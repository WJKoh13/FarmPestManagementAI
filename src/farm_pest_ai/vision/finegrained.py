"""E8: an auxiliary objective that separates visually similar pests.

Cross-entropy only asks that the correct logit be largest. It says nothing about
the *geometry* of the representation, so classes that look alike can sit almost
on top of one another in feature space and still be classified correctly most of
the time — which is exactly the residual error structure Phase 7.2 measured on
rice10: the three plant hoppers confuse one another 16-25% each way, the two
borers swap 12-16%, and rice leaf caterpillar leaks 21% into rice leaf roller.

**Supervised contrastive loss** is chosen over triplet margin loss, on a measured
property of the data rather than a preference. At batch size 64 the rice10
training distribution yields ~9.94 of 10 classes present per batch and **~7
same-class partners per anchor**. SupCon consumes every one of those positives
and every negative in one term, whereas triplet loss would need a mining strategy
to select among them and would discard most of the batch. On full102 the same
batch gives only ~1.68 positives per anchor, which is why E8 screens on rice10
first and why promoting it to full102 would require reconsidering batch
composition — a second variable this phase does not introduce.

The inference contract is unchanged
    ``forward()`` still returns raw class logits, and nothing about evaluation
    changes. The embedding is produced by a separate :meth:`forward_features`
    path and a projection head that exists **only during training**: it receives
    no gradient at inference and is not consulted by any prediction. A model
    trained with this objective is loaded and served exactly like one trained
    without it.

Total objective
    ``cross_entropy + auxiliary_weight * supervised_contrastive``

Degenerate batches
    An anchor with no same-class partner contributes no term at all rather than
    a zero, and a batch with no valid anchor returns exactly zero with the graph
    intact. Both are common enough on a 102-class problem that treating them as
    errors would abort a run, and treating them as zeros would silently shrink
    the effective auxiliary weight.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

__all__ = [
    "FINE_GRAINED_METHODS",
    "FineGrainedConfig",
    "FineGrainedError",
    "ProjectionHead",
    "fine_grained_config_from_config",
    "supervised_contrastive_loss",
]

#: Selectable auxiliary objectives. ``none`` is the default and is a no-op.
FINE_GRAINED_METHODS: tuple[str, ...] = ("none", "supcon")


class FineGrainedError(ValueError):
    """Raised when the auxiliary objective is misconfigured or misused."""


@dataclass(frozen=True)
class FineGrainedConfig:
    """The resolved ``training.fine_grained`` section.

    Attributes:
        method: ``none`` or ``supcon``.
        weight: Multiplier on the auxiliary term in the total objective. The
            cross-entropy term always carries weight 1.
        temperature: SupCon temperature. Lower values sharpen the contrast and
            increase the gradient on hard negatives; 0.07 is the value the
            method was published with and is used unless evidence says otherwise.
        embedding_dim: Output width of the projection head.
        mining: Positive/negative selection strategy. ``all_positives`` uses
            every same-class partner in the batch and every other sample as a
            negative, which is what makes the loss usable without a sampler
            change on rice10.
    """

    method: str = "none"
    weight: float = 0.1
    temperature: float = 0.07
    embedding_dim: int = 128
    mining: str = "all_positives"

    def validate(self) -> FineGrainedConfig:
        """Check every field.

        Returns:
            ``self``, so the call can be chained.

        Raises:
            FineGrainedError: On the first inconsistency found.
        """
        if self.method not in FINE_GRAINED_METHODS:
            raise FineGrainedError(
                f"unknown training.fine_grained.method {self.method!r}; expected "
                f"one of {list(FINE_GRAINED_METHODS)}"
            )
        if self.method == "none":
            return self
        if self.weight < 0.0:
            raise FineGrainedError(
                f"training.fine_grained.weight must be non-negative, got "
                f"{self.weight}"
            )
        if self.temperature <= 0.0:
            raise FineGrainedError(
                f"training.fine_grained.temperature must be positive, got "
                f"{self.temperature}"
            )
        if self.embedding_dim < 2:
            raise FineGrainedError(
                f"training.fine_grained.embedding_dim must be at least 2, got "
                f"{self.embedding_dim}"
            )
        if self.mining != "all_positives":
            raise FineGrainedError(
                f"unknown training.fine_grained.mining {self.mining!r}; expected "
                f"'all_positives'"
            )
        return self

    @property
    def enabled(self) -> bool:
        """Whether the auxiliary term is applied."""
        return self.method != "none"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable mapping recorded with every checkpoint."""
        return {
            "method": self.method,
            "weight": self.weight,
            "temperature": self.temperature,
            "embedding_dim": self.embedding_dim,
            "mining": self.mining,
            "objective": (
                "cross_entropy + weight * supervised_contrastive"
                if self.enabled
                else "cross_entropy"
            ),
            "applies_to": "train split only",
            "inference_contract": "forward() returns raw class logits, unchanged",
        }


def fine_grained_config_from_config(config: Any) -> FineGrainedConfig:
    """Build a :class:`FineGrainedConfig` from ``training.fine_grained``.

    A configuration with no such section resolves to ``method: none``, so every
    historical config keeps its exact meaning.
    """
    training = config.section("training") if hasattr(config, "section") else {}
    section = training.get("fine_grained") or {}
    if not isinstance(section, Mapping):
        raise FineGrainedError(
            f"training.fine_grained must be a mapping, got {section!r}"
        )
    defaults = FineGrainedConfig()
    return FineGrainedConfig(
        method=str(section.get("method", defaults.method)),
        weight=float(section.get("weight", defaults.weight)),
        temperature=float(section.get("temperature", defaults.temperature)),
        embedding_dim=int(section.get("embedding_dim", defaults.embedding_dim)),
        mining=str(section.get("mining", defaults.mining)),
    ).validate()


class ProjectionHead(nn.Module):
    """Maps pooled backbone features to a normalised embedding.

    A two-layer MLP, as SupCon specifies: contrasting the pooled features
    directly ties the metric geometry to the same vector the classifier reads,
    which degrades classification. The projection gives the contrastive loss its
    own space to shape.

    Outputs are L2-normalised, so the dot product used by the loss is a cosine
    similarity bounded in ``[-1, 1]`` and the temperature has a stable meaning.

    This module is **training-only scaffolding**: it is never called by
    ``forward()`` and never participates in a prediction.
    """

    def __init__(self, in_features: int, embedding_dim: int) -> None:
        """Build the head.

        Args:
            in_features: Width of the pooled backbone features.
            embedding_dim: Output embedding width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, embedding_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return L2-normalised embeddings of shape ``(N, embedding_dim)``."""
        return nn.functional.normalize(self.net(features), dim=1)


def supervised_contrastive_loss(
    embeddings: Tensor, targets: Tensor, *, temperature: float = 0.07
) -> Tensor:
    """Supervised contrastive loss over a batch of normalised embeddings.

    For each anchor ``i`` with positive set ``P(i)`` — the other samples sharing
    its label — the loss is

        ``-1/|P(i)| * sum_{p in P(i)} log( exp(z_i·z_p/T) / sum_{a != i} exp(z_i·z_a/T) )``

    averaged over anchors that have at least one positive.

    Numerical stability
        The similarity row has its maximum subtracted before exponentiation.
        Without that, ``exp(1/0.07)`` overflows fp16 immediately and produces a
        NaN loss on the first batch under AMP — the failure mode that makes a
        contrastive run look like a broken model rather than a broken loss.

    Degenerate batches
        Anchors without a positive are excluded from the average rather than
        counted as zero, so the reported magnitude stays comparable across
        batches with different class composition. A batch where no anchor has a
        positive returns ``0.0`` **with the graph attached**, so ``backward()``
        still works and the step is a genuine no-op for this term.

    Args:
        embeddings: ``(N, D)`` L2-normalised embeddings.
        targets: ``(N,)`` integer labels.
        temperature: Softmax temperature; must be positive.

    Returns:
        A scalar loss tensor.

    Raises:
        FineGrainedError: If shapes disagree or the temperature is not positive.
    """
    if temperature <= 0.0:
        raise FineGrainedError(f"temperature must be positive, got {temperature}")
    if embeddings.ndim != 2:
        raise FineGrainedError(
            f"expected embeddings of shape (N, D), got {tuple(embeddings.shape)}"
        )
    labels = targets.reshape(-1)
    if labels.shape[0] != embeddings.shape[0]:
        raise FineGrainedError(
            f"{embeddings.shape[0]} embeddings against {labels.shape[0]} targets"
        )

    batch_size = int(embeddings.shape[0])
    if batch_size < 2:
        return (embeddings.sum() * 0.0).to(embeddings.dtype)

    # Contrastive similarities are computed in float32 even under autocast: the
    # exponentials span a wide range at T=0.07 and fp16 loses them.
    features = embeddings.float()
    similarity = features @ features.t() / float(temperature)

    identity = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    # Subtract the row max over the non-self entries for stability. Self-terms
    # are masked out of both numerator and denominator, so they must not set the
    # maximum either.
    masked = similarity.masked_fill(identity, float("-inf"))
    row_max = masked.max(dim=1, keepdim=True).values.detach()
    # A row that is entirely -inf cannot happen for batch_size >= 2, but guard
    # against a non-finite max propagating into the exponent regardless.
    row_max = torch.nan_to_num(row_max, neginf=0.0)
    logits = similarity - row_max

    exponentiated = torch.exp(logits).masked_fill(identity, 0.0)
    denominator = exponentiated.sum(dim=1, keepdim=True)
    log_probability = logits - torch.log(denominator.clamp_min(1e-12))

    positive_mask = (labels.reshape(-1, 1) == labels.reshape(1, -1)) & ~identity
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0

    if not bool(valid.any()):
        # No anchor has a partner: contribute nothing, but keep the graph so the
        # caller's backward() is still well-formed.
        return (embeddings.sum() * 0.0).to(embeddings.dtype)

    positive_log_probability = (
        (log_probability * positive_mask).sum(dim=1)[valid]
        / positive_counts[valid].clamp_min(1)
    )
    return (-positive_log_probability.mean()).to(embeddings.dtype)
