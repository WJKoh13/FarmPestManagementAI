"""E7: training-only MixUp and CutMix batch mixing.

Both methods build a convex combination of two views of a batch and train
against the correspondingly mixed target. MixUp blends pixels globally; CutMix
pastes a rectangle of one image over another and sets lambda to the *actual*
pasted area.

Five rules this module is responsible for, each of which fails silently if it is
merely intended rather than enforced:

**Training only.** :meth:`BatchMixer.apply` raises unless the caller declares it
is in a training pass. Validation preprocessing stays deterministic and
unchanged, so a mixed validation batch cannot happen by accident.

**Metrics compare against the original hard labels.** Mixing returns the mixed
images together with *both* original label vectors and lambda. The accuracy
reported for a mixed epoch is measured against the true labels, never against a
soft target, so the training curve stays comparable with every historical run.

**CutMix corrects lambda.** The sampled box is clipped to the image bounds, so
the area actually pasted is usually smaller than the area drawn. Using the
uncorrected lambda would train against a target that does not match the pixels —
the classic CutMix implementation bug. Lambda is recomputed from the clipped box.

**Randomness comes from the run seed.** The mixer owns a ``torch.Generator``
seeded from the experiment seed, so a mixed run is as reproducible as an
unmixed one and does not perturb the global RNG stream the dataloader uses.

**Disabled by default.** ``method: none`` is the default and every historical
configuration resolves to exactly that, so no existing run changes meaning.

Interaction with label smoothing
    These compose rather than conflict, but they are **not** independent, and
    the phase treats that explicitly. The mixed objective is

        ``lam * CE(logits, targets_a) + (1 - lam) * CE(logits, targets_b)``

    where each ``CE`` term carries the configured ``label_smoothing``. That is
    the standard formulation and is mathematically identical to computing a
    single cross-entropy against the smoothed, lambda-blended soft target,
    because cross-entropy is linear in the target distribution. Smoothing is
    therefore left at its E0 value of 0.1 for the E7 arms: changing both the
    augmentation and the smoothing would make E7 a two-variable experiment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

__all__ = [
    "MIXING_METHODS",
    "BatchMixer",
    "MixedBatch",
    "MixingConfig",
    "MixingError",
    "mixed_criterion",
    "mixing_config_from_config",
]

#: Selectable methods. ``none`` is the default and is a strict no-op.
MIXING_METHODS: tuple[str, ...] = ("none", "mixup", "cutmix")


class MixingError(ValueError):
    """Raised when mixing configuration or usage is invalid."""


@dataclass(frozen=True)
class MixingConfig:
    """The resolved ``training.mixing`` section.

    Attributes:
        method: ``none``, ``mixup`` or ``cutmix``.
        alpha: Symmetric Beta shape parameter. Lambda is drawn from
            ``Beta(alpha, alpha)``. At ``alpha=0.2`` the draw concentrates near 0
            and 1, giving mostly-unmixed batches with occasional strong mixes;
            at ``alpha=1.0`` it is uniform.
        probability: Chance that any given batch is mixed at all. Batches that
            are not selected pass through untouched with ``lam = 1``.
    """

    method: str = "none"
    alpha: float = 0.2
    probability: float = 1.0

    def validate(self) -> MixingConfig:
        """Check every field.

        Returns:
            ``self``, so the call can be chained.

        Raises:
            MixingError: On the first inconsistency found.
        """
        if self.method not in MIXING_METHODS:
            raise MixingError(
                f"unknown training.mixing.method {self.method!r}; expected one of "
                f"{list(MIXING_METHODS)}"
            )
        if self.method != "none":
            if self.alpha <= 0.0:
                raise MixingError(
                    f"training.mixing.alpha must be positive when mixing is "
                    f"enabled, got {self.alpha}"
                )
            if not 0.0 <= self.probability <= 1.0:
                raise MixingError(
                    f"training.mixing.probability must be in [0, 1], got "
                    f"{self.probability}"
                )
        return self

    @property
    def enabled(self) -> bool:
        """Whether any mixing is applied."""
        return self.method != "none"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable mapping recorded with every checkpoint."""
        return {
            "method": self.method,
            "alpha": self.alpha,
            "probability": self.probability,
            # Recorded explicitly so a checkpoint states how its targets and
            # loss behaved, rather than leaving it to be inferred from `method`.
            "target_mode": "soft_pair" if self.enabled else "hard",
            "loss": (
                "lam * CE(logits, targets_a) + (1 - lam) * CE(logits, targets_b)"
                if self.enabled
                else "CE(logits, targets)"
            ),
            "metrics_against": "original hard labels",
            "applies_to": "train split only",
        }


def mixing_config_from_config(config: Any) -> MixingConfig:
    """Build a :class:`MixingConfig` from the ``training.mixing`` section.

    A configuration with no ``mixing`` section resolves to ``method: none``, so
    every historical config keeps its exact meaning.
    """
    training = config.section("training") if hasattr(config, "section") else {}
    section = training.get("mixing") or {}
    if not isinstance(section, Mapping):
        raise MixingError(
            f"training.mixing must be a mapping, got {section!r}"
        )
    defaults = MixingConfig()
    return MixingConfig(
        method=str(section.get("method", defaults.method)),
        alpha=float(section.get("alpha", defaults.alpha)),
        probability=float(section.get("probability", defaults.probability)),
    ).validate()


@dataclass(frozen=True)
class MixedBatch:
    """One mixed batch and everything needed to score and supervise it.

    Attributes:
        images: The mixed images, same shape as the input.
        targets_a: The original labels, in batch order.
        targets_b: The labels of the permuted partner batch.
        lam: The mixing weight on ``targets_a``. Exactly ``1.0`` when the batch
            was not mixed.
        mixed: Whether mixing was actually applied to this batch.
        method: Which method produced it.

    Note:
        ``targets_a`` is always the batch's true labels, so metrics accumulated
        against it describe the model's real accuracy on real labels. Nothing
        downstream has to know whether the batch was mixed.
    """

    images: Tensor
    targets_a: Tensor
    targets_b: Tensor
    lam: float
    mixed: bool
    method: str

    @property
    def hard_targets(self) -> Tensor:
        """The original labels, for metric accumulation."""
        return self.targets_a


def mixed_criterion(
    criterion: nn.Module, logits: Tensor, batch: MixedBatch
) -> Tensor:
    """Apply a loss to a possibly-mixed batch.

    For an unmixed batch this is exactly ``criterion(logits, targets)``, so the
    disabled path is bit-identical to the pre-E7 engine rather than merely
    equivalent.

    For a mixed batch the two cross-entropy terms are weighted by lambda. Since
    cross-entropy is linear in the target distribution, this equals a single
    cross-entropy against the blended soft target — including under label
    smoothing, which is applied identically inside each term.

    Args:
        criterion: The configured loss, carrying class weights and smoothing.
        logits: Raw model logits.
        batch: The mixed batch.

    Returns:
        The scalar loss.
    """
    if not batch.mixed:
        return criterion(logits, batch.targets_a)
    lam = batch.lam
    return lam * criterion(logits, batch.targets_a) + (1.0 - lam) * criterion(
        logits, batch.targets_b
    )


def _rand_bbox(
    height: int, width: int, lam: float, generator: torch.Generator
) -> tuple[int, int, int, int]:
    """Sample a CutMix box whose area is ``1 - lam`` of the image, then clip it.

    The centre is uniform over the image, so the box routinely extends past an
    edge. Clipping is what makes the pasted area smaller than the drawn area,
    and is precisely why the caller must recompute lambda afterwards rather than
    trusting the lambda that sized the box.

    Returns:
        ``(y1, x1, y2, x2)`` with ``y1 <= y2`` and ``x1 <= x2``.
    """
    ratio = (1.0 - lam) ** 0.5
    cut_h = int(height * ratio)
    cut_w = int(width * ratio)

    center_y = int(torch.randint(0, height, (1,), generator=generator).item())
    center_x = int(torch.randint(0, width, (1,), generator=generator).item())

    y1 = max(0, center_y - cut_h // 2)
    x1 = max(0, center_x - cut_w // 2)
    y2 = min(height, center_y + (cut_h + 1) // 2)
    x2 = min(width, center_x + (cut_w + 1) // 2)
    return y1, x1, y2, x2


class BatchMixer:
    """Applies MixUp or CutMix to training batches, deterministically.

    Owns its own :class:`torch.Generator` so that mixing draws are reproducible
    from the run seed and do not consume the global RNG stream the dataloader
    workers depend on — Phase 5 established that changing how that stream is
    consumed changes the augmentations drawn.

    Example:
        >>> mixer = BatchMixer(MixingConfig(method="mixup", alpha=0.2), seed=1337)
        >>> batch = mixer.apply(images, targets, training=True)
        >>> loss = mixed_criterion(criterion, model(batch.images), batch)
    """

    def __init__(self, config: MixingConfig, *, seed: int) -> None:
        """Build a mixer.

        Args:
            config: Resolved mixing configuration.
            seed: The run seed. The generator is derived from it, so two runs
                with the same seed draw the same mixes.
        """
        self.config = config.validate()
        self.seed = int(seed)
        # A dedicated CPU generator: mixing decisions are scalars and index
        # permutations, so there is nothing to gain from generating them on the
        # device, and keeping them on CPU makes the draws device-independent.
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)

    def _sample_lambda(self) -> float:
        """Draw lambda from ``Beta(alpha, alpha)`` using the owned generator.

        ``torch.distributions`` does not accept an explicit generator, so the
        Beta is built from two Gamma draws, which is its definition:
        ``X/(X+Y)`` with ``X, Y ~ Gamma(alpha, 1)``.
        """
        alpha = self.config.alpha
        concentration = torch.tensor([alpha], dtype=torch.float64)
        x = torch._standard_gamma(concentration, generator=self.generator)
        y = torch._standard_gamma(concentration, generator=self.generator)
        total = x + y
        if float(total) <= 0.0:
            return 1.0
        return float(x / total)

    def apply(
        self, images: Tensor, targets: Tensor, *, training: bool
    ) -> MixedBatch:
        """Mix one batch, or pass it through unchanged.

        Args:
            images: ``(N, C, H, W)`` training images.
            targets: ``(N,)`` ground-truth labels.
            training: Whether this is a training pass. **Must** be true when
                mixing is enabled.

        Returns:
            The :class:`MixedBatch`. When mixing is disabled, not selected by
            ``probability``, or the batch has fewer than two images, the images
            are returned untouched with ``lam = 1.0`` and ``mixed = False``.

        Raises:
            MixingError: If mixing is enabled but ``training`` is false, which
                would mean an evaluation split was about to be augmented.
        """
        if not self.config.enabled:
            return MixedBatch(
                images=images,
                targets_a=targets,
                targets_b=targets,
                lam=1.0,
                mixed=False,
                method="none",
            )

        if not training:
            raise MixingError(
                "batch mixing is a training-only augmentation; it may never be "
                "applied to a validation or test batch, whose preprocessing must "
                "stay deterministic"
            )

        batch_size = int(images.shape[0])
        if batch_size < 2:
            return MixedBatch(
                images=images,
                targets_a=targets,
                targets_b=targets,
                lam=1.0,
                mixed=False,
                method=self.config.method,
            )

        if self.config.probability < 1.0:
            draw = float(torch.rand((), generator=self.generator))
            if draw >= self.config.probability:
                return MixedBatch(
                    images=images,
                    targets_a=targets,
                    targets_b=targets,
                    lam=1.0,
                    mixed=False,
                    method=self.config.method,
                )

        lam = self._sample_lambda()
        permutation = torch.randperm(batch_size, generator=self.generator).to(
            images.device
        )
        targets_b = targets[permutation]

        if self.config.method == "mixup":
            mixed_images = lam * images + (1.0 - lam) * images[permutation]
            return MixedBatch(
                images=mixed_images,
                targets_a=targets,
                targets_b=targets_b,
                lam=lam,
                mixed=True,
                method="mixup",
            )

        # CutMix.
        height, width = int(images.shape[2]), int(images.shape[3])
        y1, x1, y2, x2 = _rand_bbox(height, width, lam, self.generator)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[permutation][:, :, y1:y2, x1:x2]

        # THE correction. The drawn box was sized for `lam`, but clipping to the
        # image bounds means the pasted area is usually smaller. Training against
        # the uncorrected lambda supervises a blend the pixels do not show.
        pasted = (y2 - y1) * (x2 - x1)
        corrected_lam = 1.0 - pasted / float(height * width)

        return MixedBatch(
            images=mixed_images,
            targets_a=targets,
            targets_b=targets_b,
            lam=corrected_lam,
            mixed=True,
            method="cutmix",
        )
