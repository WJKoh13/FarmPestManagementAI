"""Classification metrics, computed from a running confusion matrix.

**Validation macro F1 is the project's primary model-selection metric.** Both
scopes are imbalanced — ``full102`` by 82x — so accuracy would be dominated by
the largest classes and would happily reward a model that ignores the rare ones
entirely.

Everything is accumulated into a ``num_classes x num_classes`` confusion matrix
on the device, which costs one scatter per batch and no host synchronisation.
Every headline metric is then derived from that single matrix, so accuracy,
macro F1 and balanced accuracy are guaranteed to describe the same predictions.

Macro averaging convention
    A class with no predictions and no ground-truth instances has an undefined
    F1. Here it contributes **zero** to the macro average rather than being
    dropped, which is the stricter and more honest reading: a model that never
    predicts a rare class should not be rewarded by having that class quietly
    excluded from its score. ``full102`` validation has classes with as few as
    seven images, so this choice is not hypothetical.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "ClassificationMetrics",
    "MetricsAccumulator",
    "MetricsError",
    "confusion_matrix",
    "label_smoothing_loss_floor",
    "macro_f1",
    "top_k_accuracy",
]


def label_smoothing_loss_floor(label_smoothing: float, num_classes: int) -> float:
    """Minimum cross-entropy achievable under label smoothing.

    With smoothing ``eps`` the target is no longer one-hot but
    ``1 - eps + eps/C`` on the true class and ``eps/C`` elsewhere. The minimum of
    the cross-entropy is then the **entropy of that smoothed target**, not zero:
    a perfectly fitted model still pays this much.

    This matters for the Phase 6 overfit check. At ``eps=0.1`` over 10 classes
    the floor is 0.5003, so comparing a converged run against a "near zero" loss
    target would report a healthy model as broken. The floor grows with the class
    count — 0.78 at 102 classes — so a fixed threshold cannot serve both scopes.

    Args:
        label_smoothing: The smoothing ``eps`` in ``[0, 1)``.
        num_classes: Number of classes.

    Returns:
        The minimum achievable mean cross-entropy, ``0.0`` when smoothing is off.

    Raises:
        MetricsError: If ``label_smoothing`` is outside ``[0, 1)`` or
            ``num_classes`` is below 2.
    """
    import math

    if not 0.0 <= label_smoothing < 1.0:
        raise MetricsError(
            f"label_smoothing must be in [0, 1), got {label_smoothing}"
        )
    if num_classes < 2:
        raise MetricsError(f"num_classes must be at least 2, got {num_classes}")
    if label_smoothing <= 0.0:
        return 0.0

    on_target = 1.0 - label_smoothing + label_smoothing / num_classes
    off_target = label_smoothing / num_classes
    return float(
        -(
            on_target * math.log(on_target)
            + (num_classes - 1) * off_target * math.log(off_target)
        )
    )


class MetricsError(ValueError):
    """Raised when metric inputs are malformed."""


@dataclass(frozen=True)
class ClassificationMetrics:
    """A resolved set of metrics for one split at one epoch.

    Attributes:
        accuracy: Top-1 accuracy over all samples.
        macro_f1: Unweighted mean of per-class F1. **The selection metric.**
        weighted_f1: Per-class F1 weighted by ground-truth support.
        balanced_accuracy: Unweighted mean of per-class recall.
        top5_accuracy: Top-5 accuracy, or ``None`` when the scope has fewer than
            six classes and the metric would be meaningless.
        loss: Mean loss over the split, when tracked.
        samples: Number of samples scored.
        per_class_precision: Precision indexed by project label.
        per_class_recall: Recall indexed by project label.
        per_class_f1: F1 indexed by project label.
        per_class_support: Ground-truth count indexed by project label.
        classes_never_predicted: Project labels the model never predicted. Worth
            recording separately: a model can post a respectable accuracy while
            silently abandoning a dozen rare classes.
    """

    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    top5_accuracy: float | None = None
    loss: float | None = None
    samples: int = 0
    per_class_precision: tuple[float, ...] = field(default_factory=tuple)
    per_class_recall: tuple[float, ...] = field(default_factory=tuple)
    per_class_f1: tuple[float, ...] = field(default_factory=tuple)
    per_class_support: tuple[int, ...] = field(default_factory=tuple)
    classes_never_predicted: tuple[int, ...] = field(default_factory=tuple)

    def to_dict(self, *, per_class: bool = True) -> dict[str, Any]:
        """Return a JSON-serialisable mapping.

        Args:
            per_class: Whether to include the per-class arrays. The epoch log
                sets this false for ``full102``, where 102 classes times four
                arrays would dominate every line.
        """
        payload: dict[str, Any] = {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "loss": self.loss,
            "samples": self.samples,
            "classes_never_predicted": list(self.classes_never_predicted),
        }
        if per_class:
            payload["per_class"] = {
                "precision": list(self.per_class_precision),
                "recall": list(self.per_class_recall),
                "f1": list(self.per_class_f1),
                "support": list(self.per_class_support),
            }
        return payload

    def get(self, name: str) -> float:
        """Look up a scalar metric by name, for early stopping and checkpointing.

        Raises:
            MetricsError: If ``name`` is not a scalar metric, or is one that was
                not computed for this split.
        """
        values: dict[str, float | None] = {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "loss": self.loss,
        }
        if name not in values:
            raise MetricsError(
                f"unknown metric {name!r}; expected one of {sorted(values)}"
            )
        value = values[name]
        if value is None:
            raise MetricsError(f"metric {name!r} was not computed for this split")
        return float(value)


def confusion_matrix(
    predictions: Tensor, targets: Tensor, num_classes: int
) -> Tensor:
    """Build a confusion matrix indexed ``[true, predicted]``.

    Args:
        predictions: Predicted project labels, shape ``(N,)``.
        targets: Ground-truth project labels, shape ``(N,)``.
        num_classes: Number of classes.

    Returns:
        An ``int64`` matrix of shape ``(num_classes, num_classes)``.

    Raises:
        MetricsError: If shapes disagree or a label is out of range. An
            out-of-range label would otherwise wrap silently through the flat
            index and corrupt an unrelated cell.
    """
    if predictions.shape != targets.shape:
        raise MetricsError(
            f"predictions shape {tuple(predictions.shape)} does not match targets "
            f"{tuple(targets.shape)}"
        )
    if num_classes < 2:
        raise MetricsError(f"num_classes must be at least 2, got {num_classes}")

    predictions = predictions.reshape(-1).to(torch.int64)
    targets = targets.reshape(-1).to(torch.int64)
    if predictions.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for name, values in (("prediction", predictions), ("target", targets)):
        if int(values.min()) < 0 or int(values.max()) >= num_classes:
            raise MetricsError(
                f"{name} label outside 0..{num_classes - 1}: min "
                f"{int(values.min())}, max {int(values.max())}"
            )

    flat = targets * num_classes + predictions
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def top_k_accuracy(logits: Tensor, targets: Tensor, k: int = 5) -> float:
    """Fraction of samples whose true label is among the top ``k`` logits.

    Args:
        logits: Raw logits, shape ``(N, C)``.
        targets: Ground-truth labels, shape ``(N,)``.
        k: How many predictions to consider. Clamped to the class count.

    Returns:
        The accuracy in ``[0, 1]``, or ``0.0`` for an empty batch.
    """
    if logits.ndim != 2:
        raise MetricsError(
            f"expected logits of shape (N, C), got {tuple(logits.shape)}"
        )
    if logits.shape[0] == 0:
        return 0.0
    k = max(1, min(int(k), int(logits.shape[1])))
    top = logits.topk(k, dim=1).indices
    hits = (top == targets.reshape(-1, 1)).any(dim=1)
    return float(hits.float().mean())


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Elementwise division that yields 0 where the denominator is 0.

    Used for precision, recall and F1: a class with no predictions has undefined
    precision, and the project's convention is to score that as zero rather than
    to exclude the class from the macro average.

    The zero denominator is replaced *only* to keep the division itself finite;
    the ``where`` then discards that branch's value entirely. A positive
    denominator is always divided by unchanged, whatever its magnitude.

    Corrected in Phase 7.1
        This previously clamped the denominator to ``min=1`` before dividing.
        For precision and recall that is invisible — their denominators are
        integer counts, so a positive one is already at least 1. F1's
        denominator is ``precision + recall``, a **fraction**, and clamping it
        silently rewrote every value in ``(0, 1)`` as 1, under-reporting F1 for
        exactly the weakest classes. Replacing only the zeros is what makes the
        two cases behave the same way.
    """
    safe_denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )
    return torch.where(
        denominator > 0,
        numerator / safe_denominator,
        torch.zeros_like(numerator),
    )


def macro_f1(matrix: Tensor) -> float:
    """Compute macro F1 directly from a confusion matrix."""
    _, _, f1, _ = _per_class_from_matrix(matrix)
    return float(f1.mean())


def _per_class_from_matrix(matrix: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return per-class precision, recall, F1 and support from a matrix."""
    matrix = matrix.to(torch.float64)
    true_positive = matrix.diagonal()
    predicted = matrix.sum(dim=0)
    support = matrix.sum(dim=1)

    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1, support


class MetricsAccumulator:
    """Accumulates predictions across batches and resolves them once at the end.

    The confusion matrix lives on the compute device and is updated per batch
    without a host synchronisation, so metric collection does not stall the GPU.
    Top-5 needs the logits themselves rather than the argmax, so it is tracked as
    a running hit count instead of being derivable from the matrix.

    Example:
        >>> accumulator = MetricsAccumulator(num_classes=10)
        >>> accumulator.update(logits, targets, loss=batch_loss)
        >>> metrics = accumulator.compute()
    """

    def __init__(
        self,
        num_classes: int,
        *,
        device: str | torch.device = "cpu",
        track_top5: bool = True,
    ) -> None:
        """Build an accumulator.

        Args:
            num_classes: Number of classes; must match the active scope.
            device: Device the confusion matrix lives on.
            track_top5: Whether to track top-5. Ignored when the scope has fewer
                than six classes, where the metric carries no information.

        Raises:
            MetricsError: If ``num_classes`` is below 2.
        """
        if num_classes < 2:
            raise MetricsError(f"num_classes must be at least 2, got {num_classes}")
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.track_top5 = bool(track_top5) and num_classes > 5
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64, device=self.device
        )
        self._top5_hits = torch.zeros((), dtype=torch.int64, device=self.device)
        self._loss_total = 0.0
        self._loss_samples = 0
        self._samples = 0

    def update(
        self, logits: Tensor, targets: Tensor, *, loss: float | None = None
    ) -> None:
        """Accumulate one batch.

        Args:
            logits: Raw logits, shape ``(N, C)``. Softmax is neither expected nor
                applied: ``argmax`` and ``topk`` are invariant under it, so
                converting first would only cost time.
            targets: Ground-truth project labels, shape ``(N,)``.
            loss: Mean loss for this batch, weighted by batch size when averaged.

        Raises:
            MetricsError: If the logits' class count disagrees with the
                accumulator's, which would mean the model and the scope have
                drifted apart.
        """
        if logits.ndim != 2:
            raise MetricsError(
                f"expected logits of shape (N, C), got {tuple(logits.shape)}"
            )
        if logits.shape[1] != self.num_classes:
            raise MetricsError(
                f"model produced {logits.shape[1]} logits but the accumulator expects "
                f"{self.num_classes}; the model and the active scope disagree"
            )
        batch = int(logits.shape[0])
        if batch == 0:
            return

        logits = logits.detach()
        targets = targets.detach().reshape(-1).to(self.device, torch.int64)
        predictions = logits.argmax(dim=1).to(self.device)

        flat = targets * self.num_classes + predictions
        self._matrix += torch.bincount(
            flat, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

        if self.track_top5:
            top = logits.topk(5, dim=1).indices.to(self.device)
            self._top5_hits += (
                (top == targets.reshape(-1, 1)).any(dim=1).sum().to(torch.int64)
            )

        if loss is not None:
            self._loss_total += float(loss) * batch
            self._loss_samples += batch
        self._samples += batch

    @property
    def matrix(self) -> Tensor:
        """The accumulated confusion matrix, indexed ``[true, predicted]``."""
        return self._matrix.clone()

    @property
    def samples(self) -> int:
        """Number of samples accumulated so far."""
        return self._samples

    def compute(self) -> ClassificationMetrics:
        """Resolve every metric from the accumulated state.

        Returns:
            The metrics. An accumulator that saw no samples returns zeros rather
            than raising, so an empty split does not abort a run mid-epoch.
        """
        matrix = self._matrix.cpu()
        precision, recall, f1, support = _per_class_from_matrix(matrix)

        total = int(matrix.sum())
        correct = int(matrix.diagonal().sum())
        accuracy = correct / total if total else 0.0

        weights = support / support.sum().clamp(min=1)
        weighted = float((f1 * weights).sum()) if total else 0.0

        # Balanced accuracy averages recall over classes that are actually
        # present. A class absent from this split has no recall to average, and
        # including it as zero would penalise the model for the split's
        # composition rather than its predictions.
        present = support > 0
        balanced = float(recall[present].mean()) if bool(present.any()) else 0.0

        predicted_counts = matrix.sum(dim=0)
        never_predicted = tuple(
            int(i) for i in torch.nonzero(predicted_counts == 0).reshape(-1)
        )

        return ClassificationMetrics(
            accuracy=accuracy,
            macro_f1=float(f1.mean()) if total else 0.0,
            weighted_f1=weighted,
            balanced_accuracy=balanced,
            top5_accuracy=(
                float(self._top5_hits) / total if self.track_top5 and total else None
            ),
            loss=(self._loss_total / self._loss_samples if self._loss_samples else None),
            samples=self._samples,
            per_class_precision=tuple(float(v) for v in precision),
            per_class_recall=tuple(float(v) for v in recall),
            per_class_f1=tuple(float(v) for v in f1),
            per_class_support=tuple(int(v) for v in support),
            classes_never_predicted=never_predicted,
        )


def metrics_from_predictions(
    predictions: Sequence[int] | Tensor,
    targets: Sequence[int] | Tensor,
    num_classes: int,
) -> ClassificationMetrics:
    """Compute metrics from already-argmaxed predictions.

    Convenience for tests and for Phase 9's per-image prediction files, where
    the logits are no longer available. Top-5 cannot be recovered from argmax
    output and is reported as ``None``.
    """
    prediction_tensor = torch.as_tensor(predictions, dtype=torch.int64).reshape(-1)
    target_tensor = torch.as_tensor(targets, dtype=torch.int64).reshape(-1)
    matrix = confusion_matrix(prediction_tensor, target_tensor, num_classes)

    precision, recall, f1, support = _per_class_from_matrix(matrix)
    total = int(matrix.sum())
    correct = int(matrix.diagonal().sum())
    weights = support / support.sum().clamp(min=1)
    present = support > 0
    predicted_counts = matrix.sum(dim=0)

    return ClassificationMetrics(
        accuracy=correct / total if total else 0.0,
        macro_f1=float(f1.mean()) if total else 0.0,
        weighted_f1=float((f1 * weights).sum()) if total else 0.0,
        balanced_accuracy=float(recall[present].mean()) if bool(present.any()) else 0.0,
        top5_accuracy=None,
        loss=None,
        samples=total,
        per_class_precision=tuple(float(v) for v in precision),
        per_class_recall=tuple(float(v) for v in recall),
        per_class_f1=tuple(float(v) for v in f1),
        per_class_support=tuple(int(v) for v in support),
        classes_never_predicted=tuple(
            int(i) for i in torch.nonzero(predicted_counts == 0).reshape(-1)
        ),
    )
