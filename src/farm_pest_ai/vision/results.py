"""Read completed run artifacts and recompute corrected metrics from them.

Phase 7 recorded its per-class **precision, recall and support** alongside every
F1 it derived. Those three were never affected by the Phase 7.1 F1 denominator
bug — their own denominators are integer counts, where the clamp was a no-op —
so every corrected F1 can be recovered arithmetically from the saved artifacts.
No retraining and no model forward pass is required.

That is the whole reason this module exists rather than a rerun script:

* the original artifacts stay **untouched**, so the reported figures remain
  auditable next to the corrected ones;
* the correction is reproducible from files on disk by anyone, at any time;
* the recomputation is exact, not an approximation of what a rerun would give.

What cannot be recovered
    ``weighted_f1`` and ``macro_f1`` are recomputed exactly, since both are
    functions of the per-class values. ``accuracy``, ``balanced_accuracy`` and
    ``top5_accuracy`` were never wrong — none of them routes through the F1
    denominator — and are carried through unchanged.

    A run whose **best epoch moves** under correction is flagged rather than
    silently re-pointed: its saved ``best.pt`` holds the weights of the epoch the
    buggy metric selected, which is a real artifact of the run and not something
    a recomputation can undo. See :func:`load_run`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EpochRecord",
    "ResultsError",
    "RunResults",
    "SplitMetrics",
    "compare_runs",
    "confusion_matrix_for_run",
    "corrected_f1",
    "discover_runs",
    "load_run",
]

#: Splits a completed training run records. The test split is never among them.
TRACKED_SPLITS = ("train", "validation")


class ResultsError(ValueError):
    """Raised when a run directory is missing or malformed."""


def corrected_f1(precision: float, recall: float) -> float:
    """F1 from precision and recall, with the project's zero convention.

    This is the reference implementation of the correction. The denominator
    ``precision + recall`` is used **as-is** whenever it is positive; only an
    exactly-zero denominator falls back to zero, matching
    ``sklearn`` ``zero_division=0``.

    Args:
        precision: Precision in ``[0, 1]``.
        recall: Recall in ``[0, 1]``.

    Returns:
        The harmonic mean, or ``0.0`` when both inputs are zero.
    """
    denominator = precision + recall
    if denominator <= 0.0:
        return 0.0
    return 2.0 * precision * recall / denominator


@dataclass(frozen=True)
class SplitMetrics:
    """One split's metrics at one epoch, as reported and as corrected.

    Attributes:
        split: ``train`` or ``validation``.
        accuracy: Unaffected by the correction.
        balanced_accuracy: Unaffected by the correction.
        loss: Unaffected by the correction.
        top5_accuracy: Unaffected by the correction.
        reported_macro_f1: The value the run wrote to disk.
        corrected_macro_f1: Recomputed from saved precision and recall.
        reported_weighted_f1: The value the run wrote to disk.
        corrected_weighted_f1: Recomputed from saved precision and recall.
        per_class_precision: Saved, unaffected by the correction.
        per_class_recall: Saved, unaffected by the correction.
        per_class_support: Saved, unaffected by the correction.
        reported_per_class_f1: The per-class values the run wrote to disk.
        corrected_per_class_f1: Recomputed per-class values.
    """

    split: str
    accuracy: float
    balanced_accuracy: float
    loss: float | None
    top5_accuracy: float | None
    reported_macro_f1: float
    corrected_macro_f1: float
    reported_weighted_f1: float
    corrected_weighted_f1: float
    per_class_precision: tuple[float, ...] = field(default_factory=tuple)
    per_class_recall: tuple[float, ...] = field(default_factory=tuple)
    per_class_support: tuple[int, ...] = field(default_factory=tuple)
    reported_per_class_f1: tuple[float, ...] = field(default_factory=tuple)
    corrected_per_class_f1: tuple[float, ...] = field(default_factory=tuple)

    @property
    def macro_f1_delta(self) -> float:
        """How far the corrected macro F1 sits above the reported one."""
        return self.corrected_macro_f1 - self.reported_macro_f1

    @property
    def affected_classes(self) -> tuple[int, ...]:
        """Classes whose F1 changed under the correction."""
        return tuple(
            index
            for index, (old, new) in enumerate(
                zip(
                    self.reported_per_class_f1,
                    self.corrected_per_class_f1,
                    strict=True,
                )
            )
            if abs(old - new) > 1e-12
        )


@dataclass(frozen=True)
class EpochRecord:
    """One epoch of a completed run.

    Attributes:
        epoch: One-based epoch number.
        learning_rate: The rate the epoch ran at.
        train: Training-split metrics, when recorded.
        validation: Validation-split metrics, when recorded.
        train_seconds: Wall-clock training time.
        validation_seconds: Wall-clock validation time.
        optimizer_steps: Optimiser steps taken.
        amp_skipped_steps: Steps AMP skipped for gradient overflow.
        peak_vram_mib: Peak allocated VRAM.
        raw: The original JSON record, unmodified.
    """

    epoch: int
    learning_rate: float
    train: SplitMetrics | None
    validation: SplitMetrics | None
    train_seconds: float | None = None
    validation_seconds: float | None = None
    optimizer_steps: int | None = None
    amp_skipped_steps: int | None = None
    peak_vram_mib: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def split(self, name: str) -> SplitMetrics | None:
        """Return metrics for ``name``, or ``None`` when the split was absent."""
        if name == "train":
            return self.train
        if name == "validation":
            return self.validation
        raise ResultsError(
            f"unknown split {name!r}; a training run records {TRACKED_SPLITS}"
        )


@dataclass(frozen=True)
class RunResults:
    """A completed run, with corrected metrics derived from its artifacts.

    Attributes:
        run_id: The run's identifier, taken from its summary.
        run_dir: Directory the artifacts were read from.
        scope: Dataset scope the run used.
        model_name: Architecture name.
        parameters: Total parameter count.
        epochs: Every recorded epoch, in order.
        summary: The run's original summary block, unmodified.
        config_sources: Configuration files the run was composed from.
    """

    run_id: str
    run_dir: Path
    scope: str
    model_name: str
    parameters: int | None
    epochs: tuple[EpochRecord, ...]
    summary: dict[str, Any] = field(default_factory=dict, repr=False)
    config_sources: tuple[str, ...] = field(default_factory=tuple)

    def curve(self, split: str, metric: str) -> list[float | None]:
        """Extract one metric across every epoch, for plotting.

        Args:
            split: ``train`` or ``validation``.
            metric: A :class:`SplitMetrics` attribute name, such as
                ``corrected_macro_f1`` or ``accuracy``.

        Returns:
            One value per epoch, ``None`` where the split was not recorded.
        """
        values: list[float | None] = []
        for record in self.epochs:
            metrics = record.split(split)
            if metrics is None:
                values.append(None)
                continue
            if not hasattr(metrics, metric):
                raise ResultsError(
                    f"unknown metric {metric!r} for split {split!r}"
                )
            value = getattr(metrics, metric)
            values.append(None if value is None else float(value))
        return values

    @property
    def epoch_numbers(self) -> list[int]:
        """The epoch axis."""
        return [record.epoch for record in self.epochs]

    @property
    def learning_rates(self) -> list[float]:
        """The learning rate at each epoch."""
        return [record.learning_rate for record in self.epochs]

    def best_epoch(self, *, corrected: bool = True) -> int | None:
        """Epoch with the highest validation macro F1.

        Args:
            corrected: Select on the corrected metric. ``False`` reproduces the
                epoch the original run selected, which is what its ``best.pt``
                actually holds.

        Returns:
            The one-based epoch number, or ``None`` if nothing was recorded.
        """
        attribute = "corrected_macro_f1" if corrected else "reported_macro_f1"
        best_epoch: int | None = None
        best_value = float("-inf")
        for record in self.epochs:
            if record.validation is None:
                continue
            value = getattr(record.validation, attribute)
            if value > best_value:
                best_value, best_epoch = value, record.epoch
        return best_epoch

    def best_validation(self, *, corrected: bool = True) -> SplitMetrics | None:
        """The validation metrics of :meth:`best_epoch`."""
        epoch = self.best_epoch(corrected=corrected)
        if epoch is None:
            return None
        for record in self.epochs:
            if record.epoch == epoch:
                return record.validation
        return None

    @property
    def best_epoch_moved(self) -> bool:
        """Whether the correction selects a different epoch than the run did.

        When true, the saved ``best.pt`` holds the weights of an epoch that is
        no longer the best one. The checkpoint is not wrong — it is what the run
        chose — but any claim about "the best model" must say which metric
        selected it.
        """
        return self.best_epoch(corrected=True) != self.best_epoch(corrected=False)

    def preprocessing_config(self) -> Any:
        """Rebuild the preprocessing this run actually trained with.

        A run's own preprocessing — not the ambient configuration — is what a
        checkpoint must be scored through. E2 trained at 224x224; evaluating it
        through the default 160x160 pipeline loads cleanly, because
        ``strict_preprocessing`` defaults off, and produces a plausible but
        wrong result.

        Returns:
            The run's :class:`~farm_pest_ai.data.transforms.PreprocessingConfig`,
            or ``None`` when the summary did not record one.
        """
        data = self.summary.get("data")
        if not isinstance(data, dict):
            return None
        payload = data.get("preprocessing")
        if not isinstance(payload, dict):
            return None

        from farm_pest_ai.data.transforms import (
            AugmentationConfig,
            PreprocessingConfig,
        )

        recorded = payload.get("augmentation") or {}
        defaults = AugmentationConfig()

        def _pair(name: str, fallback: tuple[float, float]) -> tuple[float, float]:
            value = recorded.get(name)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return (float(value[0]), float(value[1]))
            return fallback

        def _number(name: str, fallback: float) -> float:
            value = recorded.get(name)
            return float(value) if isinstance(value, (int, float)) else fallback

        def _flag(name: str, fallback: bool) -> bool:
            value = recorded.get(name)
            return bool(value) if isinstance(value, bool) else fallback

        augment = AugmentationConfig(
            enabled=_flag("enabled", defaults.enabled),
            random_resized_crop=_flag(
                "random_resized_crop", defaults.random_resized_crop
            ),
            scale=_pair("scale", defaults.scale),
            ratio=_pair("ratio", defaults.ratio),
            horizontal_flip=_number("horizontal_flip", defaults.horizontal_flip),
            vertical_flip=_number("vertical_flip", defaults.vertical_flip),
            rotation_degrees=_number("rotation_degrees", defaults.rotation_degrees),
            color_jitter_brightness=_number(
                "color_jitter_brightness", defaults.color_jitter_brightness
            ),
            color_jitter_contrast=_number(
                "color_jitter_contrast", defaults.color_jitter_contrast
            ),
            color_jitter_saturation=_number(
                "color_jitter_saturation", defaults.color_jitter_saturation
            ),
            color_jitter_hue=_number("color_jitter_hue", defaults.color_jitter_hue),
            random_erasing=_number("random_erasing", defaults.random_erasing),
        )
        size = payload.get("image_size") or [160, 160]
        return PreprocessingConfig(
            image_size=(int(size[0]), int(size[1])),
            interpolation=str(payload.get("interpolation", "bilinear")),
            resize_shorter_side=payload.get("resize_shorter_side"),
            mean=tuple(payload.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(payload.get("std", (0.229, 0.224, 0.225))),
            augmentation=augment,
            version=str(payload.get("version", "1.0.0")),
        )

    @property
    def warmup_epochs(self) -> int | None:
        """Warm-up epochs from the resolved training config, when recorded."""
        training = self.summary.get("training")
        if isinstance(training, dict):
            value = training.get("warmup_epochs")
            if isinstance(value, (int, float)):
                return int(value)
        return None


def _split_metrics(name: str, payload: dict[str, Any]) -> SplitMetrics:
    """Build corrected metrics for one split from a recorded epoch payload."""
    per_class = payload.get("per_class") or {}
    precision = tuple(float(v) for v in per_class.get("precision", ()))
    recall = tuple(float(v) for v in per_class.get("recall", ()))
    support = tuple(int(v) for v in per_class.get("support", ()))
    reported_f1 = tuple(float(v) for v in per_class.get("f1", ()))

    if len(precision) != len(recall):
        raise ResultsError(
            f"{name}: {len(precision)} precision values against {len(recall)} recall"
        )
    if support and len(support) != len(precision):
        raise ResultsError(
            f"{name}: {len(support)} support values against {len(precision)} classes"
        )

    corrected = tuple(
        corrected_f1(p, r) for p, r in zip(precision, recall, strict=True)
    )

    # Macro averages over every class, including ones the model never predicted.
    # Weighted averages by ground-truth support. Both mirror the definitions in
    # farm_pest_ai.vision.metrics so the two paths cannot drift.
    if corrected:
        corrected_macro = sum(corrected) / len(corrected)
        total_support = sum(support)
        corrected_weighted = (
            sum(f * s for f, s in zip(corrected, support, strict=True)) / total_support
            if total_support
            else 0.0
        )
    else:
        # No per-class arrays were recorded for this split; fall back to the
        # reported scalars rather than inventing a correction.
        corrected_macro = float(payload.get("macro_f1", 0.0))
        corrected_weighted = float(payload.get("weighted_f1", 0.0))

    loss = payload.get("loss")
    top5 = payload.get("top5_accuracy")
    return SplitMetrics(
        split=name,
        accuracy=float(payload.get("accuracy", 0.0)),
        balanced_accuracy=float(payload.get("balanced_accuracy", 0.0)),
        loss=None if loss is None else float(loss),
        top5_accuracy=None if top5 is None else float(top5),
        reported_macro_f1=float(payload.get("macro_f1", 0.0)),
        corrected_macro_f1=corrected_macro,
        reported_weighted_f1=float(payload.get("weighted_f1", 0.0)),
        corrected_weighted_f1=corrected_weighted,
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_support=support,
        reported_per_class_f1=reported_f1,
        corrected_per_class_f1=corrected,
    )


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each JSON object from a JSON Lines file."""
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ResultsError(f"{path}:{number}: {error}") from error
            if not isinstance(record, dict):
                raise ResultsError(f"{path}:{number}: expected a JSON object")
            yield record


def load_run(run_dir: str | Path) -> RunResults:
    """Load a completed run and correct every F1 it recorded.

    Reads ``metrics.jsonl`` for the epoch series and ``summary.json`` for run
    identity. Neither file is modified.

    Args:
        run_dir: A run directory, such as
            ``artifacts/checkpoints/rice10_custom_protocolA``.

    Returns:
        The run with reported and corrected metrics side by side.

    Raises:
        ResultsError: If the directory or its ``metrics.jsonl`` is missing, or
            if a record is malformed.
    """
    directory = Path(run_dir)
    if not directory.is_dir():
        raise ResultsError(f"run directory does not exist: {directory}")

    metrics_path = directory / "metrics.jsonl"
    if not metrics_path.is_file():
        raise ResultsError(f"no metrics.jsonl in {directory}")

    summary: dict[str, Any] = {}
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            summary = loaded

    records: list[EpochRecord] = []
    for payload in _read_jsonl(metrics_path):
        if payload.get("smoke"):
            raise ResultsError(
                f"{metrics_path} is a smoke run; its metrics are not results"
            )
        splits: dict[str, SplitMetrics | None] = {}
        for name in TRACKED_SPLITS:
            block = payload.get(name)
            splits[name] = (
                _split_metrics(name, block) if isinstance(block, dict) else None
            )
        records.append(
            EpochRecord(
                epoch=int(payload.get("epoch", len(records) + 1)),
                learning_rate=float(payload.get("learning_rate", 0.0)),
                train=splits["train"],
                validation=splits["validation"],
                train_seconds=payload.get("train_seconds"),
                validation_seconds=payload.get("validation_seconds"),
                optimizer_steps=payload.get("optimizer_steps"),
                amp_skipped_steps=payload.get("amp_skipped_steps"),
                peak_vram_mib=payload.get("peak_vram_mib"),
                raw=payload,
            )
        )

    if not records:
        raise ResultsError(f"{metrics_path} contains no epochs")

    records.sort(key=lambda record: record.epoch)
    raw_model = summary.get("model")
    model: dict[str, Any] = raw_model if isinstance(raw_model, dict) else {}
    parameters = summary.get("parameters")
    total_parameters = (
        int(parameters["total"])
        if isinstance(parameters, dict) and "total" in parameters
        else None
    )

    return RunResults(
        run_id=str(summary.get("run_id", directory.name)),
        run_dir=directory,
        scope=str(summary.get("scope", "")),
        model_name=str(model.get("name", "")),
        parameters=total_parameters,
        epochs=tuple(records),
        summary=summary,
        config_sources=tuple(str(p) for p in summary.get("config_sources", ())),
    )


def discover_runs(root: str | Path) -> list[RunResults]:
    """Load every run directory beneath ``root``.

    A directory qualifies when it holds a ``metrics.jsonl``. Smoke runs raise
    in :func:`load_run` and are skipped here rather than aborting a sweep.

    Args:
        root: Directory to search, typically ``artifacts/checkpoints``.

    Returns:
        Runs sorted by ``run_id``.
    """
    directory = Path(root)
    if not directory.is_dir():
        raise ResultsError(f"not a directory: {directory}")

    runs: list[RunResults] = []
    for candidate in sorted(directory.iterdir()):
        if not (candidate / "metrics.jsonl").is_file():
            continue
        try:
            runs.append(load_run(candidate))
        except ResultsError:
            continue
    return sorted(runs, key=lambda run: run.run_id)


def confusion_matrix_for_run(
    run: RunResults,
    config: Any,
    *,
    checkpoint: str = "best.pt",
    split: str = "validation",
) -> list[list[int]]:
    """Recompute a run's validation confusion matrix from its checkpoint.

    Training records per-class precision, recall and support but not the matrix
    itself, so it is regenerated here by evaluating the checkpoint. This is
    exact rather than approximate: evaluation preprocessing contains no random
    step and the loader preserves manifest order, so re-scoring the same
    checkpoint over the same split reproduces the original epoch's predictions.

    Args:
        run: The run to score.
        config: Resolved project configuration supplying the data pipeline.
        checkpoint: Which checkpoint to score.
        split: Split to evaluate. **The test split is refused.**

    Returns:
        Counts indexed ``[true, predicted]``.

    Raises:
        ResultsError: If ``split`` is the test split, or the checkpoint is
            absent.
    """
    if split not in ("train", "validation"):
        raise ResultsError(
            f"refusing to score split {split!r}; the test split is reserved for "
            "Phase 9 and is never used to produce a figure"
        )

    path = run.run_dir / checkpoint
    if not path.is_file():
        raise ResultsError(f"checkpoint not found: {path}")

    import torch

    from farm_pest_ai.data.loaders import build_loaders
    from farm_pest_ai.vision.checkpoints import load_checkpoint
    from farm_pest_ai.vision.metrics import confusion_matrix

    # The run's OWN preprocessing, not the ambient configuration. A run trained
    # at 224x224 scored through a 160x160 pipeline loads without complaint —
    # `strict_preprocessing` defaults off — and yields a plausible but wrong
    # matrix. Rebuilding from the summary makes the figure match the model.
    preprocessing = run.preprocessing_config()
    bundle = build_loaders(config, (split,), preprocessing=preprocessing)

    model, _, _ = load_checkpoint(
        path,
        scope=config.scope,
        map_location="cpu",
        preprocessing_fingerprint=(
            preprocessing.fingerprint if preprocessing is not None else None
        ),
        strict_preprocessing=preprocessing is not None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for images, labels in bundle.loaders[split]:
            logits = model(images.to(device))
            predictions.extend(int(v) for v in logits.argmax(dim=1).cpu())
            targets.extend(int(v) for v in labels.cpu())

    matrix = confusion_matrix(
        torch.tensor(predictions), torch.tensor(targets), config.num_classes
    )
    return [[int(v) for v in row] for row in matrix]


def compare_runs(
    runs: Sequence[RunResults], *, corrected: bool = True
) -> list[dict[str, Any]]:
    """Summarise several runs for a comparison table.

    Args:
        runs: Runs to summarise.
        corrected: Select and report the corrected macro F1.

    Returns:
        One row per run, ordered as given.
    """
    rows: list[dict[str, Any]] = []
    for run in runs:
        best = run.best_validation(corrected=corrected)
        rows.append(
            {
                "run_id": run.run_id,
                "model": run.model_name,
                "scope": run.scope,
                "parameters": run.parameters,
                "epochs": len(run.epochs),
                "best_epoch": run.best_epoch(corrected=corrected),
                "reported_macro_f1": None if best is None else best.reported_macro_f1,
                "corrected_macro_f1": None if best is None else best.corrected_macro_f1,
                "accuracy": None if best is None else best.accuracy,
                "balanced_accuracy": None if best is None else best.balanced_accuracy,
                "best_epoch_moved": run.best_epoch_moved,
            }
        )
    return rows
