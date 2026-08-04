"""The training engine: one epoch loop, checkpointing, early stopping, logging.

Phase 6 builds the machinery and proves it works on a smoke run. Real
experiments are Phases 7 and 8; nothing here selects a hyperparameter.

The rules from ``docs/TRAINING.md`` that this module is responsible for:

* AdamW with cosine decay and linear warmup, unless configuration says otherwise
* mixed precision when CUDA is available, with a gradient scaler
* **validation macro F1** as the monitored metric for both early stopping and
  best-checkpoint selection
* separate ``best.pt`` and ``last.pt``, both resumable
* the fully resolved configuration, seed, environment and preprocessing
  fingerprint recorded with the run
* structured JSON Lines metrics, one record per epoch

Two things this engine will not do. It never touches the test split — the loader
layer already omits it unless named explicitly, and nothing here names it. And
it never computes a class statistic from anything but the training split; the
class weights it applies to the loss arrive pre-computed from
:func:`~farm_pest_ai.data.loaders.build_loaders`, which enforces that.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

from ..data.loaders import LoaderBundle
from ..logging_config import get_logger
from ..reproducibility import environment_snapshot
from ..scopes import CLASS_MAPPING_VERSION
from .checkpoints import (
    CheckpointError,
    CheckpointMetadata,
    best_checkpoint_path,
    capture_rng_state,
    last_checkpoint_path,
    load_checkpoint,
    restore_rng_state,
    write_metadata_sidecar,
)
from .finegrained import (
    FineGrainedConfig,
    ProjectionHead,
    fine_grained_config_from_config,
    supervised_contrastive_loss,
)
from .metrics import ClassificationMetrics, MetricsAccumulator
from .mixing import BatchMixer, MixingConfig, mixed_criterion, mixing_config_from_config
from .models import ModelConfig, build_model, model_config_from_config, summarize_model

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from ..config import Config

__all__ = [
    "EpochResult",
    "TrainPassResult",
    "Trainer",
    "TrainingConfig",
    "TrainingError",
    "build_optimizer",
    "build_scheduler",
    "build_trainer",
    "iter_epoch_records",
    "training_config_from_config",
]

logger = get_logger("training")


class TrainingError(RuntimeError):
    """Raised when a training run cannot start or cannot continue safely."""


@dataclass(frozen=True)
class TrainingConfig:
    """The resolved ``training`` section for one run.

    Attributes:
        optimizer: Optimiser name. AdamW unless justified in writing.
        learning_rate: Peak learning rate, reached at the end of warmup.
        weight_decay: Decoupled weight decay. Not applied to norm and bias
            parameters, see :func:`build_optimizer`.
        batch_size: Images per optimiser step.
        epochs: Maximum epochs; early stopping may end the run sooner.
        scheduler: Learning-rate schedule name.
        warmup_epochs: Epochs of linear warmup before the main schedule.
        label_smoothing: Cross-entropy label smoothing.
        class_weighting: Scheme name, resolved by the loader from training data.
        class_weighting_beta: Strength parameter of the ``effective`` scheme,
            ignored by every other scheme. It governs that scheme entirely: on
            full102's 82x imbalance, 0.9999 yields a 69.5x weight ratio and
            0.999 yields 23.5x, so it is the difference between a nearly-maximal
            correction and a moderate one.
        grad_clip_norm: Gradient-norm clip; ``0`` disables it.
        early_stopping_metric: Monitored metric name.
        early_stopping_mode: ``"max"`` or ``"min"``.
        early_stopping_patience: Epochs without improvement before stopping.
        early_stopping_min_delta: Improvement below this does not count.
        save_best: Whether to write ``best.pt``.
        save_last: Whether to write ``last.pt``.
        monitor: Metric that decides which checkpoint is best.
        amp: Whether to use mixed precision when CUDA is available.
        mixing: Training-only MixUp/CutMix settings. Defaults to ``none``, so a
            configuration that says nothing about mixing behaves exactly as it
            did before E7 existed.
    """

    optimizer: str = "adamw"
    learning_rate: float = 0.002
    weight_decay: float = 0.05
    batch_size: int = 64
    epochs: int = 80
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    class_weighting: str = "none"
    class_weighting_beta: float = 0.9999
    grad_clip_norm: float = 1.0
    early_stopping_metric: str = "macro_f1"
    early_stopping_mode: str = "max"
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    save_best: bool = True
    save_last: bool = True
    monitor: str = "macro_f1"
    amp: bool = True
    mixing: MixingConfig = field(default_factory=MixingConfig)
    fine_grained: FineGrainedConfig = field(default_factory=FineGrainedConfig)

    def validate(self) -> TrainingConfig:
        """Check every field.

        Returns:
            ``self``, so the call can be chained.

        Raises:
            TrainingError: On the first inconsistency found.
        """
        if self.optimizer not in ("adamw", "adam", "sgd"):
            raise TrainingError(
                f"unknown training.optimizer {self.optimizer!r}; expected 'adamw', "
                f"'adam' or 'sgd'"
            )
        if self.scheduler not in ("cosine", "step", "none"):
            raise TrainingError(
                f"unknown training.scheduler {self.scheduler!r}; expected 'cosine', "
                f"'step' or 'none'"
            )
        if self.learning_rate <= 0:
            raise TrainingError(
                f"training.learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0:
            raise TrainingError(
                f"training.weight_decay must be non-negative, got {self.weight_decay}"
            )
        if self.batch_size <= 0:
            raise TrainingError(
                f"training.batch_size must be positive, got {self.batch_size}"
            )
        if self.epochs <= 0:
            raise TrainingError(f"training.epochs must be positive, got {self.epochs}")
        if self.warmup_epochs < 0:
            raise TrainingError(
                f"training.warmup_epochs must be non-negative, got {self.warmup_epochs}"
            )
        if self.warmup_epochs >= self.epochs:
            raise TrainingError(
                f"training.warmup_epochs={self.warmup_epochs} leaves no epochs for the "
                f"main schedule, which runs for training.epochs={self.epochs}"
            )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise TrainingError(
                f"training.label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )
        if not 0.0 <= self.class_weighting_beta < 1.0:
            raise TrainingError(
                f"training.class_weighting_beta must be in [0, 1), got "
                f"{self.class_weighting_beta}"
            )
        if self.grad_clip_norm < 0:
            raise TrainingError(
                f"training.grad_clip_norm must be non-negative, got {self.grad_clip_norm}"
            )
        if self.early_stopping_mode not in ("max", "min"):
            raise TrainingError(
                f"training.early_stopping.mode must be 'max' or 'min', got "
                f"{self.early_stopping_mode!r}"
            )
        if self.early_stopping_patience < 1:
            raise TrainingError(
                f"training.early_stopping.patience must be at least 1, got "
                f"{self.early_stopping_patience}"
            )
        self.mixing.validate()
        self.fine_grained.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "scheduler": self.scheduler,
            "warmup_epochs": self.warmup_epochs,
            "label_smoothing": self.label_smoothing,
            "class_weighting": self.class_weighting,
            "class_weighting_beta": self.class_weighting_beta,
            "grad_clip_norm": self.grad_clip_norm,
            "early_stopping": {
                "metric": self.early_stopping_metric,
                "mode": self.early_stopping_mode,
                "patience": self.early_stopping_patience,
                "min_delta": self.early_stopping_min_delta,
            },
            "checkpoint": {
                "save_best": self.save_best,
                "save_last": self.save_last,
                "monitor": self.monitor,
            },
            "amp": self.amp,
            "mixing": self.mixing.to_dict(),
            "fine_grained": self.fine_grained.to_dict(),
        }


def training_config_from_config(config: Config) -> TrainingConfig:
    """Build a :class:`TrainingConfig` from the ``training`` section."""
    section = config.section("training")
    defaults = TrainingConfig()

    early = section.get("early_stopping") or {}
    if not isinstance(early, Mapping):
        raise TrainingError(
            f"training.early_stopping must be a mapping, got {early!r}"
        )
    checkpoint = section.get("checkpoint") or {}
    if not isinstance(checkpoint, Mapping):
        raise TrainingError(
            f"training.checkpoint must be a mapping, got {checkpoint!r}"
        )

    resolved = TrainingConfig(
        optimizer=str(section.get("optimizer", defaults.optimizer)),
        learning_rate=float(section.get("learning_rate", defaults.learning_rate)),
        weight_decay=float(section.get("weight_decay", defaults.weight_decay)),
        batch_size=int(section.get("batch_size", defaults.batch_size)),
        epochs=int(section.get("epochs", defaults.epochs)),
        scheduler=str(section.get("scheduler", defaults.scheduler)),
        warmup_epochs=int(section.get("warmup_epochs", defaults.warmup_epochs)),
        label_smoothing=float(section.get("label_smoothing", defaults.label_smoothing)),
        class_weighting=str(section.get("class_weighting", defaults.class_weighting)),
        class_weighting_beta=float(
            section.get("class_weighting_beta", defaults.class_weighting_beta)
        ),
        grad_clip_norm=float(section.get("grad_clip_norm", defaults.grad_clip_norm)),
        early_stopping_metric=str(
            early.get("metric", defaults.early_stopping_metric)
        ),
        early_stopping_mode=str(early.get("mode", defaults.early_stopping_mode)),
        early_stopping_patience=int(
            early.get("patience", defaults.early_stopping_patience)
        ),
        early_stopping_min_delta=float(
            early.get("min_delta", defaults.early_stopping_min_delta)
        ),
        save_best=bool(checkpoint.get("save_best", defaults.save_best)),
        save_last=bool(checkpoint.get("save_last", defaults.save_last)),
        monitor=str(checkpoint.get("monitor", defaults.monitor)),
        amp=bool(config.get("runtime.amp", defaults.amp)),
        mixing=mixing_config_from_config(config),
        fine_grained=fine_grained_config_from_config(config),
    )
    return resolved.validate()


# -- optimiser and schedule ---------------------------------------------


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Build the optimiser with weight decay applied selectively.

    Normalisation parameters and biases are placed in a **zero weight-decay**
    group. Decaying a BatchNorm scale pulls it toward zero, which suppresses the
    channel it normalises; decaying a bias just shifts the decision boundary for
    no regularisation benefit. Only weight matrices and convolution kernels are
    decayed, which is the standard treatment and matters more here because
    ``weight_decay`` is 0.05, high enough for the difference to show.

    Args:
        model: The model whose parameters are optimised.
        config: Resolved training configuration.

    Returns:
        The configured optimiser.

    Raises:
        TrainingError: If the optimiser name is unsupported.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if config.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=config.learning_rate)
    if config.optimizer == "adam":
        return torch.optim.Adam(groups, lr=config.learning_rate)
    if config.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=config.learning_rate, momentum=0.9, nesterov=True)
    raise TrainingError(f"unsupported optimizer {config.optimizer!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig, steps_per_epoch: int
) -> Any:
    """Build a per-step learning-rate schedule with linear warmup.

    Stepping per batch rather than per epoch makes the warmup smooth: an
    epoch-level warmup on ``rice10``, which has only ~67 batches per epoch at
    batch size 64, would jump the learning rate in a handful of large steps.

    Args:
        optimizer: The optimiser to schedule.
        config: Resolved training configuration.
        steps_per_epoch: Optimiser steps in one epoch.

    Returns:
        A ``LambdaLR`` whose multiplier starts near 0, reaches 1 at the end of
        warmup, and then follows the configured decay.
    """
    warmup_steps = max(0, config.warmup_epochs * steps_per_epoch)
    total_steps = max(1, config.epochs * steps_per_epoch)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            # +1 so the very first step has a non-zero learning rate; starting
            # at exactly 0 wastes a step and can leave BatchNorm statistics
            # initialised from an untrained forward pass.
            return (step + 1) / warmup_steps
        if config.scheduler == "none":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        if config.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if config.scheduler == "step":
            # Decay by 10x at each third of the post-warmup schedule.
            return 0.1 ** int(progress * 3)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


# -- results -------------------------------------------------------------


@dataclass(frozen=True)
class TrainPassResult:
    """What one training pass produced, beyond its metrics.

    Attributes:
        metrics: Metrics accumulated over the augmented training batches.
        seconds: Wall-clock time of the pass.
        images_per_second: Training throughput.
        optimizer_steps: Optimiser steps that actually applied.
        amp_skipped_steps: Steps the gradient scaler skipped because the
            gradients overflowed at the current loss scale.
        amp_final_scale: The loss scale at the end of the pass, or ``None`` when
            AMP is off.
    """

    metrics: ClassificationMetrics
    seconds: float
    images_per_second: float
    optimizer_steps: int = 0
    amp_skipped_steps: int = 0
    amp_final_scale: float | None = None


@dataclass(frozen=True)
class EpochResult:
    """Everything recorded for one completed epoch.

    Attributes:
        epoch: One-based epoch number.
        train: Metrics over the training split, computed from the same augmented
            batches the model actually trained on.
        validation: Metrics over the validation split.
        learning_rate: Learning rate at the end of the epoch.
        train_seconds: Wall-clock time of the training pass.
        validation_seconds: Wall-clock time of the validation pass.
        images_per_second: Training throughput.
        peak_vram_mib: Peak CUDA memory during the epoch, or ``None`` on CPU.
        improved: Whether the monitored metric improved this epoch.
        best_metric: Best monitored value seen so far.
        optimizer_steps: Optimiser steps that actually applied this epoch.
        amp_skipped_steps: Batches whose optimiser step AMP skipped because the
            gradients overflowed at the current loss scale. A handful during the
            first epoch is normal scale calibration; a count that stays high, or
            reappears late in a run, means the loss scale is fighting the
            gradients and those batches contributed no learning at all.
        amp_final_scale: The gradient scaler's loss scale at the end of the
            epoch, or ``None`` when AMP is off. Recorded alongside the skip
            count because the two only make sense together: a scale that keeps
            halving is what produces skips.
    """

    epoch: int
    train: ClassificationMetrics
    validation: ClassificationMetrics
    learning_rate: float
    train_seconds: float
    validation_seconds: float
    images_per_second: float
    peak_vram_mib: float | None = None
    improved: bool = False
    best_metric: float | None = None
    optimizer_steps: int = 0
    amp_skipped_steps: int = 0
    amp_final_scale: float | None = None

    def to_dict(self, *, per_class: bool = False) -> dict[str, Any]:
        """Return the JSON Lines record written for this epoch."""
        return {
            "epoch": self.epoch,
            "train": self.train.to_dict(per_class=per_class),
            "validation": self.validation.to_dict(per_class=per_class),
            "learning_rate": self.learning_rate,
            "train_seconds": round(self.train_seconds, 3),
            "validation_seconds": round(self.validation_seconds, 3),
            "images_per_second": round(self.images_per_second, 1),
            "peak_vram_mib": self.peak_vram_mib,
            "improved": self.improved,
            "best_metric": self.best_metric,
            "optimizer_steps": self.optimizer_steps,
            "amp_skipped_steps": self.amp_skipped_steps,
            "amp_final_scale": self.amp_final_scale,
        }


@dataclass
class _EarlyStopping:
    """Tracks the monitored metric and decides when to stop."""

    metric: str
    mode: str
    patience: int
    min_delta: float
    best: float | None = None
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def update(self, value: float, epoch: int) -> bool:
        """Record a new value; return whether it is an improvement."""
        if self.best is None:
            self.best, self.best_epoch = value, epoch
            self.epochs_without_improvement = 0
            return True
        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta
        if improved:
            self.best, self.best_epoch = value, epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved

    @property
    def should_stop(self) -> bool:
        """Whether patience has been exhausted."""
        return self.epochs_without_improvement >= self.patience


# -- the trainer ---------------------------------------------------------


class Trainer:
    """Runs the training loop for one experiment.

    Owns the model, optimiser, schedule, checkpointing and the per-epoch metric
    record. It does not own the data: a :class:`LoaderBundle` is passed in, which
    is what keeps the test-split exclusion and the training-only-statistics rule
    enforced in one place rather than two.
    """

    def __init__(
        self,
        model: nn.Module,
        bundle: LoaderBundle,
        config: TrainingConfig,
        *,
        run_dir: Path,
        run_id: str = "",
        resolved_config: Mapping[str, Any] | None = None,
        model_config: ModelConfig | None = None,
        manifest_version: str = "1.0.0",
        smoke: bool = False,
        max_train_batches: int | None = None,
        max_validation_batches: int | None = None,
    ) -> None:
        """Build a trainer.

        Args:
            model: The model to train.
            bundle: Loaders and datasets. Must contain ``train``; ``validation``
                is required for early stopping and best-checkpoint selection.
            config: Resolved training configuration.
            run_dir: Directory for checkpoints, metrics and the resolved config.
            run_id: Identifier for this run, embedded in every checkpoint.
            resolved_config: The full resolved configuration to record.
            model_config: The model configuration to record. Read from the model
                when omitted.
            manifest_version: Manifest version to record.
            smoke: Marks the run's checkpoints and metrics as a smoke test whose
                numbers are meaningless. Recorded so a smoke checkpoint can never
                be mistaken for an experiment result.
            max_train_batches: Stop each training epoch after this many batches.
                Smoke runs only.
            max_validation_batches: Score at most this many batches, spread
                evenly across the split. Smoke runs only, and its metrics are
                still not comparable with a full pass. Batches are strided
                rather than taken from the front because evaluation loaders
                preserve official manifest order, which is grouped by class: the
                first N batches of ``rice10`` validation are entirely class 0,
                so a front-truncated pass reports macro F1 0.0 for any model and
                would mask a genuine regression.

        Raises:
            TrainingError: If the bundle has no training loader, or the model's
                output width disagrees with the scope.
        """
        if "train" not in bundle.loaders:
            raise TrainingError(
                "the loader bundle has no training split; a trainer cannot run "
                "without one"
            )
        model_classes = getattr(model, "num_classes", None)
        if model_classes is not None and model_classes != bundle.num_classes:
            raise TrainingError(
                f"model produces {model_classes} classes but the data bundle is scope "
                f"{bundle.scope.name!r} with {bundle.num_classes}; refusing to train a "
                f"model against labels it cannot represent"
            )

        self.model = model
        self.bundle = bundle
        self.config = config.validate()
        self.run_dir = Path(run_dir)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.resolved_config = dict(resolved_config or {})
        self.manifest_version = manifest_version
        self.smoke = bool(smoke)
        self.max_train_batches = max_train_batches
        self.max_validation_batches = max_validation_batches

        self.device = torch.device(bundle.device)
        self.model.to(self.device)

        stored = model_config or getattr(model, "config", None)
        self.model_config: ModelConfig | None = (
            stored if isinstance(stored, ModelConfig) else None
        )

        self.criterion = self._build_criterion()
        # The projection head is training-only scaffolding. It is built before
        # the optimiser so its parameters are optimised, and it is deliberately
        # NOT part of `self.model`: keeping it separate means the saved
        # state_dict is byte-compatible with a model trained without the
        # auxiliary objective, so an E8 checkpoint loads into the ordinary
        # inference path with no special case.
        self.projection: ProjectionHead | None = None
        if self.config.fine_grained.enabled:
            feature_dim = getattr(self.model, "feature_dim", None)
            if feature_dim is None:
                raise TrainingError(
                    "training.fine_grained is enabled but the model exposes no "
                    "feature_dim; the auxiliary objective needs an embedding path"
                )
            self.projection = ProjectionHead(
                int(feature_dim), self.config.fine_grained.embedding_dim
            ).to(self.device)

        self.optimizer = build_optimizer(self.model, self.config)
        if self.projection is not None:
            # The head trains under the same schedule and weight decay policy as
            # the backbone, so the auxiliary term is one variable rather than a
            # second optimiser configuration.
            self.optimizer.add_param_group(
                {
                    "params": list(self.projection.parameters()),
                    "weight_decay": self.config.weight_decay,
                }
            )
        # Seeded from the run seed, so a mixed run is as reproducible as an
        # unmixed one, and owning its own generator keeps it from consuming the
        # global RNG stream the dataloader workers draw augmentations from.
        self.mixer = BatchMixer(self.config.mixing, seed=bundle.seed)

        train_loader = bundle.loaders["train"]
        self.steps_per_epoch = max(1, len(train_loader))
        if max_train_batches is not None:
            self.steps_per_epoch = max(1, min(self.steps_per_epoch, max_train_batches))
        self.scheduler = build_scheduler(
            self.optimizer, self.config, self.steps_per_epoch
        )

        # AMP only pays off on CUDA. On CPU the autocast path adds casts without
        # tensor cores to exploit, so it is disabled regardless of configuration.
        self.amp_enabled = bool(self.config.amp) and self.device.type == "cuda"
        # torch's default init_scale of 65536 overflows on the first few steps of
        # this model and the scaler halves it each time, skipping those optimiser
        # steps entirely. On a full run that is a negligible warmup cost, but on a
        # short capped run it consumes most of the budget and makes the model look
        # like it never learned. Starting at 2**12 reaches a usable scale in one
        # or two steps instead of four or five, and costs nothing thereafter since
        # the scaler grows back on its own.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled, init_scale=2.0**12
        )

        self.early_stopping = _EarlyStopping(
            metric=self.config.early_stopping_metric,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )
        self.start_epoch = 1
        self.global_step = 0
        self.history: list[EpochResult] = []
        # Cumulative across the run, so a resumed run's totals are per-epoch in
        # `metrics.jsonl` and the run total is the sum of those records.
        self.amp_skipped_steps = 0

    # -- setup ---------------------------------------------------------

    def _build_criterion(self) -> nn.Module:
        """Build the loss, applying training-derived class weights if present.

        The weights come from :class:`LoaderBundle`, which computes them from the
        training split alone. This method never derives them itself, so there is
        no path by which validation labels could reach the loss.
        """
        weight: Tensor | None = None
        if self.bundle.class_weights is not None:
            weight = torch.tensor(
                self.bundle.class_weights, dtype=torch.float32, device=self.device
            )
            if weight.numel() != self.bundle.num_classes:
                raise TrainingError(
                    f"class weights have {weight.numel()} entries but the scope defines "
                    f"{self.bundle.num_classes} classes"
                )
        return nn.CrossEntropyLoss(
            weight=weight, label_smoothing=self.config.label_smoothing
        )

    def _metadata(
        self, epoch: int, metrics: ClassificationMetrics | None
    ) -> CheckpointMetadata:
        """Assemble the provenance block embedded in every checkpoint."""
        return CheckpointMetadata(
            scope=self.bundle.scope.name,
            num_classes=self.bundle.num_classes,
            class_mapping_version=CLASS_MAPPING_VERSION,
            manifest_version=self.manifest_version,
            preprocessing_version=self.bundle.preprocessing.version,
            preprocessing_fingerprint=self.bundle.preprocessing.fingerprint,
            model=self.model_config.to_dict() if self.model_config else {},
            epoch=epoch,
            global_step=self.global_step,
            metrics=metrics.to_dict(per_class=True) if metrics else {},
            best_metric=self.early_stopping.best,
            monitor=self.config.monitor,
            seed=self.bundle.seed,
            run_id=self.run_id,
            environment=environment_snapshot(),
            config=self.resolved_config,
            smoke=self.smoke,
        )

    # -- passes --------------------------------------------------------

    def train_epoch(self, epoch: int) -> TrainPassResult:
        """Run one training pass.

        Metrics are accumulated from the same augmented batches the model trains
        on, so training accuracy is measured under augmentation and will read
        lower than a clean pass over the same data. That is the honest number:
        it describes what the model actually saw.

        When ``training.mixing`` is enabled the model trains on mixed images and
        a lambda-weighted loss, but metrics are still accumulated against the
        **original hard labels**. A mixed run's training accuracy therefore
        remains directly comparable with an unmixed run's, and reads lower
        because the images are genuinely harder rather than because the target
        changed.

        Returns:
            A :class:`TrainPassResult` carrying the metrics, timing and the AMP
            step accounting for this epoch.
        """
        loader = self.bundle.loaders["train"]
        self.model.train()
        if self.projection is not None:
            self.projection.train()
        accumulator = MetricsAccumulator(
            self.bundle.num_classes, device=self.device
        )
        started = time.perf_counter()
        images_seen = 0
        stepped_count = 0
        skipped_count = 0

        for batch_index, (images, targets) in enumerate(loader):
            if self.max_train_batches is not None and batch_index >= self.max_train_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # set_to_none frees the gradient buffers rather than filling them
            # with zeros; it is both faster and avoids a stale-gradient class of
            # bug where a parameter that gets no gradient keeps its previous one.
            self.optimizer.zero_grad(set_to_none=True)

            # Training-only mixing. When disabled this returns the batch
            # untouched and `mixed_criterion` reduces to the plain criterion
            # call, so the pre-E7 path is preserved exactly rather than merely
            # reproduced.
            mixed = self.mixer.apply(images, targets, training=True)

            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                if self.projection is None:
                    logits = self.model(mixed.images)
                    loss = mixed_criterion(self.criterion, logits, mixed)
                else:
                    # One backbone pass feeds both terms. Two passes would cost
                    # twice as much and, with stochastic depth active, would
                    # describe two different random subnetworks.
                    # Cast because ``nn.Module.__getattr__`` is typed as
                    # returning ``Tensor | Module``, which would make the call
                    # look invalid. The attribute is guaranteed present: the
                    # constructor already required ``feature_dim`` before
                    # building the projection head.
                    combined = cast(
                        "Callable[[Tensor], tuple[Tensor, Tensor]]",
                        self.model.forward_logits_and_features,
                    )
                    logits, features = combined(mixed.images)
                    loss = mixed_criterion(self.criterion, logits, mixed)
                    # The contrastive term is applied to UNMIXED batches only.
                    # A mixed image has no single label, so contrasting it
                    # against a hard label would supervise the embedding with a
                    # target the pixels do not support. When mixing is off —
                    # which is the case for every E8 arm — this is every batch.
                    if not mixed.mixed:
                        auxiliary = supervised_contrastive_loss(
                            self.projection(features),
                            mixed.hard_targets,
                            temperature=self.config.fine_grained.temperature,
                        )
                        loss = loss + self.config.fine_grained.weight * auxiliary

            if not torch.isfinite(loss):
                raise TrainingError(
                    f"epoch {epoch}, batch {batch_index}: loss is {loss.item()}; "
                    f"training cannot continue with a non-finite loss"
                )

            self.scaler.scale(loss).backward()
            if self.config.grad_clip_norm > 0:
                # Gradients must be unscaled before their norm is meaningful.
                self.scaler.unscale_(self.optimizer)
                # The projection head is optimised, so it must be clipped too;
                # leaving it out would let the auxiliary term produce an
                # unbounded update while the backbone stayed clipped.
                clipped = list(self.model.parameters())
                if self.projection is not None:
                    clipped.extend(self.projection.parameters())
                torch.nn.utils.clip_grad_norm_(clipped, self.config.grad_clip_norm)
            # With AMP the scaler skips the optimiser step on any batch whose
            # gradients overflowed, which routinely includes the first one while
            # the scale is being calibrated. Advancing the schedule on a step
            # that did not happen would desynchronise the learning rate from the
            # optimiser, so the scale is compared before and after to detect it.
            scale_before = self.scaler.get_scale() if self.amp_enabled else None
            self.scaler.step(self.optimizer)
            self.scaler.update()
            stepped = (
                scale_before is None or self.scaler.get_scale() >= scale_before
            )
            if stepped:
                # torch warns when the scheduler steps before the optimiser ever
                # has. That is exactly the AMP-calibration case handled above:
                # the guard means the schedule is already correctly aligned, so
                # the warning here is noise rather than signal.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*lr_scheduler\.step\(\) before.*",
                        category=UserWarning,
                    )
                    self.scheduler.step()
                self.global_step += 1
                stepped_count += 1
            else:
                skipped_count += 1

            # Scored against the ORIGINAL hard labels, never a mixed target, so
            # training accuracy stays comparable with every historical run and a
            # mixed epoch's curve means the same thing as an unmixed one's.
            accumulator.update(
                logits.detach().float(),
                mixed.hard_targets,
                loss=float(loss.detach()),
            )
            images_seen += int(images.shape[0])

        elapsed = time.perf_counter() - started
        rate = images_seen / elapsed if elapsed > 0 else 0.0
        self.amp_skipped_steps += skipped_count
        return TrainPassResult(
            metrics=accumulator.compute(),
            seconds=elapsed,
            images_per_second=rate,
            optimizer_steps=stepped_count,
            amp_skipped_steps=skipped_count,
            amp_final_scale=(
                float(self.scaler.get_scale()) if self.amp_enabled else None
            ),
        )

    @torch.no_grad()
    def evaluate(self, split: str = "validation") -> tuple[ClassificationMetrics, float]:
        """Score one split without updating any parameter.

        ``eval`` mode disables dropout and stochastic depth and switches
        BatchNorm to its running statistics, so this pass is deterministic given
        deterministic preprocessing — which Phase 5 verified it is.

        Args:
            split: The split to score. Never ``"test"`` before Phase 9; the
                bundle does not carry a test loader unless one was requested
                explicitly.

        Returns:
            The metrics and the elapsed seconds.

        Raises:
            TrainingError: If the bundle has no loader for ``split``.
        """
        loader = self.bundle.loaders.get(split)
        if loader is None:
            raise TrainingError(
                f"the loader bundle has no {split!r} split; it was not requested when "
                f"the loaders were built"
            )

        self.model.eval()
        accumulator = MetricsAccumulator(self.bundle.num_classes, device=self.device)
        started = time.perf_counter()

        # Evaluation loaders keep official manifest order, which is grouped by
        # class. Taking the first N batches would therefore score one or two
        # classes only. Striding covers the whole label range instead.
        stride = 1
        if self.max_validation_batches is not None:
            stride = max(1, len(loader) // max(1, self.max_validation_batches))

        scored = 0
        for batch_index, (images, targets) in enumerate(loader):
            if self.max_validation_batches is not None:
                if batch_index % stride != 0:
                    continue
                if scored >= self.max_validation_batches:
                    break
                scored += 1
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            accumulator.update(logits.float(), targets, loss=float(loss))

        return accumulator.compute(), time.perf_counter() - started

    # -- the loop ------------------------------------------------------

    def fit(self, epochs: int | None = None) -> list[EpochResult]:
        """Run the training loop.

        Writes ``last.pt`` every epoch and ``best.pt`` whenever the monitored
        metric improves, appends one JSON Lines record per epoch to
        ``metrics.jsonl``, and stops early when patience is exhausted.

        Args:
            epochs: Override ``training.epochs``.

        Returns:
            One :class:`EpochResult` per completed epoch.
        """
        total_epochs = int(epochs if epochs is not None else self.config.epochs)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.run_dir / "metrics.jsonl"

        self._write_run_record()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(self.start_epoch, total_epochs + 1):
            train_pass = self.train_epoch(epoch)
            train_metrics = train_pass.metrics
            validation_metrics, validation_seconds = self.evaluate("validation")

            monitored = validation_metrics.get(self.config.early_stopping_metric)
            improved = self.early_stopping.update(monitored, epoch)

            peak_vram = None
            if self.device.type == "cuda":
                peak_vram = round(
                    torch.cuda.max_memory_allocated(self.device) / 2**20, 1
                )

            result = EpochResult(
                epoch=epoch,
                train=train_metrics,
                validation=validation_metrics,
                learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                train_seconds=train_pass.seconds,
                validation_seconds=validation_seconds,
                images_per_second=train_pass.images_per_second,
                peak_vram_mib=peak_vram,
                improved=improved,
                best_metric=self.early_stopping.best,
                optimizer_steps=train_pass.optimizer_steps,
                amp_skipped_steps=train_pass.amp_skipped_steps,
                amp_final_scale=train_pass.amp_final_scale,
            )
            self.history.append(result)
            self._append_metrics(metrics_path, result)

            logger.info(
                "epoch %d/%d loss=%.4f val_macro_f1=%.4f val_acc=%.4f lr=%.2e "
                "amp_skipped=%d%s",
                epoch,
                total_epochs,
                train_metrics.loss if train_metrics.loss is not None else float("nan"),
                validation_metrics.macro_f1,
                validation_metrics.accuracy,
                result.learning_rate,
                train_pass.amp_skipped_steps,
                " *" if improved else "",
                extra={
                    "event": "epoch",
                    "epoch": epoch,
                    "scope": self.bundle.scope.name,
                    "macro_f1": validation_metrics.macro_f1,
                    "improved": improved,
                    "optimizer_steps": train_pass.optimizer_steps,
                    "amp_skipped_steps": train_pass.amp_skipped_steps,
                    "amp_skipped_steps_total": self.amp_skipped_steps,
                    "amp_final_scale": train_pass.amp_final_scale,
                },
            )

            self._save_checkpoints(epoch, validation_metrics, improved)

            if self.early_stopping.should_stop:
                logger.info(
                    "early stopping at epoch %d; %s has not improved for %d epochs "
                    "(best %.4f at epoch %d)",
                    epoch,
                    self.config.early_stopping_metric,
                    self.early_stopping.epochs_without_improvement,
                    self.early_stopping.best or 0.0,
                    self.early_stopping.best_epoch,
                    extra={"event": "early_stop", "epoch": epoch},
                )
                break

        return self.history

    def _save_checkpoints(
        self, epoch: int, metrics: ClassificationMetrics, improved: bool
    ) -> None:
        """Write ``last.pt``, and ``best.pt`` when the metric improved."""
        from .checkpoints import save_checkpoint

        metadata = self._metadata(epoch, metrics)
        rng_state = capture_rng_state()

        if self.config.save_last:
            save_checkpoint(
                last_checkpoint_path(self.run_dir),
                self.model,
                metadata,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler if self.amp_enabled else None,
                rng_state=rng_state,
            )
        if self.config.save_best and improved:
            path = save_checkpoint(
                best_checkpoint_path(self.run_dir),
                self.model,
                metadata,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler if self.amp_enabled else None,
                rng_state=rng_state,
            )
            write_metadata_sidecar(path.with_suffix(".json"), metadata)

    def _append_metrics(self, path: Path, result: EpochResult) -> None:
        """Append one epoch record to the JSON Lines metrics file.

        Per-class arrays are included only for ``rice10``: 102 classes times four
        arrays per epoch would make a ``full102`` log unreadable, and the final
        per-class breakdown is preserved in the checkpoint metadata anyway.
        """
        per_class = self.bundle.num_classes <= 10
        payload = result.to_dict(per_class=per_class)
        payload["run_id"] = self.run_id
        payload["scope"] = self.bundle.scope.name
        payload["smoke"] = self.smoke
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def _write_run_record(self) -> None:
        """Write the resolved configuration and environment for this run.

        Produced before the first epoch, so an interrupted run still leaves
        behind enough to identify exactly what it was doing.
        """
        record = {
            "run_id": self.run_id,
            "scope": self.bundle.scope.name,
            "num_classes": self.bundle.num_classes,
            "class_mapping_version": CLASS_MAPPING_VERSION,
            "manifest_version": self.manifest_version,
            "smoke": self.smoke,
            "started": datetime.now(timezone.utc).isoformat(),
            "device": str(self.device),
            "amp_enabled": self.amp_enabled,
            "steps_per_epoch": self.steps_per_epoch,
            "training": self.config.to_dict(),
            "model": self.model_config.to_dict() if self.model_config else {},
            "model_summary": summarize_model(
                self.model,
                input_size=self.bundle.preprocessing.image_size,
                device=str(self.device),
            ),
            "data": self.bundle.describe(),
            "config": self.resolved_config,
            "environment": environment_snapshot(),
        }
        path = self.run_dir / "run.json"
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    # -- resumption ----------------------------------------------------

    def resume(self, path: Path | None = None) -> int:
        """Resume from a checkpoint, restoring optimiser, schedule and RNG state.

        The checkpoint's scope is verified against the bundle's before anything
        is restored, so a ``rice10`` checkpoint cannot be resumed into a
        ``full102`` run.

        Args:
            path: Checkpoint to resume from. Defaults to ``last.pt`` in the run
                directory.

        Returns:
            The epoch training will resume at.

        Raises:
            TrainingError: If the checkpoint is missing or mismatched.
        """
        resolved = Path(path) if path is not None else last_checkpoint_path(self.run_dir)
        try:
            _, metadata, extras = load_checkpoint(
                resolved,
                scope=self.bundle.scope,
                model=self.model,
                map_location=str(self.device),
                preprocessing_fingerprint=self.bundle.preprocessing.fingerprint,
            )
        except CheckpointError as exc:
            raise TrainingError(f"cannot resume: {exc}") from exc

        if "optimizer_state" in extras:
            self.optimizer.load_state_dict(extras["optimizer_state"])
        if "scheduler_state" in extras:
            self.scheduler.load_state_dict(extras["scheduler_state"])
        if "scaler_state" in extras and self.amp_enabled:
            self.scaler.load_state_dict(extras["scaler_state"])
        if "rng_state" in extras:
            restore_rng_state(extras["rng_state"])

        self.global_step = metadata.global_step
        self.start_epoch = metadata.epoch + 1
        if metadata.best_metric is not None:
            self.early_stopping.best = metadata.best_metric
            self.early_stopping.best_epoch = metadata.epoch

        logger.info(
            "resumed from %s at epoch %d (best %s=%.4f)",
            resolved,
            metadata.epoch,
            metadata.monitor,
            metadata.best_metric if metadata.best_metric is not None else 0.0,
            extra={"event": "resume", "epoch": metadata.epoch},
        )
        return self.start_epoch


def build_trainer(
    config: Config,
    bundle: LoaderBundle,
    *,
    run_dir: Path,
    run_id: str = "",
    smoke: bool = False,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> Trainer:
    """Build a model and trainer from a resolved project configuration.

    The model's class count is derived from ``dataset.scope`` and cross-checked
    against the bundle's scope, so the three places a class count could disagree
    — configuration, model and data — are reconciled at construction.
    """
    model_config = model_config_from_config(config)
    model = build_model(model_config, scope=bundle.scope)
    return Trainer(
        model,
        bundle,
        training_config_from_config(config),
        run_dir=run_dir,
        run_id=run_id,
        resolved_config=config.to_dict(),
        model_config=model_config,
        manifest_version=config.dataset.manifest_version,
        smoke=smoke,
        max_train_batches=max_train_batches,
        max_validation_batches=max_validation_batches,
    )


def iter_epoch_records(path: Path) -> Iterator[dict[str, Any]]:
    """Read a ``metrics.jsonl`` file back into dictionaries.

    Used by the smoke script's summary and by later reporting phases.
    """
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
