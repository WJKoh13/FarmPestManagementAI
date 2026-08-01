"""Checkpoint writing, reading and provenance enforcement.

A checkpoint here is never just weights. It carries the dataset scope, the
number of classes, the class-mapping version, the manifest and preprocessing
versions, the resolved model configuration, the training seed, the environment
snapshot and the Git revision. :func:`load_checkpoint` **verifies** that
provenance before returning: loading a ``rice10`` checkpoint under ``full102``
raises rather than reinterpreting ten output units as the first ten of 102.

That check is the whole reason this module exists. A silently mismatched
checkpoint does not crash — it produces confident, wrong pest identifications,
which is the single most damaging failure this project can have.

Writes are atomic. ``torch.save`` to the final path can leave a truncated file
if the process dies mid-write, and a half-written ``best.pt`` discovered at the
end of a multi-hour run is an expensive way to learn that lesson.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..scopes import CLASS_MAPPING_VERSION, ScopeSpec, resolve_scope
from .models import ModelConfig, build_model

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from torch.optim import Optimizer

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointError",
    "CheckpointMetadata",
    "best_checkpoint_path",
    "last_checkpoint_path",
    "load_checkpoint",
    "load_model_for_inference",
    "read_metadata",
    "save_checkpoint",
]

#: Bumped when the on-disk checkpoint layout changes incompatibly.
CHECKPOINT_FORMAT_VERSION = "1.0.0"

#: Filenames used by the training engine.
BEST_CHECKPOINT_NAME = "best.pt"
LAST_CHECKPOINT_NAME = "last.pt"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, malformed or provenance-mismatched."""


@dataclass(frozen=True)
class CheckpointMetadata:
    """Everything recorded alongside the weights.

    Attributes:
        scope: Dataset scope name the model was trained for.
        num_classes: Output width, always equal to the scope's class count.
        class_mapping_version: Version of the project-label mapping.
        manifest_version: Version of the derived manifests used.
        preprocessing_version: Version stamp of the preprocessing.
        preprocessing_fingerprint: Hash of the resolved preprocessing pipeline.
        model: The resolved model configuration.
        epoch: Epoch this checkpoint was written at.
        global_step: Optimiser steps completed.
        metrics: Validation metrics at this epoch.
        best_metric: Best monitored value seen so far.
        monitor: Name of the monitored metric.
        seed: The run's global seed.
        run_id: Identifier of the run directory.
        environment: Interpreter, library and GPU snapshot.
        config: The fully resolved run configuration.
        format_version: On-disk layout version.
        smoke: Whether this came from a smoke run whose metrics are meaningless.
    """

    scope: str
    num_classes: int
    class_mapping_version: str = CLASS_MAPPING_VERSION
    manifest_version: str = "1.0.0"
    preprocessing_version: str = "1.0.0"
    preprocessing_fingerprint: str = ""
    model: dict[str, Any] = field(default_factory=dict)
    epoch: int = 0
    global_step: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    best_metric: float | None = None
    monitor: str = "macro_f1"
    seed: int = 1337
    run_id: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    format_version: str = CHECKPOINT_FORMAT_VERSION
    smoke: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "scope": self.scope,
            "num_classes": self.num_classes,
            "class_mapping_version": self.class_mapping_version,
            "manifest_version": self.manifest_version,
            "preprocessing_version": self.preprocessing_version,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "model": dict(self.model),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "metrics": dict(self.metrics),
            "best_metric": self.best_metric,
            "monitor": self.monitor,
            "seed": self.seed,
            "run_id": self.run_id,
            "environment": dict(self.environment),
            "config": dict(self.config),
            "format_version": self.format_version,
            "smoke": self.smoke,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointMetadata:
        """Rebuild metadata from a checkpoint's stored mapping.

        Raises:
            CheckpointError: If a required provenance field is absent. These are
                required precisely because their absence is what would allow a
                mismatched checkpoint through.
        """
        for required in ("scope", "num_classes"):
            if payload.get(required) is None:
                raise CheckpointError(
                    f"checkpoint metadata is missing {required!r}; it cannot be "
                    f"verified against the active scope and will not be loaded"
                )
        return cls(
            scope=str(payload["scope"]),
            num_classes=int(payload["num_classes"]),
            class_mapping_version=str(
                payload.get("class_mapping_version", CLASS_MAPPING_VERSION)
            ),
            manifest_version=str(payload.get("manifest_version", "1.0.0")),
            preprocessing_version=str(payload.get("preprocessing_version", "1.0.0")),
            preprocessing_fingerprint=str(payload.get("preprocessing_fingerprint", "")),
            model=dict(payload.get("model") or {}),
            epoch=int(payload.get("epoch", 0)),
            global_step=int(payload.get("global_step", 0)),
            metrics=dict(payload.get("metrics") or {}),
            best_metric=(
                float(payload["best_metric"])
                if payload.get("best_metric") is not None
                else None
            ),
            monitor=str(payload.get("monitor", "macro_f1")),
            seed=int(payload.get("seed", 1337)),
            run_id=str(payload.get("run_id", "")),
            environment=dict(payload.get("environment") or {}),
            config=dict(payload.get("config") or {}),
            format_version=str(payload.get("format_version", CHECKPOINT_FORMAT_VERSION)),
            smoke=bool(payload.get("smoke", False)),
        )

    def model_config(self) -> ModelConfig:
        """Reconstruct the :class:`ModelConfig` this checkpoint was trained with.

        Raises:
            CheckpointError: If the stored model section is absent or unusable.
        """
        if not self.model:
            raise CheckpointError(
                "checkpoint carries no model configuration, so the architecture "
                "cannot be rebuilt"
            )
        payload = dict(self.model)
        stored_classes = payload.pop("num_classes", self.num_classes)
        if int(stored_classes) != self.num_classes:
            raise CheckpointError(
                f"checkpoint is internally inconsistent: metadata says "
                f"{self.num_classes} classes but the model section says "
                f"{stored_classes}"
            )
        known = {f for f in ModelConfig.__dataclass_fields__ if f != "num_classes"}
        unknown = set(payload) - known
        if unknown:
            raise CheckpointError(
                f"checkpoint model configuration has unknown field(s) "
                f"{sorted(unknown)}; it was written by an incompatible version"
            )
        for key in ("stage_channels", "stage_blocks", "stage_strides"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return ModelConfig(num_classes=self.num_classes, **payload).validate()

    def verify_against(
        self,
        scope: str | ScopeSpec,
        *,
        preprocessing_fingerprint: str | None = None,
        manifest_version: str | None = None,
        strict_preprocessing: bool = False,
    ) -> None:
        """Check this checkpoint may be used with the given scope.

        Args:
            scope: The scope the caller intends to use the model under.
            preprocessing_fingerprint: Fingerprint of the caller's preprocessing.
                A difference means the model would see differently-processed
                pixels than it was trained on.
            manifest_version: The caller's manifest version.
            strict_preprocessing: Whether a preprocessing mismatch is an error
                rather than a caller-visible difference. Phase 9 sets this; a
                mid-development comparison may not.

        Raises:
            CheckpointError: If the scope, class count, class-mapping version or
                (when strict) preprocessing does not match.
        """
        spec = resolve_scope(scope)
        if self.scope != spec.name:
            raise CheckpointError(
                f"checkpoint was trained for scope {self.scope!r} but {spec.name!r} was "
                f"requested; a model may never be used under a different scope"
            )
        if self.num_classes != spec.num_classes:
            raise CheckpointError(
                f"checkpoint has {self.num_classes} output classes but scope "
                f"{spec.name!r} defines {spec.num_classes}"
            )
        if self.class_mapping_version != CLASS_MAPPING_VERSION:
            raise CheckpointError(
                f"checkpoint was trained under class mapping "
                f"{self.class_mapping_version!r} but this build uses "
                f"{CLASS_MAPPING_VERSION!r}; the project labels no longer mean the "
                f"same thing, so its predictions cannot be interpreted"
            )
        if manifest_version is not None and self.manifest_version != manifest_version:
            raise CheckpointError(
                f"checkpoint used manifest version {self.manifest_version!r} but "
                f"{manifest_version!r} is configured"
            )
        if (
            strict_preprocessing
            and preprocessing_fingerprint is not None
            and self.preprocessing_fingerprint
            and self.preprocessing_fingerprint != preprocessing_fingerprint
        ):
            raise CheckpointError(
                f"checkpoint was trained with preprocessing fingerprint "
                f"{self.preprocessing_fingerprint!r} but the active pipeline is "
                f"{preprocessing_fingerprint!r}; the model would see different pixels "
                f"than it was trained on"
            )


def best_checkpoint_path(run_dir: Path) -> Path:
    """Path of the best-metric checkpoint inside a run directory."""
    return Path(run_dir) / BEST_CHECKPOINT_NAME


def last_checkpoint_path(run_dir: Path) -> Path:
    """Path of the most recent checkpoint inside a run directory."""
    return Path(run_dir) / LAST_CHECKPOINT_NAME


def save_checkpoint(
    path: Path,
    model: nn.Module,
    metadata: CheckpointMetadata,
    *,
    optimizer: Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    rng_state: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint atomically.

    The optimiser, scheduler, AMP scaler and RNG states are stored so that a run
    can be resumed exactly. Without the optimiser's moment estimates, a resumed
    AdamW run restarts its adaptivity from scratch and produces a visible loss
    spike; without the RNG state, the augmentation stream diverges.

    Args:
        path: Destination file.
        model: The model whose ``state_dict`` is saved.
        metadata: Provenance to embed.
        optimizer: Optimiser to save, when resuming matters.
        scheduler: Learning-rate scheduler to save.
        scaler: AMP gradient scaler to save.
        rng_state: Captured RNG state from :func:`capture_rng_state`.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "metadata": metadata.to_dict(),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if rng_state is not None:
        payload["rng_state"] = rng_state

    # Same atomic pattern as the manifest writer: a torch.save interrupted
    # partway through the final file would leave a truncated checkpoint that
    # only fails when it is loaded, potentially hours later. The fsync matters
    # more here than for a manifest, since a checkpoint can represent hours of
    # GPU time that a power loss would otherwise discard.
    handle, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def read_metadata(path: Path) -> CheckpointMetadata:
    """Read only the metadata from a checkpoint.

    Loads onto the CPU, so a registry listing does not need a GPU.

    Raises:
        CheckpointError: If the file is missing or carries no metadata.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError) as exc:
        raise CheckpointError(f"failed to read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise CheckpointError(
            f"{path} is not a project checkpoint: it carries no metadata block"
        )
    return CheckpointMetadata.from_dict(payload["metadata"])


def load_checkpoint(
    path: Path,
    *,
    scope: str | ScopeSpec | None = None,
    model: nn.Module | None = None,
    map_location: str = "cpu",
    preprocessing_fingerprint: str | None = None,
    manifest_version: str | None = None,
    strict_preprocessing: bool = False,
) -> tuple[nn.Module, CheckpointMetadata, dict[str, Any]]:
    """Load a checkpoint, verifying its provenance first.

    The verification happens **before** any weight is copied into a model, so a
    mismatched checkpoint cannot leave a half-populated network behind.

    Args:
        path: Checkpoint file.
        scope: Scope the caller intends to use. When given, a mismatch raises.
        model: Existing model to load weights into. When ``None`` the
            architecture is rebuilt from the checkpoint's own configuration.
        map_location: Device to map tensors onto.
        preprocessing_fingerprint: Caller's preprocessing fingerprint.
        manifest_version: Caller's manifest version.
        strict_preprocessing: Whether a preprocessing mismatch raises.

    Returns:
        The model with weights loaded, the metadata, and the remaining payload
        (optimiser, scheduler, scaler and RNG states) for a resuming caller.

    Raises:
        CheckpointError: If the file is missing, malformed, provenance-mismatched
            or its weights do not fit the architecture.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, EOFError) as exc:
        raise CheckpointError(f"failed to read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise CheckpointError(
            f"{path} is not a project checkpoint: it carries no metadata block"
        )

    metadata = CheckpointMetadata.from_dict(payload["metadata"])
    if scope is not None:
        metadata.verify_against(
            scope,
            preprocessing_fingerprint=preprocessing_fingerprint,
            manifest_version=manifest_version,
            strict_preprocessing=strict_preprocessing,
        )

    if model is None:
        model = build_model(metadata.model_config(), scope=scope)
    elif getattr(model, "num_classes", metadata.num_classes) != metadata.num_classes:
        raise CheckpointError(
            f"target model has {getattr(model, 'num_classes', '?')} output classes but "
            f"the checkpoint has {metadata.num_classes}"
        )

    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise CheckpointError(f"{path} carries no model weights")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(
            f"checkpoint weights do not fit the architecture: {exc}"
        ) from exc

    extras = {k: v for k, v in payload.items() if k not in ("metadata", "model_state")}
    return model, metadata, extras


def load_model_for_inference(
    path: Path,
    scope: str | ScopeSpec,
    *,
    device: str = "cpu",
    preprocessing_fingerprint: str | None = None,
    strict_preprocessing: bool = True,
) -> tuple[nn.Module, CheckpointMetadata]:
    """Load a checkpoint for inference, in ``eval`` mode on ``device``.

    ``scope`` is required rather than optional: an inference caller always knows
    which scope it is serving, and making the check unavoidable is the point.
    ``strict_preprocessing`` defaults to true here for the same reason — serving
    a model with preprocessing it was not trained under is a silent accuracy
    loss, not a warning.

    Returns:
        The model in ``eval`` mode and its metadata.

    Raises:
        CheckpointError: If provenance does not match.
    """
    model, metadata, _ = load_checkpoint(
        path,
        scope=scope,
        map_location=device,
        preprocessing_fingerprint=preprocessing_fingerprint,
        strict_preprocessing=strict_preprocessing,
    )
    model.to(device)
    model.eval()
    return model, metadata


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy and PyTorch RNG state for exact resumption."""
    import random

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        state["numpy"] = np.random.get_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`.

    Missing entries are skipped rather than raising: a checkpoint written on a
    CUDA machine and resumed on a CPU one is a legitimate case, and the CPU
    streams are still restored exactly.
    """
    import random

    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        # A different GPU count on resume; CPU streams are still exact.
        with contextlib.suppress(RuntimeError, ValueError):
            torch.cuda.set_rng_state_all(
                [_as_byte_tensor(s) for s in state["torch_cuda"]]
            )
    if "numpy" in state:
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            np.random.set_state(state["numpy"])


def _as_byte_tensor(value: Any) -> torch.Tensor:
    """Coerce a stored RNG state back into the ByteTensor torch expects."""
    if isinstance(value, torch.Tensor):
        return value.cpu().to(torch.uint8)
    return torch.tensor(bytearray(value), dtype=torch.uint8)


def write_metadata_sidecar(path: Path, metadata: CheckpointMetadata) -> Path:
    """Write a checkpoint's metadata as readable JSON next to it.

    The sidecar is a convenience for inspection and for the Phase 9 registry; it
    is never the authority. :func:`load_checkpoint` always reads the metadata
    embedded in the checkpoint itself, so deleting or editing a sidecar cannot
    make a mismatched model loadable.
    """
    from ..data.manifests import atomic_write_text

    return atomic_write_text(
        Path(path),
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
    )
