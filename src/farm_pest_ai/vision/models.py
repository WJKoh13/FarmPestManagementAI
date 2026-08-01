"""The project's CNN architectures.

Two models, both assembled from the primitives in
:mod:`farm_pest_ai.vision.blocks`:

``baseline_cnn``
    Model A. A deliberately plain stack of conv-BN-ReLU stages with max pooling.
    It is the control that Model B has to beat; if the custom architecture
    cannot outperform this, the extra complexity is not earning its place.

``custom_cnn``
    Model B. A stem followed by stages of residual depthwise-separable blocks
    with squeeze-and-excitation and stochastic depth.

Three protocol rules are enforced structurally rather than by convention:

* the input is ``(N, 3, 160, 160)`` and a mismatched channel count raises,
* the output is ``num_classes`` **raw logits**, with no softmax anywhere inside
  the model, because the training losses expect logits,
* ``num_classes`` is never hard-coded or inferred locally. It arrives from
  :mod:`farm_pest_ai.scopes` through :func:`build_model`, which derives it from
  the active scope and refuses a value that contradicts it.

``torchvision.models``, prebuilt architectures, pretrained weights and
downloaded checkpoints are prohibited and are not imported anywhere here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ..scopes import ScopeSpec, resolve_scope
from .blocks import (
    ACTIVATIONS,
    NORMS,
    ConvBNAct,
    ResidualSeparableBlock,
    build_activation,
    init_weights,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from ..config import Config

__all__ = [
    "MODEL_NAMES",
    "BaselineCNN",
    "CustomCNN",
    "ModelConfig",
    "ModelError",
    "build_model",
    "count_parameters",
    "model_config_from_config",
    "summarize_model",
]

#: Architectures selectable from ``model.name``.
MODEL_NAMES: tuple[str, ...] = ("baseline_cnn", "custom_cnn")

#: The only input channel count the project supports; the preprocessing layer
#: converts every image to RGB precisely so this holds.
INPUT_CHANNELS = 3


class ModelError(ValueError):
    """Raised when a model configuration is malformed or inconsistent."""


# -- configuration ------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """The resolved ``model`` section for one run.

    ``num_classes`` is present here because a constructed model must know its
    output width, but it is always supplied by :func:`build_model` from the
    active scope. Nothing in this module derives it independently.

    Attributes:
        name: Architecture name, one of :data:`MODEL_NAMES`.
        num_classes: Output logits. Derived from ``dataset.scope``.
        stem_channels: Width of the initial convolution.
        stage_channels: Output channels per stage; the length sets the depth.
        stage_blocks: Blocks per stage. ``custom_cnn`` only.
        stage_strides: Downsampling stride per stage. ``custom_cnn`` only.
        block: Residual block type. ``custom_cnn`` only.
        se_ratio: Squeeze-and-excitation ratio; ``0`` disables it.
        dropout: Dropout probability before the classifier.
        drop_path: Maximum stochastic-depth probability, scaled linearly with
            depth so early blocks are dropped less often than late ones.
        activation: Activation name.
        norm: Normalisation name.
        head: Pooling used before the classifier.
    """

    name: str = "custom_cnn"
    num_classes: int = 10
    stem_channels: int = 32
    stage_channels: tuple[int, ...] = (64, 128, 256, 384)
    stage_blocks: tuple[int, ...] = (2, 2, 3, 2)
    stage_strides: tuple[int, ...] = (2, 2, 2, 2)
    block: str = "residual_separable"
    se_ratio: float = 0.25
    dropout: float = 0.3
    drop_path: float = 0.1
    activation: str = "silu"
    norm: str = "batchnorm"
    head: str = "global_avg_pool"

    def validate(self) -> ModelConfig:
        """Check every field for consistency.

        Returns:
            ``self``, so the call can be chained.

        Raises:
            ModelError: On the first inconsistency found.
        """
        if self.name not in MODEL_NAMES:
            raise ModelError(
                f"unknown model.name {self.name!r}; expected one of {list(MODEL_NAMES)}"
            )
        if self.num_classes < 2:
            raise ModelError(
                f"model needs at least two classes, got {self.num_classes}"
            )
        if self.stem_channels <= 0:
            raise ModelError(
                f"model.stem_channels must be positive, got {self.stem_channels}"
            )
        if not self.stage_channels:
            raise ModelError("model.stage_channels must list at least one stage")
        if any(c <= 0 for c in self.stage_channels):
            raise ModelError(
                f"model.stage_channels must be positive, got {list(self.stage_channels)}"
            )
        if self.activation not in ACTIVATIONS:
            raise ModelError(
                f"unknown model.activation {self.activation!r}; expected one of "
                f"{list(ACTIVATIONS)}"
            )
        if self.norm not in NORMS:
            raise ModelError(
                f"unknown model.norm {self.norm!r}; expected one of {list(NORMS)}"
            )
        if self.head not in ("global_avg_pool", "global_max_pool"):
            raise ModelError(
                f"unknown model.head {self.head!r}; expected 'global_avg_pool' or "
                f"'global_max_pool'"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ModelError(f"model.dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.drop_path < 1.0:
            raise ModelError(f"model.drop_path must be in [0, 1), got {self.drop_path}")
        if self.se_ratio < 0.0:
            raise ModelError(f"model.se_ratio must be non-negative, got {self.se_ratio}")

        if self.name == "custom_cnn":
            stages = len(self.stage_channels)
            if len(self.stage_blocks) != stages:
                raise ModelError(
                    f"model.stage_blocks has {len(self.stage_blocks)} entries but "
                    f"model.stage_channels defines {stages} stages"
                )
            if len(self.stage_strides) != stages:
                raise ModelError(
                    f"model.stage_strides has {len(self.stage_strides)} entries but "
                    f"model.stage_channels defines {stages} stages"
                )
            if any(b <= 0 for b in self.stage_blocks):
                raise ModelError(
                    f"model.stage_blocks must be positive, got {list(self.stage_blocks)}"
                )
            if any(s <= 0 for s in self.stage_strides):
                raise ModelError(
                    f"model.stage_strides must be positive, got "
                    f"{list(self.stage_strides)}"
                )
            if self.block != "residual_separable":
                raise ModelError(
                    f"unknown model.block {self.block!r}; expected 'residual_separable'"
                )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping, stored with every checkpoint."""
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "stem_channels": self.stem_channels,
            "stage_channels": list(self.stage_channels),
            "stage_blocks": list(self.stage_blocks),
            "stage_strides": list(self.stage_strides),
            "block": self.block,
            "se_ratio": self.se_ratio,
            "dropout": self.dropout,
            "drop_path": self.drop_path,
            "activation": self.activation,
            "norm": self.norm,
            "head": self.head,
        }


def _as_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    """Coerce a sequence of integers into a tuple."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return tuple(int(v) for v in values)
    raise ModelError(f"{name} must be a list of integers, got {value!r}")


def model_config_from_config(
    config: Config, *, num_classes: int | None = None
) -> ModelConfig:
    """Build a :class:`ModelConfig` from a resolved :class:`Config`.

    ``num_classes`` comes from ``config.num_classes``, which
    :mod:`farm_pest_ai.scopes` derives from ``dataset.scope``. A ``model``
    section that states its own ``num_classes`` is a hard error rather than a
    silently honoured override, exactly as it is under ``dataset``: two places
    stating the class count is how a 10-way checkpoint ends up being read as a
    102-way one.

    Args:
        config: Resolved project configuration.
        num_classes: Override the derived count. Reserved for tests that build a
            model without a full configuration; production callers omit it.

    Returns:
        The validated model configuration.

    Raises:
        ModelError: If the section is malformed or states ``num_classes``.
    """
    section = config.section("model")
    defaults = ModelConfig()

    stated = section.get("num_classes")
    if stated is not None:
        raise ModelError(
            f"model.num_classes={stated!r} must not be stated in configuration; it is "
            f"derived from dataset.scope, which defines {config.num_classes} classes"
        )

    resolved = ModelConfig(
        name=str(section.get("name", defaults.name)),
        num_classes=int(num_classes if num_classes is not None else config.num_classes),
        stem_channels=int(section.get("stem_channels", defaults.stem_channels)),
        stage_channels=_as_int_tuple(
            section.get("stage_channels", defaults.stage_channels),
            "model.stage_channels",
        ),
        stage_blocks=_as_int_tuple(
            section.get("stage_blocks", defaults.stage_blocks), "model.stage_blocks"
        ),
        stage_strides=_as_int_tuple(
            section.get("stage_strides", defaults.stage_strides), "model.stage_strides"
        ),
        block=str(section.get("block", defaults.block)),
        se_ratio=float(section.get("se_ratio", defaults.se_ratio)),
        dropout=float(section.get("dropout", defaults.dropout)),
        drop_path=float(section.get("drop_path", defaults.drop_path)),
        activation=str(section.get("activation", defaults.activation)),
        norm=str(section.get("norm", defaults.norm)),
        head=str(section.get("head", defaults.head)),
    )
    return resolved.validate()


# -- architectures ------------------------------------------------------


class _ClassifierMixin(nn.Module):
    """Shared input checking and head construction for both architectures."""

    num_classes: int

    def _build_head(
        self, in_channels: int, num_classes: int, *, head: str, dropout: float
    ) -> None:
        """Attach global pooling, dropout and the linear classifier.

        The classifier emits raw logits. No softmax is applied here or anywhere
        else in the model: the losses expect logits, and the inference policy in
        Phase 9 owns the conversion to probabilities.
        """
        self.pool: nn.Module = (
            nn.AdaptiveAvgPool2d(1)
            if head == "global_avg_pool"
            else nn.AdaptiveMaxPool2d(1)
        )
        self.flatten = nn.Flatten(1)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(in_channels, num_classes)

    def _check_input(self, x: Tensor) -> None:
        """Reject a malformed input batch with a diagnosable message.

        Raises:
            ModelError: If the batch is not 4-D or does not carry exactly three
                channels. The three-channel rule is the same one the
                preprocessing layer guarantees; checking it here too means a
                four-channel tensor from a hand-built pipeline fails loudly at
                the model boundary instead of at a shape mismatch deep inside a
                convolution.
        """
        if x.ndim != 4:
            raise ModelError(
                f"expected a 4-D (N, C, H, W) batch, got shape {tuple(x.shape)}"
            )
        if x.shape[1] != INPUT_CHANNELS:
            raise ModelError(
                f"expected {INPUT_CHANNELS} input channels, got {x.shape[1]}; every "
                f"image must be converted to RGB before it reaches the model"
            )

    def _classify(self, features: Tensor) -> Tensor:
        """Pool, drop and project features to raw logits."""
        pooled = self.flatten(self.pool(features))
        return self.classifier(self.dropout(pooled))


class BaselineCNN(_ClassifierMixin):
    """Model A: a plain conv-BN-ReLU stack with max pooling.

    Each stage is two ``3x3`` convolutions followed by a ``2x2`` max pool, so a
    160x160 input is halved once per stage. Deliberately unremarkable: it exists
    to establish the score that the custom architecture must beat, and any
    cleverness here would weaken that comparison.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the network from a validated :class:`ModelConfig`."""
        super().__init__()
        config.validate()
        self.config = config
        self.num_classes = config.num_classes

        layers: list[nn.Module] = []
        in_channels = INPUT_CHANNELS
        for out_channels in config.stage_channels:
            layers.append(
                ConvBNAct(
                    in_channels,
                    out_channels,
                    3,
                    norm=config.norm,
                    activation=config.activation,
                )
            )
            layers.append(
                ConvBNAct(
                    out_channels,
                    out_channels,
                    3,
                    norm=config.norm,
                    activation=config.activation,
                )
            )
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self._build_head(
            in_channels, config.num_classes, head=config.head, dropout=config.dropout
        )
        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw logits of shape ``(N, num_classes)``."""
        self._check_input(x)
        return self._classify(self.features(x))


class CustomCNN(_ClassifierMixin):
    """Model B: residual depthwise-separable stages with SE and stochastic depth.

    A strided stem halves the input once, then each stage applies its first
    block at ``stage_strides[i]`` and the remainder at stride 1. Stochastic
    depth ramps linearly from 0 at the first block to ``drop_path`` at the last,
    the usual schedule: dropping early blocks as often as late ones removes
    low-level features the whole network depends on.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the network from a validated :class:`ModelConfig`."""
        super().__init__()
        config.validate()
        self.config = config
        self.num_classes = config.num_classes

        self.stem = ConvBNAct(
            INPUT_CHANNELS,
            config.stem_channels,
            3,
            stride=2,
            norm=config.norm,
            activation=config.activation,
        )

        total_blocks = sum(config.stage_blocks)
        built = 0
        stages: list[nn.Module] = []
        in_channels = config.stem_channels

        for out_channels, blocks, stride in zip(
            config.stage_channels, config.stage_blocks, config.stage_strides, strict=True
        ):
            stage: list[nn.Module] = []
            for index in range(blocks):
                # Linear ramp: the first block of the network gets 0, the last
                # gets the configured maximum.
                ratio = built / max(1, total_blocks - 1)
                stage.append(
                    ResidualSeparableBlock(
                        in_channels,
                        out_channels,
                        stride=stride if index == 0 else 1,
                        se_ratio=config.se_ratio,
                        drop_path=config.drop_path * ratio,
                        norm=config.norm,
                        activation=config.activation,
                    )
                )
                in_channels = out_channels
                built += 1
            stages.append(nn.Sequential(*stage))

        self.stages = nn.Sequential(*stages)
        self.head_act = build_activation(config.activation)
        self._build_head(
            in_channels, config.num_classes, head=config.head, dropout=config.dropout
        )
        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw logits of shape ``(N, num_classes)``."""
        self._check_input(x)
        return self._classify(self.head_act(self.stages(self.stem(x))))


# -- construction and inspection ----------------------------------------


def build_model(
    config: Config | ModelConfig,
    *,
    scope: str | ScopeSpec | None = None,
) -> nn.Module:
    """Construct the configured architecture.

    This is the sanctioned entry point. Given a project :class:`Config` it
    derives ``num_classes`` from ``dataset.scope`` through
    :mod:`farm_pest_ai.scopes`, so a model can never be built with a class count
    that disagrees with the data it will be trained on.

    Args:
        config: Either a resolved project configuration, or an already-built
            :class:`ModelConfig` for tests and checkpoint reconstruction.
        scope: Cross-check the derived class count against this scope. Supplying
            it when loading a checkpoint turns a scope mismatch into an error
            instead of a silently wrong 102-way read of a 10-way head.

    Returns:
        The constructed model, in ``train`` mode with initialised weights.

    Raises:
        ModelError: If the configuration is invalid or contradicts ``scope``.
    """
    resolved = (
        config if isinstance(config, ModelConfig) else model_config_from_config(config)
    )
    resolved.validate()

    if scope is not None:
        spec = resolve_scope(scope)
        if spec.num_classes != resolved.num_classes:
            raise ModelError(
                f"model has {resolved.num_classes} output classes but scope "
                f"{spec.name!r} defines {spec.num_classes}; a checkpoint may never be "
                f"reinterpreted under a different scope"
            )

    if resolved.name == "baseline_cnn":
        return BaselineCNN(resolved)
    if resolved.name == "custom_cnn":
        return CustomCNN(resolved)
    raise ModelError(
        f"unknown model.name {resolved.name!r}; expected one of {list(MODEL_NAMES)}"
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters.

    Returns:
        ``total``, ``trainable`` and ``buffers``. Buffers are counted separately
        because BatchNorm's running statistics occupy checkpoint space but are
        not optimised.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return {"total": total, "trainable": trainable, "buffers": buffers}


@dataclass(frozen=True)
class _StageShape:
    """One recorded feature-map shape, used by :func:`summarize_model`."""

    name: str
    shape: tuple[int, ...] = field(default_factory=tuple)


def summarize_model(
    model: nn.Module,
    *,
    input_size: tuple[int, int] = (160, 160),
    batch_size: int = 2,
    device: str = "cpu",
) -> dict[str, Any]:
    """Describe a model by running one batch of zeros through it.

    Recorded with every run so a checkpoint carries the exact parameter count
    and output width it was trained with. The forward pass runs under
    ``no_grad`` in ``eval`` mode, so it neither allocates gradients nor perturbs
    BatchNorm's running statistics.

    Args:
        model: The model to inspect.
        input_size: ``(height, width)`` of the probe batch.
        batch_size: Probe batch size. At least 2, so BatchNorm is exercised the
            way it will be during training.
        device: Device to run the probe on.

    Returns:
        A JSON-serialisable summary including parameter counts, the output
        shape, and the estimated parameter memory in MiB.

    Raises:
        ModelError: If the model does not return ``(batch_size, num_classes)``.
    """
    parameters = count_parameters(model)
    was_training = model.training
    model.eval()
    probe = torch.zeros(
        (max(2, batch_size), INPUT_CHANNELS, *input_size), device=device
    )
    try:
        with torch.no_grad():
            logits = model(probe.to(device))
    finally:
        model.train(was_training)

    if logits.ndim != 2 or logits.shape[0] != probe.shape[0]:
        raise ModelError(
            f"model returned shape {tuple(logits.shape)}; expected "
            f"({probe.shape[0]}, num_classes)"
        )

    config = getattr(model, "config", None)
    # 4 bytes per fp32 parameter; buffers are included because they are written
    # to the checkpoint alongside the weights.
    bytes_per_element = 4
    memory_mib = (
        (parameters["total"] + parameters["buffers"]) * bytes_per_element / 2**20
    )

    return {
        "name": type(model).__name__,
        "config": config.to_dict() if isinstance(config, ModelConfig) else None,
        "parameters": parameters,
        "parameter_memory_mib": round(memory_mib, 2),
        "input_shape": [INPUT_CHANNELS, *input_size],
        "output_shape": list(logits.shape[1:]),
        "num_classes": int(logits.shape[1]),
    }


def describe_architecture(model: nn.Module) -> Mapping[str, Any]:
    """Return the model's configuration mapping, or an empty one.

    Convenience for callers that hold an ``nn.Module`` and want its provenance
    without knowing which architecture it is.
    """
    config = getattr(model, "config", None)
    return config.to_dict() if isinstance(config, ModelConfig) else {}
