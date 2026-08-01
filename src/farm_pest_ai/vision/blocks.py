"""Primitive building blocks for the project's CNNs.

Everything here is written from primitive PyTorch layers: ``nn.Conv2d``,
``nn.BatchNorm2d``, pooling, activations, dropout and linear layers. The project
rules prohibit ``torchvision.models``, any prebuilt architecture, pretrained
weights and downloaded checkpoints, so the residual, depthwise-separable and
squeeze-and-excitation blocks below are implemented rather than imported.

Two conventions hold throughout:

Convolutions preceding a norm carry no bias
    BatchNorm subtracts a per-channel mean immediately afterwards, so a bias
    term would be cancelled and would only waste parameters and a kernel launch.

Nothing here applies a softmax
    Blocks emit activations and the head emits raw logits. The training losses
    expect logits, and applying softmax inside the model would silently change
    what the loss computes.

Torch is imported at module scope here, unlike the data layer: this module is
only reachable from a training or inference path, both of which require torch
anyway.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

__all__ = [
    "ACTIVATIONS",
    "NORMS",
    "ConvBNAct",
    "DepthwiseSeparableConv",
    "DropPath",
    "ResidualSeparableBlock",
    "SqueezeExcite",
    "build_activation",
    "build_norm",
    "init_weights",
]

#: Activation names selectable from configuration.
ACTIVATIONS: tuple[str, ...] = ("relu", "silu", "gelu", "leaky_relu")

#: Normalisation names selectable from configuration.
NORMS: tuple[str, ...] = ("batchnorm", "groupnorm", "none")


def build_activation(name: str, *, inplace: bool = True) -> nn.Module:
    """Construct an activation module by name.

    Args:
        name: One of :data:`ACTIVATIONS`.
        inplace: Whether the activation may overwrite its input. Safe for every
            activation used here, since none of them is followed by a consumer
            of the pre-activation tensor.

    Returns:
        The activation module.

    Raises:
        ValueError: If ``name`` is not a supported activation.
    """
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "gelu":
        # GELU has no in-place variant; the flag is accepted and ignored.
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=inplace)
    raise ValueError(f"unknown activation {name!r}; expected one of {list(ACTIVATIONS)}")


def build_norm(name: str, channels: int, *, groups: int = 8) -> nn.Module:
    """Construct a normalisation module by name.

    Args:
        name: One of :data:`NORMS`.
        channels: Number of channels to normalise.
        groups: Group count for ``groupnorm``. Reduced automatically when it
            does not divide ``channels``, so a narrow stage still builds.

    Returns:
        The normalisation module, or :class:`~torch.nn.Identity` for ``"none"``.

    Raises:
        ValueError: If ``name`` is not a supported normalisation.
    """
    if name == "batchnorm":
        return nn.BatchNorm2d(channels)
    if name == "groupnorm":
        divisor = groups
        while divisor > 1 and channels % divisor != 0:
            divisor -= 1
        return nn.GroupNorm(divisor, channels)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {name!r}; expected one of {list(NORMS)}")


class ConvBNAct(nn.Module):
    """Convolution, normalisation and activation in that order.

    The convolution's bias is omitted whenever a normalisation follows, since
    BatchNorm's own shift makes it redundant.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        norm: str = "batchnorm",
        activation: str | None = "relu",
    ) -> None:
        """Build the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Square kernel size.
            stride: Convolution stride.
            padding: Explicit padding. Defaults to ``kernel_size // 2``, which
                keeps the spatial size unchanged at stride 1 for odd kernels.
            groups: Convolution groups; ``in_channels`` makes it depthwise.
            norm: Normalisation name, see :data:`NORMS`.
            activation: Activation name, or ``None`` for a linear block. Used by
                the residual block, whose second convolution must stay linear so
                the skip connection is added before the non-linearity.
        """
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        has_norm = norm != "none"
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=not has_norm,
        )
        self.norm = build_norm(norm, out_channels)
        self.act: nn.Module = (
            build_activation(activation) if activation is not None else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution, normalisation and activation."""
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """A depthwise convolution followed by a pointwise projection.

    Factorising a ``k x k`` convolution this way costs roughly
    ``1/out_channels + 1/k**2`` of the parameters of the dense equivalent, which
    is what makes a deeper model fit in the ~4 GB of VRAM Phase 1 measured free.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        norm: str = "batchnorm",
        activation: str = "silu",
        pointwise_activation: str | None = None,
    ) -> None:
        """Build the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Depthwise kernel size.
            stride: Depthwise stride.
            norm: Normalisation name applied after both convolutions.
            activation: Activation after the depthwise convolution.
            pointwise_activation: Activation after the pointwise convolution, or
                ``None`` to leave the projection linear.
        """
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            groups=in_channels,
            norm=norm,
            activation=activation,
        )
        self.pointwise = ConvBNAct(
            in_channels,
            out_channels,
            1,
            norm=norm,
            activation=pointwise_activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the depthwise then pointwise convolution."""
        return self.pointwise(self.depthwise(x))


class SqueezeExcite(nn.Module):
    """Channel attention: pool to a vector, gate each channel by a learned scale.

    The gate is a sigmoid, so it can only attenuate or preserve a channel, never
    invert it. Implemented with ``1x1`` convolutions rather than linear layers so
    the block stays shape-agnostic.
    """

    def __init__(
        self, channels: int, ratio: float = 0.25, *, activation: str = "silu"
    ) -> None:
        """Build the block.

        Args:
            channels: Channels to gate.
            ratio: Bottleneck width as a fraction of ``channels``. Clamped to at
                least one channel so a narrow stage still builds.
            activation: Activation inside the bottleneck.
        """
        super().__init__()
        hidden = max(1, round(channels * ratio))
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = build_activation(activation)
        self.expand = nn.Conv2d(hidden, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Scale each channel by its learned gate."""
        # Mean over height and width, keeping dims so the gate broadcasts back.
        weights = x.mean(dim=(2, 3), keepdim=True)
        weights = self.gate(self.expand(self.act(self.reduce(weights))))
        return x * weights


class DropPath(nn.Module):
    """Stochastic depth: drop a residual branch for a whole sample.

    During training each sample independently keeps or discards the residual
    branch, and the survivors are scaled by ``1 / keep_prob`` so the expected
    activation is unchanged. At evaluation the branch is always kept, which is
    what makes the deterministic-evaluation guarantee hold for the model as well
    as for preprocessing.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Build the block.

        Args:
            drop_prob: Probability of dropping the branch for a given sample.

        Raises:
            ValueError: If ``drop_prob`` is outside ``[0, 1)``. A probability of
                exactly 1 would delete the branch entirely and is a
                configuration error, not a valid setting.
        """
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        """Drop the branch for a random subset of samples during training."""
        # Exact comparison is intended: drop_prob is stored verbatim from
        # configuration and 0.0 is the "disabled" sentinel, not a computed value.
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # One Bernoulli draw per sample, broadcast over C, H and W.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        """Show the drop probability in ``repr``."""
        return f"drop_prob={self.drop_prob}"


class ResidualSeparableBlock(nn.Module):
    """A residual block built from depthwise-separable convolutions.

    Structure::

        x -> DWSep(3x3, stride) -> DWSep(3x3) -> SE -> DropPath -> + shortcut -> act

    The second separable convolution's projection is left **linear**, so the
    shortcut is added before the final activation. That ordering is what lets
    the identity path carry an unmodified signal, which is the point of a
    residual block.

    The shortcut is an identity when the shape is unchanged, and a ``1x1``
    strided projection otherwise.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        se_ratio: float = 0.25,
        drop_path: float = 0.0,
        norm: str = "batchnorm",
        activation: str = "silu",
    ) -> None:
        """Build the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            stride: Stride of the first separable convolution.
            se_ratio: Squeeze-and-excitation bottleneck ratio; ``0`` disables it.
            drop_path: Stochastic-depth probability for the residual branch.
            norm: Normalisation name.
            activation: Activation name.
        """
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(
            in_channels,
            out_channels,
            3,
            stride=stride,
            norm=norm,
            activation=activation,
            pointwise_activation=activation,
        )
        self.conv2 = DepthwiseSeparableConv(
            out_channels,
            out_channels,
            3,
            stride=1,
            norm=norm,
            activation=activation,
            # Linear: the shortcut is added before the final activation.
            pointwise_activation=None,
        )
        self.se: nn.Module = (
            SqueezeExcite(out_channels, se_ratio, activation=activation)
            if se_ratio > 0
            else nn.Identity()
        )
        self.drop_path = DropPath(drop_path)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = ConvBNAct(
                in_channels,
                out_channels,
                1,
                stride=stride,
                norm=norm,
                activation=None,
            )
        else:
            self.shortcut = nn.Identity()

        self.act = build_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual branch and add the shortcut."""
        residual = self.se(self.conv2(self.conv1(x)))
        return self.act(self.drop_path(residual) + self.shortcut(x))


def init_weights(module: nn.Module) -> None:
    """Initialise one module's parameters in place.

    Intended for :meth:`torch.nn.Module.apply`. Convolutions use Kaiming normal
    with ``fan_out``, which keeps activation variance stable through a deep
    stack of ReLU-family non-linearities; normalisation layers start as the
    identity; linear layers use a small normal so the initial logits sit near
    zero, giving a starting loss close to ``ln(num_classes)``.

    Args:
        module: The module to initialise.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


ActivationFactory = Callable[[], nn.Module]
"""Type alias for a zero-argument activation constructor."""
