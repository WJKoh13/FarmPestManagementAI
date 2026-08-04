"""custom_cnn -- a residual depthwise-separable CNN with SE and stochastic depth.

Copied verbatim from ``notebooks/custom_cnn_ziyang.ipynb`` section 2, which is
the graded artifact and has to show the architecture inline. That duplication is
deliberate, and ``tests/test_custom_cnn.py`` fails if the two ever drift: it
execs the notebook cell and compares state-dict keys and parameter count.

The registry needs this module for a second reason. A legacy checkpoint trained
on the ``det_top15`` scope is imported by ``scripts/import_custom_cnn_run.py``,
and the app rebuilds the architecture from ``model_name`` alone -- a state dict
carries weights, not layer definitions. The parameter *names* below are
therefore load-bearing: they are what ``strict=True`` matches against, so
renaming an attribute silently invalidates every existing checkpoint.

Nothing here comes from ``torchvision.models``, and no weight is pretrained.
Everything is built from primitive PyTorch layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The submitted model is fifteen-class; see data_manifests/classes_top15.json.
# Never rely on this as a default in new code -- pass protocol.num_classes.
NUM_CLASSES = 15


class ConvBNAct(nn.Module):
    """Convolution, normalization and activation, in that order.

    The convolution's bias is omitted whenever a norm follows: BatchNorm
    subtracts a per-channel mean immediately afterwards, so a bias term would be
    cancelled and would only waste parameters.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, *,
                 stride: int = 1, groups: int = 1, activation: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=kernel_size // 2, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        # `activation=False` leaves the block linear. The residual block's second
        # projection needs that, so the skip is added *before* the non-linearity.
        self.act: nn.Module = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """A depthwise convolution followed by a pointwise projection.

    Factorizing a k x k convolution this way costs roughly ``1/out_channels +
    1/k**2`` of the parameters of the dense equivalent. That is what buys the
    depth here: the whole network is 1.4M parameters, an order of magnitude
    under ProPestNet, while going deeper.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, *,
                 stride: int = 1, pointwise_activation: bool = False):
        super().__init__()
        self.depthwise = ConvBNAct(in_channels, in_channels, kernel_size, stride=stride,
                                   groups=in_channels)
        self.pointwise = ConvBNAct(in_channels, out_channels, 1,
                                   activation=pointwise_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SqueezeExcite(nn.Module):
    """Channel attention: pool to a vector, gate each channel by a learned scale.

    Global average pooling collapses each channel to one number, so the gate's
    decision is informed by the whole feature map rather than a 3x3
    neighbourhood. The gate is a sigmoid, so a channel can be attenuated or
    preserved but never inverted. Built from 1x1 convolutions rather than linear
    layers, which keeps the block agnostic to spatial size.
    """

    def __init__(self, channels: int, ratio: float = 0.25):
        super().__init__()
        hidden = max(1, round(channels * ratio))
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.expand = nn.Conv2d(hidden, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean over H and W, keeping dims so the gate broadcasts back.
        weights = x.mean(dim=(2, 3), keepdim=True)
        weights = self.gate(self.expand(self.act(self.reduce(weights))))
        return x * weights


class DropPath(nn.Module):
    """Stochastic depth: drop a residual branch for a whole sample.

    During training each sample independently keeps or discards the residual
    branch, and survivors are scaled by ``1 / keep_prob`` so the expected
    activation is unchanged. At eval the branch is always kept, which is what
    makes evaluation deterministic.

    Holds no parameters, so it contributes nothing to the state dict.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # One Bernoulli draw per sample, broadcast over C, H and W.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class ResidualSeparableBlock(nn.Module):
    """A residual block built from depthwise-separable convolutions.

    Structure::

        x -> DWSep(3x3, stride) -> DWSep(3x3) -> SE -> DropPath -> + shortcut -> act

    The second separable convolution's projection is left **linear** so the
    shortcut is added before the final activation -- that ordering is what lets
    the identity path carry an unmodified signal, which is the point of a
    residual block. The shortcut is an identity when the shape is unchanged, and
    a 1x1 strided projection otherwise.
    """

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1,
                 se_ratio: float = 0.25, drop_path: float = 0.0):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, 3, stride=stride,
                                            pointwise_activation=True)
        # Linear: the shortcut is added before the final activation.
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, 3,
                                            pointwise_activation=False)
        self.se: nn.Module = (SqueezeExcite(out_channels, se_ratio)
                              if se_ratio > 0 else nn.Identity())
        self.drop_path = DropPath(drop_path)
        self.shortcut: nn.Module = (
            ConvBNAct(in_channels, out_channels, 1, stride=stride, activation=False)
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.se(self.conv2(self.conv1(x)))
        return self.act(self.drop_path(residual) + self.shortcut(x))


class CustomCNN(nn.Module):
    """Residual depthwise-separable stages with SE and stochastic depth.

    A strided stem halves the input once, then each stage applies its first
    block at ``stage_strides[i]`` and the remainder at stride 1. Stochastic depth
    ramps linearly from 0 at the first block to ``drop_path`` at the last: the
    usual schedule, because dropping early blocks as often as late ones removes
    low-level features the whole network depends on.

    Pooling is adaptive, so the network is not tied to one input resolution --
    160px under the current protocol, 128px or 224px without a code change.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, *, stem_channels: int = 32,
                 stage_channels: tuple[int, ...] = (64, 128, 256, 384),
                 stage_blocks: tuple[int, ...] = (2, 2, 3, 2),
                 stage_strides: tuple[int, ...] = (2, 2, 2, 2),
                 se_ratio: float = 0.25, dropout: float = 0.3, drop_path: float = 0.1):
        super().__init__()
        if not (len(stage_channels) == len(stage_blocks) == len(stage_strides)):
            raise ValueError("stage_channels, stage_blocks and stage_strides must "
                             "describe the same number of stages")
        self.num_classes = num_classes

        self.stem = ConvBNAct(3, stem_channels, 3, stride=2)

        total_blocks = sum(stage_blocks)
        built = 0
        stages: list[nn.Module] = []
        in_channels = stem_channels
        for out_channels, blocks, stride in zip(stage_channels, stage_blocks, stage_strides):
            stage: list[nn.Module] = []
            for index in range(blocks):
                # Linear ramp: the first block of the network gets 0, the last
                # gets the configured maximum.
                ratio = built / max(1, total_blocks - 1)
                stage.append(ResidualSeparableBlock(
                    in_channels, out_channels,
                    stride=stride if index == 0 else 1,
                    se_ratio=se_ratio,
                    drop_path=drop_path * ratio,
                ))
                in_channels = out_channels
                built += 1
            stages.append(nn.Sequential(*stage))

        self.stages = nn.Sequential(*stages)
        self.head_act = nn.SiLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(in_channels, num_classes)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(N, num_classes)`` -- never a softmax."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected an (N, 3, H, W) batch, got {tuple(x.shape)}")
        features = self.head_act(self.stages(self.stem(x)))
        pooled = self.flatten(self.pool(features))
        return self.classifier(self.dropout(pooled))


def init_weights(module: nn.Module) -> None:
    """Initialize one module in place; intended for ``nn.Module.apply``.

    Convolutions use Kaiming normal with ``fan_out``, which keeps activation
    variance stable through a deep stack of ReLU-family non-linearities; norms
    start as the identity; the classifier uses a small normal so the initial
    logits sit near zero, giving a starting loss close to ``ln(num_classes)``.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_custom_cnn_ziyang(num_classes: int = NUM_CLASSES, **kwargs) -> nn.Module:
    """Construct ``custom_cnn`` with randomly initialized weights."""
    return CustomCNN(num_classes=num_classes, **kwargs)
