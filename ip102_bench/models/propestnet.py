"""ProPestNet -- the architecture behind the pest chatbot.

Copied verbatim from ``notebooks/WJ_ProPestNet.ipynb`` section 5, which is the
graded artifact and has to show the architecture inline. That duplication is
deliberate, and ``tests/test_propestnet.py`` fails if the two ever drift: it
execs the notebook cell and compares state-dict keys and parameter count.

Nothing here comes from ``torchvision.models``. The design rationale, the
ablation and the measured results are in ``docs/propestnet.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The submitted model is fifteen-class; see data_manifests/classes_top15.json.
NUM_CLASSES = 15


class ChannelGate(nn.Module):
    """Squeeze-and-excitation: one learned multiplier per channel.

    Squeeze — global average pooling collapses each C x H x W channel to a single
              number, so the gate's decision is informed by the *whole* feature map
              rather than a 3x3 neighbourhood.
    Excite  — a two-layer bottleneck MLP turns those C numbers into C gates in
              (0, 1), which rescale the channels.

    This is what lets the network learn "when separating the two blister beetles,
    the channels responding to head colour matter more than the ones responding to
    body outline" — a decision no fixed convolution kernel can make.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[0], x.shape[1]
        weights = self.gate(self.pool(x).reshape(batch, channels))
        return x * weights.reshape(batch, channels, 1, 1)


class MultiScaleStem(nn.Module):
    """Three convolution sizes run in parallel over the raw pixels, concatenated.

    A 3x3 filter sees a leg or a wing edge; a 7x7 sees most of a small insect in one
    look. IP102 framing varies too much to bet on a single receptive field, so all
    three run side by side. Stride 2 halves the resolution immediately, which is also
    where most of the saving in time per epoch comes from.
    """

    KERNEL_SIZES = (3, 5, 7)

    def __init__(self, channels_per_branch: int = 32):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, channels_per_branch, kernel_size, stride=2,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(channels_per_branch),
                nn.ReLU(inplace=True),
            )
            for kernel_size in self.KERNEL_SIZES
        ])
        self.out_channels = channels_per_branch * len(self.KERNEL_SIZES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class SimpleStem(nn.Module):
    """Single-kernel stem — the control used by the ablation in section 11."""

    def __init__(self, out_channels: int = 96):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PestBlock(nn.Module):
    """Two 3x3 convolutions, a channel gate, and an identity skip.

    The block preserves its channel count and resolution, so the skip is a pure
    identity — no 1x1 projection needed anywhere in the network. All resolution and
    width changes happen in the stage transitions instead.
    """

    def __init__(self, channels: int, use_residual: bool = True,
                 use_se: bool = True, reduction: int = 16):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.gate = ChannelGate(channels, reduction) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.gate(self.norm2(self.conv2(out)))
        if self.use_residual:
            out = out + x
        return self.relu(out)


class ProPestNet(nn.Module):
    """An original CNN for IP102 pest identification.

    Every constructor flag exists so section 11 can ablate one decision at a time;
    the defaults are the submitted model.
    """

    STAGE_WIDTHS = (64, 128, 256, 384)
    STAGE_DEPTHS = (2, 2, 3, 2)
    RESIDUAL_INIT_SCALE = 0.1   # initial gamma of each block's second BatchNorm

    def __init__(self, num_classes: int = NUM_CLASSES,
                 widths: tuple = STAGE_WIDTHS, depths: tuple = STAGE_DEPTHS,
                 multiscale_stem: bool = True, use_residual: bool = True,
                 use_se: bool = True, learned_downsample: bool = True,
                 head: str = "multiscale", dropout: float = 0.4):
        super().__init__()
        if head not in {"multiscale", "gap", "flatten"}:
            raise ValueError(f"unknown head {head!r}")
        self.head = head

        self.stem = MultiScaleStem() if multiscale_stem else SimpleStem()

        in_channels = self.stem.out_channels
        stages = []
        for index, (width, depth) in enumerate(zip(widths, depths)):
            # The stem has already halved the resolution once, so stage 1 keeps it
            # and only stages 2-4 downsample.
            layers = [self._transition(in_channels, width,
                                       downsample=index > 0,
                                       learned=learned_downsample)]
            layers += [PestBlock(width, use_residual=use_residual, use_se=use_se)
                       for _ in range(depth)]
            stages.append(nn.Sequential(*layers))
            in_channels = width
        self.stages = nn.ModuleList(stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        if head == "multiscale":
            # Stage 3 carries fine texture, stage 4 carries shape and semantics.
            # The classifier gets both.
            features = widths[-1] + widths[-2]
            self.classifier = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(features, num_classes))
        elif head == "gap":
            self.classifier = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(widths[-1], num_classes))
        else:
            # The plain-stack control: keep a 4x4 grid, flatten it, and pay for a
            # wide hidden layer.
            self.flatten_pool = nn.AdaptiveAvgPool2d(4)
            self.classifier = nn.Sequential(
                nn.Linear(widths[-1] * 16, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
        self._initialize_weights()

    @staticmethod
    def _transition(in_channels: int, out_channels: int, downsample: bool,
                    learned: bool) -> nn.Sequential:
        """Change the channel count, and optionally halve the resolution.

        With learned=True the stride-2 convolution decides what to discard. With
        learned=False a fixed max-pool does it instead — the ablation control.
        """
        layers: list[nn.Module] = []
        stride = 1
        if downsample:
            if learned:
                stride = 2
            else:
                layers.append(nn.MaxPool2d(2, 2))
        layers += [
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Start each residual branch small. With gamma = 1 the branch output has the
        # same scale as the identity path, so nine stacked blocks compound their
        # variance; at RESIDUAL_INIT_SCALE the block is close to a pass-through and
        # the network effectively starts shallow, deepening itself as training grows
        # these gammas.
        #
        # Deliberately small rather than zero: gamma = 0 makes the branch output
        # exactly zero, which also zeroes the gradient to every parameter *inside*
        # the branch — 54 of this network's 95 parameter tensors would sit dead
        # through the first step. A small positive value keeps the near-identity
        # behaviour while every parameter still learns from step one.
        for module in self.modules():
            if isinstance(module, PestBlock) and module.use_residual:
                nn.init.constant_(module.norm2.weight, self.RESIDUAL_INIT_SCALE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        stage_outputs = []
        for stage in self.stages:
            x = stage(x)
            stage_outputs.append(x)

        if self.head == "multiscale":
            pooled = torch.cat([self.pool(stage_outputs[-2]).flatten(1),
                                self.pool(stage_outputs[-1]).flatten(1)], dim=1)
        elif self.head == "gap":
            pooled = self.pool(stage_outputs[-1]).flatten(1)
        else:
            pooled = self.flatten_pool(stage_outputs[-1]).flatten(1)
        return self.classifier(pooled)


def build_propestnet(num_classes: int = NUM_CLASSES, **kwargs) -> nn.Module:
    """Registry entry point. Randomly initialized -- weights are loaded separately.

    ``kwargs`` are the ablation flags (``multiscale_stem``, ``use_residual``,
    ``use_se``, ``learned_downsample``, ``head``); the defaults are the
    submitted model, 10,988,015 parameters.
    """
    return ProPestNet(num_classes=num_classes, **kwargs)
