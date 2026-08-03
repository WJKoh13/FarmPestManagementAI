"""Justin's scratch-built baseline CNN for IP102 pest classification.

The network uses only primitive PyTorch layers and starts from randomly
initialized weights. It deliberately avoids pretrained weights and imported
prebuilt architectures so it can participate in the team's controlled model
comparison.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Sequential):
    """Convolution, batch normalization, ReLU, and spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class JustinBaselineCNN(nn.Module):
    """Small three-block CNN for close-up pest image classification."""

    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, ...] | list[int] = (32, 64, 128),
        hidden_dim: int = 128,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        channels = tuple(channels)
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not channels or any(channel < 1 for channel in channels):
            raise ValueError("channels must contain positive integers")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")

        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        self.num_classes = num_classes
        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=channels[-1], out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every trainable layer without loading external weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, images: Tensor) -> Tensor:
        """Extract one pooled feature vector from each input image."""
        features = self.features(images)
        features = self.global_pool(features)
        return torch.flatten(features, start_dim=1)

    def forward(self, images: Tensor) -> Tensor:
        """Return raw class logits for a batch of RGB images."""
        features = self.features(images)
        pooled = self.global_pool(features)
        return self.classifier(pooled)


def build_justin_baseline_cnn(
    num_classes: int = 10, **kwargs
) -> JustinBaselineCNN:
    """Build a randomly initialized Justin baseline CNN."""
    return JustinBaselineCNN(num_classes=num_classes, **kwargs)


# Backward-compatible alias used by the original FinalProject notebook.
build_model = build_justin_baseline_cnn

