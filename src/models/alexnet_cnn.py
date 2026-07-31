"""AlexNet-style CNN - the controlled baseline (Member 1).

Written from primitive PyTorch layers. AlexNet (Krizhevsky et al., 2012) is the
design reference only; nothing is imported and no pretrained weights are used.

Deliberate departures from the 2012 paper, all recorded here so the team can
justify them in the report:

  * Batch normalization replaces local response normalization, which is what
    everyone does now and makes the net trainable without careful LR tuning.
  * Input is 160x160 instead of 224x224, so the stem uses an 7x7 stride-2
    convolution rather than the original 11x11 stride-4 - a stride-4 stem would
    throw away too much detail on small pest images.
  * The three large fully connected layers (which hold ~58M of AlexNet's 61M
    parameters) are replaced by global average pooling plus a single Linear.
    This is what keeps the model inside the project's 0.5-5M budget.

Architecture (width_mult = 1.0):

    Input 3 x 160 x 160
    Stem:    Conv 7x7 s2, 64  -> BN -> ReLU -> MaxPool 3x3 s2      ->  64 x 40 x 40
    Stage 2: Conv 5x5,   192  -> BN -> ReLU -> MaxPool 3x3 s2      -> 192 x 20 x 20
    Stage 3: Conv 3x3,   256  -> BN -> ReLU                        -> 256 x 20 x 20
    Stage 4: Conv 3x3,   256  -> BN -> ReLU                        -> 256 x 20 x 20
    Stage 5: Conv 3x3,   192  -> BN -> ReLU -> MaxPool 3x3 s2      -> 192 x 10 x 10
    Head:    GlobalAvgPool -> Dropout -> Linear(192 -> num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
    """Conv -> BN -> ReLU. Bias is omitted because BN immediately re-centres."""
    return nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AlexNetCNN(nn.Module):
    """Five convolutional stages, three max-pools, GAP classifier head."""

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        def w(channels: int) -> int:
            # Round to a multiple of 8 - keeps the tensor shapes tidy.
            return max(8, int(round(channels * width_mult / 8)) * 8)

        c1, c2, c3, c4, c5 = w(64), w(192), w(256), w(256), w(192)

        self.features = nn.Sequential(
            _conv_bn_relu(3, c1, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _conv_bn_relu(c1, c2, kernel_size=5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _conv_bn_relu(c2, c3, kernel_size=3),
            _conv_bn_relu(c3, c4, kernel_size=3),
            _conv_bn_relu(c4, c5, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(c5, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming/He initialization, which is the right choice ahead of ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)  # raw logits


def build_alexnet_cnn(num_classes: int = 10, **kwargs) -> AlexNetCNN:
    return AlexNetCNN(num_classes=num_classes, **kwargs)
