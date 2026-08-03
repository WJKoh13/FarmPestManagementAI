"""Lightweight depthwise-separable CNN - Member 5. STUB: implement me.

Research question:
    How much accuracy can a small, fast custom model retain for offline edge
    deployment?

Design reference: Howard et al. (2017), MobileNet-v1. Reference only - do NOT
import MobileNet or any other lightweight architecture.

The depthwise-separable idea: factor one standard convolution into two cheap
ones. A standard 3x3 conv costs in_ch * out_ch * 9 parameters; the separable
version costs in_ch * 9 + in_ch * out_ch, which is roughly 8-9x cheaper for
typical channel counts.

The block you must build:

    Depthwise 3x3 conv   -> nn.Conv2d(in_ch, in_ch, 3, stride, padding=1,
                                      groups=in_ch, bias=False)
    BatchNorm -> ReLU
    Pointwise 1x1 conv   -> nn.Conv2d(in_ch, out_ch, 1, bias=False)
    BatchNorm -> ReLU

``groups=in_channels`` is what makes it depthwise: each input channel gets its
own filter and there is no mixing across channels. The 1x1 pointwise conv is
what then mixes them.

Suggested layout (MobileNet-v1 stage pattern at width_mult=0.5):

    Input 3 x 160 x 160
    Stem:      Conv 3x3 s2, 16 -> BN -> ReLU        ->  16 x 80 x 80
    Separable: 16  ->  32,  stride 1                ->  32 x 80 x 80
    Separable: 32  ->  64,  stride 2                ->  64 x 40 x 40
    Separable: 64  ->  64,  stride 1                ->  64 x 40 x 40
    Separable: 64  -> 128,  stride 2                -> 128 x 20 x 20
    Separable: 128 -> 128,  stride 1                -> 128 x 20 x 20
    Separable: 128 -> 256,  stride 2                -> 256 x 10 x 10
    Separable: 256 -> 256,  stride 1  (x2)          -> 256 x 10 x 10
    Separable: 256 -> 512,  stride 2                -> 512 x 5 x 5
    Head:      GlobalAvgPool -> Dropout -> Linear(512 -> num_classes)

That lands around 0.4-0.8M parameters - the smallest model in the comparison.

This model matters most for the final recommendation: the application is
intended for offline use, so its CPU latency and .pt file size carry as much
weight as its macro F1. Note in your write-up whether the accuracy it gives up
against the residual or VGG models is worth the speed it buys.

Acceptance checks before you train:
  * accepts [B, 3, 160, 160] and returns [B, 10]
  * a forward AND backward pass completes without error
  * no softmax inside the model - CrossEntropyLoss wants raw logits
  * Kaiming/He init on every conv and linear layer
  * record the exact parameter count (see `python -m src.summarize`)

See src/models/alexnet_cnn.py for the conventions to follow.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 followed by pointwise 1x1, each with BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        raise NotImplementedError(
            "Member 5: implement the separable block in src/models/lightweight_cnn.py"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LightweightCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 0.5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        raise NotImplementedError(
            "Member 5: implement the lightweight network in src/models/lightweight_cnn.py"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_lightweight_cnn(num_classes: int = 10, **kwargs) -> LightweightCNN:
    return LightweightCNN(num_classes=num_classes, **kwargs)
