"""VGG-style deep sequential CNN - Member 2. STUB: implement me.

Research question:
    Does additional depth improve fine-grained pest recognition, and does
    VGG19's extra depth beat VGG16's on a small 10-class dataset?

Design reference: Simonyan & Zisserman (2014). Reference only - do NOT import
``torchvision.models.vgg16``. Build it from nn.Conv2d / nn.BatchNorm2d / nn.ReLU.

The defining VGG idea: never use a large kernel. Stack small 3x3 convolutions
instead, because two stacked 3x3 layers see the same 5x5 receptive field with
fewer parameters and an extra non-linearity in between.

Layer configurations (M = MaxPool 2x2 stride 2). These are the real VGG configs
D and E, and the ONLY difference between the two models - keep them exact so the
depth ablation actually measures depth:

    config D (VGG16, 13 conv layers):
        [64, 64, M, 128, 128, M, 256, 256, 256, M,
         512, 512, 512, M, 512, 512, 512, M]

    config E (VGG19, 16 conv layers):
        [64, 64, M, 128, 128, M, 256, 256, 256, 256, M,
         512, 512, 512, 512, M, 512, 512, 512, 512, M]

Two mandatory departures from the paper, both required by the project rules:

  1. width_mult=0.5 halves every channel count. Full-width VGG16 is ~138M
     parameters; the project budget is 0.5-5M. Halving the widths cuts the conv
     parameters to roughly a quarter, landing around 3.7M (D) and 5.0M (E).
  2. Global average pooling replaces the three 4096-unit FC layers, which is
     where almost all of VGG's parameters live.

Add BatchNorm after every conv (the original predates it and was notoriously
hard to train without it) and use progressively increasing dropout in the head.

Implementation sketch:

    def _make_layers(config, width_mult):
        layers, in_ch = [], 3
        for item in config:
            if item == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                out_ch = int(item * width_mult)
                layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                           nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
                in_ch = out_ch
        return nn.Sequential(*layers), in_ch

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

CONFIG_D = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
            512, 512, 512, "M", 512, 512, 512, "M"]

CONFIG_E = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
            512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]


class VGGCNN(nn.Module):
    def __init__(
        self,
        config: list,
        num_classes: int = 10,
        width_mult: float = 0.5,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        raise NotImplementedError(
            "Member 2: implement the VGG-style network in src/models/vgg_cnn.py"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_vgg16_cnn(num_classes: int = 10, **kwargs) -> VGGCNN:
    return VGGCNN(config=CONFIG_D, num_classes=num_classes, **kwargs)


def build_vgg19_cnn(num_classes: int = 10, **kwargs) -> VGGCNN:
    return VGGCNN(config=CONFIG_E, num_classes=num_classes, **kwargs)
