"""Custom residual CNN - Member 3. STUB: implement me.

Research question:
    Do manually implemented skip connections improve optimization and accuracy?

Design reference: He et al. (2015), ResNet-18 shape. Reference only - do NOT
import a ResNet implementation from torchvision or anywhere else. The whole
point of this experiment is that the residual block is written by hand.

The basic block, which you must build yourself:

    Input --------------------------+
      |                             |
      +-> Conv3x3 -> BN -> ReLU     | (identity, or a projection when shapes differ)
          -> Conv3x3 -> BN          |
                                    |
                 Add <--------------+
                  |
                 ReLU

Critical detail: the second BN comes BEFORE the addition, and the final ReLU
comes AFTER it. Putting the ReLU before the add is the classic mistake - it
destroys the identity path and the model then trains no better than a plain net.

When the spatial size halves or the channel count changes, the identity branch
cannot be added directly. Build the projection shortcut manually:

    nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                  nn.BatchNorm2d(out_ch))

Suggested layout (ResNet-18 shaped, narrowed to fit the 0.5-5M budget):

    Input 3 x 160 x 160
    Stem:    Conv 3x3 s1, 32 -> BN -> ReLU            ->  32 x 160 x 160
             MaxPool 3x3 s2                           ->  32 x 80 x 80
    Stage 1: 2 basic blocks,  32 ch, stride 1         ->  32 x 80 x 80
    Stage 2: 2 basic blocks,  64 ch, stride 2 first   ->  64 x 40 x 40
    Stage 3: 2 basic blocks, 128 ch, stride 2 first   -> 128 x 20 x 20
    Stage 4: 2 basic blocks, 256 ch, stride 2 first   -> 256 x 10 x 10
    Head:    GlobalAvgPool -> Dropout -> Linear(256 -> num_classes)

That lands around 2.8M parameters. A worthwhile extra for the report: zero-init
the last BN gamma in each block so every block starts as an exact identity,
which measurably helps early training.

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


class BasicBlock(nn.Module):
    """Two 3x3 convolutions plus a skip connection. Build this by hand."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        raise NotImplementedError(
            "Member 3: implement the residual block in src/models/residual_cnn.py"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ResidualCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        stage_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        raise NotImplementedError(
            "Member 3: implement the residual network in src/models/residual_cnn.py"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_residual_cnn(num_classes: int = 10, **kwargs) -> ResidualCNN:
    return ResidualCNN(num_classes=num_classes, **kwargs)
