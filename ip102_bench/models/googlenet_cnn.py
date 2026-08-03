"""GoogLeNet / Inception-style multi-scale CNN - Member 4. STUB: implement me.

Research question:
    Do multi-scale features improve recognition of pests with different body
    sizes and visual structures?

Design reference: Szegedy et al. (2014), GoogLeNet / Inception-v1. Reference
only - do NOT import ``torchvision.models.googlenet``.

The Inception idea: instead of choosing one kernel size per layer, run several
receptive fields in PARALLEL and concatenate them along the channel dimension,
letting the network learn which scale matters for each pest.

The Inception block you must build (all four branches keep the same H x W so
they can be concatenated):

                +-> 1x1 conv, c1 ------------------------+
                |                                        |
    Input ------+-> 1x1 conv, r3 -> 3x3 conv, c3 --------+-> Concat -> BN -> ReLU
                |                                        |
                +-> 1x1 conv, r5 -> 5x5 conv, c5 --------+
                |                                        |
                +-> MaxPool 3x3 s1 -> 1x1 conv, cp ------+

    Output channels = c1 + c3 + c5 + cp

The 1x1 convolutions marked r3 and r5 are the "reduce" bottlenecks. They exist
purely to cut the channel count before the expensive 3x3 and 5x5 convolutions -
without them the parameter count explodes. Cheaper alternative worth trying, and
worth a sentence in the report: replace the 5x5 branch with two stacked 3x3
convolutions, which covers the same receptive field for fewer parameters.

Suggested layout (narrowed from the paper to fit the 0.5-5M budget):

    Input 3 x 160 x 160
    Stem:  Conv 7x7 s2, 64 -> BN -> ReLU -> MaxPool 3x3 s2   ->  64 x 40 x 40
           Conv 1x1, 64 -> Conv 3x3, 128 -> BN -> ReLU       -> 128 x 40 x 40
           MaxPool 3x3 s2                                    -> 128 x 20 x 20
    Inception 3a, 3b -> MaxPool 3x3 s2                       ->  ~ x 10 x 10
    Inception 4a, 4b -> MaxPool 3x3 s2                       ->  ~ x 5 x 5
    Inception 5a
    Head:  GlobalAvgPool -> Dropout -> Linear(C -> num_classes)

Auxiliary classifiers: the paper adds two extra heads mid-network whose losses
are added at 0.3 weight. Leave them OFF by default. They change the loss
function, and the five-model comparison is only controlled if every model uses
the same plain weighted cross-entropy. If you want to test them, run it as a
clearly labelled extra experiment.

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


class InceptionBlock(nn.Module):
    """Four parallel branches concatenated on the channel axis. Build by hand."""

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        reduce3x3: int,
        ch3x3: int,
        reduce5x5: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        raise NotImplementedError(
            "Member 4: implement the inception block in src/models/googlenet_cnn.py"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GoogLeNetCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.4,
        use_aux: bool = False,
    ) -> None:
        super().__init__()
        raise NotImplementedError(
            "Member 4: implement the multi-scale network in src/models/googlenet_cnn.py"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_googlenet_cnn(num_classes: int = 10, **kwargs) -> GoogLeNetCNN:
    return GoogLeNetCNN(num_classes=num_classes, **kwargs)
