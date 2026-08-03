"""Shallow baseline CNN - the team's own architecture. STUB: implement me.

This is the open slot: an original, straightforward CNN that is not modelled on
any published network. Its role is to establish how well a small hand-rolled CNN
does before the classic architectures are brought in, which is what makes the
comparison meaningful.

Suggested architecture, straight from Step 5 of the instructions - change it if
you have a better idea, but record what you changed and why:

    Input 3 x 160 x 160

    Block 1:  Conv 3x3, 32  -> BatchNorm -> ReLU -> MaxPool 2x2   ->  32 x 80 x 80
    Block 2:  Conv 3x3, 64  -> BatchNorm -> ReLU -> MaxPool 2x2   ->  64 x 40 x 40
    Block 3:  Conv 3x3, 128 -> BatchNorm -> ReLU -> MaxPool 2x2   -> 128 x 20 x 20

    Global average pooling
    Dense 128 -> ReLU -> Dropout -> Dense num_classes

Note this design has a Dense 128 hidden layer in the head, unlike the other
models here which go straight from pooling to the classifier. That is fine - it
is part of what makes this architecture yours.

Only three pooling stages means the feature map stays fairly large, so this model
is cheap in parameters but not necessarily cheap in compute. Worth a sentence in
your write-up: parameter count and inference latency are not the same thing.

Acceptance checks before you train:
  * accepts [B, 3, 160, 160] and returns [B, 10]
  * a forward AND backward pass completes without error
  * no softmax inside the model - CrossEntropyLoss wants raw logits
  * Kaiming/He init on every conv and linear layer
  * record the exact parameter count (see `python -m src.summarize`)

See src/models/alexnet_cnn.py for a fully worked example of the conventions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, ...] = (32, 64, 128),
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        raise NotImplementedError(
            "Implement your own shallow baseline in src/models/baseline_cnn.py"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_baseline_cnn(num_classes: int = 10, **kwargs) -> BaselineCNN:
    return BaselineCNN(num_classes=num_classes, **kwargs)
