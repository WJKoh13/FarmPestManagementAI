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


# ---------------------------------------------------------------------------
# Beatrice's VGG19, as trained in notebooks/Beatrice_vgg19_xml_cropped.ipynb.
#
# Copied verbatim from section 9 of that notebook, which is the graded artifact
# and has to show the architecture inline. ``tests/test_vgg19_beatrice.py``
# fails if the two ever drift: it execs the notebook cell and compares
# state-dict keys and parameter count.
#
# This is deliberately NOT the ``VGGCNN`` stub above. That stub specifies a
# half-width, BatchNorm, global-average-pool variant built to a 0.5-5M parameter
# budget; this is the full-width paper architecture with the three fully
# connected layers, and it is what her trained weights actually fit. Redefining
# the ``vgg16``/``vgg19`` keys to mean this instead would orphan any checkpoint
# saved against the stub's spec, so this gets its own registry key.
#
# Every parameter name below is load-bearing: ``features``, ``adaptive_pool``
# and ``classifier`` are what ``strict=True`` matches when the app rebuilds the
# network from ``model_name`` alone, so renaming an attribute silently
# invalidates the checkpoint.
# ---------------------------------------------------------------------------


class VGG19(nn.Module):
    """Full-width VGG19: 16 convolutions, then three fully connected layers.

    No BatchNorm and no global average pooling -- this is the 2014 architecture
    as published, which is what the notebook set out to train from scratch.
    Almost all of its ~139M parameters sit in the first ``Linear(25088, 4096)``.
    """

    def __init__(
        self,
        num_classes,
        strict_classifier=True,
        small_classifier_units=512,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        hidden_units = (
            4096 if strict_classifier else small_classifier_units
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def build_vgg19_beatrice(num_classes: int = 15, **kwargs) -> VGG19:
    """Construct Beatrice's VGG19 with randomly initialized weights.

    The adaptive pool means this accepts any input resolution; her run trained
    at 128px, which the checkpoint records so the app serves it the same way.
    """
    return VGG19(num_classes=num_classes, **kwargs)
