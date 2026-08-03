"""Model registry.

Every architecture in this project is written by hand from primitive PyTorch
layers. Nothing is imported from ``torchvision.models`` and no pretrained weights
are ever loaded.

Shared contract, identical for all models:

    model = build_model("alexnet", num_classes=10)
    logits = model(images)      # [B, 3, 160, 160] -> [B, 10], raw logits

Do not apply softmax inside the model - ``CrossEntropyLoss`` expects logits.
"""

from __future__ import annotations

import torch.nn as nn

from .alexnet_cnn import build_alexnet_cnn
from .baseline_cnn import build_baseline_cnn
from .googlenet_cnn import build_googlenet_cnn
from .justin_baseline_cnn import build_justin_baseline_cnn
from .justin_deep_cnn import build_justin_deep_cnn
from .lightweight_cnn import build_lightweight_cnn
from .residual_cnn import build_residual_cnn
from .vgg_cnn import build_vgg16_cnn, build_vgg19_cnn

MODEL_REGISTRY = {
    # Assigned - one member each
    "alexnet": build_alexnet_cnn,    # AlexNet-style, implemented as the reference
    "vgg16": build_vgg16_cnn,        # deep sequential, VGG config D
    "vgg19": build_vgg19_cnn,        # deep sequential, VGG config E
    "baseline": build_baseline_cnn,  # the team's own shallow architecture
    "justin_baseline": build_justin_baseline_cnn,
    "justin_deep_v2": build_justin_deep_cnn,
    # Unassigned spares - implement only if the group wants more comparison rows
    "googlenet": build_googlenet_cnn,      # multi-scale / inception blocks
    "residual": build_residual_cnn,        # manual skip connections
    "lightweight": build_lightweight_cnn,  # depthwise separable, edge deployment
}


def build_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """Instantiate a registered architecture with randomly initialized weights."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Registered models: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


__all__ = ["build_model", "available_models", "MODEL_REGISTRY"]
