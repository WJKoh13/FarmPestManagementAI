"""Optional reference architectures.

Nothing requires you to use these. The normal way to add a model is to define it
in your own notebook -- see ``notebooks/_TEMPLATE.ipynb``. This package exists
so the from-scratch architectures already written for the project stay usable,
and so the pretrained baseline is one line rather than fifty.

Shared contract for every builder here::

    model = build_model("alexnet", num_classes=10)
    logits = model(images)          # [B, 3, 160, 160] -> [B, 10], raw logits

Never apply softmax inside a model -- ``CrossEntropyLoss`` expects logits.

Several of these are still stubs that raise ``NotImplementedError``; each stub
file carries the architecture spec, the target parameter count and its
acceptance checks. ``alexnet_cnn.py`` is the worked example.
"""

from __future__ import annotations

import torch.nn as nn

from .alexnet_cnn import build_alexnet_cnn
from .baseline_cnn import build_baseline_cnn
from .googlenet_cnn import build_googlenet_cnn
from .justin_deep_cnn import build_justin_deep_cnn
from .lightweight_cnn import build_lightweight_cnn
from .propestnet import build_propestnet
from .residual_cnn import build_residual_cnn
from .vgg_cnn import build_vgg16_cnn, build_vgg19_cnn

SCRATCH_REGISTRY = {
    "propestnet": build_propestnet,
    "alexnet": build_alexnet_cnn,
    "vgg16": build_vgg16_cnn,
    "vgg19": build_vgg19_cnn,
    "baseline": build_baseline_cnn,
    "googlenet": build_googlenet_cnn,
    "justin_deep_v2": build_justin_deep_cnn,
    "residual": build_residual_cnn,
    "lightweight": build_lightweight_cnn,
}


def build_model(name: str, num_classes: int = 15, **kwargs) -> nn.Module:
    """Instantiate a from-scratch architecture with randomly initialized weights."""
    if name not in SCRATCH_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(SCRATCH_REGISTRY)}")
    return SCRATCH_REGISTRY[name](num_classes=num_classes, **kwargs)


def build_pretrained(
    name: str = "resnet18",
    num_classes: int = 10,
    *,
    freeze_features: bool = True,
    weights: str = "DEFAULT",
) -> nn.Module:
    """A torchvision backbone with an ImageNet checkpoint and a fresh head.

    This is the external benchmark, not a from-scratch entry. Whatever you do
    with it, pass ``pretrained=True`` to ``save_run`` so the comparison table
    keeps it in the right group.

    Args:
        freeze_features: Train only the new classifier head. The usual recipe is
            to start frozen, then unfreeze the last block for a few low-learning-
            rate epochs -- see ``archive/pretrained_ip102_benchmark.py`` for the
            two-phase version.
    """
    from torchvision import models as tv_models

    factory = getattr(tv_models, name, None)
    if factory is None:
        raise KeyError(f"torchvision has no model named '{name}'")

    model = factory(weights=weights)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # Replace whichever head this architecture uses, and leave it trainable.
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
        else:
            last = classifier[-1]
            classifier[-1] = nn.Linear(last.in_features, num_classes)
    else:
        raise TypeError(f"Don't know how to replace the head of '{name}'")

    return model


def available_models() -> list[str]:
    return sorted(SCRATCH_REGISTRY)


__all__ = [
    "SCRATCH_REGISTRY",
    "available_models",
    "build_model",
    "build_pretrained",
]
