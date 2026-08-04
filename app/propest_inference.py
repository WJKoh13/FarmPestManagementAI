"""Inference that preprocesses a photo exactly the way training did.

This module exists because of a real bug, recorded in docs/propestnet.md:
``predict_pest()`` once opened images raw and fed uncropped photos to a model
trained on crops. It was caught only because top-1 accuracy disagreed with the
confusion matrix on identical images. Every transform here is therefore derived
from the checkpoint's own recorded settings, never from a constant typed twice.

Test-time augmentation is on by default. The notebook selected the setting on
the validation split -- centre + whole + mirror, four passes -- and it was worth
+3.5 points of accuracy and +0.030 macro-F1 on test. Note *which* view earned
that: mirroring alone was slightly negative, and the entire gain came from the
``whole`` view, which squeezes the full frame in instead of centre-cropping it.
That matters here more than it did in the notebook, because a farmer's upload
has no bounding box to crop to and its framing is unknown.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from app.cnn_model import DEFAULT_CROP_MARGIN
from ip102_bench.transforms import build_eval_transform

# The notebook's validation-selected setting. Views are averaged in softmax
# space, and `flip` adds a mirrored pass of each.
TTA_VIEWS = ("centre", "whole")
TTA_FLIP = True


def build_views(image_size: int, mean: list[float], std: list[float]) -> dict:
    """The deterministic views TTA averages over.

    ``centre`` is the harness eval transform -- its resize-to-1.14x then
    centre-crop is already identical to the notebook's Resize(146)/CenterCrop(128)
    at image_size 128, so it is reused rather than restated.

    ``whole`` has no harness equivalent: it squeezes the entire frame into the
    input instead of cropping, which is the view that carries the TTA gain.
    """
    from torchvision import transforms

    return {
        "centre": build_eval_transform(image_size, mean, std),
        "whole": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
    }


def crop_to_box(image: Image.Image, box: list[float] | None,
                margin: float = DEFAULT_CROP_MARGIN) -> Image.Image:
    """Crop to an annotated insect box, keeping ``margin`` of context each side.

    Only the offline evaluation has boxes. A farmer's photo has none, and the
    image is returned untouched -- the same behaviour the notebook's
    ``crop_to_insect`` has for unannotated images. For a model trained purely on
    box crops that untouched path is distribution shift; the caller is expected
    to say so rather than pretend the prediction is as reliable.
    """
    if not box:
        return image
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * margin
    pad_y = (y2 - y1) * margin
    return image.crop(
        (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(image.width, x2 + pad_x),
            min(image.height, y2 + pad_y),
        )
    )


@torch.inference_mode()
def predict_probabilities(model, image: Image.Image, views: dict, *, tta: bool = True,
                          device: str | torch.device = "cpu") -> torch.Tensor:
    """Class probabilities for one image, averaged over the TTA views."""
    used = TTA_VIEWS if tta else ("centre",)
    probabilities = None
    for name in used:
        batch = views[name](image).unsqueeze(0).to(device)
        passes = [batch]
        if tta and TTA_FLIP:
            # dims=[3] is the width axis: a mirror of the tensor, cheaper than
            # and exactly equivalent to flipping the PIL image.
            passes.append(torch.flip(batch, dims=[3]))
        for pass_batch in passes:
            softmax = torch.softmax(model(pass_batch), dim=1)[0].cpu()
            probabilities = softmax if probabilities is None else probabilities + softmax
    return probabilities / probabilities.sum()


def predict_topk(model, image: Image.Image | str | Path, class_names: list[str], views: dict,
                 *, k: int = 3, tta: bool = True, device: str | torch.device = "cpu",
                 box: list[float] | None = None,
                 crop_margin: float = DEFAULT_CROP_MARGIN) -> list[tuple[str, float]]:
    """Top-k ``(class_name, probability)`` for one photo, most confident first.

    Top-3 rather than top-1 is the deliberate presentation: the model reaches
    86.5% top-3 against 69.2% top-1, so three candidates with confidences is
    both more accurate and more useful to a farmer than one confident-looking
    guess.

    ``crop_margin`` must be the margin the weights were trained with -- the
    caller reads it off the checkpoint. It defaults to the locked protocol's
    0.25 so checkpoints that record no margin behave exactly as before.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            image = handle.convert("RGB")
    else:
        image = image.convert("RGB")

    image = crop_to_box(image, box, margin=crop_margin)
    probabilities = predict_probabilities(model, image, views, tta=tta, device=device)
    confidences, indices = probabilities.topk(min(k, len(class_names)))
    return [(class_names[i], float(c)) for c, i in zip(confidences, indices)]
