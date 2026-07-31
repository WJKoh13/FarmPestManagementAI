"""Shared preprocessing and augmentation (Step 3).

One normalization policy for every model. Training gets moderate augmentation;
validation and test are fully deterministic - no randomness whatsoever.
"""

from __future__ import annotations

import json
from pathlib import Path

from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS_PATH = PROJECT_ROOT / "configs" / "norm_stats.json"

# Fallback only - the real values come from configs/norm_stats.json, computed on
# the training images alone by scripts/compute_norm_stats.py.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_norm_stats(path: str | Path | None = None) -> tuple[list[float], list[float]]:
    """Return (mean, std) recorded for the training split."""
    stats_path = Path(path) if path else DEFAULT_STATS_PATH
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"{stats_path} not found. Run `python scripts/compute_norm_stats.py` first."
        )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return stats["mean"], stats["std"]


def build_train_transform(image_size: int, mean: list[float], std: list[float]):
    """Augmentation strong enough to regularize, mild enough to keep the pest visible."""
    return transforms.Compose(
        [
            # Rotate BEFORE cropping: rotating a 160x160 crop leaves black wedges in
            # the corners, which the network happily learns as a shortcut feature.
            # Rotating first lets the subsequent crop cut most of that padding away.
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(
                image_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), value="random"),
        ]
    )


def build_eval_transform(image_size: int, mean: list[float], std: list[float]):
    """Deterministic pipeline for validation and test. No random augmentation."""
    resize_to = int(round(image_size * 1.14))  # 160 -> 182, then center crop to 160
    return transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
