"""The shared preprocessing and augmentation pipeline.

One normalization policy, one augmentation recipe, for every model. Training
gets moderate augmentation; validation and test are fully deterministic.
"""

from __future__ import annotations

from torchvision import transforms


def build_train_transform(image_size: int, mean: list[float], std: list[float]):
    """Strong enough to regularize, mild enough to keep the pest recognizable."""
    return transforms.Compose(
        [
            # Rotate BEFORE cropping. Rotating a finished 160x160 crop leaves black
            # wedges in the corners, and a network will happily learn those wedges
            # as a feature. Rotating first lets the crop cut the padding away.
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
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
    """Deterministic pipeline for validation and test. No randomness at all."""
    resize_to = int(round(image_size * 1.14))  # 160 -> 182, then centre crop back to 160
    return transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
