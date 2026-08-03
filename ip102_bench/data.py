"""The shared dataset and dataloaders.

Every notebook loads data through here. That is the whole point: if two people
build their loaders differently, their macro F1 scores are not comparable and
the whole comparison table is meaningless.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .protocol import Protocol
from .runtime import seed_worker
from .transforms import build_eval_transform, build_train_transform

REQUIRED_COLUMNS = ["image_path", "original_label", "project_label", "class_name", "split"]
BOX_COLUMNS = ["x1", "y1", "x2", "y2"]


def expand_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    """Grow a box by ``margin`` of its own size on each side, clipped to the image.

    A tight crop throws away the context that tells a leafhopper on a rice stem
    apart from one pinned on white card, so a little padding usually helps.
    """
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * margin
    dy = (y2 - y1) * margin
    return (
        max(0, int(round(x1 - dx))),
        max(0, int(round(y1 - dy))),
        min(width, int(round(x2 + dx))),
        min(height, int(round(y2 + dy))),
    )


def weights_from_counts(counts) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1.

    The classes are badly imbalanced -- in ``detection_top10`` the largest has
    ~7x the images of the smallest -- so unweighted cross-entropy quietly
    optimizes for the big classes and drags macro F1 down.
    """
    counts = torch.as_tensor(list(counts), dtype=torch.float)
    weights = counts.sum() / (counts.clamp(min=1) * len(counts))
    return weights / weights.mean()


def class_counts_of(dataset, num_classes: int) -> list[int]:
    """Per-class image counts for a dataset, seeing through ``Subset`` wrappers.

    Smoke tests and quick experiments wrap the training set in a ``Subset``, and
    the wrapped slice has a different class balance from the whole set. Taking
    counts from the underlying dataset would silently apply the wrong loss
    weights, so the indices are followed through.
    """
    from torch.utils.data import Subset

    if isinstance(dataset, IP102Dataset):
        return dataset.class_counts()

    if isinstance(dataset, Subset):
        inner = dataset.dataset
        if isinstance(inner, IP102Dataset):
            counts = [0] * num_classes
            for index in dataset.indices:
                counts[inner.labels[index]] += 1
            return counts
        return class_counts_of(inner, num_classes)

    if hasattr(dataset, "class_counts"):
        return list(dataset.class_counts())

    # Last resort: read the targets one by one. Works for anything, but decodes
    # every image, so it is slow on a full split.
    counts = [0] * num_classes
    for item in dataset:
        counts[int(item[1])] += 1
    return counts


class IP102Dataset(Dataset):
    """Reads a manifest CSV and yields ``(image, label)`` or ``(image, label, path)``.

    Args:
        manifest_path: CSV written by ``scripts/setup_data.py``.
        image_root: Directory the manifest's ``image_path`` values are relative to.
        transform: Torchvision transform applied to the (optionally cropped) PIL image.
        crop_mode: ``"box"`` to crop to the annotated bounding box, ``"none"`` for
            the whole image. Rows with no box always fall back to the whole image.
        crop_margin: Passed to :func:`expand_box`.
        return_path: When True also returns the image path, which error analysis needs.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        transform=None,
        crop_mode: str = "none",
        crop_margin: float = 0.0,
        return_path: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.crop_mode = crop_mode
        self.crop_margin = crop_margin
        self.return_path = return_path

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}. "
                "Run `python scripts/setup_data.py` first."
            )
        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_root}. "
                "Check `dataset.subsets.<name>.images` in protocol.yaml."
            )

        self.frame = pd.read_csv(self.manifest_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in self.frame.columns]
        if missing:
            raise ValueError(f"{self.manifest_path} is missing columns: {missing}")

        self._paths = self.frame["image_path"].tolist()
        self._labels = self.frame["project_label"].astype(int).tolist()

        self._boxes: list[tuple[float, float, float, float] | None]
        if crop_mode == "box" and all(c in self.frame.columns for c in BOX_COLUMNS):
            boxes = self.frame[BOX_COLUMNS]
            self._boxes = [
                None if row.isna().any() else tuple(row.astype(float))
                for _, row in boxes.iterrows()
            ]
        else:
            if crop_mode == "box":
                raise ValueError(
                    f"protocol.yaml asks for crop mode 'box' but {self.manifest_path} "
                    "has no x1/y1/x2/y2 columns. Either pick a subset that ships "
                    "boxes, or set dataset.crop.mode to 'none'."
                )
            self._boxes = [None] * len(self._paths)

        # project_label -> class_name, ordered so index == label.
        pairs = self.frame[["project_label", "class_name"]].drop_duplicates()
        self.class_names: list[str] = [
            name for _, name in sorted(zip(pairs["project_label"], pairs["class_name"]))
        ]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def labels(self) -> list[int]:
        """Project label per row, in manifest order."""
        return self._labels

    def class_counts(self) -> list[int]:
        """Images per project label, index-aligned with ``class_names``."""
        counts = [0] * self.num_classes
        for label in self._labels:
            counts[label] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights, normalized to mean 1."""
        return weights_from_counts(self.class_counts())

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        rel_path = self._paths[index]
        label = self._labels[index]

        with Image.open(self.image_root / rel_path) as img:
            image = img.convert("RGB")
            box = self._boxes[index]
            if box is not None:
                image = image.crop(
                    expand_box(box, image.width, image.height, self.crop_margin)
                )

        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(label, dtype=torch.long)  # CrossEntropyLoss wants int64
        if self.return_path:
            return image, target, rel_path
        return image, target


def build_dataset(protocol: Protocol, split: str, *, train_augmentation: bool | None = None,
                  return_path: bool = False) -> IP102Dataset:
    """One dataset for one split, configured entirely from the protocol."""
    if train_augmentation is None:
        train_augmentation = split == "train"

    mean, std = protocol.norm_stats()
    transform = (
        build_train_transform(protocol.image_size, mean, std)
        if train_augmentation
        else build_eval_transform(protocol.image_size, mean, std)
    )
    return IP102Dataset(
        manifest_path=protocol.manifest(split),
        image_root=protocol.image_root,
        transform=transform,
        crop_mode=protocol.crop_mode,
        crop_margin=protocol.crop_margin,
        return_path=return_path,
    )


def build_dataloaders(
    protocol: Protocol,
    *,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    generator: torch.Generator | None = None,
) -> dict[str, DataLoader]:
    """Train/validation/test loaders under the locked protocol.

    Only the training loader shuffles and augments. Test and validation loaders
    return paths as well, because error analysis needs filenames.
    """
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(protocol.seed)

    num_workers = int(protocol.runtime.get("num_workers", 4))
    loaders: dict[str, DataLoader] = {}

    for split in splits:
        is_train = split == "train"
        dataset = build_dataset(protocol, split, return_path=not is_train)
        loaders[split] = DataLoader(
            dataset,
            batch_size=protocol.batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator if is_train else None,
        )
    return loaders
