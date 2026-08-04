"""Shared PyTorch dataset for manifest-defined IP102 classification subsets.

Every model in the project must load data through this class so that the only
difference between experiments is the architecture.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

REQUIRED_COLUMNS = ["image_path", "original_label", "project_label", "class_name", "split"]


class IP102ClassificationDataset(Dataset):
    """Reads a CSV manifest and yields (image, project_label[, path]).

    Args:
        manifest_path: CSV written by a repository manifest-building script.
        dataset_root: Directory the manifest's ``image_path`` values are relative to,
            e.g. ``IP102_v1.1/Classification/ip102_v1.1``.
        transform: Torchvision transform applied to the PIL image.
        return_path: When True, ``__getitem__`` also returns the image path, which
            evaluation and error analysis need.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        transform=None,
        return_path: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.return_path = return_path

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}. "
                "Run `python scripts/setup_data.py` first."
            )
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        self.frame = pd.read_csv(self.manifest_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in self.frame.columns]
        if missing:
            raise ValueError(f"{self.manifest_path} is missing columns: {missing}")

        self._paths = self.frame["image_path"].tolist()
        self._labels = self.frame["project_label"].astype(int).tolist()

        # project_label -> class_name, ordered by label so index == label.
        pairs = self.frame[["project_label", "class_name"]].drop_duplicates()
        self.class_names: list[str] = [
            name for _, name in sorted(zip(pairs["project_label"], pairs["class_name"]))
        ]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_counts(self) -> list[int]:
        """Images per project label, index-aligned with ``class_names``."""
        counts = [0] * self.num_classes
        for label in self._labels:
            counts[label] += 1
        return counts

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        rel_path = self._paths[index]
        label = self._labels[index]

        with Image.open(self.dataset_root / rel_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # CrossEntropyLoss expects int64 targets.
        target = torch.tensor(label, dtype=torch.long)

        if self.return_path:
            return image, target, rel_path
        return image, target
