"""Step 3: compute RGB mean and std from the TRAINING images only.

Validation and test statistics must never leak into preprocessing. The result is
written to configs/norm_stats.json and used by every model, so this only needs to
be run once for the whole team.

    python scripts/compute_norm_stats.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from src.data.dataset import IP102ClassificationDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/_base.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    image_size = config["image_size"]

    # Resize only - no augmentation, no normalization; we are measuring raw pixels.
    dataset = IP102ClassificationDataset(
        manifest_path=resolve_path(config["train_manifest"]),
        dataset_root=resolve_path(config["dataset_root"]),
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        ),
    )
    print(f"Computing statistics over {len(dataset)} TRAINING images only.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Streaming sum and sum-of-squares over pixels, per channel.
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for images, _ in tqdm(loader, desc="scanning"):
        images = images.double()
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images**2).sum(dim=[0, 2, 3])
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / pixel_count
    std = (channel_sq_sum / pixel_count - mean**2).clamp(min=0).sqrt()

    stats = {
        "mean": [round(v, 6) for v in mean.tolist()],
        "std": [round(v, 6) for v in std.tolist()],
        "computed_from": config["train_manifest"],
        "num_images": len(dataset),
        "image_size": image_size,
        "note": "Training split only. Never recompute from validation or test data.",
    }

    out_path = PROJECT_ROOT / "configs" / "norm_stats.json"
    out_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"\nmean = {stats['mean']}")
    print(f"std  = {stats['std']}")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
