"""Step 2/3 validation checks plus the augmentation preview image.

    python scripts/check_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.data.dataset import IP102ClassificationDataset  # noqa: E402
from src.data.transforms import (  # noqa: E402
    build_eval_transform,
    build_train_transform,
    load_norm_stats,
)
from src.utils.seed import seed_everything  # noqa: E402

EXPECTED = {"train_manifest": 4318, "val_manifest": 721, "test_manifest": 2166}


def check(label: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    return passed


def main() -> None:
    config = load_config("configs/_base.yaml")
    seed_everything(config["seed"])
    image_size = config["image_size"]
    mean, std = load_norm_stats(resolve_path(config["norm_stats"]))
    dataset_root = resolve_path(config["dataset_root"])
    results: list[bool] = []

    print("Dataset checks")
    datasets = {}
    for key, expected_len in EXPECTED.items():
        ds = IP102ClassificationDataset(
            manifest_path=resolve_path(config[key]),
            dataset_root=dataset_root,
            transform=build_eval_transform(image_size, mean, std),
        )
        datasets[key] = ds
        split = key.replace("_manifest", "")
        results.append(check(f"{split} length == {expected_len}", len(ds) == expected_len, str(len(ds))))

        counts = ds.class_counts()
        results.append(check(f"{split}: all 10 classes present", all(c > 0 for c in counts)))
        labels = set(ds.frame["project_label"].tolist())
        results.append(
            check(f"{split}: labels within 0-9", labels <= set(range(10)), str(sorted(labels)))
        )

    print("\nBatch checks")
    train_set = IP102ClassificationDataset(
        manifest_path=resolve_path(config["train_manifest"]),
        dataset_root=dataset_root,
        transform=build_train_transform(image_size, mean, std),
    )
    images, targets = next(iter(DataLoader(train_set, batch_size=32, shuffle=True)))
    results.append(
        check("image batch is [32, 3, 160, 160]",
              tuple(images.shape) == (32, 3, image_size, image_size), str(tuple(images.shape)))
    )
    results.append(check("labels are int64 for CrossEntropyLoss", targets.dtype == torch.int64,
                         str(targets.dtype)))
    results.append(check("images are finite", bool(torch.isfinite(images).all())))

    print("\nDeterminism check")
    eval_set = IP102ClassificationDataset(
        manifest_path=resolve_path(config["val_manifest"]),
        dataset_root=dataset_root,
        transform=build_eval_transform(image_size, mean, std),
    )
    first, _ = eval_set[0]
    second, _ = eval_set[0]
    results.append(
        check("validation transform is deterministic", bool(torch.equal(first, second)))
    )
    aug_a, _ = train_set[0]
    aug_b, _ = train_set[0]
    results.append(
        check("training transform IS random (sanity)", not bool(torch.equal(aug_a, aug_b)))
    )

    print("\nAugmentation preview")
    out_dir = resolve_path("runs") / "_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = out_dir / "augmentation_preview.png"

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    # Evenly spaced indices: the manifest is class-ordered, so range(16) would only
    # ever show class 0.
    preview_indices = [round(i * (len(train_set) - 1) / 15) for i in range(16)]
    fig, axes = plt.subplots(4, 4, figsize=(11, 11.5))
    for ax, idx in zip(axes.flat, preview_indices):
        image, label = train_set[idx]
        ax.imshow((image * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(train_set.class_names[int(label)], fontsize=8)
        ax.axis("off")
    fig.suptitle("Training augmentations - the pest must stay visible", fontsize=12)
    fig.tight_layout()
    fig.savefig(preview_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {preview_path}")
    print("  Inspect it by eye: pest visible, colours realistic, insect not cropped away.")

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
