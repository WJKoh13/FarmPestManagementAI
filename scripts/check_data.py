"""Validate the data pipeline before anyone burns hours on a training run.

    python scripts/check_data.py
    python scripts/check_data.py --preview   # also write a sample-batch image

Catches the failures that are expensive to find later: overlapping splits, a
class missing from a split, unreadable images, boxes that fall outside their
image, normalization that leaves the inputs off-centre.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from ip102_bench import build_dataloaders, build_dataset, load_protocol  # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{PASS if condition else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Write a sample-batch PNG.")
    parser.add_argument("--sample", type=int, default=250,
                        help="How many images to open when checking readability.")
    args = parser.parse_args()

    protocol = load_protocol()
    print(f"subset '{protocol.subset_name}', crop '{protocol.crop_mode}', "
          f"{protocol.num_classes} classes\n")

    frames = {s: pd.read_csv(protocol.manifest(s)) for s in ("train", "validation", "test")}
    expected = protocol.subset.get("expected_totals") or {}

    for split, frame in frames.items():
        target = expected.get(split)
        check(f"{split} count", target is None or len(frame) == target,
              f"{len(frame)} images" + (f", expected {target}" if target else ""))

    # Split overlap is the failure that silently inflates every score.
    train, val, test = (set(f["image_path"]) for f in frames.values())
    check("train/validation disjoint", not train & val, f"{len(train & val)} shared")
    check("train/test disjoint", not train & test, f"{len(train & test)} shared")
    check("validation/test disjoint", not val & test, f"{len(val & test)} shared")

    for split, frame in frames.items():
        present = set(frame["project_label"])
        missing = set(range(protocol.num_classes)) - present
        check(f"{split} has every class", not missing, f"missing {sorted(missing)}")

    check("labels contiguous from 0",
          set(frames["train"]["project_label"]) == set(range(protocol.num_classes)))

    # Boxes must lie inside their image, or the crop silently returns a sliver.
    if protocol.crop_mode == "box":
        frame = frames["train"].dropna(subset=["x1", "y1", "x2", "y2"])
        check("boxes have positive area",
              bool(((frame["x2"] > frame["x1"]) & (frame["y2"] > frame["y1"])).all()))

        bad, checked = 0, 0
        for _, row in frame.sample(min(args.sample, len(frame)), random_state=0).iterrows():
            with Image.open(protocol.image_root / row["image_path"]) as img:
                width, height = img.size
            checked += 1
            if row["x2"] > width or row["y2"] > height or row["x1"] < 0 or row["y1"] < 0:
                bad += 1
        check("boxes inside image bounds", bad == 0, f"{bad}/{checked} out of bounds")

    unreadable = []
    frame = frames["train"].sample(min(args.sample, len(frames["train"])), random_state=0)
    for path in frame["image_path"]:
        try:
            with Image.open(protocol.image_root / path) as img:
                img.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - any failure is worth reporting
            unreadable.append(f"{path}: {exc}")
    check("images readable", not unreadable, f"{len(unreadable)} of {len(frame)} failed")
    for line in unreadable[:5]:
        print(f"          {line}")

    dataset = build_dataset(protocol, "train")
    image, label = dataset[0]
    check("sample shape", tuple(image.shape) == (3, protocol.image_size, protocol.image_size),
          str(tuple(image.shape)))
    check("label dtype int64", label.dtype == torch.int64, str(label.dtype))

    counts = dataset.class_counts()
    weights = dataset.class_weights()
    print(f"\n  class counts : {counts}")
    print(f"  imbalance    : {max(counts) / max(min(counts), 1):.1f}x (largest / smallest)")
    print(f"  loss weights : {[round(float(w), 3) for w in weights]}")

    # A batch of normalized images should sit near mean 0, std 1. Far off means
    # the stats were computed on the wrong images or with the wrong crop mode.
    loaders = build_dataloaders(protocol, splits=("train",))
    batch, _ = next(iter(loaders["train"]))
    mean, std = batch.mean().item(), batch.std().item()
    check("normalized batch centred", abs(mean) < 0.6, f"mean {mean:.3f}")
    check("normalized batch scaled", 0.5 < std < 1.8, f"std {std:.3f}")

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stats_mean, stats_std = protocol.norm_stats()
        denorm = batch[:16] * torch.tensor(stats_std).view(1, 3, 1, 1) \
            + torch.tensor(stats_mean).view(1, 3, 1, 1)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for ax, img in zip(axes.ravel(), denorm.clamp(0, 1)):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis("off")
        fig.suptitle("Augmented training batch (denormalized)")
        fig.tight_layout()
        out = Path("runs/_diagnostics")
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "sample_batch.png", dpi=140)
        print(f"\n  wrote {out / 'sample_batch.png'}")

    print()
    if failures:
        sys.exit(f"{len(failures)} check(s) failed: {failures}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
