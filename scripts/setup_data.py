"""Build the shared manifests and normalization statistics.

    python scripts/setup_data.py                    # uses the subset in protocol.yaml
    python scripts/setup_data.py --subset rice10
    python scripts/setup_data.py --skip-norm-stats  # manifests only, much faster

Reads the subset definition from ``protocol.yaml`` and writes one CSV per split
into ``data_manifests/``, plus ``classes.json`` and ``norm_stats.json``. Every
notebook loads data through those files, so this script is the single place
where "which images, which labels" is decided.

Two kinds of subset are supported:

``source: detection``
    Bounding-box subset in VOC2007 layout. Splits and boxes come from the JSON
    files committed at the repo root, so everyone trains on the identical split.

``source: classification``
    The official IP102 classification splits (``train.txt`` / ``val.txt`` /
    ``test.txt``), filtered to the selected labels. No new split is created.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ip102_bench.protocol import load_protocol, resolve_path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES_TXT = PROJECT_ROOT / "IP102_v1.1" / "Classification" / "classes.txt"

# The detection JSONs name splits "val"; the rest of the project says "validation".
SPLIT_ALIASES = {"train": ("train",), "validation": ("validation", "val"), "test": ("test",)}
CLASSIFICATION_SPLIT_FILES = {"train": "train.txt", "validation": "val.txt", "test": "test.txt"}

HEADER = "image_path,original_label,project_label,class_name,split,x1,y1,x2,y2"


def slugify(name: str) -> str:
    """'Rice leaf roller ' -> 'rice_leaf_roller'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_class_names() -> dict[int, str]:
    """Map original IP102 label (0-based) -> slugified name.

    ``classes.txt`` is 1-indexed ('1  rice leaf roller'), so label n is on line n+1.
    """
    if not CLASSES_TXT.is_file():
        sys.exit(f"ERROR: {CLASSES_TXT} not found. Extract the IP102 archive first.")
    names: dict[int, str] = {}
    for line in CLASSES_TXT.read_text(encoding="utf-8", errors="replace").splitlines():
        head, _, rest = line.strip().partition(" ")
        if head.isdigit():
            names[int(head) - 1] = slugify(rest)
    return names


def pick(mapping: dict, split: str):
    """Look a split up under any of its accepted names."""
    for alias in SPLIT_ALIASES[split]:
        if alias in mapping:
            return mapping[alias]
    raise KeyError(f"Split '{split}' not found. Available: {sorted(mapping)}")


def rows_from_detection(subset: dict, label_map: dict[int, int], names: dict[int, str],
                        split: str) -> list[str]:
    splits = json.loads(resolve_path(subset["splits_json"]).read_text(encoding="utf-8"))
    boxes = json.loads(resolve_path(subset["boxes_json"]).read_text(encoding="utf-8"))

    # The splits JSON already stores contiguous project labels; recover the
    # original IP102 label from the subset's ordered list so the manifest stays
    # self-describing.
    originals = subset["original_labels"]
    rows: list[str] = []
    missing_boxes = 0

    for filename, project_label in pick(splits, split):
        project_label = int(project_label)
        original = originals[project_label]
        box = boxes.get(filename)
        if box is None:
            missing_boxes += 1
            box_fields = ",,,"  # four empty cells
        else:
            box_fields = ",".join(f"{float(v):.1f}" for v in box)
        rows.append(
            f"{filename},{original},{project_label},{names[original]},{split},{box_fields}"
        )

    if missing_boxes:
        print(f"        note: {missing_boxes} image(s) have no box; used uncropped")
    return rows


def rows_from_classification(subset: dict, label_map: dict[int, int], names: dict[int, str],
                             split: str) -> list[str]:
    root = resolve_path(subset["images"])
    split_txt = root / CLASSIFICATION_SPLIT_FILES[split]
    if not split_txt.is_file():
        sys.exit(f"ERROR: official split file missing: {split_txt}")

    rows: list[str] = []
    for line in split_txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        image_name, original = parts[0], int(parts[1])
        project_label = label_map.get(original)
        if project_label is None:
            continue
        rows.append(
            f"images/{image_name},{original},{project_label},{names[original]},{split},,,,"
        )
    return rows


def compute_norm_stats(protocol, manifest_dir: Path) -> dict:
    """RGB mean/std over the *training* split only, after cropping.

    Measured after the crop because that is what the model actually sees. Box
    crops are much greener and much closer-in than the full frames, so stats
    taken from whole images would leave the network's inputs off-centre.
    """
    import numpy as np
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from ip102_bench.data import IP102Dataset

    print("\n[norm] computing RGB mean/std over the training split "
          "(a few minutes, done once)")
    dataset = IP102Dataset(
        manifest_path=manifest_dir / "train.csv",
        image_root=protocol.image_root,
        transform=transforms.Compose(
            [transforms.Resize((protocol.image_size, protocol.image_size)),
             transforms.ToTensor()]
        ),
        crop_mode=protocol.crop_mode,
        crop_margin=protocol.crop_margin,
    )
    loader = DataLoader(dataset, batch_size=64, num_workers=4)

    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    pixels = 0
    for images, _ in loader:
        batch = images.numpy()
        channel_sum += batch.sum(axis=(0, 2, 3))
        channel_sq_sum += (batch**2).sum(axis=(0, 2, 3))
        pixels += batch.shape[0] * batch.shape[2] * batch.shape[3]

    mean = channel_sum / pixels
    std = np.sqrt(channel_sq_sum / pixels - mean**2)
    return {
        "mean": [round(float(v), 6) for v in mean],
        "std": [round(float(v), 6) for v in std],
        "computed_from": "train split only",
        "subset": protocol.subset_name,
        "crop_mode": protocol.crop_mode,
        "num_images": len(dataset),
        "image_size": protocol.image_size,
        "note": "Never recompute from validation or test data.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default=None, help="Override protocol.yaml's subset.")
    parser.add_argument("--skip-norm-stats", action="store_true",
                        help="Write manifests only. Training still needs the stats.")
    args = parser.parse_args()

    protocol = load_protocol()
    if args.subset:
        protocol.dataset["subset"] = args.subset

    subset = protocol.subset
    source = subset.get("source", "classification")
    names = load_class_names()

    originals = subset.get("original_labels")
    if originals is None:  # the 'all' subset: every class, identity mapping
        originals = sorted(names)
    label_map = {orig: proj for proj, orig in enumerate(originals)}

    image_root = protocol.image_root
    if not image_root.is_dir():
        sys.exit(f"ERROR: image directory not found: {image_root}\n"
                 "Extract the IP102 archive, or fix `images` in protocol.yaml.")

    manifest_dir = protocol.manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f"subset  : {protocol.subset_name}  ({source}, {len(originals)} classes)")
    print(f"images  : {image_root}")
    print(f"crop    : {protocol.crop_mode} (margin {protocol.crop_margin})\n")

    expected = subset.get("expected_totals") or {}
    failures: list[str] = []

    for split in ("train", "validation", "test"):
        builder = rows_from_detection if source == "detection" else rows_from_classification
        rows = builder(subset, label_map, names, split)

        per_class: dict[int, int] = dict.fromkeys(range(len(originals)), 0)
        for row in rows:
            per_class[int(row.split(",")[2])] += 1

        target = expected.get(split)
        status = "OK" if target in (None, len(rows)) else "MISMATCH"
        if target is not None and len(rows) != target:
            failures.append(f"{split}: got {len(rows)}, expected {target}")
        absent = [names[originals[p]] for p, n in per_class.items() if n == 0]
        if absent:
            failures.append(f"{split}: classes with no images -> {absent}")

        out_path = manifest_dir / f"{split}.csv"
        out_path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
        print(f"[{status}] {out_path.name}: {len(rows)} images"
              + (f" (expected {target})" if target else ""))
        print("        per class: "
              + ", ".join(f"{p}={n}" for p, n in sorted(per_class.items())))

    meta = {
        "subset": protocol.subset_name,
        "source": source,
        "description": subset.get("description", "").strip(),
        "num_classes": len(originals),
        "classes": [
            {"project_label": proj, "original_label": orig, "class_name": names[orig]}
            for proj, orig in enumerate(originals)
        ],
        "expected_totals": expected or None,
    }
    (manifest_dir / "classes.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote classes.json ({len(originals)} classes)")

    if failures:
        sys.exit("\nFAILED validation:\n  " + "\n  ".join(failures))

    if not args.skip_norm_stats:
        stats = compute_norm_stats(protocol, manifest_dir)
        protocol.norm_stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"[ok] wrote {protocol.norm_stats_path.name}: "
              f"mean {stats['mean']}  std {stats['std']}")

    print("\nAll counts match. Data is ready.")


if __name__ == "__main__":
    main()
