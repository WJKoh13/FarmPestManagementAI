"""Step 1: extract the IP102 classification archive and build the shared manifests.

Reads the OFFICIAL split files (train.txt / val.txt / test.txt) shipped with the
dataset, keeps only the ten selected rice-pest classes, remaps their original
IP102 ids to contiguous project labels 0-9, and writes one CSV per split.

No new random splits are ever created here.

    python scripts/setup_data.py
    python scripts/setup_data.py --tar ~/Downloads/ip102_v1.1.tar --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Project label -> original IP102 id, straight from the team instructions.
SELECTED_CLASSES: list[tuple[int, int]] = [
    (0, 0),
    (1, 1),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
]

EXPECTED_TOTALS = {"train": 4318, "validation": 721, "test": 2166}

# Official split file -> our split name. IP102 calls the validation split "val".
SPLIT_FILES = {"train": "train.txt", "validation": "val.txt", "test": "test.txt"}


def slugify(name: str) -> str:
    """'rice leaf roller ' -> 'rice_leaf_roller'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_class_names(classes_txt: Path) -> dict[int, str]:
    """Map original IP102 id (0-based) -> slugified class name.

    classes.txt is 1-indexed ('1  rice leaf roller'), so id n sits on line n+1.
    """
    names: dict[int, str] = {}
    for line in classes_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        if not head.isdigit():
            continue
        names[int(head) - 1] = slugify(rest)
    return names


def extract_archive(tar_path: Path, dest: Path, force: bool) -> Path:
    """Extract <tar>/ip102_v1.1/ into dest. Returns the dataset root."""
    dataset_root = dest / "ip102_v1.1"
    images_dir = dataset_root / "images"

    if images_dir.is_dir() and not force:
        print(f"[skip] dataset already extracted at {dataset_root}")
        return dataset_root

    if not tar_path.is_file():
        sys.exit(
            f"ERROR: archive not found: {tar_path}\n"
            "Pass the correct path with --tar <path to ip102_v1.1.tar>"
        )

    print(f"[extract] {tar_path} -> {dest} (this takes a few minutes, ~3.2 GB)")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(dest, filter="data")
    return dataset_root


def read_split(split_txt: Path) -> list[tuple[str, int]]:
    """Parse '<filename> <original_label>' lines."""
    rows: list[tuple[str, int]] = []
    for line in split_txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        rows.append((parts[0], int(parts[1])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tar",
        type=Path,
        default=Path.home() / "Downloads" / "ip102_v1.1.tar",
        help="Path to the IP102 classification archive.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=PROJECT_ROOT / "IP102_v1.1" / "Classification",
        help="Where to extract the archive.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data_manifests",
        help="Where to write the CSV manifests.",
    )
    parser.add_argument("--force", action="store_true", help="Re-extract even if present.")
    args = parser.parse_args()

    dataset_root = extract_archive(args.tar, args.dest, args.force)

    classes_txt = args.dest / "classes.txt"
    if not classes_txt.is_file():
        sys.exit(f"ERROR: classes.txt not found at {classes_txt}")
    all_names = load_class_names(classes_txt)

    original_to_project = {orig: proj for proj, orig in SELECTED_CLASSES}
    class_names = {proj: all_names[orig] for proj, orig in SELECTED_CLASSES}

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for split, filename in SPLIT_FILES.items():
        split_txt = dataset_root / filename
        if not split_txt.is_file():
            sys.exit(f"ERROR: official split file missing: {split_txt}")

        lines = ["image_path,original_label,project_label,class_name,split"]
        per_class = dict.fromkeys(class_names, 0)

        for image_name, original_label in read_split(split_txt):
            project_label = original_to_project.get(original_label)
            if project_label is None:
                continue
            name = class_names[project_label]
            lines.append(
                f"images/{image_name},{original_label},{project_label},{name},{split}"
            )
            per_class[project_label] += 1

        count = len(lines) - 1
        expected = EXPECTED_TOTALS[split]
        status = "OK" if count == expected else "MISMATCH"
        if count != expected:
            failures.append(f"{split}: got {count}, expected {expected}")
        missing = [class_names[p] for p, n in per_class.items() if n == 0]
        if missing:
            failures.append(f"{split}: classes absent -> {missing}")

        out_path = args.manifest_dir / f"{split}.csv"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[{status}] {out_path.name}: {count} images (expected {expected})")
        print("        per class: " + ", ".join(f"{p}={n}" for p, n in sorted(per_class.items())))

    meta = {
        "dataset_root": str(dataset_root.relative_to(PROJECT_ROOT)),
        "num_classes": len(SELECTED_CLASSES),
        "classes": [
            {
                "project_label": proj,
                "original_label": orig,
                "class_name": class_names[proj],
            }
            for proj, orig in SELECTED_CLASSES
        ],
        "expected_totals": EXPECTED_TOTALS,
    }
    meta_path = args.manifest_dir / "selected_classes.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {meta_path.name}")

    if failures:
        sys.exit("FAILED validation:\n  " + "\n  ".join(failures))
    print("\nAll manifest counts match the expected totals.")


if __name__ == "__main__":
    main()
