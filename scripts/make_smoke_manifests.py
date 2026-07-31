"""Step 7: build the five-class smoke manifests.

Filters the official splits down to five classes remapped to project labels 0-4,
purely to exercise the training -> checkpoint -> evaluate -> predict pipeline
before committing to the ten-class runs. These are not experimental results.

    python scripts/make_smoke_manifests.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.setup_data import SPLIT_FILES, load_class_names, read_split  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

# Project label -> original IP102 id, from the Step 7 table.
SMOKE_CLASSES = [(0, 0), (1, 3), (2, 7), (3, 8), (4, 10)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=PROJECT_ROOT / "IP102_v1.1" / "Classification" / "ip102_v1.1")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data_manifests")
    args = parser.parse_args()

    all_names = load_class_names(args.dataset_root.parent / "classes.txt")
    original_to_project = {orig: proj for proj, orig in SMOKE_CLASSES}
    class_names = {proj: all_names[orig] for proj, orig in SMOKE_CLASSES}

    for split, filename in SPLIT_FILES.items():
        lines = ["image_path,original_label,project_label,class_name,split"]
        for image_name, original_label in read_split(args.dataset_root / filename):
            project_label = original_to_project.get(original_label)
            if project_label is None:
                continue
            lines.append(
                f"images/{image_name},{original_label},{project_label},"
                f"{class_names[project_label]},{split}"
            )
        out_path = args.manifest_dir / f"smoke_{split}.csv"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[ok] {out_path.name}: {len(lines) - 1} images")

    print("\nRun the smoke experiment with:")
    print("  python -m src.train --config configs/alexnet.yaml --epochs 3 --num-classes 5 \\")
    print("      --train-manifest data_manifests/smoke_train.csv \\")
    print("      --val-manifest data_manifests/smoke_validation.csv \\")
    print("      --test-manifest data_manifests/smoke_test.csv \\")
    print("      --output-root runs/_smoke")


if __name__ == "__main__":
    main()
