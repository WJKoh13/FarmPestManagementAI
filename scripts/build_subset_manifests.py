"""Build reproducible IP102 manifests from a tracked subset definition.

The official IP102 train/validation/test lists remain unchanged. This script
only filters them to the requested original labels and remaps those labels to
contiguous project labels.

Example:
    python scripts/build_subset_manifests.py \
        --definition data_manifests/broad15_classes.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = {"train": "train.txt", "validation": "val.txt", "test": "test.txt"}


def slugify(name: str) -> str:
    """Convert an IP102 display name to the manifest naming convention."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_ip102_class_names(classes_file: Path) -> dict[int, str]:
    """Map zero-based IP102 labels to normalized class names."""
    names: dict[int, str] = {}
    for line in classes_file.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        line = line.strip()
        if not line:
            continue
        raw_number, separator, raw_name = line.partition(" ")
        if separator and raw_number.isdigit():
            names[int(raw_number) - 1] = slugify(raw_name)
    return names


def read_official_split(split_file: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line_number, line in enumerate(
        split_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Malformed line {line_number} in {split_file}: {line!r}"
            )
        rows.append((parts[0], int(parts[1])))
    return rows


def validate_definition(definition: dict, official_names: dict[int, str]) -> None:
    classes = definition.get("classes", [])
    expected_num_classes = definition.get("num_classes")
    if len(classes) != expected_num_classes:
        raise ValueError(
            f"Definition contains {len(classes)} classes but num_classes is "
            f"{expected_num_classes}."
        )

    project_labels = [int(item["project_label"]) for item in classes]
    expected_labels = list(range(len(classes)))
    if project_labels != expected_labels:
        raise ValueError(
            "Project labels must be ordered and contiguous: "
            f"expected {expected_labels}, got {project_labels}."
        )

    original_labels = [int(item["original_label"]) for item in classes]
    if len(set(original_labels)) != len(original_labels):
        raise ValueError("Every original IP102 label must be unique.")

    for item in classes:
        original_label = int(item["original_label"])
        expected_name = official_names.get(original_label)
        if expected_name is None:
            raise ValueError(f"Unknown original IP102 label {original_label}.")
        if item["class_name"] != expected_name:
            raise ValueError(
                f"Class-name mismatch for original label {original_label}: "
                f"definition={item['class_name']!r}, IP102={expected_name!r}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--definition",
        type=Path,
        default=Path("data_manifests/broad15_classes.json"),
        help="Tracked JSON subset definition, relative to the repository root.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data_manifests"),
        help="Output directory, relative to the repository root.",
    )
    args = parser.parse_args()

    definition_path = resolve_project_path(args.definition)
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    dataset_root = resolve_project_path(definition["dataset_root"])
    classes_file = resolve_project_path(definition["classes_file"])
    output_dir = resolve_project_path(args.manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    official_names = load_ip102_class_names(classes_file)
    validate_definition(definition, official_names)

    classes = definition["classes"]
    original_to_class = {
        int(item["original_label"]): item for item in classes
    }
    expected_totals = definition["expected_totals"]
    prefix = definition["manifest_prefix"]

    for split_name, official_filename in SPLIT_FILES.items():
        official_path = dataset_root / official_filename
        rows: list[dict] = []
        per_class = dict.fromkeys(range(len(classes)), 0)

        for image_name, original_label in read_official_split(official_path):
            selected = original_to_class.get(original_label)
            if selected is None:
                continue
            project_label = int(selected["project_label"])
            rows.append(
                {
                    "image_path": f"images/{image_name}",
                    "original_label": original_label,
                    "project_label": project_label,
                    "class_name": selected["class_name"],
                    "split": split_name,
                }
            )
            per_class[project_label] += 1

        expected = int(expected_totals[split_name])
        if len(rows) != expected:
            raise ValueError(
                f"{split_name} contains {len(rows)} selected images; "
                f"expected {expected}."
            )
        if any(count == 0 for count in per_class.values()):
            raise ValueError(f"{split_name} has an empty selected class: {per_class}")

        output_path = output_dir / f"{prefix}_{split_name}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "image_path",
                    "original_label",
                    "project_label",
                    "class_name",
                    "split",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        counts = ", ".join(
            f"{project_label}={count}"
            for project_label, count in per_class.items()
        )
        print(f"[OK] {output_path.name}: {len(rows)} images")
        print(f"     per class: {counts}")

    print(
        f"\nBuilt {definition['subset_name']} from the official IP102 splits. "
        "No random split was created."
    )


if __name__ == "__main__":
    main()
