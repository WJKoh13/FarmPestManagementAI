#!/usr/bin/env python3
"""Audit the detection bounding boxes that the E4/E5 crop arms consume.

Read-only. Nothing under ``ip102_v1.1`` is renamed, moved, deleted, re-encoded
or rewritten; contact sheets are new files written under the reports directory.

The audit exists because "cropping helps" and "cropping helps *here*" are
different claims. If most boxes already cover most of the frame, a crop arm can
only differ marginally from its control, and a null result would say more about
the boxes than about cropping. This script measures that up front:

* total boxes, and how many are missing, structurally invalid or degenerate,
* percentiles of box area as a fraction of image area,
* how many boxes fall below the 10%, 25% and 50% area thresholds,
* a reproducible contact sheet of padded crops sampled across classes,
* boxes that touch an image boundary, where padding clamps rather than grows.

Examples:
    python scripts/audit_crops.py --scope det_top10
    python scripts/audit_crops.py --scope det_top15 --contact-sheets
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.data.detection import (
    DEFAULT_PADDING,
    BoundingBox,
    DetectionDataError,
    box_statistics,
    build_detection_records,
    crop_with_padding,
    detection_root,
    load_boxes,
    partition_records,
    scope_suffix,
)
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.scopes import ScopeSpec, is_detection_scope

LOGGER = get_logger(__name__)

#: Splits the audit covers. The test split is deliberately absent: these
#: experiments never open it, and an audit is not a reason to start.
AUDIT_SPLITS: tuple[str, ...] = ("train", "validation")

#: Contact-sheet geometry. Each cell shows one padded crop.
SHEET_COLUMNS = 5
SHEET_CELL = 160


def _image_sizes(
    images_dir: Path, filenames: list[str]
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """Read (width, height) for each file without decoding pixel data.

    Pillow reads the dimensions from the header, so this stays fast over
    thousands of images.
    """
    from PIL import Image

    sizes: dict[str, tuple[int, int]] = {}
    unreadable: list[str] = []
    for name in filenames:
        try:
            with Image.open(images_dir / name) as image:
                sizes[name] = (int(image.width), int(image.height))
        except (OSError, ValueError):
            unreadable.append(name)
    return sizes, unreadable


def _boundary_cases(
    boxes: dict[str, BoundingBox],
    sizes: dict[str, tuple[int, int]],
    padding: float,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Find boxes whose padded region clamps against an image edge.

    These are the cases where padding cannot deliver the requested context on
    every side, so they are the ones most likely to behave unlike the rest.
    """
    from farm_pest_ai.data.detection import pad_and_clamp

    found: list[dict[str, Any]] = []
    for name, box in boxes.items():
        size = sizes.get(name)
        if size is None:
            continue
        width, height = size
        padded = pad_and_clamp(box, width, height, padding)
        clamped = [
            side
            for side, touching in (
                ("left", padded.x1 <= 0.0 and box.x1 - box.width * padding < 0.0),
                ("top", padded.y1 <= 0.0 and box.y1 - box.height * padding < 0.0),
                (
                    "right",
                    padded.x2 >= width and box.x2 + box.width * padding > width,
                ),
                (
                    "bottom",
                    padded.y2 >= height and box.y2 + box.height * padding > height,
                ),
            )
            if touching
        ]
        if clamped:
            found.append(
                {
                    "filename": name,
                    "image_size": [width, height],
                    "box": box.to_dict(),
                    "padded": padded.to_dict(),
                    "clamped_sides": clamped,
                }
            )
        if len(found) >= limit:
            break
    return found


def _contact_sheet(
    images_dir: Path,
    entries: list[tuple[str, int]],
    boxes: dict[str, BoundingBox],
    padding: float,
    output: Path,
) -> None:
    """Render padded crops into one PNG grid.

    Sources are opened read-only and cropped in memory; the originals are never
    written back.
    """
    from PIL import Image, ImageDraw

    if not entries:
        return
    rows = (len(entries) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    label_band = 14
    sheet = Image.new(
        "RGB",
        (SHEET_COLUMNS * SHEET_CELL, rows * (SHEET_CELL + label_band)),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (name, label) in enumerate(entries):
        box = boxes.get(name)
        if box is None:
            continue
        try:
            with Image.open(images_dir / name) as image:
                image.load()
                crop = crop_with_padding(image.convert("RGB"), box, padding)
        except (OSError, ValueError):
            continue
        crop = crop.resize((SHEET_CELL, SHEET_CELL), Image.BILINEAR)
        column = index % SHEET_COLUMNS
        row = index // SHEET_COLUMNS
        x = column * SHEET_CELL
        y = row * (SHEET_CELL + label_band)
        sheet.paste(crop, (x, y + label_band))
        draw.text((x + 2, y + 2), f"c{label} {name[:16]}", fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    LOGGER.info("wrote contact sheet %s (%d crops)", output, len(entries))


def audit_scope(
    config: Config,
    spec: ScopeSpec,
    *,
    padding: float,
    contact_sheets: bool,
    per_class_samples: int,
) -> dict[str, Any]:
    """Audit every box a scope's train and validation splits reference."""
    dataset_root = config.paths.dataset_root
    images_dir = detection_root(dataset_root) / "JPEGImages"
    boxes, invalid = load_boxes(dataset_root, spec)

    report: dict[str, Any] = {
        "scope": spec.name,
        "num_classes": spec.num_classes,
        "padding": padding,
        "boxes_file": f"boxes_{scope_suffix(spec)}.json",
        "boxes_total_in_file": len(boxes) + len(invalid),
        "boxes_valid_in_file": len(boxes),
        "boxes_structurally_invalid": len(invalid),
        "invalid_detail": invalid,
        "splits": {},
    }

    referenced: list[str] = []
    per_class: dict[int, list[str]] = defaultdict(list)
    for split in AUDIT_SPLITS:
        records = build_detection_records(dataset_root, spec, split)
        partition = partition_records(records, boxes, invalid)
        counts: dict[int, int] = defaultdict(int)
        for record in records:
            counts[record.project_label] += 1
        for record in partition.kept:
            referenced.append(record.filename)
            per_class[record.project_label].append(record.filename)
        report["splits"][split] = {
            "records": len(records),
            "usable": len(partition.kept),
            "dropped": len(partition.dropped),
            "dropped_detail": [
                {"filename": r.filename, "reason": reason}
                for r, reason in partition.dropped
            ],
            "per_class_counts": {str(k): counts[k] for k in sorted(counts)},
        }

    sizes, unreadable = _image_sizes(images_dir, referenced)
    referenced_boxes = {name: boxes[name] for name in referenced if name in boxes}
    report["images_unreadable"] = unreadable
    report["area_statistics"] = box_statistics(referenced_boxes, sizes)
    report["boundary_examples"] = _boundary_cases(referenced_boxes, sizes, padding)

    # A crop arm can only differ from its control where the padded box is
    # smaller than the frame. Quantifying this is the difference between "the
    # crop did not help" and "the crop barely changed the input".
    from farm_pest_ai.data.detection import pad_and_clamp

    full_frame = 0
    for name, box in referenced_boxes.items():
        size = sizes.get(name)
        if size is None:
            continue
        width, height = size
        padded = pad_and_clamp(box, width, height, padding)
        if (
            padded.x1 <= 0.0
            and padded.y1 <= 0.0
            and padded.x2 >= width
            and padded.y2 >= height
        ):
            full_frame += 1
    measured = report["area_statistics"]["boxes_measured"]
    report["padded_crop_covers_full_frame"] = {
        "count": full_frame,
        "percent": round(100.0 * full_frame / measured, 3) if measured else 0.0,
    }

    if contact_sheets:
        sheet_dir = Path(config.paths.reports_dir) / "crop_audit" / spec.name
        entries: list[tuple[str, int]] = []
        for label in sorted(per_class):
            for name in sorted(per_class[label])[:per_class_samples]:
                entries.append((name, label))
        _contact_sheet(
            images_dir, entries, referenced_boxes, padding, sheet_dir / "crops.png"
        )
        boundary_entries = [
            (case["filename"], -1) for case in report["boundary_examples"]
        ]
        _contact_sheet(
            images_dir,
            boundary_entries,
            referenced_boxes,
            padding,
            sheet_dir / "boundary_crops.png",
        )
        report["contact_sheets"] = {
            "crops": str(sheet_dir / "crops.png"),
            "boundary_crops": str(sheet_dir / "boundary_crops.png"),
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = base_parser(description=__doc__ or "")
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING,
        help="Padding fraction applied per side (default: %(default)s).",
    )
    parser.add_argument(
        "--contact-sheets",
        action="store_true",
        help="Render PNG contact sheets of representative padded crops.",
    )
    parser.add_argument(
        "--per-class-samples",
        type=int,
        default=3,
        help="Crops per class on the contact sheet (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path (default: <reports_dir>/crop_audit_<scope>.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the audit and write its report."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config, _ = bootstrap(args)
    spec = config.dataset.scope

    if not is_detection_scope(spec):
        parser.error(
            f"--scope must be a detection scope (det_top10 or det_top15), got "
            f"{spec.name!r}"
        )

    try:
        report = audit_scope(
            config,
            spec,
            padding=float(args.padding),
            contact_sheets=bool(args.contact_sheets),
            per_class_samples=int(args.per_class_samples),
        )
    except DetectionDataError as exc:
        LOGGER.error("crop audit failed: %s", exc)
        return 1

    output = args.output or (
        Path(config.paths.reports_dir) / f"crop_audit_{spec.name}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    stats = report["area_statistics"]
    thresholds = stats["thresholds"]
    print(f"\nCrop audit - {spec.name} (padding {report['padding']:.0%})")
    print(f"  boxes in file        : {report['boxes_total_in_file']}")
    print(f"  structurally invalid : {report['boxes_structurally_invalid']}")
    for split, payload in report["splits"].items():
        print(
            f"  {split:10}         : {payload['usable']} usable, "
            f"{payload['dropped']} dropped"
        )
    print(f"  boxes measured       : {stats['boxes_measured']}")
    percentiles = stats["area_ratio_percentiles"]
    print(
        f"  area ratio p10/p50/p90: {percentiles['p10']:.4f} / "
        f"{percentiles['p50']:.4f} / {percentiles['p90']:.4f}"
    )
    for label in ("below_10pct", "below_25pct", "below_50pct"):
        entry = thresholds[label]
        print(f"  {label:20}: {entry['count']} ({entry['percent']}%)")
    covers = report["padded_crop_covers_full_frame"]
    print(
        f"  padded crop = full frame: {covers['count']} ({covers['percent']}%) "
        f"- these samples are identical in both arms"
    )
    print(f"\nreport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
