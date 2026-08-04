#!/usr/bin/env python3
"""Build a read-only image-quality review manifest and contact sheets.

The Phase 7.3 gate. Measures objective image properties, optionally records what
a trained checkpoint predicted, and writes a manifest whose
``reviewer_decision`` and ``reviewer_notes`` columns are **empty** for a person
to fill in.

What this script will not do
    It never renames, moves, deletes, re-encodes or relabels a source image, and
    it never edits an official derived manifest. It proposes a *suspected*
    category and stops. A curated split, if a completed review ever justifies
    one, goes to a new versioned directory under
    ``data/processed/<scope>/curated/<version>/`` — the benchmark manifests stay
    byte-identical.

    The test split cannot be reviewed. Reviewing it would let its contents
    influence a data decision, which is exactly what Phase 9's discipline
    forbids; ``--split test`` is rejected rather than quietly ignored.

Objective versus suspected
    Only ``low_resolution`` and ``blurry`` are measured from pixels and asserted
    as quality flags. Everything else — ``tiny_subject``, ``symptom_only``,
    ``diagram_text``, ``unrelated``, ``ambiguous``, ``suspected_mislabel`` —
    needs human judgement. A confident model/label disagreement is queued as
    ``suspected_mislabel`` because it is worth a person's attention, not because
    it is evidence the label is wrong.

Examples:
    python scripts/review_images.py --split validation --limit 200
    python scripts/review_images.py --split validation --contact-sheets
    python scripts/review_images.py \
        --checkpoint artifacts/checkpoints/rice10_custom_protocolA/best.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.data.manifests import read_derived_manifest
from farm_pest_ai.data.review import (
    REVIEWABLE_SPLITS,
    ReviewError,
    ReviewRecord,
    ReviewThresholds,
    build_contact_sheet,
    measure_image,
    read_review_manifest,
    suggest_issue,
    write_review_manifest,
)
from farm_pest_ai.logging_config import get_logger

logger = get_logger("review_images")

#: How many thumbnails go on one contact sheet before it is split.
SHEET_CAPACITY = 42


def predict_split(
    config: Config, checkpoint: Path, split: str
) -> dict[str, tuple[int, float]]:
    """Predict every image in ``split`` with a trained checkpoint.

    Evaluation preprocessing is deterministic and augmentation-free, and the
    loader preserves official manifest order, so predictions join back to the
    manifest by position.

    Returns:
        ``{filename: (predicted_label, confidence)}``.
    """
    import dataclasses

    import torch

    from farm_pest_ai.data.loaders import (
        build_dataset,
        build_loader,
        runtime_config_from_config,
    )
    from farm_pest_ai.vision.checkpoints import load_checkpoint

    # A review scores images, it does not train on them, so every split is
    # loaded with *evaluation* semantics. Three training defaults would
    # otherwise corrupt the join back to the manifest:
    #
    #   * `runtime.drop_last` is true for training — right for BatchNorm, but it
    #     silently discards the final partial batch, 30 of rice10's 4,318 train
    #     images;
    #   * the train loader shuffles, so predictions would not line up with
    #     manifest order;
    #   * the train pipeline augments, and a review must describe the image on
    #     disk rather than a randomly cropped variant of it.
    runtime = dataclasses.replace(runtime_config_from_config(config), drop_last=False)
    dataset = build_dataset(config, split, augment=False)
    loader = build_loader(
        dataset,
        runtime,
        batch_size=int(config.section("training").get("batch_size", 64)),
        seed=config.seed,
        shuffle=False,
    )

    model, metadata, _ = load_checkpoint(
        checkpoint, scope=config.scope, map_location="cpu"
    )
    logger.info(
        "checkpoint loaded",
        extra={"scope": metadata.scope, "epoch": metadata.epoch},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    labels: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits.float(), dim=1)
            best, index = probabilities.max(dim=1)
            labels.extend(int(value) for value in index.cpu())
            confidences.extend(float(value) for value in best.cpu())

    records = dataset.records
    if len(labels) != len(records):
        raise ReviewError(
            f"{len(labels)} predictions for {len(records)} images; the loader did "
            "not preserve manifest coverage"
        )
    return {
        record.filename: (label, confidence)
        for record, label, confidence in zip(records, labels, confidences, strict=True)
    }


def build_records(
    config: Config,
    split: str,
    *,
    thresholds: ReviewThresholds,
    predictions: dict[str, tuple[int, float]],
    class_names: dict[int, str],
    limit: int | None,
) -> list[ReviewRecord]:
    """Measure every image in ``split`` and assemble its review rows."""
    records, _ = read_derived_manifest(
        config.paths.processed_dir, config.scope, split
    )
    if limit is not None:
        records = records[:limit]

    images_dir = config.paths.images_dir
    rows: list[ReviewRecord] = []
    for index, record in enumerate(records, start=1):
        path = images_dir / record.filename
        try:
            measured = measure_image(path, thresholds=thresholds)
        except ReviewError as error:
            logger.warning("undecodable image", extra={"file": record.filename})
            print(f"warning: {error}", file=sys.stderr)
            continue

        predicted = predictions.get(record.filename)
        label = predicted[0] if predicted else None
        confidence = predicted[1] if predicted else None
        correct = None if label is None else label == record.project_label

        rows.append(
            ReviewRecord(
                filename=record.filename,
                split=split,
                current_label=record.project_label,
                current_class_name=record.class_name,
                width=measured["width"],
                height=measured["height"],
                model_prediction=label,
                model_prediction_name=(
                    class_names.get(label, "") if label is not None else ""
                ),
                confidence=confidence,
                quality_flags=measured["quality_flags"],
                suspected_issue=suggest_issue(
                    measured["quality_flags"],
                    confidence=confidence,
                    prediction_correct=correct,
                    thresholds=thresholds,
                ),
            )
        )
        if index % 500 == 0:
            logger.info("measured", extra={"images": index, "split": split})

    return rows


def write_contact_sheets(
    rows: list[ReviewRecord], config: Config, destination: Path
) -> list[Path]:
    """Write one contact sheet per suspected issue, for human review."""
    grouped: dict[str, list[ReviewRecord]] = {}
    for row in rows:
        if row.suspected_issue:
            grouped.setdefault(row.suspected_issue, []).append(row)

    images_dir = config.paths.images_dir
    written: list[Path] = []
    for issue, members in sorted(grouped.items()):
        for start in range(0, len(members), SHEET_CAPACITY):
            page = members[start : start + SHEET_CAPACITY]
            entries = [
                (
                    images_dir / row.filename,
                    f"{row.filename} {row.current_class_name[:20]}",
                )
                for row in page
            ]
            sheet = build_contact_sheet(entries)
            number = start // SHEET_CAPACITY + 1
            path = destination / f"{issue}_{number:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(path)
            written.append(path)
    return written


def _print_summary(
    rows: list[ReviewRecord],
    *,
    scope: str,
    split: str,
    thresholds: ReviewThresholds,
    has_predictions: bool,
) -> None:
    """Print the review summary: what was measured, and what needs a human."""
    flags = Counter(flag for row in rows for flag in row.quality_flags)
    issues = Counter(row.suspected_issue for row in rows if row.suspected_issue)

    print()
    print(f"Image-quality review — {scope} / {split}")
    print("=" * 70)
    print(f"  images reviewed        {len(rows):,}")
    print(
        f"  thresholds             short side < {thresholds.min_short_side}px, "
        f"focus < {thresholds.blur_variance}, "
        f"confidence < {thresholds.low_confidence}"
    )
    print(
        f"  predictions            "
        f"{'yes' if has_predictions else 'no checkpoint given'}"
    )

    for title, counts in (
        ("objective quality flags (measured from pixels):", flags),
        ("suspected issues queued for human review:", issues),
    ):
        print()
        print(f"  {title}")
        if not counts:
            print("    none")
            continue
        for name, count in sorted(counts.items()):
            print(f"    {name:<22}{count:>6}  ({count / len(rows) * 100:.1f}%)")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Build a read-only image-quality review manifest.",
        default_configs=("base.yaml",),
    )
    parser.add_argument(
        "--split",
        default="validation",
        # The test split is deliberately absent from the choices, so no
        # invocation can name it.
        choices=list(REVIEWABLE_SPLITS),
        help="Split to review. The test split cannot be reviewed.",
    )
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="Optional checkpoint used to record predictions and confidence.",
    )
    parser.add_argument(
        "--limit", type=int, help="Review only the first N images, for a quick pass."
    )
    parser.add_argument(
        "--contact-sheets",
        action="store_true",
        help="Also write contact sheets grouped by suspected issue.",
    )
    parser.add_argument(
        "--min-short-side",
        type=int,
        default=None,
        help="Short side below which an image is flagged low_resolution.",
    )
    parser.add_argument(
        "--blur-variance",
        type=float,
        default=100.0,
        help="Focus measure below which an image is flagged blurry.",
    )
    parser.add_argument(
        "--low-confidence",
        type=float,
        default=0.35,
        help="Confidence below which an image is queued as ambiguous.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Where to write the review manifest CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing manifest that has more rows than this run "
            "would write, or that already carries reviewer decisions."
        ),
    )
    return parser


def _refuse_to_clobber(destination: Path, incoming: int, force: bool) -> str | None:
    """Return a refusal message when writing would destroy existing work.

    Two ways a write loses information, both of which happened or nearly
    happened in practice:

    * a ``--limit`` pass overwriting a complete review with a partial one;
    * any pass overwriting decisions a human has already entered.

    Returns:
        The reason to refuse, or ``None`` when writing is safe.
    """
    if force or not destination.is_file():
        return None

    try:
        existing = read_review_manifest(destination)
    except (ReviewError, OSError):
        # An unreadable file is not evidence of work worth protecting.
        return None

    decided = sum(1 for row in existing if (row.get("reviewer_decision") or "").strip())
    if decided:
        return (
            f"{destination} already carries {decided} reviewer decision(s). "
            "Refusing to overwrite completed human review; pass --force to "
            "discard it."
        )
    if len(existing) > incoming:
        return (
            f"{destination} holds {len(existing)} rows and this run would write "
            f"only {incoming}. Refusing to replace a fuller review with a "
            "partial one; pass --force if that is intended."
        )
    return None


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config: Config
    config, _ = bootstrap(args)

    if args.split not in REVIEWABLE_SPLITS:  # defence in depth behind `choices`
        print(
            f"error: {args.split!r} cannot be reviewed; the test split is "
            "reserved for Phase 9",
            file=sys.stderr,
        )
        return 2

    image_size = config.get("dataset.image_size") or [160, 160]
    thresholds = ReviewThresholds(
        min_short_side=args.min_short_side or int(min(image_size)),
        blur_variance=args.blur_variance,
        low_confidence=args.low_confidence,
    )

    class_names: dict[int, str] = {}
    predictions: dict[str, tuple[int, float]] = {}
    if args.checkpoint:
        try:
            predictions = predict_split(config, Path(args.checkpoint), args.split)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

    try:
        rows = build_records(
            config,
            args.split,
            thresholds=thresholds,
            predictions=predictions,
            class_names=class_names,
            limit=args.limit,
        )
    except (ReviewError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not rows:
        print("error: no images were reviewed", file=sys.stderr)
        return 2

    # Fill in class names now that every record's label is known.
    names = {row.current_label: row.current_class_name for row in rows}
    rows = [
        (
            row
            if row.model_prediction is None
            else ReviewRecord(
                **{
                    **row.__dict__,
                    "model_prediction_name": names.get(row.model_prediction, ""),
                }
            )
        )
        for row in rows
    ]

    destination = (
        Path(args.output)
        if args.output
        else config.paths.reports_dir
        / f"image_review_{config.scope.name}_{args.split}.csv"
    )
    refusal = _refuse_to_clobber(destination, len(rows), args.force)
    if refusal is not None:
        print(f"error: {refusal}", file=sys.stderr)
        return 2

    write_review_manifest(rows, destination)

    _print_summary(
        rows,
        scope=config.scope.name,
        split=args.split,
        thresholds=thresholds,
        has_predictions=bool(predictions),
    )

    sheets: list[Path] = []
    if args.contact_sheets:
        sheet_dir = config.paths.reports_dir / "contact_sheets" / args.split
        sheets = write_contact_sheets(rows, config, sheet_dir)
        print()
        print(f"  contact sheets         {len(sheets)} written to {sheet_dir}")

    print()
    print(f"Review manifest: {destination}")
    print(
        "  reviewer_decision and reviewer_notes are empty by design. "
        "No image was modified and no label was changed."
    )
    logger.info(
        "review manifest written",
        extra={"rows": len(rows), "output": str(destination), "sheets": len(sheets)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
