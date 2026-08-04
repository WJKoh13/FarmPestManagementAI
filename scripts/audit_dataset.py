#!/usr/bin/env python3
"""Audit the IP102 dataset: integrity, duplicates, leakage and image properties.

Implements the checks Phase 1 deferred, all of which read image *content*:

* full decode of every image in the scope, catching truncated files,
* SHA-256 content hashing,
* exact-content duplicate groups,
* exact-content cross-split leakage, which filename checks cannot rule out,
* source dimensions and the sub-160px upscale cohort,
* per-class distributions for the derived manifests.

The source tree is only ever read. Reports are written to ``data/reports/``.

Hashing and decoding 75,222 images takes a while; ``--scope rice10`` (7,205
images) is the fast path, and ``--limit`` bounds the work further for a smoke
test. Progress is logged as it goes.

Examples:
    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --scope full102
    python scripts/audit_dataset.py --scope rice10 --limit 200 --no-report
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.data.audit import (
    DecodeResult,
    check_integrity,
    find_leakage,
    probe_many,
    split_distribution,
)
from farm_pest_ai.data.manifests import (
    SPLITS,
    ManifestError,
    ManifestRecord,
    atomic_write_text,
    read_derived_manifest,
    read_source_manifest,
)
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.reproducibility import environment_snapshot
from farm_pest_ai.scopes import CLASS_MAPPING_VERSION

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_manifests import source_manifest_paths


def audit_integrity(config: Config) -> dict[str, Any]:
    """Run filename-level integrity checks against the source manifests.

    Integrity is a property of the dataset as a whole, so this always reads the
    full source manifests regardless of the active scope. Auditing only a scope
    subset would report every out-of-scope image as unreferenced.
    """
    logger = get_logger("audit_dataset")
    splits = {
        split: read_source_manifest(path)
        for split, path in source_manifest_paths(config).items()
    }
    report = check_integrity(splits, config.paths.images_dir)
    logger.info(
        "integrity: %d records, %d files on disk, ok=%s",
        report.total_records,
        report.images_on_disk,
        report.ok,
        extra={"event": "integrity", **report.to_dict()},
    )
    return report.to_dict()


def load_manifests(
    config: Config, limit: int | None
) -> dict[str, tuple[ManifestRecord, ...]]:
    """Read the derived manifests for the active scope.

    Raises:
        ManifestError: If a manifest is missing, which means
            ``scripts/build_manifests.py`` has not been run for this scope.
    """
    scope = config.dataset.scope
    processed_dir = config.paths.processed_dir
    manifests: dict[str, tuple[ManifestRecord, ...]] = {}
    for split in SPLITS:
        records, _ = read_derived_manifest(processed_dir, scope, split)
        manifests[split] = records[:limit] if limit else records
    return manifests


def audit_distributions(
    config: Config, manifests: dict[str, tuple[ManifestRecord, ...]]
) -> dict[str, Any]:
    """Compute per-class distributions for every split of the active scope."""
    num_classes = config.dataset.num_classes
    return {
        split: split_distribution(split, records, num_classes).to_dict()
        for split, records in manifests.items()
    }


def audit_images(
    config: Config,
    manifests: dict[str, tuple[ManifestRecord, ...]],
    *,
    full_decode: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Decode, measure and hash every image in the active scope.

    Returns:
        A per-split summary, and the ``split -> filename -> sha256`` mapping the
        leakage check consumes.
    """
    logger = get_logger("audit_dataset")
    images_dir = config.paths.images_dir
    summaries: dict[str, Any] = {}
    hashes: dict[str, dict[str, str]] = {}

    for split, records in manifests.items():

        def report(done: int, total: int, split: str = split) -> None:
            logger.info(
                "%s: probed %d/%d images",
                split,
                done,
                total,
                extra={"event": "probe_progress", "split": split, "done": done},
            )

        probes = probe_many(
            records,
            images_dir,
            full_decode=full_decode,
            compute_hash=True,
            progress=report,
        )
        result = DecodeResult(probes=probes, split=split)
        summaries[split] = result.to_dict()
        hashes[split] = {
            p.filename: p.sha256 for p in probes if p.sha256 is not None
        }
        logger.info(
            "%s: %d decoded, %d failed",
            split,
            len(result.ok_probes),
            len(result.failures),
            extra={
                "event": "decode_summary",
                "split": split,
                "failed": len(result.failures),
            },
        )
    return summaries, hashes


def audit_leakage(
    config: Config,
    manifests: dict[str, tuple[ManifestRecord, ...]],
    hashes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Find exact-content duplicates and cross-split leakage."""
    logger = get_logger("audit_dataset")
    scope = config.dataset.scope
    labels = {
        split: {r.filename: r.project_label for r in records}
        for split, records in manifests.items()
    }
    report = find_leakage(hashes, labels, scope.name)
    summary = report.summary()
    logger.info(
        "leakage: %d duplicate groups, %d cross-split, %d label conflicts",
        summary["duplicate_groups"],
        summary["cross_split_groups"],
        summary["label_conflict_groups"],
        extra={"event": "leakage", **summary},
    )
    # Cross-split groups are the ones Phase 9 must exclude from headline
    # metrics, so they are listed in full rather than merely counted. The
    # within-split groups matter too — a duplicated training image is silently
    # weighted twice — so every group is recorded, not just the leaking ones.
    summary["cross_split_detail"] = [g.to_dict() for g in report.cross_split_groups]
    summary["label_conflict_detail"] = [
        g.to_dict() for g in report.label_conflict_groups
    ]
    summary["duplicate_detail"] = [g.to_dict() for g in report.duplicate_groups]
    summary["leaked_files"] = {
        split: list(report.leaked_files(split)) for split in SPLITS
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Audit dataset integrity, duplicates, leakage and image properties.",
        default_configs=("base.yaml",),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Probe at most this many images per split. For a quick smoke test.",
    )
    parser.add_argument(
        "--header-only",
        action="store_true",
        help=(
            "Read image headers instead of decoding pixels. Much faster, but "
            "will not detect truncated files."
        ),
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Run only the manifest-level checks; skip decoding and hashing.",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Print results without writing a report."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    config, _ = bootstrap(args)
    logger = get_logger("audit_dataset")

    if args.print_config:
        print(config.to_yaml())
        return 0

    scope = config.dataset.scope
    started = datetime.now(timezone.utc)

    try:
        manifests = load_manifests(config, args.limit)
    except ManifestError:
        logger.exception(
            "derived manifests unavailable", extra={"event": "manifest_missing"}
        )
        return 1

    report: dict[str, Any] = {
        "scope": scope.name,
        "num_classes": scope.num_classes,
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "manifest_version": config.dataset.manifest_version,
        "limit": args.limit,
        "full_decode": not args.header_only,
        "started_at": started.isoformat(timespec="seconds"),
        "integrity": audit_integrity(config),
        "distributions": audit_distributions(config, manifests),
    }

    if not args.skip_images:
        images, hashes = audit_images(
            config, manifests, full_decode=not args.header_only
        )
        report["images"] = images
        report["leakage"] = audit_leakage(config, manifests, hashes)

    report["environment"] = environment_snapshot()
    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print_report(report)

    if not args.no_report:
        path = config.paths.reports_dir / f"dataset_audit_{scope.name}.json"
        atomic_write_text(path, json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nReport written to {path}")

    return 0 if audit_passed(report) else 1


def audit_passed(report: dict[str, Any]) -> bool:
    """Decide the exit status.

    Integrity failures and decode failures are hard errors: they mean the data
    cannot be trusted. Duplicates and leakage are *findings* — they are real
    properties of IP102 that the project must account for in Phase 9, not
    reasons for this script to fail.
    """
    if not report["integrity"]["ok"]:
        return False
    images = report.get("images", {})
    return all(split.get("failed", 0) == 0 for split in images.values())


def _print_integrity(integrity: dict[str, Any]) -> None:
    """Print the filename-level integrity section."""
    print(f"\nIntegrity: {'OK' if integrity['ok'] else 'FAILED'}")
    print(f"  source records      {integrity['total_records']}")
    print(f"  images on disk      {integrity['images_on_disk']}")
    print(f"  missing from disk   {integrity['missing_from_disk']}")
    print(f"  unreferenced        {integrity['unreferenced_on_disk']}")
    print(f"  cross-split names   {integrity['cross_split_filenames']}")
    print(f"  conflicting labels  {integrity['conflicting_labels']}")


def _print_distributions(distributions: dict[str, Any]) -> None:
    """Print the per-split class distribution table."""
    print("\nDistribution")
    print(
        f"  {'SPLIT'.ljust(12)}{'TOTAL'.rjust(8)}{'CLASSES'.rjust(9)}"
        f"{'MIN'.rjust(7)}{'MAX'.rjust(7)}{'MEDIAN'.rjust(8)}{'IMBAL'.rjust(8)}"
    )
    for split, dist in distributions.items():
        ratio = dist["imbalance_ratio"]
        print(
            f"  {split.ljust(12)}{dist['total']:>8}{dist['present_classes']:>9}"
            f"{dist['min']:>7}{dist['max']:>7}{dist['median']:>8}"
            f"{(f'{ratio}x' if ratio else '-'):>8}"
        )
        if dist["empty_classes"]:
            print(f"    classes with no records: {dist['empty_classes']}")


def _print_images(images: dict[str, Any]) -> None:
    """Print the decode and dimension section."""
    print("\nImages")
    for split, summary in images.items():
        dims = summary["dimensions"]
        print(
            f"  {split}: {summary['decoded']}/{summary['inspected']} decoded, "
            f"{summary['failed']} failed"
        )
        print(f"    formats {summary['formats']} modes {summary['modes']}")
        if not dims.get("count"):
            continue
        short = dims["short_side"]
        print(
            f"    short side min/median/max "
            f"{short['min']}/{short['median']}/{short['max']}"
        )
        print(
            f"    below 160px: {dims['short_side_below_160']} "
            f"({dims['short_side_below_160_pct']}%)  "
            f"below 224px: {dims['short_side_below_224']} "
            f"({dims['short_side_below_224_pct']}%)"
        )


def _print_leakage(leakage: dict[str, Any]) -> None:
    """Print the duplicate and cross-split leakage section."""
    print("\nExact-content duplicates and leakage")
    print(f"  duplicate groups     {leakage['duplicate_groups']}")
    print(f"  duplicate files      {leakage['duplicate_files']}")
    print(f"  within-split groups  {leakage['within_split_groups']}")
    print(f"  CROSS-SPLIT groups   {leakage['cross_split_groups']}")
    print(f"  label conflicts      {leakage['label_conflict_groups']}")
    if leakage["cross_split_pairs"]:
        print(f"  split pairs          {leakage['cross_split_pairs']}")
    leaked = leakage["leaked_files_per_split"]
    if any(leaked.values()):
        print(f"  leaked files/split   {leaked}")


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable summary of the audit."""
    print(f"\n=== Dataset audit: scope {report['scope']} ===")
    _print_integrity(report["integrity"])
    _print_distributions(report["distributions"])
    if images := report.get("images"):
        _print_images(images)
    if leakage := report.get("leakage"):
        _print_leakage(leakage)


if __name__ == "__main__":
    raise SystemExit(main())
