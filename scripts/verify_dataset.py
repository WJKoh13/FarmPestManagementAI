#!/usr/bin/env python3
"""Verify the derived manifests against the source data and audited counts.

A fast gate meant to run before any training job. Unlike
``scripts/audit_dataset.py`` it does not decode or hash images, so it finishes
in seconds and can be run routinely.

Checks:

* every derived manifest exists and its metadata matches the active scope and
  the current class-mapping version,
* record counts match the ``expected_counts`` measured in Phase 1,
* project labels are in range and every class is represented,
* the derived labels agree with the source manifests record for record,
* the class mapping on disk matches :mod:`farm_pest_ai.scopes`,
* every referenced image file exists (a stat, not a decode),
* no filename appears in more than one split.

Examples:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --scope full102
    python scripts/verify_dataset.py --skip-file-check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.data.manifests import (
    SPLITS,
    ManifestError,
    ManifestRecord,
    manifest_csv_path,
    read_source_manifest,
)
from farm_pest_ai.data.manifests import read_derived_manifest as _read_derived
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.scopes import CLASS_MAPPING_VERSION, ScopeSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_manifests import source_manifest_paths

#: Collected human-readable problems. Checks append rather than raise so that
#: one run reports every fault, not just the first.
Problems = list[str]


def check_manifests_present(
    config: Config, problems: Problems
) -> dict[str, tuple[ManifestRecord, ...]]:
    """Read every derived manifest, recording any that cannot be loaded."""
    scope = config.dataset.scope
    processed_dir = config.paths.processed_dir
    manifests: dict[str, tuple[ManifestRecord, ...]] = {}
    for split in SPLITS:
        try:
            records, metadata = _read_derived(processed_dir, scope, split)
        except ManifestError as exc:
            problems.append(str(exc))
            continue
        manifests[split] = records
        if metadata.get("manifest_version") != config.dataset.manifest_version:
            problems.append(
                f"{split}: manifest_version {metadata.get('manifest_version')!r} "
                f"does not match configured {config.dataset.manifest_version!r}"
            )
        if metadata.get("records") != len(records):
            problems.append(
                f"{split}: metadata claims {metadata.get('records')} records but the "
                f"CSV holds {len(records)}"
            )
    return manifests


def check_counts(
    config: Config, manifests: dict[str, tuple[ManifestRecord, ...]], problems: Problems
) -> None:
    """Compare record counts against the audited ``expected_counts``."""
    expected = config.section("expected_counts")
    if not expected:
        return
    for split, records in manifests.items():
        want = expected.get(split)
        if want is not None and len(records) != want:
            problems.append(
                f"{split}: {len(records)} records, Phase 1 measured {want}"
            )
    total_want = expected.get("total")
    if total_want is not None:
        total_got = sum(len(r) for r in manifests.values())
        if total_got != total_want:
            problems.append(
                f"total: {total_got} records, Phase 1 measured {total_want}"
            )


def check_labels(
    config: Config, manifests: dict[str, tuple[ManifestRecord, ...]], problems: Problems
) -> None:
    """Check label ranges, class coverage and the scope remapping itself."""
    scope = config.dataset.scope
    num_classes = scope.num_classes

    for split, records in manifests.items():
        seen: set[int] = set()
        for record in records:
            if not 0 <= record.project_label < num_classes:
                problems.append(
                    f"{split}: {record.filename} has project label "
                    f"{record.project_label}, outside 0..{num_classes - 1}"
                )
                continue
            # The remap is the single most likely source of a silent labelling
            # bug, so it is re-derived here rather than trusted.
            expected_original = scope.to_original_label(record.project_label)
            if record.ip102_label != expected_original:
                problems.append(
                    f"{split}: {record.filename} maps project label "
                    f"{record.project_label} to IP102 {record.ip102_label}, "
                    f"but the scope defines {expected_original}"
                )
            seen.add(record.project_label)
        missing = sorted(set(range(num_classes)) - seen)
        if missing:
            problems.append(f"{split}: no records for project labels {missing}")


def check_against_source(
    config: Config, manifests: dict[str, tuple[ManifestRecord, ...]], problems: Problems
) -> None:
    """Confirm the derived manifests agree with the read-only source files.

    Catches a derived manifest that has drifted from the source data, whether
    through a stale build or a hand edit.
    """
    scope = config.dataset.scope
    for split, path in source_manifest_paths(config).items():
        records = manifests.get(split)
        if records is None:
            continue
        source = read_source_manifest(path)
        in_scope = [
            (filename, label)
            for filename, label in source
            if scope.includes_original(label)
        ]
        if len(in_scope) != len(records):
            problems.append(
                f"{split}: source yields {len(in_scope)} in-scope records but the "
                f"derived manifest holds {len(records)}"
            )
            continue
        for (filename, label), record in zip(in_scope, records, strict=True):
            if record.filename != filename or record.ip102_label != label:
                problems.append(
                    f"{split}: derived record {record.filename}/{record.ip102_label} "
                    f"does not match source {filename}/{label}"
                )
                break


def check_class_mapping(config: Config, problems: Problems) -> None:
    """Confirm the on-disk class mapping still matches :mod:`farm_pest_ai.scopes`."""
    scope = config.dataset.scope
    path = config.paths.processed_dir / scope.name / "class_mapping.json"
    if not path.is_file():
        problems.append(f"class mapping not found: {path}")
        return
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("class_mapping_version") != CLASS_MAPPING_VERSION:
        problems.append(
            f"{path}: class_mapping_version "
            f"{document.get('class_mapping_version')!r} does not match "
            f"{CLASS_MAPPING_VERSION!r}"
        )
    if document.get("num_classes") != scope.num_classes:
        problems.append(
            f"{path}: num_classes {document.get('num_classes')} does not match the "
            f"{scope.num_classes} defined by scope {scope.name!r}"
        )
    on_disk = {
        int(entry["project_label"]): int(entry["ip102_label"])
        for entry in document.get("classes", [])
    }
    if on_disk != dict(scope.project_to_original):
        problems.append(
            f"{path}: the stored project->IP102 mapping no longer matches "
            f"farm_pest_ai.scopes; rebuild the manifests"
        )


def check_files_exist(
    config: Config, manifests: dict[str, tuple[ManifestRecord, ...]], problems: Problems
) -> None:
    """Confirm every referenced image is present, and no filename spans splits."""
    images_dir = config.paths.images_dir
    on_disk = {p.name for p in images_dir.iterdir() if p.is_file()}

    seen_in: dict[str, str] = {}
    for split, records in manifests.items():
        missing = [r.filename for r in records if r.filename not in on_disk]
        if missing:
            problems.append(
                f"{split}: {len(missing)} referenced image(s) missing from "
                f"{images_dir}, e.g. {missing[:5]}"
            )
        for record in records:
            previous = seen_in.get(record.filename)
            if previous is not None and previous != split:
                problems.append(
                    f"{record.filename} appears in both {previous} and {split}"
                )
            seen_in[record.filename] = split


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Verify derived manifests against the source data and audited counts.",
        default_configs=("base.yaml",),
    )
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Skip confirming that every referenced image exists on disk.",
    )
    parser.add_argument(
        "--skip-source-check",
        action="store_true",
        help="Skip re-reading the source manifests for a record-by-record comparison.",
    )
    return parser


def summarise(
    scope: ScopeSpec, manifests: dict[str, tuple[ManifestRecord, ...]]
) -> dict[str, Any]:
    """Build a small summary of what was verified."""
    return {
        "scope": scope.name,
        "num_classes": scope.num_classes,
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "records": {split: len(records) for split, records in manifests.items()},
        "total": sum(len(r) for r in manifests.values()),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    config, _ = bootstrap(args)
    logger = get_logger("verify_dataset")

    if args.print_config:
        print(config.to_yaml())
        return 0

    scope = config.dataset.scope
    problems: Problems = []

    manifests = check_manifests_present(config, problems)
    if manifests:
        check_counts(config, manifests, problems)
        check_labels(config, manifests, problems)
        check_class_mapping(config, problems)
        if not args.skip_source_check:
            check_against_source(config, manifests, problems)
        if not args.skip_file_check:
            check_files_exist(config, manifests, problems)

    summary = summarise(scope, manifests)
    print(f"\n=== Verify dataset: scope {scope.name} ===")
    print(f"  classes             {summary['num_classes']}")
    print(f"  class mapping ver.  {summary['class_mapping_version']}")
    for split in SPLITS:
        count = summary["records"].get(split)
        shown = count if count is not None else "MISSING"
        path = manifest_csv_path(config.paths.processed_dir, scope, split)
        print(f"  {split.ljust(12)}{shown:>8}  {path.name}")
    print(f"  {'total'.ljust(12)}{summary['total']:>8}")

    if problems:
        print(f"\nFAILED: {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        logger.error(
            "dataset verification failed",
            extra={"event": "verify_dataset", "problems": len(problems)},
        )
        return 1

    print("\nOK: derived manifests agree with the source data and audited counts.")
    logger.info(
        "dataset verification passed",
        extra={"event": "verify_dataset", **summary},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
