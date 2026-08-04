#!/usr/bin/env python3
"""Build scope-aware derived manifests from the read-only IP102 source files.

Reads ``classes.txt`` and the three official split manifests, applies the
``classes.txt`` off-by-one, filters and remaps labels for the active scope, and
writes ``data/processed/<scope>/{train,validation,test}.csv`` plus a metadata
sidecar and a class-mapping file.

The source tree is never modified. Writes are atomic, so re-running after an
interruption is safe, and the operation is idempotent: identical inputs produce
byte-identical CSVs.

Examples:
    python scripts/build_manifests.py
    python scripts/build_manifests.py --scope full102
    python scripts/build_manifests.py --config data_full102.yaml --check
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
    ClassInfo,
    DerivedManifest,
    ManifestError,
    atomic_write_text,
    build_derived_manifest,
    manifest_csv_path,
    read_classes,
    read_source_manifest,
    render_manifest_csv,
    write_derived_manifest,
)
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.scopes import CLASS_MAPPING_VERSION, ScopeSpec

#: Source manifest filenames used when configuration does not name them.
DEFAULT_SOURCE_MANIFESTS = {
    "train": "train.txt",
    "validation": "val.txt",
    "test": "test.txt",
}


def source_manifest_paths(config: Config) -> dict[str, Path]:
    """Resolve the three source manifest paths from configuration.

    ``dataset.source_manifests`` maps the project's canonical split names onto
    IP102's filenames, which differ (``val.txt`` rather than ``validation.txt``).
    """
    configured = config.get("dataset.source_manifests", {}) or {}
    root = config.paths.classification_root
    paths: dict[str, Path] = {}
    for split in SPLITS:
        name = configured.get(split, DEFAULT_SOURCE_MANIFESTS[split])
        paths[split] = root / str(name)
    return paths


def classes_file_path(config: Config) -> Path:
    """Resolve the ``classes.txt`` path from configuration."""
    name = config.get("dataset.classes_file", "classes.txt")
    return config.paths.classification_root / str(name)


def class_mapping_document(
    scope: ScopeSpec, classes: tuple[ClassInfo, ...]
) -> dict[str, Any]:
    """Build the scope's class-mapping record.

    Written next to the manifests so every later phase — training, evaluation,
    the API and the knowledge base — reads one authoritative file rather than
    re-deriving the mapping. The raw ``classes.txt`` name is preserved verbatim
    alongside a normalised ``canonical_name``; taxonomy is never corrected.
    """
    by_label = {info.ip102_label: info for info in classes}
    entries = []
    for project_label, ip102_label in scope.project_to_original.items():
        info = by_label[ip102_label]
        entries.append(
            {
                "project_label": project_label,
                "ip102_label": ip102_label,
                "classes_txt_id": info.classes_txt_id,
                "raw_name": info.raw_name,
                "canonical_name": info.canonical_name,
            }
        )
    return {
        "scope": scope.name,
        "description": scope.description,
        "num_classes": scope.num_classes,
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "identity_mapping": scope.is_identity,
        "classes": entries,
    }


def build_all(config: Config) -> dict[str, DerivedManifest]:
    """Build a derived manifest for every split under the active scope."""
    logger = get_logger("build_manifests")
    scope = config.dataset.scope

    classes = read_classes(classes_file_path(config), expected=102)
    logger.info(
        "parsed %d classes (classes.txt ids 1-%d -> IP102 labels 0-%d)",
        len(classes),
        len(classes),
        len(classes) - 1,
        extra={"event": "classes_parsed", "classes": len(classes)},
    )

    manifests: dict[str, DerivedManifest] = {}
    for split, path in source_manifest_paths(config).items():
        source = read_source_manifest(path)
        manifest = build_derived_manifest(
            split,
            source,
            scope,
            classes,
            manifest_version=config.dataset.manifest_version,
        )
        manifests[split] = manifest
        logger.info(
            "%s: %d source records -> %d in scope %s (%d excluded)",
            split,
            manifest.source_records,
            len(manifest),
            scope.name,
            manifest.excluded_records,
            extra={
                "event": "manifest_built",
                "split": split,
                "scope": scope.name,
                "source_records": manifest.source_records,
                "records": len(manifest),
            },
        )
    return manifests


def compare_with_expected(
    config: Config, manifests: dict[str, DerivedManifest]
) -> list[str]:
    """Compare record counts against the audited ``expected_counts`` section.

    Returns:
        A list of human-readable mismatches; empty when everything agrees or no
        expectations are configured.
    """
    expected = config.section("expected_counts")
    if not expected:
        return []
    problems: list[str] = []
    for split, manifest in manifests.items():
        want = expected.get(split)
        if want is not None and len(manifest) != want:
            problems.append(
                f"{split}: built {len(manifest)} records, config expects {want}"
            )
    total_want = expected.get("total")
    total_got = sum(len(m) for m in manifests.values())
    if total_want is not None and total_got != total_want:
        problems.append(f"total: built {total_got} records, config expects {total_want}")
    return problems


def check_on_disk(
    config: Config, manifests: dict[str, DerivedManifest]
) -> list[str]:
    """Compare freshly built manifests against the files already on disk.

    Bytes are compared rather than parsed records, so column-order or
    formatting drift is caught as well as changed content.

    Returns:
        A list of human-readable differences; empty when everything matches.
    """
    scope = config.dataset.scope
    processed_dir = config.paths.processed_dir
    differences: list[str] = []
    for split, manifest in manifests.items():
        path = manifest_csv_path(processed_dir, scope, split)
        if not path.is_file():
            differences.append(f"{split}: {path} does not exist")
            continue
        if path.read_bytes() != render_manifest_csv(manifest).encode("utf-8"):
            differences.append(f"{split}: {path} differs from a fresh build")
    return differences


def print_summary(
    scope: ScopeSpec, manifests: dict[str, DerivedManifest], written: list[Path]
) -> None:
    """Print the per-split record table and the list of files written."""
    print(f"\nScope {scope.name}: {scope.num_classes} classes")
    header = "SPLIT".ljust(12) + "RECORDS".rjust(9) + "SOURCE".rjust(9)
    print(header + "EXCLUDED".rjust(10))
    print("-" * 40)
    for split in SPLITS:
        manifest = manifests[split]
        print(
            f"{split.ljust(12)}{len(manifest):>9}{manifest.source_records:>9}"
            f"{manifest.excluded_records:>10}"
        )
    print("-" * 40)
    print(f"{'total'.ljust(12)}{sum(len(m) for m in manifests.values()):>9}")
    print("\nWritten:")
    for path in written:
        print(f"  {path}")
        if path.suffix == ".csv":
            print(f"  {path.with_suffix('.metadata.json')}")


def write_all(config: Config, manifests: dict[str, DerivedManifest]) -> list[Path]:
    """Write every derived manifest plus the scope's class-mapping document."""
    scope = config.dataset.scope
    processed_dir = config.paths.processed_dir

    written = [
        write_derived_manifest(manifests[split], processed_dir) for split in SPLITS
    ]

    mapping_path = processed_dir / scope.name / "class_mapping.json"
    classes = read_classes(classes_file_path(config), expected=102)
    atomic_write_text(
        mapping_path,
        json.dumps(class_mapping_document(scope, classes), indent=2) + "\n",
    )
    written.append(mapping_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Build derived, scope-aware manifests from the IP102 source files.",
        default_configs=("base.yaml",),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Rebuild in memory and compare against the manifests already on "
            "disk without writing. Exits non-zero if they differ."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite the manifests even when the existing files already match.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    config, _ = bootstrap(args)
    logger = get_logger("build_manifests")

    if args.print_config:
        print(config.to_yaml())
        return 0

    scope = config.dataset.scope

    try:
        manifests = build_all(config)
    except ManifestError:
        logger.exception("failed to build manifests", extra={"event": "manifest_error"})
        return 1

    problems = compare_with_expected(config, manifests)
    for problem in problems:
        logger.error("count mismatch: %s", problem, extra={"event": "count_mismatch"})

    if args.check:
        differences = check_on_disk(config, manifests)
        for difference in differences:
            logger.error("%s", difference, extra={"event": "manifest_drift"})
        if differences or problems:
            print(f"\nFAILED: {len(differences) + len(problems)} problem(s) found.")
            return 1
        print(f"\nOK: derived manifests for scope {scope.name} are up to date.")
        return 0

    if problems:
        return 1

    written = write_all(config, manifests)
    print_summary(scope, manifests, written)

    logger.info(
        "derived manifests written for scope %s",
        scope.name,
        extra={"event": "manifests_written", "scope": scope.name, "files": len(written)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
