"""Reading IP102 source manifests and building derived, scope-aware manifests.

This module owns the two places where IP102's own conventions are translated
into the project's:

``classes.txt`` numbering
    The file numbers classes 1-102 while the split manifests use labels 0-101.
    The relationship ``ip102_label = classes_txt_id - 1`` is applied exactly
    once, in :func:`read_classes`, and nowhere else in the project.

Scope remapping
    Project labels are derived from :mod:`farm_pest_ai.scopes`. A derived
    manifest for ``rice10`` carries both the original IP102 label and the
    project label, so nothing downstream has to re-derive the mapping.

The source tree is read-only. Everything written by this module goes to
``data/processed/`` via atomic replace, so an interrupted run never leaves a
half-written manifest that a later phase would silently consume.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..scopes import CLASS_MAPPING_VERSION, ScopeSpec, resolve_scope

__all__ = [
    "SPLITS",
    "ClassInfo",
    "DerivedManifest",
    "ManifestError",
    "ManifestRecord",
    "atomic_write_bytes",
    "atomic_write_text",
    "build_derived_manifest",
    "class_distribution",
    "manifest_csv_path",
    "read_classes",
    "read_derived_manifest",
    "read_source_manifest",
    "render_manifest_csv",
    "write_derived_manifest",
]

#: Canonical split names used throughout the project. The source filenames
#: differ (``val.txt``), which is why the mapping lives in configuration.
SPLITS: tuple[str, ...] = ("train", "validation", "test")

#: Column order of a derived manifest CSV. Stable: later phases parse by name,
#: but a fixed order keeps diffs readable and file hashes reproducible.
DERIVED_COLUMNS: tuple[str, ...] = (
    "filename",
    "relative_path",
    "ip102_label",
    "project_label",
    "class_name",
    "split",
)


class ManifestError(ValueError):
    """Raised when a manifest or class file is missing, malformed or unusable."""


@dataclass(frozen=True)
class ClassInfo:
    """One entry of ``classes.txt``, after the off-by-one is resolved.

    Attributes:
        classes_txt_id: The 1-based identifier printed in ``classes.txt``.
        ip102_label: The 0-based label used in the split manifests.
        raw_name: The name exactly as it appears in the file, minus surrounding
            whitespace. Preserved verbatim; taxonomy is never corrected.
        canonical_name: A normalised form for display and knowledge lookup:
            lower-cased with internal whitespace collapsed.
    """

    classes_txt_id: int
    ip102_label: int
    raw_name: str
    canonical_name: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "classes_txt_id": self.classes_txt_id,
            "ip102_label": self.ip102_label,
            "raw_name": self.raw_name,
            "canonical_name": self.canonical_name,
        }


@dataclass(frozen=True)
class ManifestRecord:
    """One image in a derived manifest.

    Attributes:
        filename: Bare filename as written in the source manifest.
        ip102_label: Original IP102 label, 0-101.
        project_label: Label in the active scope's numbering.
        class_name: Raw class name from ``classes.txt``.
        split: One of :data:`SPLITS`.
    """

    filename: str
    ip102_label: int
    project_label: int
    class_name: str
    split: str

    @property
    def relative_path(self) -> str:
        """Path relative to ``classification_root``, using forward slashes.

        Stored in the CSV so a manifest remains meaningful on Windows and inside
        a Linux container without re-deriving the layout.
        """
        return f"images/{self.filename}"

    def to_row(self) -> dict[str, Any]:
        """Return this record as a CSV row keyed by :data:`DERIVED_COLUMNS`."""
        return {
            "filename": self.filename,
            "relative_path": self.relative_path,
            "ip102_label": self.ip102_label,
            "project_label": self.project_label,
            "class_name": self.class_name,
            "split": self.split,
        }


@dataclass(frozen=True)
class DerivedManifest:
    """A scope-filtered, remapped manifest for one split.

    Attributes:
        scope: The scope this manifest was built for.
        split: One of :data:`SPLITS`.
        records: The retained records, in source-manifest order.
        source_records: How many records the source split contained before
            scope filtering. Equal to ``len(records)`` for ``full102``.
        manifest_version: Version stamp copied from configuration.
    """

    scope: ScopeSpec
    split: str
    records: tuple[ManifestRecord, ...]
    source_records: int
    manifest_version: str = "1.0.0"

    def __len__(self) -> int:
        """Number of records retained after scope filtering."""
        return len(self.records)

    def __iter__(self) -> Iterator[ManifestRecord]:
        """Iterate the retained records in source-manifest order."""
        return iter(self.records)

    @property
    def excluded_records(self) -> int:
        """Records dropped because their class is outside the scope."""
        return self.source_records - len(self.records)

    def class_counts(self) -> dict[int, int]:
        """Count records per project label, including classes with zero."""
        counts = dict.fromkeys(range(self.scope.num_classes), 0)
        for record in self.records:
            counts[record.project_label] += 1
        return counts

    def metadata(self) -> dict[str, Any]:
        """Provenance recorded beside the CSV.

        Every consumer checks ``scope`` and ``class_mapping_version`` before
        trusting the labels, so a manifest built under an older mapping is
        rejected rather than silently misread.
        """
        counts = self.class_counts()
        return {
            "scope": self.scope.name,
            "split": self.split,
            "num_classes": self.scope.num_classes,
            "class_mapping_version": CLASS_MAPPING_VERSION,
            "manifest_version": self.manifest_version,
            "records": len(self.records),
            "source_records": self.source_records,
            "excluded_records": self.excluded_records,
            "project_to_ip102": {
                str(p): o for p, o in self.scope.project_to_original.items()
            },
            "class_counts": {str(k): v for k, v in counts.items()},
            "classes_with_no_records": [k for k, v in counts.items() if v == 0],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


# -- atomic writes ------------------------------------------------------


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    """Write ``payload`` to ``path`` atomically.

    The data lands in a temporary file in the same directory, is flushed to
    disk, then replaced into place. A crash mid-write leaves either the old file
    or nothing, never a truncated manifest.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write ``text`` atomically with explicit LF line endings.

    Newlines are not translated, so a manifest generated on Windows is
    byte-identical to one generated in a container.
    """
    return atomic_write_bytes(path, text.encode(encoding))


# -- reading source files -----------------------------------------------


def _parse_class_line(path: Path, number: int, raw_line: str) -> ClassInfo:
    """Parse one ``classes.txt`` line into a :class:`ClassInfo`.

    Splits on the first space, falling back to a tab, so class names containing
    spaces survive intact.
    """
    line = raw_line.strip()
    head, sep, tail = line.partition(" ")
    if not sep:
        head, sep, tail = line.partition("\t")
    if not sep:
        raise ManifestError(f"{path}:{number}: expected '<id> <name>', got {raw_line!r}")
    try:
        classes_txt_id = int(head)
    except ValueError:
        raise ManifestError(
            f"{path}:{number}: class id must be an integer, got {head!r}"
        ) from None
    name = tail.strip()
    if not name:
        raise ManifestError(f"{path}:{number}: class name is empty")
    return ClassInfo(
        classes_txt_id=classes_txt_id,
        # The single point where the classes.txt off-by-one is applied.
        ip102_label=classes_txt_id - 1,
        raw_name=name,
        canonical_name=" ".join(name.split()).lower(),
    )


def read_classes(path: Path, *, expected: int | None = None) -> tuple[ClassInfo, ...]:
    """Parse ``classes.txt`` and resolve its 1-based numbering.

    Each line is ``<id> <name>``, where ``id`` runs from 1. Names may contain
    spaces and, in the shipped file, trailing tabs; the file uses CRLF endings
    while the split manifests use LF. All of that is normalised here.

    Args:
        path: Path to ``classes.txt``.
        expected: Optional exact number of classes to require.

    Returns:
        Entries ordered by ``ip102_label``.

    Raises:
        ManifestError: If the file is missing, a line is malformed, ids are not
            exactly ``1..n``, or ``expected`` is not met.
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"classes file not found: {path}")

    entries: list[ClassInfo] = []
    seen: dict[int, int] = {}
    # ``splitlines`` handles the shipped file's CRLF endings, which differ from
    # the LF used by the split manifests. ``utf-8-sig`` drops any BOM.
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    for number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        info = _parse_class_line(path, number, raw_line)
        if info.classes_txt_id in seen:
            raise ManifestError(
                f"{path}:{number}: duplicate class id {info.classes_txt_id} "
                f"(first seen on line {seen[info.classes_txt_id]})"
            )
        seen[info.classes_txt_id] = number
        entries.append(info)

    if not entries:
        raise ManifestError(f"{path} contains no class definitions")

    entries.sort(key=lambda info: info.ip102_label)
    ids = [info.classes_txt_id for info in entries]
    if ids != list(range(1, len(ids) + 1)):
        raise ManifestError(
            f"{path}: class ids must be exactly 1..{len(ids)}; "
            f"got {ids[:5]}...{ids[-3:]}"
        )
    if expected is not None and len(entries) != expected:
        raise ManifestError(
            f"{path}: expected {expected} classes, found {len(entries)}"
        )
    return tuple(entries)


def _parse_manifest_line(path: Path, number: int, raw_line: str) -> tuple[str, int]:
    """Parse one ``<filename> <label>`` manifest line."""
    parts = raw_line.split()
    if len(parts) != 2:
        raise ManifestError(
            f"{path}:{number}: expected '<filename> <label>', got {raw_line!r}"
        )
    filename, raw_label = parts
    try:
        label = int(raw_label)
    except ValueError:
        raise ManifestError(
            f"{path}:{number}: label must be an integer, got {raw_label!r}"
        ) from None
    if label < 0:
        raise ManifestError(f"{path}:{number}: label must be >= 0, got {label}")
    if "/" in filename or "\\" in filename:
        raise ManifestError(f"{path}:{number}: filename must be bare, got {filename!r}")
    return filename, label


def read_source_manifest(path: Path) -> tuple[tuple[str, int], ...]:
    """Parse an IP102 split manifest into ``(filename, ip102_label)`` pairs.

    Order is preserved exactly as in the source file: the official split
    assignment is never reshuffled here.

    Raises:
        ManifestError: If the file is missing, a line is malformed, a label is
            not an integer, or a filename repeats within the split.
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest file not found: {path}")

    records: list[tuple[str, int]] = []
    first_seen: dict[str, int] = {}
    text = path.read_text(encoding="utf-8", errors="strict")
    for number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        filename, label = _parse_manifest_line(path, number, raw_line)
        if filename in first_seen:
            raise ManifestError(
                f"{path}:{number}: duplicate filename {filename!r} "
                f"(first seen on line {first_seen[filename]})"
            )
        first_seen[filename] = number
        records.append((filename, label))

    if not records:
        raise ManifestError(f"{path} contains no records")
    return tuple(records)


# -- deriving -----------------------------------------------------------


def build_derived_manifest(
    split: str,
    source: Iterable[tuple[str, int]],
    scope: str | ScopeSpec,
    classes: Sequence[ClassInfo],
    *,
    manifest_version: str = "1.0.0",
) -> DerivedManifest:
    """Filter a source split to a scope and remap its labels.

    For ``full102`` every record is retained and ``project_label`` equals
    ``ip102_label``. For ``rice10`` only the ten mapped classes survive, and
    their labels are renumbered 0-9 in the order fixed by
    :mod:`farm_pest_ai.scopes`.

    Args:
        split: One of :data:`SPLITS`.
        source: ``(filename, ip102_label)`` pairs from :func:`read_source_manifest`.
        scope: Scope name or spec.
        classes: Parsed ``classes.txt`` entries, used for the name column.
        manifest_version: Version stamp recorded in the metadata.

    Returns:
        The derived manifest, preserving source order.

    Raises:
        ManifestError: If ``split`` is unknown or a record carries a label with
            no matching ``classes.txt`` entry.
    """
    if split not in SPLITS:
        raise ManifestError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
    spec = resolve_scope(scope)
    by_label = {info.ip102_label: info for info in classes}

    records: list[ManifestRecord] = []
    total = 0
    for filename, ip102_label in source:
        total += 1
        info = by_label.get(ip102_label)
        if info is None:
            raise ManifestError(
                f"{split}: image {filename!r} has IP102 label {ip102_label}, which "
                f"has no entry in classes.txt (known labels 0..{len(by_label) - 1})"
            )
        if not spec.includes_original(ip102_label):
            continue
        records.append(
            ManifestRecord(
                filename=filename,
                ip102_label=ip102_label,
                project_label=spec.to_project_label(ip102_label),
                class_name=info.raw_name,
                split=split,
            )
        )

    return DerivedManifest(
        scope=spec,
        split=split,
        records=tuple(records),
        source_records=total,
        manifest_version=manifest_version,
    )


def class_distribution(manifest: DerivedManifest) -> Counter[int]:
    """Return a counter of project labels present in ``manifest``."""
    return Counter(record.project_label for record in manifest.records)


# -- writing and reading derived manifests -------------------------------


def manifest_csv_path(processed_dir: Path, scope: str | ScopeSpec, split: str) -> Path:
    """Return the canonical CSV location for a derived manifest.

    The scope is part of the directory name so ``rice10`` and ``full102``
    artifacts can never overwrite one another.
    """
    spec = resolve_scope(scope)
    if split not in SPLITS:
        raise ManifestError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
    return Path(processed_dir) / spec.name / f"{split}.csv"


def render_manifest_csv(manifest: DerivedManifest) -> str:
    """Render a derived manifest as CSV text.

    Uses an explicit LF terminator so the output is byte-identical on Windows
    and inside a Linux container. This is the single renderer: writing and the
    ``--check`` comparison both go through it, so they can never disagree.
    """
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(DERIVED_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    for record in manifest.records:
        writer.writerow(record.to_row())
    return stream.getvalue()


def write_derived_manifest(manifest: DerivedManifest, processed_dir: Path) -> Path:
    """Write a derived manifest and its metadata sidecar atomically.

    Produces ``<processed_dir>/<scope>/<split>.csv`` plus a matching
    ``<split>.metadata.json``.

    Returns:
        The CSV path.
    """
    csv_path = manifest_csv_path(processed_dir, manifest.scope, manifest.split)
    atomic_write_text(csv_path, render_manifest_csv(manifest))
    atomic_write_text(
        csv_path.with_suffix(".metadata.json"),
        json.dumps(manifest.metadata(), indent=2, sort_keys=True) + "\n",
    )
    return csv_path


def read_derived_manifest(
    processed_dir: Path, scope: str | ScopeSpec, split: str
) -> tuple[tuple[ManifestRecord, ...], dict[str, Any]]:
    """Read back a derived manifest and its metadata.

    The metadata's scope and class-mapping version are checked against the
    request, so a manifest generated under a different mapping is rejected
    rather than silently misinterpreted.

    Returns:
        The records and the metadata mapping.

    Raises:
        ManifestError: If either file is missing, the CSV columns differ, or the
            recorded scope or class-mapping version does not match.
    """
    spec = resolve_scope(scope)
    csv_path = manifest_csv_path(processed_dir, spec, split)
    meta_path = csv_path.with_suffix(".metadata.json")
    if not csv_path.is_file():
        raise ManifestError(
            f"derived manifest not found: {csv_path}; run scripts/build_manifests.py"
        )
    if not meta_path.is_file():
        raise ManifestError(f"manifest metadata not found: {meta_path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("scope") != spec.name:
        raise ManifestError(
            f"{meta_path}: manifest was built for scope {metadata.get('scope')!r}, "
            f"not {spec.name!r}"
        )
    if metadata.get("class_mapping_version") != CLASS_MAPPING_VERSION:
        raise ManifestError(
            f"{meta_path}: manifest uses class mapping version "
            f"{metadata.get('class_mapping_version')!r}, but this build expects "
            f"{CLASS_MAPPING_VERSION!r}; rebuild the manifests"
        )

    records: list[ManifestRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(DERIVED_COLUMNS):
            raise ManifestError(
                f"{csv_path}: expected columns {list(DERIVED_COLUMNS)}, "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            records.append(
                ManifestRecord(
                    filename=row["filename"],
                    ip102_label=int(row["ip102_label"]),
                    project_label=int(row["project_label"]),
                    class_name=row["class_name"],
                    split=row["split"],
                )
            )
    return tuple(records), metadata
