"""Dataset auditing: integrity, duplicates, leakage and image properties.

Phase 1 established the split counts and filename-level integrity of IP102.
This module implements the checks that were deferred to Phase 4, all of which
depend on reading image *content* rather than manifest text:

* exact-content duplicate detection by SHA-256,
* exact-content cross-split leakage, which filename checks cannot rule out,
* a full decode of every image, and
* source dimensions, recorded so the sub-160px cohort can be analysed later.

Everything here is read-only with respect to ``ip102_v1.1``. Results are
returned as dataclasses and written to ``data/reports/`` by the audit script.

Pillow is imported lazily so that manifest building, configuration and the test
suite keep working in an environment without it.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .manifests import ManifestRecord

__all__ = [
    "DecodeResult",
    "DuplicateGroup",
    "ImageProbe",
    "IntegrityReport",
    "LeakageReport",
    "SplitDistribution",
    "check_integrity",
    "find_duplicates",
    "find_leakage",
    "hash_file",
    "probe_image",
    "probe_many",
    "split_distribution",
    "summarise_dimensions",
]

#: Read images in chunks rather than whole, so a multi-megabyte JPEG never sits
#: in memory purely to be hashed.
_HASH_CHUNK_BYTES = 1 << 20

#: The model input's short side. Images below it must be upscaled.
_MODEL_SHORT_SIDE = 160


# -- hashing and probing ------------------------------------------------


def hash_file(path: Path, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """Return the SHA-256 hex digest of a file's bytes.

    Byte-level hashing detects only exact duplicates: two visually identical
    images saved at different JPEG qualities hash differently. That is the
    intended scope — an exact duplicate across splits is unambiguous leakage,
    whereas near-duplicate detection needs perceptual hashing and is not
    claimed here.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ImageProbe:
    """Measured properties of one image file.

    Attributes:
        filename: Bare filename, matching the manifest.
        ok: Whether the image decoded without error.
        width: Pixel width, or ``None`` when the probe failed.
        height: Pixel height, or ``None`` when the probe failed.
        image_format: Format reported by Pillow, e.g. ``"JPEG"``.
        mode: Colour mode reported by Pillow, e.g. ``"RGB"``.
        size_bytes: File size on disk.
        sha256: Content hash, present only when hashing was requested.
        error: Failure description when ``ok`` is false.
    """

    filename: str
    ok: bool
    width: int | None = None
    height: int | None = None
    image_format: str | None = None
    mode: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    error: str | None = None

    @property
    def short_side(self) -> int | None:
        """The smaller of width and height, or ``None`` if unknown."""
        if self.width is None or self.height is None:
            return None
        return min(self.width, self.height)

    @property
    def needs_upscale(self) -> bool:
        """Whether this image must be enlarged to reach the model input."""
        side = self.short_side
        return side is not None and side < _MODEL_SHORT_SIDE

    @property
    def aspect_ratio(self) -> float | None:
        """Width divided by height, or ``None`` if unknown."""
        if not self.width or not self.height:
            return None
        return self.width / self.height

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "filename": self.filename,
            "ok": self.ok,
            "width": self.width,
            "height": self.height,
            "format": self.image_format,
            "mode": self.mode,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "error": self.error,
        }


def probe_image(
    path: Path, *, full_decode: bool = True, compute_hash: bool = False
) -> ImageProbe:
    """Measure one image, optionally decoding it fully and hashing it.

    Args:
        path: Image file to inspect.
        full_decode: When true, decode the pixel data rather than reading only
            the header. Header-only reads are far faster but miss truncated
            files, which is precisely what a decode audit is looking for.
        compute_hash: Whether to also compute the SHA-256 content hash.

    Returns:
        An :class:`ImageProbe`. Failures are reported in the result rather than
        raised, so one corrupt file does not abort an audit of 75,222 images.
    """
    from PIL import Image

    path = Path(path)
    filename = path.name
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return ImageProbe(filename=filename, ok=False, error=f"stat failed: {exc}")

    digest = hash_file(path) if compute_hash else None

    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            mode = image.mode
            if full_decode:
                # load() forces the pixel data through the decoder, which is
                # what surfaces truncated JPEGs; open() alone reads the header.
                image.load()
    # UnidentifiedImageError subclasses OSError, so it is covered here.
    except (OSError, ValueError) as exc:
        return ImageProbe(
            filename=filename,
            ok=False,
            size_bytes=size_bytes,
            sha256=digest,
            error=f"{type(exc).__name__}: {exc}",
        )

    return ImageProbe(
        filename=filename,
        ok=True,
        width=width,
        height=height,
        image_format=image_format,
        mode=mode,
        size_bytes=size_bytes,
        sha256=digest,
    )


# -- integrity ----------------------------------------------------------


@dataclass(frozen=True)
class IntegrityReport:
    """Filename-level integrity of the manifests against the image directory.

    Attributes:
        total_records: Records across all audited splits.
        images_on_disk: Files present in the images directory.
        missing_from_disk: Referenced filenames with no file.
        unreferenced_on_disk: Files no split refers to.
        cross_split_filenames: Filenames appearing in more than one split.
        conflicting_labels: Filenames carrying different labels in different
            splits, mapped to the labels observed.
    """

    total_records: int
    images_on_disk: int
    missing_from_disk: tuple[str, ...] = ()
    unreferenced_on_disk: tuple[str, ...] = ()
    cross_split_filenames: tuple[str, ...] = ()
    conflicting_labels: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether every integrity check passed."""
        return not (
            self.missing_from_disk
            or self.unreferenced_on_disk
            or self.cross_split_filenames
            or self.conflicting_labels
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary with bounded example lists."""
        return {
            "ok": self.ok,
            "total_records": self.total_records,
            "images_on_disk": self.images_on_disk,
            "missing_from_disk": len(self.missing_from_disk),
            "unreferenced_on_disk": len(self.unreferenced_on_disk),
            "cross_split_filenames": len(self.cross_split_filenames),
            "conflicting_labels": len(self.conflicting_labels),
            "examples": {
                "missing_from_disk": list(self.missing_from_disk[:10]),
                "unreferenced_on_disk": list(self.unreferenced_on_disk[:10]),
                "cross_split_filenames": list(self.cross_split_filenames[:10]),
            },
        }


def check_integrity(
    splits: Mapping[str, Sequence[tuple[str, int]]],
    images_dir: Path,
    *,
    check_unreferenced: bool = True,
) -> IntegrityReport:
    """Check manifest records against the files in ``images_dir``.

    Args:
        splits: Split name to ``(filename, label)`` pairs. Pass the *source*
            manifests: integrity is a property of the dataset, not of a scope.
        images_dir: Directory holding the images.
        check_unreferenced: Whether to also list files no split refers to. Skip
            it when auditing a scope subset, where most files are legitimately
            unreferenced.

    Returns:
        An :class:`IntegrityReport`.
    """
    images_dir = Path(images_dir)
    on_disk = {p.name for p in images_dir.iterdir() if p.is_file()}

    labels_by_file: defaultdict[str, set[int]] = defaultdict(set)
    splits_by_file: defaultdict[str, set[str]] = defaultdict(set)
    total = 0
    for split, records in splits.items():
        for filename, label in records:
            total += 1
            labels_by_file[filename].add(label)
            splits_by_file[filename].add(split)

    referenced = set(labels_by_file)
    missing = sorted(referenced - on_disk)
    unreferenced = sorted(on_disk - referenced) if check_unreferenced else []
    cross_split = sorted(f for f, s in splits_by_file.items() if len(s) > 1)
    conflicting = {
        f: tuple(sorted(labels))
        for f, labels in labels_by_file.items()
        if len(labels) > 1
    }

    return IntegrityReport(
        total_records=total,
        images_on_disk=len(on_disk),
        missing_from_disk=tuple(missing),
        unreferenced_on_disk=tuple(unreferenced),
        cross_split_filenames=tuple(cross_split),
        conflicting_labels=dict(sorted(conflicting.items())),
    )


# -- duplicates and leakage ---------------------------------------------


@dataclass(frozen=True)
class DuplicateGroup:
    """A set of files sharing one content hash.

    Attributes:
        sha256: The shared content hash.
        members: ``(split, filename, project_label)`` for each copy.
    """

    sha256: str
    members: tuple[tuple[str, str, int], ...]

    @property
    def splits(self) -> tuple[str, ...]:
        """Distinct splits the copies fall into, sorted."""
        return tuple(sorted({split for split, _, _ in self.members}))

    @property
    def labels(self) -> tuple[int, ...]:
        """Distinct project labels the copies carry, sorted."""
        return tuple(sorted({label for _, _, label in self.members}))

    @property
    def crosses_splits(self) -> bool:
        """Whether copies exist in more than one split. This is leakage."""
        return len(self.splits) > 1

    @property
    def label_conflict(self) -> bool:
        """Whether identical bytes carry more than one label.

        A genuine annotation contradiction: the same image cannot be two
        different classes, so at least one of the labels is wrong.
        """
        return len(self.labels) > 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "sha256": self.sha256,
            "count": len(self.members),
            "splits": list(self.splits),
            "labels": list(self.labels),
            "crosses_splits": self.crosses_splits,
            "label_conflict": self.label_conflict,
            "members": [
                {"split": s, "filename": f, "project_label": lbl}
                for s, f, lbl in self.members
            ],
        }


def find_duplicates(
    hashes: Mapping[str, Mapping[str, str]],
    labels: Mapping[str, Mapping[str, int]],
) -> tuple[DuplicateGroup, ...]:
    """Group files that share a content hash.

    Args:
        hashes: Split name -> filename -> SHA-256 hex digest.
        labels: Split name -> filename -> project label.

    Returns:
        Groups of size two or more, ordered by descending group size then hash,
        so the worst offenders appear first in a report.
    """
    by_hash: defaultdict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for split, per_file in hashes.items():
        split_labels = labels.get(split, {})
        for filename, digest in per_file.items():
            by_hash[digest].append((split, filename, split_labels.get(filename, -1)))

    groups = [
        DuplicateGroup(sha256=digest, members=tuple(sorted(members)))
        for digest, members in by_hash.items()
        if len(members) > 1
    ]
    groups.sort(key=lambda g: (-len(g.members), g.sha256))
    return tuple(groups)


@dataclass(frozen=True)
class LeakageReport:
    """Exact-content overlap between splits.

    Attributes:
        duplicate_groups: Every group of identical files.
        scope: Scope the audit was run for.
    """

    duplicate_groups: tuple[DuplicateGroup, ...]
    scope: str

    @property
    def within_split_groups(self) -> tuple[DuplicateGroup, ...]:
        """Duplicate groups confined to a single split."""
        return tuple(g for g in self.duplicate_groups if not g.crosses_splits)

    @property
    def cross_split_groups(self) -> tuple[DuplicateGroup, ...]:
        """Duplicate groups spanning splits. These inflate headline metrics."""
        return tuple(g for g in self.duplicate_groups if g.crosses_splits)

    @property
    def label_conflict_groups(self) -> tuple[DuplicateGroup, ...]:
        """Duplicate groups whose copies disagree about the label."""
        return tuple(g for g in self.duplicate_groups if g.label_conflict)

    def leaked_files(self, split: str) -> tuple[str, ...]:
        """Filenames in ``split`` that also appear, byte-identical, elsewhere.

        These are the evaluation images whose content the model may have seen
        during training. Phase 9 reports metrics with and without them.
        """
        leaked = {
            filename
            for group in self.cross_split_groups
            for member_split, filename, _ in group.members
            if member_split == split
        }
        return tuple(sorted(leaked))

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        cross = self.cross_split_groups
        pairs = Counter(
            tuple(group.splits) for group in cross
        )
        return {
            "scope": self.scope,
            "duplicate_groups": len(self.duplicate_groups),
            "within_split_groups": len(self.within_split_groups),
            "cross_split_groups": len(cross),
            "label_conflict_groups": len(self.label_conflict_groups),
            "duplicate_files": sum(
                len(g.members) for g in self.duplicate_groups
            ),
            "cross_split_pairs": {
                " + ".join(splits): count for splits, count in sorted(pairs.items())
            },
            "leaked_files_per_split": {
                split: len(self.leaked_files(split))
                for split in ("train", "validation", "test")
            },
        }


def find_leakage(
    hashes: Mapping[str, Mapping[str, str]],
    labels: Mapping[str, Mapping[str, int]],
    scope: str,
) -> LeakageReport:
    """Build a :class:`LeakageReport` from per-split content hashes."""
    return LeakageReport(
        duplicate_groups=find_duplicates(hashes, labels), scope=scope
    )


# -- distributions and dimensions ---------------------------------------


@dataclass(frozen=True)
class SplitDistribution:
    """Per-class counts for one split.

    Attributes:
        split: Split name.
        counts: Project label -> record count, including empty classes.
        class_names: Project label -> raw class name.
    """

    split: str
    counts: Mapping[int, int]
    class_names: Mapping[int, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Total records in the split."""
        return sum(self.counts.values())

    @property
    def present_classes(self) -> int:
        """Classes with at least one record."""
        return sum(1 for v in self.counts.values() if v > 0)

    @property
    def empty_classes(self) -> tuple[int, ...]:
        """Project labels with no records at all."""
        return tuple(sorted(k for k, v in self.counts.items() if v == 0))

    @property
    def imbalance_ratio(self) -> float | None:
        """Largest class divided by smallest non-empty class.

        ``None`` when the split is empty. Reported because it drives the choice
        of macro F1 over accuracy as the selection metric.
        """
        values = [v for v in self.counts.values() if v > 0]
        if not values:
            return None
        return max(values) / min(values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        values = [v for v in self.counts.values() if v > 0]
        min_label = min(self.counts, key=lambda k: self.counts[k]) if self.counts else None
        max_label = max(self.counts, key=lambda k: self.counts[k]) if self.counts else None
        return {
            "split": self.split,
            "total": self.total,
            "classes": len(self.counts),
            "present_classes": self.present_classes,
            "empty_classes": list(self.empty_classes),
            "min": min(values) if values else 0,
            "min_label": min_label,
            "max": max(values) if values else 0,
            "max_label": max_label,
            "median": median(values) if values else 0,
            "mean": round(sum(values) / len(values), 1) if values else 0.0,
            "imbalance_ratio": (
                round(self.imbalance_ratio, 1)
                if self.imbalance_ratio is not None
                else None
            ),
            "counts": {str(k): v for k, v in sorted(self.counts.items())},
            "class_names": {str(k): v for k, v in sorted(self.class_names.items())},
        }


def split_distribution(
    split: str,
    records: Iterable[ManifestRecord],
    num_classes: int,
) -> SplitDistribution:
    """Compute per-class counts for one split's derived records."""
    counts = dict.fromkeys(range(num_classes), 0)
    names: dict[int, str] = {}
    for record in records:
        counts[record.project_label] = counts.get(record.project_label, 0) + 1
        names.setdefault(record.project_label, record.class_name)
    return SplitDistribution(split=split, counts=counts, class_names=names)


@dataclass(frozen=True)
class DecodeResult:
    """Outcome of decoding and measuring a collection of images.

    Attributes:
        probes: One :class:`ImageProbe` per file inspected.
        split: Split the probes belong to, or ``"all"``.
    """

    probes: tuple[ImageProbe, ...]
    split: str = "all"

    @property
    def failures(self) -> tuple[ImageProbe, ...]:
        """Probes that could not be decoded."""
        return tuple(p for p in self.probes if not p.ok)

    @property
    def ok_probes(self) -> tuple[ImageProbe, ...]:
        """Probes that decoded successfully."""
        return tuple(p for p in self.probes if p.ok)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        good = self.ok_probes
        formats = Counter(p.image_format for p in good)
        modes = Counter(p.mode for p in good)
        return {
            "split": self.split,
            "inspected": len(self.probes),
            "decoded": len(good),
            "failed": len(self.failures),
            "formats": {str(k): v for k, v in formats.most_common()},
            "modes": {str(k): v for k, v in modes.most_common()},
            "failures": [p.to_dict() for p in self.failures[:50]],
            "dimensions": summarise_dimensions(good),
        }


def _percentile(sorted_values: Sequence[int], fraction: float) -> int:
    """Return a nearest-rank percentile from an already-sorted sequence."""
    if not sorted_values:
        return 0
    index = max(0, min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def summarise_dimensions(probes: Sequence[ImageProbe]) -> dict[str, Any]:
    """Summarise widths, heights, aspect ratios and the upscale cohort.

    The sub-160px share is the figure Phase 5 needs when choosing an
    interpolation policy, and Phase 9 revisits it to check whether errors
    concentrate in images that had to be enlarged.
    """
    good = [p for p in probes if p.ok and p.width and p.height]
    if not good:
        return {"count": 0}

    widths = sorted(p.width for p in good if p.width is not None)
    heights = sorted(p.height for p in good if p.height is not None)
    shorts = sorted(s for p in good if (s := p.short_side) is not None)
    aspects = sorted(a for p in good if (a := p.aspect_ratio) is not None)
    below_160 = sum(1 for s in shorts if s < _MODEL_SHORT_SIDE)
    below_224 = sum(1 for s in shorts if s < 224)

    def spread(values: Sequence[int]) -> dict[str, int]:
        return {
            "min": values[0],
            "p05": _percentile(values, 0.05),
            "median": values[len(values) // 2],
            "p95": _percentile(values, 0.95),
            "max": values[-1],
        }

    return {
        "count": len(good),
        "width": spread(widths),
        "height": spread(heights),
        "short_side": spread(shorts),
        "aspect_ratio": {
            "min": round(aspects[0], 2),
            "median": round(aspects[len(aspects) // 2], 2),
            "max": round(aspects[-1], 2),
        },
        "short_side_below_160": below_160,
        "short_side_below_160_pct": round(100 * below_160 / len(good), 1),
        "short_side_below_224": below_224,
        "short_side_below_224_pct": round(100 * below_224 / len(good), 1),
    }


def probe_many(
    records: Sequence[ManifestRecord],
    images_dir: Path,
    *,
    full_decode: bool = True,
    compute_hash: bool = True,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 2000,
) -> tuple[ImageProbe, ...]:
    """Probe every record's image, reporting progress periodically.

    Args:
        records: Derived manifest records to inspect.
        images_dir: Directory holding the images.
        full_decode: Whether to force a full pixel decode.
        compute_hash: Whether to compute content hashes.
        progress: Optional ``(done, total)`` callback.
        progress_every: How many images between progress callbacks.

    Returns:
        One probe per record, in record order.
    """
    images_dir = Path(images_dir)
    total = len(records)
    probes: list[ImageProbe] = []
    for index, record in enumerate(records, start=1):
        probes.append(
            probe_image(
                images_dir / record.filename,
                full_decode=full_decode,
                compute_hash=compute_hash,
            )
        )
        if progress and (index % progress_every == 0 or index == total):
            progress(index, total)
    return tuple(probes)
