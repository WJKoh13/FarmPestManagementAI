"""The IP102 detection subset: split manifests, bounding boxes and cropping.

This module supports exactly one question — *does supplying the model a padded
bounding-box crop instead of the full frame improve classification?* — and it is
built so that the answer cannot be confounded by anything else.

Three facts about the source data were established by measurement rather than
assumption, and are pinned by tests:

* ``boxes_top*.json`` stores ``[x1, y1, x2, y2]`` in **absolute pixels**. Over a
  500-box sample, that reading produced zero out-of-bounds or degenerate boxes,
  while an ``[x, y, w, h]`` reading produced 408 violations.
* ``splits_top*.json`` already carries an official ``train``/``val``/``test``
  assignment with a zero-based label per image, so no split is invented here.
  All 9,135 top10 filenames are unique and no filename spans two splits.
* Exactly one top10 image, ``IP022000163.jpg``, has no box.

The crop and full-frame arms of an experiment **share one manifest**. The image
region is the only thing that differs, which is what makes the comparison
paired: :func:`build_detection_records` takes no cropping argument, and the
records it returns are identical for both arms. Images whose box is missing or
invalid are dropped from *both* arms by :func:`partition_records`, so the two
arms always consume exactly the same samples.

Nothing here writes to ``ip102_v1.1``. Cropping happens on the fly, in memory,
as a transform applied before the existing resize/augment pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..scopes import ScopeSpec, is_detection_scope, resolve_scope
from .manifests import ManifestRecord

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from PIL.Image import Image as PILImage

__all__ = [
    "DEFAULT_PADDING",
    "DETECTION_SPLIT_ALIASES",
    "BoundingBox",
    "BoxCropTransform",
    "DetectionDataError",
    "RecordPartition",
    "box_statistics",
    "build_detection_records",
    "crop_with_padding",
    "detection_root",
    "load_boxes",
    "load_splits",
    "pad_and_clamp",
    "partition_records",
    "scope_suffix",
]

#: Fraction of box width and height added to each side. 0.15 is the experiment's
#: fixed padding; it is a constructor argument so a follow-up crop-plus-context
#: arm can vary it without editing this module.
DEFAULT_PADDING: float = 0.15

#: The split files name the held-out split ``val``; the project calls it
#: ``validation`` everywhere else. Mapped here rather than renaming anything on
#: disk, since ``ip102_v1.1`` is read-only.
DETECTION_SPLIT_ALIASES: Mapping[str, str] = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


class DetectionDataError(RuntimeError):
    """Raised when detection splits, boxes or crops cannot be produced."""


@dataclass(frozen=True)
class BoundingBox:
    """One axis-aligned box in absolute pixel coordinates.

    Attributes:
        x1: Left edge, inclusive.
        y1: Top edge, inclusive.
        x2: Right edge, exclusive.
        y2: Bottom edge, exclusive.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return max(0.0, self.width) * max(0.0, self.height)

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(x1, y1, x2, y2)``."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable mapping."""
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


def scope_suffix(scope: str | ScopeSpec) -> str:
    """Return the ``top10`` / ``top15`` file suffix for a detection scope.

    Raises:
        DetectionDataError: If ``scope`` is not a detection scope.
    """
    spec = resolve_scope(scope)
    if not is_detection_scope(spec):
        raise DetectionDataError(
            f"scope {spec.name!r} is not a detection scope; expected det_top10 "
            f"or det_top15"
        )
    return spec.name.removeprefix("det_")


def detection_root(dataset_root: Path) -> Path:
    """Return the ``Detection/VOC2007`` directory under ``dataset_root``."""
    return Path(dataset_root) / "Detection" / "VOC2007"


def _coerce_box(value: Any, filename: str) -> BoundingBox:
    """Validate one raw JSON box entry into a :class:`BoundingBox`.

    Every box must be exactly four finite numbers. Booleans are rejected
    explicitly: ``bool`` is a subclass of ``int``, so ``[True, 0, 5, 5]`` would
    otherwise pass a naive numeric check and silently become ``x1=1``.

    Raises:
        DetectionDataError: If the entry is not four numeric coordinates.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DetectionDataError(
            f"{filename}: box must be a sequence of four numbers, got {value!r}"
        )
    coords = list(value)
    if len(coords) != 4:
        raise DetectionDataError(
            f"{filename}: box must have four coordinates, got {len(coords)}: {coords!r}"
        )
    numbers: list[float] = []
    for axis, raw in zip(("x1", "y1", "x2", "y2"), coords, strict=True):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise DetectionDataError(
                f"{filename}: box coordinate {axis} is not numeric: {raw!r}"
            )
        number = float(raw)
        if number != number or number in (float("inf"), float("-inf")):
            raise DetectionDataError(
                f"{filename}: box coordinate {axis} is not finite: {raw!r}"
            )
        numbers.append(number)
    return BoundingBox(*numbers)


def load_boxes(
    dataset_root: Path, scope: str | ScopeSpec
) -> tuple[dict[str, BoundingBox], dict[str, str]]:
    """Read the boxes JSON for a detection scope.

    Structurally invalid entries are collected rather than raised, so that a
    single bad box reports alongside every other and the audit can quantify
    them. A box that is *absent* is simply not in the returned mapping.

    Args:
        dataset_root: Root of ``ip102_v1.1``.
        scope: A detection scope.

    Returns:
        ``(boxes, invalid)`` where ``boxes`` maps filename to a validated box
        and ``invalid`` maps filename to the reason it was rejected.

    Raises:
        DetectionDataError: If the file is missing or is not a JSON object.
    """
    path = detection_root(dataset_root) / f"boxes_{scope_suffix(scope)}.json"
    if not path.is_file():
        raise DetectionDataError(f"detection boxes file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DetectionDataError(f"failed to read {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DetectionDataError(
            f"{path}: expected a JSON object mapping filename to box, got "
            f"{type(payload).__name__}"
        )

    boxes: dict[str, BoundingBox] = {}
    invalid: dict[str, str] = {}
    for filename, raw in payload.items():
        try:
            boxes[str(filename)] = _coerce_box(raw, str(filename))
        except DetectionDataError as exc:
            invalid[str(filename)] = str(exc)
    return boxes, invalid


def load_splits(
    dataset_root: Path, scope: str | ScopeSpec
) -> dict[str, tuple[tuple[str, int], ...]]:
    """Read the official detection split assignment.

    The file stores ``[[filename, label], ...]`` per split under the keys
    ``train``, ``val`` and ``test``. Those are returned under the project's own
    split names, so callers never deal with the ``val``/``validation`` mismatch.

    Args:
        dataset_root: Root of ``ip102_v1.1``.
        scope: A detection scope.

    Returns:
        Project split name to a tuple of ``(filename, label)`` pairs, in file
        order.

    Raises:
        DetectionDataError: If the file is missing, malformed, a label is out of
            range for the scope, or a filename appears in more than one split.
    """
    spec = resolve_scope(scope)
    path = detection_root(dataset_root) / f"splits_{scope_suffix(spec)}.json"
    if not path.is_file():
        raise DetectionDataError(f"detection splits file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DetectionDataError(f"failed to read {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DetectionDataError(f"{path}: expected a JSON object, got {type(payload)}")

    result: dict[str, tuple[tuple[str, int], ...]] = {}
    seen: dict[str, str] = {}
    for project_split, file_key in DETECTION_SPLIT_ALIASES.items():
        if file_key not in payload:
            raise DetectionDataError(f"{path}: missing split {file_key!r}")
        entries: list[tuple[str, int]] = []
        for item in payload[file_key]:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
                raise DetectionDataError(
                    f"{path}: {file_key} entry must be [filename, label], got {item!r}"
                )
            pair = list(item)
            if len(pair) != 2:
                raise DetectionDataError(
                    f"{path}: {file_key} entry must be [filename, label], got {pair!r}"
                )
            filename = str(pair[0])
            if isinstance(pair[1], bool) or not isinstance(pair[1], int):
                raise DetectionDataError(
                    f"{path}: {filename} has a non-integer label {pair[1]!r}"
                )
            label = int(pair[1])
            if not 0 <= label < spec.num_classes:
                raise DetectionDataError(
                    f"{path}: {filename} carries label {label}, outside "
                    f"0..{spec.num_classes - 1} for scope {spec.name!r}"
                )
            # Cross-split leakage would make validation meaningless, so it is a
            # hard error rather than something the caller may opt into.
            if filename in seen:
                raise DetectionDataError(
                    f"{path}: {filename} appears in both {seen[filename]!r} and "
                    f"{project_split!r}; the splits must be disjoint"
                )
            seen[filename] = project_split
            entries.append((filename, label))
        result[project_split] = tuple(entries)
    return result


def build_detection_records(
    dataset_root: Path,
    scope: str | ScopeSpec,
    split: str,
    *,
    class_names: Mapping[int, str] | None = None,
) -> tuple[ManifestRecord, ...]:
    """Build manifest records for one split of a detection scope.

    Deliberately takes no cropping argument: the crop and full-frame arms of an
    experiment call this identically and receive identical records. Which pixels
    reach the model is decided later, by the transform.

    ``ip102_label`` is set to the detection project label. The detection subset
    does not expose a mapping back to IP102 classification labels, and inventing
    one would let a detection result be silently joined to a ``full102`` one.

    Args:
        dataset_root: Root of ``ip102_v1.1``.
        scope: A detection scope.
        split: ``"train"``, ``"validation"`` or ``"test"``.
        class_names: Optional label-to-name mapping for reporting.

    Returns:
        Records in official split order.

    Raises:
        DetectionDataError: If the split is unknown or its file is malformed.
    """
    spec = resolve_scope(scope)
    if split not in DETECTION_SPLIT_ALIASES:
        raise DetectionDataError(
            f"unknown split {split!r}; expected one of "
            f"{list(DETECTION_SPLIT_ALIASES)}"
        )
    splits = load_splits(dataset_root, spec)
    names = dict(class_names or {})
    return tuple(
        ManifestRecord(
            filename=filename,
            ip102_label=label,
            project_label=label,
            class_name=names.get(label, f"{spec.name}_class_{label}"),
            split=split,
        )
        for filename, label in splits[split]
    )


@dataclass(frozen=True)
class RecordPartition:
    """Records split into those usable by both arms and those dropped.

    Attributes:
        kept: Records whose box is present and valid.
        dropped: ``(record, reason)`` for each excluded record.
    """

    kept: tuple[ManifestRecord, ...]
    dropped: tuple[tuple[ManifestRecord, str], ...]

    @property
    def dropped_filenames(self) -> tuple[str, ...]:
        """Filenames excluded from both arms."""
        return tuple(record.filename for record, _ in self.dropped)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "kept": len(self.kept),
            "dropped": len(self.dropped),
            "dropped_detail": [
                {"filename": record.filename, "reason": reason}
                for record, reason in self.dropped
            ],
        }


def partition_records(
    records: Iterable[ManifestRecord],
    boxes: Mapping[str, BoundingBox],
    invalid: Mapping[str, str] | None = None,
) -> RecordPartition:
    """Split records into those with a usable box and those without.

    Applied to **both** arms of a pair, including the full-frame arm which does
    not itself need a box. Dropping an image from only the crop arm would leave
    the two arms scoring different sample sets, and their difference would no
    longer be attributable to the crop alone.

    A box is unusable when it is missing, structurally invalid, or degenerate
    (zero or negative width or height).

    Args:
        records: Candidate records.
        boxes: Validated boxes by filename.
        invalid: Structural rejections from :func:`load_boxes`.

    Returns:
        The partition, preserving input order in both groups.
    """
    rejected = dict(invalid or {})
    kept: list[ManifestRecord] = []
    dropped: list[tuple[ManifestRecord, str]] = []
    for record in records:
        reason = rejected.get(record.filename)
        if reason is not None:
            dropped.append((record, f"invalid box: {reason}"))
            continue
        box = boxes.get(record.filename)
        if box is None:
            dropped.append((record, "missing box"))
            continue
        if box.width <= 0 or box.height <= 0:
            dropped.append(
                (record, f"degenerate box {box.as_tuple()} (width/height <= 0)")
            )
            continue
        kept.append(record)
    return RecordPartition(kept=tuple(kept), dropped=tuple(dropped))


def pad_and_clamp(
    box: BoundingBox,
    image_width: int,
    image_height: int,
    padding: float = DEFAULT_PADDING,
) -> BoundingBox:
    """Grow a box by ``padding`` on every side and clamp it to the image.

    Padding is measured **relative to the box's own width and height**, so a
    small box grows by a small number of pixels. Each side grows independently:
    a box against the left edge still gains its full padding on the right, which
    keeps the padded region's area closer to the intent than shifting would.

    The result is clamped to ``[0, image_width] x [0, image_height]`` and is
    guaranteed to be at least one pixel in each dimension, so the crop can never
    be empty.

    Args:
        box: The unpadded box.
        image_width: Source image width in pixels.
        image_height: Source image height in pixels.
        padding: Fraction of width/height added per side.

    Returns:
        The padded, clamped box.

    Raises:
        DetectionDataError: If the image dimensions or padding are invalid.
    """
    if image_width <= 0 or image_height <= 0:
        raise DetectionDataError(
            f"image dimensions must be positive, got {image_width}x{image_height}"
        )
    if padding < 0:
        raise DetectionDataError(f"padding must be non-negative, got {padding}")

    pad_x = box.width * padding
    pad_y = box.height * padding
    x1 = max(0.0, box.x1 - pad_x)
    y1 = max(0.0, box.y1 - pad_y)
    x2 = min(float(image_width), box.x2 + pad_x)
    y2 = min(float(image_height), box.y2 + pad_y)

    # A box lying entirely outside the frame, or one whose rounding collapses,
    # would otherwise yield a zero-area crop that Pillow turns into an empty
    # image and the resize silently accepts.
    if x2 - x1 < 1.0:
        x1 = max(0.0, min(x1, float(image_width) - 1.0))
        x2 = x1 + 1.0
    if y2 - y1 < 1.0:
        y1 = max(0.0, min(y1, float(image_height) - 1.0))
        y2 = y1 + 1.0
    return BoundingBox(x1, y1, x2, y2)


def crop_with_padding(
    image: PILImage,
    box: BoundingBox,
    padding: float = DEFAULT_PADDING,
) -> PILImage:
    """Return the padded crop of ``image``.

    The source image is never modified: :meth:`PIL.Image.Image.crop` returns a
    new image.

    Args:
        image: Decoded RGB source image.
        box: Unpadded box in absolute pixels.
        padding: Fraction of width/height added per side.

    Returns:
        The cropped region.
    """
    width, height = image.size
    padded = pad_and_clamp(box, width, height, padding)
    left = round(padded.x1)
    top = round(padded.y1)
    right = max(left + 1, round(padded.x2))
    bottom = max(top + 1, round(padded.y2))
    right = min(right, width)
    bottom = min(bottom, height)
    left = min(left, right - 1)
    top = min(top, bottom - 1)
    return image.crop((left, top, right, bottom))


class BoxCropTransform:
    """Crop to a padded box, then apply the project's normal pipeline.

    This is the **only** difference between a crop arm and its full-frame
    control. Resizing, augmentation and normalisation all happen inside
    ``inner``, which is the transform :func:`farm_pest_ai.data.transforms.build_transform`
    already builds, so the crop arm inherits every preprocessing decision
    unchanged instead of reimplementing them.

    The transform is keyed by filename because a map-style dataset applies its
    transform to a decoded image with no other context. ``PestImageDataset``
    passes the filename through when the transform declares it wants one.

    Attributes are read-only after construction; instances are shared with
    DataLoader workers by fork/spawn and nothing mutates them.
    """

    #: Marks this transform as needing the record's filename, which the dataset
    #: checks for rather than inspecting the call signature.
    wants_filename: bool = True

    def __init__(
        self,
        boxes: Mapping[str, BoundingBox],
        inner: Callable[[PILImage], Any] | None = None,
        *,
        padding: float = DEFAULT_PADDING,
    ) -> None:
        """Build the transform.

        Args:
            boxes: Validated boxes by filename.
            inner: Transform applied to the cropped image.
            padding: Fraction of box width/height added per side.

        Raises:
            DetectionDataError: If ``padding`` is negative.
        """
        if padding < 0:
            raise DetectionDataError(f"padding must be non-negative, got {padding}")
        self._boxes = dict(boxes)
        self._inner = inner
        self._padding = float(padding)

    @property
    def padding(self) -> float:
        """Padding fraction applied per side."""
        return self._padding

    @property
    def inner(self) -> Callable[[PILImage], Any] | None:
        """The wrapped preprocessing pipeline."""
        return self._inner

    def __call__(self, image: PILImage, filename: str | None = None) -> Any:
        """Crop ``image`` to its padded box and apply the inner transform.

        Raises:
            DetectionDataError: If no filename is supplied or the file has no
                box. Both are failures of the caller's contract rather than
                recoverable conditions: falling back to the full frame here
                would silently turn a crop-arm sample into a full-frame one and
                corrupt the very comparison the experiment exists to make.
        """
        if filename is None:
            raise DetectionDataError(
                "BoxCropTransform requires the record's filename; the dataset must "
                "pass it through"
            )
        box = self._boxes.get(filename)
        if box is None:
            raise DetectionDataError(
                f"{filename}: no bounding box available; records without a usable "
                f"box must be dropped from both arms before training"
            )
        cropped = crop_with_padding(image, box, self._padding)
        if self._inner is None:
            return cropped
        return self._inner(cropped)

    def __repr__(self) -> str:
        """Return a concise description for logs."""
        return (
            f"{type(self).__name__}(boxes={len(self._boxes)}, "
            f"padding={self._padding})"
        )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description recorded with every run."""
        return {
            "kind": "box_crop",
            "padding": self._padding,
            "boxes": len(self._boxes),
        }


def box_statistics(
    boxes: Mapping[str, BoundingBox],
    sizes: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    """Summarise box area as a fraction of image area.

    Args:
        boxes: Validated boxes by filename.
        sizes: ``(width, height)`` per filename.

    Returns:
        Counts, percentiles of the area ratio, and the number and percentage of
        boxes below the 10%, 25% and 50% thresholds the audit reports.
    """
    ratios: list[float] = []
    missing_size: list[str] = []
    for filename, box in boxes.items():
        size = sizes.get(filename)
        if size is None:
            missing_size.append(filename)
            continue
        width, height = size
        if width <= 0 or height <= 0:
            missing_size.append(filename)
            continue
        ratios.append(box.area / float(width * height))

    ratios.sort()

    def percentile(fraction: float) -> float | None:
        """Nearest-rank percentile of the sorted ratios."""
        if not ratios:
            return None
        index = min(len(ratios) - 1, max(0, round(fraction * (len(ratios) - 1))))
        return round(ratios[index], 6)

    total = len(ratios)
    thresholds: dict[str, dict[str, float | int]] = {}
    for label, cutoff in (("below_10pct", 0.10), ("below_25pct", 0.25), ("below_50pct", 0.50)):
        count = sum(1 for r in ratios if r < cutoff)
        thresholds[label] = {
            "count": count,
            "percent": round(100.0 * count / total, 3) if total else 0.0,
        }

    return {
        "boxes_measured": total,
        "sizes_unavailable": len(missing_size),
        "area_ratio_percentiles": {
            "p01": percentile(0.01),
            "p05": percentile(0.05),
            "p10": percentile(0.10),
            "p25": percentile(0.25),
            "p50": percentile(0.50),
            "p75": percentile(0.75),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        },
        "area_ratio_min": round(ratios[0], 6) if ratios else None,
        "area_ratio_max": round(ratios[-1], 6) if ratios else None,
        "area_ratio_mean": round(sum(ratios) / total, 6) if total else None,
        "thresholds": thresholds,
    }
