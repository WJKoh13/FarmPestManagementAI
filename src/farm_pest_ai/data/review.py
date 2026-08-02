"""Read-only image-quality review: flags, manifests and contact sheets.

This module **proposes**; it never decides. It measures objective image
properties, records what a trained model predicted, and writes a manifest with
empty ``reviewer_decision`` and ``reviewer_notes`` columns for a human to fill
in. Nothing here relabels an image, moves a file, or edits a benchmark manifest.

Why the automated part stops at "suspected"
    A low-confidence prediction is evidence about the *model*, not proof about
    the *label*. A blurry photograph of the correct pest is still correctly
    labelled, and a sharp photograph the model finds easy can still be labelled
    wrong. Only :data:`OBJECTIVE_FLAGS` are measured from pixels and are safe to
    assert; every category that requires judgement — including
    ``suspected_mislabel`` — is a queue for a person, which is why the decision
    column ships empty.

Dataset safety
    ``ip102_v1.1`` is opened read-only. Images are decoded to measure them and
    to build contact sheets; no source file is written, renamed or re-encoded.
    Any curated split that a review eventually justifies belongs in a **new
    versioned derived-manifest directory**, never on top of the official one —
    see :func:`curated_manifest_dir`.

Test-split discipline
    :func:`build_review_manifest` refuses the test split outright. The review
    exists to inform decisions about data and preprocessing, which is exactly
    the kind of decision the test split must never participate in.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "OBJECTIVE_FLAGS",
    "REVIEWABLE_SPLITS",
    "REVIEW_CATEGORIES",
    "REVIEW_COLUMNS",
    "ReviewError",
    "ReviewRecord",
    "ReviewThresholds",
    "build_contact_sheet",
    "curated_manifest_dir",
    "measure_image",
    "read_review_manifest",
    "suggest_issue",
    "write_review_manifest",
]

#: Every category a reviewer may assign. ``valid_close_up`` and
#: ``difficult_but_valid`` both mean "keep"; the distinction records whether the
#: image is *easy*, which matters when reading a per-class score.
REVIEW_CATEGORIES: tuple[str, ...] = (
    "valid_close_up",
    "difficult_but_valid",
    "blurry",
    "low_resolution",
    "tiny_subject",
    "symptom_only",
    "diagram_text",
    "unrelated",
    "ambiguous",
    "suspected_mislabel",
)

#: Categories this module may assert from pixels alone. Everything else in
#: :data:`REVIEW_CATEGORIES` needs a human, and is only ever *suggested*.
OBJECTIVE_FLAGS: tuple[str, ...] = ("blurry", "low_resolution")

#: Column order of the review manifest. ``reviewer_decision`` and
#: ``reviewer_notes`` are written empty and are the human's to fill.
REVIEW_COLUMNS: tuple[str, ...] = (
    "filename",
    "split",
    "current_label",
    "current_class_name",
    "width",
    "height",
    "short_side",
    "aspect_ratio",
    "model_prediction",
    "model_prediction_name",
    "confidence",
    "prediction_correct",
    "quality_flags",
    "suspected_issue",
    "reviewer_decision",
    "reviewer_notes",
)

#: Splits a review may read. The test split is excluded by policy, not by
#: oversight: reviewing it would let its contents shape a data decision.
REVIEWABLE_SPLITS: tuple[str, ...] = ("train", "validation")


class ReviewError(ValueError):
    """Raised when a review would be unsafe or is malformed."""


@dataclass(frozen=True)
class ReviewThresholds:
    """Cut-offs for the objectively measurable flags.

    Attributes:
        min_short_side: Below this, the image is upscaled by preprocessing and
            is flagged ``low_resolution``. Defaults to the model's input size.
        blur_variance: Variance-of-Laplacian below which an image is flagged
            ``blurry``. A focus measure, not a quality verdict: low-texture
            subjects score low while being perfectly sharp, which is why the
            flag is a queue rather than a conclusion.
        low_confidence: Predicted probability below which an image is queued as
            ``ambiguous`` for review.
    """

    min_short_side: int = 160
    blur_variance: float = 100.0
    low_confidence: float = 0.35

    def __post_init__(self) -> None:
        """Validate the cut-offs.

        Raises:
            ReviewError: If any threshold is outside its meaningful range.
        """
        if self.min_short_side < 1:
            raise ReviewError(
                f"min_short_side must be positive, got {self.min_short_side}"
            )
        if self.blur_variance < 0:
            raise ReviewError(
                f"blur_variance must be non-negative, got {self.blur_variance}"
            )
        if not 0.0 <= self.low_confidence <= 1.0:
            raise ReviewError(
                f"low_confidence must be in [0, 1], got {self.low_confidence}"
            )


@dataclass(frozen=True)
class ReviewRecord:
    """One row of the review manifest.

    Attributes:
        filename: Source filename; the join key back to the manifest.
        split: ``train`` or ``validation``.
        current_label: The project label the official manifest assigns.
        current_class_name: That label's class name.
        width: Decoded pixel width.
        height: Decoded pixel height.
        model_prediction: Predicted project label, when a model was supplied.
        model_prediction_name: That prediction's class name.
        confidence: Predicted probability for the predicted class.
        quality_flags: Objectively measured flags; see :data:`OBJECTIVE_FLAGS`.
        suspected_issue: A single suggested category for the reviewer to
            confirm or reject. Never a decision.
    """

    filename: str
    split: str
    current_label: int
    current_class_name: str
    width: int
    height: int
    model_prediction: int | None = None
    model_prediction_name: str = ""
    confidence: float | None = None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    suspected_issue: str = ""

    @property
    def short_side(self) -> int:
        """The smaller dimension, which drives the upscaling flag."""
        return min(self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height."""
        return self.width / self.height if self.height else 0.0

    @property
    def prediction_correct(self) -> bool | None:
        """Whether the model agreed with the current label."""
        if self.model_prediction is None:
            return None
        return self.model_prediction == self.current_label

    def to_row(self) -> dict[str, Any]:
        """Render as a manifest row, with the reviewer columns left empty."""
        correct = self.prediction_correct
        return {
            "filename": self.filename,
            "split": self.split,
            "current_label": self.current_label,
            "current_class_name": self.current_class_name,
            "width": self.width,
            "height": self.height,
            "short_side": self.short_side,
            "aspect_ratio": f"{self.aspect_ratio:.4f}",
            "model_prediction": (
                "" if self.model_prediction is None else self.model_prediction
            ),
            "model_prediction_name": self.model_prediction_name,
            "confidence": "" if self.confidence is None else f"{self.confidence:.4f}",
            "prediction_correct": "" if correct is None else str(correct).lower(),
            "quality_flags": "|".join(self.quality_flags),
            "suspected_issue": self.suspected_issue,
            # The human's columns. Deliberately empty.
            "reviewer_decision": "",
            "reviewer_notes": "",
        }


def _laplacian_variance(grey: Any) -> float:
    """Variance of the Laplacian of a greyscale image, as a focus measure.

    Implemented with an explicit 3x3 kernel over the raw pixel buffer rather
    than a dependency, so the audit needs nothing beyond Pillow.
    """
    width, height = grey.size
    if width < 3 or height < 3:
        return 0.0

    # `getdata` is deprecated in Pillow 12 in favour of `get_flattened_data`.
    # Both return the same flat buffer; prefer the new name where it exists so
    # the audit does not emit a DeprecationWarning on newer Pillow.
    reader = getattr(grey, "get_flattened_data", None) or grey.getdata
    pixels = list(reader())
    values: list[float] = []
    for y in range(1, height - 1):
        row = y * width
        above = row - width
        below = row + width
        for x in range(1, width - 1):
            centre = pixels[row + x]
            value = (
                pixels[above + x]
                + pixels[below + x]
                + pixels[row + x - 1]
                + pixels[row + x + 1]
                - 4 * centre
            )
            values.append(float(value))

    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def measure_image(
    path: Path, *, thresholds: ReviewThresholds | None = None, sample_size: int = 256
) -> dict[str, Any]:
    """Measure one image's objective quality properties.

    The image is opened read-only and never written back.

    Args:
        path: Image to measure.
        thresholds: Cut-offs; defaults to :class:`ReviewThresholds`.
        sample_size: The focus measure is computed on a copy downscaled to fit
            this box, which keeps a full-scope audit tractable while preserving
            the relative ordering the flag depends on.

    Returns:
        Width, height, the focus measure and any objective flags raised.

    Raises:
        ReviewError: If the file cannot be decoded.
    """
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - Pillow is a dependency
        raise ReviewError("Pillow is required for the image-quality review") from error

    limits = thresholds or ReviewThresholds()
    try:
        with Image.open(path) as handle:
            handle.load()
            width, height = handle.size
            grey = handle.convert("L")
            grey.thumbnail((sample_size, sample_size))
            focus = _laplacian_variance(grey)
    except OSError as error:
        raise ReviewError(f"could not decode {path}: {error}") from error

    flags: list[str] = []
    if min(width, height) < limits.min_short_side:
        flags.append("low_resolution")
    if focus < limits.blur_variance:
        flags.append("blurry")

    return {
        "width": width,
        "height": height,
        "focus_measure": focus,
        "quality_flags": tuple(flags),
    }


def suggest_issue(
    record_flags: Sequence[str],
    *,
    confidence: float | None,
    prediction_correct: bool | None,
    thresholds: ReviewThresholds | None = None,
) -> str:
    """Suggest one category for the reviewer to confirm.

    This is a triage hint and nothing more. In particular a confident
    disagreement between model and label is reported as ``suspected_mislabel``
    — *suspected*, because a model can be confidently wrong, and only a person
    looking at the image can tell the two apart.

    Returns:
        A member of :data:`REVIEW_CATEGORIES`, or ``""`` when nothing stands out.
    """
    limits = thresholds or ReviewThresholds()

    # Objective measurements first: they describe the file, not the model.
    if "low_resolution" in record_flags:
        return "low_resolution"
    if "blurry" in record_flags:
        return "blurry"

    if confidence is None or prediction_correct is None:
        return ""
    if confidence < limits.low_confidence:
        return "ambiguous"
    if not prediction_correct:
        # Confident and disagreeing with the label. Worth a human's time; not
        # evidence on its own.
        return "suspected_mislabel"
    return ""


def write_review_manifest(records: Iterable[ReviewRecord], path: Path) -> Path:
    """Write the review manifest as CSV with LF endings.

    The ``reviewer_decision`` and ``reviewer_notes`` columns are written empty.

    Args:
        records: Rows to write.
        path: Destination CSV.

    Returns:
        The path written.
    """
    from farm_pest_ai.data.manifests import atomic_write_text

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(REVIEW_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(record.to_row())

    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_text(path, buffer.getvalue())


def read_review_manifest(path: Path) -> list[dict[str, str]]:
    """Read a review manifest back, including any reviewer decisions.

    Raises:
        ReviewError: If a row carries a ``reviewer_decision`` outside
            :data:`REVIEW_CATEGORIES`, which would otherwise propagate a typo
            into whatever consumes the review.
    """
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    allowed = set(REVIEW_CATEGORIES)
    for number, row in enumerate(rows, start=2):
        decision = (row.get("reviewer_decision") or "").strip()
        if decision and decision not in allowed:
            raise ReviewError(
                f"{path}:{number}: reviewer_decision {decision!r} is not one of "
                f"{sorted(allowed)}"
            )
    return rows


def curated_manifest_dir(processed_dir: Path, scope: str, version: str) -> Path:
    """Resolve the directory a curated manifest version may be written to.

    A curated split never overwrites the official derived manifest. It goes to
    ``<processed_dir>/<scope>/curated/<version>/``, so the benchmark manifests
    stay byte-identical and any curated experiment states which version it used.

    Raises:
        ReviewError: If ``version`` is empty or is a path rather than a name.
    """
    if not version or version != Path(version).name or version in {".", ".."}:
        raise ReviewError(f"curated version must be a simple name, got {version!r}")
    return Path(processed_dir) / scope / "curated" / version


def build_contact_sheet(
    images: Sequence[tuple[Path, str]],
    *,
    columns: int = 6,
    thumbnail: int = 180,
    label_height: int = 22,
) -> Any:
    """Build one contact sheet from image paths and captions.

    Contact sheets exist because a human cannot review thousands of rows in a
    spreadsheet, but can scan a page of thumbnails quickly. Source images are
    opened read-only and copies are scaled into the sheet.

    Args:
        images: ``(path, caption)`` pairs, in reading order.
        columns: Thumbnails per row.
        thumbnail: Bounding box for each thumbnail, in pixels.
        label_height: Caption strip height beneath each thumbnail.

    Returns:
        The assembled ``PIL.Image``.

    Raises:
        ReviewError: If no images are given or Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - Pillow is a dependency
        raise ReviewError("Pillow is required to build contact sheets") from error

    if not images:
        raise ReviewError("a contact sheet needs at least one image")
    if columns < 1:
        raise ReviewError(f"columns must be positive, got {columns}")

    cell_height = thumbnail + label_height
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * thumbnail, rows * cell_height), color=(252, 252, 251)
    )
    draw = ImageDraw.Draw(sheet)

    for index, (path, caption) in enumerate(images):
        column, row = index % columns, index // columns
        x, y = column * thumbnail, row * cell_height
        try:
            with Image.open(path) as handle:
                handle.load()
                tile = handle.convert("RGB")
                tile.thumbnail((thumbnail, thumbnail))
        except OSError:
            # A file that will not decode is still worth a slot, so the sheet
            # keeps its alignment with the manifest rows.
            draw.rectangle(
                [x, y, x + thumbnail - 1, y + thumbnail - 1], fill=(224, 224, 220)
            )
            tile = None

        if tile is not None:
            offset_x = x + (thumbnail - tile.width) // 2
            offset_y = y + (thumbnail - tile.height) // 2
            sheet.paste(tile, (offset_x, offset_y))

        draw.text(
            (x + 3, y + thumbnail + 4),
            caption[:44],
            fill=(11, 11, 11),
        )

    return sheet
