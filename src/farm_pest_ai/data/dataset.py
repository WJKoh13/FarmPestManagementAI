"""The PyTorch ``Dataset`` that turns derived manifests into tensors.

:class:`PestImageDataset` is deliberately thin. It reads a derived manifest
produced in Phase 4, decodes the referenced image, and applies the split's
transform pipeline. Every decision that could change a label lives elsewhere:
the scope mapping in :mod:`farm_pest_ai.scopes`, the manifest in
:mod:`farm_pest_ai.data.manifests`, the pixels in
:mod:`farm_pest_ai.data.transforms`.

Two invariants are enforced here rather than assumed:

* the manifest's recorded scope and class-mapping version must match the scope
  being requested, so a ``rice10`` manifest can never be loaded as ``full102``;
* decoding dispatches on file content and converts to RGB, because Phase 4 found
  ten ``.jpg`` files that are really PNG, seven of them RGBA.

Torch and Pillow are imported inside the functions that need them so that
manifest tooling and the configuration tests keep working in an environment
without the training extras installed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..scopes import CLASS_MAPPING_VERSION, ScopeSpec, resolve_scope
from .manifests import SPLITS, ManifestError, ManifestRecord, read_derived_manifest

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from PIL.Image import Image as PILImage

__all__ = [
    "DatasetError",
    "PestImageDataset",
    "Sample",
    "class_counts",
    "class_weights",
    "load_image",
]


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be constructed or a sample cannot be read."""


@dataclass(frozen=True)
class Sample:
    """One decoded, transformed example.

    Batches are collated as plain tuples for speed; this dataclass is what
    :meth:`PestImageDataset.sample_metadata` returns when a caller needs the
    provenance of an index, for example when writing per-image predictions in
    Phase 9.

    Attributes:
        index: Position in the dataset.
        filename: Bare filename, matching the manifest and the source data.
        path: Absolute path to the image file.
        project_label: Label in the active scope's numbering.
        ip102_label: Original IP102 label.
        class_name: Raw class name from ``classes.txt``.
        split: Split the record belongs to.
    """

    index: int
    filename: str
    path: Path
    project_label: int
    ip102_label: int
    class_name: str
    split: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "index": self.index,
            "filename": self.filename,
            "path": str(self.path),
            "project_label": self.project_label,
            "ip102_label": self.ip102_label,
            "class_name": self.class_name,
            "split": self.split,
        }


def load_image(path: Path) -> PILImage:
    """Decode one image file into an RGB :class:`PIL.Image.Image`.

    Pillow dispatches on the file's magic bytes, not its extension, which is
    what makes the ten mislabelled PNG files work. The conversion to RGB is
    unconditional and applied here as well as in the transform pipeline, so a
    caller that bypasses the pipeline still cannot obtain a four-channel array.

    Args:
        path: Image file to read.

    Returns:
        The decoded image in RGB mode, fully loaded and detached from the file
        handle.

    Raises:
        DatasetError: If the file is missing, unreadable, or not an image. The
            path is included, because a bare Pillow error during a DataLoader
            worker's run is otherwise very hard to trace.
    """
    from PIL import Image

    from .transforms import to_rgb

    try:
        with Image.open(path) as image:
            # load() completes the decode before the file handle closes; without
            # it, Pillow's lazy read would fail after the context manager exits.
            image.load()
            return to_rgb(image)
    except (OSError, ValueError) as exc:
        raise DatasetError(
            f"failed to decode image {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _dataset_base() -> Any:
    """Return ``torch.utils.data.Dataset``, or :class:`object` without torch.

    Subclassing the real base when it is available keeps ``DataLoader`` happy
    for both static type checkers and ``isinstance`` checks in later phases,
    while keeping torch out of this module's import requirements.
    """
    try:
        from torch.utils.data import Dataset
    except ImportError:  # pragma: no cover - environments without torch
        return object
    return Dataset


class PestImageDataset(_dataset_base()):  # type: ignore[misc]
    """A map-style dataset over one split of a derived manifest.

    Subclasses ``torch.utils.data.Dataset`` when torch is installed, and plain
    :class:`object` otherwise, so manifest tooling still imports this module in
    an environment without the training extras.

    Attributes are read-only after construction. The record tuple is shared with
    DataLoader workers by fork/spawn, which is safe because nothing mutates it.
    """

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        images_dir: Path,
        scope: str | ScopeSpec,
        split: str,
        *,
        transform: Callable[[PILImage], Any] | None = None,
        verify_files: bool = False,
        manifest_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Build a dataset from already-read manifest records.

        Args:
            records: Derived manifest records for one split.
            images_dir: Directory holding the image files.
            scope: The active scope; labels are checked against it.
            split: One of :data:`farm_pest_ai.data.manifests.SPLITS`.
            transform: Callable applied to the decoded RGB image. When ``None``
                the raw :class:`PIL.Image.Image` is returned, which is what the
                dimension spot-check uses.
            verify_files: Whether to stat every referenced file at construction
                time. Off by default: ``scripts/verify_dataset.py`` already does
                this once, and repeating it per epoch would cost thousands of
                stats for no new information.
            manifest_metadata: Provenance sidecar, when the records came from a
                derived manifest on disk. Recorded verbatim by :meth:`describe`.

        Raises:
            DatasetError: If ``split`` is unknown, ``records`` is empty, a label
                is outside the scope's range, or ``verify_files`` finds a
                missing file.
        """
        if split not in SPLITS:
            raise DatasetError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
        spec = resolve_scope(scope)
        if not records:
            raise DatasetError(f"{split}: dataset has no records")

        images_dir = Path(images_dir)
        for record in records:
            if not 0 <= record.project_label < spec.num_classes:
                raise DatasetError(
                    f"{split}: {record.filename} carries project label "
                    f"{record.project_label}, outside 0..{spec.num_classes - 1} for "
                    f"scope {spec.name!r}"
                )

        if verify_files:
            missing = [
                r.filename for r in records if not (images_dir / r.filename).is_file()
            ]
            if missing:
                raise DatasetError(
                    f"{split}: {len(missing)} referenced image(s) missing from "
                    f"{images_dir}, e.g. {missing[:5]}"
                )

        self._records: tuple[ManifestRecord, ...] = tuple(records)
        self._images_dir = images_dir
        self._scope = spec
        self._split = split
        self._transform = transform
        self._metadata: dict[str, Any] | None = (
            dict(manifest_metadata) if manifest_metadata is not None else None
        )

    # -- construction -------------------------------------------------

    @classmethod
    def from_manifest(
        cls,
        processed_dir: Path,
        scope: str | ScopeSpec,
        split: str,
        images_dir: Path,
        *,
        transform: Callable[[PILImage], Any] | None = None,
        manifest_version: str | None = None,
        verify_files: bool = False,
    ) -> PestImageDataset:
        """Read a derived manifest from disk and build a dataset from it.

        The manifest's metadata is checked before its records are trusted:
        :func:`~farm_pest_ai.data.manifests.read_derived_manifest` rejects a
        mismatched scope or class-mapping version, and ``manifest_version`` is
        checked here when supplied.

        Args:
            processed_dir: Root of ``data/processed``.
            scope: Scope name or spec.
            split: Split to load.
            images_dir: Directory holding the image files.
            transform: Transform applied to the decoded image.
            manifest_version: When given, the manifest must record this version.
            verify_files: Whether to stat every file at construction time.

        Returns:
            The constructed dataset.

        Raises:
            DatasetError: If the manifest is missing or its recorded provenance
                does not match what was requested.
        """
        spec = resolve_scope(scope)
        try:
            records, metadata = read_derived_manifest(processed_dir, spec, split)
        except ManifestError as exc:
            raise DatasetError(str(exc)) from exc

        recorded_version = metadata.get("manifest_version")
        if manifest_version is not None and recorded_version != manifest_version:
            raise DatasetError(
                f"{split}: manifest version {recorded_version!r} does not match the "
                f"configured {manifest_version!r}; rebuild the manifests"
            )

        return cls(
            records,
            images_dir,
            spec,
            split,
            transform=transform,
            verify_files=verify_files,
            manifest_metadata=metadata,
        )

    # -- Dataset protocol ---------------------------------------------

    def __len__(self) -> int:
        """Number of records in this split."""
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Return ``(image, project_label)`` for ``index``.

        The label is a plain :class:`int`; the default collate turns a batch of
        them into an ``int64`` tensor, which is what the cross-entropy loss
        expects.

        Raises:
            DatasetError: If the image cannot be decoded. The failure names the
                file, so a corrupt image found mid-epoch is identifiable.
        """
        record = self._records[index]
        image = load_image(self._images_dir / record.filename)
        if self._transform is not None:
            # A transform that declares `wants_filename` needs the record's
            # provenance, not just its pixels - the detection crop transform
            # looks its bounding box up by filename. Checking for the attribute
            # keeps every existing transform's one-argument call unchanged.
            if getattr(self._transform, "wants_filename", False):
                filename_aware = cast(
                    "Callable[[PILImage, str], Any]", self._transform
                )
                return filename_aware(image, record.filename), record.project_label
            return self._transform(image), record.project_label
        return image, record.project_label

    def __repr__(self) -> str:
        """Return a concise description for logs and error messages."""
        return (
            f"{type(self).__name__}(scope={self._scope.name!r}, split={self._split!r}, "
            f"records={len(self._records)}, num_classes={self._scope.num_classes})"
        )

    # -- accessors ----------------------------------------------------

    @property
    def records(self) -> tuple[ManifestRecord, ...]:
        """The underlying manifest records, in official split order."""
        return self._records

    @property
    def scope(self) -> ScopeSpec:
        """The scope this dataset was built for."""
        return self._scope

    @property
    def split(self) -> str:
        """The split this dataset covers."""
        return self._split

    @property
    def num_classes(self) -> int:
        """Number of output classes, derived from the scope."""
        return self._scope.num_classes

    @property
    def images_dir(self) -> Path:
        """Directory the images are read from."""
        return self._images_dir

    @property
    def transform(self) -> Callable[[PILImage], Any] | None:
        """The transform applied to each decoded image."""
        return self._transform

    @property
    def targets(self) -> tuple[int, ...]:
        """Project labels in dataset order.

        Exposed so a weighted sampler or a class-weight computation can be built
        without decoding a single image.
        """
        return tuple(record.project_label for record in self._records)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Raw class names indexed by project label.

        Raises:
            DatasetError: If a class has no records in this split, since the name
                cannot then be recovered from the manifest.
        """
        names: dict[int, str] = {}
        for record in self._records:
            names.setdefault(record.project_label, record.class_name)
        missing = [label for label in range(self.num_classes) if label not in names]
        if missing:
            raise DatasetError(
                f"{self._split}: no records for project label(s) {missing}, so their "
                f"class names are unknown in this split"
            )
        return tuple(names[label] for label in range(self.num_classes))

    def sample_metadata(self, index: int) -> Sample:
        """Return the provenance of one index without decoding the image."""
        record = self._records[index]
        return Sample(
            index=index,
            filename=record.filename,
            path=self._images_dir / record.filename,
            project_label=record.project_label,
            ip102_label=record.ip102_label,
            class_name=record.class_name,
            split=self._split,
        )

    def class_counts(self) -> dict[int, int]:
        """Records per project label, including classes with zero."""
        return class_counts(self.targets, self.num_classes)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary recorded with every run."""
        counts = self.class_counts()
        present = [v for v in counts.values() if v > 0]
        return {
            "scope": self._scope.name,
            "split": self._split,
            "num_classes": self.num_classes,
            "class_mapping_version": CLASS_MAPPING_VERSION,
            "records": len(self._records),
            "images_dir": str(self._images_dir),
            "classes_present": len(present),
            "empty_classes": [k for k, v in counts.items() if v == 0],
            "min_class_count": min(present) if present else 0,
            "max_class_count": max(present) if present else 0,
            "imbalance_ratio": (
                round(max(present) / min(present), 2) if present else None
            ),
            "manifest_metadata": self._metadata,
        }


def class_counts(targets: Sequence[int], num_classes: int) -> dict[int, int]:
    """Count occurrences of each project label.

    Args:
        targets: Project labels.
        num_classes: Number of classes; every label in ``0..num_classes-1``
            appears in the result, including those with no records.

    Returns:
        Label to count, in ascending label order.

    Raises:
        ValueError: If a target falls outside the valid range.
    """
    counts = dict.fromkeys(range(num_classes), 0)
    for label in targets:
        if not 0 <= label < num_classes:
            raise ValueError(
                f"label {label} is outside 0..{num_classes - 1}"
            )
        counts[label] += 1
    return counts


def class_weights(
    targets: Sequence[int],
    num_classes: int,
    *,
    scheme: str = "inverse",
    normalize: bool = True,
    beta: float = 0.9999,
) -> tuple[float, ...]:
    """Compute per-class loss weights from label counts.

    **Only training labels may be passed here.** Deriving weights from
    validation or test labels would leak evaluation information into training,
    which the project rules forbid; the loader enforces this by computing
    weights from the training dataset alone.

    Args:
        targets: Project labels from the training split.
        num_classes: Number of classes.
        scheme: ``"none"`` for uniform weights, ``"inverse"`` for ``1/count``,
            ``"inverse_sqrt"`` for ``1/sqrt(count)`` (a gentler correction), or
            ``"effective"`` for the effective-number-of-samples reweighting,
            ``(1-beta)/(1-beta**count)``.
        normalize: Whether to scale the weights to average 1.0, which keeps the
            loss magnitude comparable to an unweighted run.
        beta: Parameter of the ``"effective"`` scheme; closer to 1 gives a
            stronger correction.

    Returns:
        One weight per project label, in ascending label order.

    Raises:
        ValueError: If ``scheme`` is unknown, ``beta`` is outside ``[0, 1)``, or
            a class has no training examples under a count-based scheme. The
            last case is a hard error rather than a silent infinity: both scopes
            are verified to have every class present in every split, so an empty
            class means the manifest is wrong.
    """
    counts = class_counts(targets, num_classes)
    if scheme == "none":
        return tuple(1.0 for _ in range(num_classes))

    if scheme not in ("inverse", "inverse_sqrt", "effective"):
        raise ValueError(
            f"unknown class-weighting scheme {scheme!r}; expected 'none', 'inverse', "
            f"'inverse_sqrt' or 'effective'"
        )

    empty = [label for label, count in counts.items() if count == 0]
    if empty:
        raise ValueError(
            f"cannot weight classes with no training examples: {empty}; every class is "
            f"expected in every split for both scopes, so this indicates a bad manifest"
        )

    if scheme == "effective":
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        raw = [(1.0 - beta) / (1.0 - beta ** counts[label]) for label in range(num_classes)]
    elif scheme == "inverse_sqrt":
        raw = [1.0 / counts[label] ** 0.5 for label in range(num_classes)]
    else:
        raw = [1.0 / counts[label] for label in range(num_classes)]

    if normalize:
        mean = sum(raw) / len(raw)
        raw = [value / mean for value in raw]
    return tuple(raw)


