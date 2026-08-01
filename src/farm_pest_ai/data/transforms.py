"""Image preprocessing and augmentation pipelines.

This module owns every pixel-level decision between a file on disk and the
tensor the CNN consumes. Three constraints from earlier phases shape it:

Explicit RGB conversion
    Phase 4 found ten ``.jpg`` files that are really PNG, seven of them RGBA
    (all IP102 label 56). Left alone, an RGBA image would hand the model a
    fourth input channel. :func:`to_rgb` converts unconditionally, and the
    decode path in :mod:`farm_pest_ai.data.dataset` dispatches on file
    *content*, never on the extension.

Deterministic evaluation
    Only the training pipeline randomises. Validation and test share one
    resize-and-normalise path, so a validation number computed today is
    comparable with one computed in Phase 9.

A recorded preprocessing version
    :func:`preprocessing_fingerprint` hashes the resolved pipeline description.
    Checkpoints carry it, so a model evaluated under changed preprocessing is
    detectable rather than silently mis-scored.

Only :mod:`torchvision.transforms` and primitive operations are used;
``torchvision.models`` and pretrained weights remain prohibited.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from PIL.Image import Image as PILImage

__all__ = [
    "EVAL_SPLITS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INTERPOLATION_MODES",
    "PREPROCESSING_VERSION",
    "AugmentationConfig",
    "PreprocessingConfig",
    "TransformError",
    "build_transform",
    "build_transforms",
    "denormalize",
    "describe_transform",
    "preprocessing_config_from_config",
    "preprocessing_fingerprint",
    "to_rgb",
]

#: Bumped whenever a change here alters the tensor a given image produces.
#: Recorded with every checkpoint and every metrics file.
PREPROCESSING_VERSION = "1.0.0"

#: Splits that must never see random augmentation.
EVAL_SPLITS: tuple[str, ...] = ("validation", "test")

#: Channel statistics. These are the standard ImageNet constants, used purely as
#: fixed normalisation numbers - no pretrained weights are involved, which the
#: CNN protocol prohibits. Phase 7 may replace them with statistics measured on
#: the training split; doing so requires bumping PREPROCESSING_VERSION.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Interpolation modes selectable from configuration.
INTERPOLATION_MODES: tuple[str, ...] = ("bilinear", "bicubic", "nearest", "lanczos")


class TransformError(ValueError):
    """Raised when a preprocessing configuration is malformed or unusable."""


def to_rgb(image: PILImage) -> PILImage:
    """Return ``image`` in three-channel RGB mode.

    Converts unconditionally unless the image is already ``RGB``. Palette and
    RGBA images are composited by Pillow's own conversion, which discards the
    alpha channel; that is the intended behaviour, since the model input is
    three channels and the ten affected files are ordinary photographs with an
    unused alpha plane.
    """
    return image if image.mode == "RGB" else image.convert("RGB")


# -- configuration ------------------------------------------------------


@dataclass(frozen=True)
class AugmentationConfig:
    """Random augmentation applied to the **training split only**.

    Every field defaults to a conservative value. Augmentation strength is a
    Phase 7 tuning decision made against validation macro F1; Phase 5 only
    guarantees that the knobs exist, are recorded, and are never applied to
    evaluation data.

    Attributes:
        enabled: Master switch. When false the training pipeline degrades to the
            deterministic evaluation pipeline, which is what the smoke test and
            the loader-determinism check use.
        random_resized_crop: Whether to sample a random area/aspect crop instead
            of the deterministic resize.
        scale: ``(min, max)`` area fraction for the random resized crop.
        ratio: ``(min, max)`` aspect-ratio bounds for the random resized crop.
        horizontal_flip: Probability of a horizontal flip. Pest photographs have
            no canonical handedness, so this is safe.
        vertical_flip: Probability of a vertical flip. Left at zero by default:
            most images are ground-referenced and an upside-down insect is not a
            realistic input.
        rotation_degrees: Maximum absolute rotation in degrees.
        color_jitter_brightness: Brightness jitter magnitude.
        color_jitter_contrast: Contrast jitter magnitude.
        color_jitter_saturation: Saturation jitter magnitude.
        color_jitter_hue: Hue jitter magnitude. Kept small; pest identification
            leans on colour, so large hue shifts would destroy signal.
        random_erasing: Probability of erasing a random patch after
            normalisation.
    """

    enabled: bool = True
    random_resized_crop: bool = True
    scale: tuple[float, float] = (0.6, 1.0)
    ratio: tuple[float, float] = (0.75, 1.3333)
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    rotation_degrees: float = 15.0
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.02
    random_erasing: float = 0.0

    def validate(self) -> None:
        """Check ranges and ordering.

        Raises:
            TransformError: If a probability is outside ``[0, 1]``, a jitter
                magnitude is negative, hue exceeds Pillow's ``0.5`` limit, or a
                ``(min, max)`` pair is inverted or non-positive.
        """
        probabilities = {
            "horizontal_flip": self.horizontal_flip,
            "vertical_flip": self.vertical_flip,
            "random_erasing": self.random_erasing,
        }
        for name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise TransformError(
                    f"augmentation.{name} must be a probability in [0, 1], got {value}"
                )

        magnitudes = {
            "color_jitter_brightness": self.color_jitter_brightness,
            "color_jitter_contrast": self.color_jitter_contrast,
            "color_jitter_saturation": self.color_jitter_saturation,
            "color_jitter_hue": self.color_jitter_hue,
            "rotation_degrees": self.rotation_degrees,
        }
        for name, value in magnitudes.items():
            if value < 0:
                raise TransformError(
                    f"augmentation.{name} must be non-negative, got {value}"
                )
        if self.color_jitter_hue > 0.5:
            raise TransformError(
                f"augmentation.color_jitter_hue must be <= 0.5, got "
                f"{self.color_jitter_hue}"
            )
        if self.rotation_degrees > 180:
            raise TransformError(
                f"augmentation.rotation_degrees must be <= 180, got "
                f"{self.rotation_degrees}"
            )

        for name, pair in (("scale", self.scale), ("ratio", self.ratio)):
            if len(pair) != 2:
                raise TransformError(
                    f"augmentation.{name} must be two numbers, got {pair!r}"
                )
            low, high = pair
            if low <= 0 or high <= 0:
                raise TransformError(
                    f"augmentation.{name} bounds must be positive, got {pair!r}"
                )
            if low > high:
                raise TransformError(
                    f"augmentation.{name} lower bound {low} exceeds upper bound {high}"
                )
        if self.scale[1] > 1.0:
            raise TransformError(
                f"augmentation.scale is an area fraction and must be <= 1.0, "
                f"got {self.scale[1]}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        payload = asdict(self)
        payload["scale"] = list(self.scale)
        payload["ratio"] = list(self.ratio)
        return payload


@dataclass(frozen=True)
class PreprocessingConfig:
    """The complete, resolved preprocessing description for one run.

    Attributes:
        image_size: ``(height, width)`` of the tensor handed to the model.
        interpolation: Resampling filter name, one of
            :data:`INTERPOLATION_MODES`.
        resize_shorter_side: When set, evaluation resizes the shorter side to
            this length and centre-crops to ``image_size``, preserving aspect
            ratio. When ``None``, evaluation resizes directly to ``image_size``,
            which distorts aspect ratio but keeps the whole frame.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation standard deviation.
        augmentation: Training-only random augmentation.
        version: Preprocessing version stamp recorded with checkpoints.
    """

    image_size: tuple[int, int] = (160, 160)
    interpolation: str = "bilinear"
    resize_shorter_side: int | None = None
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    version: str = PREPROCESSING_VERSION

    def validate(self) -> PreprocessingConfig:
        """Validate every field.

        Returns:
            ``self``, so the call can be chained.

        Raises:
            TransformError: On the first inconsistency found.
        """
        if len(self.image_size) != 2:
            raise TransformError(
                f"image_size must be (height, width), got {self.image_size!r}"
            )
        if any(side <= 0 for side in self.image_size):
            raise TransformError(f"image_size must be positive, got {self.image_size!r}")
        if self.interpolation not in INTERPOLATION_MODES:
            raise TransformError(
                f"unknown interpolation {self.interpolation!r}; expected one of "
                f"{list(INTERPOLATION_MODES)}"
            )
        if self.resize_shorter_side is not None:
            if self.resize_shorter_side <= 0:
                raise TransformError(
                    f"resize_shorter_side must be positive, got "
                    f"{self.resize_shorter_side}"
                )
            if self.resize_shorter_side < max(self.image_size):
                raise TransformError(
                    f"resize_shorter_side={self.resize_shorter_side} is smaller than "
                    f"the crop {self.image_size}; the centre crop would pad or fail"
                )
        for name, values in (("mean", self.mean), ("std", self.std)):
            if len(values) != 3:
                raise TransformError(
                    f"{name} must have three channel values, got {values!r}"
                )
        if any(s <= 0 for s in self.std):
            raise TransformError(f"std values must be positive, got {self.std!r}")
        self.augmentation.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping, used for the fingerprint."""
        return {
            "version": self.version,
            "image_size": list(self.image_size),
            "interpolation": self.interpolation,
            "resize_shorter_side": self.resize_shorter_side,
            "mean": list(self.mean),
            "std": list(self.std),
            "augmentation": self.augmentation.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        """Short stable hash of this configuration."""
        return preprocessing_fingerprint(self)


def _as_pair(value: Any, name: str) -> tuple[float, float]:
    """Coerce a two-element sequence of numbers into a tuple."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
            return (float(values[0]), float(values[1]))
    raise TransformError(f"{name} must be two numbers, got {value!r}")


def _as_triple(value: Any, name: str) -> tuple[float, float, float]:
    """Coerce a three-element sequence of numbers into a tuple."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) == 3 and all(isinstance(v, (int, float)) for v in values):
            return (float(values[0]), float(values[1]), float(values[2]))
    raise TransformError(f"{name} must be three numbers, got {value!r}")


def _augmentation_from_mapping(section: Mapping[str, Any]) -> AugmentationConfig:
    """Build an :class:`AugmentationConfig` from a configuration mapping."""
    defaults = AugmentationConfig()
    return AugmentationConfig(
        enabled=bool(section.get("enabled", defaults.enabled)),
        random_resized_crop=bool(
            section.get("random_resized_crop", defaults.random_resized_crop)
        ),
        scale=_as_pair(section.get("scale", defaults.scale), "augmentation.scale"),
        ratio=_as_pair(section.get("ratio", defaults.ratio), "augmentation.ratio"),
        horizontal_flip=float(
            section.get("horizontal_flip", defaults.horizontal_flip)
        ),
        vertical_flip=float(section.get("vertical_flip", defaults.vertical_flip)),
        rotation_degrees=float(
            section.get("rotation_degrees", defaults.rotation_degrees)
        ),
        color_jitter_brightness=float(
            section.get("color_jitter_brightness", defaults.color_jitter_brightness)
        ),
        color_jitter_contrast=float(
            section.get("color_jitter_contrast", defaults.color_jitter_contrast)
        ),
        color_jitter_saturation=float(
            section.get("color_jitter_saturation", defaults.color_jitter_saturation)
        ),
        color_jitter_hue=float(
            section.get("color_jitter_hue", defaults.color_jitter_hue)
        ),
        random_erasing=float(section.get("random_erasing", defaults.random_erasing)),
    )


def preprocessing_config_from_config(config: Any) -> PreprocessingConfig:
    """Build a :class:`PreprocessingConfig` from a resolved :class:`Config`.

    Reads ``dataset.image_size`` and the ``preprocessing`` section. The
    ``preprocessing_version`` recorded in the result comes from configuration
    when present, so a run can be pinned to an older stamp deliberately; a
    mismatch with :data:`PREPROCESSING_VERSION` is surfaced by
    ``scripts/verify_loader.py`` rather than silently accepted.

    Args:
        config: A :class:`farm_pest_ai.config.Config`.

    Returns:
        The validated preprocessing configuration.

    Raises:
        TransformError: If any value is malformed.
    """
    dataset = config.dataset
    section = config.section("preprocessing")
    defaults = PreprocessingConfig()

    shorter = section.get("resize_shorter_side", defaults.resize_shorter_side)
    if shorter is not None and (isinstance(shorter, bool) or not isinstance(shorter, int)):
        raise TransformError(
            f"preprocessing.resize_shorter_side must be an integer or null, "
            f"got {shorter!r}"
        )

    augmentation_section = section.get("augmentation", {})
    if not isinstance(augmentation_section, Mapping):
        raise TransformError(
            f"preprocessing.augmentation must be a mapping, got "
            f"{augmentation_section!r}"
        )

    resolved = PreprocessingConfig(
        image_size=dataset.image_size,
        interpolation=str(section.get("interpolation", defaults.interpolation)),
        resize_shorter_side=shorter,
        mean=_as_triple(section.get("mean", defaults.mean), "preprocessing.mean"),
        std=_as_triple(section.get("std", defaults.std), "preprocessing.std"),
        augmentation=_augmentation_from_mapping(augmentation_section),
        version=str(dataset.preprocessing_version),
    )
    return resolved.validate()


def preprocessing_fingerprint(preprocessing: PreprocessingConfig) -> str:
    """Return a short stable hash of a preprocessing configuration.

    Two runs producing the same fingerprint applied identical preprocessing.
    The digest is over the canonicalised JSON description, so it is stable
    across processes and platforms, unlike :func:`hash`.
    """
    payload = json.dumps(preprocessing.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# -- pipeline construction ----------------------------------------------


def _interpolation_mode(name: str) -> Any:
    """Translate an interpolation name into a torchvision enum member."""
    from torchvision.transforms import InterpolationMode

    modes = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
        "lanczos": InterpolationMode.LANCZOS,
    }
    try:
        return modes[name]
    except KeyError:
        raise TransformError(
            f"unknown interpolation {name!r}; expected one of {sorted(modes)}"
        ) from None


def build_transform(preprocessing: PreprocessingConfig, split: str) -> Any:
    """Build the transform pipeline for one split.

    The training pipeline randomises; ``validation`` and ``test`` share one
    deterministic pipeline, so the two are never accidentally different.
    Augmentation for training is disabled entirely when
    ``augmentation.enabled`` is false, which makes the training pipeline
    byte-identical to the evaluation one - the property the determinism check in
    ``scripts/verify_loader.py`` relies on.

    Args:
        preprocessing: Validated preprocessing configuration.
        split: ``"train"``, ``"validation"`` or ``"test"``.

    Returns:
        A ``torchvision.transforms.Compose``.

    Raises:
        TransformError: If ``split`` is unknown.
    """
    from torchvision import transforms

    if split not in ("train", *EVAL_SPLITS):
        raise TransformError(
            f"unknown split {split!r}; expected 'train', 'validation' or 'test'"
        )
    preprocessing.validate()
    interpolation = _interpolation_mode(preprocessing.interpolation)
    height, width = preprocessing.image_size
    augmentation = preprocessing.augmentation

    steps: list[Any] = [
        # First and unconditional: Phase 4 found RGBA files behind .jpg names.
        transforms.Lambda(to_rgb)
    ]

    if split == "train" and augmentation.enabled:
        if augmentation.random_resized_crop:
            steps.append(
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=augmentation.scale,
                    ratio=augmentation.ratio,
                    interpolation=interpolation,
                    antialias=True,
                )
            )
        else:
            steps.append(
                transforms.Resize(
                    (height, width), interpolation=interpolation, antialias=True
                )
            )
        if augmentation.horizontal_flip > 0:
            steps.append(transforms.RandomHorizontalFlip(augmentation.horizontal_flip))
        if augmentation.vertical_flip > 0:
            steps.append(transforms.RandomVerticalFlip(augmentation.vertical_flip))
        if augmentation.rotation_degrees > 0:
            steps.append(
                transforms.RandomRotation(
                    augmentation.rotation_degrees, interpolation=interpolation
                )
            )
        jitter = (
            augmentation.color_jitter_brightness,
            augmentation.color_jitter_contrast,
            augmentation.color_jitter_saturation,
            augmentation.color_jitter_hue,
        )
        if any(value > 0 for value in jitter):
            steps.append(
                transforms.ColorJitter(
                    brightness=augmentation.color_jitter_brightness,
                    contrast=augmentation.color_jitter_contrast,
                    saturation=augmentation.color_jitter_saturation,
                    hue=augmentation.color_jitter_hue,
                )
            )
    else:
        if preprocessing.resize_shorter_side is not None:
            steps.append(
                transforms.Resize(
                    preprocessing.resize_shorter_side,
                    interpolation=interpolation,
                    antialias=True,
                )
            )
            steps.append(transforms.CenterCrop((height, width)))
        else:
            steps.append(
                transforms.Resize(
                    (height, width), interpolation=interpolation, antialias=True
                )
            )

    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=preprocessing.mean, std=preprocessing.std))

    # Erasing operates on the normalised tensor, so it comes last and only for
    # an augmented training pipeline.
    if split == "train" and augmentation.enabled and augmentation.random_erasing > 0:
        steps.append(transforms.RandomErasing(p=augmentation.random_erasing))

    return transforms.Compose(steps)


def build_transforms(preprocessing: PreprocessingConfig) -> dict[str, Any]:
    """Build the transform pipeline for every split.

    Returns:
        A mapping from split name to pipeline. ``validation`` and ``test`` are
        separately constructed but describe identically, which the tests pin.
    """
    return {
        split: build_transform(preprocessing, split)
        for split in ("train", *EVAL_SPLITS)
    }


def describe_transform(transform: Any) -> tuple[str, ...]:
    """Return the class names of a pipeline's steps, in order.

    Used by tests and by the loader report to assert that evaluation contains no
    random step, without depending on ``repr`` formatting.
    """
    steps = getattr(transform, "transforms", None)
    if steps is None:
        return (type(transform).__name__,)
    return tuple(type(step).__name__ for step in steps)


def denormalize(tensor: Any, preprocessing: PreprocessingConfig) -> Any:
    """Invert :class:`~torchvision.transforms.Normalize` for inspection.

    Only used for saving sample grids and for the visual spot-check in
    ``scripts/verify_loader.py``; it is never part of the training path.

    Args:
        tensor: A ``(C, H, W)`` or ``(N, C, H, W)`` normalised tensor.
        preprocessing: The configuration the tensor was produced under.

    Returns:
        The tensor with channel statistics reapplied, clamped to ``[0, 1]``.
    """
    import torch

    mean = torch.tensor(preprocessing.mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(preprocessing.std, dtype=tensor.dtype, device=tensor.device)
    shape = (1, 3, 1, 1) if tensor.ndim == 4 else (3, 1, 1)
    return (tensor * std.view(shape) + mean.view(shape)).clamp_(0.0, 1.0)
