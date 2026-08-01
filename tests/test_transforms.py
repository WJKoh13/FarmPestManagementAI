"""Tests for preprocessing and augmentation pipelines.

These use synthetic images, so they run without ``ip102_v1.1`` present. The
checks that matter most are the two the project's correctness depends on:

* evaluation preprocessing is deterministic and contains no random step;
* every decoded image reaches the model as exactly three channels, including
  the RGBA files Phase 4 found hiding behind a ``.jpg`` extension.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from farm_pest_ai.config import load_config
from farm_pest_ai.data.transforms import (
    EVAL_SPLITS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESSING_VERSION,
    AugmentationConfig,
    PreprocessingConfig,
    TransformError,
    build_transform,
    build_transforms,
    denormalize,
    describe_transform,
    preprocessing_config_from_config,
    preprocessing_fingerprint,
    to_rgb,
)

# Building a pipeline needs torch, torchvision and Pillow; the module under test
# imports them lazily, so the whole file skips rather than each test.
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
PIL_Image = pytest.importorskip("PIL.Image")

#: Steps whose presence in an evaluation pipeline would be a correctness bug.
RANDOM_MARKERS = ("Random", "Jitter", "Erasing")


@pytest.fixture()
def preprocessing() -> PreprocessingConfig:
    """A validated default preprocessing configuration."""
    return PreprocessingConfig().validate()


def make_image(mode: str = "RGB", size: tuple[int, int] = (200, 150)) -> PIL_Image.Image:
    """Build a deterministic synthetic image in the requested mode."""
    image = PIL_Image.new(mode, size)
    pixels = image.load()
    channels = len(image.getbands())
    for x in range(size[0]):
        for y in range(size[1]):
            value = ((x * 7 + y * 13) % 256,) * channels
            pixels[x, y] = value if channels > 1 else value[0]
    return image


# -- RGB conversion -----------------------------------------------------


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "CMYK", "LA"])
def test_to_rgb_always_yields_three_channels(mode: str) -> None:
    """The Phase 4 RGBA finding: a fourth channel must never reach the model."""
    converted = to_rgb(make_image(mode))
    assert converted.mode == "RGB"
    assert len(converted.getbands()) == 3


def test_to_rgb_passes_rgb_through_unchanged() -> None:
    image = make_image("RGB")
    assert to_rgb(image) is image


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_pipeline_output_is_three_channel_for_any_mode(
    preprocessing: PreprocessingConfig, mode: str
) -> None:
    """An RGBA input must not widen the tensor the CNN receives."""
    tensor = build_transform(preprocessing, "validation")(make_image(mode))
    assert tensor.shape == (3, *preprocessing.image_size)


# -- shapes and dtypes --------------------------------------------------


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_pipeline_produces_the_configured_shape(
    preprocessing: PreprocessingConfig, split: str
) -> None:
    tensor = build_transform(preprocessing, split)(make_image())
    assert tensor.shape == (3, 160, 160)
    assert tensor.dtype is torch.float32


@pytest.mark.parametrize("size", [(64, 48), (4000, 3000), (160, 160), (37, 900)])
def test_any_source_size_resizes_to_the_model_input(
    preprocessing: PreprocessingConfig, size: tuple[int, int]
) -> None:
    """Both the upscale cohort and very large images must land on 160x160."""
    tensor = build_transform(preprocessing, "validation")(make_image(size=size))
    assert tensor.shape == (3, 160, 160)


def test_non_square_image_size_is_honoured() -> None:
    preprocessing = PreprocessingConfig(image_size=(120, 200)).validate()
    tensor = build_transform(preprocessing, "validation")(make_image())
    assert tensor.shape == (3, 120, 200)


def test_normalisation_is_applied(preprocessing: PreprocessingConfig) -> None:
    """A mid-grey image must map to the expected normalised value per channel."""
    grey = PIL_Image.new("RGB", (200, 200), (128, 128, 128))
    tensor = build_transform(preprocessing, "validation")(grey)
    for channel in range(3):
        expected = (128 / 255 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert tensor[channel].mean().item() == pytest.approx(expected, abs=1e-4)


def test_denormalize_round_trips(preprocessing: PreprocessingConfig) -> None:
    grey = PIL_Image.new("RGB", (200, 200), (128, 128, 128))
    tensor = build_transform(preprocessing, "validation")(grey)
    restored = denormalize(tensor, preprocessing)
    assert restored.min() >= 0.0 and restored.max() <= 1.0
    assert restored.mean().item() == pytest.approx(128 / 255, abs=1e-3)


def test_denormalize_handles_a_batch(preprocessing: PreprocessingConfig) -> None:
    batch = torch.zeros(4, 3, 8, 8)
    assert denormalize(batch, preprocessing).shape == (4, 3, 8, 8)


# -- determinism: the central guarantee ---------------------------------


@pytest.mark.parametrize("split", EVAL_SPLITS)
def test_evaluation_pipeline_has_no_random_step(
    preprocessing: PreprocessingConfig, split: str
) -> None:
    steps = describe_transform(build_transform(preprocessing, split))
    assert not [s for s in steps if any(m in s for m in RANDOM_MARKERS)], steps


@pytest.mark.parametrize("split", EVAL_SPLITS)
def test_evaluation_pipeline_is_bit_identical_across_calls(
    preprocessing: PreprocessingConfig, split: str
) -> None:
    """Repeated evaluation must give the same tensor, or metrics drift."""
    transform = build_transform(preprocessing, split)
    image = make_image()
    first = transform(image)
    for _ in range(5):
        assert torch.equal(transform(image), first)


def test_validation_and_test_share_one_pipeline(
    preprocessing: PreprocessingConfig,
) -> None:
    """The two evaluation splits must never diverge."""
    image = make_image()
    validation = build_transform(preprocessing, "validation")(image)
    test = build_transform(preprocessing, "test")(image)
    assert torch.equal(validation, test)


def test_evaluation_ignores_augmentation_settings() -> None:
    """Turning augmentation up must not change evaluation preprocessing."""
    calm = PreprocessingConfig().validate()
    wild = replace(
        calm,
        augmentation=AugmentationConfig(
            horizontal_flip=1.0, rotation_degrees=90.0, color_jitter_hue=0.4
        ),
    ).validate()
    image = make_image()
    assert torch.equal(
        build_transform(calm, "validation")(image),
        build_transform(wild, "validation")(image),
    )


# -- augmentation -------------------------------------------------------


def test_training_pipeline_is_random_when_enabled(
    preprocessing: PreprocessingConfig,
) -> None:
    transform = build_transform(preprocessing, "train")
    image = make_image()
    outputs = [transform(image) for _ in range(8)]
    assert any(not torch.equal(outputs[0], other) for other in outputs[1:])


def test_disabling_augmentation_makes_training_deterministic() -> None:
    """The property the loader determinism check relies on."""
    preprocessing = replace(
        PreprocessingConfig(), augmentation=AugmentationConfig(enabled=False)
    ).validate()
    transform = build_transform(preprocessing, "train")
    image = make_image()
    assert torch.equal(transform(image), transform(image))
    steps = describe_transform(transform)
    assert not [s for s in steps if any(m in s for m in RANDOM_MARKERS)], steps


def test_disabled_augmentation_matches_the_evaluation_pipeline() -> None:
    preprocessing = replace(
        PreprocessingConfig(), augmentation=AugmentationConfig(enabled=False)
    ).validate()
    image = make_image()
    assert torch.equal(
        build_transform(preprocessing, "train")(image),
        build_transform(preprocessing, "validation")(image),
    )


def test_zero_probability_steps_are_omitted() -> None:
    """A disabled knob must not leave a no-op random step in the pipeline."""
    preprocessing = replace(
        PreprocessingConfig(),
        augmentation=AugmentationConfig(
            horizontal_flip=0.0,
            vertical_flip=0.0,
            rotation_degrees=0.0,
            color_jitter_brightness=0.0,
            color_jitter_contrast=0.0,
            color_jitter_saturation=0.0,
            color_jitter_hue=0.0,
        ),
    ).validate()
    steps = describe_transform(build_transform(preprocessing, "train"))
    assert "RandomHorizontalFlip" not in steps
    assert "RandomRotation" not in steps
    assert "ColorJitter" not in steps
    # The random resized crop is separately controlled and stays.
    assert "RandomResizedCrop" in steps


def test_random_erasing_is_added_after_normalisation() -> None:
    preprocessing = replace(
        PreprocessingConfig(), augmentation=AugmentationConfig(random_erasing=0.5)
    ).validate()
    steps = describe_transform(build_transform(preprocessing, "train"))
    assert steps[-1] == "RandomErasing"
    assert steps.index("Normalize") < steps.index("RandomErasing")


def test_random_erasing_never_reaches_evaluation() -> None:
    preprocessing = replace(
        PreprocessingConfig(), augmentation=AugmentationConfig(random_erasing=1.0)
    ).validate()
    for split in EVAL_SPLITS:
        assert "RandomErasing" not in describe_transform(
            build_transform(preprocessing, split)
        )


def test_build_transforms_covers_every_split(
    preprocessing: PreprocessingConfig,
) -> None:
    built = build_transforms(preprocessing)
    assert set(built) == {"train", "validation", "test"}


def test_unknown_split_is_rejected(preprocessing: PreprocessingConfig) -> None:
    with pytest.raises(TransformError, match=r"unknown split"):
        build_transform(preprocessing, "holdout")


# -- shorter-side resize ------------------------------------------------


def test_resize_shorter_side_centre_crops() -> None:
    preprocessing = PreprocessingConfig(resize_shorter_side=180).validate()
    steps = describe_transform(build_transform(preprocessing, "validation"))
    assert steps == ("Lambda", "Resize", "CenterCrop", "ToTensor", "Normalize")
    tensor = build_transform(preprocessing, "validation")(make_image())
    assert tensor.shape == (3, 160, 160)


def test_resize_shorter_side_below_the_crop_is_rejected() -> None:
    """A shorter side under the crop size would pad or fail at run time."""
    with pytest.raises(TransformError, match=r"smaller than the crop"):
        PreprocessingConfig(resize_shorter_side=100).validate()


# -- validation ---------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"image_size": (0, 160)}, r"must be positive"),
        ({"image_size": (-1, -1)}, r"must be positive"),
        ({"interpolation": "spline"}, r"unknown interpolation"),
        ({"resize_shorter_side": 0}, r"must be positive"),
        ({"mean": (0.5, 0.5)}, r"three channel values"),
        ({"std": (0.0, 0.2, 0.2)}, r"std values must be positive"),
    ],
)
def test_invalid_preprocessing_is_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(TransformError, match=match):
        PreprocessingConfig(**kwargs).validate()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"horizontal_flip": 1.5}, r"probability in \[0, 1\]"),
        ({"vertical_flip": -0.1}, r"probability in \[0, 1\]"),
        ({"random_erasing": 2.0}, r"probability in \[0, 1\]"),
        ({"rotation_degrees": -5.0}, r"must be non-negative"),
        ({"rotation_degrees": 400.0}, r"must be <= 180"),
        ({"color_jitter_hue": 0.8}, r"hue must be <= 0\.5"),
        ({"color_jitter_brightness": -1.0}, r"must be non-negative"),
        ({"scale": (0.9, 0.2)}, r"lower bound 0\.9 exceeds upper bound"),
        ({"scale": (0.5, 1.5)}, r"area fraction and must be <= 1\.0"),
        ({"ratio": (0.0, 2.0)}, r"bounds must be positive"),
    ],
)
def test_invalid_augmentation_is_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(TransformError, match=match):
        AugmentationConfig(**kwargs).validate()


# -- fingerprinting -----------------------------------------------------


def test_fingerprint_is_stable(preprocessing: PreprocessingConfig) -> None:
    assert preprocessing.fingerprint == PreprocessingConfig().validate().fingerprint


def test_fingerprint_changes_with_preprocessing(
    preprocessing: PreprocessingConfig,
) -> None:
    """A silent preprocessing change must be detectable after the fact."""
    for changed in (
        replace(preprocessing, image_size=(224, 224)),
        replace(preprocessing, interpolation="bicubic"),
        replace(preprocessing, mean=(0.5, 0.5, 0.5)),
        replace(preprocessing, resize_shorter_side=200),
        replace(
            preprocessing,
            augmentation=replace(preprocessing.augmentation, horizontal_flip=0.25),
        ),
    ):
        assert changed.validate().fingerprint != preprocessing.fingerprint


def test_fingerprint_is_a_short_hex_digest(preprocessing: PreprocessingConfig) -> None:
    digest = preprocessing_fingerprint(preprocessing)
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


# -- configuration integration ------------------------------------------


def test_preprocessing_comes_from_the_shipped_config() -> None:
    config = load_config("base.yaml")
    preprocessing = preprocessing_config_from_config(config)
    assert preprocessing.image_size == (160, 160)
    assert preprocessing.interpolation in ("bilinear", "bicubic", "nearest", "lanczos")
    assert preprocessing.version == PREPROCESSING_VERSION


def test_image_size_override_flows_into_preprocessing() -> None:
    config = load_config("base.yaml", cli_overrides=["dataset.image_size=[96, 96]"])
    assert preprocessing_config_from_config(config).image_size == (96, 96)


def test_augmentation_override_flows_into_preprocessing() -> None:
    config = load_config(
        "base.yaml", cli_overrides=["preprocessing.augmentation.horizontal_flip=0.0"]
    )
    assert preprocessing_config_from_config(config).augmentation.horizontal_flip == 0.0


def test_malformed_config_section_is_rejected() -> None:
    config = load_config("base.yaml", cli_overrides=["preprocessing.mean=[0.5, 0.5]"])
    with pytest.raises(TransformError, match=r"three numbers"):
        preprocessing_config_from_config(config)


def test_invalid_interpolation_in_config_is_rejected() -> None:
    config = load_config("base.yaml", cli_overrides=["preprocessing.interpolation=spline"])
    with pytest.raises(TransformError, match=r"unknown interpolation"):
        preprocessing_config_from_config(config)
