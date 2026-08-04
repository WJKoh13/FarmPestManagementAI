"""Loader tests that read the real IP102 images and derived manifests.

Every test skips when the dataset or the built manifests are absent, so the
suite still runs on a machine without them. These cover what synthetic fixtures
cannot: the ten real PNG-behind-``.jpg`` files, the real class distributions,
and the sub-160px upscale cohort measured in Phase 4.

Nothing here writes to ``ip102_v1.1``, and nothing reads the test split - Phase 9
is the only phase permitted to.
"""

from __future__ import annotations

import pytest

from farm_pest_ai.config import load_config
from farm_pest_ai.data.dataset import load_image
from farm_pest_ai.data.loaders import build_dataset, build_loaders
from farm_pest_ai.data.transforms import (
    build_transform,
    describe_transform,
    preprocessing_config_from_config,
)

# The loader path needs torch and torchvision, imported lazily by the modules
# under test, so the whole file skips rather than each test.
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

pytestmark = pytest.mark.dataset

#: The ten `.jpg` files that are really PNG, found by the Phase 4 full decode.
#: Seven carry an alpha channel. All belong to IP102 label 56, so they are
#: outside rice10 and present only under full102.
PNG_MASQUERADING_AS_JPG = (
    "40256.jpg", "40314.jpg", "40549.jpg", "40557.jpg", "40563.jpg",
    "40574.jpg", "40591.jpg", "40601.jpg", "40630.jpg", "40577.jpg",
)

#: The seven of those that are RGBA. An untouched one would give the CNN a
#: fourth input channel.
RGBA_FILES = (
    "40549.jpg", "40563.jpg", "40577.jpg",
    "40314.jpg", "40574.jpg", "40591.jpg", "40601.jpg",
)

#: rice10 counts, confirmed in Phase 1 and rebuilt in Phase 4.
RICE10_COUNTS = {"train": 4318, "validation": 721}


@pytest.fixture(scope="module")
def config():
    """The rice10 configuration, or a skip when the dataset is absent."""
    config = load_config("data_rice10.yaml")
    if not config.paths.images_dir.is_dir():
        pytest.skip(f"IP102 images not found at {config.paths.images_dir}")
    if not (config.paths.processed_dir / "rice10" / "train.csv").is_file():
        pytest.skip("derived manifests not built; run scripts/build_manifests.py")
    return config


@pytest.fixture(scope="module")
def full102_config():
    """The full102 configuration, or a skip when the dataset is absent."""
    config = load_config("data_full102.yaml")
    if not config.paths.images_dir.is_dir():
        pytest.skip(f"IP102 images not found at {config.paths.images_dir}")
    if not (config.paths.processed_dir / "full102" / "train.csv").is_file():
        pytest.skip("derived manifests not built; run scripts/build_manifests.py")
    return config


# -- the Phase 4 format anomaly, against the real files -----------------


@pytest.mark.parametrize("filename", PNG_MASQUERADING_AS_JPG)
def test_real_png_behind_jpg_decodes_to_rgb(config, filename: str) -> None:
    """Pillow must dispatch on content, and the result must be three channels."""
    path = config.paths.images_dir / filename
    if not path.is_file():
        pytest.skip(f"{filename} not present")
    image = load_image(path)
    assert image.mode == "RGB"
    assert len(image.getbands()) == 3


@pytest.mark.parametrize("filename", RGBA_FILES)
def test_real_rgba_file_yields_a_three_channel_tensor(config, filename: str) -> None:
    """The fourth channel must be gone by the time the model sees the tensor."""
    path = config.paths.images_dir / filename
    if not path.is_file():
        pytest.skip(f"{filename} not present")
    preprocessing = preprocessing_config_from_config(config)
    tensor = build_transform(preprocessing, "validation")(load_image(path))
    assert tensor.shape == (3, *preprocessing.image_size)


def test_the_known_png_files_are_still_png(config) -> None:
    """If the source tree changed, this fails rather than the training run."""
    from PIL import Image

    checked = 0
    for filename in PNG_MASQUERADING_AS_JPG:
        path = config.paths.images_dir / filename
        if not path.is_file():
            continue
        checked += 1
        with Image.open(path) as image:
            assert image.format == "PNG", f"{filename} is now {image.format}"
    assert checked == len(PNG_MASQUERADING_AS_JPG)


# -- real datasets ------------------------------------------------------


@pytest.mark.parametrize("split", ["train", "validation"])
def test_rice10_dataset_matches_the_audited_counts(config, split: str) -> None:
    assert len(build_dataset(config, split)) == RICE10_COUNTS[split]


def test_rice10_classes_are_all_present(config) -> None:
    dataset = build_dataset(config, "train")
    counts = dataset.class_counts()
    assert len(counts) == 10
    assert all(count > 0 for count in counts.values())


def test_full102_dataset_has_102_classes(full102_config) -> None:
    dataset = build_dataset(full102_config, "validation")
    assert dataset.num_classes == 102
    assert len(dataset) == 7508


def test_real_batches_have_the_right_shape(config) -> None:
    bundle = build_loaders(
        config,
        ("train", "validation"),
        batch_size=8,
    )
    images, labels = next(iter(bundle.loaders["validation"]))
    assert images.shape == (8, 3, 160, 160)
    assert images.dtype is torch.float32
    assert labels.dtype is torch.int64
    assert int(labels.max()) < 10


def test_real_evaluation_pipeline_is_deterministic(config) -> None:
    """Two passes over the same real images must be bit-identical."""
    preprocessing = preprocessing_config_from_config(config)
    transform = build_transform(preprocessing, "validation")
    dataset = build_dataset(config, "validation")
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        image = load_image(dataset.images_dir / dataset.records[index].filename)
        assert torch.equal(transform(image), transform(image))


def test_real_evaluation_pipeline_has_no_random_step(config) -> None:
    dataset = build_dataset(config, "validation")
    steps = describe_transform(dataset.transform)
    assert not [s for s in steps if "Random" in s or "Jitter" in s], steps


def test_real_training_pipeline_actually_varies(config) -> None:
    preprocessing = preprocessing_config_from_config(config)
    transform = build_transform(preprocessing, "train")
    dataset = build_dataset(config, "train")
    image = load_image(dataset.images_dir / dataset.records[0].filename)
    outputs = [transform(image) for _ in range(8)]
    assert any(not torch.equal(outputs[0], other) for other in outputs[1:])


def test_small_images_upscale_to_the_model_input(config) -> None:
    """Phase 4 measured 6.3% of rice10 train below 160px on the short side.

    Those images are upscaled rather than padded or skipped, so they must still
    produce a correctly shaped tensor.
    """
    preprocessing = preprocessing_config_from_config(config)
    transform = build_transform(preprocessing, "validation")
    dataset = build_dataset(config, "validation")

    checked = 0
    for index in range(0, len(dataset), 17):
        path = dataset.images_dir / dataset.records[index].filename
        image = load_image(path)
        if min(image.size) >= 160:
            continue
        checked += 1
        assert transform(image).shape == (3, 160, 160)
        if checked >= 5:
            break
    assert checked > 0, "no sub-160px image found in the sampled validation records"


def test_labels_agree_with_the_manifest(config) -> None:
    """The dataset must not reorder or relabel what the manifest recorded."""
    dataset = build_dataset(config, "validation")
    for index in (0, 100, len(dataset) - 1):
        record = dataset.records[index]
        assert dataset.sample_metadata(index).project_label == record.project_label
        assert dataset[index][1] == record.project_label


def test_scope_is_carried_into_the_run_record(config) -> None:
    described = build_loaders(config, ("train", "validation")).describe()
    assert described["scope"] == "rice10"
    assert described["num_classes"] == 10
    assert described["preprocessing_fingerprint"]
    assert described["splits"]["validation"]["augmented"] is False
