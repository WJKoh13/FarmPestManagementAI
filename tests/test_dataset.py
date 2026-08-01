"""Tests for :class:`~farm_pest_ai.data.dataset.PestImageDataset`.

Synthetic images and manifests throughout, so the suite runs without
``ip102_v1.1``. The scope guard and the class-weight rules get the most
attention: both are places where a silent mistake would corrupt training
without raising anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from farm_pest_ai.data.dataset import (
    DatasetError,
    PestImageDataset,
    class_counts,
    class_weights,
    load_image,
)
from farm_pest_ai.data.manifests import ManifestRecord
from farm_pest_ai.scopes import FULL102, RICE10

# Decoding needs Pillow, which the module under test imports lazily.
PIL_Image = pytest.importorskip("PIL.Image")


def write_image(
    path: Path, mode: str = "RGB", size: tuple[int, int] = (64, 48)
) -> Path:
    """Write a small synthetic image in ``mode`` and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build in RGB, then convert: Pillow picks the right fill for each mode.
    PIL_Image.new("RGB", size, (120, 90, 60)).convert(mode).save(path)
    return path


@pytest.fixture()
def images_dir(tmp_path: Path) -> Path:
    """A directory holding one image per rice10 project label."""
    directory = tmp_path / "images"
    for label in range(RICE10.num_classes):
        write_image(directory / f"{label:05d}.jpg")
    return directory


@pytest.fixture()
def records() -> tuple[ManifestRecord, ...]:
    """One record per rice10 project label, in project-label order."""
    return tuple(
        ManifestRecord(
            filename=f"{label:05d}.jpg",
            ip102_label=RICE10.to_original_label(label),
            project_label=label,
            class_name=f"class {label}",
            split="train",
        )
        for label in range(RICE10.num_classes)
    )


@pytest.fixture()
def dataset(records, images_dir: Path) -> PestImageDataset:
    """A ten-record rice10 training dataset with no transform."""
    return PestImageDataset(records, images_dir, RICE10, "train")


# -- decoding -----------------------------------------------------------


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_load_image_always_returns_rgb(tmp_path: Path, mode: str) -> None:
    """Phase 4's RGBA finding, enforced at the decode boundary."""
    path = write_image(tmp_path / f"{mode}.png", mode=mode)
    assert load_image(path).mode == "RGB"


def test_load_image_dispatches_on_content_not_extension(tmp_path: Path) -> None:
    """The ten real files are PNG behind a .jpg name; both must decode."""
    path = tmp_path / "actually_png.jpg"
    PIL_Image.new("RGBA", (32, 32), (10, 20, 30, 255)).save(path, format="PNG")
    image = load_image(path)
    assert image.mode == "RGB"
    assert len(image.getbands()) == 3


def test_load_image_reports_the_path_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not an image")
    with pytest.raises(DatasetError, match=r"broken\.jpg"):
        load_image(path)


def test_load_image_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"failed to decode"):
        load_image(tmp_path / "absent.jpg")


def test_truncated_image_is_reported(tmp_path: Path) -> None:
    """A half-written JPEG must raise, not yield a partial tensor."""
    full = write_image(tmp_path / "full.jpg", size=(200, 200))
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(full.read_bytes()[: 40])
    with pytest.raises(DatasetError):
        load_image(truncated)


# -- dataset protocol ---------------------------------------------------


def test_length_matches_the_records(dataset: PestImageDataset) -> None:
    assert len(dataset) == RICE10.num_classes


def test_getitem_returns_an_image_and_an_int_label(dataset: PestImageDataset) -> None:
    image, label = dataset[3]
    assert image.mode == "RGB"
    assert label == 3
    assert isinstance(label, int)


def test_transform_is_applied(records, images_dir: Path) -> None:
    dataset = PestImageDataset(
        records, images_dir, RICE10, "train", transform=lambda image: image.size
    )
    assert dataset[0][0] == (64, 48)


def test_targets_follow_dataset_order(dataset: PestImageDataset) -> None:
    assert dataset.targets == tuple(range(RICE10.num_classes))


def test_num_classes_comes_from_the_scope(dataset: PestImageDataset) -> None:
    assert dataset.num_classes == 10
    assert dataset.scope is RICE10


def test_sample_metadata_needs_no_decode(dataset: PestImageDataset) -> None:
    sample = dataset.sample_metadata(2)
    assert sample.project_label == 2
    assert sample.ip102_label == RICE10.to_original_label(2)
    assert sample.filename == "00002.jpg"
    assert sample.path.name == "00002.jpg"
    assert sample.to_dict()["split"] == "train"


def test_class_names_are_indexed_by_project_label(dataset: PestImageDataset) -> None:
    assert dataset.class_names == tuple(f"class {i}" for i in range(10))


def test_class_names_raise_when_a_class_is_absent(records, images_dir: Path) -> None:
    dataset = PestImageDataset(records[:5], images_dir, RICE10, "train")
    with pytest.raises(DatasetError, match=r"no records for project label"):
        _ = dataset.class_names


def test_describe_is_json_serialisable(dataset: PestImageDataset) -> None:
    import json

    described = dataset.describe()
    assert described["scope"] == "rice10"
    assert described["num_classes"] == 10
    assert described["records"] == 10
    json.dumps(described)


def test_repr_names_the_scope_and_split(dataset: PestImageDataset) -> None:
    text = repr(dataset)
    assert "rice10" in text and "train" in text


# -- guards -------------------------------------------------------------

def test_empty_records_are_rejected(images_dir: Path) -> None:
    with pytest.raises(DatasetError, match=r"no records"):
        PestImageDataset([], images_dir, RICE10, "train")


def test_unknown_split_is_rejected(records, images_dir: Path) -> None:
    with pytest.raises(DatasetError, match=r"unknown split"):
        PestImageDataset(records, images_dir, RICE10, "holdout")


def test_out_of_range_label_is_rejected(records, images_dir: Path) -> None:
    """A full102 label loaded under rice10 must raise, not silently truncate."""
    bad = (
        *records,
        ManifestRecord(
            filename="99999.jpg",
            ip102_label=101,
            project_label=101,
            class_name="wrong scope",
            split="train",
        ),
    )
    with pytest.raises(DatasetError, match=r"outside 0\.\.9 for scope 'rice10'"):
        PestImageDataset(bad, images_dir, RICE10, "train")


def test_verify_files_reports_missing_images(records, tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"referenced image\(s\) missing"):
        PestImageDataset(
            records, tmp_path / "empty", RICE10, "train", verify_files=True
        )


def test_verify_files_is_off_by_default(records, tmp_path: Path) -> None:
    """Construction must not stat thousands of files every epoch."""
    dataset = PestImageDataset(records, tmp_path / "empty", RICE10, "train")
    assert len(dataset) == RICE10.num_classes


# -- class counts and weights -------------------------------------------


def test_class_counts_include_empty_classes() -> None:
    assert class_counts([0, 0, 2], 4) == {0: 2, 1: 0, 2: 1, 3: 0}


def test_class_counts_reject_out_of_range_labels() -> None:
    with pytest.raises(ValueError, match=r"outside 0\.\.2"):
        class_counts([0, 5], 3)


def test_dataset_class_counts_match(dataset: PestImageDataset) -> None:
    assert dataset.class_counts() == dict.fromkeys(range(10), 1)


def test_none_scheme_gives_uniform_weights() -> None:
    assert class_weights([0, 0, 0, 1], 2, scheme="none") == (1.0, 1.0)


def test_inverse_weighting_favours_the_rare_class() -> None:
    weights = class_weights([0] * 90 + [1] * 10, 2, scheme="inverse")
    assert weights[1] > weights[0]
    # 9x rarer, so 9x the weight.
    assert weights[1] / weights[0] == pytest.approx(9.0)


def test_inverse_sqrt_is_gentler_than_inverse() -> None:
    targets = [0] * 90 + [1] * 10
    inverse = class_weights(targets, 2, scheme="inverse")
    sqrt = class_weights(targets, 2, scheme="inverse_sqrt")
    assert sqrt[1] / sqrt[0] < inverse[1] / inverse[0]


def test_effective_weighting_favours_the_rare_class() -> None:
    weights = class_weights([0] * 90 + [1] * 10, 2, scheme="effective")
    assert weights[1] > weights[0]


def test_normalised_weights_average_to_one() -> None:
    weights = class_weights([0] * 90 + [1] * 10, 2, scheme="inverse")
    assert sum(weights) / len(weights) == pytest.approx(1.0)


def test_unnormalised_weights_are_raw_reciprocals() -> None:
    weights = class_weights([0] * 4 + [1] * 2, 2, scheme="inverse", normalize=False)
    assert weights == pytest.approx((0.25, 0.5))


def test_balanced_classes_get_equal_weights() -> None:
    weights = class_weights([0, 0, 1, 1], 2, scheme="inverse")
    assert weights == pytest.approx((1.0, 1.0))


def test_unknown_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"unknown class-weighting scheme"):
        class_weights([0, 1], 2, scheme="magic")


def test_empty_class_is_a_hard_error() -> None:
    """An unweightable class means a bad manifest, not an infinite weight."""
    with pytest.raises(ValueError, match=r"no training examples"):
        class_weights([0, 0], 2, scheme="inverse")


def test_effective_scheme_rejects_a_bad_beta() -> None:
    with pytest.raises(ValueError, match=r"beta must be in"):
        class_weights([0, 1], 2, scheme="effective", beta=1.0)


def test_weights_cover_all_102_classes() -> None:
    """The scope drives the length, so full102 must yield 102 weights."""
    weights = class_weights(list(range(102)), FULL102.num_classes, scheme="inverse")
    assert len(weights) == 102
