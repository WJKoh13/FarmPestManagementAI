"""Tests for the IP102 detection subset and the E4/E5 cropping experiments.

The properties pinned here are the ones that make the experiment interpretable
rather than merely runnable. In particular:

* the box coordinate convention is ``[x1, y1, x2, y2]``, verified against real
  image dimensions rather than assumed;
* padding is measured against the box's own size and clamps to the frame;
* the crop and full-frame arms consume **identical** records, so their
  difference is attributable to the image region alone;
* a missing box is a hard error inside the crop transform, never a silent
  fall back to the full frame.

Tests needing the real dataset skip cleanly when it is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from farm_pest_ai.config import load_config
from farm_pest_ai.data.detection import (
    DEFAULT_PADDING,
    BoundingBox,
    BoxCropTransform,
    DetectionDataError,
    box_statistics,
    build_detection_records,
    crop_with_padding,
    detection_root,
    load_boxes,
    load_splits,
    pad_and_clamp,
    partition_records,
    scope_suffix,
)
from farm_pest_ai.data.manifests import ManifestRecord
from farm_pest_ai.scopes import (
    DETECTION_SCOPES,
    get_scope,
    is_detection_scope,
    num_classes_for,
)

#: The four arm configs, paired as (control, treatment).
ARM_PAIRS = (
    ("exp_det_top10_e4a_fullframe.yaml", "exp_det_top10_e4b_crop15.yaml", "det_top10"),
    ("exp_det_top15_e5a_fullframe.yaml", "exp_det_top15_e5b_crop15.yaml", "det_top15"),
)


# -- scopes -------------------------------------------------------------


def test_detection_scopes_derive_num_classes() -> None:
    """num_classes comes from the scope, never from configuration."""
    assert num_classes_for("det_top10") == 10
    assert num_classes_for("det_top15") == 15


def test_detection_scopes_are_flagged() -> None:
    """Detection scopes are distinguishable from classification scopes."""
    assert is_detection_scope("det_top10")
    assert is_detection_scope("det_top15")
    assert not is_detection_scope("rice10")
    assert not is_detection_scope("full102")
    assert set(DETECTION_SCOPES) == {"det_top10", "det_top15"}


def test_classification_scopes_are_unchanged() -> None:
    """Adding detection scopes must not disturb the existing ones."""
    assert num_classes_for("rice10") == 10
    assert num_classes_for("full102") == 102
    assert get_scope("rice10").original_labels == (0, 1, 3, 4, 5, 7, 8, 9, 10, 11)


def test_scope_suffix_rejects_classification_scopes() -> None:
    """A classification scope has no detection box file."""
    assert scope_suffix("det_top10") == "top10"
    assert scope_suffix("det_top15") == "top15"
    with pytest.raises(DetectionDataError, match="not a detection scope"):
        scope_suffix("rice10")


# -- padding geometry ---------------------------------------------------


def test_padding_is_relative_to_box_size() -> None:
    """15% padding grows a box by 15% of its own width and height per side."""
    box = BoundingBox(100.0, 100.0, 200.0, 300.0)  # 100 wide, 200 tall
    padded = pad_and_clamp(box, 1000, 1000, 0.15)
    assert padded.x1 == pytest.approx(85.0)  # 100 - 15
    assert padded.x2 == pytest.approx(215.0)  # 200 + 15
    assert padded.y1 == pytest.approx(70.0)  # 100 - 30
    assert padded.y2 == pytest.approx(330.0)  # 300 + 30


def test_padding_clamps_to_image_bounds() -> None:
    """A box against the edge cannot pad outside the image."""
    box = BoundingBox(0.0, 0.0, 50.0, 50.0)
    padded = pad_and_clamp(box, 100, 100, 0.15)
    assert padded.x1 == 0.0
    assert padded.y1 == 0.0
    assert padded.x2 == pytest.approx(57.5)
    assert padded.y2 == pytest.approx(57.5)


def test_padding_never_produces_an_empty_crop() -> None:
    """A degenerate or out-of-frame box still yields at least one pixel."""
    padded = pad_and_clamp(BoundingBox(99.5, 99.5, 99.6, 99.6), 100, 100, 0.15)
    assert padded.x2 - padded.x1 >= 1.0
    assert padded.y2 - padded.y1 >= 1.0


def test_zero_padding_returns_the_box() -> None:
    """Padding 0 is the identity, which the crop-plus-context arm would use."""
    box = BoundingBox(10.0, 20.0, 30.0, 40.0)
    assert pad_and_clamp(box, 100, 100, 0.0).as_tuple() == box.as_tuple()


def test_padding_rejects_bad_inputs() -> None:
    """Invalid dimensions and negative padding are hard errors."""
    box = BoundingBox(0.0, 0.0, 10.0, 10.0)
    with pytest.raises(DetectionDataError, match="dimensions must be positive"):
        pad_and_clamp(box, 0, 100, 0.15)
    with pytest.raises(DetectionDataError, match="padding must be non-negative"):
        pad_and_clamp(box, 100, 100, -0.1)


# -- box validation -----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        [1.0, 2.0, 3.0],  # too few
        [1.0, 2.0, 3.0, 4.0, 5.0],  # too many
        [1.0, 2.0, "3", 4.0],  # non-numeric
        [True, 2.0, 3.0, 4.0],  # bool is an int subclass but not a coordinate
        "1,2,3,4",  # not a sequence of numbers
        None,
    ],
)
def test_invalid_boxes_are_rejected(raw: object, tmp_path: Path) -> None:
    """Structurally invalid boxes are collected, not silently accepted."""
    root = tmp_path / "Detection" / "VOC2007"
    root.mkdir(parents=True)
    (root / "boxes_top10.json").write_text(json.dumps({"a.jpg": raw}), encoding="utf-8")
    boxes, invalid = load_boxes(tmp_path, "det_top10")
    assert boxes == {}
    assert "a.jpg" in invalid


def test_valid_box_is_accepted(tmp_path: Path) -> None:
    """A well-formed box round-trips into a BoundingBox."""
    root = tmp_path / "Detection" / "VOC2007"
    root.mkdir(parents=True)
    (root / "boxes_top10.json").write_text(
        json.dumps({"a.jpg": [1, 2, 30, 40]}), encoding="utf-8"
    )
    boxes, invalid = load_boxes(tmp_path, "det_top10")
    assert invalid == {}
    assert boxes["a.jpg"].as_tuple() == (1.0, 2.0, 30.0, 40.0)
    assert boxes["a.jpg"].area == pytest.approx(29.0 * 38.0)


# -- record partitioning (the pairing invariant) ------------------------


def _record(name: str, label: int = 0) -> ManifestRecord:
    """Build a throwaway record."""
    return ManifestRecord(
        filename=name,
        ip102_label=label,
        project_label=label,
        class_name=f"c{label}",
        split="train",
    )


def test_records_without_a_box_are_dropped() -> None:
    """Missing, invalid and degenerate boxes are all excluded."""
    records = [_record(n) for n in ("ok.jpg", "missing.jpg", "bad.jpg", "flat.jpg")]
    boxes = {
        "ok.jpg": BoundingBox(0.0, 0.0, 10.0, 10.0),
        "flat.jpg": BoundingBox(5.0, 5.0, 5.0, 20.0),  # zero width
    }
    partition = partition_records(records, boxes, {"bad.jpg": "invalid"})
    assert [r.filename for r in partition.kept] == ["ok.jpg"]
    assert set(partition.dropped_filenames) == {"missing.jpg", "bad.jpg", "flat.jpg"}


def test_partition_preserves_order() -> None:
    """Record order is official split order and must survive filtering."""
    names = [f"{i}.jpg" for i in range(6)]
    boxes = {n: BoundingBox(0.0, 0.0, 5.0, 5.0) for n in names if n != "3.jpg"}
    partition = partition_records([_record(n) for n in names], boxes)
    assert [r.filename for r in partition.kept] == [
        "0.jpg",
        "1.jpg",
        "2.jpg",
        "4.jpg",
        "5.jpg",
    ]


# -- the crop transform -------------------------------------------------


def test_crop_transform_requires_a_filename() -> None:
    """Without provenance the transform cannot find its box, and says so."""
    pytest.importorskip("PIL")
    from PIL import Image

    transform = BoxCropTransform({"a.jpg": BoundingBox(0.0, 0.0, 5.0, 5.0)})
    with pytest.raises(DetectionDataError, match="requires the record's filename"):
        transform(Image.new("RGB", (10, 10)))


def test_crop_transform_refuses_to_fall_back_to_full_frame() -> None:
    """A missing box must raise rather than quietly become a full-frame sample.

    This is the single most important guard in the module: a silent fallback
    would put full-frame samples inside the crop arm and corrupt the comparison
    without any visible failure.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    transform = BoxCropTransform({"a.jpg": BoundingBox(0.0, 0.0, 5.0, 5.0)})
    with pytest.raises(DetectionDataError, match="no bounding box available"):
        transform(Image.new("RGB", (10, 10)), "absent.jpg")


def test_crop_transform_applies_inner_pipeline() -> None:
    """The crop arm reuses the project's preprocessing rather than its own."""
    pytest.importorskip("PIL")
    from PIL import Image

    seen: dict[str, tuple[int, int]] = {}

    def inner(image):
        seen["size"] = image.size
        return "transformed"

    transform = BoxCropTransform(
        {"a.jpg": BoundingBox(10.0, 10.0, 30.0, 30.0)}, inner, padding=0.0
    )
    assert transform(Image.new("RGB", (100, 100)), "a.jpg") == "transformed"
    assert seen["size"] == (20, 20)


def test_crop_does_not_modify_the_source_image() -> None:
    """Cropping returns a new image; the original is untouched."""
    pytest.importorskip("PIL")
    from PIL import Image

    source = Image.new("RGB", (100, 100), (10, 20, 30))
    cropped = crop_with_padding(source, BoundingBox(10.0, 10.0, 40.0, 40.0), 0.15)
    assert source.size == (100, 100)
    assert cropped.size != source.size


def test_box_statistics_thresholds() -> None:
    """Area-ratio thresholds count boxes strictly below each cutoff."""
    boxes = {
        "tiny.jpg": BoundingBox(0.0, 0.0, 10.0, 10.0),  # 1% of 100x100
        "small.jpg": BoundingBox(0.0, 0.0, 40.0, 40.0),  # 16%
        "big.jpg": BoundingBox(0.0, 0.0, 90.0, 90.0),  # 81%
    }
    sizes = dict.fromkeys(boxes, (100, 100))
    stats = box_statistics(boxes, sizes)
    assert stats["boxes_measured"] == 3
    assert stats["thresholds"]["below_10pct"]["count"] == 1
    assert stats["thresholds"]["below_25pct"]["count"] == 2
    assert stats["thresholds"]["below_50pct"]["count"] == 2


# -- shipped configuration ----------------------------------------------


@pytest.mark.parametrize("control,treatment,scope", ARM_PAIRS)
def test_arms_differ_only_in_the_crop_flag(
    control: str, treatment: str, scope: str
) -> None:
    """The paired configs must differ in exactly one resolved field.

    This is the property that makes the experiment controlled. It is checked
    against the *resolved* configuration, not the YAML text, so an accidental
    difference introduced through the extends chain is still caught.
    """
    a = load_config(["model_custom.yaml", control])
    b = load_config(["model_custom.yaml", treatment])

    assert a.dataset.scope_name == scope
    assert b.dataset.scope_name == scope
    assert a.section("dataset")["use_bbox_crop"] is False
    assert b.section("dataset")["use_bbox_crop"] is True

    differing = [
        key
        for key in set(a.data) | set(b.data)
        if a.data.get(key) != b.data.get(key)
    ]
    assert differing == ["dataset"], f"arms differ outside dataset: {differing}"

    da, db = dict(a.section("dataset")), dict(b.section("dataset"))
    changed = [k for k in set(da) | set(db) if da.get(k) != db.get(k)]
    assert changed == ["use_bbox_crop"], f"expected one difference, got {changed}"


@pytest.mark.parametrize("control,treatment,scope", ARM_PAIRS)
def test_arms_share_the_training_protocol(
    control: str, treatment: str, scope: str
) -> None:
    """Neither arm may be tuned independently of the other."""
    a = load_config(["model_custom.yaml", control])
    b = load_config(["model_custom.yaml", treatment])
    assert a.section("training") == b.section("training")
    assert a.section("preprocessing") == b.section("preprocessing")
    assert a.seed == b.seed == 1337
    assert a.section("model") == b.section("model")


def test_detection_protocol_matches_the_e0_recipe() -> None:
    """The detection protocol carries E0 across unchanged.

    Reusing the settled recipe is what keeps the crop the only new variable; a
    drifted hyperparameter here would make E4/E5 incomparable with the rice10
    lineage they are meant to extend.
    """
    e0 = load_config(["model_custom.yaml", "exp_rice10_protocol_a.yaml"]).section(
        "training"
    )
    det = load_config(
        ["model_custom.yaml", "exp_det_top10_e4a_fullframe.yaml"]
    ).section("training")
    assert det == e0


def test_no_pretrained_weights_are_configured() -> None:
    """EfficientNet and pretrained weights remain prohibited project-wide."""
    for control, treatment, _ in ARM_PAIRS:
        for name in (control, treatment):
            config = load_config(["model_custom.yaml", name])
            model = config.section("model")
            assert model.get("name") == "custom_cnn"
            assert not model.get("pretrained", False)


# -- real data ----------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_root() -> Path:
    """The IP102 root, skipping when the source data is absent."""
    root = load_config("base.yaml").paths.dataset_root
    if not detection_root(root).is_dir():
        pytest.skip(f"IP102 detection data not present under {root}")
    return root


@pytest.mark.parametrize("scope", ["det_top10", "det_top15"])
def test_real_splits_are_disjoint_and_unique(dataset_root: Path, scope: str) -> None:
    """No filename may appear twice, within or across splits."""
    splits = load_splits(dataset_root, scope)
    assert set(splits) == {"train", "validation", "test"}
    everything = [name for entries in splits.values() for name, _ in entries]
    assert len(everything) == len(set(everything))


@pytest.mark.parametrize(
    "scope,expected",
    [
        ("det_top10", {"train": 6395, "validation": 1370, "test": 1370}),
        ("det_top15", {"train": 6748, "validation": 1446, "test": 1447}),
    ],
)
def test_real_split_sizes(
    dataset_root: Path, scope: str, expected: dict[str, int]
) -> None:
    """Pin the official split sizes so a changed source file is noticed."""
    splits = load_splits(dataset_root, scope)
    assert {k: len(v) for k, v in splits.items()} == expected


@pytest.mark.parametrize("scope", ["det_top10", "det_top15"])
def test_real_labels_are_in_scope_range(dataset_root: Path, scope: str) -> None:
    """Every label must be valid for the scope's derived class count."""
    limit = num_classes_for(scope)
    splits = load_splits(dataset_root, scope)
    labels = {label for entries in splits.values() for _, label in entries}
    assert labels == set(range(limit))


def test_real_boxes_are_xyxy_not_xywh(dataset_root: Path) -> None:
    """Pin the coordinate convention, measured against real image sizes.

    This was determined empirically: on a 500-box sample the xyxy reading
    produced zero violations while an xywh reading produced 408. Reading the
    boxes the wrong way would still train and still produce plausible numbers,
    so the convention is pinned rather than trusted.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    boxes, invalid = load_boxes(dataset_root, "det_top10")
    assert invalid == {}
    images = detection_root(dataset_root) / "JPEGImages"
    sample = sorted(boxes)[:200]

    xyxy_violations = 0
    xywh_violations = 0
    for name in sample:
        box = boxes[name]
        with Image.open(images / name) as image:
            width, height = image.size
        if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
            xyxy_violations += 1
        if not (box.x1 + box.x2 <= width and box.y1 + box.y2 <= height):
            xywh_violations += 1

    assert xyxy_violations == 0
    assert xywh_violations > 0.5 * len(sample)


def test_real_top10_has_exactly_one_missing_box(dataset_root: Path) -> None:
    """Pin the known gap so a change in the source data is noticed."""
    boxes, _ = load_boxes(dataset_root, "det_top10")
    records = [
        record
        for split in ("train", "validation", "test")
        for record in build_detection_records(dataset_root, "det_top10", split)
    ]
    partition = partition_records(records, boxes)
    assert partition.dropped_filenames == ("IP022000163.jpg",)


@pytest.mark.parametrize("control,treatment,scope", ARM_PAIRS)
def test_real_arms_consume_identical_records(
    dataset_root: Path, control: str, treatment: str, scope: str
) -> None:
    """The paired arms must see the same samples in the same order.

    Verified through the real loader, because this is the invariant that makes
    the comparison paired: if the crop arm dropped an image the control kept,
    the two runs would be scored on different data.
    """
    pytest.importorskip("torch")
    from farm_pest_ai.data.loaders import build_dataset

    for split in ("train", "validation"):
        a = build_dataset(load_config(["model_custom.yaml", control]), split)
        b = build_dataset(load_config(["model_custom.yaml", treatment]), split)
        assert [r.filename for r in a.records] == [r.filename for r in b.records]
        assert [r.project_label for r in a.records] == [
            r.project_label for r in b.records
        ]
        assert a.num_classes == b.num_classes == num_classes_for(scope)


@pytest.mark.parametrize("control,treatment,_scope", ARM_PAIRS)
def test_real_crop_arm_changes_the_pixels(
    dataset_root: Path, control: str, treatment: str, _scope: str
) -> None:
    """The crop arm must actually differ from its control on most samples.

    Not all: the audit measured ~10% of padded boxes covering the whole frame,
    where the two arms are legitimately identical.
    """
    torch = pytest.importorskip("torch")
    from farm_pest_ai.data.loaders import build_dataset

    a = build_dataset(load_config(["model_custom.yaml", control]), "validation")
    b = build_dataset(load_config(["model_custom.yaml", treatment]), "validation")
    differing = 0
    checked = list(range(0, len(a), max(1, len(a) // 25)))[:25]
    for index in checked:
        assert a.records[index].filename == b.records[index].filename
        if not torch.allclose(a[index][0], b[index][0]):
            differing += 1
    assert differing >= int(0.7 * len(checked))


def test_real_detection_eval_pipeline_is_deterministic(dataset_root: Path) -> None:
    """Cropping must not introduce randomness into evaluation."""
    torch = pytest.importorskip("torch")
    from farm_pest_ai.data.loaders import build_dataset

    dataset = build_dataset(
        load_config(["model_custom.yaml", "exp_det_top10_e4b_crop15.yaml"]),
        "validation",
    )
    for index in (0, 17, 251):
        assert torch.equal(dataset[index][0], dataset[index][0])


def test_real_detection_tensors_have_the_expected_shape(dataset_root: Path) -> None:
    """Both arms hand the model the same tensor contract."""
    torch = pytest.importorskip("torch")
    from farm_pest_ai.data.loaders import build_dataset

    for name in ("exp_det_top10_e4a_fullframe.yaml", "exp_det_top10_e4b_crop15.yaml"):
        dataset = build_dataset(load_config(["model_custom.yaml", name]), "validation")
        tensor, label = dataset[0]
        assert tensor.shape == (3, 160, 160)
        assert tensor.dtype == torch.float32
        assert 0 <= label < 10


def test_real_source_images_are_not_modified(dataset_root: Path) -> None:
    """Cropping is in-memory: the source JPEG is byte-identical afterwards."""
    pytest.importorskip("PIL")
    from PIL import Image

    boxes, _ = load_boxes(dataset_root, "det_top10")
    name = sorted(boxes)[0]
    path = detection_root(dataset_root) / "JPEGImages" / name
    before = path.read_bytes()
    with Image.open(path) as image:
        image.load()
        crop_with_padding(image.convert("RGB"), boxes[name], DEFAULT_PADDING)
    assert path.read_bytes() == before


# -- paired comparison script -------------------------------------------


def _load_comparison_script():
    """Import scripts/compare_crop_experiments.py as a module."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "compare_crop_experiments.py"
    spec = importlib.util.spec_from_file_location("compare_crop_experiments", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_pairs_carry_both_arm_configs() -> None:
    """Each pair must name its own config for BOTH arms.

    Regression test. Scoring both arms through the control's configuration fed
    the crop model full frames: it loads without error, because cropping happens
    before the pipeline and the two arms therefore share a preprocessing
    fingerprint, and it silently produced inverted flip counts (221 "broken"
    against 95 "corrected" for a pair whose accuracy had risen).
    """
    module = _load_comparison_script()
    for label, control, treatment, control_cfg, treatment_cfg, scope in module.PAIRS:
        assert control_cfg != treatment_cfg, f"{label} reuses one config for both arms"
        assert "fullframe" in control_cfg
        assert "crop" in treatment_cfg
        assert scope in control_cfg and scope.replace("det_", "") in treatment_cfg


def test_comparison_noise_threshold_matches_project_convention() -> None:
    """The 0.01 threshold is inherited from Phase 7.2, not reinvented."""
    module = _load_comparison_script()
    assert module.NOISE_THRESHOLD == 0.01
