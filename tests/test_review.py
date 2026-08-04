"""Tests for the read-only image-quality review.

The safety properties matter more than the measurements here: the review must
never modify a source image, never assert a label change, and never touch the
test split. Those are checked explicitly rather than assumed from the docstring.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from farm_pest_ai.data.review import (
    OBJECTIVE_FLAGS,
    REVIEW_CATEGORIES,
    REVIEW_COLUMNS,
    REVIEWABLE_SPLITS,
    ReviewError,
    ReviewRecord,
    ReviewThresholds,
    build_contact_sheet,
    curated_manifest_dir,
    measure_image,
    read_review_manifest,
    suggest_issue,
    write_review_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_IMAGES = REPO_ROOT / "ip102_v1.1" / "Classification" / "images"

Image = pytest.importorskip("PIL.Image")


def _record(**overrides: object) -> ReviewRecord:
    """A review record with sensible defaults."""
    defaults: dict[str, object] = {
        "filename": "00001.jpg",
        "split": "validation",
        "current_label": 3,
        "current_class_name": "yellow rice borer",
        "width": 400,
        "height": 300,
    }
    defaults.update(overrides)
    return ReviewRecord(**defaults)  # type: ignore[arg-type]


# -- policy -------------------------------------------------------------


def test_the_test_split_is_not_reviewable() -> None:
    """Reviewing the test split would let it shape a data decision."""
    assert "test" not in REVIEWABLE_SPLITS
    assert set(REVIEWABLE_SPLITS) == {"train", "validation"}


def test_only_pixel_measurable_categories_are_asserted() -> None:
    """Everything needing judgement is a suggestion, never a measured flag."""
    assert set(OBJECTIVE_FLAGS) <= set(REVIEW_CATEGORIES)
    assert set(OBJECTIVE_FLAGS) == {"blurry", "low_resolution"}
    # The judgement categories must not be assertable from pixels.
    for category in ("suspected_mislabel", "unrelated", "symptom_only", "diagram_text"):
        assert category not in OBJECTIVE_FLAGS


def test_every_taxonomy_category_is_present() -> None:
    """The ten categories the review policy defines."""
    assert set(REVIEW_CATEGORIES) == {
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
    }


def test_reviewer_columns_ship_empty(tmp_path: Path) -> None:
    """The decision is the human's; the manifest must not pre-fill it."""
    path = write_review_manifest(
        [_record(suspected_issue="suspected_mislabel")], tmp_path / "review.csv"
    )
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["reviewer_decision"] == ""
    assert rows[0]["reviewer_notes"] == ""
    # The suspicion is recorded separately from the decision.
    assert rows[0]["suspected_issue"] == "suspected_mislabel"


def test_manifest_carries_every_required_column(tmp_path: Path) -> None:
    path = write_review_manifest([_record()], tmp_path / "review.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(REVIEW_COLUMNS)
    for column in (
        "filename",
        "split",
        "current_label",
        "model_prediction",
        "confidence",
        "quality_flags",
        "suspected_issue",
        "reviewer_decision",
        "reviewer_notes",
    ):
        assert column in REVIEW_COLUMNS


def test_manifest_uses_lf_endings(tmp_path: Path) -> None:
    """Byte-identical output on Windows and in a container."""
    path = write_review_manifest([_record()], tmp_path / "review.csv")
    assert b"\r\n" not in path.read_bytes()


def test_reading_back_rejects_an_invalid_decision(tmp_path: Path) -> None:
    """A typo in the human's column must not propagate downstream."""
    path = write_review_manifest([_record()], tmp_path / "review.csv")
    text = path.read_text(encoding="utf-8").replace(
        ",,\n", ",not_a_category,\n", 1
    )
    path.write_text(text, encoding="utf-8", newline="")
    with pytest.raises(ReviewError, match="not one of"):
        read_review_manifest(path)


def test_reading_back_accepts_a_valid_decision(tmp_path: Path) -> None:
    path = write_review_manifest([_record()], tmp_path / "review.csv")
    text = path.read_text(encoding="utf-8").replace(",,\n", ",diagram_text,\n", 1)
    path.write_text(text, encoding="utf-8", newline="")
    rows = read_review_manifest(path)
    assert rows[0]["reviewer_decision"] == "diagram_text"


# -- not clobbering existing review work --------------------------------
#
# The review manifest is the one artifact a human writes into by hand, so an
# accidental overwrite destroys work no rerun can recreate. Both guards below
# come from real incidents: a `--limit 40` smoke pass silently replaced a
# complete 721-row review, and the same path would have overwritten reviewer
# decisions had any been entered.

sys.path.insert(0, str(REPO_ROOT / "scripts"))
review_images = pytest.importorskip("review_images")


def test_partial_review_refuses_to_replace_a_fuller_one(tmp_path: Path) -> None:
    path = write_review_manifest(
        [_record(filename=f"{i}.jpg") for i in range(50)], tmp_path / "review.csv"
    )
    refusal = review_images._refuse_to_clobber(path, incoming=10, force=False)
    assert refusal is not None
    assert "partial" in refusal


def test_a_fuller_review_may_replace_a_smaller_one(tmp_path: Path) -> None:
    """Re-running a complete review over a partial one is the normal fix."""
    path = write_review_manifest([_record()], tmp_path / "review.csv")
    assert review_images._refuse_to_clobber(path, incoming=721, force=False) is None


def test_reviewer_decisions_are_never_silently_overwritten(tmp_path: Path) -> None:
    path = write_review_manifest([_record()], tmp_path / "review.csv")
    text = path.read_text(encoding="utf-8").replace(",,\n", ",diagram_text,\n", 1)
    path.write_text(text, encoding="utf-8", newline="")

    # Even a strictly larger review must not discard a human's decisions.
    refusal = review_images._refuse_to_clobber(path, incoming=10_000, force=False)
    assert refusal is not None
    assert "reviewer decision" in refusal


def test_force_overrides_both_guards(tmp_path: Path) -> None:
    path = write_review_manifest(
        [_record(filename=f"{i}.jpg") for i in range(50)], tmp_path / "review.csv"
    )
    assert review_images._refuse_to_clobber(path, incoming=1, force=True) is None


def test_writing_a_new_manifest_is_never_refused(tmp_path: Path) -> None:
    assert (
        review_images._refuse_to_clobber(tmp_path / "absent.csv", 10, force=False)
        is None
    )


# -- curated manifests --------------------------------------------------


def test_curated_manifests_go_to_a_new_versioned_directory(tmp_path: Path) -> None:
    """A curated split never overwrites the official benchmark manifest."""
    official = tmp_path / "rice10"
    curated = curated_manifest_dir(tmp_path, "rice10", "v1")
    assert curated == official / "curated" / "v1"
    # It is strictly below the scope directory, never beside the official CSVs.
    assert curated.name == "v1"
    assert "curated" in curated.parts


@pytest.mark.parametrize("version", ["", ".", "..", "../escape", "a/b"])
def test_curated_version_must_be_a_simple_name(tmp_path: Path, version: str) -> None:
    """A path in the version would let a curated write escape its directory."""
    with pytest.raises(ReviewError, match="simple name"):
        curated_manifest_dir(tmp_path, "rice10", version)


# -- suggestions --------------------------------------------------------


def test_objective_flags_take_priority_over_model_opinion() -> None:
    """A measured property outranks a guess about the label."""
    assert (
        suggest_issue(
            ["low_resolution"], confidence=0.99, prediction_correct=False
        )
        == "low_resolution"
    )


def test_low_confidence_is_queued_as_ambiguous() -> None:
    assert suggest_issue([], confidence=0.1, prediction_correct=True) == "ambiguous"


def test_confident_disagreement_is_only_suspected() -> None:
    """The strongest signal available still yields 'suspected', not a verdict."""
    issue = suggest_issue([], confidence=0.95, prediction_correct=False)
    assert issue == "suspected_mislabel"
    assert issue.startswith("suspected")


def test_confident_agreement_raises_nothing() -> None:
    assert suggest_issue([], confidence=0.95, prediction_correct=True) == ""


def test_no_model_means_no_label_suspicion() -> None:
    """Without predictions the review cannot suspect a mislabel at all."""
    assert suggest_issue([], confidence=None, prediction_correct=None) == ""


def test_every_suggestion_is_a_known_category() -> None:
    allowed = set(REVIEW_CATEGORIES) | {""}
    for flags in ([], ["blurry"], ["low_resolution"]):
        for confidence in (None, 0.1, 0.95):
            for correct in (None, True, False):
                assert (
                    suggest_issue(
                        flags, confidence=confidence, prediction_correct=correct
                    )
                    in allowed
                )


# -- thresholds and measurement -----------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [("min_short_side", 0), ("blur_variance", -1.0), ("low_confidence", 1.5)],
)
def test_invalid_thresholds_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ReviewError):
        ReviewThresholds(**{field: value})  # type: ignore[arg-type]


def test_low_resolution_flag_follows_the_threshold(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    Image.new("RGB", (100, 80), color=(120, 130, 140)).save(path)
    measured = measure_image(path, thresholds=ReviewThresholds(min_short_side=160))
    assert measured["width"] == 100
    assert measured["height"] == 80
    assert "low_resolution" in measured["quality_flags"]


def test_large_image_is_not_flagged_low_resolution(tmp_path: Path) -> None:
    path = tmp_path / "large.png"
    Image.new("RGB", (400, 300), color=(120, 130, 140)).save(path)
    measured = measure_image(path, thresholds=ReviewThresholds(min_short_side=160))
    assert "low_resolution" not in measured["quality_flags"]


def test_flat_image_scores_zero_focus_and_is_flagged_blurry(tmp_path: Path) -> None:
    """A uniform image has no edges, so the focus measure is zero."""
    path = tmp_path / "flat.png"
    Image.new("RGB", (300, 300), color=(128, 128, 128)).save(path)
    measured = measure_image(path)
    assert measured["focus_measure"] == pytest.approx(0.0)
    assert "blurry" in measured["quality_flags"]


def test_sharp_image_scores_higher_focus_than_a_blurred_copy(tmp_path: Path) -> None:
    """The focus measure must order a sharp image above its blurred version."""
    from PIL import ImageFilter

    sharp_path = tmp_path / "sharp.png"
    blurred_path = tmp_path / "blurred.png"

    # A high-contrast checkerboard: plenty of edges to lose.
    sharp = Image.new("RGB", (300, 300), color=(0, 0, 0))
    for y in range(0, 300, 20):
        for x in range(0, 300, 20):
            if (x // 20 + y // 20) % 2 == 0:
                for dy in range(20):
                    for dx in range(20):
                        sharp.putpixel((x + dx, y + dy), (255, 255, 255))
    sharp.save(sharp_path)
    sharp.filter(ImageFilter.GaussianBlur(radius=4)).save(blurred_path)

    assert (
        measure_image(sharp_path)["focus_measure"]
        > measure_image(blurred_path)["focus_measure"]
    )


def test_undecodable_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(ReviewError, match="could not decode"):
        measure_image(path)


# -- contact sheets -----------------------------------------------------


def test_contact_sheet_lays_out_a_grid(tmp_path: Path) -> None:
    paths = []
    for index in range(7):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (200, 150), color=(index * 30, 100, 100)).save(path)
        paths.append((path, f"image {index}"))

    sheet = build_contact_sheet(paths, columns=3, thumbnail=100, label_height=20)
    # 7 images over 3 columns is 3 rows.
    assert sheet.size == (300, 3 * 120)


def test_contact_sheet_survives_an_undecodable_file(tmp_path: Path) -> None:
    """One bad file must not break the sheet's alignment with the manifest."""
    good = tmp_path / "good.png"
    Image.new("RGB", (100, 100), color=(10, 20, 30)).save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")

    sheet = build_contact_sheet(
        [(good, "good"), (bad, "bad")], columns=2, thumbnail=80, label_height=20
    )
    assert sheet.size == (160, 100)


def test_contact_sheet_requires_images() -> None:
    with pytest.raises(ReviewError, match="at least one image"):
        build_contact_sheet([])


# -- the real dataset stays read-only -----------------------------------

real_data = pytest.mark.skipif(
    not SOURCE_IMAGES.is_dir(), reason="the source dataset is not present"
)


@real_data
def test_measuring_a_real_image_does_not_modify_it() -> None:
    """The audit opens source files read-only, and must leave them untouched."""
    path = next(iter(sorted(SOURCE_IMAGES.glob("*.jpg"))))
    before = (path.stat().st_mtime_ns, path.stat().st_size)

    measured = measure_image(path)
    assert measured["width"] > 0 and measured["height"] > 0

    after = (path.stat().st_mtime_ns, path.stat().st_size)
    assert before == after, "the audit modified a source image"


@real_data
def test_building_a_contact_sheet_does_not_modify_sources() -> None:
    paths = sorted(SOURCE_IMAGES.glob("*.jpg"))[:6]
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in paths}

    build_contact_sheet([(p, p.name) for p in paths], columns=3)

    after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in paths}
    assert before == after, "building a contact sheet modified a source image"
