"""Tests that read the real IP102 source data and the derived manifests.

Every test here skips when the dataset or the built manifests are absent, so
the suite still runs on a machine without them. Assertions pin the figures
verified in Phase 1 and re-verified in Phase 4: if the source data or the build
ever drifts, these fail rather than a training run silently learning the wrong
labels.

Nothing here writes to ``ip102_v1.1``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from farm_pest_ai.config import load_config
from farm_pest_ai.data.manifests import (
    SPLITS,
    read_classes,
    read_derived_manifest,
    read_source_manifest,
)
from farm_pest_ai.scopes import CLASS_MAPPING_VERSION, FULL102, RICE10, get_scope

#: Counts measured in Phase 1 from the official manifests.
SOURCE_COUNTS = {"train": 45095, "validation": 7508, "test": 22619}

#: Derived rice10 counts, confirmed in Phase 1 and rebuilt in Phase 4.
RICE10_COUNTS = {"train": 4318, "validation": 721, "test": 2166}

#: The ten rice10 classes, in project-label order, from classes.txt.
RICE10_NAMES = (
    "rice leaf roller",
    "rice leaf caterpillar",
    "asiatic rice borer",
    "yellow rice borer",
    "rice gall midge",
    "brown plant hopper",
    "white backed plant hopper",
    "small brown plant hopper",
    "rice water weevil",
    "rice leafhopper",
)


@pytest.fixture(scope="module")
def config():
    """The base configuration, or a skip when it cannot be loaded."""
    return load_config("base.yaml")


@pytest.fixture(scope="module")
def classification_root(config) -> Path:
    """The IP102 classification directory, skipping when absent."""
    root = config.paths.classification_root
    if not root.is_dir():
        pytest.skip(f"IP102 source data not present at {root}")
    return root


@pytest.fixture(scope="module")
def processed_dir(config) -> Path:
    """The derived-manifest directory, skipping when it has not been built."""
    path = config.paths.processed_dir
    if not path.is_dir():
        pytest.skip("derived manifests not built; run scripts/build_manifests.py")
    return path


def require_manifests(processed_dir: Path, scope):
    """Read all three derived manifests for a scope, skipping when unbuilt."""
    try:
        return {
            split: read_derived_manifest(processed_dir, scope, split)[0]
            for split in SPLITS
        }
    except Exception as exc:
        pytest.skip(f"derived manifests for {scope.name} unavailable: {exc}")


# -- source data --------------------------------------------------------


def test_classes_txt_has_102_entries(classification_root: Path) -> None:
    classes = read_classes(classification_root / "classes.txt", expected=102)
    assert len(classes) == 102
    assert classes[0].ip102_label == 0
    assert classes[-1].ip102_label == 101


def test_classes_txt_ids_are_one_ahead_of_labels(classification_root: Path) -> None:
    """The off-by-one that is the most likely source of a silent labelling bug."""
    for info in read_classes(classification_root / "classes.txt"):
        assert info.ip102_label == info.classes_txt_id - 1


def test_classes_txt_names_carry_no_stray_whitespace(
    classification_root: Path,
) -> None:
    """The shipped file pads names with tabs and uses CRLF endings."""
    for info in read_classes(classification_root / "classes.txt"):
        assert info.raw_name == info.raw_name.strip()
        assert "\t" not in info.raw_name
        assert "\r" not in info.raw_name
        assert info.raw_name


@pytest.mark.parametrize(("split", "filename"), [
    ("train", "train.txt"), ("validation", "val.txt"), ("test", "test.txt")
])
def test_source_split_counts_match_phase1(
    classification_root: Path, split: str, filename: str
) -> None:
    records = read_source_manifest(classification_root / filename)
    assert len(records) == SOURCE_COUNTS[split]


def test_source_labels_stay_within_ip102_range(classification_root: Path) -> None:
    for filename in ("train.txt", "val.txt", "test.txt"):
        labels = {label for _, label in read_source_manifest(classification_root / filename)}
        assert min(labels) == 0
        assert max(labels) == 101
        assert len(labels) == 102


# -- derived manifests --------------------------------------------------


@pytest.mark.parametrize("split", SPLITS)
def test_rice10_counts_match_phase1(processed_dir: Path, split: str) -> None:
    manifests = require_manifests(processed_dir, RICE10)
    assert len(manifests[split]) == RICE10_COUNTS[split]


def test_rice10_total_is_7205(processed_dir: Path) -> None:
    manifests = require_manifests(processed_dir, RICE10)
    assert sum(len(r) for r in manifests.values()) == 7205


@pytest.mark.parametrize("split", SPLITS)
def test_full102_counts_match_the_source(processed_dir: Path, split: str) -> None:
    """full102 is an identity mapping, so nothing may be dropped."""
    manifests = require_manifests(processed_dir, FULL102)
    assert len(manifests[split]) == SOURCE_COUNTS[split]


def test_full102_project_labels_equal_ip102_labels(processed_dir: Path) -> None:
    manifests = require_manifests(processed_dir, FULL102)
    for records in manifests.values():
        assert all(r.project_label == r.ip102_label for r in records)


def test_rice10_class_names_match_classes_txt(processed_dir: Path) -> None:
    """The exact ten names, in the exact project-label order."""
    manifests = require_manifests(processed_dir, RICE10)
    names: dict[int, str] = {}
    for records in manifests.values():
        for record in records:
            names.setdefault(record.project_label, record.class_name)
    assert tuple(names[label] for label in range(10)) == RICE10_NAMES


def test_rice10_labels_follow_the_scope_mapping(processed_dir: Path) -> None:
    manifests = require_manifests(processed_dir, RICE10)
    observed: dict[int, int] = {}
    for records in manifests.values():
        for record in records:
            observed[record.project_label] = record.ip102_label
    assert observed == dict(RICE10.project_to_original)


def test_rice10_excludes_the_neighbouring_labels(processed_dir: Path) -> None:
    """IP102 2 (paddy stem maggot) and 6 sit in range but are not rice pests here."""
    manifests = require_manifests(processed_dir, RICE10)
    present = {r.ip102_label for records in manifests.values() for r in records}
    assert 2 not in present
    assert 6 not in present


@pytest.mark.parametrize("scope_name", ["rice10", "full102"])
def test_every_class_appears_in_every_split(
    processed_dir: Path, scope_name: str
) -> None:
    scope = get_scope(scope_name)
    manifests = require_manifests(processed_dir, scope)
    for split, records in manifests.items():
        labels = {r.project_label for r in records}
        assert labels == set(range(scope.num_classes)), (
            f"{scope_name}/{split} is missing classes "
            f"{sorted(set(range(scope.num_classes)) - labels)}"
        )


@pytest.mark.parametrize("scope_name", ["rice10", "full102"])
def test_no_filename_appears_in_two_splits(
    processed_dir: Path, scope_name: str
) -> None:
    """Filename-level split integrity, re-checked against the built manifests."""
    manifests = require_manifests(processed_dir, get_scope(scope_name))
    seen: dict[str, str] = {}
    for split, records in manifests.items():
        for record in records:
            previous = seen.get(record.filename)
            assert previous is None, (
                f"{record.filename} appears in both {previous} and {split}"
            )
            seen[record.filename] = split


@pytest.mark.parametrize("scope_name", ["rice10", "full102"])
def test_derived_manifests_agree_with_the_source(
    classification_root: Path, processed_dir: Path, scope_name: str
) -> None:
    """Every derived record must trace back to the read-only source manifest."""
    scope = get_scope(scope_name)
    manifests = require_manifests(processed_dir, scope)
    sources = {
        "train": "train.txt", "validation": "val.txt", "test": "test.txt"
    }
    for split, filename in sources.items():
        source = read_source_manifest(classification_root / filename)
        in_scope = [
            (name, label) for name, label in source if scope.includes_original(label)
        ]
        derived = manifests[split]
        assert len(in_scope) == len(derived)
        assert [r.filename for r in derived] == [n for n, _ in in_scope]
        assert [r.ip102_label for r in derived] == [lbl for _, lbl in in_scope]


@pytest.mark.parametrize("scope_name", ["rice10", "full102"])
def test_manifest_metadata_records_the_scope_and_mapping_version(
    processed_dir: Path, scope_name: str
) -> None:
    """Scope and mapping version travel with every artifact, by contract."""
    scope = get_scope(scope_name)
    for split in SPLITS:
        try:
            _, metadata = read_derived_manifest(processed_dir, scope, split)
        except Exception as exc:
            pytest.skip(f"manifest unavailable: {exc}")
        assert metadata["scope"] == scope_name
        assert metadata["num_classes"] == scope.num_classes
        assert metadata["class_mapping_version"] == CLASS_MAPPING_VERSION


@pytest.mark.parametrize("scope_name", ["rice10", "full102"])
def test_class_mapping_file_matches_the_scope_module(
    processed_dir: Path, scope_name: str
) -> None:
    """The on-disk mapping must never drift from farm_pest_ai.scopes."""
    scope = get_scope(scope_name)
    path = processed_dir / scope_name / "class_mapping.json"
    if not path.is_file():
        pytest.skip(f"class mapping not built for {scope_name}")

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["scope"] == scope_name
    assert document["num_classes"] == scope.num_classes
    assert document["class_mapping_version"] == CLASS_MAPPING_VERSION

    mapping = {
        int(entry["project_label"]): int(entry["ip102_label"])
        for entry in document["classes"]
    }
    assert mapping == dict(scope.project_to_original)

    for entry in document["classes"]:
        assert entry["classes_txt_id"] == entry["ip102_label"] + 1


#: Files whose extension says .jpg but whose content is PNG, found by the
#: Phase 4 full-decode audit. Phase 1's 2,000-image sample missed them. All ten
#: sit in IP102 label 56, and seven carry an alpha channel.
NON_JPEG_FILES = frozenset({
    "40256.jpg", "40314.jpg", "40549.jpg", "40557.jpg", "40563.jpg",
    "40574.jpg", "40577.jpg", "40591.jpg", "40601.jpg", "40630.jpg",
})

#: The subset of the above that decode as RGBA rather than RGB.
RGBA_FILES = frozenset({
    "40314.jpg", "40549.jpg", "40563.jpg", "40574.jpg", "40577.jpg",
    "40591.jpg", "40601.jpg",
})


#: Exact-content cross-split duplicate pairs found in full102 by the Phase 4
#: audit. Both are train/test pairs within a single class, so the test set is
#: contaminated by two images. Phase 9 reports metrics with and without them.
KNOWN_LEAKED_PAIRS = (
    ("40410.jpg", "40432.jpg", 56),
    ("65553.jpg", "66152.jpg", 92),
)


@pytest.mark.parametrize(("train_file", "test_file", "label"), KNOWN_LEAKED_PAIRS)
def test_known_cross_split_duplicates_are_byte_identical(
    classification_root: Path, train_file: str, test_file: str, label: int
) -> None:
    """Pin the measured leakage so a silent change to the source data is caught.

    These pairs are byte-identical, which is unambiguous contamination: the
    model can memorise the training copy and be rewarded for it at test time.
    """
    from farm_pest_ai.data.audit import hash_file

    images = classification_root / "images"
    assert hash_file(images / train_file) == hash_file(images / test_file)


def test_full102_leakage_is_limited_to_the_known_pairs(processed_dir: Path) -> None:
    """Guard the audit's headline finding: only two test images are affected."""
    report_path = (
        processed_dir.parent / "reports" / "dataset_audit_full102.json"
    )
    if not report_path.is_file():
        pytest.skip("full102 audit report not present; run scripts/audit_dataset.py")

    leakage = json.loads(report_path.read_text(encoding="utf-8"))["leakage"]
    assert leakage["cross_split_groups"] == len(KNOWN_LEAKED_PAIRS)
    assert leakage["label_conflict_groups"] == 0
    assert leakage["leaked_files_per_split"]["test"] == len(KNOWN_LEAKED_PAIRS)
    assert leakage["leaked_files_per_split"]["validation"] == 0


def test_rice10_has_no_cross_split_leakage(processed_dir: Path) -> None:
    """The development scope's validation figures are uncontaminated."""
    report_path = processed_dir.parent / "reports" / "dataset_audit_rice10.json"
    if not report_path.is_file():
        pytest.skip("rice10 audit report not present; run scripts/audit_dataset.py")

    leakage = json.loads(report_path.read_text(encoding="utf-8"))["leakage"]
    assert leakage["cross_split_groups"] == 0
    assert leakage["label_conflict_groups"] == 0


@pytest.mark.parametrize("filename", sorted(NON_JPEG_FILES))
def test_known_png_files_still_decode(
    classification_root: Path, filename: str
) -> None:
    """The mislabelled-extension files must still load.

    Pillow dispatches on content rather than extension, so these decode fine.
    The Phase 5 loader must not switch to an extension-based reader, and must
    convert to RGB rather than assuming three channels.
    """
    pytest.importorskip("PIL")
    from farm_pest_ai.data.audit import probe_image

    probe = probe_image(classification_root / "images" / filename)
    assert probe.ok, f"{filename} failed to decode: {probe.error}"
    assert probe.image_format == "PNG"
    assert probe.mode == ("RGBA" if filename in RGBA_FILES else "RGB")


def test_rgba_images_convert_to_three_channels(classification_root: Path) -> None:
    """An alpha channel would give the CNN a fourth input plane if left alone."""
    pytest.importorskip("PIL")
    from PIL import Image

    path = classification_root / "images" / sorted(RGBA_FILES)[0]
    with Image.open(path) as image:
        assert image.mode == "RGBA"
        assert image.convert("RGB").mode == "RGB"


def test_rice10_and_full102_artifacts_are_separate(processed_dir: Path) -> None:
    """The two scopes are different tasks; their outputs must not collide."""
    rice = processed_dir / "rice10"
    full = processed_dir / "full102"
    if not (rice.is_dir() and full.is_dir()):
        pytest.skip("both scopes must be built for this check")
    assert rice != full
    assert (rice / "train.csv").read_bytes() != (full / "train.csv").read_bytes()
