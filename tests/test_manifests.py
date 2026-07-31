"""Tests for reading source manifests and deriving scope-aware ones.

These use small synthetic fixtures rather than the real dataset, so the suite
runs without ``ip102_v1.1`` present. The tests that touch the actual source data
live in ``test_dataset_integration.py`` and skip when it is absent.

The ``classes.txt`` off-by-one and the rice10 remap are the two places a silent
labelling bug could hide, so both are pinned by exact assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from farm_pest_ai.data.manifests import (
    DERIVED_COLUMNS,
    SPLITS,
    ClassInfo,
    ManifestError,
    atomic_write_text,
    build_derived_manifest,
    manifest_csv_path,
    read_classes,
    read_derived_manifest,
    read_source_manifest,
    render_manifest_csv,
    write_derived_manifest,
)
from farm_pest_ai.scopes import CLASS_MAPPING_VERSION, FULL102, RICE10

# -- fixtures -----------------------------------------------------------


@pytest.fixture()
def classes_file(tmp_path: Path) -> Path:
    """A 12-line ``classes.txt`` mimicking the real file's quirks.

    Reproduces the shipped file's CRLF endings and trailing tabs, plus a name
    containing spaces, so parsing is tested against the real shape of the data.
    """
    names = [
        "rice leaf roller",
        "rice leaf caterpillar",
        "paddy stem maggot",
        "asiatic rice borer",
        "yellow rice borer",
        "rice gall midge",
        "rice stemfly",
        "brown plant hopper",
        "white backed plant hopper",
        "small brown plant hopper",
        "rice water weevil",
        "rice leafhopper",
    ]
    lines = [f"{index} {name} \t\t" for index, name in enumerate(names, start=1)]
    path = tmp_path / "classes.txt"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return path


@pytest.fixture()
def classes(classes_file: Path) -> tuple[ClassInfo, ...]:
    """Parsed synthetic classes."""
    return read_classes(classes_file)


@pytest.fixture()
def source_records() -> tuple[tuple[str, int], ...]:
    """Two images for each of IP102 labels 0-11."""
    return tuple(
        (f"{label * 2 + offset:05d}.jpg", label)
        for label in range(12)
        for offset in range(2)
    )


# -- classes.txt --------------------------------------------------------


def test_read_classes_applies_the_off_by_one(classes: tuple[ClassInfo, ...]) -> None:
    """classes.txt id N describes IP102 label N-1. The single riskiest mapping."""
    assert len(classes) == 12
    for info in classes:
        assert info.ip102_label == info.classes_txt_id - 1
    assert classes[0].classes_txt_id == 1
    assert classes[0].ip102_label == 0
    assert classes[0].raw_name == "rice leaf roller"


def test_read_classes_strips_trailing_tabs_and_handles_crlf(
    classes: tuple[ClassInfo, ...],
) -> None:
    """The shipped file uses CRLF and pads names with tabs; neither may leak."""
    for info in classes:
        assert info.raw_name == info.raw_name.strip()
        assert "\t" not in info.raw_name
        assert "\r" not in info.raw_name


def test_read_classes_preserves_names_with_spaces(
    classes: tuple[ClassInfo, ...],
) -> None:
    """Splitting on the first space must not truncate multi-word names."""
    assert classes[8].raw_name == "white backed plant hopper"


def test_read_classes_canonical_name_is_normalised(tmp_path: Path) -> None:
    """The canonical form lower-cases and collapses whitespace, keeping raw intact."""
    path = tmp_path / "classes.txt"
    path.write_text("1  Sternochetus   Frigidus \n", encoding="utf-8")
    (info,) = read_classes(path)
    assert info.canonical_name == "sternochetus frigidus"
    assert "Sternochetus" in info.raw_name


def test_read_classes_returns_sorted_by_label(tmp_path: Path) -> None:
    path = tmp_path / "classes.txt"
    path.write_text("3 third\n1 first\n2 second\n", encoding="utf-8")
    classes = read_classes(path)
    assert [c.ip102_label for c in classes] == [0, 1, 2]
    assert [c.raw_name for c in classes] == ["first", "second", "third"]


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("1 first\n3 third\n", r"must be exactly 1\.\.2"),
        ("1 first\n1 duplicate\n", r"duplicate class id"),
        ("notanumber name\n", r"class id must be an integer"),
        ("1\n", r"expected '<id> <name>'"),
        ("1  \n", r"expected '<id> <name>'|class name is empty"),
        ("", r"contains no class definitions"),
    ],
)
def test_read_classes_rejects_malformed_files(
    tmp_path: Path, content: str, match: str
) -> None:
    path = tmp_path / "classes.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestError, match=match):
        read_classes(path)


def test_read_classes_enforces_expected_count(classes_file: Path) -> None:
    with pytest.raises(ManifestError, match=r"expected 102 classes, found 12"):
        read_classes(classes_file, expected=102)


def test_read_classes_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match=r"classes file not found"):
        read_classes(tmp_path / "absent.txt")


# -- source manifests ---------------------------------------------------


def test_read_source_manifest_preserves_order(tmp_path: Path) -> None:
    """Official split order is never reshuffled."""
    path = tmp_path / "train.txt"
    path.write_text("00009.jpg 3\n00002.jpg 0\n00005.jpg 1\n", encoding="utf-8")
    assert read_source_manifest(path) == (
        ("00009.jpg", 3),
        ("00002.jpg", 0),
        ("00005.jpg", 1),
    )


def test_read_source_manifest_tolerates_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "train.txt"
    path.write_text("00002.jpg 0\n\n00003.jpg 1\n", encoding="utf-8")
    assert len(read_source_manifest(path)) == 2


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("00002.jpg\n", r"expected '<filename> <label>'"),
        ("00002.jpg 0 extra\n", r"expected '<filename> <label>'"),
        ("00002.jpg abc\n", r"label must be an integer"),
        ("00002.jpg -1\n", r"label must be >= 0"),
        ("images/00002.jpg 0\n", r"filename must be bare"),
        ("00002.jpg 0\n00002.jpg 1\n", r"duplicate filename"),
        ("", r"contains no records"),
    ],
)
def test_read_source_manifest_rejects_malformed_files(
    tmp_path: Path, content: str, match: str
) -> None:
    path = tmp_path / "train.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestError, match=match):
        read_source_manifest(path)


# -- deriving -----------------------------------------------------------


def test_full102_is_an_identity_mapping(
    source_records: tuple[tuple[str, int], ...], classes: tuple[ClassInfo, ...]
) -> None:
    """For full102 nothing is dropped and project labels equal IP102 labels."""
    manifest = build_derived_manifest("train", source_records, FULL102, classes)
    assert len(manifest) == len(source_records)
    assert manifest.excluded_records == 0
    for record in manifest:
        assert record.project_label == record.ip102_label


def test_rice10_filters_and_remaps(
    source_records: tuple[tuple[str, int], ...], classes: tuple[ClassInfo, ...]
) -> None:
    """rice10 keeps ten classes and renumbers them 0-9 in the fixed order."""
    manifest = build_derived_manifest("train", source_records, RICE10, classes)

    # 12 labels x 2 images, minus the two excluded classes (IP102 2 and 6).
    assert len(manifest) == 20
    assert manifest.source_records == 24
    assert manifest.excluded_records == 4

    observed = {r.ip102_label: r.project_label for r in manifest}
    assert observed == {
        0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9
    }


def test_rice10_excludes_paddy_stem_maggot_and_rice_stemfly(
    source_records: tuple[tuple[str, int], ...], classes: tuple[ClassInfo, ...]
) -> None:
    """IP102 labels 2 and 6 sit in the same numeric range but are not rice10."""
    manifest = build_derived_manifest("train", source_records, RICE10, classes)
    present = {r.ip102_label for r in manifest}
    assert 2 not in present
    assert 6 not in present
    names = {r.class_name for r in manifest}
    assert "paddy stem maggot" not in names
    assert "rice stemfly" not in names


def test_derived_records_carry_the_class_name(
    source_records: tuple[tuple[str, int], ...], classes: tuple[ClassInfo, ...]
) -> None:
    manifest = build_derived_manifest("train", source_records, RICE10, classes)
    first = manifest.records[0]
    assert first.project_label == 0
    assert first.class_name == "rice leaf roller"
    assert first.relative_path == f"images/{first.filename}"


def test_build_preserves_source_order(
    classes: tuple[ClassInfo, ...],
) -> None:
    source = (("c.jpg", 1), ("a.jpg", 0), ("b.jpg", 1))
    manifest = build_derived_manifest("train", source, RICE10, classes)
    assert [r.filename for r in manifest] == ["c.jpg", "a.jpg", "b.jpg"]


def test_build_rejects_a_label_absent_from_classes_txt(
    classes: tuple[ClassInfo, ...],
) -> None:
    """A label with no class entry is a hard error, never a silent drop."""
    with pytest.raises(ManifestError, match=r"has no entry in classes.txt"):
        build_derived_manifest("train", (("x.jpg", 99),), FULL102, classes)


def test_build_rejects_an_unknown_split(classes: tuple[ClassInfo, ...]) -> None:
    with pytest.raises(ManifestError, match=r"unknown split 'val'"):
        build_derived_manifest("val", (), RICE10, classes)


def test_class_counts_include_empty_classes(classes: tuple[ClassInfo, ...]) -> None:
    """A class with no records must appear as zero, not vanish from the counts."""
    manifest = build_derived_manifest("train", (("a.jpg", 0),), RICE10, classes)
    counts = manifest.class_counts()
    assert len(counts) == 10
    assert counts[0] == 1
    assert all(counts[label] == 0 for label in range(1, 10))
    assert manifest.metadata()["classes_with_no_records"] == list(range(1, 10))


# -- writing and reading ------------------------------------------------


def test_round_trip_through_csv(
    tmp_path: Path,
    source_records: tuple[tuple[str, int], ...],
    classes: tuple[ClassInfo, ...],
) -> None:
    manifest = build_derived_manifest("validation", source_records, RICE10, classes)
    write_derived_manifest(manifest, tmp_path)

    records, metadata = read_derived_manifest(tmp_path, RICE10, "validation")
    assert records == manifest.records
    assert metadata["scope"] == "rice10"
    assert metadata["num_classes"] == 10
    assert metadata["class_mapping_version"] == CLASS_MAPPING_VERSION
    assert metadata["records"] == len(manifest)
    assert metadata["source_records"] == 24


def test_written_csv_has_the_expected_columns(
    tmp_path: Path,
    source_records: tuple[tuple[str, int], ...],
    classes: tuple[ClassInfo, ...],
) -> None:
    manifest = build_derived_manifest("train", source_records, RICE10, classes)
    path = write_derived_manifest(manifest, tmp_path)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(DERIVED_COLUMNS)


def test_written_csv_uses_lf_endings(
    tmp_path: Path,
    source_records: tuple[tuple[str, int], ...],
    classes: tuple[ClassInfo, ...],
) -> None:
    """Output must be byte-identical on Windows and in a Linux container."""
    manifest = build_derived_manifest("train", source_records, RICE10, classes)
    path = write_derived_manifest(manifest, tmp_path)
    assert b"\r\n" not in path.read_bytes()


def test_writing_is_idempotent(
    tmp_path: Path,
    source_records: tuple[tuple[str, int], ...],
    classes: tuple[ClassInfo, ...],
) -> None:
    """Rebuilding identical inputs must produce identical bytes."""
    manifest = build_derived_manifest("train", source_records, RICE10, classes)
    path = write_derived_manifest(manifest, tmp_path)
    first = path.read_bytes()
    write_derived_manifest(manifest, tmp_path)
    assert path.read_bytes() == first


def test_render_matches_what_is_written(
    tmp_path: Path,
    source_records: tuple[tuple[str, int], ...],
    classes: tuple[ClassInfo, ...],
) -> None:
    """The --check comparison relies on these two agreeing exactly."""
    manifest = build_derived_manifest("test", source_records, RICE10, classes)
    path = write_derived_manifest(manifest, tmp_path)
    assert path.read_text(encoding="utf-8") == render_manifest_csv(manifest)


def test_scope_is_part_of_the_manifest_path(tmp_path: Path) -> None:
    """rice10 and full102 artifacts must never collide."""
    rice = manifest_csv_path(tmp_path, RICE10, "train")
    full = manifest_csv_path(tmp_path, FULL102, "train")
    assert rice != full
    assert rice.parent.name == "rice10"
    assert full.parent.name == "full102"


def test_reading_rejects_a_scope_mismatch(
    tmp_path: Path,
    source_records: tuple[tuple[str, int], ...],
    classes: tuple[ClassInfo, ...],
) -> None:
    """A manifest built for one scope must not be readable as another."""
    manifest = build_derived_manifest("train", source_records, RICE10, classes)
    csv_path = write_derived_manifest(manifest, tmp_path)

    # Move the rice10 files into the full102 location, keeping the metadata.
    full_dir = tmp_path / "full102"
    full_dir.mkdir()
    (full_dir / "train.csv").write_bytes(csv_path.read_bytes())
    (full_dir / "train.metadata.json").write_bytes(
        csv_path.with_suffix(".metadata.json").read_bytes()
    )

    with pytest.raises(ManifestError, match=r"was built for scope 'rice10'"):
        read_derived_manifest(tmp_path, FULL102, "train")


def test_reading_rejects_a_stale_class_mapping_version(
    tmp_path: Path,
    source_records: tuple[tuple[str, int], ...],
    classes: tuple[ClassInfo, ...],
) -> None:
    """A manifest from an older mapping is rejected, not silently misread."""
    manifest = build_derived_manifest("train", source_records, RICE10, classes)
    csv_path = write_derived_manifest(manifest, tmp_path)
    meta_path = csv_path.with_suffix(".metadata.json")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["class_mapping_version"] = "0.9.0"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ManifestError, match=r"class mapping version"):
        read_derived_manifest(tmp_path, RICE10, "train")


def test_reading_requires_the_metadata_sidecar(
    tmp_path: Path,
    source_records: tuple[tuple[str, int], ...],
    classes: tuple[ClassInfo, ...],
) -> None:
    manifest = build_derived_manifest("train", source_records, RICE10, classes)
    csv_path = write_derived_manifest(manifest, tmp_path)
    csv_path.with_suffix(".metadata.json").unlink()
    with pytest.raises(ManifestError, match=r"manifest metadata not found"):
        read_derived_manifest(tmp_path, RICE10, "train")


def test_reading_a_missing_manifest_names_the_build_script(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match=r"build_manifests\.py"):
        read_derived_manifest(tmp_path, RICE10, "train")


# -- atomic writes ------------------------------------------------------


def test_atomic_write_creates_parents(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c.txt"
    atomic_write_text(path, "payload")
    assert path.read_text(encoding="utf-8") == "payload"


def test_atomic_write_replaces_existing_content(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    atomic_write_text(path, "first")
    atomic_write_text(path, "second")
    assert path.read_text(encoding="utf-8") == "second"


def test_atomic_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    atomic_write_text(tmp_path / "file.txt", "payload")
    assert [p.name for p in tmp_path.iterdir()] == ["file.txt"]


def test_atomic_write_does_not_translate_newlines(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    atomic_write_text(path, "a\nb\n")
    assert path.read_bytes() == b"a\nb\n"


# -- split names --------------------------------------------------------


def test_split_names_are_canonical() -> None:
    """The project says 'validation'; IP102's file is val.txt. Mapping is config."""
    assert SPLITS == ("train", "validation", "test")
