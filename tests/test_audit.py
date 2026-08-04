"""Tests for the dataset audit: integrity, duplicates, leakage and probes.

Synthetic images are generated with Pillow so the suite exercises real decode
paths without depending on ``ip102_v1.1``. Tests needing Pillow skip when it is
absent, keeping the harness usable in a bare environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from farm_pest_ai.data.audit import (
    DecodeResult,
    check_integrity,
    find_duplicates,
    find_leakage,
    hash_file,
    probe_image,
    probe_many,
    split_distribution,
    summarise_dimensions,
)
from farm_pest_ai.data.manifests import ManifestRecord

PIL = pytest.importorskip("PIL", reason="Pillow is required for image probes")


def make_image(path: Path, size: tuple[int, int], colour: tuple[int, int, int]) -> Path:
    """Write a small JPEG of the requested size and colour."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, format="JPEG", quality=95)
    return path


def record(
    filename: str, project_label: int, split: str = "train", ip102_label: int | None = None
) -> ManifestRecord:
    """Build a manifest record for tests."""
    return ManifestRecord(
        filename=filename,
        ip102_label=project_label if ip102_label is None else ip102_label,
        project_label=project_label,
        class_name=f"class {project_label}",
        split=split,
    )


# -- hashing ------------------------------------------------------------


def test_identical_bytes_hash_identically(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")
    assert hash_file(a) == hash_file(b)


def test_different_bytes_hash_differently(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert hash_file(a) != hash_file(b)


def test_hashing_is_chunk_size_independent(tmp_path: Path) -> None:
    """A large file must hash the same however it is chunked."""
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * 100_000)
    assert hash_file(path, chunk_bytes=7) == hash_file(path, chunk_bytes=1 << 20)


# -- probing ------------------------------------------------------------


def test_probe_reports_dimensions_and_format(tmp_path: Path) -> None:
    path = make_image(tmp_path / "img.jpg", (320, 240), (10, 120, 30))
    probe = probe_image(path)
    assert probe.ok
    assert (probe.width, probe.height) == (320, 240)
    assert probe.image_format == "JPEG"
    assert probe.mode == "RGB"
    assert probe.short_side == 240
    assert probe.size_bytes and probe.size_bytes > 0


def test_probe_flags_images_needing_upscale(tmp_path: Path) -> None:
    """The sub-160px cohort drives the Phase 5 interpolation decision."""
    small = probe_image(make_image(tmp_path / "s.jpg", (400, 120), (1, 2, 3)))
    large = probe_image(make_image(tmp_path / "l.jpg", (400, 300), (1, 2, 3)))
    assert small.needs_upscale
    assert not large.needs_upscale


def test_probe_computes_a_hash_only_when_asked(tmp_path: Path) -> None:
    path = make_image(tmp_path / "img.jpg", (64, 64), (5, 5, 5))
    assert probe_image(path, compute_hash=False).sha256 is None
    assert probe_image(path, compute_hash=True).sha256 is not None


def test_probe_reports_failure_rather_than_raising(tmp_path: Path) -> None:
    """One corrupt file must not abort an audit of 75,222 images."""
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is definitely not a jpeg")
    probe = probe_image(path)
    assert not probe.ok
    assert probe.error
    assert probe.short_side is None
    assert not probe.needs_upscale


def test_probe_reports_a_missing_file(tmp_path: Path) -> None:
    probe = probe_image(tmp_path / "absent.jpg")
    assert not probe.ok
    assert "stat failed" in (probe.error or "")


def test_full_decode_detects_a_truncated_jpeg(tmp_path: Path) -> None:
    """Header-only reads miss truncation; this is why the audit decodes fully."""
    path = make_image(tmp_path / "img.jpg", (256, 256), (200, 30, 30))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    assert not probe_image(path, full_decode=True).ok
    # The header still parses, which is precisely the blind spot being covered.
    assert probe_image(path, full_decode=False).ok


def test_probe_many_reports_progress(tmp_path: Path) -> None:
    for index in range(5):
        make_image(tmp_path / f"{index}.jpg", (64, 64), (index, index, index))
    records = [record(f"{index}.jpg", 0) for index in range(5)]

    seen: list[tuple[int, int]] = []
    probes = probe_many(
        records,
        tmp_path,
        progress=lambda done, total: seen.append((done, total)),
        progress_every=2,
    )
    assert len(probes) == 5
    assert (4, 5) in seen
    assert seen[-1] == (5, 5)


# -- integrity ----------------------------------------------------------


def test_integrity_passes_on_a_clean_dataset(tmp_path: Path) -> None:
    images = tmp_path / "images"
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        make_image(images / name, (32, 32), (1, 1, 1))

    report = check_integrity(
        {"train": [("a.jpg", 0), ("b.jpg", 1)], "test": [("c.jpg", 0)]}, images
    )
    assert report.ok
    assert report.total_records == 3
    assert report.images_on_disk == 3


def test_integrity_detects_a_missing_image(tmp_path: Path) -> None:
    images = tmp_path / "images"
    make_image(images / "a.jpg", (32, 32), (1, 1, 1))
    report = check_integrity({"train": [("a.jpg", 0), ("gone.jpg", 1)]}, images)
    assert not report.ok
    assert report.missing_from_disk == ("gone.jpg",)


def test_integrity_detects_unreferenced_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    make_image(images / "a.jpg", (32, 32), (1, 1, 1))
    make_image(images / "orphan.jpg", (32, 32), (2, 2, 2))
    report = check_integrity({"train": [("a.jpg", 0)]}, images)
    assert report.unreferenced_on_disk == ("orphan.jpg",)


def test_integrity_can_skip_the_unreferenced_check(tmp_path: Path) -> None:
    """Auditing a scope subset legitimately leaves most files unreferenced."""
    images = tmp_path / "images"
    make_image(images / "a.jpg", (32, 32), (1, 1, 1))
    make_image(images / "orphan.jpg", (32, 32), (2, 2, 2))
    report = check_integrity(
        {"train": [("a.jpg", 0)]}, images, check_unreferenced=False
    )
    assert report.unreferenced_on_disk == ()
    assert report.ok


def test_integrity_detects_a_filename_in_two_splits(tmp_path: Path) -> None:
    images = tmp_path / "images"
    make_image(images / "a.jpg", (32, 32), (1, 1, 1))
    report = check_integrity(
        {"train": [("a.jpg", 0)], "test": [("a.jpg", 0)]}, images
    )
    assert not report.ok
    assert report.cross_split_filenames == ("a.jpg",)


def test_integrity_detects_conflicting_labels(tmp_path: Path) -> None:
    images = tmp_path / "images"
    make_image(images / "a.jpg", (32, 32), (1, 1, 1))
    report = check_integrity(
        {"train": [("a.jpg", 0)], "test": [("a.jpg", 7)]}, images
    )
    assert not report.ok
    assert report.conflicting_labels == {"a.jpg": (0, 7)}


# -- duplicates and leakage ---------------------------------------------


def test_no_duplicates_when_every_hash_is_distinct() -> None:
    hashes = {"train": {"a.jpg": "h1", "b.jpg": "h2"}, "test": {"c.jpg": "h3"}}
    labels = {"train": {"a.jpg": 0, "b.jpg": 1}, "test": {"c.jpg": 0}}
    assert find_duplicates(hashes, labels) == ()


def test_within_split_duplicates_are_not_leakage() -> None:
    """A duplicated training image is a weighting issue, not contamination."""
    hashes = {"train": {"a.jpg": "same", "b.jpg": "same"}}
    labels = {"train": {"a.jpg": 3, "b.jpg": 3}}
    report = find_leakage(hashes, labels, "rice10")

    assert len(report.duplicate_groups) == 1
    assert len(report.within_split_groups) == 1
    assert report.cross_split_groups == ()
    assert report.leaked_files("train") == ()


def test_cross_split_duplicates_are_leakage() -> None:
    """Identical bytes in train and test inflate any headline test metric."""
    hashes = {"train": {"a.jpg": "same"}, "test": {"b.jpg": "same"}}
    labels = {"train": {"a.jpg": 3}, "test": {"b.jpg": 3}}
    report = find_leakage(hashes, labels, "rice10")

    assert len(report.cross_split_groups) == 1
    group = report.cross_split_groups[0]
    assert group.crosses_splits
    assert group.splits == ("test", "train")
    assert report.leaked_files("test") == ("b.jpg",)
    assert report.leaked_files("train") == ("a.jpg",)


def test_label_conflicts_are_reported() -> None:
    """The same bytes cannot be two classes; one annotation must be wrong."""
    hashes = {"train": {"a.jpg": "same", "b.jpg": "same"}}
    labels = {"train": {"a.jpg": 3, "b.jpg": 8}}
    report = find_leakage(hashes, labels, "rice10")

    assert len(report.label_conflict_groups) == 1
    assert report.label_conflict_groups[0].labels == (3, 8)


def test_leakage_summary_counts_every_category() -> None:
    hashes = {
        "train": {"a.jpg": "x", "b.jpg": "x", "c.jpg": "y"},
        "test": {"d.jpg": "y", "e.jpg": "z"},
    }
    labels = {
        "train": {"a.jpg": 0, "b.jpg": 0, "c.jpg": 1},
        "test": {"d.jpg": 1, "e.jpg": 2},
    }
    summary = find_leakage(hashes, labels, "rice10").summary()

    assert summary["duplicate_groups"] == 2
    assert summary["within_split_groups"] == 1
    assert summary["cross_split_groups"] == 1
    assert summary["duplicate_files"] == 4
    assert summary["leaked_files_per_split"]["test"] == 1


def test_duplicate_groups_are_ordered_worst_first() -> None:
    hashes = {"train": {"a.jpg": "x", "b.jpg": "x", "c.jpg": "x", "d.jpg": "y", "e.jpg": "y"}}
    labels = {"train": dict.fromkeys(["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"], 0)}
    groups = find_duplicates(hashes, labels)
    assert [len(g.members) for g in groups] == [3, 2]


def test_duplicates_found_in_real_files(tmp_path: Path) -> None:
    """End-to-end: byte-identical files on disk are detected as duplicates."""
    images = tmp_path / "images"
    make_image(images / "a.jpg", (64, 64), (9, 9, 9))
    (images / "copy.jpg").write_bytes((images / "a.jpg").read_bytes())
    make_image(images / "other.jpg", (64, 64), (200, 10, 10))

    records = [
        record("a.jpg", 0, "train"),
        record("other.jpg", 1, "train"),
        record("copy.jpg", 0, "test"),
    ]
    probes = {r.filename: probe_image(images / r.filename, compute_hash=True) for r in records}
    hashes = {
        "train": {n: p.sha256 for n, p in probes.items() if n != "copy.jpg" if p.sha256},
        "test": {"copy.jpg": probes["copy.jpg"].sha256 or ""},
    }
    labels = {"train": {"a.jpg": 0, "other.jpg": 1}, "test": {"copy.jpg": 0}}

    report = find_leakage(hashes, labels, "rice10")
    assert len(report.cross_split_groups) == 1
    assert report.leaked_files("test") == ("copy.jpg",)


# -- distributions ------------------------------------------------------


def test_distribution_includes_classes_with_no_records() -> None:
    records = [record("a.jpg", 0), record("b.jpg", 0), record("c.jpg", 2)]
    dist = split_distribution("train", records, num_classes=4)

    assert dist.total == 3
    assert dist.counts == {0: 2, 1: 0, 2: 1, 3: 0}
    assert dist.present_classes == 2
    assert dist.empty_classes == (1, 3)


def test_distribution_imbalance_ratio_ignores_empty_classes() -> None:
    """Dividing by an empty class would be undefined, not infinitely imbalanced."""
    records = [record(f"{i}.jpg", 0) for i in range(10)] + [record("x.jpg", 1)]
    dist = split_distribution("train", records, num_classes=3)
    assert dist.imbalance_ratio == pytest.approx(10.0)


def test_distribution_of_an_empty_split() -> None:
    dist = split_distribution("train", [], num_classes=3)
    assert dist.total == 0
    assert dist.imbalance_ratio is None
    assert dist.to_dict()["min"] == 0


def test_distribution_records_class_names() -> None:
    dist = split_distribution("train", [record("a.jpg", 1)], num_classes=2)
    assert dist.class_names[1] == "class 1"


# -- dimension summaries ------------------------------------------------


def test_dimension_summary_counts_the_upscale_cohort(tmp_path: Path) -> None:
    sizes = [(400, 100), (400, 150), (400, 200), (400, 300)]
    probes = [
        probe_image(make_image(tmp_path / f"{i}.jpg", size, (i, i, i)))
        for i, size in enumerate(sizes)
    ]
    summary = summarise_dimensions(probes)

    assert summary["count"] == 4
    # Short sides 100 and 150 fall below the 160px model input.
    assert summary["short_side_below_160"] == 2
    assert summary["short_side_below_160_pct"] == 50.0
    assert summary["short_side_below_224"] == 3
    assert summary["short_side"]["min"] == 100
    assert summary["short_side"]["max"] == 300


def test_dimension_summary_ignores_failed_probes(tmp_path: Path) -> None:
    good = probe_image(make_image(tmp_path / "ok.jpg", (300, 300), (1, 1, 1)))
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not a jpeg")
    summary = summarise_dimensions([good, probe_image(broken)])
    assert summary["count"] == 1


def test_dimension_summary_of_nothing() -> None:
    assert summarise_dimensions([]) == {"count": 0}


def test_decode_result_separates_failures(tmp_path: Path) -> None:
    good = probe_image(make_image(tmp_path / "ok.jpg", (64, 64), (1, 1, 1)))
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"nope")
    result = DecodeResult(probes=(good, probe_image(broken)), split="train")

    assert len(result.ok_probes) == 1
    assert len(result.failures) == 1
    summary = result.to_dict()
    assert summary["decoded"] == 1
    assert summary["failed"] == 1
    assert summary["formats"] == {"JPEG": 1}
