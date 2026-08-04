"""Tests for the Phase 8.1 Stage 1 analysis.

The judgement calls this script makes are the ones worth pinning: how a
difference is classified against the noise thresholds, that a mixed-training
arm's train-validation gap is flagged as incomparable rather than tabulated as
if it meant the same thing, and that the rice10 confusion groups name the
classes they claim to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_phase81 import (  # noqa: E402
    NOISE_DELTA,
    RICE10_ARMS,
    RICE10_CONFUSION_GROUPS,
    SEED_NOISE_DELTA,
    _last_n,
)


def test_noise_thresholds_match_the_established_evidence() -> None:
    """0.01 is the reporting threshold; 0.02 is the seed-stability threshold.

    Both come from measurements: Phase 7.2 set 0.01 as indistinguishable on a
    721-image split, and E4 showed a +0.0138 single-seed margin falling to
    +0.0079 and reversing on one of three seeds.
    """
    assert NOISE_DELTA == 0.01
    assert SEED_NOISE_DELTA == 0.02
    assert SEED_NOISE_DELTA > NOISE_DELTA


def test_last_n_reports_mean_and_sample_deviation() -> None:
    """The late-run statistics are a plain mean and sample sd of the tail."""
    values = [float(v) for v in range(1, 21)]
    mean, sd = _last_n(values, n=10)

    assert mean == pytest.approx(15.5)
    assert sd == pytest.approx(3.0276503541, abs=1e-6)


def test_last_n_handles_a_short_series() -> None:
    """A run that stopped early still yields statistics rather than raising."""
    mean, sd = _last_n([0.5], n=10)
    assert mean == pytest.approx(0.5)
    assert sd == 0.0


def test_last_n_ignores_a_missing_tail() -> None:
    """An empty series yields zeros rather than a division error."""
    assert _last_n([], n=10) == (0.0, 0.0)


def test_confusion_groups_name_the_documented_taxa() -> None:
    """The group indices must match the real rice10 class mapping.

    Hard-coding indices is exactly the kind of thing that silently drifts, and
    a wrong index would attribute E8's per-group result to the wrong classes.
    """
    mapping_path = (
        REPO_ROOT / "data" / "processed" / "rice10" / "class_mapping.json"
    )
    if not mapping_path.is_file():
        pytest.skip("rice10 class mapping is not built")

    names = {
        entry["project_label"]: entry["canonical_name"]
        for entry in json.loads(mapping_path.read_text(encoding="utf-8"))["classes"]
    }

    hoppers = RICE10_CONFUSION_GROUPS[
        "plant hoppers (brown / white-backed / small brown)"
    ]
    assert {names[i] for i in hoppers} == {
        "brown plant hopper",
        "white backed plant hopper",
        "small brown plant hopper",
    }

    borers = RICE10_CONFUSION_GROUPS["borers (asiatic / yellow rice)"]
    assert {names[i] for i in borers} == {"asiatic rice borer", "yellow rice borer"}

    leaves = RICE10_CONFUSION_GROUPS["leaf roller vs leaf caterpillar"]
    assert {names[i] for i in leaves} == {
        "rice leaf roller",
        "rice leaf caterpillar",
    }


def test_confusion_groups_are_disjoint_and_in_range() -> None:
    """No class belongs to two groups, and every index is a real rice10 label."""
    seen: set[int] = set()
    for labels in RICE10_CONFUSION_GROUPS.values():
        assert not (seen & set(labels)), "a class appears in two confusion groups"
        seen.update(labels)
        assert all(0 <= label < 10 for label in labels)


def test_the_control_is_listed_first() -> None:
    """Deltas are computed against the first arm, so E0 must lead the list."""
    assert RICE10_ARMS[0][0] == "E0"
    assert "control" in RICE10_ARMS[0][2]


def test_analysis_never_names_the_test_split() -> None:
    """The Stage 1 analysis is validation-only."""
    import analyze_phase81

    source = Path(analyze_phase81.__file__).read_text(encoding="utf-8")
    body = "\n".join(
        line
        for line in source.splitlines()
        if not line.strip().startswith("#")
    )
    assert 'split="test"' not in body
    assert '"test"' not in body.replace('"test split"', "")


@pytest.mark.skipif(
    not (
        REPO_ROOT / "data" / "reports" / "phase81_stage1_rice10.json"
    ).is_file(),
    reason="the Stage 1 rice10 report has not been generated",
)
def test_generated_report_flags_mixed_arms_as_incomparable() -> None:
    """A mixed arm's gap must be marked as not comparable with the control's.

    E7a records a *negative* train-validation accuracy gap purely because its
    training accuracy is measured on blended images. Reported without the flag,
    that reads as a model generalising better than it fits, which is false.
    """
    report = json.loads(
        (REPO_ROOT / "data" / "reports" / "phase81_stage1_rice10.json").read_text(
            encoding="utf-8"
        )
    )
    arms = {block["label"]: block for block in report["arms"]}

    for label in ("E7a", "E7b"):
        if label not in arms:
            continue
        assert arms[label]["mixed_training"] is True
        assert arms[label]["gap_comparable_to_control"] is False

    for label in ("E0", "E6a", "E6b", "E8"):
        if label not in arms:
            continue
        assert arms[label]["mixed_training"] is False
        assert arms[label]["gap_comparable_to_control"] is True


@pytest.mark.skipif(
    not (
        REPO_ROOT
        / "artifacts"
        / "checkpoints"
        / "full102_custom_protocolA"
        / "best.json"
    ).is_file(),
    reason="the full102 control run is not present",
)
def test_per_class_falls_back_to_the_sidecar_for_full102() -> None:
    """full102 records no per-class arrays per epoch, so the sidecar supplies them.

    Without the fallback the support-quartile analysis silently produces empty
    groups — the failure that motivated it.
    """
    from analyze_phase81 import _per_class_from_sidecar
    from farm_pest_ai.vision.results import load_run

    run = load_run(
        REPO_ROOT / "artifacts" / "checkpoints" / "full102_custom_protocolA"
    )
    best_epoch = run.best_epoch(corrected=True)
    assert best_epoch is not None

    # The per-epoch records genuinely lack the arrays for this scope.
    best = run.best_validation(corrected=True)
    assert best is not None
    assert not best.corrected_per_class_f1

    f1, support = _per_class_from_sidecar(run, best_epoch)

    assert len(f1) == 102
    assert len(support) == 102
    assert all(0.0 <= v <= 1.0 for v in f1)
    # The recorded support must account for the whole validation split.
    assert sum(support) == 7508


@pytest.mark.skipif(
    not (
        REPO_ROOT
        / "artifacts"
        / "checkpoints"
        / "full102_custom_protocolA"
        / "best.json"
    ).is_file(),
    reason="the full102 control run is not present",
)
def test_sidecar_fallback_refuses_a_mismatched_epoch() -> None:
    """A sidecar describing a different epoch is not silently misattributed."""
    from analyze_phase81 import _per_class_from_sidecar
    from farm_pest_ai.vision.results import load_run

    run = load_run(
        REPO_ROOT / "artifacts" / "checkpoints" / "full102_custom_protocolA"
    )
    best_epoch = run.best_epoch(corrected=True)
    assert best_epoch is not None

    assert _per_class_from_sidecar(run, best_epoch + 1) == ([], [])


@pytest.mark.skipif(
    not (
        REPO_ROOT / "data" / "reports" / "phase81_stage1_rice10.json"
    ).is_file(),
    reason="the Stage 1 rice10 report has not been generated",
)
def test_generated_report_records_no_test_split_use() -> None:
    """The report asserts its own discipline."""
    report = json.loads(
        (REPO_ROOT / "data" / "reports" / "phase81_stage1_rice10.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["test_split_used"] is False
    assert report["retrained"] is False
    assert report["scope"] == "rice10"
