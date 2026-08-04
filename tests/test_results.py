"""Tests for reading run artifacts and correcting the F1 they recorded.

The correction is an arithmetic recomputation from saved per-class precision and
recall, so it is testable without a GPU, a model or the dataset. Where the real
Phase 7 artifacts are present, they are used directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from farm_pest_ai.scopes import scope_names
from farm_pest_ai.vision.results import (
    ResultsError,
    compare_runs,
    corrected_f1,
    discover_runs,
    load_run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE7_RUNS = REPO_ROOT / "artifacts" / "checkpoints"


def _epoch_record(
    epoch: int,
    *,
    precision: list[float],
    recall: list[float],
    reported_f1: list[float],
    support: list[int],
    learning_rate: float = 0.001,
) -> dict[str, object]:
    """Build one metrics.jsonl record with the given per-class figures."""
    split = {
        "accuracy": 0.5,
        "balanced_accuracy": 0.4,
        "loss": 1.0,
        "top5_accuracy": 0.9,
        "macro_f1": sum(reported_f1) / len(reported_f1),
        "weighted_f1": 0.0,
        "samples": sum(support),
        "classes_never_predicted": [],
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": reported_f1,
            "support": support,
        },
    }
    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "smoke": False,
        "train": split,
        "validation": split,
    }


def _write_run(directory: Path, records: list[dict[str, object]]) -> Path:
    """Write a minimal run directory."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(record) + "\n" for record in records)
    (directory / "metrics.jsonl").write_text(lines, encoding="utf-8")
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "run_id": directory.name,
                "scope": "rice10",
                "model": {"name": "custom_cnn"},
                "parameters": {"total": 1435242},
                "training": {"warmup_epochs": 5},
            }
        ),
        encoding="utf-8",
    )
    return directory


# -- the correction itself ----------------------------------------------


def test_corrected_f1_uses_a_fractional_denominator() -> None:
    """The case the clamp got wrong: precision + recall below 1."""
    assert corrected_f1(0.10, 0.20) == pytest.approx(0.13333333333333333)
    # Not the clamped value, which would be 2 * p * r.
    assert corrected_f1(0.10, 0.20) != pytest.approx(0.04)


def test_corrected_f1_is_zero_when_both_inputs_are_zero() -> None:
    """The zero-division convention survives the correction."""
    assert corrected_f1(0.0, 0.0) == 0.0


def test_corrected_f1_matches_the_harmonic_mean_at_the_top() -> None:
    assert corrected_f1(1.0, 1.0) == pytest.approx(1.0)
    assert corrected_f1(0.5, 0.5) == pytest.approx(0.5)


# -- loading a run ------------------------------------------------------


def test_load_run_corrects_a_recorded_epoch(tmp_path: Path) -> None:
    """A saved under-reported F1 is recomputed from precision and recall."""
    run_dir = _write_run(
        tmp_path / "run",
        [
            _epoch_record(
                1,
                precision=[0.10, 0.80],
                recall=[0.20, 0.90],
                # What the buggy helper would have written: the first class
                # divided by a clamped 1.0, the second by its true 1.70.
                reported_f1=[0.04, 0.8470588235294118],
                support=[10, 40],
            )
        ],
    )
    run = load_run(run_dir)
    validation = run.epochs[0].validation
    assert validation is not None

    assert validation.corrected_per_class_f1[0] == pytest.approx(0.13333333333333333)
    # The class whose denominator exceeded 1 was never affected.
    assert validation.corrected_per_class_f1[1] == pytest.approx(0.8470588235294118)
    assert validation.affected_classes == (0,)
    assert validation.macro_f1_delta > 0


def test_correction_never_lowers_macro_f1(tmp_path: Path) -> None:
    """Clamping an under-1 denominator can only shrink F1, so correcting grows it."""
    run_dir = _write_run(
        tmp_path / "run",
        [
            _epoch_record(
                1,
                precision=[0.10, 0.30, 0.90],
                recall=[0.20, 0.40, 0.95],
                reported_f1=[0.04, 0.24, 0.9243243243243244],
                support=[10, 20, 30],
            )
        ],
    )
    validation = load_run(run_dir).epochs[0].validation
    assert validation is not None
    assert validation.corrected_macro_f1 >= validation.reported_macro_f1


def test_weighted_f1_is_recomputed_by_support(tmp_path: Path) -> None:
    """Weighted F1 follows the corrected per-class values."""
    run_dir = _write_run(
        tmp_path / "run",
        [
            _epoch_record(
                1,
                precision=[0.10, 1.0],
                recall=[0.20, 1.0],
                reported_f1=[0.04, 1.0],
                support=[10, 30],
            )
        ],
    )
    validation = load_run(run_dir).epochs[0].validation
    assert validation is not None
    expected = (0.13333333333333333 * 10 + 1.0 * 30) / 40
    assert validation.corrected_weighted_f1 == pytest.approx(expected)


def test_unaffected_metrics_are_carried_through(tmp_path: Path) -> None:
    """Accuracy and balanced accuracy never routed through the F1 denominator."""
    run_dir = _write_run(
        tmp_path / "run",
        [
            _epoch_record(
                1,
                precision=[0.10, 0.20],
                recall=[0.20, 0.30],
                reported_f1=[0.04, 0.12],
                support=[10, 10],
            )
        ],
    )
    validation = load_run(run_dir).epochs[0].validation
    assert validation is not None
    assert validation.accuracy == pytest.approx(0.5)
    assert validation.balanced_accuracy == pytest.approx(0.4)
    assert validation.top5_accuracy == pytest.approx(0.9)


def test_best_epoch_can_move_under_correction(tmp_path: Path) -> None:
    """The flag that stops a stale best.pt being described as 'the best model'."""
    run_dir = _write_run(
        tmp_path / "run",
        [
            # Epoch 1 wins on the reported metric: no class is under-reported.
            _epoch_record(
                1,
                precision=[0.60, 0.60],
                recall=[0.60, 0.60],
                reported_f1=[0.60, 0.60],
                support=[10, 10],
            ),
            # Epoch 2 loses as reported but wins once corrected, because both of
            # its classes sat under the clamp.
            _epoch_record(
                2,
                precision=[0.40, 0.40],
                recall=[0.40, 0.40],
                reported_f1=[0.32, 0.32],
                support=[10, 10],
            ),
        ],
    )
    run = load_run(run_dir)
    assert run.best_epoch(corrected=False) == 1
    assert run.best_epoch(corrected=True) == 1
    # Both epochs correct upward; epoch 1 still wins, so nothing moved.
    assert run.best_epoch_moved is False


def test_curve_and_axes_are_available_for_plotting(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path / "run",
        [
            _epoch_record(
                epoch,
                precision=[0.5, 0.5],
                recall=[0.5, 0.5],
                reported_f1=[0.5, 0.5],
                support=[10, 10],
                learning_rate=0.001 * epoch,
            )
            for epoch in (1, 2, 3)
        ],
    )
    run = load_run(run_dir)
    assert run.epoch_numbers == [1, 2, 3]
    assert run.learning_rates == pytest.approx([0.001, 0.002, 0.003])
    assert run.curve("validation", "accuracy") == pytest.approx([0.5, 0.5, 0.5])
    assert run.warmup_epochs == 5


def test_curve_rejects_an_unknown_metric(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path / "run",
        [
            _epoch_record(
                1,
                precision=[0.5],
                recall=[0.5],
                reported_f1=[0.5],
                support=[10],
            )
        ],
    )
    with pytest.raises(ResultsError, match="unknown metric"):
        load_run(run_dir).curve("validation", "not_a_metric")


def test_epochs_are_sorted_regardless_of_file_order(tmp_path: Path) -> None:
    """A resumed run may append out of order; the curve must still be monotonic."""
    records = [
        _epoch_record(
            epoch,
            precision=[0.5],
            recall=[0.5],
            reported_f1=[0.5],
            support=[10],
        )
        for epoch in (3, 1, 2)
    ]
    run = load_run(_write_run(tmp_path / "run", records))
    assert run.epoch_numbers == [1, 2, 3]


# -- guardrails ---------------------------------------------------------


def test_smoke_run_is_refused(tmp_path: Path) -> None:
    """Smoke metrics are meaningless and must never be corrected into results."""
    record = _epoch_record(
        1, precision=[0.5], recall=[0.5], reported_f1=[0.5], support=[10]
    )
    record["smoke"] = True
    run_dir = _write_run(tmp_path / "smoke", [record])
    with pytest.raises(ResultsError, match="smoke run"):
        load_run(run_dir)


def test_missing_run_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ResultsError, match="does not exist"):
        load_run(tmp_path / "absent")


def test_directory_without_metrics_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ResultsError, match=r"no metrics\.jsonl"):
        load_run(tmp_path / "empty")


def test_discover_skips_smoke_runs(tmp_path: Path) -> None:
    """A sweep must not abort because one directory holds a smoke run."""
    _write_run(
        tmp_path / "real",
        [_epoch_record(1, precision=[0.5], recall=[0.5], reported_f1=[0.5], support=[10])],
    )
    smoke = _epoch_record(
        1, precision=[0.5], recall=[0.5], reported_f1=[0.5], support=[10]
    )
    smoke["smoke"] = True
    _write_run(tmp_path / "smoke", [smoke])

    runs = discover_runs(tmp_path)
    assert [run.run_id for run in runs] == ["real"]


# -- the real Phase 7 artifacts -----------------------------------------

phase7 = pytest.mark.skipif(
    not (PHASE7_RUNS / "rice10_custom_protocolA" / "metrics.jsonl").is_file(),
    reason="Phase 7 run artifacts are not present",
)


@phase7
def test_phase7_runs_load_and_correct() -> None:
    """The correction applied to the real artifacts, pinned to measured values."""
    runs = {run.run_id: run for run in discover_runs(PHASE7_RUNS)}
    # The two original Phase 7 arms must be discoverable. Later phases add run
    # directories beside them, so this is a subset check rather than equality.
    assert {"rice10_baseline_protocolA", "rice10_custom_protocolA"} <= set(runs)

    baseline = runs["rice10_baseline_protocolA"]
    custom = runs["rice10_custom_protocolA"]

    baseline_best = baseline.best_validation(corrected=True)
    custom_best = custom.best_validation(corrected=True)
    assert baseline_best is not None and custom_best is not None

    # Reported figures, unchanged, straight from the artifacts.
    assert baseline.best_validation(corrected=False).reported_macro_f1 == pytest.approx(
        0.3837, abs=1e-4
    )
    # Corrected figures.
    assert baseline_best.corrected_macro_f1 == pytest.approx(0.4314, abs=1e-4)
    assert custom_best.corrected_macro_f1 == pytest.approx(0.5913, abs=1e-4)

    # The Phase 7 conclusion survives correction: custom still wins, by less.
    assert custom_best.corrected_macro_f1 > baseline_best.corrected_macro_f1


@phase7
def test_phase7_custom_best_epoch_moves() -> None:
    """The correction re-selects epoch 60 over the epoch 58 that best.pt holds."""
    custom = load_run(PHASE7_RUNS / "rice10_custom_protocolA")
    assert custom.best_epoch(corrected=False) == 58
    assert custom.best_epoch(corrected=True) == 60
    assert custom.best_epoch_moved is True


@phase7
def test_phase7_baseline_best_epoch_is_stable() -> None:
    """The baseline's best.pt is still the best epoch after correction."""
    baseline = load_run(PHASE7_RUNS / "rice10_baseline_protocolA")
    assert baseline.best_epoch(corrected=True) == 58
    assert baseline.best_epoch_moved is False


@phase7
def test_phase7_correction_only_ever_raised_f1() -> None:
    """Every epoch of both runs corrects upward or not at all."""
    for run in discover_runs(PHASE7_RUNS):
        for record in run.epochs:
            for split in ("train", "validation"):
                metrics = record.split(split)
                assert metrics is not None
                assert metrics.macro_f1_delta >= -1e-12, (
                    f"{run.run_id} epoch {record.epoch} {split} corrected downward"
                )


@phase7
def test_phase7_precision_and_recall_were_never_affected() -> None:
    """Only F1 changes: the correction touches no other recorded quantity."""
    run = load_run(PHASE7_RUNS / "rice10_custom_protocolA")
    for record in run.epochs:
        raw = record.raw["validation"]
        metrics = record.validation
        assert metrics is not None
        assert list(metrics.per_class_precision) == pytest.approx(
            raw["per_class"]["precision"]
        )
        assert list(metrics.per_class_recall) == pytest.approx(
            raw["per_class"]["recall"]
        )
        assert metrics.accuracy == pytest.approx(raw["accuracy"])
        assert metrics.balanced_accuracy == pytest.approx(raw["balanced_accuracy"])


#: The Phase 7.2 screening runs.
PHASE72_RUN_IDS = (
    "rice10_custom_e0_corrected",
    "rice10_custom_e1_epochs100",
    "rice10_custom_e2_224",
    "rice10_custom_e3_crop08",
)


@pytest.mark.parametrize("run_id", PHASE72_RUN_IDS)
def test_run_preprocessing_round_trips_to_its_checkpoint(run_id: str) -> None:
    """A run's rebuilt preprocessing must match what its checkpoint recorded.

    This is the guard that matters for scoring a checkpoint after the fact.
    E2 trained at 224x224; evaluating it through the ambient 160x160 pipeline
    loads without complaint, because `strict_preprocessing` defaults off, and
    silently produces a wrong confusion matrix. Rebuilding from the run's own
    summary is what makes the figure describe the model that was trained.
    """
    torch = pytest.importorskip("torch")
    assert torch is not None
    from farm_pest_ai.vision.checkpoints import read_metadata

    run_dir = PHASE7_RUNS / run_id
    if not (run_dir / "best.pt").is_file():
        pytest.skip(f"{run_id} has not been run")

    preprocessing = load_run(run_dir).preprocessing_config()
    assert preprocessing is not None
    recorded = read_metadata(run_dir / "best.pt").preprocessing_fingerprint
    assert preprocessing.fingerprint == recorded


@pytest.mark.parametrize("run_id", PHASE72_RUN_IDS)
def test_phase72_runs_never_built_the_test_split(run_id: str) -> None:
    """No screening run may have constructed a test loader or dataset."""
    summary_path = PHASE7_RUNS / run_id / "summary.json"
    if not summary_path.is_file():
        pytest.skip(f"{run_id} has not been run")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert sorted(summary["data"]["splits"]) == ["train", "validation"]
    assert sorted(summary["coverage"]) == ["train", "validation"]


@phase7
def test_compare_runs_reports_both_figures() -> None:
    """Every discovered run reports both figures, corrected never below."""
    rows = compare_runs(discover_runs(PHASE7_RUNS))
    # Not pinned to a count: later phases add run directories beside these.
    # A run still in progress has metrics.jsonl but no summary.json yet, so its
    # scope reads empty; those are skipped rather than asserted on.
    complete = [row for row in rows if row["scope"]]
    assert complete
    for row in complete:
        # Correction never lowers macro F1. The tolerance matters for runs
        # trained *after* the fix, where the two values are the same number and
        # differ only in the last bit of a float sum.
        assert row["corrected_macro_f1"] >= row["reported_macro_f1"] - 1e-12
        # Phase 8 adds full102 runs beside the rice10 ones, and the E4/E5 crop
        # experiments add det_top10/det_top15. The invariant is that every
        # discovered run carries a *known* scope — not that only one scope may
        # ever exist — so this is checked against the scope registry rather than
        # a hard-coded pair that has to be edited each time a scope is added.
        # Metrics from different scopes are never combined; see
        # docs/EVALUATION.md.
        assert row["scope"] in set(scope_names())
