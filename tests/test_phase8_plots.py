"""Tests for the Phase 8 validation figures.

The properties that matter here are safety properties, not cosmetic ones: a
Phase 8 figure must never be built from test data, and a per-class figure must
read the checkpoint sidecar rather than silently drawing nothing because
``metrics.jsonl`` omits per-class arrays for ``full102``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from farm_pest_ai.vision.plots import PlotError

pytest.importorskip("matplotlib")

from scripts.plot_phase8 import (
    PHASE8_RUNS,
    VALIDATION_NOTE,
    build_arg_parser,
    read_best_per_class,
    support_quartiles,
    top_confusions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "artifacts" / "checkpoints"


def _run_dir(name: str) -> Path:
    return RUNS_DIR / name


def _completed(name: str) -> bool:
    return (_run_dir(name) / "summary.json").is_file()


def test_no_cli_flag_can_name_the_test_split() -> None:
    """No option exposes a split, so a figure cannot be pointed at test data."""
    parser = build_arg_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--split" not in options
    assert not any("test" in option for option in options)


def test_validation_note_names_the_split_and_disclaims_test() -> None:
    """Every figure carries provenance that survives being separated from docs."""
    assert "validation" in VALIDATION_NOTE
    assert "test split is unused" in VALIDATION_NOTE


def test_confusion_helper_refuses_the_test_split() -> None:
    """The rescoring helper rejects the test split outright."""
    from farm_pest_ai.vision.results import ResultsError, confusion_matrix_for_run

    class _Stub:
        run_dir = Path()
        run_id = "stub"

    with pytest.raises(ResultsError, match="refusing to score split"):
        confusion_matrix_for_run(_Stub(), object(), split="test")


def test_top_confusions_excludes_the_diagonal() -> None:
    """A confusion pair is a *mis*classification; correct predictions are not one."""
    matrix = [
        [8, 2, 0],
        [1, 7, 2],
        [0, 3, 6],
    ]
    pairs = top_confusions(matrix, None, limit=10)
    assert all(pair["true"] != pair["predicted"] for pair in pairs)
    # Ordered by count, descending.
    assert [pair["count"] for pair in pairs] == sorted(
        (pair["count"] for pair in pairs), reverse=True
    )
    # The share is of the true class's row total, not of the whole matrix.
    top = next(p for p in pairs if p["true"] == 2 and p["predicted"] == 1)
    assert top["share_of_true_class"] == pytest.approx(3 / 9)


def test_top_confusions_uses_class_names_when_given() -> None:
    """Names make a 102-class chart readable; indices do not."""
    matrix = [[0, 5], [1, 0]]
    pairs = top_confusions(matrix, ["aphids", "miridae"], limit=5)
    assert pairs[0]["true_name"] == "aphids"
    assert pairs[0]["predicted_name"] == "miridae"


@pytest.mark.parametrize("name", PHASE8_RUNS)
def test_per_class_comes_from_the_checkpoint_sidecar(name: str) -> None:
    """``full102`` per-class arrays live in ``best.json``, not ``metrics.jsonl``.

    The training engine omits them from the metrics log for a 102-class run, so a
    figure that reads only the log would draw nothing. This pins where they are
    actually read from, and that all four arrays cover every class.
    """
    if not _completed(name):
        pytest.skip(f"{name} has not been run")
    from farm_pest_ai.vision.results import load_run

    run = load_run(_run_dir(name))
    per_class = read_best_per_class(run)
    assert set(per_class) == {"f1", "precision", "recall", "support"}
    assert all(len(values) == 102 for values in per_class.values())
    # The support must account for the whole validation split, exactly once.
    assert sum(per_class["support"]) == 7508

    # And the log genuinely lacks them, which is why the sidecar is needed.
    best = run.best_validation(corrected=True)
    assert best is not None
    assert not best.corrected_per_class_f1


def test_support_quartiles_partition_every_class() -> None:
    """Quartiles cover all 102 classes with no overlap or omission."""
    if not all(_completed(name) for name in PHASE8_RUNS):
        pytest.skip("Phase 8 runs have not completed")
    from farm_pest_ai.vision.results import load_run

    runs = [load_run(_run_dir(name)) for name in PHASE8_RUNS]
    rows = support_quartiles(runs)
    assert len(rows) == 4
    assert sum(int(row["classes"]) for row in rows) == 102
    # Quartiles are ordered rarest-first and do not overlap.
    for earlier, later in itertools.pairwise(rows):
        assert earlier["support_max"] <= later["support_min"]


def test_read_best_per_class_raises_without_a_sidecar(tmp_path: Path) -> None:
    """A missing sidecar fails loudly rather than producing an empty figure."""

    class _Stub:
        run_dir = tmp_path
        run_id = "absent"

    with pytest.raises(PlotError, match=r"no best\.json sidecar"):
        read_best_per_class(_Stub())


def test_report_records_that_no_test_data_was_used() -> None:
    """The written report asserts its own provenance."""
    path = REPO_ROOT / "data" / "reports" / "phase8_validation_figures.json"
    if not path.is_file():
        pytest.skip("Phase 8 figures have not been rendered")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert payload["validation_only"] is True
    assert payload["test_split_used"] is False
