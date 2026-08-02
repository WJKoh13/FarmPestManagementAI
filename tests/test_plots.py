"""Tests for the Matplotlib plotting entry point.

Figures are checked for the properties that make them readable and honest —
percentage formatting, a zero-based share axis, a single y-axis, raw values
rather than a smoothed curve, and the marked best epoch — rather than by
comparing rendered pixels, which would pin the theme instead of the content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from farm_pest_ai.vision.plots import (  # noqa: E402
    FIGURE_FORMATS,
    SERIES_COLOURS,
    PlotError,
    close_figure,
    plot_accuracy,
    plot_confusion_matrix,
    plot_experiment_comparison,
    plot_learning_rate,
    plot_loss,
    plot_macro_f1,
    plot_per_class_f1,
    render_run_plots,
    save_figure,
)
from farm_pest_ai.vision.results import RunResults, load_run  # noqa: E402


def _write_run(directory: Path, *, epochs: int = 6, run_id: str = "run") -> RunResults:
    """Write a small but complete run directory and load it."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for epoch in range(1, epochs + 1):
        # A rising curve with one class permanently under the clamp, so the
        # corrected and reported series genuinely differ.
        recall = min(0.95, 0.1 * epoch)
        split = {
            "accuracy": 0.1 * epoch,
            "balanced_accuracy": 0.09 * epoch,
            "loss": 2.0 - 0.1 * epoch,
            "top5_accuracy": 0.5 + 0.05 * epoch,
            "macro_f1": 0.05 * epoch,
            "weighted_f1": 0.05 * epoch,
            "samples": 20,
            "classes_never_predicted": [],
            "per_class": {
                "precision": [0.10, recall],
                "recall": [0.20, recall],
                "f1": [0.04, recall],
                "support": [10, 10],
            },
        }
        lines.append(
            json.dumps(
                {
                    "epoch": epoch,
                    "learning_rate": 0.001 * (epochs - epoch + 1),
                    "smoke": False,
                    "train": split,
                    "validation": split,
                }
            )
        )
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "scope": "rice10",
                "model": {"name": "custom_cnn"},
                "parameters": {"total": 1435242},
                "training": {"warmup_epochs": 2},
            }
        ),
        encoding="utf-8",
    )
    return load_run(directory)


@pytest.fixture
def run(tmp_path: Path) -> RunResults:
    return _write_run(tmp_path / "run")


# -- figure content -----------------------------------------------------


def test_accuracy_axis_is_percentage_formatted_and_zero_based(
    run: RunResults,
) -> None:
    """Accuracy is a share: it reads as a percentage and starts at zero."""
    figure = plot_accuracy(run)
    axes = figure.axes[0]
    assert axes.get_ylim()[0] == pytest.approx(0.0)
    assert axes.yaxis.get_major_formatter()(0.42) == "42%"
    close_figure(figure)


def test_figures_use_a_single_y_axis(run: RunResults) -> None:
    """A second scale would make unrelated curves look correlated."""
    for builder in (plot_accuracy, plot_macro_f1, plot_loss, plot_learning_rate):
        figure = builder(run)
        assert len(figure.axes) == 1, f"{builder.__name__} produced a twin axis"
        close_figure(figure)


def test_macro_f1_plots_raw_corrected_values(run: RunResults) -> None:
    """The plotted series must be the corrected metric, unsmoothed."""
    figure = plot_macro_f1(run)
    axes = figure.axes[0]
    expected = run.curve("validation", "corrected_macro_f1")

    solid = [
        line
        for line in axes.get_lines()
        if line.get_label() == "validation" and line.get_linestyle() == "-"
    ]
    assert len(solid) == 1
    assert list(solid[0].get_ydata()) == pytest.approx(expected)
    close_figure(figure)


def test_macro_f1_ghosts_the_reported_curve(run: RunResults) -> None:
    """Both figures are shown: corrected solid, reported dashed behind it."""
    figure = plot_macro_f1(run)
    axes = figure.axes[0]
    labels = {line.get_label() for line in axes.get_lines()}
    assert "validation" in labels
    assert "validation (as reported)" in labels

    ghost = next(
        line
        for line in axes.get_lines()
        if line.get_label() == "validation (as reported)"
    )
    assert list(ghost.get_ydata()) == pytest.approx(
        run.curve("validation", "reported_macro_f1")
    )
    # The ghost sits behind the corrected curve.
    assert ghost.get_zorder() < 3
    close_figure(figure)


def test_no_ghost_when_nothing_was_under_reported(tmp_path: Path) -> None:
    """A run the bug never touched gets a clean single curve."""
    directory = tmp_path / "clean"
    directory.mkdir()
    split = {
        "accuracy": 0.9,
        "balanced_accuracy": 0.9,
        "loss": 0.2,
        "top5_accuracy": 1.0,
        "macro_f1": 0.9,
        "weighted_f1": 0.9,
        "samples": 20,
        "classes_never_predicted": [],
        # precision + recall = 1.8, comfortably above 1, so no correction.
        "per_class": {
            "precision": [0.9],
            "recall": [0.9],
            "f1": [0.9],
            "support": [20],
        },
    }
    (directory / "metrics.jsonl").write_text(
        json.dumps(
            {"epoch": 1, "learning_rate": 0.001, "smoke": False, "validation": split}
        )
        + "\n",
        encoding="utf-8",
    )
    figure = plot_macro_f1(load_run(directory))
    labels = {line.get_label() for line in figure.axes[0].get_lines()}
    assert "validation (as reported)" not in labels
    close_figure(figure)


def test_best_epoch_and_warmup_are_marked(run: RunResults) -> None:
    """Both boundaries are drawn as vertical rules."""
    figure = plot_macro_f1(run)
    axes = figure.axes[0]
    verticals = [
        line.get_xdata()[0]
        for line in axes.get_lines()
        if len(set(line.get_xdata())) == 1 and len(line.get_xdata()) == 2
    ]
    best = run.best_epoch(corrected=True)
    assert best in verticals
    # Warm-up ends after epoch 2, drawn at the boundary.
    assert pytest.approx(2.5) in verticals
    close_figure(figure)


def test_series_colours_are_assigned_by_identity_not_rank(run: RunResults) -> None:
    """Train is always slot 1 and validation slot 2, in every figure."""
    for builder in (plot_accuracy, plot_loss):
        figure = builder(run)
        by_label = {
            line.get_label(): line.get_color() for line in figure.axes[0].get_lines()
        }
        assert by_label["train"] == SERIES_COLOURS[0]
        assert by_label["validation"] == SERIES_COLOURS[1]
        close_figure(figure)


def test_learning_rate_axis_is_not_forced_to_percentage(run: RunResults) -> None:
    """The learning rate is not a share and must not be shown as one."""
    figure = plot_learning_rate(run)
    axes = figure.axes[0]
    line = axes.get_lines()[0]
    assert list(line.get_ydata()) == pytest.approx(run.learning_rates)
    close_figure(figure)


# -- per-class and comparison -------------------------------------------


def test_per_class_uses_supplied_class_names(run: RunResults) -> None:
    figure = plot_per_class_f1([run], class_names=["alpha", "beta"])
    labels = [text.get_text() for text in figure.axes[0].get_xticklabels()]
    assert labels == ["alpha", "beta"]
    close_figure(figure)


def test_per_class_rejects_a_wrong_length_name_list(run: RunResults) -> None:
    with pytest.raises(PlotError, match="class names"):
        plot_per_class_f1([run], class_names=["only-one"])


def test_per_class_bars_are_the_corrected_values(run: RunResults) -> None:
    figure = plot_per_class_f1([run])
    best = run.best_validation(corrected=True)
    assert best is not None
    heights = [patch.get_height() for patch in figure.axes[0].patches]
    assert heights == pytest.approx(list(best.corrected_per_class_f1))
    close_figure(figure)


def test_comparison_draws_one_series_per_run(tmp_path: Path) -> None:
    first = _write_run(tmp_path / "a", run_id="a")
    second = _write_run(tmp_path / "b", run_id="b")
    figure = plot_experiment_comparison([first, second])
    labels = {line.get_label() for line in figure.axes[0].get_lines()}
    assert {"a", "b"} <= labels
    close_figure(figure)


def test_comparison_leaves_room_for_the_peak_label(tmp_path: Path) -> None:
    """Peaks sit near the last epoch, so the label must not be clipped."""
    run = _write_run(tmp_path / "a", run_id="a", epochs=60)
    figure = plot_experiment_comparison([run])
    assert figure.axes[0].get_xlim()[1] > 60
    close_figure(figure)


def test_comparison_requires_validation_metrics(tmp_path: Path) -> None:
    directory = tmp_path / "train_only"
    directory.mkdir()
    (directory / "metrics.jsonl").write_text(
        json.dumps({"epoch": 1, "learning_rate": 0.001, "smoke": False}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PlotError, match="no run carries validation metrics"):
        plot_experiment_comparison([load_run(directory)])


# -- confusion matrix ---------------------------------------------------


def test_confusion_matrix_normalises_by_row() -> None:
    """Row normalisation stops a large class dominating the colour scale."""
    # Class 0 has 100 images, class 1 has 4. Both are 50% correct.
    matrix = [[50, 50], [2, 2]]
    figure = plot_confusion_matrix(matrix, normalise=True)
    image = figure.axes[0].images[0]
    flat = [value for row in image.get_array().tolist() for value in row]
    assert flat == pytest.approx([0.5, 0.5, 0.5, 0.5])
    close_figure(figure)


def test_confusion_matrix_can_show_raw_counts() -> None:
    figure = plot_confusion_matrix([[3, 1], [0, 6]], normalise=False)
    assert figure.axes[0].images[0].get_array().tolist() == [[3, 1], [0, 6]]
    close_figure(figure)


def test_confusion_matrix_uses_a_single_hue_ramp() -> None:
    """Magnitude needs one hue light-to-dark, never a rainbow."""
    figure = plot_confusion_matrix([[1, 0], [0, 1]])
    assert figure.axes[0].images[0].get_cmap().name == "Blues"
    close_figure(figure)


def test_confusion_matrix_labels_axes_by_class_name() -> None:
    figure = plot_confusion_matrix([[1, 0], [0, 1]], class_names=["alpha", "beta"])
    axes = figure.axes[0]
    assert [t.get_text() for t in axes.get_xticklabels()] == ["alpha", "beta"]
    assert [t.get_text() for t in axes.get_yticklabels()] == ["alpha", "beta"]
    # Indexed [true, predicted]: the row axis is the ground truth.
    assert axes.get_ylabel() == "true label"
    assert axes.get_xlabel() == "predicted"
    close_figure(figure)


@pytest.mark.parametrize(
    "matrix", [[], [[1, 2, 3], [4, 5, 6]]], ids=["empty", "not-square"]
)
def test_confusion_matrix_rejects_a_malformed_matrix(
    matrix: list[list[int]],
) -> None:
    with pytest.raises(PlotError):
        plot_confusion_matrix(matrix)


def test_confusion_matrix_rejects_wrong_length_names() -> None:
    with pytest.raises(PlotError, match="class names"):
        plot_confusion_matrix([[1, 0], [0, 1]], class_names=["only-one"])


def test_confusion_matrix_for_run_refuses_the_test_split(tmp_path: Path) -> None:
    """No figure may ever be produced from the test split."""
    from farm_pest_ai.vision.results import ResultsError, confusion_matrix_for_run

    run = _write_run(tmp_path / "run")
    with pytest.raises(ResultsError, match="test split is reserved"):
        confusion_matrix_for_run(run, object(), split="test")


# -- output -------------------------------------------------------------


def test_save_figure_writes_png_and_svg(run: RunResults, tmp_path: Path) -> None:
    figure = plot_accuracy(run)
    written = save_figure(figure, tmp_path / "plots", "accuracy")
    close_figure(figure)

    assert [path.suffix.lstrip(".") for path in written] == list(FIGURE_FORMATS)
    for path in written:
        assert path.is_file()
        assert path.stat().st_size > 0
    # The SVG is really vector output, not a wrapped bitmap.
    svg = next(path for path in written if path.suffix == ".svg")
    assert "<svg" in svg.read_text(encoding="utf-8", errors="ignore")[:2000]


def test_render_run_plots_produces_every_figure(
    run: RunResults, tmp_path: Path
) -> None:
    written = render_run_plots(run, tmp_path / "plots")
    stems = {path.stem for path in written}
    assert stems == {
        "accuracy",
        "macro_f1",
        "loss",
        "learning_rate",
        "per_class_f1",
    }
    # Each figure in both formats, all under the run's own directory.
    assert len(written) == len(stems) * len(FIGURE_FORMATS)
    assert all(path.parent.name == run.run_id for path in written)
