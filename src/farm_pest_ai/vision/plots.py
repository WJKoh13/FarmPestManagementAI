"""Matplotlib figures built from completed run artifacts.

Every figure reads what a run actually recorded — no retraining and no forward
pass — and every F1 shown is the **corrected** one from
:mod:`farm_pest_ai.vision.results`. Where the correction changes a curve, the
reported values are drawn behind it as a dashed ghost, so a plot shows what was
claimed alongside what is true rather than quietly replacing one with the other.

Plotting decisions
    Raw per-epoch values, never smoothed. A smoothed validation curve hides the
    epoch-to-epoch variance that decides whether a gap between two runs is real,
    which is the main thing these plots exist to show.

    The best epoch and the warm-up boundary are marked on every epoch axis. The
    Phase 7 runs both peaked at epoch 58 of 60, and that fact is far easier to
    see as a marked line than to read off a table.

    Accuracy and F1 axes are percentage-formatted and, being shares, start at
    zero. Loss and learning rate are not shares and are not forced to zero.

    A single y-axis per figure, always. Two scales on one plot make any pair of
    curves look correlated, which is exactly the mistake these plots must not
    invite.

Colours are a fixed, colour-vision-validated order — slot 1 blue, slot 2 orange,
slot 3 aqua — assigned by series identity, never by rank, so a series keeps its
colour across every figure in the set.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from farm_pest_ai.vision.results import RunResults

__all__ = [
    "PlotError",
    "close_figure",
    "plot_accuracy",
    "plot_confusion_matrix",
    "plot_experiment_comparison",
    "plot_learning_rate",
    "plot_loss",
    "plot_macro_f1",
    "plot_per_class_f1",
    "render_run_plots",
    "save_figure",
]

#: Categorical slots, in fixed assignment order. Validated for colour-vision
#: separation against the light chart surface; see the dataviz palette.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

#: Ink and chrome, kept recessive so the data carries the figure.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOUR = "#e1e0d9"
SURFACE = "#fcfcfb"

#: Formats every figure is written in. SVG for documents that must stay
#: scalable, PNG for quick viewing.
FIGURE_FORMATS = ("png", "svg")


class PlotError(RuntimeError):
    """Raised when a figure cannot be produced."""


def _pyplot() -> Any:
    """Import pyplot with a non-interactive backend.

    Selecting ``Agg`` explicitly keeps the module importable on a headless
    container and stops a stray window from opening on Windows.
    """
    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - matplotlib is optional
        raise PlotError(
            "matplotlib is required for plotting; install the 'train' extra"
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _style_axes(axes: Any, *, percent: bool = False) -> None:
    """Apply the shared chart chrome to one axes."""
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID_COLOUR, linewidth=0.8, alpha=0.9)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID_COLOUR)
    axes.tick_params(colors=INK_MUTED, labelsize=9)
    axes.xaxis.label.set_color(INK_SECONDARY)
    axes.yaxis.label.set_color(INK_SECONDARY)
    axes.title.set_color(INK_PRIMARY)

    if percent:
        from matplotlib.ticker import FuncFormatter

        axes.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value * 100:.0f}%")
        )


def _mark_epochs(axes: Any, run: RunResults, *, label: bool = True) -> None:
    """Draw the warm-up boundary and the corrected best epoch."""
    warmup = run.warmup_epochs
    if warmup:
        axes.axvline(
            warmup + 0.5,
            color=INK_MUTED,
            linewidth=1.0,
            linestyle=":",
            zorder=1,
        )
        if label:
            axes.annotate(
                f"warm-up ends (epoch {warmup})",
                xy=(warmup + 0.5, 0.02),
                xycoords=("data", "axes fraction"),
                rotation=90,
                fontsize=7.5,
                color=INK_MUTED,
                ha="right",
                va="bottom",
            )

    best = run.best_epoch(corrected=True)
    if best is not None:
        axes.axvline(best, color=INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=1)
        if label:
            axes.annotate(
                f"best epoch {best}",
                xy=(best, 0.98),
                xycoords=("data", "axes fraction"),
                rotation=90,
                fontsize=7.5,
                color=INK_SECONDARY,
                ha="right",
                va="top",
            )


def _plot_split_curves(
    axes: Any,
    run: RunResults,
    metric: str,
    *,
    ghost_metric: str | None = None,
) -> None:
    """Draw one metric for both splits, with an optional reported ghost."""
    epochs = run.epoch_numbers
    for index, split in enumerate(("train", "validation")):
        values = run.curve(split, metric)
        if all(value is None for value in values):
            continue
        axes.plot(
            epochs,
            values,
            color=SERIES_COLOURS[index],
            linewidth=2.0,
            label=split,
            zorder=3,
        )
        if ghost_metric is not None:
            reported = run.curve(split, ghost_metric)
            if any(
                a is not None and b is not None and abs(a - b) > 1e-12
                for a, b in zip(values, reported, strict=True)
            ):
                axes.plot(
                    epochs,
                    reported,
                    color=SERIES_COLOURS[index],
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.55,
                    label=f"{split} (as reported)",
                    zorder=2,
                )


def close_figure(figure: Any) -> None:
    """Release a figure's resources.

    Callers that build many figures in one process must close each one;
    matplotlib otherwise keeps every figure alive in its global registry.
    """
    _pyplot().close(figure)


def save_figure(figure: Any, directory: Path, stem: str) -> list[Path]:
    """Write one figure as PNG and SVG.

    Args:
        figure: The matplotlib figure.
        directory: Destination directory, created if absent.
        stem: Filename without an extension.

    Returns:
        The paths written, in :data:`FIGURE_FORMATS` order.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in FIGURE_FORMATS:
        path = directory / f"{stem}.{suffix}"
        figure.savefig(path, format=suffix, dpi=150, bbox_inches="tight")
        written.append(path)
    return written


def plot_accuracy(run: RunResults) -> Any:
    """Train and validation accuracy against epoch."""
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _plot_split_curves(axes, run, "accuracy")
    _mark_epochs(axes, run)
    _style_axes(axes, percent=True)
    axes.set_ylim(bottom=0.0)
    axes.set_xlabel("epoch")
    axes.set_ylabel("accuracy")
    axes.set_title(f"{run.run_id} — accuracy", fontsize=11, loc="left")
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    figure.tight_layout()
    return figure


def plot_macro_f1(run: RunResults) -> Any:
    """Train and validation corrected macro F1 against epoch.

    The reported curve is drawn behind as a dashed ghost wherever the correction
    changed it, which is the whole point of the figure.
    """
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _plot_split_curves(
        axes, run, "corrected_macro_f1", ghost_metric="reported_macro_f1"
    )
    _mark_epochs(axes, run)
    _style_axes(axes, percent=True)
    axes.set_ylim(bottom=0.0)
    axes.set_xlabel("epoch")
    axes.set_ylabel("macro F1")
    axes.set_title(
        f"{run.run_id} — corrected macro F1", fontsize=11, loc="left"
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    figure.tight_layout()
    return figure


def plot_loss(run: RunResults) -> Any:
    """Train and validation loss against epoch."""
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _plot_split_curves(axes, run, "loss")
    _mark_epochs(axes, run)
    _style_axes(axes)
    axes.set_xlabel("epoch")
    axes.set_ylabel("loss")
    axes.set_title(f"{run.run_id} — loss", fontsize=11, loc="left")
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    figure.tight_layout()
    return figure


def plot_learning_rate(run: RunResults) -> Any:
    """Learning rate against epoch."""
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    axes.plot(
        run.epoch_numbers,
        run.learning_rates,
        color=SERIES_COLOURS[0],
        linewidth=2.0,
        label="learning rate",
        zorder=3,
    )
    _mark_epochs(axes, run)
    _style_axes(axes)
    axes.set_xlabel("epoch")
    axes.set_ylabel("learning rate")
    axes.set_title(
        f"{run.run_id} — learning rate schedule", fontsize=11, loc="left"
    )
    figure.tight_layout()
    return figure


def plot_per_class_f1(
    runs: Sequence[RunResults], *, class_names: Sequence[str] | None = None
) -> Any:
    """Per-class validation F1 at each run's corrected best epoch.

    Args:
        runs: One or more runs to compare. Bars are grouped by class.
        class_names: Optional axis labels; defaults to project label indices.

    Raises:
        PlotError: If no run carries per-class validation figures.
    """
    pyplot = _pyplot()
    series: list[tuple[str, tuple[float, ...]]] = []
    for run in runs:
        best = run.best_validation(corrected=True)
        if best is None or not best.corrected_per_class_f1:
            continue
        series.append((run.run_id, best.corrected_per_class_f1))
    if not series:
        raise PlotError("no run carries per-class validation metrics")

    num_classes = len(series[0][1])
    labels = (
        list(class_names)
        if class_names is not None
        else [str(index) for index in range(num_classes)]
    )
    if len(labels) != num_classes:
        raise PlotError(
            f"{len(labels)} class names for {num_classes} classes"
        )

    figure, axes = pyplot.subplots(
        figsize=(max(8.0, num_classes * 0.95), 4.8), facecolor=SURFACE
    )
    positions = range(num_classes)
    # A 2px surface gap between adjacent bars keeps groups legible.
    width = 0.8 / len(series)
    for index, (name, values) in enumerate(series):
        offsets = [p + index * width - 0.4 + width / 2 for p in positions]
        axes.bar(
            offsets,
            values,
            width=width * 0.92,
            color=SERIES_COLOURS[index % len(SERIES_COLOURS)],
            label=name,
            zorder=3,
        )

    _style_axes(axes, percent=True)
    axes.set_ylim(0.0, 1.0)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    axes.set_xlabel("class")
    axes.set_ylabel("validation F1 (corrected)")
    axes.set_title(
        "Per-class validation F1 at each run's best epoch", fontsize=11, loc="left"
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    figure.tight_layout()
    return figure


def plot_confusion_matrix(
    matrix: Sequence[Sequence[float]],
    *,
    class_names: Sequence[str] | None = None,
    title: str = "Confusion matrix",
    normalise: bool = True,
) -> Any:
    """Plot a confusion matrix indexed ``[true, predicted]``.

    A confusion matrix encodes **magnitude**, so it uses one hue light-to-dark
    rather than a rainbow: with a multi-hue scale the reader has to consult the
    legend to order two cells, and mid-scale hues read as categories rather than
    as quantities.

    Args:
        matrix: Counts indexed ``[true, predicted]``.
        class_names: Axis labels; defaults to label indices.
        title: Figure title.
        normalise: Divide each row by its support, so a large class cannot
            dominate the colour scale. Row-normalised diagonals are per-class
            recall.

    Raises:
        PlotError: If the matrix is empty or not square.
    """
    pyplot = _pyplot()

    rows = [list(row) for row in matrix]
    if not rows or not rows[0]:
        raise PlotError("confusion matrix is empty")
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise PlotError("confusion matrix must be square")

    counts = [row[:] for row in rows]
    if normalise:
        rows = [
            [value / total if (total := sum(row)) else 0.0 for value in row]
            for row in rows
        ]

    labels = (
        list(class_names) if class_names is not None else [str(i) for i in range(size)]
    )
    if len(labels) != size:
        raise PlotError(f"{len(labels)} class names for {size} classes")

    figure, axes = pyplot.subplots(
        figsize=(max(6.5, size * 0.78), max(5.5, size * 0.68)), facecolor=SURFACE
    )
    # One hue, light to dark: the project's sequential blue ramp.
    image = axes.imshow(
        rows, cmap="Blues", vmin=0.0, vmax=1.0 if normalise else None, aspect="auto"
    )

    axes.set_xticks(range(size))
    axes.set_yticks(range(size))
    axes.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axes.set_yticklabels(labels, fontsize=8)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true label")
    axes.set_title(title, fontsize=11, loc="left")

    # Direct-label every cell: at 10 classes the grid is small enough that the
    # numbers are more useful than a colour lookup, and they keep the figure
    # readable for anyone who cannot separate the ramp's steps.
    for y in range(size):
        for x in range(size):
            value = rows[y][x]
            if normalise:
                text = f"{value * 100:.0f}" if value >= 0.005 else ""
            else:
                text = f"{int(counts[y][x])}" if counts[y][x] else ""
            if text:
                axes.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    # Ink flips on the dark end of the ramp to hold contrast.
                    color="#ffffff" if value > 0.55 else INK_PRIMARY,
                )

    axes.grid(False)
    axes.tick_params(colors=INK_MUTED, labelsize=8)
    axes.xaxis.label.set_color(INK_SECONDARY)
    axes.yaxis.label.set_color(INK_SECONDARY)
    axes.title.set_color(INK_PRIMARY)

    bar = figure.colorbar(image, ax=axes, fraction=0.045)
    bar.set_label(
        "share of true class (%)" if normalise else "images", color=INK_SECONDARY
    )
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    if normalise:
        from matplotlib.ticker import FuncFormatter

        # A formatter rather than fixed labels: set_yticklabels on a colourbar
        # with a dynamic locator can attach labels to the wrong ticks.
        bar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value * 100:.0f}")
        )

    figure.tight_layout()
    return figure


def plot_experiment_comparison(
    runs: Sequence[RunResults], *, labels: Sequence[str] | None = None
) -> Any:
    """Validation macro F1 against epoch for several runs on one axis.

    Args:
        runs: Runs to overlay. Colour follows position, so a run keeps its
            colour across the figure set.
        labels: Optional short legend names, one per run. Defaults to
            ``run_id``, which is unwieldy once several experiments share a
            prefix.

    Raises:
        PlotError: If no run carries validation metrics, or ``labels`` does not
            match ``runs`` in length.
    """
    pyplot = _pyplot()
    if labels is not None and len(labels) != len(runs):
        raise PlotError(f"{len(labels)} labels for {len(runs)} runs")

    figure, axes = pyplot.subplots(figsize=(9.0, 5.0), facecolor=SURFACE)

    drawn = 0
    for index, run in enumerate(runs):
        values = run.curve("validation", "corrected_macro_f1")
        if all(value is None for value in values):
            continue
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        axes.plot(
            run.epoch_numbers,
            values,
            color=colour,
            linewidth=2.0,
            label=labels[index] if labels else run.run_id,
            zorder=3,
        )
        best_epoch = run.best_epoch(corrected=True)
        best = run.best_validation(corrected=True)
        if best_epoch is not None and best is not None:
            # Direct-label the peak rather than every point.
            axes.plot(
                [best_epoch],
                [best.corrected_macro_f1],
                marker="o",
                markersize=8,
                color=colour,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                zorder=4,
            )
            # Stagger the peak labels: several runs peak near the same epoch
            # and at similar scores, so a fixed offset overlaps them.
            axes.annotate(
                f"{best.corrected_macro_f1 * 100:.1f}%",
                xy=(best_epoch, best.corrected_macro_f1),
                xytext=(7, 6 + 11 * (len(runs) - 1 - index)),
                textcoords="offset points",
                fontsize=9,
                color=colour,
            )
        drawn += 1

    if not drawn:
        raise PlotError("no run carries validation metrics")

    _style_axes(axes, percent=True)
    axes.set_ylim(bottom=0.0)
    # The peak label is drawn to the right of its marker, and every run's peak
    # tends to sit near the final epoch. Widen the axis so it is not clipped.
    right = max(max(run.epoch_numbers) for run in runs if run.epochs)
    axes.set_xlim(right=right * 1.06)
    axes.set_xlabel("epoch")
    axes.set_ylabel("validation macro F1 (corrected)")
    axes.set_title("Experiment comparison", fontsize=11, loc="left")
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    figure.tight_layout()
    return figure


def render_run_plots(
    run: RunResults,
    plots_dir: Path,
    *,
    class_names: Sequence[str] | None = None,
) -> list[Path]:
    """Render every per-run figure into ``plots_dir/<run_id>/``.

    Returns:
        Every file written.
    """
    destination = Path(plots_dir) / run.run_id
    written: list[Path] = []

    figures = {
        "accuracy": plot_accuracy(run),
        "macro_f1": plot_macro_f1(run),
        "loss": plot_loss(run),
        "learning_rate": plot_learning_rate(run),
    }
    # A run without per-class figures still gets its curve plots.
    with contextlib.suppress(PlotError):
        figures["per_class_f1"] = plot_per_class_f1([run], class_names=class_names)

    for stem, figure in figures.items():
        written.extend(save_figure(figure, destination, stem))
        close_figure(figure)
    return written
