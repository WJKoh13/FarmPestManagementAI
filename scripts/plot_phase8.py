"""Phase 8 figures: the two full102 architectures on **validation data only**.

Every figure here is produced from completed run artifacts. Nothing is retrained,
the test split is never built, and no training artifact is overwritten — figures
go to a dedicated ``phase8`` subdirectory of the configured plots directory.

Two things this script does differently from ``plot_results.py``:

**Per-class figures read the checkpoint sidecar, not ``metrics.jsonl``.** The
training engine omits per-class arrays from a ``full102`` metrics log by design
(102 classes x 4 arrays x 60 epochs would be unreadable), so
:func:`farm_pest_ai.vision.plots.plot_per_class_f1` finds nothing to draw. The
arrays *are* preserved in ``best.json`` beside each checkpoint, which is where
the per-class figures below get them. The consequence is that per-class data
exists for the **best epoch only**, not per epoch as on ``rice10``.

**Confusion matrices are rescored through each run's own recorded
preprocessing.** :func:`farm_pest_ai.vision.results.confusion_matrix_for_run`
rebuilds the pipeline from the run summary and passes
``strict_preprocessing=True``, so a checkpoint can never be scored through a
mismatched pipeline and quietly produce a plausible wrong matrix. It also refuses
any split other than train or validation.

The two scopes are never ranked against one another: ``rice10`` and ``full102``
are different classification tasks, and a chart placing their macro F1 side by
side would invite exactly the comparison ``docs/EVALUATION.md`` forbids.

Usage:
    python scripts/plot_phase8.py
    python scripts/plot_phase8.py --no-confusion   # skip the GPU rescoring pass
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.vision.plots import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES_COLOURS,
    SURFACE,
    PlotError,
    _pyplot,
    _style_axes,
    close_figure,
    plot_confusion_matrix,
    save_figure,
)
from farm_pest_ai.vision.results import RunResults, confusion_matrix_for_run, load_run

#: The two Phase 8 arms, in fixed series order so a model keeps its colour
#: across every figure in the set.
PHASE8_RUNS = ("full102_baseline_protocolA", "full102_custom_protocolA")

#: Appended to every figure so a plot separated from this document still says
#: which split produced it. No figure here uses test data.
VALIDATION_NOTE = "full102 validation split only — the test split is unused (Phase 9)"


def _annotate(figure: Any) -> None:
    """Stamp the validation-only provenance onto a figure."""
    figure.text(
        0.005,
        0.005,
        VALIDATION_NOTE,
        fontsize=7.5,
        color=INK_MUTED,
        ha="left",
        va="bottom",
    )


def read_best_per_class(run: RunResults) -> dict[str, list[float]]:
    """Read per-class arrays from a run's ``best.json`` checkpoint sidecar.

    ``metrics.jsonl`` omits these for ``full102`` (see the module docstring), so
    the sidecar is the only place they survive.

    Raises:
        PlotError: If the sidecar is missing or carries no per-class block.
    """
    path = run.run_dir / "best.json"
    if not path.is_file():
        raise PlotError(f"no best.json sidecar for {run.run_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_class = payload.get("metrics", {}).get("per_class")
    if not per_class:
        raise PlotError(f"{path} carries no per-class metrics")
    return {
        "f1": [float(v) for v in per_class["f1"]],
        "precision": [float(v) for v in per_class["precision"]],
        "recall": [float(v) for v in per_class["recall"]],
        "support": [float(v) for v in per_class["support"]],
    }


def plot_loss_curves(runs: list[RunResults]) -> Any:
    """Train and validation loss for both arms on one axis."""
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(figsize=(9.0, 5.0), facecolor=SURFACE)
    for index, run in enumerate(runs):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        # An epoch record can carry either split as None; plot only the epochs
        # that actually recorded the series, rather than assuming completeness.
        train = [(r.epoch, r.train.loss) for r in run.epochs if r.train is not None]
        validation = [
            (r.epoch, r.validation.loss) for r in run.epochs if r.validation is not None
        ]
        axes.plot(
            [e for e, _ in train],
            [v for _, v in train],
            color=colour,
            linewidth=1.4,
            linestyle="--",
            alpha=0.75,
            label=f"{run.run_id} — train",
        )
        axes.plot(
            [e for e, _ in validation],
            [v for _, v in validation],
            color=colour,
            linewidth=2.0,
            label=f"{run.run_id} — validation",
        )
    _style_axes(axes)
    axes.set_xlabel("epoch")
    axes.set_ylabel("cross-entropy loss")
    axes.set_title(
        "Phase 8 full102 — training and validation loss", fontsize=11, loc="left"
    )
    axes.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    figure.tight_layout()
    _annotate(figure)
    return figure


def plot_validation_metrics(runs: list[RunResults]) -> Any:
    """Validation macro F1 and accuracy, one panel each."""
    pyplot = _pyplot()
    figure, (left, right) = pyplot.subplots(
        1, 2, figsize=(12.0, 4.8), facecolor=SURFACE
    )
    for index, run in enumerate(runs):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        scored = [
            (r.epoch, r.validation) for r in run.epochs if r.validation is not None
        ]
        epochs = [epoch for epoch, _ in scored]
        left.plot(
            epochs,
            [metrics.corrected_macro_f1 for _, metrics in scored],
            color=colour,
            linewidth=2.0,
            label=run.run_id,
        )
        right.plot(
            epochs,
            [metrics.accuracy for _, metrics in scored],
            color=colour,
            linewidth=2.0,
            label=run.run_id,
        )
        best = run.best_epoch(corrected=True)
        if best:
            left.axvline(best, color=colour, linewidth=0.9, linestyle=":", alpha=0.8)
    for axes, title, ylabel in (
        (left, "Validation macro F1", "macro F1 (corrected)"),
        (right, "Validation accuracy", "accuracy"),
    ):
        _style_axes(axes, percent=True)
        axes.set_ylim(0.0, 0.75)
        axes.set_xlabel("epoch")
        axes.set_ylabel(ylabel)
        axes.set_title(title, fontsize=11, loc="left")
        axes.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    figure.suptitle(
        "Phase 8 full102 — validation metrics (dotted line marks best epoch)",
        fontsize=12,
        color=INK_PRIMARY,
        x=0.005,
        ha="left",
    )
    figure.tight_layout()
    _annotate(figure)
    return figure


def plot_metric_comparison(runs: list[RunResults]) -> Any:
    """Grouped bars over the headline validation metrics at each best epoch."""
    pyplot = _pyplot()
    metrics = [
        ("macro F1", "corrected_macro_f1"),
        ("balanced acc.", "balanced_accuracy"),
        ("accuracy", "accuracy"),
        ("weighted F1", "corrected_weighted_f1"),
        ("top-5 acc.", "top5_accuracy"),
    ]
    figure, axes = pyplot.subplots(figsize=(9.0, 4.8), facecolor=SURFACE)
    width = 0.8 / len(runs)
    for index, run in enumerate(runs):
        best = run.best_validation(corrected=True)
        values = [float(getattr(best, attr, 0.0) or 0.0) for _, attr in metrics]
        offsets = [p + index * width - 0.4 + width / 2 for p in range(len(metrics))]
        bars = axes.bar(
            offsets,
            values,
            width=width * 0.9,
            color=SERIES_COLOURS[index % len(SERIES_COLOURS)],
            label=run.run_id,
            zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                fontsize=8,
                color=INK_SECONDARY,
            )
    _style_axes(axes, percent=True)
    axes.set_ylim(0.0, 1.0)
    axes.set_xticks(list(range(len(metrics))))
    axes.set_xticklabels([label for label, _ in metrics])
    axes.set_ylabel("score at best epoch")
    axes.set_title(
        "Phase 8 full102 — baseline vs custom on validation", fontsize=11, loc="left"
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    figure.tight_layout()
    _annotate(figure)
    return figure


def plot_per_class_by_support(runs: list[RunResults]) -> Any:
    """Per-class F1 for both arms, ordered by validation support.

    Ordering by support rather than by label index is what makes the figure
    answer the question that matters for an 82x-imbalanced task: the rare classes
    all sit together on the left, so a model that abandons the tail is visible
    immediately rather than scattered across the axis.
    """
    pyplot = _pyplot()
    data = [(run, read_best_per_class(run)) for run in runs]
    support = data[0][1]["support"]
    order = sorted(range(len(support)), key=lambda i: (support[i], i))

    figure, axes = pyplot.subplots(figsize=(15.0, 5.2), facecolor=SURFACE)
    width = 0.8 / len(data)
    for index, (run, per_class) in enumerate(data):
        values = [per_class["f1"][i] for i in order]
        offsets = [p + index * width - 0.4 + width / 2 for p in range(len(order))]
        axes.bar(
            offsets,
            values,
            width=width * 0.9,
            color=SERIES_COLOURS[index % len(SERIES_COLOURS)],
            label=run.run_id,
            zorder=3,
        )
    _style_axes(axes, percent=True)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlim(-1.0, len(order))
    step = 5
    axes.set_xticks(list(range(0, len(order), step)))
    axes.set_xticklabels(
        [f"{order[i]}\nn={int(support[order[i]])}" for i in range(0, len(order), step)],
        fontsize=7.5,
    )
    axes.set_xlabel("class label, ordered by validation support (rarest first)")
    axes.set_ylabel("validation F1 at best epoch")
    axes.set_title(
        "Phase 8 full102 — per-class validation F1, ordered by support",
        fontsize=11,
        loc="left",
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    figure.tight_layout()
    _annotate(figure)
    return figure


def plot_support_vs_f1(runs: list[RunResults]) -> Any:
    """Scatter of validation support against per-class F1, with a trend line."""
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(figsize=(9.0, 5.2), facecolor=SURFACE)
    for index, run in enumerate(runs):
        per_class = read_best_per_class(run)
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        axes.scatter(
            per_class["support"],
            per_class["f1"],
            s=26,
            color=colour,
            alpha=0.7,
            edgecolors="none",
            label=run.run_id,
            zorder=3,
        )
    _style_axes(axes, percent=True)
    axes.set_xscale("log")
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("validation support (images, log scale)")
    axes.set_ylabel("validation F1 at best epoch")
    axes.set_title(
        "Phase 8 full102 — class support vs per-class F1", fontsize=11, loc="left"
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    figure.tight_layout()
    _annotate(figure)
    return figure


def support_quartiles(runs: list[RunResults]) -> list[dict[str, Any]]:
    """Mean per-class F1 per validation-support quartile, for each arm."""
    data = [(run, read_best_per_class(run)) for run in runs]
    support = data[0][1]["support"]
    order = sorted(range(len(support)), key=lambda i: (support[i], i))
    size = len(order) // 4
    rows: list[dict[str, Any]] = []
    for quartile in range(4):
        lo = quartile * size
        hi = (quartile + 1) * size if quartile < 3 else len(order)
        chunk = order[lo:hi]
        row: dict[str, Any] = {
            "quartile": f"Q{quartile + 1}",
            "support_min": int(support[chunk[0]]),
            "support_max": int(support[chunk[-1]]),
            "classes": len(chunk),
        }
        for run, per_class in data:
            row[run.run_id] = st.fmean(per_class["f1"][i] for i in chunk)
        rows.append(row)
    return rows


def plot_support_quartiles(runs: list[RunResults]) -> Any:
    """Grouped bars of mean per-class F1 by validation-support quartile."""
    pyplot = _pyplot()
    rows = support_quartiles(runs)
    figure, axes = pyplot.subplots(figsize=(9.0, 4.8), facecolor=SURFACE)
    width = 0.8 / len(runs)
    for index, run in enumerate(runs):
        values = [float(row[run.run_id]) for row in rows]
        offsets = [p + index * width - 0.4 + width / 2 for p in range(len(rows))]
        bars = axes.bar(
            offsets,
            values,
            width=width * 0.9,
            color=SERIES_COLOURS[index % len(SERIES_COLOURS)],
            label=run.run_id,
            zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                fontsize=8,
                color=INK_SECONDARY,
            )
    _style_axes(axes, percent=True)
    axes.set_ylim(0.0, 1.0)
    axes.set_xticks(list(range(len(rows))))
    axes.set_xticklabels(
        [
            f"{row['quartile']}\nn={row['support_min']}-{row['support_max']}"
            for row in rows
        ],
        fontsize=9,
    )
    axes.set_xlabel("validation-support quartile (rarest to most common)")
    axes.set_ylabel("mean per-class validation F1")
    axes.set_title(
        "Phase 8 full102 — performance by class-support quartile",
        fontsize=11,
        loc="left",
    )
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    figure.tight_layout()
    _annotate(figure)
    return figure


def top_confusions(
    matrix: list[list[int]], class_names: list[str] | None, *, limit: int = 20
) -> list[dict[str, Any]]:
    """The most frequent off-diagonal (true, predicted) pairs."""
    pairs: list[dict[str, Any]] = []
    for true_index, row in enumerate(matrix):
        total = sum(row)
        for predicted_index, count in enumerate(row):
            if true_index == predicted_index or not count:
                continue
            pairs.append(
                {
                    "true": true_index,
                    "predicted": predicted_index,
                    "count": int(count),
                    "share_of_true_class": count / total if total else 0.0,
                    "true_name": (
                        class_names[true_index] if class_names else str(true_index)
                    ),
                    "predicted_name": (
                        class_names[predicted_index]
                        if class_names
                        else str(predicted_index)
                    ),
                }
            )
    pairs.sort(key=lambda item: (-item["count"], item["true"]))
    return pairs[:limit]


def plot_top_confusions(
    pairs: list[dict[str, Any]], run_id: str, *, limit: int = 15
) -> Any:
    """Horizontal bars of the most frequent confusion pairs."""
    pyplot = _pyplot()
    shown = pairs[:limit]
    if not shown:
        raise PlotError(f"{run_id} has no off-diagonal confusions to plot")
    figure, axes = pyplot.subplots(
        figsize=(10.0, max(4.0, len(shown) * 0.42)), facecolor=SURFACE
    )
    positions = list(range(len(shown)))[::-1]
    axes.barh(
        positions,
        [pair["count"] for pair in shown],
        color=SERIES_COLOURS[1],
        height=0.72,
        zorder=3,
    )
    for position, pair in zip(positions, shown, strict=True):
        axes.text(
            pair["count"] + 0.4,
            position,
            f"{pair['count']}  ({pair['share_of_true_class'] * 100:.0f}% of class)",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    _style_axes(axes)
    axes.set_yticks(positions)
    axes.set_yticklabels(
        [
            f"{pair['true_name'][:28]} → {pair['predicted_name'][:28]}"
            for pair in shown
        ],
        fontsize=8,
    )
    axes.set_xlabel("misclassified validation images")
    axes.set_title(
        f"Phase 8 full102 — most frequent confusions, {run_id}",
        fontsize=11,
        loc="left",
    )
    axes.set_xlim(0, max(pair["count"] for pair in shown) * 1.35)
    figure.tight_layout()
    _annotate(figure)
    return figure


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Render the Phase 8 full102 validation figures.",
        default_configs=("data_full102.yaml",),
    )
    parser.add_argument(
        "--no-confusion",
        action="store_true",
        help=(
            "Skip the confusion-matrix figures, which require a forward pass "
            "over the validation split for each checkpoint."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory holding the run directories. Defaults to the checkpoints dir.",
    )
    return parser


def _class_names(config: Config) -> list[str] | None:
    """Read class names for the active scope.

    ``plot_results.read_class_names`` lives in a sibling script rather than the
    package, so it is imported by path: a plain ``from scripts.plot_results``
    only resolves when the repository root happens to be on ``sys.path``, which
    silently yielded bare label indices when it was not. Failures are raised
    rather than swallowed — a confusion chart labelled ``24 -> 70`` instead of
    ``rice leafhopper -> ...`` is far less useful, and quietly degrading to it
    hides the cause.
    """
    import sys

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.plot_results import read_class_names

    return read_class_names(config)


def _render_confusion_figures(
    runs: list[RunResults],
    config: Config,
    names: list[str] | None,
    plots_dir: Path,
    report: dict[str, Any],
) -> list[Path]:
    """Rescore each run and write its confusion and top-confusion figures.

    The rescoring goes through each run's **own** recorded preprocessing with
    ``strict_preprocessing`` enabled, and the helper refuses any split but train
    or validation, so a Phase 8 figure cannot be built from test data.
    """
    written: list[Path] = []
    for run in runs:
        matrix = confusion_matrix_for_run(run, config, split="validation")
        figure = plot_confusion_matrix(
            matrix,
            class_names=names,
            title=f"Phase 8 full102 — normalised validation confusion, {run.run_id}",
            normalise=True,
        )
        _annotate(figure)
        stem = f"phase8_confusion_{run.run_id}"
        written.extend(save_figure(figure, plots_dir, stem))
        close_figure(figure)
        print(f"  wrote {stem}")

        pairs = top_confusions(matrix, names)
        report["runs"][run.run_id]["top_confusions"] = pairs
        figure = plot_top_confusions(pairs, run.run_id)
        stem = f"phase8_top_confusions_{run.run_id}"
        written.extend(save_figure(figure, plots_dir, stem))
        close_figure(figure)
        print(f"  wrote {stem}")
    return written


def main(argv: list[str] | None = None) -> int:
    """Render every Phase 8 figure and write the supporting JSON report."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config, _ = bootstrap(args)

    if config.dataset.scope_name != "full102":
        parser.error(
            f"Phase 8 figures describe full102; got scope "
            f"{config.dataset.scope_name!r}"
        )

    runs_dir = args.runs_dir or config.paths.checkpoints_dir
    runs = [load_run(Path(runs_dir) / name) for name in PHASE8_RUNS]
    plots_dir = Path(config.paths.plots_dir) / "phase8"

    names = _class_names(config)
    written: list[Path] = []
    report: dict[str, Any] = {
        "validation_only": True,
        "test_split_used": False,
        "runs": {},
        "support_quartiles": support_quartiles(runs),
    }

    figures = [
        ("phase8_loss_curves", plot_loss_curves(runs)),
        ("phase8_validation_metrics", plot_validation_metrics(runs)),
        ("phase8_metric_comparison", plot_metric_comparison(runs)),
        ("phase8_per_class_f1_by_support", plot_per_class_by_support(runs)),
        ("phase8_support_vs_f1", plot_support_vs_f1(runs)),
        ("phase8_support_quartiles", plot_support_quartiles(runs)),
    ]
    for stem, figure in figures:
        written.extend(save_figure(figure, plots_dir, stem))
        close_figure(figure)
        print(f"  wrote {stem}")

    for run in runs:
        best = run.best_validation(corrected=True)
        report["runs"][run.run_id] = {
            "best_epoch": run.best_epoch(corrected=True),
            "corrected_macro_f1": best.corrected_macro_f1 if best else None,
            "accuracy": best.accuracy if best else None,
            "balanced_accuracy": best.balanced_accuracy if best else None,
            "per_class": read_best_per_class(run),
        }

    if not args.no_confusion:
        written.extend(
            _render_confusion_figures(runs, config, names, plots_dir, report)
        )

    report["figures"] = [str(path) for path in written]
    report_path = Path(config.paths.reports_dir) / "phase8_validation_figures.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"\n  {len(written)} files under {plots_dir}")
    print(f"  report {report_path}")
    print("  validation split only; no test loader was built.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
