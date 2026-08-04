#!/usr/bin/env python3
"""Render figures from completed run artifacts.

Reads whatever a run recorded and produces, under the configured plots
directory:

* train and validation accuracy against epoch,
* train and validation **corrected** macro F1 against epoch,
* train and validation loss against epoch,
* the learning-rate schedule against epoch,
* a per-class F1 comparison at each run's corrected best epoch,
* an experiment-comparison chart across every run.

Nothing is retrained and no checkpoint is loaded: every value comes from
``metrics.jsonl``, and every F1 is corrected from the per-class precision and
recall recorded there. The original artifacts are never modified.

Each figure is written as both PNG and SVG. Class names come from the scope's
derived ``class_mapping.json`` when it is available, so the per-class chart is
labelled with pest names rather than label indices.

Examples:
    python scripts/plot_results.py
    python scripts/plot_results.py --run artifacts/checkpoints/rice10_custom_protocolA
    python scripts/plot_results.py --no-comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.vision.plots import (
    PlotError,
    close_figure,
    plot_confusion_matrix,
    plot_experiment_comparison,
    plot_per_class_f1,
    render_run_plots,
    save_figure,
)
from farm_pest_ai.vision.results import (
    ResultsError,
    RunResults,
    confusion_matrix_for_run,
    discover_runs,
    load_run,
)

logger = get_logger("plot_results")


def read_class_names(config: Config) -> list[str] | None:
    """Read canonical class names for the active scope, if available.

    Returns:
        Names indexed by project label, or ``None`` when the derived mapping
        has not been built. A missing mapping is not an error: the per-class
        chart falls back to label indices.
    """
    path = config.paths.processed_dir / config.scope.name / "class_mapping.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    entries = payload.get("classes")
    if not isinstance(entries, list):
        return None
    names: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = entry.get("project_label")
        name = entry.get("canonical_name") or entry.get("raw_name")
        if isinstance(label, int) and isinstance(name, str):
            names[label] = name
    if not names:
        return None
    return [names.get(index, str(index)) for index in range(max(names) + 1)]


def _render_confusion(
    run: RunResults,
    config: Config,
    plots_dir: Path,
    class_names: list[str] | None,
) -> list[Path]:
    """Recompute and plot one run's validation confusion matrix.

    A missing checkpoint or an unavailable GPU is reported and skipped rather
    than aborting a whole plotting sweep.
    """
    try:
        matrix = confusion_matrix_for_run(run, config)
    except (ResultsError, OSError, RuntimeError, ValueError) as error:
        print(f"  {run.run_id}: confusion matrix skipped ({error})", file=sys.stderr)
        return []

    figure = plot_confusion_matrix(
        matrix,
        class_names=class_names,
        title=f"{run.run_id} — validation confusion matrix (best.pt)",
    )
    written = save_figure(figure, plots_dir / run.run_id, "confusion_matrix")
    close_figure(figure)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Render figures from completed run artifacts.",
        default_configs=("base.yaml",),
    )
    parser.add_argument(
        "--run",
        action="append",
        metavar="DIR",
        help=(
            "A run directory to plot. Repeatable. Defaults to every run under "
            "the configured checkpoints directory."
        ),
    )
    parser.add_argument(
        "--plots-dir",
        metavar="DIR",
        help="Where to write figures. Defaults to the configured plots directory.",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip the cross-run comparison figures.",
    )
    parser.add_argument(
        "--confusion",
        action="store_true",
        help=(
            "Also plot each run's validation confusion matrix, recomputed by "
            "scoring its best.pt. Never touches the test split."
        ),
    )
    parser.add_argument(
        "--in-run-dir",
        action="store_true",
        help="Write each run's figures into its own run directory as well.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config: Config
    config, _ = bootstrap(args)

    try:
        if args.run:
            runs: list[RunResults] = [load_run(directory) for directory in args.run]
        else:
            runs = discover_runs(config.paths.checkpoints_dir)
    except ResultsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not runs:
        print(
            f"error: no completed runs under {config.paths.checkpoints_dir}",
            file=sys.stderr,
        )
        return 2

    plots_dir = Path(args.plots_dir) if args.plots_dir else config.paths.plots_dir
    class_names = read_class_names(config)

    written: list[Path] = []
    try:
        for run in runs:
            paths = render_run_plots(run, plots_dir, class_names=class_names)

            if args.confusion:
                paths.extend(
                    _render_confusion(run, config, plots_dir, class_names)
                )

            # Condition 7 of the Phase 7.2 approval: figures live with the run
            # artifacts, not only in the shared plots tree.
            if args.in_run_dir:
                paths.extend(
                    render_run_plots(run, run.run_dir / "plots", class_names=class_names)
                )
                if args.confusion:
                    paths.extend(
                        _render_confusion(
                            run, config, run.run_dir / "plots", class_names
                        )
                    )

            written.extend(paths)
            print(f"{run.run_id}: {len(paths)} files")

        if not args.no_comparison and len(runs) > 1:
            comparison = plot_experiment_comparison(runs)
            written.extend(save_figure(comparison, plots_dir, "comparison_macro_f1"))
            close_figure(comparison)

            per_class = plot_per_class_f1(runs, class_names=class_names)
            written.extend(save_figure(per_class, plots_dir, "comparison_per_class_f1"))
            close_figure(per_class)
            print("comparison: 4 files")
    except PlotError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print()
    print(f"{len(written)} files written under {plots_dir}")
    logger.info(
        "figures written", extra={"files": len(written), "plots_dir": str(plots_dir)}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
