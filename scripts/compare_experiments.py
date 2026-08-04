#!/usr/bin/env python3
"""Compare the Phase 7.2 screening experiments and write the comparison report.

Reads the E0-E3 run directories, ranks them on **corrected** validation macro
F1, and writes both a JSON report and the comparison figures. Nothing is
retrained; every figure and figure comes from recorded artifacts.

Each experiment changes exactly one documented variable relative to E0, with the
deliberate exception of E1: raising the epoch cap necessarily stretches the
cosine schedule, because the schedule is defined over ``training.epochs``. E1 is
therefore reported as **"longer budget with a stretched cosine"**, never as
isolating epoch count.

Examples:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --output data/reports/phase72.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.data.manifests import atomic_write_text
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.vision.plots import (
    close_figure,
    plot_experiment_comparison,
    plot_per_class_f1,
    save_figure,
)
from farm_pest_ai.vision.results import ResultsError, RunResults, load_run

logger = get_logger("compare_experiments")

#: The screening arms, in reporting order: short label, run directory name and
#: the one variable each changes relative to the E0 control.
EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("E0", "rice10_custom_e0_corrected", "control (160px, 60 epochs, crop 0.6-1.0)"),
    (
        "E1",
        "rice10_custom_e1_epochs100",
        "longer budget with a stretched cosine (100 epochs)",
    ),
    ("E2", "rice10_custom_e2_224", "input size 224x224"),
    ("E3", "rice10_custom_e3_crop08", "RandomResizedCrop scale 0.8-1.0"),
)

#: Below this, a macro F1 difference on a 721-image validation split is not
#: distinguishable from seed noise and must not be called an improvement.
MEANINGFUL_DELTA = 0.01


def load_experiments(
    checkpoints_dir: Path,
) -> list[tuple[str, str, RunResults]]:
    """Load whichever screening runs are present.

    Returns:
        ``(label, description, run)`` for each run found, in reporting order.
    """
    found: list[tuple[str, str, RunResults]] = []
    for label, directory, description in EXPERIMENTS:
        path = checkpoints_dir / directory
        if not (path / "metrics.jsonl").is_file():
            continue
        found.append((label, description, load_run(path)))
    return found


def build_report(
    experiments: list[tuple[str, str, RunResults]],
) -> dict[str, Any]:
    """Assemble the screening comparison."""
    control = next((run for label, _, run in experiments if label == "E0"), None)
    control_best = control.best_validation(corrected=True) if control else None
    baseline = control_best.corrected_macro_f1 if control_best else None

    rows: list[dict[str, Any]] = []
    for label, description, run in experiments:
        best = run.best_validation(corrected=True)
        if best is None:
            continue
        delta = (
            None if baseline is None else best.corrected_macro_f1 - baseline
        )
        rows.append(
            {
                "experiment": label,
                "variable": description,
                "run_id": run.run_id,
                "epochs_completed": len(run.epochs),
                "best_epoch": run.best_epoch(corrected=True),
                "macro_f1": best.corrected_macro_f1,
                "delta_vs_control": delta,
                "meaningful": (
                    None if delta is None else abs(delta) >= MEANINGFUL_DELTA
                ),
                "accuracy": best.accuracy,
                "balanced_accuracy": best.balanced_accuracy,
                "weighted_f1": best.corrected_weighted_f1,
                "top5_accuracy": best.top5_accuracy,
                "per_class_f1": list(best.corrected_per_class_f1),
                "per_class_support": list(best.per_class_support),
            }
        )

    ranked = sorted(rows, key=lambda row: row["macro_f1"], reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "7.2",
        "scope": "rice10",
        "metric": "corrected validation macro F1",
        "meaningful_delta_threshold": MEANINGFUL_DELTA,
        "notes": {
            "E1": (
                "E1 changes the training budget AND the cosine schedule shape, "
                "because the schedule is defined over training.epochs. It is a "
                "combined 'longer budget with a stretched cosine' experiment "
                "and does not isolate epoch count."
            ),
            "single_seed": (
                "Every arm ran one seed (1337). A difference below the "
                "meaningful threshold is not distinguishable from seed noise "
                "on a 721-image validation split."
            ),
            "test_split": "No arm built, inspected or evaluated the test split.",
        },
        "experiments": rows,
        "ranking": [row["experiment"] for row in ranked],
    }


def _print_summary(report: dict[str, Any]) -> None:
    """Print the screening comparison table."""
    print()
    print("Phase 7.2 screening — corrected validation macro F1")
    print("=" * 78)
    print(f"{'':<4}{'variable':<42}{'macro F1':>10}  {'Δ vs E0':>10}{'best ep':>9}")
    print("-" * 78)
    for row in report["experiments"]:
        delta = row["delta_vs_control"]
        if delta is None or row["experiment"] == "E0":
            delta_text = "—"
        else:
            marker = "" if row["meaningful"] else " ns"
            delta_text = f"{delta:+.4f}{marker}"
        print(
            f"{row['experiment']:<4}{row['variable'][:41]:<42}"
            f"{row['macro_f1']:>10.4f}  {delta_text:>10}{row['best_epoch']:>9}"
        )
    print()
    print(f"  ranking: {' > '.join(report['ranking'])}")
    print(
        f"  'ns' marks a difference below {report['meaningful_delta_threshold']}, "
        "which one seed cannot separate from noise."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Compare the Phase 7.2 screening experiments.",
        default_configs=("base.yaml",),
    )
    parser.add_argument(
        "--output", metavar="PATH", help="Where to write the JSON report."
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip the comparison figures."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config: Config
    config, _ = bootstrap(args)

    try:
        experiments = load_experiments(config.paths.checkpoints_dir)
    except ResultsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not experiments:
        print(
            f"error: no screening runs under {config.paths.checkpoints_dir}",
            file=sys.stderr,
        )
        return 2

    report = build_report(experiments)
    destination = (
        Path(args.output)
        if args.output
        else config.paths.reports_dir / "phase72_experiment_comparison.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, json.dumps(report, indent=2, sort_keys=True) + "\n")

    _print_summary(report)

    if not args.no_plots:
        runs = [run for _, _, run in experiments]
        short = [label for label, _, _ in experiments]
        names = _class_names(config)
        comparison = plot_experiment_comparison(runs, labels=short)
        written = save_figure(
            comparison, config.paths.plots_dir, "phase72_comparison_macro_f1"
        )
        close_figure(comparison)

        per_class = plot_per_class_f1(runs, class_names=names)
        written += save_figure(
            per_class, config.paths.plots_dir, "phase72_comparison_per_class_f1"
        )
        close_figure(per_class)
        print()
        print(f"  {len(written)} comparison figures written")
        logger.info("comparison figures written", extra={"files": len(written)})

    print()
    print(f"Report: {destination}")
    return 0


def _class_names(config: Config) -> list[str] | None:
    """Read canonical class names for the active scope, when available."""
    path = config.paths.processed_dir / config.scope.name / "class_mapping.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("classes")
    if not isinstance(entries, list):
        return None
    names = {
        entry["project_label"]: entry.get("canonical_name") or entry.get("raw_name")
        for entry in entries
        if isinstance(entry, dict) and "project_label" in entry
    }
    return [names.get(i, str(i)) for i in range(max(names) + 1)] if names else None


if __name__ == "__main__":
    raise SystemExit(main())
