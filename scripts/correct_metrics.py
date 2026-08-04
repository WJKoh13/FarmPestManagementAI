#!/usr/bin/env python3
"""Recompute corrected macro F1 for completed runs and write a correction report.

The Phase 7.1 gate. Phase 7's shared safe-division helper clamped its
denominator to ``min=1``. For precision and recall that is invisible — their
denominators are integer counts — but F1's denominator is ``precision + recall``,
a fraction, so every class whose precision and recall summed to less than 1 had
its F1 divided by 1 instead of by the true, smaller denominator. The effect is
always to **under-report**, and it falls hardest on the weakest classes, which
is exactly where macro F1 is supposed to be sensitive.

Every run recorded per-class precision, recall and support alongside the F1 it
derived, and those three were never wrong. The correction is therefore an exact
arithmetic recomputation from files already on disk — no retraining, no forward
pass, no GPU.

**The original artifacts are never modified.** The report is written to the
reports directory and holds reported and corrected values side by side, so the
Phase 7 record stays auditable.

Checkpoint verification
    ``--verify-checkpoints`` loads each run's ``best.pt`` and ``last.pt`` and
    checks their embedded metadata against the run summary: scope, class count
    and recorded epoch. It reports whether ``best.pt`` still corresponds to the
    best epoch under the corrected metric. When the correction moves the best
    epoch, the checkpoint is **not** rewritten: it holds the weights the run
    selected, which is a fact about the run, and re-pointing it would fabricate
    a checkpoint that never existed. Loading requires torch and is skipped when
    torch is unavailable.

Examples:
    python scripts/correct_metrics.py
    python scripts/correct_metrics.py --verify-checkpoints
    python scripts/correct_metrics.py --run artifacts/checkpoints/rice10_custom_protocolA
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
from farm_pest_ai.vision.results import (
    ResultsError,
    RunResults,
    compare_runs,
    discover_runs,
    load_run,
)

logger = get_logger("correct_metrics")

#: Filename of the correction report, written under the reports directory.
REPORT_NAME = "phase7_metric_correction.json"

#: Checkpoint filenames a completed run writes.
BEST_CHECKPOINT = "best.pt"
LAST_CHECKPOINT = "last.pt"


def _epoch_rows(run: RunResults) -> list[dict[str, Any]]:
    """Recalculated epoch curves, reported against corrected, for one run."""
    rows: list[dict[str, Any]] = []
    for record in run.epochs:
        row: dict[str, Any] = {
            "epoch": record.epoch,
            "learning_rate": record.learning_rate,
        }
        for split in ("train", "validation"):
            metrics = record.split(split)
            if metrics is None:
                continue
            row[split] = {
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "loss": metrics.loss,
                "reported_macro_f1": metrics.reported_macro_f1,
                "corrected_macro_f1": metrics.corrected_macro_f1,
                "macro_f1_delta": metrics.macro_f1_delta,
                "reported_weighted_f1": metrics.reported_weighted_f1,
                "corrected_weighted_f1": metrics.corrected_weighted_f1,
                "classes_affected": list(metrics.affected_classes),
            }
        rows.append(row)
    return rows


def _per_class_table(run: RunResults) -> list[dict[str, Any]]:
    """Per-class validation figures at the corrected best epoch."""
    best = run.best_validation(corrected=True)
    if best is None:
        return []
    return [
        {
            "class": index,
            "support": best.per_class_support[index],
            "precision": best.per_class_precision[index],
            "recall": best.per_class_recall[index],
            "reported_f1": best.reported_per_class_f1[index],
            "corrected_f1": best.corrected_per_class_f1[index],
            "delta": (
                best.corrected_per_class_f1[index] - best.reported_per_class_f1[index]
            ),
        }
        for index in range(len(best.per_class_support))
    ]


def verify_checkpoints(run: RunResults) -> dict[str, Any]:
    """Check a run's saved checkpoints against its summary and the correction.

    Loads metadata only — weights are read but not instantiated into a model.

    Returns:
        A report naming each checkpoint, its embedded scope, class count and
        epoch, and whether that epoch is still the best one after correction.
    """
    report: dict[str, Any] = {"available": True, "checkpoints": {}}
    try:
        from farm_pest_ai.vision.checkpoints import CheckpointError, read_metadata
    except ImportError as error:  # pragma: no cover - torch is a dependency
        return {"available": False, "reason": f"torch is not importable: {error}"}

    corrected_best = run.best_epoch(corrected=True)
    reported_best = run.best_epoch(corrected=False)

    for name in (BEST_CHECKPOINT, LAST_CHECKPOINT):
        path = run.run_dir / name
        if not path.is_file():
            report["checkpoints"][name] = {"present": False}
            continue

        try:
            metadata = read_metadata(path)
        except CheckpointError as error:
            report["checkpoints"][name] = {"present": True, "error": str(error)}
            continue

        entry: dict[str, Any] = {
            "present": True,
            "epoch": metadata.epoch,
            "scope": metadata.scope,
            "num_classes": metadata.num_classes,
            "class_mapping_version": metadata.class_mapping_version,
            "preprocessing_fingerprint": metadata.preprocessing_fingerprint,
            "scope_matches_run": metadata.scope == run.scope,
        }
        if name == BEST_CHECKPOINT:
            # The checkpoint holds whatever epoch the *buggy* metric selected.
            # It is only stale if the correction moves the best epoch.
            entry["selected_by_reported_metric"] = reported_best
            entry["best_epoch_after_correction"] = corrected_best
            entry["still_the_best_epoch"] = metadata.epoch == corrected_best
        report["checkpoints"][name] = entry

    return report


def build_report(runs: list[RunResults], *, verify: bool) -> dict[str, Any]:
    """Assemble the full correction report."""
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "7.1",
        "correction": {
            "defect": (
                "the shared safe-division helper clamped its denominator to "
                "min=1, which is invalid for the F1 denominator precision + "
                "recall whenever that sum lies strictly between 0 and 1"
            ),
            "effect": (
                "per-class F1 was under-reported for every class whose "
                "precision + recall was below 1, dragging macro F1 down; "
                "precision, recall, accuracy, balanced accuracy and top-5 were "
                "never affected"
            ),
            "method": (
                "recomputed exactly from the per-class precision and recall "
                "already recorded in each run's metrics.jsonl; no retraining"
            ),
            "artifacts_modified": False,
        },
        "runs": [],
    }

    for run in runs:
        best_corrected = run.best_validation(corrected=True)
        best_reported = run.best_validation(corrected=False)
        entry: dict[str, Any] = {
            "run_id": run.run_id,
            "run_dir": str(run.run_dir),
            "scope": run.scope,
            "model": run.model_name,
            "parameters": run.parameters,
            "epochs_completed": len(run.epochs),
            "config_sources": list(run.config_sources),
            "best_epoch_reported_metric": run.best_epoch(corrected=False),
            "best_epoch_corrected_metric": run.best_epoch(corrected=True),
            "best_epoch_moved": run.best_epoch_moved,
            "headline": {
                "reported_best_macro_f1": (
                    None if best_reported is None else best_reported.reported_macro_f1
                ),
                "corrected_best_macro_f1": (
                    None if best_corrected is None else best_corrected.corrected_macro_f1
                ),
                "accuracy_at_corrected_best": (
                    None if best_corrected is None else best_corrected.accuracy
                ),
                "balanced_accuracy_at_corrected_best": (
                    None if best_corrected is None else best_corrected.balanced_accuracy
                ),
            },
            "per_class_at_corrected_best": _per_class_table(run),
            "epochs": _epoch_rows(run),
        }
        if verify:
            entry["checkpoint_verification"] = verify_checkpoints(run)
        payload["runs"].append(entry)

    payload["comparison"] = compare_runs(runs, corrected=True)
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    """Print the human-readable correction summary."""
    print()
    print("Phase 7.1 metric correction")
    print("=" * 78)
    print("Defect:", payload["correction"]["defect"])
    print()
    header = (
        f"{'run':<28}{'reported':>10}{'corrected':>11}{'delta':>9}  {'best epoch'}"
    )
    print(header)
    print("-" * 78)

    for run in payload["runs"]:
        reported = run["headline"]["reported_best_macro_f1"]
        corrected = run["headline"]["corrected_best_macro_f1"]
        delta = (corrected or 0.0) - (reported or 0.0)
        moved = " (moved)" if run["best_epoch_moved"] else ""
        epoch = f"{run['best_epoch_corrected_metric']}{moved}"
        print(
            f"{run['run_id']:<28}{reported:>10.4f}{corrected:>11.4f}"
            f"{delta:>+9.4f}  {epoch}"
        )

    print()
    for run in payload["runs"]:
        if not run["best_epoch_moved"]:
            continue
        print(
            f"NOTE: {run['run_id']} best epoch moves "
            f"{run['best_epoch_reported_metric']} -> "
            f"{run['best_epoch_corrected_metric']}. Its best.pt holds the "
            f"epoch the original metric selected and was not rewritten."
        )

    verified = [
        (run["run_id"], run["checkpoint_verification"])
        for run in payload["runs"]
        if "checkpoint_verification" in run
    ]
    if verified:
        print()
        print("Checkpoint verification")
        print("-" * 78)
        for run_id, report in verified:
            _print_checkpoint_report(run_id, report)
    print()


def _print_checkpoint_report(run_id: str, report: dict[str, Any]) -> None:
    """Print one run's checkpoint verification lines."""
    if not report.get("available", False):
        print(f"{run_id}: skipped ({report.get('reason')})")
        return

    for name, entry in report["checkpoints"].items():
        if not entry.get("present"):
            print(f"{run_id}/{name}: missing")
            continue
        if "error" in entry:
            print(f"{run_id}/{name}: FAILED to load - {entry['error']}")
            continue

        status = "scope OK" if entry["scope_matches_run"] else "SCOPE MISMATCH"
        extra = ""
        if name == BEST_CHECKPOINT and not entry.get("still_the_best_epoch", True):
            extra = " (no longer the best epoch after correction)"
        print(
            f"{run_id}/{name}: epoch {entry['epoch']}, "
            f"{entry['scope']}, {status}{extra}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Recompute corrected macro F1 from completed run artifacts.",
        default_configs=("base.yaml",),
    )
    parser.add_argument(
        "--run",
        action="append",
        metavar="DIR",
        help=(
            "A run directory to correct. Repeatable. Defaults to every run "
            "under the configured checkpoints directory."
        ),
    )
    parser.add_argument(
        "--verify-checkpoints",
        action="store_true",
        help="Load best.pt and last.pt and check their embedded metadata.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help=f"Where to write the report. Defaults to <reports_dir>/{REPORT_NAME}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config: Config
    config, _ = bootstrap(args)

    if args.run:
        runs: list[RunResults] = []
        for directory in args.run:
            try:
                runs.append(load_run(directory))
            except ResultsError as error:
                logger.error("could not load run", extra={"run": directory})
                print(f"error: {error}", file=sys.stderr)
                return 2
    else:
        checkpoints_dir = config.paths.checkpoints_dir
        try:
            runs = discover_runs(checkpoints_dir)
        except ResultsError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        if not runs:
            print(f"error: no completed runs under {checkpoints_dir}", file=sys.stderr)
            return 2

    payload = build_report(runs, verify=args.verify_checkpoints)

    destination = (
        Path(args.output) if args.output else config.paths.reports_dir / REPORT_NAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    _print_summary(payload)
    print(f"Report written to {destination}")
    logger.info(
        "correction report written",
        extra={"runs": len(runs), "output": str(destination)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
