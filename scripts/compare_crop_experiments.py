#!/usr/bin/env python3
"""Compare each bounding-box crop arm against its full-frame control.

Answers one question per pair - *did cropping improve generalisation?* - and is
deliberately conservative about how it answers:

* the verdict is taken from **validation macro F1 and balanced accuracy**, never
  from training accuracy, which rises whenever a task gets easier to memorise;
* a difference below 0.01 is reported as indistinguishable from seed noise, the
  threshold this project has used since Phase 7.2;
* per-sample flips are computed by re-scoring both checkpoints through **their
  own** recorded preprocessing, so a mismatched pipeline raises instead of
  producing a plausible but wrong answer.

Inference only. Nothing is retrained, no checkpoint is rewritten, and the test
split is never opened.

Examples:
    python scripts/compare_crop_experiments.py
    python scripts/compare_crop_experiments.py --pair E4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import load_config
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.vision.results import RunResults, load_run

LOGGER = get_logger(__name__)

#: A difference below this is treated as seed noise rather than an effect.
#: Inherited from Phase 7.2 so verdicts stay comparable across experiments.
NOISE_THRESHOLD = 0.01

#: The pairs, as (label, control run, treatment run, control config,
#: treatment config, scope). Both configs are carried because each arm must be
#: re-scored through its own image pipeline.
PAIRS: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "E4",
        "det_top10_e4a_fullframe",
        "det_top10_e4b_crop15",
        "exp_det_top10_e4a_fullframe.yaml",
        "exp_det_top10_e4b_crop15.yaml",
        "det_top10",
    ),
    (
        "E5",
        "det_top15_e5a_fullframe",
        "det_top15_e5b_crop15",
        "exp_det_top15_e5a_fullframe.yaml",
        "exp_det_top15_e5b_crop15.yaml",
        "det_top15",
    ),
)


def _relative(delta: float, base: float) -> float | None:
    """Return the relative change, or None when the base is zero."""
    if base == 0:
        return None
    return round(100.0 * delta / base, 3)


def _headline(run: RunResults) -> dict[str, Any]:
    """Extract the metrics a verdict may rest on."""
    best = run.best_validation()
    if best is None:
        raise SystemExit(f"{run.run_dir}: no validation metrics recorded")
    epoch = run.best_epoch()
    # The corrected macro F1 throughout: the Phase 7.1 defect under-reported the
    # value every earlier decision was made on, and mixing the two series would
    # make this comparison incompatible with E0-E3.
    curve = [v for v in run.curve("validation", "corrected_macro_f1") if v is not None]
    train_acc = [v for v in run.curve("train", "accuracy") if v is not None]
    val_acc = [v for v in run.curve("validation", "accuracy") if v is not None]
    return {
        "run_dir": str(run.run_dir),
        "best_epoch": epoch,
        "epochs_run": len(curve),
        "macro_f1": best.corrected_macro_f1,
        "accuracy": best.accuracy,
        "balanced_accuracy": best.balanced_accuracy,
        # The mean of the last ten epochs is the more conservative reading:
        # "best epoch" is the maximum of a noisy series and flatters whichever
        # run happened to spike.
        "last10_mean_macro_f1": (
            round(sum(curve[-10:]) / len(curve[-10:]), 6) if curve else None
        ),
        "final_train_accuracy": train_acc[-1] if train_acc else None,
        "final_val_accuracy": val_acc[-1] if val_acc else None,
        # The train-validation gap at the end of training is the overfitting
        # indicator; a crop that raises train accuracy while widening the gap
        # has made memorisation easier, not generalisation better.
        "final_train_val_gap": (
            round(train_acc[-1] - val_acc[-1], 6) if train_acc and val_acc else None
        ),
        "per_class": {
            str(index): {
                "precision": (
                    best.per_class_precision[index]
                    if index < len(best.per_class_precision)
                    else None
                ),
                "recall": (
                    best.per_class_recall[index]
                    if index < len(best.per_class_recall)
                    else None
                ),
                "f1": (
                    best.corrected_per_class_f1[index]
                    if index < len(best.corrected_per_class_f1)
                    else None
                ),
                "support": (
                    best.per_class_support[index]
                    if index < len(best.per_class_support)
                    else None
                ),
            }
            for index in range(len(best.per_class_support))
        },
    }


def _predictions(
    run: RunResults, config: Any, split: str = "validation"
) -> tuple[list[int], list[int], list[str], list[float]]:
    """Score a run's best checkpoint through its own preprocessing.

    Returns:
        ``(predictions, targets, filenames, confidences)`` in manifest order.
    """
    import torch

    from farm_pest_ai.data.loaders import build_loaders
    from farm_pest_ai.vision.checkpoints import load_checkpoint

    path = run.run_dir / "best.pt"
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")

    preprocessing = run.preprocessing_config()
    bundle = build_loaders(config, (split,), preprocessing=preprocessing)
    model, _, _ = load_checkpoint(
        path,
        scope=config.scope,
        map_location="cpu",
        preprocessing_fingerprint=(
            preprocessing.fingerprint if preprocessing is not None else None
        ),
        strict_preprocessing=preprocessing is not None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    predictions: list[int] = []
    targets: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for images, labels in bundle.loaders[split]:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            best, index = probabilities.max(dim=1)
            predictions.extend(int(v) for v in index.cpu())
            confidences.extend(float(v) for v in best.cpu())
            targets.extend(int(v) for v in labels.cpu())

    filenames = [r.filename for r in bundle.datasets[split].records]
    return predictions, targets, filenames, confidences


def _per_class_from_predictions(run: RunResults, config: Any) -> dict[str, Any]:
    """Recompute per-class precision, recall, F1 and support from predictions.

    Used when a run's ``metrics.jsonl`` carries no per-class arrays. Uses the
    Phase 7.1 corrected F1 formula, so the values are directly comparable with
    runs whose arrays were logged.
    """
    predictions, targets, _, _ = _predictions(run, config)
    classes = sorted(set(targets))
    result: dict[str, Any] = {}
    for label in classes:
        true_positive = sum(
            1 for p, t in zip(predictions, targets, strict=True) if p == t == label
        )
        predicted = sum(1 for p in predictions if p == label)
        actual = sum(1 for t in targets if t == label)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        denominator = precision + recall
        result[str(label)] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(2 * precision * recall / denominator, 6) if denominator else 0.0,
            "support": actual,
        }
    return result


def _write_predictions(
    path: Path,
    filenames: list[str],
    targets: list[int],
    predictions: list[int],
    confidences: list[float],
) -> None:
    """Write per-image predictions as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["filename,true_label,predicted_label,confidence"]
    lines.extend(
        f"{name},{true},{pred},{conf:.6f}"
        for name, true, pred, conf in zip(
            filenames, targets, predictions, confidences, strict=True
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def compare_pair(
    label: str,
    control_name: str,
    treatment_name: str,
    control_config_name: str,
    treatment_config_name: str,
    scope: str,
    checkpoints_dir: Path,
    reports_dir: Path,
    *,
    with_flips: bool,
) -> dict[str, Any]:
    """Compare one crop arm against its full-frame control."""
    control = load_run(checkpoints_dir / control_name)
    treatment = load_run(checkpoints_dir / treatment_name)
    a = _headline(control)
    b = _headline(treatment)

    metrics: dict[str, Any] = {}
    for key in (
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "last10_mean_macro_f1",
        "final_train_val_gap",
    ):
        base, other = a.get(key), b.get(key)
        if base is None or other is None:
            continue
        delta = round(other - base, 6)
        metrics[key] = {
            "full_frame": base,
            "crop": other,
            "absolute_change": delta,
            "relative_change_percent": _relative(delta, base),
            "beats_noise_threshold": abs(delta) >= NOISE_THRESHOLD,
        }

    # `training.py` only logs per-class arrays when num_classes <= 10, a
    # threshold set when the only scopes were rice10 (10) and full102 (102).
    # det_top15 falls outside it, so its per-class breakdown is recomputed here
    # from predictions rather than read back. This is exact arithmetic over the
    # same checkpoints, not a re-run, and it leaves both runs' artifacts alone.
    if not a["per_class"] or not b["per_class"]:
        control_config = load_config(["model_custom.yaml", control_config_name])
        treatment_config = load_config(["model_custom.yaml", treatment_config_name])
        a["per_class"] = _per_class_from_predictions(control, control_config)
        b["per_class"] = _per_class_from_predictions(treatment, treatment_config)
        a["per_class_source"] = "recomputed from predictions"
        b["per_class_source"] = "recomputed from predictions"

    per_class: dict[str, Any] = {}
    for index, entry in a["per_class"].items():
        other = b["per_class"].get(index, {})
        for metric in ("recall", "f1"):
            base_value = entry.get(metric)
            new_value = other.get(metric)
            if base_value is None or new_value is None:
                continue
            per_class.setdefault(index, {"support": entry.get("support")})[metric] = {
                "full_frame": base_value,
                "crop": new_value,
                "absolute_change": round(new_value - base_value, 6),
            }

    report: dict[str, Any] = {
        "pair": label,
        "scope": scope,
        "control": {"name": control_name, **a},
        "treatment": {"name": treatment_name, **b},
        "metrics": metrics,
        "per_class": per_class,
        "noise_threshold": NOISE_THRESHOLD,
    }

    if with_flips:
        # Each arm must be scored through ITS OWN configuration. Scoring both
        # through the control's config would feed the crop model full frames -
        # it loads cleanly, since the two arms share a preprocessing
        # fingerprint (cropping happens before the pipeline), and silently
        # produces a wrong answer. This is the same failure Phase 7.2 hit when
        # a 224x224 run was scored through a 160x160 pipeline; here the
        # fingerprint cannot catch it, so the configs are kept separate.
        control_config = load_config(["model_custom.yaml", control_config_name])
        treatment_config = load_config(["model_custom.yaml", treatment_config_name])
        pred_a, targets_a, files_a, conf_a = _predictions(control, control_config)
        pred_b, targets_b, files_b, conf_b = _predictions(treatment, treatment_config)
        if files_a != files_b or targets_a != targets_b:
            raise SystemExit(
                f"{label}: the two arms scored different samples; the pairing "
                f"invariant is broken and no comparison is valid"
            )

        # Re-scoring must reproduce the accuracy training recorded. Without this
        # check a mis-built pipeline yields a self-consistent set of predictions
        # that silently disagrees with the run's own metrics - which is exactly
        # what happened when both arms were scored through the control's config.
        for name, predictions, targets, recorded in (
            (control_name, pred_a, targets_a, a["accuracy"]),
            (treatment_name, pred_b, targets_b, b["accuracy"]),
        ):
            rescored = sum(
                1 for p, t in zip(predictions, targets, strict=True) if p == t
            ) / len(targets)
            if abs(rescored - recorded) > 0.005:
                raise SystemExit(
                    f"{name}: re-scored accuracy {rescored:.4f} does not match the "
                    f"{recorded:.4f} recorded during training. The evaluation "
                    f"pipeline does not match the one the model was trained under, "
                    f"so the flip analysis would be wrong."
                )

        _write_predictions(
            reports_dir / "crop_experiments" / f"{control_name}_predictions.csv",
            files_a,
            targets_a,
            pred_a,
            conf_a,
        )
        _write_predictions(
            reports_dir / "crop_experiments" / f"{treatment_name}_predictions.csv",
            files_b,
            targets_b,
            pred_b,
            conf_b,
        )

        fixed = [
            {"filename": f, "true_label": t, "full_frame_predicted": x}
            for f, t, x, y in zip(files_a, targets_a, pred_a, pred_b, strict=True)
            if x != t and y == t
        ]
        broken = [
            {"filename": f, "true_label": t, "crop_predicted": y}
            for f, t, x, y in zip(files_a, targets_a, pred_a, pred_b, strict=True)
            if x == t and y != t
        ]
        report["flips"] = {
            "corrected_by_cropping": len(fixed),
            "broken_by_cropping": len(broken),
            "net": len(fixed) - len(broken),
            "samples_scored": len(files_a),
            "corrected_examples": fixed[:25],
            "broken_examples": broken[:25],
        }
        report["predictions"] = {
            "full_frame": str(
                reports_dir / "crop_experiments" / f"{control_name}_predictions.csv"
            ),
            "crop": str(
                reports_dir / "crop_experiments" / f"{treatment_name}_predictions.csv"
            ),
        }

    macro = metrics.get("macro_f1", {})
    balanced = metrics.get("balanced_accuracy", {})
    decisive = bool(macro.get("beats_noise_threshold")) and bool(
        balanced.get("beats_noise_threshold")
    )
    direction = macro.get("absolute_change", 0.0)
    report["verdict"] = {
        "improved": decisive and direction > 0,
        "regressed": decisive and direction < 0,
        "indistinguishable_from_noise": not decisive,
        "basis": "validation macro F1 and balanced accuracy, single seed",
        "caveat": (
            "Single seed. Phase 7.2/E4 showed a single-seed margin shrinking and "
            "even reversing under replication, so the direction is better "
            "supported than the magnitude."
        ),
    }
    return report


def _plot_pair(report: dict[str, Any], plots_dir: Path) -> list[str]:
    """Render accuracy and loss curves for one pair on shared axes."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        LOGGER.warning("matplotlib unavailable; skipping figures")
        return []

    control = load_run(Path(report["control"]["run_dir"]))
    treatment = load_run(Path(report["treatment"]["run_dir"]))
    written: list[str] = []
    plots_dir.mkdir(parents=True, exist_ok=True)

    panels = (
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
        ("corrected_macro_f1", "Macro F1 (corrected)"),
    )
    figure, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5))
    for axis, (metric, title) in zip(axes, panels, strict=True):
        for run, name, colour in (
            (control, "full frame", "tab:blue"),
            (treatment, "crop 15%", "tab:orange"),
        ):
            epochs = run.epoch_numbers
            for split, style in (("train", "--"), ("validation", "-")):
                series = run.curve(split, metric)
                points = [
                    (e, v) for e, v in zip(epochs, series, strict=False) if v is not None
                ]
                if not points:
                    continue
                axis.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    style,
                    color=colour,
                    label=f"{name} {split}",
                    linewidth=1.4,
                )
        axis.set_title(f"{report['pair']} - {title}")
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
        if metric in ("accuracy", "corrected_macro_f1"):
            axis.set_ylim(0, 1)

    figure.suptitle(
        f"{report['pair']} ({report['scope']}): bounding-box crop vs full frame"
    )
    figure.tight_layout()
    for suffix in ("png", "svg"):
        path = plots_dir / f"{report['pair'].lower()}_crop_vs_fullframe.{suffix}"
        figure.savefig(path, dpi=150)
        written.append(str(path))
    plt.close(figure)
    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = base_parser(description=__doc__ or "")
    parser.add_argument(
        "--pair",
        choices=[p[0] for p in PAIRS],
        default=None,
        help="Compare only one pair (default: all).",
    )
    parser.add_argument(
        "--no-flips",
        action="store_true",
        help="Skip per-sample flip analysis, which re-scores both checkpoints.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure rendering.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Compare every pair and write the reports."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config, _ = bootstrap(args)

    checkpoints_dir = Path(config.paths.checkpoints_dir)
    reports_dir = Path(config.paths.reports_dir)
    plots_dir = Path(config.paths.plots_dir) / "crop_experiments"

    selected = [p for p in PAIRS if args.pair is None or p[0] == args.pair]
    reports: list[dict[str, Any]] = []
    for (
        label,
        control,
        treatment,
        control_config,
        treatment_config,
        scope,
    ) in selected:
        if not (checkpoints_dir / control).is_dir():
            LOGGER.warning("skipping %s: %s not found", label, control)
            continue
        if not (checkpoints_dir / treatment).is_dir():
            LOGGER.warning("skipping %s: %s not found", label, treatment)
            continue
        report = compare_pair(
            label,
            control,
            treatment,
            control_config,
            treatment_config,
            scope,
            checkpoints_dir,
            reports_dir,
            with_flips=not args.no_flips,
        )
        if not args.no_plots:
            report["figures"] = _plot_pair(report, plots_dir)
        reports.append(report)

    if not reports:
        LOGGER.error("no completed pairs found under %s", checkpoints_dir)
        return 1

    output = reports_dir / "crop_experiment_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"pairs": reports}, indent=2, sort_keys=True), encoding="utf-8"
    )

    for report in reports:
        macro = report["metrics"]["macro_f1"]
        balanced = report["metrics"]["balanced_accuracy"]
        accuracy = report["metrics"]["accuracy"]
        print(f"\n=== {report['pair']} ({report['scope']}) crop vs full frame ===")
        print(
            f"  macro F1          {macro['full_frame']:.4f} -> {macro['crop']:.4f}  "
            f"{macro['absolute_change']:+.4f} "
            f"({macro['relative_change_percent']:+.2f}%)"
        )
        print(
            f"  balanced accuracy {balanced['full_frame']:.4f} -> "
            f"{balanced['crop']:.4f}  {balanced['absolute_change']:+.4f}"
        )
        print(
            f"  accuracy          {accuracy['full_frame']:.4f} -> "
            f"{accuracy['crop']:.4f}  {accuracy['absolute_change']:+.4f}"
        )
        gap = report["metrics"].get("final_train_val_gap")
        if gap:
            print(
                f"  train-val gap     {gap['full_frame']:+.4f} -> "
                f"{gap['crop']:+.4f}  {gap['absolute_change']:+.4f}"
            )
        print(
            f"  best epoch        {report['control']['best_epoch']} -> "
            f"{report['treatment']['best_epoch']}"
        )
        flips = report.get("flips")
        if flips:
            print(
                f"  flips             +{flips['corrected_by_cropping']} corrected / "
                f"-{flips['broken_by_cropping']} broken = "
                f"{flips['net']:+d} of {flips['samples_scored']}"
            )
        verdict = report["verdict"]
        state = (
            "IMPROVED"
            if verdict["improved"]
            else "REGRESSED"
            if verdict["regressed"]
            else "INDISTINGUISHABLE FROM NOISE"
        )
        print(f"  verdict           {state} (single seed)")

    print(f"\nreport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
