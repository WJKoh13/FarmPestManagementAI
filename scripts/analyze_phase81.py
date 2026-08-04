#!/usr/bin/env python3
"""Analyse the Phase 8.1 Stage 1 screening arms against their controls.

Reads completed run artifacts and, for every arm, reports the full quantity set
the phase requires: best macro F1, last-10 mean and standard deviation,
full-coverage accuracy, balanced accuracy, weighted F1, top-5, validation loss,
**train-versus-validation gaps**, per-class F1 changes, classes never predicted,
best epoch, runtime, throughput and peak VRAM. Selective-accuracy curves are
computed by rescoring each arm's ``best.pt`` through **its own** recorded
preprocessing.

Nothing is retrained and no artifact is modified. The test split is refused
everywhere — rescoring routes through the Phase 8.1 ensemble module, which only
accepts ``train`` and ``validation``.

Macro F1 is taken from each run's **corrected** per-class precision and recall
(Phase 7.1), so these figures are directly comparable with every earlier phase.

Examples:
    python scripts/analyze_phase81.py --scope rice10
    python scripts/analyze_phase81.py --scope full102 --config data_full102.yaml
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.vision.ensemble import (
    EnsembleError,
    score_checkpoint,
    selective_accuracy,
    write_report,
)
from farm_pest_ai.vision.results import ResultsError, RunResults, load_run

logger = get_logger("analyze_phase81")

#: Below this, a macro F1 difference on rice10's 721-image validation split is
#: not distinguishable from seed noise. E4 established the stronger form: a
#: single-seed margin under ~0.02 can shrink or reverse across seeds.
NOISE_DELTA = 0.01
SEED_NOISE_DELTA = 0.02

#: rice10 Stage 1: the control first, then each arm and the variable it changed.
RICE10_ARMS: tuple[tuple[str, str, str], ...] = (
    ("E0", "rice10_custom_e0_corrected", "control (lr 0.0015, wd 0.05, no mixing)"),
    ("E6a", "rice10_custom_e6a_lr0008", "learning rate 0.0015 -> 0.0008"),
    ("E6b", "rice10_custom_e6b_lr0030", "learning rate 0.0015 -> 0.0030"),
    ("E7a", "rice10_custom_e7a_mixup", "MixUp alpha 0.2"),
    ("E7b", "rice10_custom_e7b_cutmix", "CutMix alpha 1.0"),
    ("E8", "rice10_custom_e8_supcon", "SupCon aux weight 0.1, T 0.07"),
)

#: full102 Stage 1: the Phase 8 control, then the two weighting arms.
FULL102_ARMS: tuple[tuple[str, str, str], ...] = (
    ("E0-102", "full102_custom_protocolA", "control (class_weighting none)"),
    ("E9a", "full102_custom_e9a_inverse_sqrt", "inverse_sqrt weighting (9.06x)"),
    (
        "E9b",
        "full102_custom_e9b_effective",
        "effective weighting, beta 0.999 (23.53x)",
    ),
)


#: The rice10 confusion groups Phase 7.2's E0 matrix identified. E8 exists to
#: separate exactly these, so its per-class result is reported on them
#: specifically rather than only as a macro average, where a real gain on six
#: classes could be hidden by noise on the other four.
#: Indices verified against data/processed/rice10/class_mapping.json.
RICE10_CONFUSION_GROUPS: dict[str, tuple[int, ...]] = {
    "plant hoppers (brown / white-backed / small brown)": (5, 6, 7),
    "borers (asiatic / yellow rice)": (2, 3),
    "leaf roller vs leaf caterpillar": (0, 1),
}


def _last_n(values: list[float], n: int = 10) -> tuple[float, float]:
    """Mean and sample standard deviation of the final ``n`` values.

    The late-run mean is the more conservative reading: "best epoch" is the
    maximum of a noisy series and systematically favours the luckiest epoch.
    Phase 7.2's E1 ranked second on peak and last on this measure.
    """
    tail = [v for v in values[-n:] if v is not None]
    if not tail:
        return 0.0, 0.0
    if len(tail) == 1:
        return tail[0], 0.0
    return statistics.mean(tail), statistics.stdev(tail)


def _per_class_from_sidecar(
    run: RunResults, best_epoch: int
) -> tuple[list[float], list[int]]:
    """Recover per-class F1 and support from a run's ``best.json`` sidecar.

    ``metrics.jsonl`` omits per-class arrays for ``full102`` by design — 102
    classes x 4 arrays x 60 epochs would make the log unreadable — so for that
    scope the breakdown exists only in the best-epoch sidecar. F1 is recomputed
    from the recorded precision and recall with the **corrected** Phase 7.1
    formula rather than read back, so these values match the rest of the report.

    Returns:
        Per-class F1 and support, or two empty lists when no usable sidecar
        exists. A sidecar describing a different epoch than the one selected is
        rejected rather than silently misattributed.
    """
    import json

    from farm_pest_ai.vision.results import corrected_f1

    sidecar = run.run_dir / "best.json"
    if not sidecar.is_file():
        return [], []
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if int(payload.get("epoch", -1)) != int(best_epoch):
        logger.warning(
            "%s: best.json holds epoch %s but epoch %d was selected; per-class "
            "arrays not used",
            run.run_dir.name,
            payload.get("epoch"),
            best_epoch,
        )
        return [], []

    per_class = (payload.get("metrics") or {}).get("per_class") or {}
    precision = [float(v) for v in per_class.get("precision", ())]
    recall = [float(v) for v in per_class.get("recall", ())]
    support = [int(v) for v in per_class.get("support", ())]
    if not precision or len(precision) != len(recall):
        return [], []
    return [corrected_f1(p, r) for p, r in zip(precision, recall, strict=True)], support


def summarize_run(label: str, description: str, run: RunResults) -> dict[str, Any]:
    """Build the metric block for one arm, from its recorded artifacts."""
    best_epoch = run.best_epoch(corrected=True)
    best = run.best_validation(corrected=True)
    if best is None or best_epoch is None:
        raise ResultsError(f"{run.run_dir} recorded no validation metrics")

    validation_f1 = [
        v for v in run.curve("validation", "corrected_macro_f1") if v is not None
    ]
    last10_mean, last10_sd = _last_n(validation_f1)

    # The train-validation gap at the selected epoch, which is the quantity the
    # whole phase is trying to move.
    train_at_best = next(
        (r.train for r in run.epochs if r.epoch == best_epoch), None
    )
    accuracy_gap = (
        train_at_best.accuracy - best.accuracy if train_at_best else None
    )
    loss_gap = (
        train_at_best.loss - best.loss
        if train_at_best and train_at_best.loss is not None and best.loss is not None
        else None
    )

    # Under MixUp/CutMix the training pass classifies *blended* images while
    # scoring against the original hard labels, so its accuracy is not a
    # generalization measure and its gap against validation is not comparable
    # with an unmixed arm's. E7a reads 0.3764 train against 0.5908 validation —
    # a "negative gap" that describes the difficulty of the augmented images,
    # not a model that generalises better than it fits. Flagged rather than
    # silently tabulated.
    mixing = (run.summary.get("training") or {}).get("mixing") or {}
    mixed_training = str(mixing.get("method", "none")) != "none"

    # full102 records no per-class arrays per epoch, so fall back to the
    # best-epoch sidecar for that scope.
    per_class_f1 = list(best.corrected_per_class_f1)
    per_class_support = list(best.per_class_support)
    if not per_class_f1:
        per_class_f1, per_class_support = _per_class_from_sidecar(run, best_epoch)

    seconds = sum(
        (r.train_seconds or 0.0) + (r.validation_seconds or 0.0) for r in run.epochs
    )
    peak_vram = max((r.peak_vram_mib or 0.0) for r in run.epochs)
    throughput = [
        r.raw.get("images_per_second")
        for r in run.epochs
        if r.raw.get("images_per_second")
    ]

    return {
        "label": label,
        "description": description,
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "scope": run.scope,
        "epochs_run": len(run.epochs),
        "best_epoch": best_epoch,
        "macro_f1": best.corrected_macro_f1,
        "last10_mean": last10_mean,
        "last10_sd": last10_sd,
        "accuracy": best.accuracy,
        "balanced_accuracy": best.balanced_accuracy,
        "weighted_f1": best.corrected_weighted_f1,
        "top5_accuracy": best.top5_accuracy,
        "validation_loss": best.loss,
        "train_accuracy": train_at_best.accuracy if train_at_best else None,
        "train_loss": train_at_best.loss if train_at_best else None,
        "train_validation_accuracy_gap": accuracy_gap,
        "train_validation_loss_gap": loss_gap,
        "mixed_training": mixed_training,
        "gap_comparable_to_control": not mixed_training,
        # Taken from the epoch record the run itself wrote, which is present for
        # both scopes, rather than re-derived from per-class arrays that
        # full102 does not record per epoch.
        "classes_never_predicted": list(
            (
                next(
                    (r.raw for r in run.epochs if r.epoch == best_epoch), {}
                ).get("validation")
                or {}
            ).get("classes_never_predicted", [])
        ),
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
        "runtime_minutes": seconds / 60.0,
        "images_per_second_median": (
            statistics.median(throughput) if throughput else None
        ),
        "peak_vram_mib": peak_vram,
    }


def add_selective(
    block: dict[str, Any], run: RunResults, config: Config
) -> dict[str, Any]:
    """Rescore ``best.pt`` through the run's own preprocessing for abstention."""
    try:
        member = score_checkpoint(
            run, config, checkpoint="best.pt", split="validation", tta="none"
        )
    except EnsembleError as exc:
        logger.warning("%s: cannot rescore for selective accuracy: %s", block["label"], exc)
        block["selective"] = None
        return block

    block["selective"] = [
        entry.to_dict() for entry in selective_accuracy(member.logits, member.targets)
    ]
    block["rescored_checkpoint_epoch"] = member.epoch
    block["rescored_checkpoint_sha256"] = member.checkpoint_sha256
    return block


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = base_parser(
        "Analyse Phase 8.1 Stage 1 screening arms (read-only).",
        default_configs=("base.yaml",),
    )
    parser.add_argument(
        "--no-selective",
        action="store_true",
        help="Skip rescoring checkpoints for selective-accuracy curves.",
    )
    parser.add_argument(
        "--output", metavar="PATH", help="Report destination JSON."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config, _ = bootstrap(args)
    if args.print_config:
        print(config.to_yaml())
        return 0

    scope = config.dataset.scope_name
    arms = RICE10_ARMS if scope == "rice10" else FULL102_ARMS
    checkpoints = config.paths.checkpoints_dir

    blocks: list[dict[str, Any]] = []
    for label, directory, description in arms:
        path = checkpoints / directory
        if not (path / "metrics.jsonl").is_file():
            logger.warning("skipping %s: %s not found", label, path.name)
            continue
        run = load_run(path)
        block = summarize_run(label, description, run)
        if not args.no_selective:
            block = add_selective(block, run, config)
        blocks.append(block)
        logger.info(
            "%s macro_f1=%.4f last10=%.4f+-%.4f acc=%.4f gap=%.4f",
            label,
            block["macro_f1"],
            block["last10_mean"],
            block["last10_sd"],
            block["accuracy"],
            block["train_validation_accuracy_gap"] or 0.0,
        )

    if not blocks:
        logger.error("no runs found under %s", checkpoints)
        return 1

    control = blocks[0]
    for block in blocks:
        delta = block["macro_f1"] - control["macro_f1"]
        late = block["last10_mean"] - control["last10_mean"]
        block["delta_macro_f1"] = delta
        block["delta_last10_mean"] = late
        block["delta_accuracy"] = block["accuracy"] - control["accuracy"]
        block["delta_balanced_accuracy"] = (
            block["balanced_accuracy"] - control["balanced_accuracy"]
        )
        # Both readings must agree before a difference is called real, and even
        # then a single seed cannot confirm it.
        block["verdict"] = (
            "control"
            if block is control
            else "worse"
            if delta < -SEED_NOISE_DELTA
            else "indistinguishable"
            if abs(delta) < NOISE_DELTA
            else "unresolved (inside seed noise)"
            if abs(delta) < SEED_NOISE_DELTA
            else "candidate"
            if delta > 0 and late > 0
            else "unresolved (readings disagree)"
        )
        block["per_class_f1_delta"] = [
            arm - ctrl
            for arm, ctrl in zip(
                block["per_class_f1"], control["per_class_f1"], strict=False
            )
        ]

        if scope == "rice10":
            # E8 targets these groups specifically, so a real gain confined to
            # them would be diluted in a 10-class macro average.
            block["confusion_groups"] = {
                name: {
                    "labels": list(labels),
                    "mean_f1": statistics.mean(
                        block["per_class_f1"][i] for i in labels
                    ),
                    "delta_vs_control": statistics.mean(
                        block["per_class_f1_delta"][i] for i in labels
                    ),
                }
                for name, labels in RICE10_CONFUSION_GROUPS.items()
            }
        else:
            # full102: the Phase 8 support-quartile grouping, so E9's tail
            # effect is measured the same way the control was.
            support = control["per_class_support"]
            order = sorted(range(len(support)), key=lambda i: support[i])
            size = max(1, len(order) // 4)
            quartiles = {
                "Q1 rarest": order[:size],
                "Q2": order[size : 2 * size],
                "Q3": order[2 * size : 3 * size],
                "Q4 largest": order[3 * size :],
            }
            block["support_quartiles"] = {
                name: {
                    "classes": len(labels),
                    "support_range": [
                        min(support[i] for i in labels),
                        max(support[i] for i in labels),
                    ],
                    "mean_f1": statistics.mean(
                        block["per_class_f1"][i] for i in labels
                    ),
                    "delta_vs_control": statistics.mean(
                        block["per_class_f1_delta"][i] for i in labels
                    ),
                }
                for name, labels in quartiles.items()
                if labels
            }

    report = {
        "phase": "8.1",
        "stage": "1 screening",
        "generated": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "control": control["label"],
        "seed": config.seed,
        "noise_delta": NOISE_DELTA,
        "seed_noise_delta": SEED_NOISE_DELTA,
        "test_split_used": False,
        "retrained": False,
        "arms": blocks,
    }
    destination = (
        Path(args.output)
        if args.output
        else config.paths.reports_dir / f"phase81_stage1_{scope}.json"
    )
    write_report(destination, report)

    print(f"\nPhase 8.1 Stage 1 — {scope}, validation split, seed {config.seed}\n")
    header = (
        f"{'arm':<6} {'macroF1':>8} {'Δ':>8} {'last10':>8} {'Δ':>8} "
        f"{'acc':>7} {'balacc':>7} {'gap':>7} {'ep':>4}  verdict"
    )
    print(header)
    print("-" * len(header))
    flagged = False
    for block in blocks:
        # A mixed arm's training accuracy is measured on blended images, so its
        # gap is not the same quantity as an unmixed arm's.
        marker = "*" if block["mixed_training"] else " "
        flagged = flagged or block["mixed_training"]
        print(
            f"{block['label']:<6} {block['macro_f1']:>8.4f} "
            f"{block['delta_macro_f1']:>+8.4f} {block['last10_mean']:>8.4f} "
            f"{block['delta_last10_mean']:>+8.4f} {block['accuracy']:>7.4f} "
            f"{block['balanced_accuracy']:>7.4f} "
            f"{(block['train_validation_accuracy_gap'] or 0.0):>6.4f}{marker} "
            f"{block['best_epoch']:>4}  {block['verdict']}"
        )
    if flagged:
        print(
            "\n* mixed-training arm: its train accuracy is measured on BLENDED "
            "images against\n  hard labels, so the gap is not comparable with an "
            "unmixed arm's and a negative\n  value does not mean the model "
            "generalises better than it fits."
        )

    group_key = "confusion_groups" if scope == "rice10" else "support_quartiles"
    if any(block.get(group_key) for block in blocks):
        title = (
            "per-group mean F1 (Δ vs control)"
            if scope == "rice10"
            else "per-support-quartile mean F1 (Δ vs control)"
        )
        names = list(next(b[group_key] for b in blocks if b.get(group_key)))
        print(f"\n{title}")
        print(f"{'arm':<6}" + "".join(f"{n[:26]:>28}" for n in names))
        for block in blocks:
            groups = block.get(group_key)
            if not groups:
                continue
            cells = "".join(
                f"{groups[n]['mean_f1']:>18.4f}"
                f"({groups[n]['delta_vs_control']:>+7.4f})"
                for n in names
            )
            print(f"{block['label']:<6}{cells}")

    if not args.no_selective:
        print("\nselective accuracy (coverage / accuracy among answered)")
        print(f"{'arm':<6} {'@0.5':>16} {'@0.7':>16} {'@0.9':>16}")
        for block in blocks:
            if not block.get("selective"):
                continue
            cells = "".join(
                f"{e['coverage']:>7.3f}/{e['selective_accuracy']:<8.3f}"
                for e in block["selective"]
            )
            print(f"{block['label']:<6} {cells}")
        print(
            "\nSelective accuracy is accuracy among ANSWERED predictions only; "
            "it is not full-coverage accuracy."
        )

    print(f"\nreport: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
