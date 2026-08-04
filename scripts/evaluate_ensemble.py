#!/usr/bin/env python3
"""E5: score single checkpoints, flip-TTA and uniform ensembles on validation.

Inference only. Nothing here trains, fine-tunes or rewrites a checkpoint, which
is what makes E5 runnable before Phase 8.1's training approval: it can only
observe models that already exist.

Every member is scored through **its own** recorded preprocessing with
``strict_preprocessing=True``, so a 224x224 model can never be scored through a
160x160 pipeline. Raw logits are averaged, never predicted labels, and members
are refused unless their scope, class count, sample count and target vectors all
agree. Ensemble weights are uniform: tuning per-member weights on the same
validation split that judges the ensemble would fit the split rather than
measure a method.

The test split cannot be reached from this script. ``--split`` offers only
``validation`` and ``train``, and the scoring layer re-checks.

Examples:
    python scripts/evaluate_ensemble.py --scope rice10
    python scripts/evaluate_ensemble.py --scope full102 --config data_full102.yaml
    python scripts/evaluate_ensemble.py --scope rice10 --checkpoint last.pt
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.reproducibility import environment_snapshot
from farm_pest_ai.vision.ensemble import (
    EnsembleError,
    MemberScores,
    ensemble_logits,
    load_member_run,
    score_checkpoint,
    summarize_scores,
    write_report,
)
from farm_pest_ai.vision.results import RunResults

logger = get_logger("evaluate_ensemble")

#: rice10 members, grouped by the input size they were trained at. The three
#: 160px seeds are the E4 arm that was retained; the three 224px seeds are the
#: arm E4 declined to adopt. Both are ensembled within their own group, and then
#: across groups, because each member carries its own preprocessing.
RICE10_SEED_RUNS_160: tuple[str, ...] = (
    "rice10_custom_e4_s160_seed1337",
    "rice10_custom_e4_s160_seed2024",
    "rice10_custom_e4_s160_seed7",
)
RICE10_SEED_RUNS_224: tuple[str, ...] = (
    "rice10_custom_e4_s224_seed1337",
    "rice10_custom_e4_s224_seed2024",
    "rice10_custom_e4_s224_seed7",
)

#: The full102 custom control. The weaker full102 baseline is deliberately NOT
#: ensembled with it by default: a 0.4258-macro-F1 model averaged uniformly into
#: a 0.5443 one is as likely to drag it down as to help, and the phase brief
#: requires a validation-based reason plus both components reported before such
#: an ensemble is evaluated at all.
FULL102_CUSTOM_RUN = "full102_custom_protocolA"
FULL102_BASELINE_RUN = "full102_baseline_protocolA"


def _member_or_none(
    run_dir: Path,
    config: Config,
    *,
    checkpoint: str,
    split: str,
    tta: str,
) -> MemberScores | None:
    """Score one member, skipping it with a warning when it is unavailable."""
    if not (run_dir / "metrics.jsonl").is_file():
        logger.warning("skipping %s: no metrics.jsonl", run_dir.name)
        return None
    if not (run_dir / checkpoint).is_file():
        logger.warning("skipping %s: no %s", run_dir.name, checkpoint)
        return None
    try:
        run: RunResults = load_member_run(run_dir)
        member = score_checkpoint(
            run, config, checkpoint=checkpoint, split=split, tta=tta
        )
    except EnsembleError as exc:
        logger.warning("skipping %s: %s", run_dir.name, exc)
        return None
    logger.info(
        "scored %s: %d samples, epoch %d, %s, %.1fs",
        member.member_id,
        member.samples,
        member.epoch,
        member.preprocessing_fingerprint,
        member.seconds,
    )
    return member


def _arm(
    label: str,
    members: list[MemberScores],
    num_classes: int,
    *,
    kind: str,
    per_class: bool,
) -> dict[str, Any] | None:
    """Build one report arm, single-member or ensembled."""
    if not members:
        return None
    if len(members) == 1:
        member = members[0]
        return summarize_scores(
            label,
            member.logits,
            member.targets,
            num_classes,
            members=members,
            kind=kind,
            per_class=per_class,
        )
    logits, targets = ensemble_logits(members)
    return summarize_scores(
        label, logits, targets, num_classes, members=members, kind=kind,
        per_class=per_class,
    )


def evaluate_rice10(
    config: Config, checkpoints_dir: Path, *, checkpoint: str, split: str
) -> list[dict[str, Any]]:
    """Run the rice10 arms: singles, flip-TTA, per-size and combined ensembles."""
    num_classes = config.num_classes
    arms: list[dict[str, Any]] = []

    group_members: dict[str, list[MemberScores]] = {"160": [], "224": []}
    for size, names in (("160", RICE10_SEED_RUNS_160), ("224", RICE10_SEED_RUNS_224)):
        for name in names:
            run_dir = checkpoints_dir / name
            plain = _member_or_none(
                run_dir, config, checkpoint=checkpoint, split=split, tta="none"
            )
            if plain is None:
                continue
            group_members[size].append(plain)
            arm = _arm(
                f"rice10 single {name}", [plain], num_classes,
                kind="single", per_class=True,
            )
            if arm:
                arms.append(arm)

            flipped = _member_or_none(
                run_dir, config, checkpoint=checkpoint, split=split, tta="hflip"
            )
            if flipped is not None:
                arm = _arm(
                    f"rice10 hflip-TTA {name}", [flipped], num_classes,
                    kind="tta", per_class=True,
                )
                if arm:
                    arms.append(arm)

    for size in ("160", "224"):
        members = group_members[size]
        if len(members) >= 2:
            arm = _arm(
                f"rice10 uniform ensemble {size}px x{len(members)}",
                members,
                num_classes,
                kind="ensemble",
                per_class=True,
            )
            if arm:
                arms.append(arm)

    # The combined 160+224 ensemble. Legitimate precisely because each member
    # was scored through its own pipeline: the logits describe the same images
    # in the same manifest order regardless of the size they were produced at.
    combined = group_members["160"] + group_members["224"]
    both_sizes_present = bool(group_members["160"]) and bool(group_members["224"])
    if both_sizes_present and len(combined) >= 2:
        arm = _arm(
            f"rice10 uniform ensemble 160+224 x{len(combined)}",
            combined,
            num_classes,
            kind="ensemble",
            per_class=True,
        )
        if arm:
            arm["note"] = (
                "Members trained at two different input sizes. Each was scored "
                "through its own recorded preprocessing (fingerprints differ by "
                "design); alignment is guaranteed by the shared target vector, "
                "which is verified before averaging."
            )
            arms.append(arm)

    return arms


def evaluate_full102(
    config: Config, checkpoints_dir: Path, *, checkpoint: str, split: str
) -> list[dict[str, Any]]:
    """Run the full102 arms: the custom single and its flip-TTA.

    The weaker baseline is scored as a **standalone reference only**. It is not
    ensembled with the custom model: uniform averaging of a much weaker member is
    as likely to hurt as help, and the phase requires a validation-based reason
    plus both components reported before such an ensemble is evaluated.
    """
    num_classes = config.num_classes
    arms: list[dict[str, Any]] = []

    # per_class is False here: 102 classes x 4 arrays per arm would dominate the
    # report exactly as it would dominate metrics.jsonl.
    custom_dir = checkpoints_dir / FULL102_CUSTOM_RUN
    plain = _member_or_none(
        custom_dir, config, checkpoint=checkpoint, split=split, tta="none"
    )
    if plain is not None:
        arm = _arm(
            f"full102 single {FULL102_CUSTOM_RUN}", [plain], num_classes,
            kind="single", per_class=True,
        )
        if arm:
            arms.append(arm)

    flipped = _member_or_none(
        custom_dir, config, checkpoint=checkpoint, split=split, tta="hflip"
    )
    if flipped is not None:
        arm = _arm(
            f"full102 hflip-TTA {FULL102_CUSTOM_RUN}", [flipped], num_classes,
            kind="tta", per_class=True,
        )
        if arm:
            arms.append(arm)

    baseline = _member_or_none(
        checkpoints_dir / FULL102_BASELINE_RUN,
        config,
        checkpoint=checkpoint,
        split=split,
        tta="none",
    )
    if baseline is not None:
        arm = _arm(
            f"full102 single {FULL102_BASELINE_RUN} (reference only)",
            [baseline],
            num_classes,
            kind="single",
            per_class=True,
        )
        if arm:
            arm["note"] = (
                "Reported as an individual component only. Not ensembled with "
                "the custom model: E5 requires a validation-based reason before "
                "uniformly averaging a substantially weaker member."
            )
            arms.append(arm)

    return arms


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = base_parser(
        "Score checkpoints, flip-TTA and uniform ensembles on validation (E5).",
        default_configs=("base.yaml",),
    )
    parser.add_argument(
        "--checkpoint",
        default="best.pt",
        choices=["best.pt", "last.pt"],
        help=(
            "Which checkpoint to score. best.pt holds the epoch the run's "
            "monitored metric selected, which is not always the numerically best "
            "epoch under the corrected metric."
        ),
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "train"],
        help="Split to score. The test split is not offered and is refused.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Report destination. Defaults to data/reports/phase81_e5_<scope>.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config, seed_state = bootstrap(args)

    if args.print_config:
        print(config.to_yaml())
        return 0

    scope = config.dataset.scope_name
    checkpoints_dir = config.paths.checkpoints_dir

    logger.info(
        "E5 %s: scoring %s over the %s split (inference only)",
        scope,
        args.checkpoint,
        args.split,
    )

    if scope == "rice10":
        arms = evaluate_rice10(
            config, checkpoints_dir, checkpoint=args.checkpoint, split=args.split
        )
    else:
        arms = evaluate_full102(
            config, checkpoints_dir, checkpoint=args.checkpoint, split=args.split
        )

    if not arms:
        logger.error("no scorable runs found under %s", checkpoints_dir)
        return 1

    ranked = sorted(
        arms, key=lambda arm: arm["full_coverage"]["macro_f1"], reverse=True
    )
    report: dict[str, Any] = {
        "phase": "8.1",
        "experiment": "E5",
        "generated": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "num_classes": config.num_classes,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "seed": seed_state.seed,
        "inference_only": True,
        "test_split_used": False,
        "ensemble_weights": "uniform",
        "logit_averaging": "raw logits averaged before softmax",
        "arms": arms,
        "ranking": [
            {
                "label": arm["label"],
                "kind": arm["kind"],
                "macro_f1": arm["full_coverage"]["macro_f1"],
                "accuracy": arm["full_coverage"]["accuracy"],
                "balanced_accuracy": arm["full_coverage"]["balanced_accuracy"],
            }
            for arm in ranked
        ],
        "environment": environment_snapshot(),
    }

    destination = (
        Path(args.output)
        if args.output
        else config.paths.reports_dir / f"phase81_e5_{scope}.json"
    )
    write_report(destination, report)

    print(f"\nE5 {scope} — {args.split} split, {args.checkpoint}, inference only\n")
    header = f"{'arm':<52} {'macro F1':>9} {'accuracy':>9} {'bal acc':>9}"
    print(header)
    print("-" * len(header))
    for arm in ranked:
        metrics = arm["full_coverage"]
        print(
            f"{arm['label'][:52]:<52} {metrics['macro_f1']:>9.4f} "
            f"{metrics['accuracy']:>9.4f} {metrics['balanced_accuracy']:>9.4f}"
        )

    best = ranked[0]
    print(f"\nselective accuracy — {best['label']}")
    print(f"{'threshold':>10} {'coverage':>10} {'answered':>10} {'sel. acc':>10}")
    for entry in best["selective"]:
        print(
            f"{entry['threshold']:>10.2f} {entry['coverage']:>10.4f} "
            f"{entry['answered']:>10d} {entry['selective_accuracy']:>10.4f}"
        )
    print(
        "\nSelective accuracy is accuracy among ANSWERED predictions only. "
        "It is not full-coverage accuracy."
    )
    print(f"\nreport: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
