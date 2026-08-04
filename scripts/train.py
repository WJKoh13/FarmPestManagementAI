#!/usr/bin/env python3
"""Run one real training experiment.

This is the entry point for every experiment from Phase 7 onwards.
``scripts/smoke_train.py`` is **not** an alternative: it caps batches per epoch,
runs a single epoch and marks everything it writes ``smoke: true`` precisely so
its numbers can never be read as a result. This script does the opposite — it
refuses to cap anything, and fails rather than quietly training on a slice.

What it guarantees, and checks before the first batch:

* the **entire** training and validation splits are used. The manifest row count
  for each split is compared against the dataset the loader actually built, and
  a mismatch aborts the run,
* the **test split is never built**. ``build_loaders`` is called with exactly
  ``("train", "validation")``, the resulting bundle is asserted to contain no
  test loader or dataset, and naming the test split anywhere is refused,
* an explicit CUDA request never degrades to CPU, which would silently turn an
  approved GPU run into a multi-day one,
* free VRAM is measured and reported before training starts,
* AMP skipped-step counts are recorded per epoch, so a run whose gradients keep
  overflowing is visible rather than looking like slow learning.

``--plan`` resolves and prints everything above, including a runtime estimate
measured from a few real batches, and exits **without training**. That is the
form to run before requesting approval for a full experiment.

On Windows every project command uses the venv interpreter explicitly, since a
bare ``python`` resolves to MSYS2, which cannot run PyTorch.

Examples:
    .venv/Scripts/python.exe scripts/train.py --config model_custom.yaml --plan
    .venv/Scripts/python.exe scripts/train.py --config model_custom.yaml
        --run-name rice10_custom_e1
    .venv/Scripts/python.exe scripts/train.py --config model_baseline.yaml
        --run-name rice10_baseline_e1 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.data.dataset import DatasetError
from farm_pest_ai.data.detection import (
    DetectionDataError,
    build_detection_records,
    detection_root,
    load_boxes,
    partition_records,
    scope_suffix,
)
from farm_pest_ai.data.loaders import LoaderBundle, LoaderError, build_loaders
from farm_pest_ai.data.manifests import (
    ManifestError,
    atomic_write_text,
    manifest_csv_path,
    read_derived_manifest,
)
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.reproducibility import environment_snapshot
from farm_pest_ai.scopes import is_detection_scope
from farm_pest_ai.vision.models import MODEL_NAMES, ModelError, count_parameters
from farm_pest_ai.vision.training import (
    EpochResult,
    Trainer,
    TrainingError,
    build_trainer,
    training_config_from_config,
)

logger = get_logger("train")

#: The only splits a training run may touch. The test split is evaluated once,
#: in Phase 9, after the inference policy is frozen; every architecture,
#: hyperparameter, augmentation, epoch and scope decision uses validation data.
TRAINING_SPLITS: tuple[str, ...] = ("train", "validation")


class TrainingRunError(RuntimeError):
    """Raised when a run must not start, or must not be trusted."""


# -- pre-flight checks ---------------------------------------------------


def assert_no_test_split(bundle: LoaderBundle) -> None:
    """Fail if a test loader or dataset reached the run.

    ``build_loaders`` already omits ``test`` unless it is named, so this is a
    second, independent check on the property that matters most in this phase.
    Belt and braces is warranted: a leaked test loader produces a plausible
    number that silently invalidates every later decision, and nothing
    downstream would notice.
    """
    leaked = sorted(
        {*bundle.loaders, *bundle.datasets} - set(TRAINING_SPLITS)
    )
    if leaked:
        raise TrainingRunError(
            f"the loader bundle carries {leaked}; a training run may build only "
            f"{list(TRAINING_SPLITS)}. The test split is evaluated once, in Phase 9."
        )


def assert_full_splits(config: Config, bundle: LoaderBundle) -> dict[str, Any]:
    """Confirm each split's dataset covers its entire source manifest.

    Counts are read back from disk rather than from a configured expectation,
    so this catches a truncated manifest, a filtered dataset and a subset
    override alike.

    Detection scopes are checked against ``splits_top*.json`` instead of a
    derived CSV, since that file is their manifest. Their expected count is the
    split size **minus** the images dropped for a missing or unusable box: those
    are dropped identically from both arms of a pair, so the run still covers
    every sample the experiment can legitimately use.

    Returns:
        Per-split record counts, for the run report.

    Raises:
        TrainingRunError: If any split is short of its manifest.
    """
    scope = config.dataset.scope
    detection = is_detection_scope(scope)
    coverage: dict[str, Any] = {}

    for split in TRAINING_SPLITS:
        dataset = bundle.datasets.get(split)
        if dataset is None:
            raise TrainingRunError(
                f"the {split!r} split is missing from the bundle; a real run needs "
                f"both {list(TRAINING_SPLITS)}"
            )

        if detection:
            manifest_path = (
                detection_root(config.paths.dataset_root)
                / f"splits_{scope_suffix(scope)}.json"
            )
            try:
                records = build_detection_records(
                    config.paths.dataset_root, scope, split
                )
                boxes, invalid = load_boxes(config.paths.dataset_root, scope)
            except DetectionDataError as exc:
                raise TrainingRunError(
                    f"could not read the {split!r} detection split {manifest_path}: "
                    f"{exc}"
                ) from exc
            partition = partition_records(records, boxes, invalid)
            expected = len(partition.kept)
            dropped: Any = len(partition.dropped)
        else:
            manifest_path = manifest_csv_path(
                config.paths.processed_dir, scope, split
            )
            try:
                records, _ = read_derived_manifest(
                    config.paths.processed_dir, scope, split
                )
            except ManifestError as exc:
                raise TrainingRunError(
                    f"could not read the {split!r} manifest {manifest_path}: {exc}"
                ) from exc
            expected = len(records)
            dropped = None

        actual = len(dataset)
        if actual != expected:
            raise TrainingRunError(
                f"the {split!r} dataset holds {actual} images but its manifest "
                f"{manifest_path} lists {expected}; a real experiment must use the "
                f"entire split, not a subset"
            )
        coverage[split] = {
            "images": actual,
            "manifest": str(manifest_path),
            "batches": len(bundle.loaders[split]),
        }
        if dropped is not None:
            coverage[split]["dropped_for_box"] = dropped
    return coverage


def assert_no_caps(trainer: Trainer) -> None:
    """Fail if the trainer was given a smoke-run batch cap.

    ``max_train_batches`` and ``max_validation_batches`` exist for the smoke
    gate. A capped experiment would report a metric computed over a slice while
    looking exactly like a full run in every artifact it writes.
    """
    if trainer.max_train_batches is not None or trainer.max_validation_batches is not None:
        raise TrainingRunError(
            f"batch caps are set (train={trainer.max_train_batches}, "
            f"validation={trainer.max_validation_batches}); those belong to "
            f"scripts/smoke_train.py, not to an experiment"
        )
    if trainer.smoke:
        raise TrainingRunError("this trainer is marked smoke=true; it is not an experiment")


def vram_snapshot(device: str) -> dict[str, Any]:
    """Report total, free and reserved device memory in MiB.

    Free VRAM is a real constraint here: Phase 1 measured 4,091 MiB free under
    desktop load against 8,188 MiB total, so a batch size chosen against the
    idle figure can fail an hour into a run.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {"device": device, "cuda": False}
    index = torch.device(device).index or 0
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": device,
        "cuda": True,
        "name": properties.name,
        "capability": f"{properties.major}.{properties.minor}",
        "total_mib": round(total_bytes / 2**20, 1),
        "free_mib": round(free_bytes / 2**20, 1),
        "used_mib": round((total_bytes - free_bytes) / 2**20, 1),
        "torch_reserved_mib": round(torch.cuda.memory_reserved(index) / 2**20, 1),
    }


def measure_step_time(
    trainer: Trainer, *, batches: int = 12, warmup: int = 3
) -> dict[str, Any]:
    """Time a few real training steps to estimate the run's wall-clock cost.

    The first batches are discarded: on Windows the DataLoader workers are
    spawned rather than forked, and cuDNN picks its algorithms on the first
    call, so an unwarmed measurement overstates the per-step cost several times
    over.

    This runs real optimiser steps, so the model is left perturbed. Callers that
    go on to train must rebuild the trainer — ``--plan`` exits instead.
    """
    loader = trainer.bundle.loaders["train"]
    model = trainer.model
    model.train()

    timings: list[float] = []
    peak_before = 0
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
        peak_before = torch.cuda.max_memory_allocated(trainer.device)

    for seen, (images, targets) in enumerate(loader, start=1):
        if seen > warmup + batches:
            break
        images = images.to(trainer.device, non_blocking=True)
        targets = targets.to(trainer.device, non_blocking=True)
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
        started = time.perf_counter()

        trainer.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=trainer.amp_enabled):
            loss = trainer.criterion(model(images), targets)
        trainer.scaler.scale(loss).backward()
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()

        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
        elapsed = time.perf_counter() - started
        if seen > warmup:
            timings.append(elapsed)

    if not timings:
        return {"measured": False}

    mean_step = sum(timings) / len(timings)
    peak_mib = None
    if trainer.device.type == "cuda":
        peak_mib = round(
            (torch.cuda.max_memory_allocated(trainer.device) - peak_before) / 2**20, 1
        )
    return {
        "measured": True,
        "batches_timed": len(timings),
        "mean_step_seconds": round(mean_step, 4),
        "min_step_seconds": round(min(timings), 4),
        "max_step_seconds": round(max(timings), 4),
        "images_per_second": round(trainer.bundle.batch_size / mean_step, 1),
        "peak_step_vram_mib": peak_mib,
    }


def estimate_runtime(
    trainer: Trainer, step: dict[str, Any], epochs: int
) -> dict[str, Any]:
    """Turn a measured step time into a per-epoch and whole-run estimate.

    The validation pass is estimated at 40% of a training step's cost per batch:
    it runs under ``no_grad`` with no backward pass and no optimiser step. This
    is an estimate for planning, not a measurement — the reported figure is an
    upper bound in practice, since early stopping usually ends the run sooner.
    """
    if not step.get("measured"):
        return {"estimated": False}

    mean_step = float(step["mean_step_seconds"])
    train_batches = len(trainer.bundle.loaders["train"])
    validation_batches = len(trainer.bundle.loaders["validation"])

    train_seconds = mean_step * train_batches
    validation_seconds = mean_step * 0.4 * validation_batches
    epoch_seconds = train_seconds + validation_seconds
    total_seconds = epoch_seconds * epochs

    return {
        "estimated": True,
        "train_batches_per_epoch": train_batches,
        "validation_batches_per_epoch": validation_batches,
        "epoch_seconds": round(epoch_seconds, 1),
        "epoch_minutes": round(epoch_seconds / 60, 2),
        "max_epochs": epochs,
        "total_seconds": round(total_seconds, 1),
        "total_minutes": round(total_seconds / 60, 1),
        "total_hours": round(total_seconds / 3600, 2),
        "note": (
            "upper bound at the configured epoch cap; early stopping normally "
            "ends the run sooner"
        ),
    }


# -- reporting -----------------------------------------------------------


def summarize_history(history: list[EpochResult]) -> dict[str, Any]:
    """Condense the epoch history into the run's headline record."""
    if not history:
        return {"epochs_completed": 0}

    best = max(history, key=lambda result: result.validation.macro_f1)
    return {
        "epochs_completed": len(history),
        "best_epoch": best.epoch,
        "best_validation_macro_f1": best.validation.macro_f1,
        "best_validation_accuracy": best.validation.accuracy,
        "best_validation_weighted_f1": best.validation.weighted_f1,
        "best_validation_balanced_accuracy": best.validation.balanced_accuracy,
        "final_epoch": history[-1].epoch,
        "final_validation_macro_f1": history[-1].validation.macro_f1,
        "final_train_loss": history[-1].train.loss,
        # Summed across the run: the calibration skips at the start are expected,
        # a total that keeps climbing is not.
        "amp_skipped_steps_total": sum(r.amp_skipped_steps for r in history),
        "amp_skipped_steps_by_epoch": [r.amp_skipped_steps for r in history],
        "optimizer_steps_total": sum(r.optimizer_steps for r in history),
        "amp_final_scale": history[-1].amp_final_scale,
        "total_train_seconds": round(sum(r.train_seconds for r in history), 1),
        "total_validation_seconds": round(
            sum(r.validation_seconds for r in history), 1
        ),
        "peak_vram_mib": max(
            (r.peak_vram_mib for r in history if r.peak_vram_mib is not None),
            default=None,
        ),
    }


def print_plan(
    config: Config,
    bundle: LoaderBundle,
    trainer: Trainer,
    coverage: dict[str, Any],
    vram: dict[str, Any],
    run_dir: Path,
) -> None:
    """Print the resolved plan for this run."""
    training = trainer.config
    model_name = trainer.model_config.name if trainer.model_config else "?"
    parameters = count_parameters(trainer.model)

    print(f"\n=== Training plan: scope {bundle.scope.name} ===")
    print(f"  config sources       {', '.join(p.name for p in config.sources)}")
    print(f"  model                {model_name}")
    print(f"  parameters           {parameters['total']:,}")
    print(f"  classes              {bundle.num_classes} (derived from scope)")
    print(f"  seed                 {bundle.seed}")
    print(f"  device               {bundle.device}")
    print(f"  amp                  {'on' if trainer.amp_enabled else 'off'}")
    print(f"  batch size           {bundle.batch_size}")
    print(f"  optimizer            {training.optimizer} lr={training.learning_rate}")
    print(f"  weight decay         {training.weight_decay}")
    print(f"  schedule             {training.scheduler}, warmup {training.warmup_epochs}")
    print(f"  label smoothing      {training.label_smoothing}")
    print(f"  class weighting      {training.class_weighting}")
    print(f"  epochs (max)         {training.epochs}")
    print(
        f"  early stopping       {training.early_stopping_metric} "
        f"{training.early_stopping_mode}, patience {training.early_stopping_patience}"
    )
    print(f"  fingerprint          {bundle.preprocessing.fingerprint}")
    print(f"  run directory        {run_dir}")

    print("\n  splits (full, no caps):")
    for split, record in coverage.items():
        print(f"    {split:<11} {record['images']:>7,} images  {record['batches']:>5} batches")
    print("    test        not built (Phase 9 only)")

    if vram.get("cuda"):
        print(
            f"\n  vram                 {vram['free_mib']:,.0f} MiB free of "
            f"{vram['total_mib']:,.0f} MiB on {vram['name']}"
        )
    else:
        print(f"\n  vram                 n/a (device {vram['device']})")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Run one real training experiment on the full train and validation splits.",
        default_configs=("model_custom.yaml",),
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_NAMES),
        help="Shorthand for --set model.name=<name>.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Shorthand for --set runtime.device=<device>.",
    )
    parser.add_argument(
        "--epochs", type=int, help="Shorthand for --set training.epochs=<n>."
    )
    parser.add_argument(
        "--run-name",
        metavar="NAME",
        help=(
            "Name of the run directory under the checkpoints directory. Defaults "
            "to <scope>_<model>_<timestamp>."
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help=(
            "Resolve everything, measure a few real batches for a runtime "
            "estimate, print the plan and exit without training."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last.pt in the run directory. Requires --run-name.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "Permit a CPU run. Without this an explicitly requested CUDA device "
            "that is unavailable aborts rather than silently training on CPU."
        ),
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a JSON report to <reports_dir>/train_<run_id>.json.",
    )
    return parser


def run_plan(
    args: argparse.Namespace,
    config: Config,
    bundle: LoaderBundle,
    trainer: Trainer,
    *,
    coverage: dict[str, Any],
    vram: dict[str, Any],
    run_id: str,
    run_dir: Path,
    seed_state: Any,
) -> int:
    """Measure the step cost, print the estimate and exit without training.

    Nothing is written to the run directory: the point of ``--plan`` is to be
    able to inspect exactly what a run would do before approving it, and a
    half-populated checkpoints directory would be indistinguishable from an
    aborted experiment.
    """
    print("\n  measuring step time on real batches ...")
    step = measure_step_time(trainer)
    estimate = estimate_runtime(trainer, step, trainer.config.epochs)
    if estimate.get("estimated"):
        print(
            f"    {step['mean_step_seconds']:.3f} s/step  "
            f"{step['images_per_second']:,.0f} img/s  "
            f"peak step VRAM {step['peak_step_vram_mib']} MiB"
        )
        print(
            f"    ~{estimate['epoch_minutes']:.2f} min/epoch  "
            f"~{estimate['total_minutes']:.1f} min "
            f"({estimate['total_hours']:.2f} h) for {estimate['max_epochs']} epochs"
        )
    print("\n  --plan: nothing was trained and no checkpoint was written.")

    if args.report:
        path = config.paths.reports_dir / f"plan_{run_id}.json"
        atomic_write_text(
            path,
            json.dumps(
                {
                    "plan_only": True,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "scope": bundle.scope.name,
                    "num_classes": config.num_classes,
                    "config_sources": [str(p) for p in config.sources],
                    "training": trainer.config.to_dict(),
                    "model": (
                        trainer.model_config.to_dict() if trainer.model_config else {}
                    ),
                    "parameters": count_parameters(trainer.model),
                    "coverage": coverage,
                    "data": bundle.describe(),
                    "vram": vram,
                    "step_timing": step,
                    "runtime_estimate": estimate,
                    "seed_state": seed_state.to_dict(),
                    "environment": environment_snapshot(),
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
        )
        print(f"  report               {path}")
    return 0


def _run_id_for(args: argparse.Namespace, config: Config) -> str:
    """Resolve the run identifier, which also names the run directory."""
    if args.run_name:
        return str(args.run_name)
    model_name = str(config.get("model.name", "model"))
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return f"{config.dataset.scope_name}_{model_name}_{stamp}"


def _fold_shorthand_overrides(args: argparse.Namespace) -> None:
    """Fold the convenience flags into the single ``--set`` override list.

    Keeping one override list means one precedence rule applies to everything:
    files, then ``FPA__`` environment variables, then overrides.
    """
    extra: list[str] = []
    if args.model:
        extra.append(f"model.name={args.model}")
    if args.device:
        extra.append(f"runtime.device={args.device}")
    if args.epochs is not None:
        extra.append(f"training.epochs={args.epochs}")
    if extra:
        args.overrides = [*(args.overrides or []), *extra]


def _refuse_unusable_invocation(args: argparse.Namespace, config: Config) -> str | None:
    """Return an error message when this invocation must not proceed."""
    if args.resume and not args.run_name:
        return "--resume needs --run-name to know which run to continue."
    # A smoke config caps batches per epoch. Refuse it rather than silently
    # producing a capped "experiment" whose artifacts look entirely normal.
    if config.section("smoke"):
        return (
            "this configuration carries a `smoke` section, which caps batches per "
            "epoch. Use configs/model_custom.yaml or configs/model_baseline.yaml "
            "for a real experiment."
        )
    return None


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    _fold_shorthand_overrides(args)

    config, seed_state = bootstrap(args)

    if args.print_config:
        print(config.to_yaml())
        return 0

    refusal = _refuse_unusable_invocation(args, config)
    if refusal is not None:
        print(f"\nFAILED: {refusal}")
        return 2

    scope = config.dataset.scope
    run_id = _run_id_for(args, config)
    run_dir = config.paths.checkpoints_dir / run_id

    # Training and validation only, named explicitly. build_loaders omits the
    # test split by default; assert_no_test_split re-checks the result.
    try:
        bundle = build_loaders(
            config,
            TRAINING_SPLITS,
            allow_cpu_fallback=bool(args.allow_cpu),
        )
    except (LoaderError, DatasetError) as exc:
        print(f"\nFAILED: could not build loaders: {exc}")
        return 1

    try:
        assert_no_test_split(bundle)
        coverage = assert_full_splits(config, bundle)
    except (TrainingRunError, DatasetError) as exc:
        print(f"\nFAILED: {exc}")
        return 1

    try:
        trainer = build_trainer(config, bundle, run_dir=run_dir, run_id=run_id)
        assert_no_caps(trainer)
    except (TrainingError, ModelError, TrainingRunError) as exc:
        print(f"\nFAILED: could not build the trainer: {exc}")
        return 1

    vram = vram_snapshot(bundle.device)
    print_plan(config, bundle, trainer, coverage, vram, run_dir)

    if args.plan:
        return run_plan(
            args,
            config,
            bundle,
            trainer,
            coverage=coverage,
            vram=vram,
            run_id=run_id,
            run_dir=run_dir,
            seed_state=seed_state,
        )

    if args.resume:
        try:
            resumed_at = trainer.resume()
        except TrainingError as exc:
            print(f"\nFAILED: {exc}")
            return 1
        print(f"\n  resuming at epoch    {resumed_at}")

    print(f"\n  training for up to {trainer.config.epochs} epochs ...\n")
    logger.info(
        "training run started",
        extra={
            "event": "train_start",
            "run_id": run_id,
            "scope": scope.name,
            "model": trainer.model_config.name if trainer.model_config else "?",
            "epochs": trainer.config.epochs,
            "device": bundle.device,
            "free_vram_mib": vram.get("free_mib"),
        },
    )

    started = time.perf_counter()
    try:
        history = trainer.fit()
    except TrainingError as exc:
        print(f"\nFAILED: training raised: {exc}")
        return 1
    except torch.cuda.OutOfMemoryError as exc:
        print(
            f"\nFAILED: CUDA ran out of memory: {exc}\n"
            f"  Free VRAM before the run was {vram.get('free_mib')} MiB. Lower "
            f"training.batch_size and retry."
        )
        return 1
    elapsed = time.perf_counter() - started

    summary = summarize_history(history)
    print(f"\n=== Run {run_id} finished in {elapsed / 60:.1f} min ===")
    if summary["epochs_completed"]:
        print(
            f"  best epoch           {summary['best_epoch']} "
            f"(macro F1 {summary['best_validation_macro_f1']:.4f}, "
            f"accuracy {summary['best_validation_accuracy']:.4f})"
        )
        print(f"  amp skipped steps    {summary['amp_skipped_steps_total']}")
        print(f"  peak vram            {summary['peak_vram_mib']} MiB")
    print(f"  checkpoints          {run_dir}")
    print(f"  metrics              {run_dir / 'metrics.jsonl'}")

    report = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "scope": scope.name,
        "num_classes": config.num_classes,
        "config_sources": [str(p) for p in config.sources],
        "model": trainer.model_config.to_dict() if trainer.model_config else {},
        "parameters": count_parameters(trainer.model),
        "training": training_config_from_config(config).to_dict(),
        "coverage": coverage,
        "data": bundle.describe(),
        "vram_before": vram,
        "vram_after": vram_snapshot(bundle.device),
        "wall_clock_seconds": round(elapsed, 1),
        "summary": summary,
        "epochs": [result.to_dict(per_class=False) for result in history],
        "seed_state": seed_state.to_dict(),
        "environment": environment_snapshot(),
    }
    summary_path = run_dir / "summary.json"
    atomic_write_text(
        summary_path, json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(f"  summary              {summary_path}")

    if args.report:
        path = config.paths.reports_dir / f"train_{run_id}.json"
        atomic_write_text(
            path, json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(f"  report               {path}")

    logger.info(
        "training run finished",
        extra={
            "event": "train_finished",
            "run_id": run_id,
            "scope": scope.name,
            "epochs_completed": summary.get("epochs_completed", 0),
            "best_macro_f1": summary.get("best_validation_macro_f1"),
            "amp_skipped_steps_total": summary.get("amp_skipped_steps_total"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
