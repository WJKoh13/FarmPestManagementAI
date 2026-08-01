#!/usr/bin/env python3
"""Prove the training pipeline works end to end, in minutes.

The Phase 6 gate. This is **not** an experiment: it caps batches per epoch and
runs a single epoch, so its metrics are meaningless and are marked ``smoke`` in
every artifact it writes. What it does prove:

* both architectures build for both scopes, output ``num_classes`` raw logits,
  and derive that count from the scope rather than from configuration,
* a forward and backward pass runs on the configured device, with AMP when CUDA
  is present,
* **gradients actually flow**: the model is trained on one small batch until the
  loss collapses. A pipeline can look healthy for an epoch and still be learning
  nothing — a detached tensor, a frozen parameter or a mis-shaped loss all
  produce a plausible-looking flat loss curve. Driving a handful of images to
  near-zero loss is the cheapest test that rules all of it out,
* checkpoints round-trip, and a checkpoint from one scope is **refused** by the
  other,
* resuming restores the optimiser, schedule and epoch,
* metrics, the resolved configuration and the environment are recorded.

Examples:
    python scripts/smoke_train.py
    python scripts/smoke_train.py --scope full102
    python scripts/smoke_train.py --model baseline_cnn --device cpu
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.data.dataset import DatasetError
from farm_pest_ai.data.loaders import LoaderError, build_loaders
from farm_pest_ai.data.manifests import atomic_write_text
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.reproducibility import environment_snapshot
from farm_pest_ai.scopes import scope_names
from farm_pest_ai.vision.checkpoints import (
    CheckpointError,
    best_checkpoint_path,
    last_checkpoint_path,
    load_checkpoint,
    read_metadata,
)
from farm_pest_ai.vision.metrics import label_smoothing_loss_floor
from farm_pest_ai.vision.models import (
    MODEL_NAMES,
    ModelConfig,
    ModelError,
    build_model,
    count_parameters,
    model_config_from_config,
    summarize_model,
)
from farm_pest_ai.vision.training import (
    Trainer,
    TrainingError,
    build_trainer,
    training_config_from_config,
)

Problems = list[str]


def check_architectures(config: Config, problems: Problems) -> dict[str, Any]:
    """Build every architecture for the active scope and check its contract.

    Confirms the output width equals the scope's class count, that the logits
    are raw (no softmax hidden in the model), and that a wrong-scope model is
    rejected.
    """
    expected = config.num_classes
    scope = config.dataset.scope
    results: dict[str, Any] = {}

    for name in MODEL_NAMES:
        model_config = ModelConfig(name=name, num_classes=expected)
        try:
            model = build_model(model_config, scope=scope)
        except ModelError as exc:
            problems.append(f"{name}: failed to build for scope {scope.name}: {exc}")
            continue

        summary = summarize_model(model, input_size=config.dataset.image_size)
        if summary["num_classes"] != expected:
            problems.append(
                f"{name}: produced {summary['num_classes']} logits, expected {expected} "
                f"for scope {scope.name!r}"
            )

        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(4, 3, *config.dataset.image_size))
        # A softmax inside the model would make every row sum to exactly 1 and
        # every value non-negative. Untrained logits do neither.
        row_sums = logits.sum(dim=1)
        if bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)):
            problems.append(
                f"{name}: output rows sum to 1, which means a softmax is being applied "
                f"inside the model; the losses expect raw logits"
            )

        results[name] = {
            "parameters": summary["parameters"],
            "parameter_memory_mib": summary["parameter_memory_mib"],
            "output_shape": summary["output_shape"],
        }

    # A model built for the other scope must be refused for this one.
    others = [s for s in scope_names() if s != scope.name]
    for other in others:
        try:
            build_model(ModelConfig(name="custom_cnn", num_classes=expected), scope=other)
        except ModelError:
            pass
        else:
            problems.append(
                f"a {expected}-class model was accepted under scope {other!r}; scope "
                f"mismatches must be refused"
            )
    return results


def check_num_classes_is_derived(config: Config, problems: Problems) -> dict[str, Any]:
    """Confirm the model refuses a configuration that states ``num_classes``.

    The class count is derived from ``dataset.scope`` in exactly one place. A
    ``model`` section that states its own would be a second source of truth, and
    the two silently disagreeing is how a 10-way checkpoint gets read as a
    102-way one.
    """
    contradictory = Config(
        data={**config.to_dict(), "model": {**config.section("model"), "num_classes": 7}}
    )
    try:
        model_config_from_config(contradictory)
    except ModelError:
        refused = True
    else:
        refused = False
        problems.append(
            "model.num_classes was accepted from configuration; it must always be "
            "derived from dataset.scope"
        )
    return {"stated_num_classes_refused": refused, "derived": config.num_classes}


def check_overfit_small_batch(
    trainer: Trainer,
    *,
    batch_size: int,
    steps: int,
    target_loss: float,
    problems: Problems,
) -> dict[str, Any]:
    """Drive the loss toward zero on a handful of images.

    This is the test that separates "the loop ran" from "the model learned". One
    fixed batch is fed repeatedly with augmentation off and the model in ``train``
    mode; a network whose gradients reach every parameter memorises a few images
    quickly. A detached tensor, a parameter excluded from the optimiser or a
    mis-shaped loss all leave the loss flat here while looking entirely normal in
    a one-epoch run.

    Note that the loss floor is not zero when label smoothing is on: smoothing at
    ``eps`` leaves a residual cross-entropy even for a perfect prediction, so the
    target is compared against that floor rather than against zero.
    """
    dataset = trainer.bundle.datasets["train"]
    device = trainer.device

    # A fixed batch, taken deterministically from the front of the training
    # split, so this check is reproducible run to run.
    images = torch.stack([dataset[i][0] for i in range(batch_size)]).to(device)
    targets = torch.tensor(
        [dataset[i][1] for i in range(batch_size)], dtype=torch.int64, device=device
    )

    model = trainer.model
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=trainer.config.label_smoothing
    )

    # Label smoothing puts a floor under the achievable loss: the minimum is the
    # entropy of the smoothed target, which is 0.50 at eps=0.1 over 10 classes
    # and 0.78 over 102. Comparing against "near zero" would fail a fully
    # converged model, so the target is measured above this floor.
    floor = label_smoothing_loss_floor(
        trainer.config.label_smoothing, trainer.bundle.num_classes
    )

    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - started

    first, last = losses[0], losses[-1]
    threshold = target_loss + floor
    converged = last <= threshold

    # Every trainable parameter must have received a gradient. A parameter with
    # None here is disconnected from the loss entirely.
    without_gradient = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if without_gradient:
        problems.append(
            f"overfit check: {len(without_gradient)} trainable parameter(s) received no "
            f"gradient, e.g. {without_gradient[:5]}"
        )
    if not converged:
        problems.append(
            f"overfit check: loss fell only from {first:.4f} to {last:.4f} over {steps} "
            f"steps on {batch_size} images, above the {threshold:.4f} target "
            f"({target_loss} above the {floor:.4f} label-smoothing floor); gradients "
            f"are not driving the model"
        )

    model.eval()
    with torch.no_grad():
        accuracy = float((model(images).argmax(1) == targets).float().mean())

    return {
        "batch_size": batch_size,
        "steps": steps,
        "first_loss": round(first, 6),
        "final_loss": round(last, 6),
        # Loss above the floor is the honest measure of what is left to learn;
        # the raw final loss cannot fall below `floor` no matter how well the
        # model fits.
        "final_loss_above_floor": round(last - floor, 6),
        "label_smoothing_floor": round(floor, 6),
        "threshold": round(threshold, 6),
        "converged": converged,
        "train_accuracy_on_batch": accuracy,
        "parameters_without_gradient": len(without_gradient),
        "seconds": round(elapsed, 2),
    }


def check_epoch_learned(
    history: list[Any], num_classes: int, problems: Problems
) -> dict[str, Any]:
    """Confirm the capped epoch moved the model off its starting point.

    One epoch of ~960 images is far too short to be a result, but it must still
    produce a measurable score. A run that reports exactly zero is either not
    learning or is being evaluated on a slice too narrow to score, and both are
    regressions the gate should catch. This check is the reason the smoke config
    trains 60 batches rather than 10: at 10, every run scored 0.0 whether the
    pipeline was healthy or not, so the epoch proved nothing.
    """
    final = history[-1].validation
    classes_present = sum(1 for support in final.per_class_support if support > 0)

    if final.macro_f1 <= 0.0:
        problems.append(
            f"after one capped epoch validation macro F1 is {final.macro_f1:.4f}; the "
            f"model has not learned anything measurable, or the validation slice is "
            f"too narrow to score it ({classes_present} of {num_classes} classes "
            f"present)"
        )
    if final.accuracy <= 0.0:
        problems.append(
            f"after one capped epoch validation accuracy is {final.accuracy:.4f}, "
            f"against {1.0 / num_classes:.4f} for random guessing"
        )

    return {
        "macro_f1": final.macro_f1,
        "accuracy": final.accuracy,
        "chance_accuracy": round(1.0 / num_classes, 4),
        "classes_present_in_validation": classes_present,
        "num_classes": num_classes,
    }


def check_checkpoint_roundtrip(
    trainer: Trainer, run_dir: Path, config: Config, problems: Problems
) -> dict[str, Any]:
    """Reload the written checkpoint and confirm provenance is enforced.

    Three things are checked: the weights round-trip bit-exactly, the metadata
    carries the scope and mapping version, and a load under the *other* scope is
    refused.
    """
    scope = config.dataset.scope
    result: dict[str, Any] = {}

    best = best_checkpoint_path(run_dir)
    last = last_checkpoint_path(run_dir)
    path = best if best.is_file() else last
    if not path.is_file():
        problems.append(f"no checkpoint was written to {run_dir}")
        return result

    try:
        metadata = read_metadata(path)
    except CheckpointError as exc:
        problems.append(f"could not read checkpoint metadata: {exc}")
        return result

    for field, expected in (
        ("scope", scope.name),
        ("num_classes", scope.num_classes),
    ):
        actual = getattr(metadata, field)
        if actual != expected:
            problems.append(
                f"checkpoint metadata {field}={actual!r}, expected {expected!r}"
            )
    if not metadata.preprocessing_fingerprint:
        problems.append("checkpoint records no preprocessing fingerprint")
    if not metadata.smoke:
        problems.append(
            "a smoke-run checkpoint is not marked smoke=true; its metrics could be "
            "mistaken for an experiment result"
        )

    # Weights must round-trip exactly.
    try:
        reloaded, _, _ = load_checkpoint(
            path,
            scope=scope,
            map_location=str(trainer.device),
            preprocessing_fingerprint=trainer.bundle.preprocessing.fingerprint,
            strict_preprocessing=True,
        )
    except CheckpointError as exc:
        problems.append(f"could not reload checkpoint: {exc}")
        return result

    original = trainer.model.state_dict()
    restored = reloaded.state_dict()
    mismatched = [
        key
        for key in original
        if key not in restored
        or not torch.equal(original[key].cpu(), restored[key].cpu())
    ]
    if mismatched:
        problems.append(
            f"{len(mismatched)} tensor(s) differ after a checkpoint round-trip, e.g. "
            f"{mismatched[:5]}"
        )

    # The core safety property: the other scope must be refused.
    others = [s for s in scope_names() if s != scope.name]
    refused = {}
    for other in others:
        try:
            load_checkpoint(path, scope=other, map_location="cpu")
        except CheckpointError:
            refused[other] = True
        else:
            refused[other] = False
            problems.append(
                f"a {scope.name!r} checkpoint loaded successfully under scope "
                f"{other!r}; cross-scope loading must be refused"
            )

    result.update(
        {
            "path": str(path),
            "size_mib": round(path.stat().st_size / 2**20, 2),
            "weights_identical": not mismatched,
            "scope": metadata.scope,
            "num_classes": metadata.num_classes,
            "class_mapping_version": metadata.class_mapping_version,
            "preprocessing_fingerprint": metadata.preprocessing_fingerprint,
            "smoke": metadata.smoke,
            "cross_scope_refused": refused,
        }
    )
    return result


def check_resume(trainer: Trainer, run_dir: Path, problems: Problems) -> dict[str, Any]:
    """Confirm a run resumes from ``last.pt`` at the right epoch."""
    path = last_checkpoint_path(run_dir)
    if not path.is_file():
        problems.append("no last.pt was written, so resumption cannot be verified")
        return {}

    try:
        metadata = read_metadata(path)
        resume_epoch = trainer.resume(path)
    except (CheckpointError, TrainingError) as exc:
        problems.append(f"resume failed: {exc}")
        return {}

    expected = metadata.epoch + 1
    if resume_epoch != expected:
        problems.append(
            f"resume returned epoch {resume_epoch}, expected {expected}"
        )
    if trainer.global_step != metadata.global_step:
        problems.append(
            f"resume restored global_step {trainer.global_step}, expected "
            f"{metadata.global_step}"
        )
    return {
        "checkpoint_epoch": metadata.epoch,
        "resume_epoch": resume_epoch,
        "global_step": trainer.global_step,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Run a fast end-to-end smoke test of the training pipeline.",
        default_configs=("smoke_test.yaml",),
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
        "--skip-overfit",
        action="store_true",
        help="Skip the small-batch overfit check, the slowest part of the run.",
    )
    parser.add_argument(
        "--keep-run",
        action="store_true",
        help=(
            "Keep the run directory. By default a smoke run's artifacts are "
            "deleted, since its metrics are meaningless and must never be "
            "mistaken for an experiment."
        ),
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a JSON report to <reports_dir>/smoke_train_<scope>.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)

    extra_overrides: list[str] = []
    if args.model:
        extra_overrides.append(f"model.name={args.model}")
    if args.device:
        extra_overrides.append(f"runtime.device={args.device}")
    if args.epochs is not None:
        extra_overrides.append(f"training.epochs={args.epochs}")
    if extra_overrides:
        args.overrides = [*(args.overrides or []), *extra_overrides]

    config, seed_state = bootstrap(args)
    logger = get_logger("smoke_train")

    if args.print_config:
        print(config.to_yaml())
        return 0

    scope = config.dataset.scope
    smoke_section = config.section("smoke")
    problems: Problems = []

    print(f"\n=== Smoke training: scope {scope.name} ===")

    architectures = check_architectures(config, problems)
    derivation = check_num_classes_is_derived(config, problems)

    # Training and validation only. The test split is untouched until Phase 9,
    # and build_loaders omits it unless it is named explicitly.
    try:
        bundle = build_loaders(config, ("train", "validation"))
    except (LoaderError, DatasetError) as exc:
        print(f"\nFAILED: could not build loaders: {exc}")
        return 1

    run_id = f"smoke_{scope.name}_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir = config.paths.checkpoints_dir / run_id

    try:
        trainer = build_trainer(
            config,
            bundle,
            run_dir=run_dir,
            run_id=run_id,
            smoke=True,
            max_train_batches=smoke_section.get("max_train_batches"),
            max_validation_batches=smoke_section.get("max_validation_batches"),
        )
    except (TrainingError, ModelError) as exc:
        print(f"\nFAILED: could not build the trainer: {exc}")
        return 1

    training_config = training_config_from_config(config)
    parameters = count_parameters(trainer.model)

    model_name = trainer.model_config.name if trainer.model_config else "?"
    print(f"  model                {model_name}")
    print(f"  classes              {bundle.num_classes}")
    print(f"  parameters           {parameters['total']:,}")
    print(f"  device               {bundle.device}")
    print(f"  amp                  {'on' if trainer.amp_enabled else 'off'}")
    print(f"  batch size           {bundle.batch_size}")
    print(f"  steps per epoch      {trainer.steps_per_epoch}")
    print(f"  fingerprint          {bundle.preprocessing.fingerprint}")

    overfit: dict[str, Any] = {}
    if not args.skip_overfit:
        print("\n  running small-batch overfit check ...")
        try:
            overfit = check_overfit_small_batch(
                trainer,
                batch_size=int(smoke_section.get("overfit_batch_size", 8)),
                steps=int(smoke_section.get("overfit_steps", 100)),
                target_loss=float(smoke_section.get("overfit_target_loss", 0.05)),
                problems=problems,
            )
        except RuntimeError as exc:
            # TrainingError derives from RuntimeError, so this covers both.
            problems.append(f"overfit check raised: {exc}")
        else:
            print(
                f"    loss {overfit['first_loss']:.4f} -> {overfit['final_loss']:.4f} "
                f"(threshold {overfit['threshold']:.4f})  "
                f"batch accuracy {overfit['train_accuracy_on_batch']:.2f}"
            )

    # The overfit check deliberately memorises eight images, so the epoch below
    # must start from fresh weights. The whole trainer is rebuilt rather than
    # just the model: the optimiser holds moment estimates for the old
    # parameters and the scheduler is bound to the old optimiser, so replacing
    # the model alone would leave the learning rate driving an optimiser that no
    # longer exists.
    if overfit:
        try:
            trainer = build_trainer(
                config,
                bundle,
                run_dir=run_dir,
                run_id=run_id,
                smoke=True,
                max_train_batches=smoke_section.get("max_train_batches"),
                max_validation_batches=smoke_section.get("max_validation_batches"),
            )
        except (TrainingError, ModelError) as exc:
            print(f"\nFAILED: could not rebuild the trainer: {exc}")
            return 1

    print("\n  running one capped epoch ...")
    try:
        history = trainer.fit()
    except TrainingError as exc:
        print(f"\nFAILED: training raised: {exc}")
        return 1

    learning: dict[str, Any] = {}
    if not history:
        problems.append("training produced no epoch results")
    else:
        learning = check_epoch_learned(history, bundle.num_classes, problems)
        for result in history:
            print(
                f"    epoch {result.epoch}  "
                f"train loss {result.train.loss:.4f}  "
                f"val macro F1 {result.validation.macro_f1:.4f}  "
                f"val acc {result.validation.accuracy:.4f}  "
                f"{result.images_per_second:.0f} img/s"
            )

    checkpoints = check_checkpoint_roundtrip(trainer, run_dir, config, problems)
    resumption = check_resume(trainer, run_dir, problems)

    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        problems.append(f"no metrics.jsonl was written to {run_dir}")
    if not (run_dir / "run.json").is_file():
        problems.append(f"no run.json was written to {run_dir}")

    report = {
        "scope": scope.name,
        "num_classes": config.num_classes,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "architectures": architectures,
        "num_classes_derivation": derivation,
        "training": training_config.to_dict(),
        "data": bundle.describe(),
        "overfit_check": overfit,
        "epoch_learned": learning,
        "epochs": [r.to_dict(per_class=False) for r in history],
        "checkpoints": checkpoints,
        "resume": resumption,
        "seed_state": seed_state.to_dict(),
        "environment": environment_snapshot(),
        "problems": problems,
    }

    if args.report:
        path = config.paths.reports_dir / f"smoke_train_{scope.name}.json"
        atomic_write_text(
            path, json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(f"\n  report               {path}")

    if not args.keep_run and run_dir.is_dir():
        # A smoke run's metrics are meaningless. Leaving them in the checkpoints
        # directory invites a later phase to read them as a result.
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"  run directory        removed ({run_dir.name}); pass --keep-run to keep it")
    elif args.keep_run:
        print(f"  run directory        {run_dir}")

    if problems:
        print(f"\nFAILED: {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        logger.error(
            "smoke training failed",
            extra={"event": "smoke_train", "problems": len(problems)},
        )
        return 1

    print(
        "\nOK: both architectures build, gradients flow, one epoch trains, "
        "checkpoints round-trip and cross-scope loading is refused."
    )
    logger.info(
        "smoke training passed",
        extra={
            "event": "smoke_train",
            "scope": scope.name,
            "num_classes": config.num_classes,
            "parameters": parameters["total"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
