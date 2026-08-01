#!/usr/bin/env python3
"""Verify the data loader and preprocessing pipeline against the real dataset.

The Phase 5 gate. Where ``verify_dataset.py`` checks that the manifests agree
with the source files, this script checks that the tensors handed to the model
are the ones the project promised:

* shapes, dtypes and label ranges are correct for the active scope,
* the ten mislabelled PNG/RGBA files decode to exactly three channels,
* evaluation preprocessing is **deterministic**: two passes over the same
  validation batch produce bit-identical tensors, and the pipeline contains no
  random step,
* training augmentation is actually random, and is applied to training only,
* evaluation loaders preserve official manifest order and drop nothing,
* class weights are derived from the training split alone,
* the preprocessing fingerprint is recorded.

By default only ``train`` and ``validation`` are touched. ``--include-test``
reads the test manifest and is intended for Phase 9; it decodes images but never
trains or tunes on them.

Examples:
    python scripts/verify_loader.py
    python scripts/verify_loader.py --scope full102
    python scripts/verify_loader.py --batches 4 --report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farm_pest_ai.cli import base_parser, bootstrap
from farm_pest_ai.config import Config
from farm_pest_ai.data.dataset import DatasetError, load_image
from farm_pest_ai.data.loaders import (
    LoaderBundle,
    LoaderError,
    build_loaders,
    sampler_weights,
)
from farm_pest_ai.data.manifests import atomic_write_text
from farm_pest_ai.data.transforms import (
    EVAL_SPLITS,
    PREPROCESSING_VERSION,
    build_transform,
    describe_transform,
    preprocessing_config_from_config,
)
from farm_pest_ai.logging_config import get_logger
from farm_pest_ai.reproducibility import environment_snapshot

#: Torchvision transform classes that introduce randomness. An evaluation
#: pipeline containing any of these is a bug, not a style choice.
RANDOM_STEP_MARKERS = ("Random", "Jitter", "Erasing")

#: The ten `.jpg` files that are really PNG, found in Phase 4. Seven are RGBA.
#: Pinned here so the loader's RGB conversion is proved against the real files
#: rather than a synthetic fixture.
PNG_MASQUERADING_AS_JPG = (
    "40256.jpg", "40314.jpg", "40549.jpg", "40557.jpg", "40563.jpg",
    "40574.jpg", "40577.jpg", "40591.jpg", "40601.jpg", "40630.jpg",
)

Problems = list[str]


def check_shapes(
    bundle: LoaderBundle, batches: int, problems: Problems
) -> dict[str, Any]:
    """Pull real batches and check shape, dtype and label range."""
    import torch

    height, width = bundle.preprocessing.image_size
    observed: dict[str, Any] = {}

    for split, loader in bundle.loaders.items():
        seen_images = 0
        seen_batches = 0
        label_min: int | None = None
        label_max: int | None = None
        started = time.perf_counter()

        for images, labels in loader:
            seen_batches += 1
            seen_images += int(images.shape[0])

            if images.ndim != 4:
                problems.append(
                    f"{split}: expected a 4-D (N, C, H, W) batch, got shape "
                    f"{tuple(images.shape)}"
                )
                break
            if tuple(images.shape[1:]) != (3, height, width):
                problems.append(
                    f"{split}: expected (3, {height}, {width}) per image, got "
                    f"{tuple(images.shape[1:])}"
                )
            if images.dtype is not torch.float32:
                problems.append(
                    f"{split}: expected float32 images, got {images.dtype}"
                )
            if labels.dtype is not torch.int64:
                problems.append(f"{split}: expected int64 labels, got {labels.dtype}")
            if not torch.isfinite(images).all():
                problems.append(f"{split}: batch contains non-finite values")

            batch_min = int(labels.min())
            batch_max = int(labels.max())
            label_min = batch_min if label_min is None else min(label_min, batch_min)
            label_max = batch_max if label_max is None else max(label_max, batch_max)

            if seen_batches >= batches:
                break

        if label_min is not None and label_min < 0:
            problems.append(f"{split}: label {label_min} is negative")
        if label_max is not None and label_max >= bundle.num_classes:
            problems.append(
                f"{split}: label {label_max} is outside 0..{bundle.num_classes - 1}"
            )

        elapsed = time.perf_counter() - started
        observed[split] = {
            "batches_read": seen_batches,
            "images_read": seen_images,
            "label_min": label_min,
            "label_max": label_max,
            "seconds": round(elapsed, 2),
            "images_per_second": round(seen_images / elapsed, 1) if elapsed > 0 else None,
        }
    return observed


def check_evaluation_is_deterministic(
    bundle: LoaderBundle, problems: Problems
) -> dict[str, Any]:
    """Prove the evaluation pipeline has no randomness.

    Two independent applications of the validation transform to the same decoded
    image must be bit-identical, and the pipeline must contain no random step.
    This is the property that makes a validation score from Phase 7 comparable
    with one from Phase 9.
    """
    import torch

    result: dict[str, Any] = {}
    preprocessing = bundle.preprocessing

    for split in EVAL_SPLITS:
        dataset = bundle.datasets.get(split)
        if dataset is None:
            continue

        steps = describe_transform(dataset.transform)
        random_steps = [
            step for step in steps if any(m in step for m in RANDOM_STEP_MARKERS)
        ]
        if random_steps:
            problems.append(
                f"{split}: evaluation pipeline contains random step(s) {random_steps}; "
                f"validation and test preprocessing must be deterministic"
            )

        transform = build_transform(preprocessing, split)
        identical = True
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            image = load_image(dataset.images_dir / dataset.records[index].filename)
            first = transform(image)
            second = transform(image)
            if not torch.equal(first, second):
                identical = False
                problems.append(
                    f"{split}: applying the evaluation transform twice to "
                    f"{dataset.records[index].filename} gave different tensors"
                )
        result[split] = {"steps": list(steps), "repeatable": identical}

    return result


def check_training_augments(
    bundle: LoaderBundle, problems: Problems
) -> dict[str, Any]:
    """Confirm training augmentation is present, random, and training-only."""
    import torch

    dataset = bundle.datasets.get("train")
    if dataset is None:
        return {}

    steps = describe_transform(dataset.transform)
    augmentation = bundle.preprocessing.augmentation
    random_steps = [step for step in steps if any(m in step for m in RANDOM_STEP_MARKERS)]

    if augmentation.enabled and not random_steps:
        problems.append(
            "train: augmentation is enabled but the pipeline contains no random step"
        )

    varies = None
    if augmentation.enabled:
        transform = build_transform(bundle.preprocessing, "train")
        image = load_image(dataset.images_dir / dataset.records[0].filename)
        # 8 draws: with the configured flip probability and crop scale, identical
        # outputs every time would mean the RNG is not being consulted at all.
        outputs = [transform(image) for _ in range(8)]
        varies = any(not torch.equal(outputs[0], other) for other in outputs[1:])
        if not varies:
            problems.append(
                "train: eight applications of the training transform produced "
                "identical tensors; augmentation is not actually random"
            )

    return {
        "steps": list(steps),
        "augmentation_enabled": augmentation.enabled,
        "random_steps": random_steps,
        "output_varies": varies,
    }


def check_rgb_conversion(config: Config, problems: Problems) -> dict[str, Any]:
    """Decode the ten mislabelled PNG files and confirm three channels.

    Phase 4 found these behind a ``.jpg`` extension, seven of them RGBA. An
    untouched RGBA image would give the CNN a fourth input channel, so this is
    checked against the real files.
    """
    images_dir = config.paths.images_dir
    preprocessing = preprocessing_config_from_config(config)
    transform = build_transform(preprocessing, "validation")
    height, width = preprocessing.image_size

    checked = 0
    for filename in PNG_MASQUERADING_AS_JPG:
        path = images_dir / filename
        if not path.is_file():
            # These files belong to IP102 label 56, which is outside rice10;
            # absence means a partial dataset, not a loader fault.
            continue
        checked += 1
        try:
            image = load_image(path)
        except DatasetError as exc:
            problems.append(str(exc))
            continue
        if image.mode != "RGB":
            problems.append(f"{filename}: decoded as mode {image.mode!r}, expected 'RGB'")
        tensor = transform(image)
        if tuple(tensor.shape) != (3, height, width):
            problems.append(
                f"{filename}: produced shape {tuple(tensor.shape)}, expected "
                f"(3, {height}, {width})"
            )

    if checked == 0:
        problems.append(
            f"none of the {len(PNG_MASQUERADING_AS_JPG)} known PNG-as-JPG files were "
            f"found in {images_dir}; the RGB conversion could not be verified"
        )
    return {"checked": checked, "expected": len(PNG_MASQUERADING_AS_JPG)}


def check_order_preserved(bundle: LoaderBundle, problems: Problems) -> dict[str, Any]:
    """Confirm evaluation loaders keep manifest order and drop no images.

    Per-image predictions in Phase 9 are joined to the manifest by position, so
    a reordered or truncated evaluation loader would silently misattribute every
    prediction.
    """
    result: dict[str, Any] = {}
    for split in EVAL_SPLITS:
        loader = bundle.loaders.get(split)
        dataset = bundle.datasets.get(split)
        if loader is None or dataset is None:
            continue

        sampler_type = type(getattr(loader, "sampler", None)).__name__
        drops = bool(getattr(loader, "drop_last", False))
        if drops:
            problems.append(
                f"{split}: loader has drop_last=True; every evaluation image must be "
                f"scored exactly once"
            )
        if "Sequential" not in sampler_type:
            problems.append(
                f"{split}: loader uses a {sampler_type}, not a sequential sampler; "
                f"official manifest order must be preserved"
            )

        covered = len(loader) * int(loader.batch_size)
        if covered < len(dataset):
            problems.append(
                f"{split}: {len(loader)} batches of {loader.batch_size} cover "
                f"{covered} of {len(dataset)} images"
            )
        result[split] = {
            "sampler": sampler_type,
            "drop_last": drops,
            "batches": len(loader),
            "records": len(dataset),
        }
    return result


def check_class_statistics(bundle: LoaderBundle, problems: Problems) -> dict[str, Any]:
    """Confirm class statistics come from the training split only."""
    train = bundle.datasets.get("train")
    if train is None:
        return {}

    result: dict[str, Any] = {
        "class_weighting": bundle.describe()["class_weighting"],
        "class_weights": (
            list(bundle.class_weights) if bundle.class_weights is not None else None
        ),
    }

    try:
        weights = sampler_weights(train)
    except LoaderError as exc:
        problems.append(str(exc))
    else:
        if len(weights) != len(train):
            problems.append(
                f"sampler weights have length {len(weights)} but the training set "
                f"holds {len(train)} records"
            )
        result["sampler_weight_range"] = [
            round(min(weights), 6),
            round(max(weights), 6),
        ]

    # Deriving weights from an evaluation split must be refused outright.
    for split in EVAL_SPLITS:
        dataset = bundle.datasets.get(split)
        if dataset is None:
            continue
        try:
            sampler_weights(dataset)
        except LoaderError:
            pass
        else:
            problems.append(
                f"{split}: sampler weights were derived from an evaluation split; "
                f"class statistics may only come from training data"
            )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Verify the data loader and preprocessing pipeline against the real dataset.",
        default_configs=("base.yaml",),
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=2,
        help="Batches to pull from each loader for the shape and dtype check.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help=(
            "Also build the test loader. Reads the test manifest, so it is meant "
            "for Phase 9; nothing is trained or tuned on it."
        ),
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a JSON report to <reports_dir>/loader_report_<scope>.json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Override runtime.num_workers for this run; 0 keeps everything in-process.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.workers is not None:
        # Folded into the normal --set path so one precedence rule applies.
        # persistent_workers is incompatible with 0 workers, so it goes too.
        args.overrides = [
            *(args.overrides or []),
            f"runtime.num_workers={args.workers}",
            "runtime.persistent_workers=false",
        ]

    config, seed_state = bootstrap(args)
    logger = get_logger("verify_loader")

    if args.print_config:
        print(config.to_yaml())
        return 0

    scope = config.dataset.scope
    splits = ("train", "validation", "test") if args.include_test else ("train", "validation")
    problems: Problems = []

    try:
        bundle = build_loaders(config, splits, verify_files=False)
    except (LoaderError, DatasetError) as exc:
        print(f"\nFAILED: could not build loaders: {exc}")
        logger.error("loader construction failed", extra={"event": "verify_loader"})
        return 1

    description = bundle.describe()
    throughput = check_shapes(bundle, max(1, args.batches), problems)
    determinism = check_evaluation_is_deterministic(bundle, problems)
    augmentation = check_training_augments(bundle, problems)
    rgb = check_rgb_conversion(config, problems)
    order = check_order_preserved(bundle, problems)
    statistics = check_class_statistics(bundle, problems)

    if description["preprocessing_version"] != PREPROCESSING_VERSION:
        problems.append(
            f"dataset.preprocessing_version is "
            f"{description['preprocessing_version']!r} but farm_pest_ai.data.transforms "
            f"is at {PREPROCESSING_VERSION!r}; reconcile them before training"
        )

    print(f"\n=== Verify loader: scope {scope.name} ===")
    print(f"  classes              {bundle.num_classes}")
    print(f"  device               {bundle.device}")
    print(f"  batch size           {bundle.batch_size}")
    print(f"  seed                 {bundle.seed}")
    print(f"  image size           {bundle.preprocessing.image_size[0]}x"
          f"{bundle.preprocessing.image_size[1]}")
    print(f"  interpolation        {bundle.preprocessing.interpolation}")
    print(f"  preprocessing ver.   {bundle.preprocessing.version}")
    print(f"  fingerprint          {bundle.preprocessing.fingerprint}")
    augment_state = "on" if augmentation.get("augmentation_enabled") else "off"
    print(f"  augmentation         {augment_state} (train only)")
    print(f"  PNG-as-JPG checked   {rgb['checked']}/{rgb['expected']}")
    for split in splits:
        stats = throughput.get(split, {})
        info = description["splits"].get(split, {})
        print(
            f"  {split.ljust(12)}{info.get('records', 0):>8} images  "
            f"{info.get('batches', 0):>5} batches  "
            f"{stats.get('images_per_second', '-'):>7} img/s  "
            f"{'aug' if info.get('augmented') else 'det'}"
        )

    report = {
        "scope": scope.name,
        "loaders": description,
        "throughput": throughput,
        "evaluation_determinism": determinism,
        "training_augmentation": augmentation,
        "rgb_conversion": rgb,
        "order_preserved": order,
        "class_statistics": statistics,
        "seed_state": seed_state.to_dict(),
        "environment": environment_snapshot(),
        "problems": problems,
    }

    if args.report:
        path = config.paths.reports_dir / f"loader_report_{scope.name}.json"
        atomic_write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\n  report               {path}")

    if problems:
        print(f"\nFAILED: {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        logger.error(
            "loader verification failed",
            extra={"event": "verify_loader", "problems": len(problems)},
        )
        return 1

    print("\nOK: loader shapes, RGB conversion, evaluation determinism and "
          "training-only augmentation all verified.")
    logger.info(
        "loader verification passed",
        extra={
            "event": "verify_loader",
            "scope": scope.name,
            "num_classes": bundle.num_classes,
            "preprocessing_fingerprint": bundle.preprocessing.fingerprint,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
