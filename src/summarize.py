"""Two reporting helpers.

Architecture check - parameter count plus a forward/backward pass for every
registered model. Run this before any long training run:

    python -m src.summarize --check

Comparison table - collects every results.json into the table the report needs:

    python -m src.summarize --compare
"""

from __future__ import annotations

import argparse
import json

import torch

from src.config import resolve_path
from src.models import MODEL_REGISTRY, build_model
from src.utils.metrics import count_parameters

BUDGET_LOW, BUDGET_HIGH = 500_000, 5_000_000


def check_models(num_classes: int, image_size: int, batch_size: int) -> int:
    print(f"{'model':<14}{'parameters':>13}{'budget':>10}  forward/backward")
    print("-" * 62)
    failures = 0

    for name in sorted(MODEL_REGISTRY):
        try:
            model = build_model(name, num_classes=num_classes)
        except NotImplementedError:
            print(f"{name:<14}{'-':>13}{'-':>10}  NOT IMPLEMENTED YET")
            failures += 1
            continue

        total, _ = count_parameters(model)
        images = torch.randn(batch_size, 3, image_size, image_size)
        try:
            logits = model(images)
            expected = (batch_size, num_classes)
            if tuple(logits.shape) != expected:
                raise ValueError(f"expected {expected}, got {tuple(logits.shape)}")
            logits.sum().backward()  # proves gradients flow to every parameter
            status = "OK"
        except Exception as exc:  # noqa: BLE001 - report and keep checking the rest
            status = f"FAILED: {exc}"
            failures += 1

        in_budget = "yes" if BUDGET_LOW <= total <= BUDGET_HIGH else "OUT"
        print(f"{name:<14}{total:>13,}{in_budget:>10}  {status}")

    print("-" * 62)
    print(f"budget: {BUDGET_LOW:,} - {BUDGET_HIGH:,} trainable parameters")
    return failures


def compare_runs(output_root: str) -> None:
    root = resolve_path(output_root)
    rows = []
    for results_path in sorted(root.glob("*/*/results.json")):
        rows.append(json.loads(results_path.read_text(encoding="utf-8")))

    if not rows:
        print(f"No results.json found under {root}. Train and evaluate a model first.")
        return

    rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    header = (
        f"| {'Model':<14} | {'Parameters':>11} | {'Accuracy':>8} | {'Macro F1':>8} | "
        f"{'CPU ms':>7} | {'Size MB':>7} |"
    )
    print(header)
    print("|" + "-" * 16 + "|" + "-" * 13 + "|" + "-" * 10 + "|" + "-" * 10 + "|"
          + "-" * 9 + "|" + "-" * 9 + "|")
    for r in rows:
        print(
            f"| {r['model']:<14} | {r['parameters']:>11,} | {r['test_accuracy']:>8.4f} | "
            f"{r['macro_f1']:>8.4f} | {r['cpu_inference_ms']:>7.2f} | "
            f"{r['model_size_mb']:>7.2f} |"
        )
    print("\nMacro F1 is the primary metric, but the final choice should also weigh "
          "CPU latency and size - the application runs offline.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Param counts + shape checks")
    parser.add_argument("--compare", action="store_true", help="Build the comparison table")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-root", default="runs")
    args = parser.parse_args()

    if not args.check and not args.compare:
        args.check = True

    failures = 0
    if args.check:
        failures = check_models(args.num_classes, args.image_size, args.batch_size)
    if args.compare:
        if args.check:
            print()
        compare_runs(args.output_root)

    if args.check and failures:
        print(f"\n{failures} model(s) not ready.")


if __name__ == "__main__":
    main()
