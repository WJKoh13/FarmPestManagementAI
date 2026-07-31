"""Step 6: small-batch overfit test.

Proves the data, model, loss and optimizer are wired together correctly before
anyone spends hours on a real run. A model that cannot memorize 64 images has a
bug, not a capacity problem.

    python scripts/overfit_test.py --model alexnet

Validation performance is irrelevant here. The only question is whether training
accuracy reaches ~100% and the loss collapses toward zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.data.dataset import IP102ClassificationDataset  # noqa: E402
from src.data.transforms import build_eval_transform, load_norm_stats  # noqa: E402
from src.models import available_models, build_model  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.metrics import count_parameters  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def run_overfit(model_name: str, config: dict, args) -> bool:
    seed_everything(config["seed"])
    device = resolve_device(args.device or config["device"])
    mean, std = load_norm_stats(resolve_path(config["norm_stats"]))

    # Eval transform on purpose: no random augmentation during this diagnostic.
    full = IP102ClassificationDataset(
        manifest_path=resolve_path(config["train_manifest"]),
        dataset_root=resolve_path(config["dataset_root"]),
        transform=build_eval_transform(config["image_size"], mean, std),
    )
    # A fixed, evenly spaced subset so every class is represented.
    stride = max(1, len(full) // args.subset_size)
    indices = list(range(0, len(full), stride))[: args.subset_size]
    subset = Subset(full, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = build_model(model_name, num_classes=config["num_classes"],
                        **config.get("model_kwargs", {})).to(device)
    total_params, _ = count_parameters(model)

    # Plain unweighted loss - class balance is meaningless on 64 images.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    print(f"\n=== overfit test: {model_name} ===")
    print(f"parameters : {total_params:,}")
    print(f"subset     : {len(subset)} images, device {device}")

    final_acc, final_loss = 0.0, float("inf")
    model.train()
    for epoch in range(1, args.iterations + 1):
        running_loss, correct, seen = 0.0, 0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)

        final_loss, final_acc = running_loss / seen, correct / seen
        if epoch % 10 == 0 or epoch == 1:
            print(f"  iter {epoch:3d} | loss {final_loss:.5f} | train acc {final_acc:.4f}")
        if final_acc >= 0.999 and final_loss < 0.01:
            print(f"  iter {epoch:3d} | loss {final_loss:.5f} | train acc {final_acc:.4f}")
            break

    passed = final_acc >= args.accuracy_threshold
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} "
          f"(train acc {final_acc:.4f}, loss {final_loss:.5f})")
    if not passed:
        print("  Investigate: label remapping, output dimension, softmax before the loss,\n"
              "  image/label misalignment, learning rate, missing zero_grad/backward/step,\n"
              "  frozen parameters, excessive augmentation, wrong normalization.")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="alexnet",
                        help="Model name, or 'all' for every registered model")
    parser.add_argument("--config", default=None, help="Defaults to configs/<model>.yaml")
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--accuracy-threshold", type=float, default=0.95)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    names = available_models() if args.model == "all" else [args.model]
    failures = []

    for name in names:
        config_path = args.config or f"configs/{name}.yaml"
        config = load_config(config_path)
        try:
            if not run_overfit(name, config, args):
                failures.append(name)
        except NotImplementedError as exc:
            print(f"\n=== overfit test: {name} ===")
            print(f"  SKIPPED - {exc}")

    if failures:
        print(f"\nFAILED: {failures}")
        sys.exit(1)
    print("\nOverfit test complete.")


if __name__ == "__main__":
    main()
