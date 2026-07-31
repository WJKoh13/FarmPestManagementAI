"""Shared trainer (Step 4). The same command trains any registered model.

    python -m src.train --config configs/alexnet.yaml

Only the config file changes between experiments. Nothing in this file is
architecture-specific, which is what makes the five-model comparison controlled.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config, resolve_path
from src.data.dataset import IP102ClassificationDataset
from src.data.transforms import build_eval_transform, build_train_transform, load_norm_stats
from src.models import build_model
from src.utils.device import resolve_device
from src.utils.metrics import compute_metrics, count_parameters
from src.utils.plots import plot_loss_curve, plot_metric_curve
from src.utils.seed import seed_everything, seed_worker

HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "learning_rate",
    "epoch_seconds",
]


def build_dataloaders(config: dict, mean, std) -> tuple[DataLoader, DataLoader, IP102ClassificationDataset]:
    image_size = config["image_size"]
    dataset_root = resolve_path(config["dataset_root"])

    train_set = IP102ClassificationDataset(
        manifest_path=resolve_path(config["train_manifest"]),
        dataset_root=dataset_root,
        transform=build_train_transform(image_size, mean, std),
    )
    val_set = IP102ClassificationDataset(
        manifest_path=resolve_path(config["val_manifest"]),
        dataset_root=dataset_root,
        transform=build_eval_transform(image_size, mean, std),
    )

    generator = torch.Generator()
    generator.manual_seed(config["seed"])
    common = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "worker_init_fn": seed_worker,
        "persistent_workers": config["num_workers"] > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, drop_last=False, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader, train_set


def compute_class_weights(train_set: IP102ClassificationDataset) -> torch.Tensor:
    """Inverse-frequency weights from the TRAINING split only.

    weight_c = N / (num_classes * count_c), so rare pests are not drowned out by
    the common ones. Macro F1 is the primary metric, so this matters.
    """
    counts = torch.tensor(train_set.class_counts(), dtype=torch.float)
    total = counts.sum()
    weights = total / (len(counts) * counts.clamp(min=1))
    return weights


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, num_classes: int) -> dict:
    model.eval()
    running_loss, seen = 0.0, 0
    all_true: list[int] = []
    all_pred: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * targets.size(0)
        seen += targets.size(0)
        all_true.extend(targets.cpu().tolist())
        all_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = compute_metrics(all_true, all_pred, num_classes)
    metrics["loss"] = running_loss / max(seen, 1)
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device, desc: str) -> tuple[float, float]:
    model.train()
    running_loss, correct, seen = 0.0, 0, 0

    # Progress bars are useless in a redirected log file, so switch them off there.
    for images, targets in tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty()):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += targets.size(0)

    return running_loss / max(seen, 1), correct / max(seen, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="e.g. configs/alexnet.yaml")
    parser.add_argument("--run-id", default=None, help="Defaults to a UTC timestamp.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke tests).")
    parser.add_argument("--device", default=None, help="cpu | mps | cuda | auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    overrides = {
        "epochs": args.epochs,
        "device": args.device,
        "num_workers": args.num_workers,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "test_manifest": args.test_manifest,
        "num_classes": args.num_classes,
        "output_root": args.output_root,
    }
    config = load_config(args.config, overrides)

    seed_everything(config["seed"])
    device = resolve_device(config["device"])

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resolve_path(config["output_root"]) / config["model"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_norm_stats(resolve_path(config["norm_stats"]))
    train_loader, val_loader, train_set = build_dataloaders(config, mean, std)
    num_classes = config["num_classes"]

    if train_set.num_classes != num_classes:
        raise ValueError(
            f"Manifest has {train_set.num_classes} classes but config says {num_classes}."
        )

    model = build_model(config["model"], num_classes=num_classes, **config.get("model_kwargs", {}))
    model.to(device)
    total_params, trainable_params = count_parameters(model)

    class_weights = compute_class_weights(train_set).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # primary metric is macro F1: higher is better
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
    )

    # Save the exact config used, so the run is reproducible from disk alone.
    saved_config = {k: v for k, v in config.items() if not k.startswith("_")}
    saved_config["run_id"] = run_id
    saved_config["device_used"] = str(device)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(saved_config, sort_keys=False), encoding="utf-8"
    )

    print(f"model            : {config['model']}")
    print(f"parameters       : {total_params:,} total / {trainable_params:,} trainable")
    print(f"device           : {device}")
    print(f"train / val      : {len(train_set)} / {len(val_loader.dataset)} images")
    print(f"class weights    : {[round(w, 3) for w in class_weights.cpu().tolist()]}")
    print(f"run directory    : {run_dir}\n")

    history: list[dict] = []
    history_path = run_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=HISTORY_FIELDS).writeheader()

    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"epoch {epoch}/{config['epochs']}",
        )
        val = evaluate_epoch(model, val_loader, criterion, device, num_classes)
        scheduler.step(val["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss": round(val["loss"], 6),
            "val_accuracy": round(val["accuracy"], 6),
            "val_macro_f1": round(val["macro_f1"], 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.perf_counter() - started, 2),
        }
        history.append(row)
        # Append immediately so a killed run still keeps its history.
        with history_path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=HISTORY_FIELDS).writerow(row)

        marker = ""
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            marker = "  <- best"
            torch.save(
                {
                    "model": config["model"],
                    "model_kwargs": config.get("model_kwargs", {}),
                    "num_classes": num_classes,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val["macro_f1"],
                    "class_names": train_set.class_names,
                    "seed": config["seed"],
                },
                run_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:3d}/{config['epochs']} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['accuracy']:.4f} "
            f"macroF1 {val['macro_f1']:.4f} | {row['epoch_seconds']:.1f}s{marker}"
        )

        if epochs_without_improvement >= config["early_stopping_patience"]:
            print(
                f"\nEarly stopping: no val macro F1 improvement for "
                f"{config['early_stopping_patience']} epochs."
            )
            break

    plot_loss_curve(history, run_dir / "loss_curve.png")
    plot_metric_curve(history, run_dir / "metric_curve.png")

    summary = {
        "model": config["model"],
        "run_id": run_id,
        "seed": config["seed"],
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_f1, 6),
        "epochs_run": len(history),
        "device": str(device),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nBest epoch {best_epoch} with val macro F1 {best_f1:.4f}")
    print(f"Artifacts written to {run_dir}")
    print(f"\nNext: python -m src.evaluate --run {run_dir.relative_to(resolve_path('.'))}")


if __name__ == "__main__":
    main()
