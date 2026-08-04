"""Shared trainer (Step 4). The same command trains any registered model.

    python -m src.train --config configs/alexnet.yaml

Only the config file changes between experiments. Nothing in this file is
architecture-specific, which is what makes the five-model comparison controlled.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
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
    "train_macro_f1",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "learning_rate",
    "epoch_seconds",
]


def build_dataloaders(
    config: dict,
    mean,
    std,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, IP102ClassificationDataset, torch.Generator]:
    image_size = config["image_size"]
    dataset_root = resolve_path(config["dataset_root"])

    train_set = IP102ClassificationDataset(
        manifest_path=resolve_path(config["train_manifest"]),
        dataset_root=dataset_root,
        transform=build_train_transform(
            image_size,
            mean,
            std,
            profile=config.get("augmentation_profile", "controlled_v1"),
        ),
    )
    val_set = IP102ClassificationDataset(
        manifest_path=resolve_path(config["val_manifest"]),
        dataset_root=dataset_root,
        transform=build_eval_transform(
            image_size,
            mean,
            std,
            profile=config.get("eval_profile", "resize_center_crop"),
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(config["seed"])
    common = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "worker_init_fn": seed_worker,
        "persistent_workers": config["num_workers"] > 0,
        "pin_memory": device.type != "cpu",
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, drop_last=False, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader, train_set, generator


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


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    num_classes: int,
    desc: str,
) -> dict:
    model.train()
    running_loss, seen = 0.0, 0
    all_true: list[int] = []
    all_pred: list[int] = []

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
        seen += targets.size(0)
        all_true.extend(targets.cpu().tolist())
        all_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    metrics = compute_metrics(all_true, all_pred, num_classes)
    metrics["loss"] = running_loss / max(seen, 1)
    return metrics


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a checkpoint without exposing a partially written target file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def capture_rng_state(generator: torch.Generator) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "dataloader_generator": generator.get_state(),
    }
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        state["xpu"] = torch.xpu.get_rng_state_all()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["dataloader_generator"])
    if "xpu" in state and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_rng_state_all(state["xpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="e.g. configs/alexnet.yaml")
    parser.add_argument("--run-id", default=None, help="Defaults to a UTC timestamp.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke tests).")
    parser.add_argument("--device", default=None, help="cpu | xpu | mps | cuda | auto")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a last_checkpoint.pt created by this trainer.",
    )
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

    resume_path = resolve_path(args.resume) if args.resume else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    if resume_path is not None and args.run_id is not None:
        raise ValueError("Do not combine --resume with --run-id; the run directory comes from the checkpoint.")

    run_id = resume_path.parent.name if resume_path else (
        args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = resume_path.parent if resume_path else (
        resolve_path(config["output_root"]) / config["model"] / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_norm_stats(resolve_path(config["norm_stats"]))
    train_loader, val_loader, train_set, train_generator = build_dataloaders(
        config, mean, std, device
    )
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
    config_path = run_dir / "config.yaml"
    if resume_path is None:
        config_path.write_text(
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
    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    start_epoch = 1

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint["model"] != config["model"]:
            raise ValueError(
                f"Checkpoint model {checkpoint['model']!r} does not match config model {config['model']!r}."
            )
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint["history"]
        best_f1 = float(checkpoint["best_val_macro_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint["rng_state"], train_generator)
        print(f"resuming         : epoch {start_epoch} from {resume_path}")

    # Rebuild the CSV from checkpointed history so it remains consistent even
    # if the process previously stopped between its CSV and checkpoint writes.
    with history_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    for epoch in range(start_epoch, config["epochs"] + 1):
        started = time.perf_counter()
        train = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            num_classes=num_classes,
            desc=f"epoch {epoch}/{config['epochs']}",
        )
        val = evaluate_epoch(model, val_loader, criterion, device, num_classes)
        current_learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(val["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": round(train["loss"], 6),
            "train_accuracy": round(train["accuracy"], 6),
            "train_macro_f1": round(train["macro_f1"], 6),
            "val_loss": round(val["loss"], 6),
            "val_accuracy": round(val["accuracy"], 6),
            "val_macro_f1": round(val["macro_f1"], 6),
            "learning_rate": current_learning_rate,
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
            atomic_torch_save(
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
            f"train loss {train['loss']:.4f} acc {train['accuracy']:.4f} "
            f"macroF1 {train['macro_f1']:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['accuracy']:.4f} "
            f"macroF1 {val['macro_f1']:.4f} | {row['epoch_seconds']:.1f}s{marker}"
        )

        atomic_torch_save(
            {
                "checkpoint_type": "last",
                "model": config["model"],
                "model_kwargs": config.get("model_kwargs", {}),
                "num_classes": num_classes,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "class_names": train_set.class_names,
                "seed": config["seed"],
                "rng_state": capture_rng_state(train_generator),
            },
            run_dir / "last_checkpoint.pt",
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
