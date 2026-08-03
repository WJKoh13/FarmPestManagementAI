"""Train the recommended second deep-CNN run on the selected IP102 classes.

This experiment deliberately uses only the official training and validation
splits. The test split remains untouched until the team selects its final
configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.justin_deep_cnn import build_model  # noqa: E402


SELECTED_CLASS_IDS = (0, 1, 3, 4, 5, 7, 8, 9, 10, 11)
CLASS_TO_INDEX = {
    original_class: index
    for index, original_class in enumerate(SELECTED_CLASS_IDS)
}
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "IP102_v1.1" / "Classification" / "ip102_v1.1"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "justin_deep_cnn_v2"


class IP102Subset(Dataset[tuple[Tensor, int]]):
    """IP102 samples filtered to the team's fixed ten-class subset."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        transform: transforms.Compose,
    ) -> None:
        self.image_dir = data_root / "images"
        self.transform = transform
        split_file = data_root / f"{split}.txt"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        samples: list[tuple[str, int]] = []
        with split_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.strip().split()
                if not fields:
                    continue
                if len(fields) != 2:
                    raise ValueError(
                        f"Malformed {split_file.name} line {line_number}: {line!r}"
                    )
                filename, original_label_text = fields
                original_label = int(original_label_text)
                if original_label in CLASS_TO_INDEX:
                    samples.append((filename, CLASS_TO_INDEX[original_label]))

        if not samples:
            raise ValueError(f"No selected-class samples found in {split_file}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        filename, label = self.samples[index]
        with Image.open(self.image_dir / filename) as image:
            rgb_image = image.convert("RGB")
            return self.transform(rgb_image), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(160, scale=(0.90, 1.00)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.10,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, validation_transform


def confusion_metrics(confusion: Tensor) -> dict[str, float]:
    confusion = confusion.to(torch.float64)
    true_positive = confusion.diag()
    predicted = confusion.sum(dim=0)
    actual = confusion.sum(dim=1)
    precision = true_positive / predicted.clamp_min(1.0)
    recall = true_positive / actual.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": (true_positive.sum() / confusion.sum()).item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            predictions = logits.argmax(dim=1)
            indices = labels.detach().cpu() * 10 + predictions.detach().cpu()
            confusion += torch.bincount(indices, minlength=100).reshape(10, 10)

    metrics = confusion_metrics(confusion)
    metrics["loss"] = total_loss / sample_count
    return metrics


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    best_epoch: int,
    best_val_f1: float,
    epochs_without_improvement: int,
    history: list[dict[str, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "config": config,
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_transform, validation_transform = make_transforms()
    train_dataset = IP102Subset(data_root, "train", train_transform)
    validation_dataset = IP102Subset(data_root, "val", validation_transform)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    counts = Counter(label for _, label in train_dataset.samples)
    class_weights = torch.tensor(
        [len(train_dataset) / (10 * counts[index]) for index in range(10)],
        dtype=torch.float32,
        device=device,
    )
    model = build_model(
        num_classes=10,
        classifier_dropout=0.20,
        stage_dropouts=(0.00, 0.00, 0.05, 0.10),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    config: dict[str, Any] = {
        "experiment": "justin_deep_cnn_v2",
        "selected_original_class_ids": list(SELECTED_CLASS_IDS),
        "num_classes": 10,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "image_size": 160,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "seed": args.seed,
        "optimizer": "AdamW",
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=3)",
        "selection_metric": "validation_macro_f1",
        "classifier_dropout": 0.20,
        "stage_dropouts": [0.00, 0.00, 0.05, 0.10],
        "train_crop_scale": [0.90, 1.00],
        "gaussian_blur": False,
        "random_erasing": False,
        "device": str(device),
        "test_split_evaluated": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    start_epoch = 1
    best_epoch = 0
    best_val_f1 = float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    if args.resume and last_checkpoint_path.is_file():
        saved = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best_epoch = int(saved["best_epoch"])
        best_val_f1 = float(saved["best_val_f1"])
        epochs_without_improvement = int(saved["epochs_without_improvement"])
        history = saved["history"]
        print(f"Resuming after epoch {start_epoch - 1}.", flush=True)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Device: {device}", flush=True)
    print(
        f"Training samples: {len(train_dataset)}; validation samples: "
        f"{len(validation_dataset)}",
        flush=True,
    )
    print(f"Trainable parameters: {parameter_count:,}", flush=True)
    print(
        f"Maximum epochs: {args.epochs}; early-stopping patience: {args.patience}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )
        scheduler.step(validation_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "val_macro_f1": validation_metrics["macro_f1"],
            "elapsed_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)

        improved = validation_metrics["macro_f1"] > best_val_f1
        if improved:
            best_val_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            best_epoch,
            best_val_f1,
            epochs_without_improvement,
            history,
            config,
        )
        torch.save(payload, last_checkpoint_path)
        if improved:
            torch.save(payload, output_dir / "best_model.pt")
        write_history(output_dir / "training_history.csv", history)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f}, "
            f"acc {train_metrics['accuracy']:.3f}, "
            f"F1 {train_metrics['macro_f1']:.3f} | "
            f"val loss {validation_metrics['loss']:.4f}, "
            f"acc {validation_metrics['accuracy']:.3f}, "
            f"F1 {validation_metrics['macro_f1']:.3f} | "
            f"lr {learning_rate:.2e} | {row['elapsed_seconds']:.1f}s"
            f"{' | best' if improved else ''}",
            flush=True,
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {args.patience} epochs without improvement.",
                flush=True,
            )
            break

    summary = {
        "status": "completed",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "test_split_evaluated": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"Completed. Best epoch: {best_epoch}; validation macro F1: "
        f"{best_val_f1:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
