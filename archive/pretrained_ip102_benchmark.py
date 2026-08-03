"""Pretrained ResNet-18 benchmark on the team's ten-class IP102 subset.

This script is intentionally separate from the scratch-built final model. It
uses ImageNet pretrained weights only to provide an external comparison.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


SELECTED_ORIGINAL_IDS = [0, 1, 3, 4, 5, 7, 8, 9, 10, 11]
ORIGINAL_TO_PROJECT = {
    original_id: project_id
    for project_id, original_id in enumerate(SELECTED_ORIGINAL_IDS)
}
CLASS_NAMES = [
    "rice leaf roller",
    "rice leaf caterpillar",
    "asiatic rice borer",
    "yellow rice borer",
    "rice gall midge",
    "brown plant hopper",
    "white backed plant hopper",
    "small brown plant hopper",
    "rice water weevil",
    "rice leafhopper",
]
EXPECTED_SPLIT_SIZES = {"train": 4318, "val": 721, "test": 2166}
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder containing images/, train.txt, val.txt, and test.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/pretrained_resnet18"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Use 0 for the safest behavior on Windows.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Validate paths and filtered split counts, then exit.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IP102Subset(Dataset):
    """Official IP102 split filtered to the shared ten rice-pest classes."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        transform: transforms.Compose,
    ) -> None:
        if split not in EXPECTED_SPLIT_SIZES:
            raise ValueError("split must be train, val, or test")

        self.data_root = data_root
        self.image_dir = data_root / "images"
        self.transform = transform
        self.records: list[tuple[str, int]] = []

        split_file = data_root / f"{split}.txt"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        for line in split_file.read_text(encoding="utf-8").splitlines():
            filename, original_label_text = line.split()
            original_label = int(original_label_text)
            if original_label in ORIGINAL_TO_PROJECT:
                self.records.append(
                    (filename, ORIGINAL_TO_PROJECT[original_label])
                )

        self.labels = [label for _, label in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        filename, label = self.records[index]
        image_path = self.image_dir / filename
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
        return image_tensor, label


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def create_datasets(data_root: Path) -> dict[str, IP102Subset]:
    train_transform, evaluation_transform = build_transforms()
    return {
        "train": IP102Subset(data_root, "train", train_transform),
        "val": IP102Subset(data_root, "val", evaluation_transform),
        "test": IP102Subset(data_root, "test", evaluation_transform),
    }


def check_datasets(datasets: dict[str, IP102Subset]) -> None:
    for split, expected_size in EXPECTED_SPLIT_SIZES.items():
        actual_size = len(datasets[split])
        print(f"{split}: {actual_size}")
        if actual_size != expected_size:
            raise ValueError(
                f"Unexpected {split} size: {actual_size}; expected {expected_size}"
            )
        if set(datasets[split].labels) != set(range(NUM_CLASSES)):
            raise ValueError(f"Not all project labels occur in the {split} split")
    print("Dataset check passed.")


def create_loaders(
    datasets: dict[str, IP102Subset],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> dict[str, DataLoader]:
    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            **options,
        ),
        "val": DataLoader(datasets["val"], shuffle=False, **options),
        "test": DataLoader(datasets["test"], shuffle=False, **options),
    }


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def confusion_update(
    confusion: Tensor,
    labels: Tensor,
    predictions: Tensor,
) -> None:
    indices = labels * NUM_CLASSES + predictions
    counts = torch.bincount(indices, minlength=NUM_CLASSES * NUM_CLASSES)
    confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES).cpu()


def metrics_from_confusion(confusion: Tensor) -> dict[str, object]:
    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    precision = true_positive / matrix.sum(dim=0).clamp_min(1)
    recall = true_positive / matrix.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    accuracy = true_positive.sum() / matrix.sum().clamp_min(1)
    return {
        "accuracy": accuracy.item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "per_class_f1": f1.tolist(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        confusion_update(confusion, labels, predictions)

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / total_examples
    return metrics


def train_phase(
    phase_name: str,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    best: dict[str, object],
    history: list[dict[str, object]],
) -> None:
    epochs_without_improvement = 0
    for local_epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
        )
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        elapsed = time.perf_counter() - start

        record = {
            "phase": phase_name,
            "epoch": local_epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        }
        history.append(record)

        improved = float(val_metrics["macro_f1"]) > float(best["macro_f1"])
        if improved:
            best["macro_f1"] = float(val_metrics["macro_f1"])
            best["phase"] = phase_name
            best["epoch"] = local_epoch
            best["state_dict"] = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        marker = " * best" if improved else ""
        print(
            f"{phase_name} {local_epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f}, "
            f"F1 {train_metrics['macro_f1']:.3f} | "
            f"val loss {val_metrics['loss']:.4f}, "
            f"F1 {val_metrics['macro_f1']:.3f} | "
            f"{elapsed:.1f}s{marker}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping {phase_name}.")
            break


def save_history(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_root = args.data_root.expanduser().resolve()
    datasets = create_datasets(data_root)
    check_datasets(datasets)
    if args.check_data:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    loaders = create_loaders(
        datasets,
        args.batch_size,
        args.num_workers,
        device,
        args.seed,
    )

    train_counts = np.bincount(datasets["train"].labels, minlength=NUM_CLASSES)
    class_weights = len(datasets["train"]) / (NUM_CLASSES * train_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    model = build_model().to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    initial_trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Total parameters: {total_parameters:,}")
    print(f"Initially trainable parameters: {initial_trainable:,}")

    history: list[dict[str, object]] = []
    best: dict[str, object] = {
        "macro_f1": -1.0,
        "phase": "",
        "epoch": 0,
        "state_dict": None,
    }
    training_start = time.perf_counter()

    head_optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=args.head_learning_rate,
        weight_decay=args.weight_decay,
    )
    train_phase(
        "head",
        model,
        loaders,
        criterion,
        head_optimizer,
        device,
        args.head_epochs,
        args.patience,
        best,
        history,
    )

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    finetune_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.finetune_learning_rate,
        weight_decay=args.weight_decay,
    )
    train_phase(
        "layer4_finetune",
        model,
        loaders,
        criterion,
        finetune_optimizer,
        device,
        args.finetune_epochs,
        args.patience,
        best,
        history,
    )

    if best["state_dict"] is None:
        raise RuntimeError("Training finished without producing a checkpoint")
    model.load_state_dict(best["state_dict"])
    test_metrics = run_epoch(model, loaders["test"], criterion, device)
    training_minutes = (time.perf_counter() - training_start) / 60

    checkpoint_path = args.output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "selected_original_ids": SELECTED_ORIGINAL_IDS,
            "best_validation_macro_f1": best["macro_f1"],
            "best_phase": best["phase"],
            "best_epoch": best["epoch"],
            "seed": args.seed,
        },
        checkpoint_path,
    )
    save_history(args.output_dir / "history.csv", history)

    results = {
        "model": "torchvision_resnet18_imagenet_pretrained",
        "benchmark_only": True,
        "seed": args.seed,
        "selected_original_ids": SELECTED_ORIGINAL_IDS,
        "class_names": CLASS_NAMES,
        "total_parameters": total_parameters,
        "best_validation_macro_f1": best["macro_f1"],
        "best_phase": best["phase"],
        "best_epoch": best["epoch"],
        "training_minutes": training_minutes,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_per_class_f1": {
            name: score
            for name, score in zip(CLASS_NAMES, test_metrics["per_class_f1"])
        },
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nFinal test results")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Results: {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
