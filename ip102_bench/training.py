"""The locked training loop.

Call ``train_model`` from your notebook and the protocol is applied for you --
optimizer, schedule, class weights, early stopping, checkpoint selection. Write
your own loop if you prefer, but then it is on you to match protocol.yaml
exactly, or your numbers do not belong in the same table as everyone else's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .protocol import Protocol
from .runtime import resolve_device, seed_everything


@dataclass
class TrainingResult:
    """Everything ``save_run`` needs from the training phase."""

    history: pd.DataFrame
    best_epoch: int
    best_val_metric: float
    best_state_dict: dict = field(repr=False)
    epochs_trained: int
    training_seconds: float
    stopped_early: bool


def result_from_external_run(
    model: nn.Module,
    history: pd.DataFrame,
    *,
    best_epoch: int | None = None,
    best_val_metric: float | None = None,
    training_seconds: float = 0.0,
    state_dict: dict | None = None,
) -> TrainingResult:
    """Wrap a run that was trained somewhere else so ``save_run`` can record it.

    For a model you already trained -- in another repo, in a Colab session, with
    your own loop -- and do not want to retrain. Give it the model with its
    trained weights loaded and whatever per-epoch history you kept, and you get
    a :class:`TrainingResult` that ``save_run`` accepts.

    ``history`` should have an ``epoch`` column; the plotting code also wants
    ``train_loss``, ``val_loss``, ``train_accuracy`` and ``val_accuracy``, and
    quietly skips the curves if they are absent.

    Note that this bypasses the protocol: nothing here can verify the external
    run used the same augmentation, schedule or split. Say so in ``notes`` when
    you call ``save_run``, so the table is not read as a clean comparison.
    """
    history = history.copy()
    if "epoch" not in history.columns:
        history.insert(0, "epoch", range(1, len(history) + 1))

    if best_epoch is None or best_val_metric is None:
        if "val_macro_f1" in history.columns and len(history):
            best_row = history.loc[history["val_macro_f1"].idxmax()]
            best_epoch = int(best_row["epoch"]) if best_epoch is None else best_epoch
            best_val_metric = (
                float(best_row["val_macro_f1"]) if best_val_metric is None else best_val_metric
            )
        else:
            best_epoch = int(best_epoch or len(history))
            best_val_metric = float(best_val_metric or float("nan"))

    if state_dict is None:
        state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return TrainingResult(
        history=history,
        best_epoch=int(best_epoch),
        best_val_metric=float(best_val_metric),
        best_state_dict=state_dict,
        epochs_trained=len(history),
        training_seconds=training_seconds,
        stopped_early=False,
    )


def build_optimizer(model: nn.Module, protocol: Protocol) -> torch.optim.Optimizer:
    spec = protocol.training
    name = str(spec["optimizer"]).lower()
    kwargs = {"lr": float(spec["learning_rate"]), "weight_decay": float(spec["weight_decay"])}
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
    raise ValueError(f"Unsupported optimizer '{name}' in protocol.yaml")


def build_criterion(protocol: Protocol, train_loader: DataLoader, device) -> nn.Module:
    """Weighted cross-entropy by default, because the classes are imbalanced."""
    from .data import class_counts_of, weights_from_counts

    loss_name = str(protocol.training["loss"]).lower()
    if loss_name == "weighted_cross_entropy":
        counts = class_counts_of(train_loader.dataset, protocol.num_classes)
        return nn.CrossEntropyLoss(weight=weights_from_counts(counts).to(device))
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss '{loss_name}' in protocol.yaml")


@torch.no_grad()
def _evaluate_epoch(model, loader, criterion, device, num_classes):
    from .metrics import compute_metrics

    model.eval()
    total_loss, seen = 0.0, 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in loader:
        images, targets = batch[0], batch[1]  # eval loaders also yield paths
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        total_loss += criterion(logits, targets).item() * targets.size(0)
        seen += targets.size(0)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = total_loss / max(seen, 1)
    return metrics


def train_model(
    model: nn.Module,
    protocol: Protocol,
    loaders: dict[str, DataLoader],
    *,
    device: torch.device | str | None = None,
    epochs: int | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train under the locked protocol and return the best checkpoint.

    The returned ``best_state_dict`` is the epoch with the highest **validation**
    macro F1. The test set is never consulted here.

    Args:
        epochs: Override only for a quick smoke test. Leave as None for real runs
            -- an epoch count that differs from the protocol makes the run
            incomparable, and ``save_run`` records the override so it shows up.
    """
    from .metrics import compute_metrics

    device = torch.device(device) if device else resolve_device(
        protocol.runtime.get("device", "auto")
    )
    seed_everything(protocol.seed)

    num_classes = protocol.num_classes
    total_epochs = epochs if epochs is not None else protocol.epochs
    patience = int(protocol.training["early_stopping_patience"])

    model = model.to(device)
    optimizer = build_optimizer(model, protocol)
    criterion = build_criterion(protocol, loaders["train"], device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # we are maximizing validation macro F1
        factor=float(protocol.training["scheduler_factor"]),
        patience=int(protocol.training["scheduler_patience"]),
    )

    rows: list[dict] = []
    best_metric, best_epoch, epochs_without_gain = -1.0, -1, 0
    best_state: dict = {}
    started = time.perf_counter()

    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        train_true: list[int] = []
        train_pred: list[int] = []

        for images, targets in loaders["train"]:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            seen += targets.size(0)
            train_true.extend(targets.cpu().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

        train_metrics = compute_metrics(train_true, train_pred, num_classes)
        val_metrics = _evaluate_epoch(model, loaders["validation"], criterion, device, num_classes)

        scheduler.step(val_metrics["macro_f1"])
        rows.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(seen, 1),
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        improved = val_metrics["macro_f1"] > best_metric
        if improved:
            best_metric = val_metrics["macro_f1"]
            best_epoch = epoch
            # Detach to CPU: keeping GPU tensors here holds the whole graph's memory.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1

        if verbose:
            mark = "  <- best" if improved else ""
            print(
                f"epoch {epoch:3d}/{total_epochs}  "
                f"train loss {rows[-1]['train_loss']:.4f}  "
                f"val loss {val_metrics['loss']:.4f}  "
                f"val acc {val_metrics['accuracy']:.4f}  "
                f"val macroF1 {val_metrics['macro_f1']:.4f}{mark}"
            )

        if epochs_without_gain >= patience:
            if verbose:
                print(f"early stop: {patience} epochs without a validation improvement")
            break

    return TrainingResult(
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_val_metric=best_metric,
        best_state_dict=best_state,
        epochs_trained=len(rows),
        training_seconds=time.perf_counter() - started,
        stopped_early=len(rows) < total_epochs,
    )
