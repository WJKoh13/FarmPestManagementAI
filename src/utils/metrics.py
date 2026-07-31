"""Metrics, parameter counting and CPU latency measurement."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, num_classes: int) -> dict:
    """Accuracy plus macro and per-class precision/recall/F1, and the confusion matrix.

    ``zero_division=0`` keeps the macro average defined if a class is never predicted.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": {
            "precision": per_p.tolist(),
            "recall": per_r.tolist(),
            "f1": per_f1.tolist(),
            "support": support.tolist(),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total parameters, trainable parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_cpu_latency(
    model: torch.nn.Module,
    image_size: int = 160,
    warmup: int = 20,
    runs: int = 100,
    batch_size: int = 1,
) -> float:
    """Mean single-image forward latency in milliseconds, always measured on CPU.

    The target application is offline, so CPU latency is the number that matters
    regardless of what the model was trained on.
    """
    was_training = model.training
    original_device = next(model.parameters()).device

    model.eval().to("cpu")
    dummy = torch.randn(batch_size, 3, image_size, image_size)

    for _ in range(warmup):
        model(dummy)

    start = time.perf_counter()
    for _ in range(runs):
        model(dummy)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    model.to(original_device)
    if was_training:
        model.train()
    return elapsed_ms / runs
