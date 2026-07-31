"""Plotting for the required run artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window during training
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_loss_curve(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax.plot(epochs, [h["val_loss"] for h in history], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_curve(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [h["val_accuracy"] for h in history], label="val accuracy")
    ax.plot(epochs, [h["val_macro_f1"] for h in history], label="val macro F1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("Validation accuracy and macro F1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix, class_names: list[str], out_path: Path) -> None:
    cm = np.asarray(matrix, dtype=float)
    # Row-normalize so classes with very different support stay comparable.
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    size = max(6.0, 0.7 * len(class_names) + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (counts, shaded by row fraction)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if normalized[i, j] > 0.5 else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
