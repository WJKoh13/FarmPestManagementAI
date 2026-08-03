"""The standard figures every run produces, so all the reports look alike."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # notebooks and headless scripts both import this
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRAIN_COLOR = "#2b6cb0"
VAL_COLOR = "#c05621"


def plot_curves(history: pd.DataFrame, out_path: str | Path) -> Path:
    """Accuracy and loss, training vs validation, side by side."""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))

    ax_acc.plot(history["epoch"], history["train_accuracy"], color=TRAIN_COLOR, lw=2, label="Training")
    ax_acc.plot(history["epoch"], history["val_accuracy"], color=VAL_COLOR, lw=2, label="Validation")
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy", loc="left", fontweight="bold")

    ax_loss.plot(history["epoch"], history["train_loss"], color=TRAIN_COLOR, lw=2, label="Training")
    ax_loss.plot(history["epoch"], history["val_loss"], color=VAL_COLOR, lw=2, label="Validation")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss", loc="left", fontweight="bold")

    for ax in (ax_acc, ax_loss):
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True); ax.legend(frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(matrix, class_names: list[str], out_path: str | Path) -> Path:
    """Counts, shaded by row fraction.

    Shading by fraction rather than raw count matters here: the classes are very
    unevenly sized, so a raw-count heatmap mostly shows which class is biggest.
    """
    matrix = np.asarray(matrix, dtype=float)
    fractions = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

    size = max(6.0, 0.62 * len(class_names) + 2.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (shaded by row fraction)", loc="left", fontweight="bold")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if fractions[i, j] > 0.5 else "#333")

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_class_f1(per_class: dict, out_path: str | Path) -> Path:
    """Per-class F1, sorted worst first -- the short bars are where errors live."""
    frame = pd.DataFrame(
        [{"class": name, "f1": s["f1"]} for name, s in per_class.items()]
    ).sort_values("f1")

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.38 * len(frame) + 1)))
    ax.barh(frame["class"], frame["f1"], color=TRAIN_COLOR, height=0.65)
    ax.set_xlim(0, 1); ax.set_xlabel("F1")
    ax.set_title("Per-class F1", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
