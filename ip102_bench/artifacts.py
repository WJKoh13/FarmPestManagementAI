"""The run artifact contract.

This is the interface between a notebook and the comparison table. Any notebook
that calls :func:`save_run` shows up in ``compare.py`` -- a from-scratch CNN, a
fine-tuned ResNet, anything -- and nothing else has to be shared between them.

Each run writes ``runs/<model>/<run_id>/``::

    results.json          metrics, params, latency, protocol snapshot, environment
    training_history.csv  per-epoch losses and metrics
    best_model.pt         weights at the best validation macro F1
    predictions.csv       per-image prediction, for error analysis
    curves.png            train vs validation accuracy and loss
    confusion_matrix.png  counts, shaded by row fraction
    per_class_f1.png      sorted worst-first
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import compute_metrics, count_parameters, measure_cpu_latency
from .plots import plot_confusion_matrix, plot_curves, plot_per_class_f1
from .protocol import Protocol, resolve_path
from .runtime import describe_environment, resolve_device
from .training import TrainingResult


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@torch.no_grad()
def predict_split(model, loader: DataLoader, device) -> tuple[list, list, list, list]:
    """Run inference and return ``(y_true, y_pred, confidence, paths)``."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    paths: list[str] = []

    for batch in loader:
        images, targets = batch[0], batch[1]
        batch_paths = batch[2] if len(batch) > 2 else [""] * targets.size(0)

        images = images.to(device, non_blocking=True)
        # Softmax lives here, never inside the model -- CrossEntropyLoss wants logits.
        probs = F.softmax(model(images), dim=1)
        confidence, predicted = probs.max(dim=1)

        y_true.extend(targets.tolist())
        y_pred.extend(predicted.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
        paths.extend(batch_paths)

    return y_true, y_pred, confidences, paths


def save_run(
    *,
    model: torch.nn.Module,
    model_name: str,
    protocol: Protocol,
    result: TrainingResult,
    test_loader: DataLoader,
    pretrained: bool,
    author: str = "",
    notes: str = "",
    architecture: str = "",
    device: torch.device | str | None = None,
    run_id: str | None = None,
    load_best: bool = True,
) -> Path:
    """Evaluate on the test set and write the full artifact set.

    Call this once, at the end of your notebook. It restores the best-validation
    checkpoint before touching the test set, so the test split never influences
    which epoch was chosen.

    Args:
        model_name: Short identifier, e.g. ``vgg16`` or ``resnet18_finetuned``.
            Becomes the directory name and the comparison-table row label.
        pretrained: True if any weight in this model started from someone else's
            training. The comparison table separates the two groups, so getting
            this wrong is the one thing that actually corrupts the results.
        author: Whose run this is. Useful when four people report the same model.
        architecture: Optional free-text description for the report.

    Returns:
        The run directory.
    """
    device = torch.device(device) if device else resolve_device(
        protocol.runtime.get("device", "auto")
    )
    run_id = run_id or new_run_id()
    output_root = resolve_path(protocol.runtime.get("output_root", "runs"))
    run_dir = output_root / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if load_best and result.best_state_dict:
        model.load_state_dict(result.best_state_dict)
    model.to(device)

    class_names = protocol.class_names
    num_classes = len(class_names)
    # The transform these weights were trained through. Recorded in the
    # checkpoint below, not only here, because the app rebuilds preprocessing
    # from the checkpoint alone.
    mean, std = protocol.norm_stats()

    y_true, y_pred, confidences, paths = predict_split(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, num_classes)
    total_params, trainable_params = count_parameters(model)
    latency_ms = measure_cpu_latency(model, image_size=protocol.image_size)

    checkpoint_path = run_dir / "best_model.pt"
    torch.save(
        {
            "model_name": model_name,
            "state_dict": result.best_state_dict or model.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes,
            "display_names": protocol.display_names,
            "epoch": result.best_epoch,
            "val_macro_f1": result.best_val_metric,
            "protocol_version": protocol.version,
            "pretrained": pretrained,
            # The preprocessing contract. A state dict carries weights, not the
            # transform that produced them, and the app cannot recover these
            # from the file otherwise -- it falls back to constants that are
            # wrong for a protocol run (128px, ImageNet statistics) and serves
            # every prediction through a distribution the model never saw. That
            # failure is silent, which is why it is written here rather than
            # left to a default.
            "image_size": protocol.image_size,
            "mean": mean,
            "std": std,
            "crop_mode": protocol.crop_mode,
            "crop_margin": protocol.crop_margin,
        },
        checkpoint_path,
    )
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    result.history.to_csv(run_dir / "training_history.csv", index=False)

    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["image_path", "true_label", "true_class", "predicted_label",
             "predicted_class", "confidence", "correct"]
        )
        for path, true, pred, conf in zip(paths, y_true, y_pred, confidences):
            writer.writerow([path, true, class_names[true], pred, class_names[pred],
                             round(conf, 4), int(true == pred)])

    per_class = {
        class_names[i]: {
            "precision": round(metrics["per_class"]["precision"][i], 4),
            "recall": round(metrics["per_class"]["recall"][i], 4),
            "f1": round(metrics["per_class"]["f1"][i], 4),
            "support": int(metrics["per_class"]["support"][i]),
        }
        for i in range(num_classes)
    }

    if not result.history.empty:
        plot_curves(result.history, run_dir / "curves.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, run_dir / "confusion_matrix.png")
    plot_per_class_f1(per_class, run_dir / "per_class_f1.png")

    results = {
        "model_name": model_name,
        "run_id": run_id,
        "author": author,
        "pretrained": bool(pretrained),
        "architecture": architecture,
        "notes": notes,
        "created": datetime.now().isoformat(timespec="seconds"),

        "protocol_version": protocol.version,
        "subset": protocol.subset_name,
        "crop_mode": protocol.crop_mode,
        "crop_margin": protocol.crop_margin,
        "image_size": protocol.image_size,
        "seed": protocol.seed,

        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(size_mb, 4),
        "cpu_inference_ms": round(latency_ms, 4),

        "best_epoch": result.best_epoch,
        "epochs_trained": result.epochs_trained,
        "stopped_early": result.stopped_early,
        "training_seconds": round(result.training_seconds, 2),
        "best_val_macro_f1": round(float(result.best_val_metric), 6),

        "test_accuracy": round(metrics["accuracy"], 6),
        "macro_precision": round(metrics["macro_precision"], 6),
        "macro_recall": round(metrics["macro_recall"], 6),
        "macro_f1": round(metrics["macro_f1"], 6),
        "per_class": per_class,
        "confusion_matrix": metrics["confusion_matrix"],
        "class_names": class_names,

        "protocol": protocol.to_dict(),
        "environment": describe_environment(),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'class':<28}{'prec':>8}{'recall':>8}{'f1':>8}{'n':>7}")
    print("-" * 59)
    for name, scores in per_class.items():
        print(f"{name:<28}{scores['precision']:>8.4f}{scores['recall']:>8.4f}"
              f"{scores['f1']:>8.4f}{scores['support']:>7d}")
    print("-" * 59)
    print(f"test accuracy : {results['test_accuracy']:.4f}")
    print(f"macro F1      : {results['macro_f1']:.4f}   <- primary metric")
    print(f"parameters    : {total_params:,} ({trainable_params:,} trainable)")
    print(f"model size    : {size_mb:.2f} MB")
    print(f"CPU latency   : {latency_ms:.2f} ms/image")
    print(f"\nArtifacts written to {run_dir}")
    return run_dir
