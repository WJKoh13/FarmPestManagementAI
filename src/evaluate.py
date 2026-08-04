"""Evaluation on the held-out test set (Step 4).

Runs in a fresh process against a saved checkpoint, which also proves the
checkpoint reloads correctly.

    python -m src.evaluate --run runs/alexnet/20260731-120000

The test set is only touched here, after the architecture and best epoch are
already fixed by validation macro F1. Never use these numbers to pick an epoch
or tune a hyperparameter.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config, resolve_path
from src.data.dataset import IP102ClassificationDataset
from src.data.transforms import build_eval_transform, load_norm_stats
from src.models import build_model
from src.utils.device import resolve_device
from src.utils.metrics import compute_metrics, count_parameters, measure_cpu_latency
from src.utils.plots import plot_confusion_matrix


def load_checkpoint(run_dir: Path, device: torch.device):
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model"],
        num_classes=checkpoint["num_classes"],
        **checkpoint.get("model_kwargs", {}),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    return model, checkpoint, size_mb


@torch.no_grad()
def run_inference(model, loader, device):
    all_true: list[int] = []
    all_pred: list[int] = []
    all_conf: list[float] = []
    all_paths: list[str] = []

    for images, targets, paths in tqdm(
        loader, desc="test inference", leave=False, disable=not sys.stderr.isatty()
    ):
        images = images.to(device, non_blocking=True)
        probs = F.softmax(model(images), dim=1)  # softmax here only, never in the model
        confidence, predicted = probs.max(dim=1)

        all_true.extend(targets.tolist())
        all_pred.extend(predicted.cpu().tolist())
        all_conf.extend(confidence.cpu().tolist())
        all_paths.extend(paths)

    return all_true, all_pred, all_conf, all_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory containing best_model.pt")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing existing artifacts for this split.",
    )
    args = parser.parse_args()

    run_dir = resolve_path(args.run)
    results_path = run_dir / f"{args.split}_results.json"
    if results_path.exists() and not args.force:
        raise FileExistsError(
            f"{results_path} already exists. Refusing to repeat {args.split} evaluation; "
            "pass --force only when replacement is intentional."
        )
    config = load_config(run_dir / "config.yaml", {"device": args.device})
    device = resolve_device(config["device"])

    model, checkpoint, size_mb = load_checkpoint(run_dir, device)
    total_params, _ = count_parameters(model)

    mean, std = load_norm_stats(resolve_path(config["norm_stats"]))
    manifest_key = "test_manifest" if args.split == "test" else "val_manifest"
    dataset = IP102ClassificationDataset(
        manifest_path=resolve_path(config[manifest_key]),
        dataset_root=resolve_path(config["dataset_root"]),
        transform=build_eval_transform(
            config["image_size"],
            mean,
            std,
            profile=config.get("eval_profile", "resize_center_crop"),
        ),
        return_path=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    class_names = checkpoint.get("class_names") or dataset.class_names
    print(f"Evaluating {config['model']} on {args.split}: {len(dataset)} images, device {device}")

    y_true, y_pred, confidences, paths = run_inference(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, config["num_classes"])

    print("\nMeasuring CPU inference latency (this is the number that matters offline)...")
    latency_ms = measure_cpu_latency(model, image_size=config["image_size"])

    # predictions.csv - filenames and readable class names, for error analysis.
    predictions_path = run_dir / f"{args.split}_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["image_path", "true_label", "true_class", "predicted_label",
             "predicted_class", "confidence", "correct"]
        )
        for path, true, pred, conf in zip(paths, y_true, y_pred, confidences):
            writer.writerow(
                [path, true, class_names[true], pred, class_names[pred],
                 round(conf, 4), int(true == pred)]
            )

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        run_dir / f"{args.split}_confusion_matrix.png",
    )

    per_class = {
        class_names[i]: {
            "precision": round(metrics["per_class"]["precision"][i], 4),
            "recall": round(metrics["per_class"]["recall"][i], 4),
            "f1": round(metrics["per_class"]["f1"][i], 4),
            "support": int(metrics["per_class"]["support"][i]),
        }
        for i in range(len(class_names))
    }

    results = {
        "model": config["model"],
        "run_id": config.get("run_id", run_dir.name),
        "seed": config["seed"],
        "parameters": total_params,
        "best_epoch": checkpoint.get("epoch", -1),
        "best_val_macro_f1": round(float(checkpoint.get("val_macro_f1", 0.0)), 6),
        "split": args.split,
        "accuracy": round(metrics["accuracy"], 6),
        "macro_precision": round(metrics["macro_precision"], 6),
        "macro_recall": round(metrics["macro_recall"], 6),
        "macro_f1": round(metrics["macro_f1"], 6),
        "cpu_inference_ms": round(latency_ms, 4),
        "model_size_mb": round(size_mb, 4),
        "per_class": per_class,
        "confusion_matrix": metrics["confusion_matrix"],
        "class_names": class_names,
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'class':<32}{'prec':>8}{'recall':>8}{'f1':>8}{'n':>7}")
    print("-" * 63)
    for name, scores in per_class.items():
        print(
            f"{name:<32}{scores['precision']:>8.4f}{scores['recall']:>8.4f}"
            f"{scores['f1']:>8.4f}{scores['support']:>7d}"
        )
    print("-" * 63)
    print(f"accuracy         : {results['accuracy']:.4f}")
    print(f"macro precision  : {results['macro_precision']:.4f}")
    print(f"macro recall     : {results['macro_recall']:.4f}")
    print(f"macro F1         : {results['macro_f1']:.4f}   <- primary metric")
    print(f"parameters       : {total_params:,}")
    print(f"model size       : {size_mb:.2f} MB")
    print(f"CPU latency      : {latency_ms:.2f} ms/image")
    print(f"\nArtifacts written to {run_dir}")


if __name__ == "__main__":
    main()
