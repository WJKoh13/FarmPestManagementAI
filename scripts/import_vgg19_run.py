"""Turn Beatrice's VGG19 checkpoint into a run bundle the app can load.

Her notebook (``notebooks/Beatrice_vgg19_xml_cropped.ipynb``) trains outside the
shared harness: its own XML crop dataset, its own loop, and a plain
``torch.save`` to ``xml_crop_vgg19_best.pth``. That payload stores weights under
``model_state_dict`` and records nothing the app understands -- not which
architecture it is, not the class list, not the preprocessing. This script
records all of it once, next to the weights:

    runs/vgg19_beatrice/<run_id>/
      best_model.pt   {model_name, num_classes, class_names, image_size, ...}
      results.json    measured metrics, provenance, and why it is not comparable

Two things it does that the other importers do not:

* **It evaluates the test split.** Her notebook selects its best epoch on
  *validation accuracy* and never computes macro F1, so an imported bundle would
  carry no score at all -- and the app ranks runs by macro F1, which would put
  hers permanently last. The evaluation here runs under *her* preprocessing, on
  the project's own test manifest, so the number is directly comparable with
  every other run's single-pass score. Her split came from the same
  ``splits_top15.json``, so test images stay test images.
* **It verifies her label ordering.** She remaps IP102 ids through
  ``splits_top15.json``; the app names classes through ``classes_top15.json``.
  Those two agreeing is what makes her label 7 and the app's label 7 the same
  insect. A silent mismatch would relabel every prediction, so it is asserted.

What it does NOT do is fall back to this repository's preprocessing. She trained
on 0.05-padding crops at 128px with ImageNet normalization; serving that through
the protocol's 0.25-margin, 160px, repo-normalized pipeline would quietly
degrade every prediction. The bundle records her values verbatim.

``runs/`` is git-ignored; nothing here copies the checkpoint into version control.

Usage:

    python scripts/import_vgg19_run.py --checkpoint /path/to/xml_crop_vgg19_best.pth
    python scripts/import_vgg19_run.py --checkpoint ... --skip-evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ip102_bench.metrics import compute_metrics, count_parameters, measure_cpu_latency  # noqa: E402
from ip102_bench.models import build_model  # noqa: E402
from ip102_bench.protocol import load_protocol  # noqa: E402

# The registry key the app rebuilds the architecture from.
MODEL_NAME = "vgg19_beatrice"

# Preprocessing her run actually used -- the defaults in her notebook's
# configuration cell. Anything the checkpoint's own ``config`` block records
# overrides these, because that is what the weights were trained with.
HER_IMAGE_SIZE = 128
HER_MEAN = [0.485, 0.456, 0.406]        # ImageNet, not this repo's norm_stats.json
HER_STD = [0.229, 0.224, 0.225]
HER_CROP_MODE = "box"
HER_CROP_MARGIN = 0.05                  # her BOX_PADDING; the protocol uses 0.25

# Marks the bundle as a different experimental regime, so ip102_bench.compare
# cannot put it in the same table as protocol_version 1.
HER_PROTOCOL_VERSION = "external-xmlcrop-vgg19"


def load_her_state(checkpoint_path: Path) -> tuple[dict, dict]:
    """Read her payload and return its weights plus the config block.

    ``weights_only=False`` because the payload is a dict of metadata around the
    tensors, not a bare state dict. The file is one the user points at
    explicitly.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        sys.exit(f"ERROR: {checkpoint_path} is not a checkpoint dictionary.")

    # Her notebook writes 'model_state_dict'. The other importers read
    # 'model_state' and 'state_dict' -- three formats, so name the one we need.
    if "model_state_dict" not in payload:
        sys.exit(
            f"ERROR: {checkpoint_path} has no 'model_state_dict' key; found "
            f"{sorted(payload)}. This importer reads the format written by "
            f"notebooks/Beatrice_vgg19_xml_cropped.ipynb."
        )

    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        sys.exit(f"ERROR: {checkpoint_path} has an empty or malformed 'model_state_dict'.")

    config = payload.get("config") or {}
    config["epoch"] = payload.get("epoch")
    config["best_val_accuracy"] = payload.get("best_val_accuracy")
    return state_dict, config


def verify_label_ordering(protocol) -> None:
    """Fail unless her label ids and the app's class list mean the same thing.

    She builds labels by remapping through ``splits_top15.json``; the app reads
    names from ``classes_top15.json``. Both order the fifteen classes by the
    same IP102 ids, which is the only reason her weights can be served under the
    app's names at all -- so check it rather than trust it.
    """
    splits = json.loads(
        (PROJECT_ROOT / "data_manifests" / "splits_top15.json").read_text(encoding="utf-8")
    )
    # splits_top15.json lists IP102 ids 1-based; protocol.yaml lists them 0-based.
    from_splits = [int(i) - 1 for i in splits["ip102_class_ids"]]
    from_classes = [int(entry["original_label"]) for entry in protocol.class_metadata]

    if from_splits != from_classes:
        sys.exit(
            "ERROR: class ordering disagrees between data_manifests/splits_top15.json "
            f"({from_splits}) and the protocol's class list ({from_classes}). Her "
            "labels were assigned from the former and would be displayed with names "
            "from the latter, so every prediction would be mislabelled. Refusing to "
            "write a bundle."
        )
    print(f"[ok] label ordering matches across both manifests ({len(from_splits)} classes)")


def evaluate_on_test_split(model, protocol, *, image_size: int, mean: list[float],
                           std: list[float], crop_margin: float, device: str) -> dict:
    """Score her model on the project test split, under her own preprocessing.

    Her evaluation transform is a plain square resize -- no resize-then-centre-
    crop -- so this rebuilds that rather than reusing the harness eval transform,
    which would feed the model a differently framed image than it was trained on.
    """
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from ip102_bench.data import IP102Dataset

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = IP102Dataset(
        protocol.manifest("test"),
        protocol.image_root,
        transform=transform,
        crop_mode=HER_CROP_MODE,
        crop_margin=crop_margin,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Her AdaptiveAvgPool2d((7, 7)) sees a 4x4 feature map at 128px, and Metal
    # has no kernel for a non-divisible adaptive pool. Probe once rather than
    # discovering it partway through scoring 1,447 images. The app itself only
    # ever runs CUDA or CPU, so this costs it nothing.
    model = model.to(device).eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, image_size, image_size, device=device))
    except RuntimeError as error:
        print(f"[!!] {device} cannot run this architecture ({str(error).splitlines()[0]});"
              f" falling back to cpu")
        device = "cpu"
        model = model.to(device)
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for index, (images, targets) in enumerate(loader, start=1):
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(targets.tolist())
            if index % 10 == 0:
                print(f"    scored {len(y_true)}/{len(dataset)} images", flush=True)

    metrics = compute_metrics(y_true, y_pred, protocol.num_classes)
    print(f"[ok] test accuracy {metrics['accuracy']:.4f}, "
          f"macro F1 {metrics['macro_f1']:.4f} on {len(y_true)} images")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="her xml_crop_vgg19_best.pth")
    parser.add_argument("--run-id", help="output folder name (default: today + best epoch)")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="do not score the test split (leaves the run unranked)")
    parser.add_argument("--device", default="cpu",
                        help="device for the evaluation pass (default: cpu)")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")

    protocol = load_protocol()
    class_names = protocol.class_names
    display_names = protocol.display_names
    verify_label_ordering(protocol)

    state_dict, config = load_her_state(args.checkpoint)

    # The class count is the checkpoint's own. If it disagrees with the app's
    # class list the import must fail loudly rather than reinterpret one head
    # under another class list.
    num_classes = int(config.get("num_classes") or len(class_names))
    if len(class_names) != num_classes:
        sys.exit(
            f"ERROR: the checkpoint is {num_classes}-class but protocol.yaml's "
            f"subset '{protocol.subset_name}' defines {len(class_names)} classes. "
            f"Refusing to relabel one class list with another."
        )

    # Her config exposes a 512-unit head as an option; infer which one these
    # weights actually carry instead of assuming the 4096 default.
    hidden_units = state_dict["classifier.0.bias"].shape[0] if "classifier.0.bias" in state_dict else 4096
    model = build_model(MODEL_NAME, num_classes=num_classes,
                        strict_classifier=hidden_units == 4096,
                        small_classifier_units=hidden_units)

    # Strict or nothing. A bundle that loads with missing, unexpected or
    # mis-shaped keys is a different model wearing this one's name.
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        sys.exit(f"ERROR: {args.checkpoint} does not load strictly into "
                 f"{MODEL_NAME}({num_classes} classes):\n\n{error}")

    parameters, _ = count_parameters(model)
    print(f"[ok] loaded strictly into {MODEL_NAME}: "
          f"{parameters:,} parameters, {num_classes} classes, {hidden_units}-unit head")

    image_size = int(config.get("image_size") or HER_IMAGE_SIZE)
    crop_margin = float(config.get("box_padding") if config.get("box_padding") is not None
                        else HER_CROP_MARGIN)
    best_epoch = int(config.get("epoch") or 0)
    best_val_accuracy = config.get("best_val_accuracy")

    test_block = None
    if not args.skip_evaluation:
        print(f"[..] scoring the test split at {image_size}px, {crop_margin} crop margin")
        metrics = evaluate_on_test_split(
            model, protocol, image_size=image_size, mean=HER_MEAN, std=HER_STD,
            crop_margin=crop_margin, device=args.device,
        )
        test_block = {
            "accuracy": round(metrics["accuracy"], 6),
            "macro_precision": round(metrics["macro_precision"], 6),
            "macro_recall": round(metrics["macro_recall"], 6),
            "macro_f1": round(metrics["macro_f1"], 6),
            "per_class_f1": [round(v, 6) for v in metrics["per_class"]["f1"]],
            "confusion_matrix": metrics["confusion_matrix"],
            "measured_by": "scripts/import_vgg19_run.py, single pass, her preprocessing",
        }

    latency_ms = measure_cpu_latency(model, image_size=image_size)

    run_id = args.run_id or (
        f"{datetime.now():%Y%m%d}_epoch{best_epoch}" if best_epoch else f"{datetime.now():%Y%m%d}"
    )
    run_dir = PROJECT_ROOT / "runs" / MODEL_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "best_model.pt"
    torch.save(
        {
            "model_name": MODEL_NAME,
            "num_classes": num_classes,
            "class_names": class_names,
            "display_names": display_names,
            # The preprocessing contract: what the app rebuilds her transform
            # from. Hers, not the protocol's.
            "image_size": image_size,
            "mean": HER_MEAN,
            "std": HER_STD,
            "crop_mode": HER_CROP_MODE,
            "crop_margin": crop_margin,
            "state_dict": state_dict,
            "hidden_units": hidden_units,
        },
        checkpoint_path,
    )
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    results = {
        "model": "VGG19 (XML-cropped, trained from scratch)",
        "model_name": MODEL_NAME,
        "author": "Beatrice",

        # Kept out of the protocol-v1 benchmark table: her training settings are
        # not the locked ones, so the comparison would be measuring two things.
        "external": True,
        "pretrained": False,          # trained from random init, just not under this protocol
        "comparable_to_main": False,
        "protocol_version": HER_PROTOCOL_VERSION,

        "subset": protocol.subset_name,
        "classes": class_names,
        "num_classes": num_classes,
        "total_parameters": parameters,
        "parameters": parameters,
        "model_size_mb": round(size_mb, 4),
        "cpu_inference_ms": round(latency_ms, 4),

        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        # Measured here, on this repository's test split, under her pipeline --
        # not copied from her report.
        "test": test_block,

        "preprocessing": {
            "image_size": image_size,
            "mean": HER_MEAN,
            "std": HER_STD,
            "crop_mode": HER_CROP_MODE,
            "crop_margin": crop_margin,
            "normalization": "imagenet",
            "resize": "square resize, no centre crop",
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.0005,
            "scheduler": "reduce_on_plateau (on validation loss)",
            "batch_size": 8,
            "epochs": 10,
            "loss": "cross_entropy (unweighted)",
            "selection_metric": "validation accuracy",
            "seed": 42,
        },
        "not_comparable_note": (
            "Trained outside the shared harness: per-object XML crops with 0.05 padding "
            "at 128px, ImageNet normalization, unweighted cross-entropy, Adam for 10 "
            "epochs, and the best epoch chosen on validation accuracy. The locked "
            "protocol uses a 0.25 margin at 160px, this repository's norm_stats.json, "
            "inverse-frequency loss weighting, 60 epochs and validation macro F1. The "
            "test macro F1 recorded here was measured by this importer on the project "
            f"test split and is directly comparable as a score, but the run itself is "
            f"NOT a protocol v{protocol.version} entry."
        ),
        "imported_from": str(args.checkpoint),
        "imported_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\n[ok] wrote {run_dir}")
    print(f"     {size_mb:.1f} MB checkpoint, {latency_ms:.1f} ms/image on CPU")
    if test_block:
        print(f"     test macro F1 {test_block['macro_f1']:.4f}, "
              f"accuracy {test_block['accuracy']:.4f}")
    else:
        print("     no test score recorded (--skip-evaluation); the app will rank it last")


if __name__ == "__main__":
    main()
