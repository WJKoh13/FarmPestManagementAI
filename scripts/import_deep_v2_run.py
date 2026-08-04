"""Package Justin's selected Broad15 DeepV2 checkpoint for the application.

This is an external run relative to the detection_top15 benchmark used by the
main application. It preserves the classification split metrics and the exact
stretch preprocessing, but deliberately marks the protocols as incomparable.
No training or evaluation is performed.

Usage:
    python scripts/import_deep_v2_run.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ip102_bench.models import build_model  # noqa: E402

DEFAULT_SOURCE = (
    PROJECT_ROOT / "runs" / "broad15" / "final" /
    "deep_v2_seed42_selected" / "best_model.pt"
)
DEFAULT_RESULTS = DEFAULT_SOURCE.with_name("test_results.json")
MODEL_NAME = "justin_deep_v2"
MODEL_KWARGS = {
    "classifier_dropout": 0.2,
    "stage_dropouts": [0.0, 0.0, 0.05, 0.1],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--test-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-id", default="broad15_epoch48_external")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")
    if not args.test_results.is_file():
        sys.exit(f"ERROR: test results not found: {args.test_results}")

    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source, dict) or "state_dict" not in source:
        sys.exit("ERROR: expected a checkpoint dictionary containing state_dict")

    measured = json.loads(args.test_results.read_text(encoding="utf-8"))
    class_names = source.get("class_names") or measured.get("class_names")
    if not class_names:
        sys.exit("ERROR: the source artifacts do not record the class order")

    model = build_model(
        MODEL_NAME, num_classes=len(class_names), **MODEL_KWARGS
    )
    try:
        model.load_state_dict(source["state_dict"], strict=True)
    except RuntimeError as error:
        sys.exit(f"ERROR: checkpoint does not strictly fit {MODEL_NAME}:\n{error}")

    output = PROJECT_ROOT / "runs" / MODEL_NAME / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": MODEL_NAME,
        "model_kwargs": MODEL_KWARGS,
        "state_dict": source["state_dict"],
        "num_classes": len(class_names),
        "class_names": class_names,
        "display_names": [name.replace("_", " ").title() for name in class_names],
        "epoch": int(source.get("epoch", measured.get("best_epoch", 48))),
        "val_macro_f1": float(source.get("val_macro_f1", measured["best_val_macro_f1"])),
        "image_size": 160,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        # DeepV2 was evaluated with a direct 160x160 stretch. Do not silently
        # apply the detection model's centre-crop, multi-view, or mirror TTA.
        "inference_views": ["whole"],
        "tta_flip": False,
        "use_box_crop": False,
    }
    packaged_checkpoint = output / "best_model.pt"
    torch.save(checkpoint, packaged_checkpoint)
    checkpoint_sha256 = hashlib.sha256(packaged_checkpoint.read_bytes()).hexdigest()

    results = {
        "model": "Justin Deep CNN V2",
        "model_name": MODEL_NAME,
        "model_kwargs": MODEL_KWARGS,
        "run_id": args.run_id,
        "classes": class_names,
        "image_size": 160,
        "inference_views": ["whole"],
        "tta_flip": False,
        "use_box_crop": False,
        "source": "external_broad15_classification_run",
        "dataset_source": "IP102 Classification",
        "split_protocol": "official IP102 train/validation/test splits",
        "benchmark_compatible": False,
        "benchmark_note": (
            "Not directly comparable with main's detection_top15 benchmark: "
            "this run uses classification images, official splits, whole-image "
            "stretch preprocessing, and different training settings."
        ),
        "best_epoch": int(measured["best_epoch"]),
        "best_val_macro_f1": float(measured["best_val_macro_f1"]),
        "test": {
            "accuracy": float(measured["accuracy"]),
            "macro_precision": float(measured["macro_precision"]),
            "macro_recall": float(measured["macro_recall"]),
            "macro_f1": float(measured["macro_f1"]),
        },
        "total_parameters": int(measured["parameters"]),
        "checkpoint_size_bytes": packaged_checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "under_trained": False,
        "imported_from": str(args.checkpoint),
        "imported_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote {output.relative_to(PROJECT_ROOT)}/")
    print("[note] external Broad15 classification result; not benchmark-compatible")


if __name__ == "__main__":
    main()
