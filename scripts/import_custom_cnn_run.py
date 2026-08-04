"""Turn a legacy custom_cnn checkpoint into a run bundle the app can load.

The checkpoint comes from an experimental branch that trained under its own
`det_top15` scope, not under this repository's protocol. Its payload stores
weights under ``model_state`` (alongside optimizer and RNG state), and nothing
in it says which architecture, how many classes, or what preprocessing produced
it in a form the app understands. This script records that once, next to the
weights:

    runs/custom_cnn_ziyang/<run_id>/
      best_model.pt   {model_name, num_classes, class_names, image_size, ...}
      results.json    metrics, provenance, and why it is not comparable

Two things it deliberately does NOT do:

* It does not evaluate the test split. Only ``save_run`` in a notebook does that.
* It does not fall back to this repository's preprocessing. The legacy model was
  trained on 0.15-margin box crops with ImageNet normalization; serving it
  through the protocol's 0.25-margin, repo-normalized pipeline would quietly
  degrade every prediction. The bundle records the legacy values verbatim.

The written ``results.json`` carries a legacy ``protocol_version``, so
``ip102_bench.compare`` filters it out of the protocol-v1 table rather than
ranking it against models that actually trained under that protocol.

It fails rather than writes if the state dict does not load *strictly* into
``custom_cnn_ziyang`` -- a bundle that loads with missing keys is silently a
different model, which is exactly the failure this file exists to prevent.

``runs/`` is git-ignored; nothing here copies the checkpoint into version control.

Usage:

    python scripts/import_custom_cnn_run.py --checkpoint /path/to/best.pt
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

from ip102_bench.models import build_model  # noqa: E402
from ip102_bench.protocol import load_protocol  # noqa: E402

# The registry key the app rebuilds the architecture from.
MODEL_NAME = "custom_cnn_ziyang"

# The scope the legacy run trained on. Its class list and ordering match this
# repository's detection_top15 subset, which is why the import is meaningful at
# all -- but the preprocessing and training settings do not match the protocol.
SOURCE_SCOPE = "det_top15"

# Preprocessing the legacy run actually used. Recorded verbatim, never replaced
# with the protocol's values.
LEGACY_IMAGE_SIZE = 160
LEGACY_MEAN = [0.485, 0.456, 0.406]        # ImageNet, not this repo's norm_stats.json
LEGACY_STD = [0.229, 0.224, 0.225]
LEGACY_CROP_MODE = "box"
LEGACY_CROP_MARGIN = 0.15                  # the protocol uses 0.25

# Marks the bundle as belonging to a different experimental regime, so
# ip102_bench.compare cannot put it in the same table as protocol_version 1.
LEGACY_PROTOCOL_VERSION = "legacy-det_top15-external"


def load_legacy_state(checkpoint_path: Path) -> tuple[dict, dict]:
    """Read the legacy payload and return its weights plus whatever metadata it carries.

    ``weights_only=False`` because the payload is a full training checkpoint --
    optimizer, scheduler, scaler and RNG state alongside the weights -- not a
    bare tensor dict. The file is one the user points at explicitly.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        sys.exit(f"ERROR: {checkpoint_path} is not a checkpoint dictionary.")

    if "model_state" not in payload:
        sys.exit(
            f"ERROR: {checkpoint_path} has no 'model_state' key; found "
            f"{sorted(payload)}. This importer reads the legacy format, which "
            f"stores weights under 'model_state' rather than 'state_dict'."
        )

    state_dict = payload["model_state"]
    if not isinstance(state_dict, dict) or not state_dict:
        sys.exit(f"ERROR: {checkpoint_path} has an empty or malformed 'model_state'.")

    return state_dict, payload.get("metadata") or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="legacy best.pt; weights live under its 'model_state' key")
    parser.add_argument("--run-id", help="output folder name (default: today + best epoch)")
    parser.add_argument("--best-val-macro-f1", type=float, default=None,
                        help="validation macro-F1 the legacy run reported "
                             "(default: read from the checkpoint, else unset)")
    parser.add_argument("--best-epoch", type=int, default=None,
                        help="epoch the weights come from (default: read from the checkpoint)")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")

    protocol = load_protocol()
    class_names = protocol.class_names
    display_names = protocol.display_names

    state_dict, metadata = load_legacy_state(args.checkpoint)

    # The class count is the checkpoint's own, not the protocol's. If they ever
    # disagree the import must fail loudly rather than reinterpret a 15-way head
    # under some other class list.
    num_classes = int(metadata.get("num_classes") or len(class_names))
    if len(class_names) != num_classes:
        sys.exit(
            f"ERROR: the checkpoint is {num_classes}-class but protocol.yaml's "
            f"subset '{protocol.subset_name}' defines {len(class_names)} classes. "
            f"Refusing to relabel one class list with another."
        )

    # Strict or nothing. A bundle that loads with missing, unexpected or
    # mis-shaped keys is a different model wearing this one's name.
    model = build_model(MODEL_NAME, num_classes=num_classes)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        sys.exit(f"ERROR: {args.checkpoint} does not load strictly into "
                 f"{MODEL_NAME}({num_classes} classes):\n\n{error}")

    parameters = sum(p.numel() for p in model.parameters())
    print(f"[ok] loaded strictly into {MODEL_NAME}: "
          f"{parameters:,} parameters, {num_classes} classes")

    best_epoch = args.best_epoch if args.best_epoch is not None else int(metadata.get("epoch", 0) or 0)
    if args.best_val_macro_f1 is not None:
        best_val_macro_f1 = args.best_val_macro_f1
    else:
        reported = (metadata.get("metrics") or {}).get("macro_f1", metadata.get("best_metric"))
        best_val_macro_f1 = float(reported) if reported is not None else None

    run_id = args.run_id or (
        f"{datetime.now():%Y%m%d}_legacy_epoch{best_epoch}" if best_epoch
        else f"{datetime.now():%Y%m%d}_legacy"
    )
    run_dir = PROJECT_ROOT / "runs" / MODEL_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": MODEL_NAME,
            "num_classes": num_classes,
            "class_names": class_names,
            "display_names": display_names,
            "image_size": LEGACY_IMAGE_SIZE,
            "mean": LEGACY_MEAN,
            "std": LEGACY_STD,
            "crop_mode": LEGACY_CROP_MODE,
            "crop_margin": LEGACY_CROP_MARGIN,
            "state_dict": state_dict,
        },
        run_dir / "best_model.pt",
    )

    results = {
        "model": "custom_cnn (legacy det_top15 import)",
        "model_name": MODEL_NAME,
        "author": "Zi Yang",

        # The fields that keep this out of the benchmark table.
        "external": True,
        "pretrained": True,          # every weight started from another training run
        "comparable_to_main": False,
        "protocol_version": LEGACY_PROTOCOL_VERSION,

        # ...and the field that keeps it from becoming the app's default. Its
        # validation score was measured under a different protocol, so letting
        # automatic discovery rank it against official runs would put a
        # deliberately non-comparable number in the top slot. Still loadable by
        # explicit path -- see app.cnn_model.load_best_model(model_path=...).
        "eligible_for_automatic_selection": False,

        "source_scope": SOURCE_SCOPE,
        "subset": protocol.subset_name,
        "classes": class_names,
        "num_classes": num_classes,
        "total_parameters": parameters,
        "parameters": parameters,

        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
        # Never scored by this import, and not copied from a report -- nothing
        # here may be mistaken for a measurement this bundle actually made.
        "test": None,

        "preprocessing": {
            "image_size": LEGACY_IMAGE_SIZE,
            "mean": LEGACY_MEAN,
            "std": LEGACY_STD,
            "crop_mode": LEGACY_CROP_MODE,
            "crop_margin": LEGACY_CROP_MARGIN,
            "normalization": "imagenet",
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.0015,
            "weight_decay": 0.05,
            "scheduler": "cosine",
            "warmup_epochs": 5,
            "batch_size": 64,
            "epochs": 60,
            "label_smoothing": 0.1,
            "class_weighting": "none",
            "seed": 1337,
        },
        "not_comparable_note": (
            "Trained outside this repository on the det_top15 scope with a 0.15 box "
            "margin, ImageNet normalization, unweighted loss with label smoothing and "
            "a cosine schedule. The locked protocol uses a 0.25 margin, this "
            "repository's norm_stats.json and inverse-frequency loss weighting, so the "
            f"reported validation macro-F1 is exploratory and must NOT be compared "
            f"with protocol v{protocol.version} runs. Retrain via "
            "notebooks/custom_cnn_ziyang.ipynb for a comparable score."
        ),
        "imported_from": str(args.checkpoint),
        "imported_at": datetime.now().isoformat(timespec="seconds"),
    }

    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[ok] wrote {run_dir.relative_to(PROJECT_ROOT)}/")
    if best_val_macro_f1 is not None:
        print(f"[!!] EXTERNAL RUN: validation macro-F1 {best_val_macro_f1:.4f} was measured "
              f"under the legacy det_top15 regime, NOT protocol v{protocol.version}.")
    print("[!!] Not comparable with benchmark runs; excluded from compare.py's table.")
    print("[!!] Excluded from the app's automatic model selection. To serve it, pass "
          "its best_model.pt explicitly as model_path.")
    print(f"[!!] Preprocessing is {LEGACY_IMAGE_SIZE}px, box crop margin "
          f"{LEGACY_CROP_MARGIN} (the protocol uses {protocol.crop_margin}). If the app "
          f"serves an image with no annotated box, that is distribution shift: this "
          f"model only ever saw box crops.")


if __name__ == "__main__":
    main()
