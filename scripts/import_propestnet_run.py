"""Turn a raw ProPestNet checkpoint into a run bundle the app can load.

The notebook saves ``torch.save(model.state_dict(), ...)`` -- a bare tensor
dict with nothing in it that says which architecture, how many classes, or what
preprocessing produced it. The app refuses to guess any of that, so this script
records it once, next to the weights:

    runs/propestnet/<run_id>/
      best_model.pt   {model_name, num_classes, class_names, image_size, ...}
      results.json    metrics, subset, and whether the run is trustworthy
      history.csv     copied verbatim when given

It fails rather than writes if the state dict does not load *strictly* into
``build_propestnet`` -- a bundle that loads with missing keys is silently a
different model, which is exactly the failure this file exists to prevent.

Usage:

    python scripts/import_propestnet_run.py \\
        --checkpoint /path/to/best_model.pt \\
        --history    /path/to/history.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ip102_bench.models import build_model  # noqa: E402
from ip102_bench.protocol import load_protocol  # noqa: E402

# Below this validation macro-F1 the run is not the submitted model, and every
# surface that shows a prediction has to say so. docs/propestnet.md reports
# 0.6045 for the finished 60-epoch run.
UNDER_TRAINED_VAL_F1 = 0.55


def read_history(path: Path) -> tuple[list[dict], dict]:
    """Rows plus the best epoch by validation macro-F1."""
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        sys.exit(f"ERROR: {path} has no rows.")
    best = max(rows, key=lambda row: float(row.get("val_macro_f1", 0) or 0))
    return rows, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="best_model.pt written by the notebook (a bare state_dict)")
    parser.add_argument("--history", type=Path,
                        help="history.csv from the same run; supplies the metrics")
    parser.add_argument("--notebook-results", type=Path,
                        help="results.json from the same run. Carries over the test "
                             "metrics and the section-13 prior correction, which this "
                             "script cannot measure for itself")
    parser.add_argument("--run-id", help="output folder name (default: today + epochs)")
    parser.add_argument("--model-name", default="propestnet",
                        help="registry key to rebuild the architecture with")
    parser.add_argument("--author", default="Wen Jun",
                        help="whose run this is; shown beside the model in the picker")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")

    protocol = load_protocol()
    class_names = protocol.class_names
    display_names = protocol.display_names
    preprocessing = protocol.subset.get("preprocessing") or {}
    image_size = int(preprocessing.get("image_size", 128))
    mean = list(preprocessing.get("mean", [0.485, 0.456, 0.406]))
    std = list(preprocessing.get("std", [0.229, 0.224, 0.225]))

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict) or "state_dict" in state_dict:
        state_dict = state_dict.get("state_dict", state_dict)

    # Strict or nothing. A bundle that loads with missing keys is a different
    # model wearing this one's name.
    model = build_model(args.model_name, num_classes=len(class_names))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        sys.exit(f"ERROR: {args.checkpoint} does not load into "
                 f"{args.model_name}({len(class_names)} classes):\n\n{error}")

    parameters = sum(p.numel() for p in model.parameters())
    print(f"[ok] loaded strictly into {args.model_name}: "
          f"{parameters:,} parameters, {len(class_names)} classes")

    history_rows: list[dict] = []
    best_row: dict = {}
    if args.history and args.history.is_file():
        history_rows, best_row = read_history(args.history)

    best_val_f1 = float(best_row.get("val_macro_f1", 0) or 0)
    epochs_run = len(history_rows)
    under_trained = best_val_f1 < UNDER_TRAINED_VAL_F1

    # Only what the notebook actually measured. Absent file -> absent numbers.
    notebook_results: dict = {}
    if args.notebook_results and args.notebook_results.is_file():
        notebook_results = json.loads(args.notebook_results.read_text(encoding="utf-8"))

    adjustment = notebook_results.get("logit_adjustment") or {}
    prior = list(adjustment.get("train_class_prior") or [])
    tau = float(adjustment.get("tau", 0.0))
    if prior and len(prior) != len(class_names):
        sys.exit(f"ERROR: {args.notebook_results} records a {len(prior)}-class prior, "
                 f"but this subset has {len(class_names)} classes.")

    run_id = args.run_id or (
        f"{datetime.now():%Y%m%d}_epoch{epochs_run}" if epochs_run else f"{datetime.now():%Y%m%d}"
    )
    run_dir = PROJECT_ROOT / "runs" / args.model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": args.model_name,
            "num_classes": len(class_names),
            "class_names": class_names,
            "display_names": display_names,
            "image_size": image_size,
            "mean": mean,
            "std": std,
            # Travels with the weights, so a bundle copied to another machine
            # keeps the correction the notebook selected for it.
            "logit_adjust_tau": tau,
            "train_class_prior": prior,
            "state_dict": state_dict,
        },
        run_dir / "best_model.pt",
    )

    results = {
        "model": "ProPestNet",
        "model_name": args.model_name,
        # Whose run this is. The model picker shows it beside the architecture,
        # because four people's models in one dropdown is a question of "whose"
        # at least as often as "which".
        "author": args.author,
        "subset": protocol.subset_name,
        "classes": class_names,
        "image_size": image_size,
        "total_parameters": parameters,
        "epochs_run": epochs_run,
        "best_epoch": int(best_row.get("epoch", 0) or 0),
        "best_val_macro_f1": best_val_f1,
        # No test split is scored by this import -- only the notebook does that.
        # Carried over when --notebook-results names the file the notebook
        # wrote, and left null otherwise, so nothing here can be mistaken for a
        # measurement this bundle actually made.
        "test": notebook_results.get("test"),
        "under_trained": under_trained,
        "imported_from": str(args.checkpoint),
        "imported_at": datetime.now().isoformat(timespec="seconds"),
    }
    for section in ("test_with_tta", "test_with_tta_and_prior", "logit_adjustment",
                    "tta_validation_sweep"):
        if notebook_results.get(section) is not None:
            results[section] = notebook_results[section]

    if under_trained:
        results["under_trained_note"] = (
            f"Best validation macro-F1 {best_val_f1:.4f} over {epochs_run} epochs. "
            "The submitted model reaches 0.6045 over 60 epochs (docs/propestnet.md). "
            "Predictions from this bundle are for testing the app, not for advice."
        )

    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.history and args.history.is_file():
        shutil.copy2(args.history, run_dir / "history.csv")

    print(f"[ok] wrote {run_dir.relative_to(PROJECT_ROOT)}/")
    if under_trained:
        print(f"[!!] UNDER-TRAINED: best val macro-F1 {best_val_f1:.4f} over "
              f"{epochs_run} epochs. The app will label it as such.")


if __name__ == "__main__":
    main()
