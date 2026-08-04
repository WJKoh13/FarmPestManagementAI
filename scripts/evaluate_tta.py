"""Score an existing run bundle the way the app will actually serve it.

The app ranks runs by the best macro F1 they can evidence, preferring the most
corrected setting a bundle records -- prior-adjusted, then TTA, then single pass
(see ``app.cnn_model._score``). That is the right rule, because it ranks each
model by the setting it will be served under. It only works if the bundles
actually carry those numbers.

They usually do not. ``ip102_bench.save_run`` records a single-pass test score,
because that is what the harness measures; only ProPestNet's bundle carries a
TTA sweep, and it carries one because its notebook did the work by hand. Left
alone, a model that never recorded TTA is ranked on its single pass against
another model's TTA-corrected number -- a comparison that rewards bookkeeping
rather than accuracy.

This script closes that gap. It re-scores any bundle on the test split through
the app's own TTA views and writes ``test_with_tta`` back into its results.json,
so automatic selection compares like with like.

Everything comes from the checkpoint: image size, normalization, crop margin. A
bundle is scored the way it was trained, never the way the protocol assumes.

Usage:

    python scripts/evaluate_tta.py runs/custom_cnn_ziyang/<run_id>
    python scripts/evaluate_tta.py runs/<model>/<run_id> --device cpu --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.cnn_model import load_best_model  # noqa: E402
from app.propest_inference import TTA_FLIP, TTA_VIEWS, build_views, crop_to_box  # noqa: E402
from ip102_bench.data import BOX_COLUMNS  # noqa: E402
from ip102_bench.metrics import compute_metrics  # noqa: E402
from ip102_bench.protocol import load_protocol  # noqa: E402


def score_split(loaded, protocol, *, device: str, limit: int | None = None) -> dict:
    """Run the app's TTA over the test manifest and return its metrics.

    Reads the manifest directly rather than going through a DataLoader: TTA runs
    several differently-framed views of the same image, so the transform has to
    be applied per view, after cropping, on the PIL image itself -- which is
    exactly what the app does for a farmer's upload.
    """
    import pandas as pd

    frame = pd.read_csv(protocol.manifest("test"))
    if limit:
        frame = frame.head(limit)

    views = build_views(loaded.image_size, loaded.mean, loaded.std)
    model = loaded.model.to(device).eval()
    has_boxes = all(column in frame.columns for column in BOX_COLUMNS)

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for position, row in enumerate(frame.itertuples(index=False), start=1):
            path = protocol.image_root / row.image_path
            with Image.open(path) as handle:
                image = handle.convert("RGB")

            # The offline split has boxes; a farmer's upload does not. Scoring
            # with them is right here -- it is the same input the single-pass
            # number was measured on, so the two remain comparable.
            box = None
            if has_boxes and loaded.crop_mode == "box":
                box = [getattr(row, column) for column in BOX_COLUMNS]
                if any(value != value for value in box):  # NaN: no annotated box
                    box = None
            image = crop_to_box(image, box, margin=loaded.crop_margin)

            probabilities = None
            for name in TTA_VIEWS:
                batch = views[name](image).unsqueeze(0).to(device)
                passes = [batch] + ([torch.flip(batch, dims=[3])] if TTA_FLIP else [])
                for pass_batch in passes:
                    softmax = torch.softmax(model(pass_batch), dim=1)[0].cpu()
                    probabilities = softmax if probabilities is None else probabilities + softmax

            y_pred.append(int(probabilities.argmax()))
            y_true.append(int(row.project_label))

            if position % 200 == 0:
                print(f"    scored {position}/{len(frame)} images", flush=True)

    return compute_metrics(y_true, y_pred, protocol.num_classes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="runs/<model>/<run_id>")
    parser.add_argument("--device", default="cpu", help="device for the passes (default: cpu)")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N images (a smoke test, not a result)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the metrics without writing results.json")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    checkpoint = run_dir / "best_model.pt"
    results_path = run_dir / "results.json"
    if not checkpoint.is_file() or not results_path.is_file():
        sys.exit(f"ERROR: {run_dir} needs both best_model.pt and results.json.")

    protocol = load_protocol()
    # Loading by explicit path, so a run excluded from automatic selection can
    # still be scored.
    loaded = load_best_model(device=args.device, model_path=checkpoint)
    if loaded.model is None:
        sys.exit(f"ERROR: cannot load {checkpoint}: {loaded.reason}")

    print(f"[ok] {checkpoint.parent.name}: {loaded.image_size}px, crop margin "
          f"{loaded.crop_margin}, {len(loaded.class_names)} classes")
    print(f"[..] scoring the test split with TTA ({' + '.join(TTA_VIEWS)}"
          f"{' + mirror' if TTA_FLIP else ''})", flush=True)

    metrics = score_split(loaded, protocol, device=args.device, limit=args.limit)
    block = {
        "setting": f"{' + '.join(TTA_VIEWS)}{' + mirror' if TTA_FLIP else ''}",
        "accuracy": round(metrics["accuracy"], 6),
        "macro_precision": round(metrics["macro_precision"], 6),
        "macro_recall": round(metrics["macro_recall"], 6),
        "macro_f1": round(metrics["macro_f1"], 6),
        "per_class_f1": [round(value, 6) for value in metrics["per_class"]["f1"]],
        "measured_by": "scripts/evaluate_tta.py",
    }

    results = json.loads(results_path.read_text(encoding="utf-8"))
    single = results.get("macro_f1") or (results.get("test") or {}).get("macro_f1")
    print(f"\n    single pass : {single if single is not None else 'not recorded'}")
    print(f"    with TTA    : {block['macro_f1']:.6f}   accuracy {block['accuracy']:.4f}")

    if args.limit:
        print(f"\n[!!] --limit {args.limit} was set: this is a smoke test, not a result. "
              f"Not writing results.json.")
        return
    if args.dry_run:
        print("\n[--] --dry-run: results.json left untouched.")
        return

    results["test_with_tta"] = block
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ok] wrote test_with_tta into {results_path}")


if __name__ == "__main__":
    main()
