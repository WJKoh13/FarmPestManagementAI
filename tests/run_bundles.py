"""Build run bundles in a temporary directory.

Tests that exercise model loading used to read whatever happened to be in the
git-ignored ``runs/`` directory, so the same test passed on one machine and
failed on another depending on which checkpoints a developer had imported. That
is not a property a suite can assert on.

Everything here writes into a ``tmp_path``, so a test states the bundle it needs
and gets exactly that. Nothing reads or writes the repository's real ``runs/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ip102_bench.models import build_model
from ip102_bench.protocol import load_protocol


def write_run_bundle(
    runs_dir: Path,
    *,
    # custom_cnn_ziyang is the default only because it is the smallest
    # architecture the registry actually builds (1.4M parameters against
    # ProPestNet's 11M), which keeps these fixtures cheap to write and delete.
    # Nothing here depends on which architecture it is.
    model_name: str = "custom_cnn_ziyang",
    run_id: str = "test_run",
    num_classes: int = 15,
    image_size: int = 128,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    crop_mode: str | None = None,
    crop_margin: float | None = None,
    results_extra: dict | None = None,
    class_names: list[str] | None = None,
) -> Path:
    """Write ``runs_dir/<model_name>/<run_id>/`` and return the run directory.

    ``crop_mode`` and ``crop_margin`` are omitted from the checkpoint unless
    given, which is how a test reproduces an older bundle that predates that
    metadata and must still fall back to the protocol's 0.25.
    """
    protocol = load_protocol()
    names = class_names if class_names is not None else protocol.class_names[:num_classes]

    run_dir = runs_dir / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_name, num_classes=num_classes)
    checkpoint = {
        "model_name": model_name,
        "num_classes": num_classes,
        "class_names": list(names),
        "display_names": [n.replace("_", " ").capitalize() for n in names],
        "image_size": image_size,
        "mean": list(mean if mean is not None else [0.485, 0.456, 0.406]),
        "std": list(std if std is not None else [0.229, 0.224, 0.225]),
        "state_dict": model.state_dict(),
    }
    # Only present when the caller asks for them, so "an old bundle" is a real
    # absence rather than a value that happens to match the default.
    if crop_mode is not None:
        checkpoint["crop_mode"] = crop_mode
    if crop_margin is not None:
        checkpoint["crop_margin"] = crop_margin
    torch.save(checkpoint, run_dir / "best_model.pt")

    results = {
        "model_name": model_name,
        "model": model_name,
        "classes": list(names),
        "image_size": image_size,
        "best_val_macro_f1": 0.5,
    }
    results.update(results_extra or {})
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    return run_dir


def write_legacy_custom_cnn_bundle(runs_dir: Path, *, run_id: str = "legacy_run",
                                   best_val_macro_f1: float = 0.6602782585720635) -> Path:
    """A bundle shaped exactly like scripts/import_custom_cnn_run.py's output."""
    return write_run_bundle(
        runs_dir,
        model_name="custom_cnn_ziyang",
        run_id=run_id,
        num_classes=15,
        image_size=160,
        crop_mode="box",
        crop_margin=0.15,
        results_extra={
            "external": True,
            "pretrained": True,
            "comparable_to_main": False,
            "eligible_for_automatic_selection": False,
            "protocol_version": "legacy-det_top15-external",
            "source_scope": "det_top15",
            "best_val_macro_f1": best_val_macro_f1,
            "best_epoch": 48,
            "test": None,
            "not_comparable_note": "Trained outside this repository on det_top15.",
        },
    )
