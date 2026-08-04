"""Freeze the validation winner, evaluate it once, and write a final report.

This command is deliberately separate from training. It compares the clean run
with the preserved warm-restart validation score, freezes exactly one model,
runs validation diagnostics, then touches the official test split once.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.config import load_config, resolve_path
from src.train import atomic_torch_save


WARM_RESTART_VAL_MACRO_F1 = 0.6247677361828722


def run_module(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", *arguments],
        cwd=resolve_path("."),
        check=True,
    )


def adapt_warm_restart_checkpoint(
    source_checkpoint: Path,
    final_run: Path,
    device: str,
) -> None:
    raw = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    standard = {
        "model": "justin_deep_v2",
        "model_kwargs": {
            "classifier_dropout": 0.20,
            "stage_dropouts": [0.00, 0.00, 0.05, 0.10],
        },
        "num_classes": 15,
        "state_dict": raw["model_state_dict"],
        "epoch": int(raw["epoch"]),
        "val_macro_f1": float(raw["val_macro_f1"]),
        "class_names": raw["class_names"],
        "seed": 42,
    }
    atomic_torch_save(standard, final_run / "best_model.pt")

    config = load_config(
        "configs/broad15_deep_v2.yaml",
        {"device": device, "num_workers": 0},
    )
    config = {key: value for key, value in config.items() if not key.startswith("_")}
    config["run_id"] = final_run.name
    config["device_used"] = device
    (final_run / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def optional_percentage(value: float | None) -> str:
    return "not completed" if value is None else percentage(value)


def write_final_report(
    final_run: Path,
    selection: dict,
    validation: dict,
    test: dict,
) -> None:
    class_names = validation["class_names"]
    lines = [
        "# Broad15 final Deep V2 result",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Frozen model selection",
        "",
        f"- Selected source: **{selection['selected_source']}**",
        f"- Selected checkpoint epoch: **{validation['best_epoch']}**",
        f"- Clean-run best validation macro-F1: **{optional_percentage(selection['clean_val_macro_f1'])}**",
        f"- Warm-restart best validation macro-F1: **{percentage(selection['warm_restart_val_macro_f1'])}**",
        "- Selection rule: highest validation macro-F1; the test split was not used for selection.",
        "",
        "## Headline metrics",
        "",
        "| Split | Accuracy | Macro precision | Macro recall | Macro-F1 |",
        "|---|---:|---:|---:|---:|",
        f"| Validation | {percentage(validation['accuracy'])} | "
        f"{percentage(validation['macro_precision'])} | {percentage(validation['macro_recall'])} | "
        f"**{percentage(validation['macro_f1'])}** |",
        f"| Test | {percentage(test['accuracy'])} | {percentage(test['macro_precision'])} | "
        f"{percentage(test['macro_recall'])} | **{percentage(test['macro_f1'])}** |",
        "",
        "## Per-class results",
        "",
        "| Class | Validation F1 | Test precision | Test recall | Test F1 | Test support |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in class_names:
        validation_scores = validation["per_class"][name]
        test_scores = test["per_class"][name]
        lines.append(
            f"| {name} | {percentage(validation_scores['f1'])} | "
            f"{percentage(test_scores['precision'])} | {percentage(test_scores['recall'])} | "
            f"{percentage(test_scores['f1'])} | {test_scores['support']} |"
        )

    weakest_validation = sorted(
        validation["per_class"].items(), key=lambda item: item[1]["f1"]
    )[:3]
    weakest_test = sorted(test["per_class"].items(), key=lambda item: item[1]["f1"])[:3]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Weakest validation classes: "
            + ", ".join(f"`{name}` ({percentage(scores['f1'])})" for name, scores in weakest_validation)
            + ".",
            "- Weakest test classes: "
            + ", ".join(f"`{name}` ({percentage(scores['f1'])})" for name, scores in weakest_test)
            + ".",
            "- Use the validation error report for any future diagnosis; do not tune against test errors.",
            "- This is one seed. It does not estimate run-to-run variance.",
            "- If the warm restart wins, report it as a two-phase optimization run rather than an uninterrupted run.",
            "",
            "## Artifacts",
            "",
            "- `best_model.pt`: frozen selected checkpoint",
            "- `selection.json`: validation-only selection provenance",
            "- `validation_results.json` and `validation_error_analysis.md`",
            "- `test_results.json`, `test_predictions.csv`, and `test_confusion_matrix.png`",
            "",
        ]
    )
    (final_run / "FINAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-run", default=None)
    parser.add_argument("--warm-checkpoint", required=True)
    parser.add_argument("--final-run", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--select-warm",
        action="store_true",
        help="Freeze the warm-restart candidate without waiting for a clean rerun.",
    )
    args = parser.parse_args()

    clean_run = resolve_path(args.clean_run) if args.clean_run else None
    warm_checkpoint = resolve_path(args.warm_checkpoint)
    final_run = resolve_path(args.final_run)
    if not warm_checkpoint.is_file():
        raise FileNotFoundError(f"Warm-restart checkpoint not found: {warm_checkpoint}")
    if final_run.exists():
        raise FileExistsError(f"Refusing to replace frozen run directory: {final_run}")

    clean_f1 = None
    if not args.select_warm:
        if clean_run is None:
            raise ValueError("Provide --clean-run or explicitly pass --select-warm.")
        clean_summary_path = clean_run / "train_summary.json"
        if not clean_summary_path.is_file():
            raise FileNotFoundError(f"Clean run is not complete: {clean_summary_path}")
        clean_summary = json.loads(clean_summary_path.read_text(encoding="utf-8"))
        clean_f1 = float(clean_summary["best_val_macro_f1"])
    clean_wins = clean_f1 is not None and clean_f1 >= WARM_RESTART_VAL_MACRO_F1

    if clean_wins:
        assert clean_run is not None
        shutil.copytree(clean_run, final_run)
        selected_source = "clean_seed42"
    else:
        final_run.mkdir(parents=True)
        adapt_warm_restart_checkpoint(warm_checkpoint, final_run, args.device)
        selected_source = "warm_restart_epoch48"

    selection = {
        "selected_source": selected_source,
        "selection_metric": "validation_macro_f1",
        "clean_val_macro_f1": clean_f1,
        "warm_restart_val_macro_f1": WARM_RESTART_VAL_MACRO_F1,
        "test_used_for_selection": False,
    }
    (final_run / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    relative_final_run = str(final_run.relative_to(resolve_path(".")))
    run_module(
        "src.evaluate", "--run", relative_final_run,
        "--split", "validation", "--device", args.device,
    )
    run_module(
        "src.analyze_errors", "--run", relative_final_run,
        "--split", "validation",
    )
    # This is intentionally last: the final model is already frozen and the
    # evaluation guard prevents an accidental second test run.
    run_module(
        "src.evaluate", "--run", relative_final_run,
        "--split", "test", "--device", args.device,
    )

    validation = json.loads(
        (final_run / "validation_results.json").read_text(encoding="utf-8")
    )
    test = json.loads((final_run / "test_results.json").read_text(encoding="utf-8"))
    write_final_report(final_run, selection, validation, test)
    print(f"Finalized Broad15 model and report at {final_run}")


if __name__ == "__main__":
    main()
