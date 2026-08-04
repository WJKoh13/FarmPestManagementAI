"""Collect every run into one comparison table.

    python -m ip102_bench.compare
    python -m ip102_bench.compare --csv comparison.csv

Runs recorded under a different ``protocol_version`` or a different dataset
subset are listed separately rather than mixed in, because putting them in one
table would silently compare models that never faced the same task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .protocol import load_protocol, resolve_path

COLUMNS = [
    "model_name", "author", "pretrained", "macro_f1", "test_accuracy",
    "best_val_macro_f1", "parameters", "model_size_mb", "cpu_inference_ms",
    "best_epoch", "epochs_trained", "training_seconds", "run_id",
]


def load_runs(output_root: Path) -> pd.DataFrame:
    """Read every ``results.json`` under ``runs/``. Latest run per model wins."""
    rows = []
    for results_path in sorted(output_root.glob("*/*/results.json")):
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[skip] unreadable: {results_path}")
            continue
        data["_path"] = str(results_path.parent)
        rows.append(data)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    for column in ("protocol_version", "subset", "author", "pretrained"):
        if column not in frame.columns:
            frame[column] = None
    return frame


def build_table(frame: pd.DataFrame, latest_only: bool = True) -> pd.DataFrame:
    if frame.empty:
        return frame
    if latest_only:
        frame = frame.sort_values("run_id").groupby("model_name", as_index=False).last()
    present = [c for c in COLUMNS if c in frame.columns]
    return frame[present].sort_values("macro_f1", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="Also write the table to CSV.")
    parser.add_argument("--all-runs", action="store_true",
                        help="Show every run, not just the latest per model.")
    args = parser.parse_args()

    protocol = load_protocol()
    output_root = resolve_path(protocol.runtime.get("output_root", "runs"))
    frame = load_runs(output_root)

    if frame.empty:
        print(f"No runs found under {output_root}. Train a model first.")
        return

    current = (frame["protocol_version"] == protocol.version) & (frame["subset"] == protocol.subset_name)
    table = build_table(frame[current], latest_only=not args.all_runs)

    print(f"\nProtocol v{protocol.version}, subset '{protocol.subset_name}' "
          f"-- primary metric: macro F1\n")
    if table.empty:
        print("No runs match the current protocol.")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(table.to_string(index=False))

        scratch = table[table["pretrained"] == False]  # noqa: E712 - pandas mask
        if not scratch.empty and (table["pretrained"] == True).any():  # noqa: E712
            best = scratch.iloc[0]
            print(f"\nBest from-scratch model: {best['model_name']} "
                  f"(macro F1 {best['macro_f1']:.4f})")

    stale = frame[~current]
    if not stale.empty:
        print(f"\n{len(stale)} run(s) excluded -- recorded under a different protocol "
              "version or subset:")
        for _, row in stale.iterrows():
            print(f"  {row['model_name']:<24} v{row['protocol_version']} "
                  f"subset={row['subset']}  {row['_path']}")
        print("Re-run these before quoting them alongside the table above.")

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
