"""Create a reproducible error-analysis report from evaluation artifacts.

Example:
    python -m src.analyze_errors --run runs/broad15/justin_deep_v2/<run_id>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.config import resolve_path


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--top-confusions", type=int, default=10)
    parser.add_argument("--examples-per-confusion", type=int, default=3)
    args = parser.parse_args()

    run_dir = resolve_path(args.run)
    results_path = run_dir / f"{args.split}_results.json"
    predictions_path = run_dir / f"{args.split}_predictions.csv"
    if not results_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(
            f"Run evaluation first; expected {results_path.name} and {predictions_path.name}."
        )

    results = json.loads(results_path.read_text(encoding="utf-8"))
    class_names = results["class_names"]
    confusion = results["confusion_matrix"]
    per_class = results["per_class"]

    ranked_classes = sorted(per_class.items(), key=lambda item: item[1]["f1"])
    ranked_confusions = sorted(
        (
            (count, class_names[true], class_names[pred])
            for true, row in enumerate(confusion)
            for pred, count in enumerate(row)
            if true != pred and count > 0
        ),
        reverse=True,
    )[: args.top_confusions]

    examples: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    wanted_pairs = {(true, pred) for _, true, pred in ranked_confusions}
    with predictions_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            pair = (row["true_class"], row["predicted_class"])
            if pair in wanted_pairs:
                examples[pair].append(row)
    for pair in examples:
        examples[pair].sort(key=lambda row: float(row["confidence"]), reverse=True)

    lines = [
        f"# {args.split.title()} error analysis",
        "",
        f"- Accuracy: **{percentage(results['accuracy'])}**",
        f"- Macro-F1: **{percentage(results['macro_f1'])}**",
        f"- Selected checkpoint epoch: **{results['best_epoch']}**",
        "",
        "## Per-class performance (weakest first)",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, scores in ranked_classes:
        lines.append(
            f"| {name} | {percentage(scores['precision'])} | "
            f"{percentage(scores['recall'])} | {percentage(scores['f1'])} | "
            f"{scores['support']} |"
        )

    lines.extend(
        [
            "",
            "## Largest off-diagonal confusions",
            "",
            "| Actual class | Predicted class | Images |",
            "|---|---|---:|",
        ]
    )
    for count, true_name, predicted_name in ranked_confusions:
        lines.append(f"| {true_name} | {predicted_name} | {count} |")

    lines.extend(["", "## High-confidence examples to inspect", ""])
    for count, true_name, predicted_name in ranked_confusions:
        pair = (true_name, predicted_name)
        lines.append(f"### {true_name} predicted as {predicted_name} ({count} images)")
        lines.append("")
        selected = examples[pair][: args.examples_per_confusion]
        if not selected:
            lines.append("No prediction rows were available.")
        else:
            for row in selected:
                lines.append(
                    f"- `{row['image_path']}` — confidence {float(row['confidence']):.4f}"
                )
        lines.append("")

    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- Inspect the listed source images before attributing an error to model capacity.",
            "- Record small/occluded pests, ambiguous life stages, composites, and suspected label noise.",
            "- Use validation errors for diagnosis only; do not tune repeatedly against test results.",
            "",
        ]
    )

    output_path = run_dir / f"{args.split}_error_analysis.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
