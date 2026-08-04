"""Score the test split through the chatbot's own inference path.

This is the check that catches the bug class docs/propestnet.md calls #4:
inference that does not preprocess the way training did. It deliberately does
*not* import the notebook's evaluation code -- it goes through
``PestAssistant.predict``, exactly as a farmer's upload does, so a disagreement
between the two paths shows up as a number rather than as a silent accuracy loss.

It reports two conditions:

  boxed    every image cropped to its annotation, as training and the notebook's
           evaluation both did. This is the number that must match the notebook.
  unboxed  no crop, which is what an uploaded photo actually gets. The gap
           between the two is the cost of not knowing where the insect is.

    python scripts/eval_chatbot.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pest_assistant import PestAssistant  # noqa: E402
from ip102_bench.protocol import load_protocol  # noqa: E402


def evaluate(assistant: PestAssistant, records: list, image_dir: Path,
             boxes: dict, use_boxes: bool, tta: bool) -> dict:
    """Top-1 and top-3 hit rate over ``records``, through the app's own path."""
    top1 = top3 = 0
    start = time.time()
    for filename, label in records:
        candidates = assistant.predict(
            image_dir / filename, box=boxes.get(filename) if use_boxes else None, tta=tta
        )
        names = [name for name, _ in candidates]
        truth = assistant.class_names[label]
        top1 += names[:1] == [truth]
        top3 += truth in names
    total = len(records)
    return {
        "images": total,
        "top1": top1 / total,
        "top3": top3 / total,
        "seconds_per_image": (time.time() - start) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=200,
                        help="images to score (0 = the whole split)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-tta", action="store_true", help="single centre view only")
    args = parser.parse_args()

    protocol = load_protocol()
    subset = protocol.subset
    splits = json.loads(Path(subset["splits_json"]).read_text(encoding="utf-8"))
    boxes = json.loads(Path(subset["boxes_json"]).read_text(encoding="utf-8"))
    image_dir = protocol.image_root
    if not image_dir.is_dir():
        sys.exit(f"ERROR: images not found at {image_dir}")

    assistant = PestAssistant()
    if assistant.model is None:
        sys.exit(f"ERROR: no model loaded. {assistant.loaded.reason}")

    records = splits[args.split]
    if args.limit and args.limit < len(records):
        random.seed(args.seed)
        records = random.sample(records, args.limit)

    print(f"model   : {assistant.loaded.path.parent.name} "
          f"({assistant.loaded.image_size}px, {len(assistant.class_names)} classes)")
    if assistant.loaded.under_trained:
        print("warning : this checkpoint is under-trained — read these as plumbing checks, "
              "not as the model's accuracy")
    print(f"split   : {args.split}, {len(records)} images")
    print(f"tta     : {'off' if args.no_tta else 'centre + whole + mirror (4 passes)'}\n")

    header = f"{'condition':<10}{'top-1':>10}{'top-3':>10}{'s/image':>10}"
    print(header)
    print("-" * len(header))
    for label, use_boxes in [("boxed", True), ("unboxed", False)]:
        scores = evaluate(assistant, records, image_dir, boxes, use_boxes, not args.no_tta)
        print(f"{label:<10}{scores['top1']:>9.1%}{scores['top3']:>10.1%}"
              f"{scores['seconds_per_image']:>10.2f}")

    print("\n'boxed' is the condition the notebook reports. A large gap there means the "
          "app preprocesses differently from training; the boxed/unboxed gap is the cost "
          "of an upload having no bounding box.")


if __name__ == "__main__":
    main()
