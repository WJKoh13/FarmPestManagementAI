"""Copy a few held-out test photos per class into sample_images/, for trying the chatbot.

Every image comes from the **test** split, so none of them was seen during
training and what the chatbot says about them is a fair check. Files are named
``<class>__<original>.jpg``, so the correct answer is in the filename and you
can tell at a glance whether a reply is right.

    python scripts/make_sample_images.py --per-class 2

The folder is git-ignored: these are IP102 images, and the dataset does not
belong in the repository.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ip102_bench.protocol import load_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "sample_images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    protocol = load_protocol()
    splits = json.loads(Path(protocol.subset["splits_json"]).read_text(encoding="utf-8"))
    class_names = protocol.class_names
    image_dir = protocol.image_root
    if not image_dir.is_dir():
        sys.exit(f"ERROR: images not found at {image_dir}")

    by_class: dict[int, list[str]] = {}
    for filename, label in splits[args.split]:
        by_class.setdefault(label, []).append(filename)

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    random.seed(args.seed)
    copied = 0
    for label, name in enumerate(class_names):
        available = by_class.get(label, [])
        chosen = random.sample(available, min(args.per_class, len(available)))
        for filename in chosen:
            source = image_dir / filename
            if not source.is_file():
                continue
            shutil.copy2(source, args.out / f"{name}__{filename}")
            copied += 1
        if len(chosen) < args.per_class:
            print(f"[!!] {name}: only {len(chosen)} image(s) available in {args.split}")

    print(f"[ok] copied {copied} images from the {args.split} split into "
          f"{args.out.relative_to(PROJECT_ROOT)}/")
    print("     Filenames carry the true class, so you can check each reply against it.")


if __name__ == "__main__":
    main()
