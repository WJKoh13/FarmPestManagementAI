"""Embed docs/knowledge/ and write the index the app searches.

Run it after editing anything in docs/knowledge/:

    python scripts/build_knowledge_index.py

The output, data_manifests/knowledge_index.json, is committed. That is
deliberate and matches how classes_top15.json is handled: a fresh clone then has
working semantic search without Ollama, without the embedding model, and without
a build step. Rebuilding is only needed when the corpus changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.knowledge import INDEX_PATH, KNOWLEDGE_DIR, build_index, load_passages  # noqa: E402
from app.ollama_client import EMBED_MODEL, OllamaClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Report whether the committed index matches the corpus, and write nothing.")
    arguments = parser.parse_args()

    passages = load_passages()
    print(f"corpus     : {len(list(KNOWLEDGE_DIR.glob('*.md'))) - 1} documents, "
          f"{len(passages)} passages")

    if arguments.check:
        from app.knowledge import load_index

        index = load_index()
        titles_on_disk = [p.title for p in index.passages]
        titles_now = [p.title for p in passages]
        if titles_on_disk != titles_now:
            print("STALE: the committed index does not match docs/knowledge/.")
            print("       Run scripts/build_knowledge_index.py to rebuild it.")
            return 1
        if not index.embedded:
            print("PLAIN: the index has passages but no embeddings — search will "
                  "fall back to keywords.")
            return 1
        print(f"index      : up to date, {len(index)} passages, {index.embed_model}")
        return 0

    client = OllamaClient()
    if not client.embeddings_available():
        print(f"\nThe embedding model {EMBED_MODEL!r} is not pulled, so the index would "
              f"have no vectors.\nRun:  ollama pull {EMBED_MODEL}")
        return 1

    print(f"embedding  : {EMBED_MODEL} (this takes a moment)")
    index = build_index(client)
    if not index.embedded:
        print("Embedding failed; nothing written.")
        return 1

    path = index.save()
    size_kb = path.stat().st_size / 1024
    print(f"wrote      : {path.relative_to(PROJECT_ROOT)} "
          f"({len(index)} passages, {len(index.vectors[0])} dimensions, {size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
