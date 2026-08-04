"""The reference library: what it retrieves, and what it must never contain.

None of these need Ollama. The committed index carries its own vectors, and the
keyword path needs nothing at all — so a bare clone runs the whole file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.knowledge import (KNOWLEDGE_DIR, KnowledgeIndex, Passage, chunk_document,
                           load_index, load_passages, parse_front_matter)
from app.treatment_guides import TREATMENT_GUIDES

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- corpus
def test_the_corpus_has_documents():
    passages = load_passages()
    assert len(passages) >= 20, "the library is too small to be worth searching"
    assert all(p.title and p.text for p in passages)


def test_front_matter_parses():
    meta, body = parse_front_matter(
        "---\ntitle: A thing\ntopic: inputs\npest_slugs: [aphids, grub]\n---\n\n## H\ntext\n"
    )
    assert meta["title"] == "A thing"
    assert meta["pest_slugs"] == ["aphids", "grub"]
    assert "## H" in body


def test_a_file_without_front_matter_still_chunks(tmp_path):
    """A contributor who forgets the header loses metadata, not their document."""
    path = tmp_path / "notes.md"
    path.write_text("## First\nSome prose.\n\n## Second\nMore prose.\n", encoding="utf-8")
    passages = chunk_document(path)
    assert len(passages) == 2
    assert passages[0].text == "Some prose."


def known_slugs() -> set[str]:
    """Every pest slug any class manifest in this repo defines.

    Checked against all of them rather than only the fifteen currently served,
    because the corpus is meant to outlive a change of class set -- a passage
    tagged for a rice pest is correct in advance, not wrong.
    """
    import json

    slugs = set(TREATMENT_GUIDES)
    for path in (PROJECT_ROOT / "data_manifests").glob("classes_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        slugs.update(entry["class_name"] for entry in data.get("classes", []))
    return slugs


def test_every_pest_slug_in_the_corpus_is_real():
    """A typo in front matter silently disables the pest boost for that document."""
    known = known_slugs()
    for passage in load_passages():
        for slug in passage.pest_slugs:
            assert slug in known, f"{passage.doc_id} tags unknown pest {slug!r}"


# ------------------------------------------------------------- the hard rule
DOSE_PATTERNS = [
    r"\d+\s*(ml|millilitre|litre|l|g|gram|kg|oz|tsp|tbsp|teaspoon|tablespoon)\b",
    r"\d+\s*%",
    r"\bevery\s+\d+\s*(day|days|week|weeks|hour|hours)\b",
    r"\d+\s*(ml|g)\s*per\b",
]


def test_no_knowledge_document_states_a_dose_or_an_interval():
    """The library explains; the treatment guides instruct.

    This is the boundary the whole two-tool design rests on. If a passage starts
    carrying "5 ml per litre every 7 days", then `search_knowledge_base` has
    quietly become a treatment authority, and the vetting that only the guides
    receive is being bypassed.
    """
    offences = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in DOSE_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                offences.append(f"{path.name}: {match.group(0)!r}")
    assert not offences, "dose-shaped text in the reference library: " + "; ".join(offences)


# --------------------------------------------------------------------- index
def test_the_committed_index_matches_the_corpus():
    """A stale index serves answers from documents that no longer say that."""
    index = load_index()
    assert index.passages, "no committed index and no corpus"
    assert [p.title for p in index.passages] == [p.title for p in load_passages()], \
        "run scripts/build_knowledge_index.py — the index is stale"


def test_the_committed_index_is_embedded():
    index = load_index()
    assert index.embedded, "the committed index has no vectors; semantic search is off"
    assert len(index.vectors[0]) > 100


def test_keyword_search_works_with_no_embeddings():
    """The offline path: no Ollama, no index, still useful."""
    index = KnowledgeIndex(passages=load_passages())
    assert not index.embedded
    hits = index.keyword_search("why does neem oil work so slowly", k=3)
    assert hits, "keyword search found nothing"
    assert any("neem" in passage.title.lower() for _, passage in hits)


def test_search_falls_back_when_the_embedder_is_unavailable():
    """`embed` returning None must degrade to keywords, not to nothing."""

    class NoEmbeddings:
        def embed(self, texts):
            return None

    index = load_index()
    hits = index.search("neem oil", NoEmbeddings(), k=2)
    assert hits, "search gave up instead of falling back to keywords"


def test_the_pest_boost_prefers_a_tagged_passage():
    index = KnowledgeIndex(passages=[
        Passage(doc_id="a", title="General watering", text="water the crop well"),
        Passage(doc_id="b", title="General watering", text="water the crop well",
                pest_slugs=["aphids"]),
    ])
    hits = index.keyword_search("water the crop", k=2, pest_slug="aphids")
    assert hits[0][1].pest_slugs == ["aphids"]


def test_search_of_an_empty_query_returns_nothing():
    assert load_index().search("   ", None) == []
