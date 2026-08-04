"""The reference library the assistant can search.

What this is, plainly: the markdown files in `docs/knowledge/` are cut into
passages, each passage is turned into a list of numbers by a local embedding
model, and a question is answered by finding the passages whose numbers point in
the most similar direction. That is retrieval-augmented generation, and it is
about forty lines of it -- there is nothing here but a dot product.

It stays deliberately small for two reasons. The corpus is a few dozen passages,
where a vector database would win nothing it does not already lose in setup
time; and the app has to run offline on a laptop, so a dependency that downloads
a model at import is not acceptable.

Two things it is *not*:

It is not the treatment guides. Those live in `treatment_guides.py`, are vetted,
and are the only sanctioned source of a product or a dose. This library explains
and gives background. `agent_tools.py` keeps the two apart on purpose.

It is not required. If the embedding model is missing, `search` falls back to
keyword scoring over the same passages, and the app carries on.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.retrieval import GENERIC_TOKENS, tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "docs" / "knowledge"
INDEX_PATH = PROJECT_ROOT / "data_manifests" / "knowledge_index.json"

# Passages returned per search. Three is about the ceiling for a 3B model: past
# that the transcript grows faster than its ability to find the relevant line.
DEFAULT_K = 3

# Added to a passage tagged with the pest currently in hand. Cosine similarity
# runs to 1.0, so 0.15 is a real thumb on the scale without letting an
# irrelevant passage win purely for being about the right insect.
PEST_BOOST = 0.15

# Below this, a passage is not about the question. Without a floor the search
# always returns its three least-bad guesses, and the model treats them as
# relevant because they were handed to it.
MIN_SIMILARITY = 0.35


@dataclass
class Passage:
    """One retrievable chunk: a heading and the prose under it."""

    doc_id: str
    title: str
    text: str
    topic: str = ""
    pest_slugs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "title": self.title, "text": self.text,
                "topic": self.topic, "pest_slugs": self.pest_slugs}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Passage":
        return cls(doc_id=data.get("doc_id", ""), title=data.get("title", ""),
                   text=data.get("text", ""), topic=data.get("topic", ""),
                   pest_slugs=list(data.get("pest_slugs", [])))


# ------------------------------------------------------------------- parsing
def parse_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    """Split a `---` YAML header off a markdown file.

    Hand-parsed rather than importing yaml for four keys, two of which are
    strings and one a flat list. The parser is deliberately dumb: anything it
    does not understand becomes a string, and a malformed header costs one
    document's metadata rather than the whole build.
    """
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw

    meta: dict[str, Any] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            meta[key] = [item.strip() for item in inner.split(",") if item.strip()]
        else:
            meta[key] = value
    return meta, parts[2]


def chunk_document(path: Path) -> list[Passage]:
    """One passage per `##` heading.

    Splitting on headings rather than a fixed token count means every passage
    starts at an idea and ends at one. A chunker that cuts at 200 words
    regularly severs a sentence about bee safety from the warning that follows
    it, and the model then retrieves half a caveat.
    """
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    doc_title = str(meta.get("title") or path.stem.replace("-", " ").capitalize())
    topic = str(meta.get("topic") or "")
    slugs = [str(slug) for slug in meta.get("pest_slugs", [])]

    passages: list[Passage] = []
    for block in re.split(r"^##\s+", body, flags=re.MULTILINE)[1:]:
        heading, _, text = block.partition("\n")
        text = text.strip()
        if not text:
            continue
        passages.append(Passage(
            doc_id=path.stem,
            # The document title carries the subject and the section title the
            # specific claim; the model sees both, so it can tell a passage about
            # neem's bee safety from one about spinosad's.
            title=f"{doc_title} — {heading.strip()}",
            text=text,
            topic=topic,
            pest_slugs=slugs,
        ))
    return passages


def load_passages(docs_dir: Path = KNOWLEDGE_DIR) -> list[Passage]:
    """Every passage in the corpus, in a stable order."""
    if not docs_dir.is_dir():
        return []
    passages: list[Passage] = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":  # the corpus's own instructions
            continue
        passages.extend(chunk_document(path))
    return passages


# ------------------------------------------------------------------- maths
def _normalise(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector))
    return [value / length for value in vector] if length else vector


def _cosine(a: list[float], b: list[float]) -> float:
    """Both vectors are stored normalised, so the dot product *is* the cosine."""
    return sum(x * y for x, y in zip(a, b))


@dataclass
class KnowledgeIndex:
    """The corpus, optionally with its embeddings."""

    passages: list[Passage] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    embed_model: str = ""

    @property
    def embedded(self) -> bool:
        return bool(self.vectors) and len(self.vectors) == len(self.passages)

    def __len__(self) -> int:
        return len(self.passages)

    # --------------------------------------------------------------- search
    def search(self, query: str, llm: Any = None, k: int = DEFAULT_K,
               pest_slug: str = "") -> list[tuple[float, Passage]]:
        """The best passages for a question, by meaning where possible.

        Falls back to keyword scoring whenever embeddings are unavailable -- no
        index built, or the embedding model not pulled. The fallback is worse,
        not broken, and losing search entirely because one model is missing
        would be the wrong trade for an app whose selling point is running
        offline.
        """
        if not query.strip() or not self.passages:
            return []

        if self.embedded and llm is not None:
            embedded_query = llm.embed([query])
            if embedded_query:
                return self._vector_search(_normalise(embedded_query[0]), k, pest_slug)
        return self.keyword_search(query, k, pest_slug)

    def _vector_search(self, query_vector: list[float], k: int,
                       pest_slug: str) -> list[tuple[float, Passage]]:
        scored: list[tuple[float, Passage]] = []
        for vector, passage in zip(self.vectors, self.passages):
            score = _cosine(query_vector, vector)
            if pest_slug and pest_slug in passage.pest_slugs:
                score += PEST_BOOST
            scored.append((score, passage))

        scored.sort(key=lambda item: -item[0])
        return [(score, passage) for score, passage in scored[:k]
                if score >= MIN_SIMILARITY]

    def keyword_search(self, query: str, k: int = DEFAULT_K,
                       pest_slug: str = "") -> list[tuple[float, Passage]]:
        """Word overlap, reusing the tokeniser the treatment-guide search uses.

        Scores are normalised by query length so the threshold in `search` means
        roughly the same thing on either path.
        """
        query_tokens = set(tokenize(query)) - GENERIC_TOKENS
        if not query_tokens:
            return []

        scored: list[tuple[float, Passage]] = []
        for passage in self.passages:
            haystack = set(tokenize(f"{passage.title} {passage.text}")) - GENERIC_TOKENS
            overlap = len(query_tokens & haystack) / len(query_tokens)
            if pest_slug and pest_slug in passage.pest_slugs:
                overlap += PEST_BOOST
            if overlap > 0:
                scored.append((overlap, passage))

        scored.sort(key=lambda item: -item[0])
        return [(score, passage) for score, passage in scored[:k] if score >= 0.2]

    # ---------------------------------------------------------- persistence
    def save(self, path: Path = INDEX_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embed_model": self.embed_model,
            "passages": [passage.to_dict() for passage in self.passages],
            # Rounded because full float64 repr triples the file size for
            # precision far below what changes any ranking.
            "vectors": [[round(value, 6) for value in vector] for vector in self.vectors],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path


def load_index(path: Path = INDEX_PATH, docs_dir: Path = KNOWLEDGE_DIR) -> KnowledgeIndex:
    """The committed index, or an unembedded one straight from the markdown.

    A bare clone with no index file still gets keyword search over the corpus,
    so the knowledge tool is never simply absent.
    """
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return KnowledgeIndex(
                passages=[Passage.from_dict(item) for item in data.get("passages", [])],
                vectors=[list(vector) for vector in data.get("vectors", [])],
                embed_model=str(data.get("embed_model", "")),
            )
        except (OSError, ValueError):
            pass  # a corrupt index must not stop the app opening
    return KnowledgeIndex(passages=load_passages(docs_dir))


def build_index(llm: Any, docs_dir: Path = KNOWLEDGE_DIR) -> KnowledgeIndex:
    """Read the corpus and embed it. Without embeddings, still returns the passages."""
    passages = load_passages(docs_dir)
    if not passages:
        return KnowledgeIndex()

    # Title and text together: a question about bee safety should match a passage
    # headed "Effects on other insects" even when the body never says "bee".
    vectors = llm.embed([f"{p.title}\n{p.text}" for p in passages])
    if not vectors:
        return KnowledgeIndex(passages=passages)

    from app.ollama_client import EMBED_MODEL
    return KnowledgeIndex(passages=passages,
                          vectors=[_normalise(vector) for vector in vectors],
                          embed_model=EMBED_MODEL)
