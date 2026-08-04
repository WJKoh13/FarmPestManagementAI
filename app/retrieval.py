"""Pick the treatment guides relevant to a question, to ground the LLM.

The guides in `treatment_guides.py` are the vetted, farmer-safe source of truth.
A small local model asked about pests unprompted will happily invent products and
dose rates, so nothing reaches the model without the relevant guide attached and
an instruction to answer from it.

Retrieval is deterministic keyword scoring, not embeddings. That is a deliberate
choice: it adds no dependency, no download and no startup cost to an app whose
selling point is running offline on a farmer's laptop, and with fifteen documents
of a few hundred words each there is nothing for a vector index to win.
"""

from __future__ import annotations

import re

from app.treatment_guides import GENERIC_GUIDE, TREATMENT_GUIDES

# Weights. Naming a pest outright must beat every amount of incidental word
# overlap, and the pest already identified from a photo must beat a passing
# mention, so that "is it safe for bees?" stays on the pest in hand.
SCORE_CONTEXT_PEST = 200.0
SCORE_FULL_NAME = 100.0
SCORE_NAME_TOKEN = 25.0
SCORE_BODY_TOKEN = 1.0

# Below this, a question is about pest management generally rather than about any
# particular pest, and naming that pest to the farmer would be a fabricated
# diagnosis.
MIN_SCORE = 20.0

# Word overlap alone, with no pest named -- "is neem oil safe for bees?" touches
# the aphid guide without being about aphids. Too weak to show as an
# identification, but the only place the answer is written down, so it is given
# to the model as background rather than thrown away.
SOFT_MIN_SCORE = 2.0

STOPWORDS = frozenset("""
a about all am an and any are as at be been but by can could did do does for from get got
had has have how i if in is it its just me my need not of on or should so some that the
their them then there these they this to too use used very was we what when where which
who why will with would you your
""".split())

# Words shared by half the guides carry no signal about which one is meant.
GENERIC_TOKENS = frozenset({
    "pest", "pests", "insect", "insects", "bug", "bugs", "crop", "crops", "plant", "plants",
    "leaf", "leaves", "spray", "sprays", "organic", "treatment", "field", "soil", "damage",
    "check", "look", "remove", "product", "label", "approved", "day", "days", "week", "weeks",
})


def tokenize(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z]+", text.lower()) if word not in STOPWORDS]


def _name_variants(slug: str, display_name: str) -> tuple[set[str], set[str]]:
    """(full-name phrases, distinctive single tokens) to match a question against."""
    phrases = {slug.replace("_", " ").lower(), display_name.lower()}
    tokens = {token for phrase in phrases for token in tokenize(phrase)}
    return phrases, tokens - GENERIC_TOKENS


def score_guides(query: str, display_names: dict[str, str] | None = None,
                 context_slug: str | None = None) -> list[tuple[float, str]]:
    """(score, slug) for every guide, best first."""
    display_names = display_names or {}
    query_lower = query.lower()
    query_tokens = set(tokenize(query)) - GENERIC_TOKENS

    scored: list[tuple[float, str]] = []
    for slug, guide in TREATMENT_GUIDES.items():
        display_name = display_names.get(slug, slug.replace("_", " "))
        phrases, name_tokens = _name_variants(slug, display_name)

        score = 0.0
        if slug == context_slug:
            score += SCORE_CONTEXT_PEST
        if any(phrase in query_lower for phrase in phrases):
            score += SCORE_FULL_NAME
        score += SCORE_NAME_TOKEN * len(query_tokens & name_tokens)
        score += SCORE_BODY_TOKEN * len(query_tokens & (set(tokenize(guide)) - GENERIC_TOKENS))
        scored.append((score, slug))

    return sorted(scored, key=lambda item: (-item[0], item[1]))


def relevant_guides(query: str, display_names: dict[str, str] | None = None,
                    context_slug: str | None = None, k: int = 2) -> list[tuple[str, str]]:
    """Up to ``k`` (display name, guide text) pairs worth showing the model.

    Empty when nothing scores above `MIN_SCORE`; the caller falls back to
    `GENERIC_GUIDE`, which is safe for any pest.
    """
    display_names = display_names or {}
    return [
        (display_names.get(slug, slug.replace("_", " ").title()), TREATMENT_GUIDES[slug])
        for score, slug in score_guides(query, display_names, context_slug)[:k]
        if score >= MIN_SCORE
    ]


def guidance_block(query: str, display_names: dict[str, str] | None = None,
                   context_slug: str | None = None, k: int = 2) -> str:
    """The retrieved guides formatted for a system prompt.

    When no guide is a confident match, the general advice leads and the closest
    partial matches follow as background, marked as not being an identification
    so the model quotes their facts without naming a pest nobody identified.
    """
    display_names = display_names or {}
    if guides := relevant_guides(query, display_names, context_slug, k=k):
        return "\n\n".join(f"### {name}\n{guide}" for name, guide in guides)

    block = f"### General organic pest guidance\n{GENERIC_GUIDE}"
    soft = [
        (display_names.get(slug, slug.replace("_", " ").title()), TREATMENT_GUIDES[slug])
        for score, slug in score_guides(query, display_names, context_slug)[:k]
        if SOFT_MIN_SCORE <= score < MIN_SCORE
    ]
    for name, guide in soft:
        block += (
            f"\n\n### Background — guidance for {name}\n"
            "(The pest here has NOT been identified. Use these facts if they answer the "
            f"question, but do not tell the farmer they have {name}.)\n{guide}"
        )
    return block
