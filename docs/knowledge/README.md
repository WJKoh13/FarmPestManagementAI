# The reference library

Background knowledge the assistant can search. Every file here is chunked,
embedded and indexed by `scripts/build_knowledge_index.py`, and reaches the
language model through the `search_knowledge_base` tool.

## The one rule

**No file in this directory states a product dose, a concentration, a spray
interval or a re-entry period.**

Those live in `app/treatment_guides.py` and nowhere else. That file is short,
hand-checked, and covered by tests that assert every class has a guide and that
every guide carries its safety warnings. This directory is none of those things:
it is long, it is prose, and it exists to be retrieved in fragments.

So the two carry different authority, and the split is enforced in three places:

- `lookup_treatment_guide` returns guide text and is described to the model as
  the only source of treatments.
- `search_knowledge_base` labels its payload *"background reference, NOT a
  treatment authority"* and repeats the boundary in its instructions.
- `tests/test_knowledge.py` greps this directory for dose-shaped strings and
  fails if one appears.

Write "apply it when the larvae are small, before they tunnel into the stem" —
that is timing tied to a biological stage, which is exactly what this library is
for. Do not write "apply 5 ml per litre every seven days".

## Front matter

Every file opens with:

```yaml
---
title: Neem oil — what it does and when it works
topic: inputs
pest_slugs: [aphids, flea_beetle]
source: Written for this project from extension-service guidance.
---
```

- `title` is what the model sees above the passage, so it should read as a
  statement, not a filename.
- `topic` is one of `practice`, `inputs`, `diagnosis`, `compliance`, `lifecycle`.
- `pest_slugs` must match the slugs in `data_manifests/classes_top15.json`. They
  boost retrieval when that pest is the one in hand; an empty list is fine for
  general material.
- `source` says where the content came from. Be honest — "written for this
  project" is a legitimate answer and better than implying a citation.

## Chunking

Split on `##` headings. Each section becomes one passage, so a heading should be
a self-contained idea of roughly 120–200 words. A section that only makes sense
after the one above it will be retrieved alone and will read as a fragment.
