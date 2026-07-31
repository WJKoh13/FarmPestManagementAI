# Architecture

An entirely offline system: after provisioning, no component requires internet
access.

## Components

| Component | Technology | Responsibility |
| --- | --- | --- |
| CNN | PyTorch, custom architecture | Pest classification, top-k, confidence, uncertainty, model metadata |
| Knowledge base | SQLite | Verified pest and organic/IPM information with provenance |
| LLM | Ollama, local model | Conversational explanation of retrieved evidence only |
| Backend | FastAPI | Validation, inference, retrieval, orchestration, safe fallbacks |
| Frontend | Streamlit | Farmer interface, manual evaluation, system status |
| Orchestration | Docker Compose | Service wiring, profiles, persistence |

## Request flow

```
image -> validate type/size -> safe decode -> preprocess (deterministic)
      -> CNN -> logits -> softmax -> top-k + confidence
      -> uncertainty policy
      -> knowledge retrieval by class ID
      -> grounded LLM request (evidence only)
      -> schema validation of LLM output
      -> combined response
```

Identification and treatment stay visibly separate at every step.

## Separation of concerns

The CNN is the **only** component that identifies a pest. The LLM never
classifies, never overrides or conceals the CNN result, and never invents
treatment facts. It may only explain and summarise evidence retrieved from the
verified knowledge base, citing knowledge record IDs.

## Scope propagation

The active dataset scope (`rice10` or `full102`) is not a display detail; it is
carried end to end. `num_classes` is derived from the scope in
`farm_pest_ai.scopes` and never hard-coded in loaders, models, losses, metrics,
evaluation, checkpoint loading, API schemas, retrieval or the frontend.

Every checkpoint and model-registry entry records the dataset scope, number of
classes, class mapping, class-mapping version, manifest version, preprocessing
version, model configuration and training seed. **A checkpoint trained for one
scope is rejected when loaded under another.** Metrics from `rice10` and
`full102` are never mixed, because they are different classification tasks.

## Knowledge coverage is independent of CNN coverage

The CNN may support all 102 classes while the verified knowledge base contains
records for a smaller reviewed subset. When the CNN identifies a class with no
verified record, the system returns the identification, preserves the
uncertainty status, and states plainly that verified treatment guidance is
unavailable. It does not ask the LLM to fill the gap.

## Failure modes

| Condition | Behaviour |
| --- | --- |
| Ollama unavailable | Return CNN results and retrieved records; state that conversational explanation is unavailable |
| CNN uncertain | Show alternatives, request another image, withhold class-specific treatment guidance |
| Knowledge missing | Return identification only; state that verified treatment information is unavailable |
| LLM output fails schema validation | Reject the output rather than repair it silently; fall back to CNN-only |

CNN-only operation is always a valid, working state.

## Package layout

```
src/farm_pest_ai/
    scopes.py            Scope definitions - the source of truth for num_classes
    config.py            Layered YAML + env + CLI configuration, validated
    logging_config.py    Console and JSON Lines logging
    reproducibility.py   Seeding and environment capture
    cli.py               Shared argument parsing and bootstrap
    data/                Manifests, audit, dataset, transforms      (Phases 4-5)
    vision/              Models, blocks, losses, metrics, engine     (Phases 6-9)
    knowledge/           Schema, repository, ingestion, retrieval    (Phase 10)
    llm/                 Ollama client, prompts, schemas, grounding  (Phase 11)
    orchestration/       Analysis and chat services                  (Phase 12)
    api/                 FastAPI app, schemas, routes                (Phase 12)
    frontend/            Streamlit app and pages                     (Phase 13)
    evaluation/          Manual review store, rubrics, reports       (Phase 13)
```

Subpackages are created when their phase requires them, rather than as empty
placeholders.

## Configuration

Three layers, lowest precedence first:

1. YAML files in `configs/`, composed with an optional `extends` key
2. `FPA__`-prefixed environment variables (`FPA__DATASET__SCOPE=full102`)
3. CLI overrides (`--set dataset.scope=full102`, `--scope full102`)

No developer-specific absolute path appears in the package. Relative paths are
anchored to the project root, so the same configuration works on Windows and
inside a Linux container.
