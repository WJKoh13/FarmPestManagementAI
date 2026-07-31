# CLAUDE.md

Repository-level instructions for Claude Code and other implementation agents.

## Required startup

Before taking any project action:

1. Run `git branch --show-current` and confirm the branch is `zy_CNN`.
2. Run `git status --short` and preserve all existing user changes.
3. Read `docs/STATUS.md` to learn the last completed phase and current risks.
4. Read the relevant section of `docs/PHASES.md` for the authorized phase.
5. Inspect the existing implementation and tests before creating or replacing files.

Stop if the branch is not `zy_CNN`. Do not assume deleted files from `main` are
missing work: this branch is an intentional clean-slate experiment.

## Instruction and documentation hierarchy

When instructions differ, use this order:

1. The user's current explicit request
2. This `CLAUDE.md`
3. `docs/STATUS.md` and `docs/PHASES.md`
4. The remaining documents in `docs/`
5. `README.md`

Treat code, shipped YAML configurations, and tests as evidence of the implemented
state. Report material conflicts instead of silently choosing one interpretation.

## Phase-gated work

- Work on exactly one authorized phase at a time.
- Do not begin the next phase until the user says `CONTINUE PHASE <number>`.
- Do not skip ahead to create later-phase modules or empty placeholders.
- Do not install software, download dependencies, pull images or models, start
  persistent services, run full training, or run long benchmarks without approval.
- Update `docs/STATUS.md` at the end of each completed phase.
- Keep `docs/PHASES.md` aligned if the user explicitly changes the plan.

At the end of a phase, report:

1. What was inspected or implemented
2. Every file created or modified
3. Commands that were run
4. Verification and test results
5. Remaining risks or decisions
6. The exact next phase

Then stop and wait for approval.

## Git safety

This branch is experimental and will not be merged directly into `main`.
Successful files will later be transferred to a new integration branch.

Without an explicit user request, do not:

- Switch branches
- Merge, rebase, cherry-pick, pull, commit, push, or rewrite history
- Restore implementation from `main`
- Delete or modify `.git`
- Discard tracked or untracked user changes
- Run `git clean -x` or `git clean -fdx`

The dataset is ignored by Git. A command that removes ignored files could destroy it.

## Project purpose and architecture

Build an entirely offline organic farm pest-management assistant:

- A custom PyTorch CNN performs pest classification.
- FastAPI validates requests and orchestrates local components.
- SQLite stores verified organic/IPM knowledge with provenance.
- Ollama provides a local conversational explanation of retrieved evidence.
- Streamlit provides the farmer and manual-evaluation interfaces.
- Docker Compose packages the final offline system when that phase is authorized.

The CNN is the only pest-identification component. The LLM must not classify an
image, override a CNN result, conceal uncertainty, or invent treatment facts.

## Dataset safety

The local source dataset is:

`ip102_v1.1/Classification/`

It contains `images/`, `classes.txt`, `train.txt`, `val.txt`, and `test.txt`.
`ip102_v1.1/Detection/` is out of scope unless the user explicitly adds an
object-detection phase.

Treat all of `ip102_v1.1/` as read-only. Never rename, move, delete, overwrite,
re-encode, reorganize, or randomly resplit source data. Never commit the dataset.
Write derived manifests and reports under `data/processed/` and `data/reports/`.

Preserve the official train, validation, and test assignments. Do not access the
test set for architecture, hyperparameter, augmentation, epoch, threshold, or
scope selection. The selected final scope's test set is evaluated once in Phase 9,
after the complete inference policy is frozen.

## Classification scopes

Both scopes are first-class configuration targets:

- `rice10`: 10 remapped rice-pest classes; the development and smoke-test scope
- `full102`: all IP102 labels 0 through 101; the broader experiment scope

`dataset.scope` selects the scope. `num_classes` must always be derived through
`farm_pest_ai.scopes`, producing 10 or 102 as appropriate. Never duplicate or
hard-code this mapping elsewhere.

Every checkpoint, registry entry, evaluation result, API response, and knowledge
lookup must retain its dataset scope and class-mapping version. Reject scope and
checkpoint mismatches. Never combine rice10 and full102 metrics as if they were
the same classification task.

## CNN protocol

- Input is 160 x 160 RGB.
- Output is `num_classes` raw logits; do not put softmax inside the model.
- Implement models from primitive PyTorch layers.
- `torchvision.transforms` and general utilities are allowed.
- `torchvision.models`, prebuilt architectures, pretrained weights, and downloaded
  CNN checkpoints are prohibited.
- Apply random augmentation only to training data.
- Keep validation and test preprocessing deterministic.
- Calculate class weights or sampling statistics from training data only.
- Use validation macro F1 as the primary model-selection metric.
- Record seeds, resolved configuration, environment, class mapping, and
  preprocessing version for every run.
- Support best/last checkpoints and resumable training.

## Configuration contract

Configuration precedence, lowest to highest, is:

1. YAML files in `configs/`, composed using `extends`
2. `FPA__` environment variables
3. CLI overrides such as `--set dataset.scope=full102`

Do not place developer-specific absolute paths in Python modules or shipped YAML.
Resolve relative paths from the repository root. Keep `configs/base.yaml` generic;
put scope- or experiment-specific settings in extending configurations.

Do not state `dataset.num_classes` in configuration. It is derived and a
contradictory value is a configuration error.

## Implementation conventions

- Use the `src/farm_pest_ai/` package layout.
- Use executable `.py` scripts, not notebooks.
- Add files only when the active phase needs them.
- Prefer small modules with clear ownership and typed interfaces.
- Add type hints to public functions and meaningful docstrings to public APIs.
- Keep scripts Windows-safe and Linux-container-safe.
- Make data preparation idempotent and resumable where practical.
- Use atomic writes for generated manifests, metadata, and checkpoints.
- Do not silently fall back from CUDA to CPU for an approved full training run.
- Preserve logs, metrics, resolved configurations, and environment information.
- Add or update tests with every implemented behavior.

## Verification commands

Use commands appropriate to the current provisioned environment:

```bash
python scripts/verify_environment.py
python scripts/verify_environment.py --scope full102
python scripts/verify_environment.py --config data_full102.yaml --print-config
python -m pytest -q
```

After Phase 3 provisions the development tools, also use when relevant:

```bash
python -m ruff check src scripts tests
python -m mypy
```

Do not install a missing command merely to run it without first obtaining the
required phase approval.

## Knowledge and response safety

- Store treatment guidance only from approved, authoritative sources.
- Retain source organization, title, date/version, jurisdiction, verification
  status, and local source reference.
- Do not infer organic approval or dosage from model memory.
- CNN coverage and verified knowledge coverage are independent.
- When knowledge is missing, return identification and state that verified
  treatment guidance is unavailable.
- When the CNN is uncertain, show alternatives and withhold class-specific
  treatment guidance.
- If Ollama fails or returns invalid structured output, preserve a safe CNN-only
  response.
- Keep identification and treatment visibly separate.

## Current sources of truth

- `docs/STATUS.md`: completed phase, next phase, verified invariants, open risks
- `docs/PHASES.md`: phase boundaries, approval gates, test-set discipline
- `docs/ARCHITECTURE.md`: component boundaries and failure modes
- `docs/DATASET.md`: verified IP102 findings
- `docs/TRAINING.md`: training and model constraints
- `docs/EVALUATION.md`: evaluation policy
- `docs/KNOWLEDGE_BASE.md`: provenance and grounding requirements
- `docs/OFFLINE_DEPLOYMENT.md`: container and offline requirements
- `docs/LIMITATIONS.md`: known limitations
- `docs/INTEGRATION_HANDOFF.md`: later transfer into an integration branch

Read the relevant document before changing its corresponding subsystem.
