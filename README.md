# FarmPestManagementAI

An entirely **offline** organic farm pest management assistant: a custom CNN
identifies a crop pest from a photograph, a verified local knowledge base
supplies organic and IPM guidance, and a local LLM explains that guidance
conversationally. After provisioning, no component needs internet access.

> **Experimental branch.** This is `zy_CNN`, an intentionally clean-slate
> experimental workspace. The earlier shared implementation was deliberately
> removed; deleted files from `main` are not missing work. This branch will not
> be merged into `main` — if the experiment succeeds, selected finished files
> will be transferred to a new integration branch created from the latest
> `main`.

## What it does

A farmer can upload or capture a crop-pest image and receive:

1. A prediction from the custom CNN, with the active classification scope shown
2. Top candidate pests with confidence levels
3. A clear uncertainty warning when identification is unreliable
4. Verified local organic/IPM information, when a reviewed record exists
5. Follow-up answers from a local LLM, grounded **only** in retrieved evidence

The CNN is the only component that identifies a pest. The LLM never classifies,
never overrides or conceals the CNN result, and never invents treatment facts.

## Classification scopes

The system supports two configurable scopes, and `num_classes` is always
**derived** from the active scope — never hard-coded anywhere.

| Scope | Classes | Description |
| --- | --- | --- |
| `rice10` | 10 | Ten-class rice-pest subset of IP102. Development scope and a focused deployment option. |
| `full102` | 102 | The complete IP102 classification task. Broader coverage, much harder imbalance. |

Select a scope in configuration, by environment variable, or on the command
line:

```yaml
dataset:
  scope: rice10   # or full102
```

```bash
python scripts/verify_environment.py --scope full102
FPA__DATASET__SCOPE=full102 python scripts/verify_environment.py
python scripts/verify_environment.py --set dataset.scope=full102
```

Stating a `num_classes` that contradicts the scope is a hard configuration
error, not a warning. Checkpoints record their scope and are rejected if loaded
under a different one, and metrics from the two scopes are never mixed — they
are different classification tasks.

## Dataset

IP102 v1.1 (classification subset): 75,222 images across 102 classes, using the
official train/validation/test splits (45,095 / 7,508 / 22,619). The dataset
lives at `ip102_v1.1/`, is Git-ignored, and is treated as strictly **read-only**
— never renamed, moved, re-encoded, reorganised or resplit. Derived manifests
and reports are written to project-controlled directories.

Full audit results are in [docs/DATASET.md](docs/DATASET.md).

## Getting started

Requires Python 3.10+. Phase 3 provisions a project virtual environment with the
CUDA 12.6 PyTorch build.

```bash
# Create the environment (Windows: use the official CPython, not MSYS2)
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

> On this machine an MSYS2 Python shadows CPython on `PATH` and cannot run
> PyTorch. Invoke `.venv\Scripts\python.exe` explicitly rather than `python`.

```bash
# Verify the harness and report on the environment
.venv\Scripts\python.exe scripts/verify_environment.py

# Fail the run if PyTorch or CUDA is missing
.venv\Scripts\python.exe scripts/verify_environment.py --require-torch

# Inspect the fully resolved configuration
.venv\Scripts\python.exe scripts/verify_environment.py --config data_full102.yaml --print-config

# Check the derived manifests against the source data (fast, no image decoding)
.venv\Scripts\python.exe scripts/verify_dataset.py --scope rice10

# Check tensor shapes, RGB conversion, evaluation determinism and augmentation
.venv\Scripts\python.exe scripts/verify_loader.py --scope rice10

# Run the test suite, linter and type checker
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check src scripts tests
.venv\Scripts\python.exe -m mypy
```

Every script supports `--help`, `--config`, `--set`, `--scope`, `--seed`,
`--log-level` and `--log-file`.

## Configuration

Three layers, lowest precedence first: YAML files in `configs/` (composed via
`extends`), then `FPA__`-prefixed environment variables, then CLI overrides. No
developer-specific absolute paths appear in the package; relative paths anchor
to the project root, so the same configuration works on Windows and inside a
Linux container.

## Architecture

| Component | Technology |
| --- | --- |
| CNN | PyTorch, custom architecture from primitive layers |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM runtime | Ollama (local) |
| Knowledge storage | SQLite |
| Orchestration | Docker Compose |

Custom CNNs are built from primitive PyTorch layers only. `torchvision.models`,
prebuilt architectures, pretrained weights and downloaded checkpoints are all
prohibited. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Safety commitments

- Confidence is never presented as certainty, and uncertainty is never hidden.
- Class-specific treatment guidance is withheld while a prediction is uncertain.
- Treatment facts, organic approval and dosage are never invented; organic
  claims carry jurisdiction context and dosage requires a verified source.
- When a predicted class has no verified knowledge record, the system returns
  the identification and states plainly that verified guidance is unavailable.
- Identification and treatment stay visibly separate.
- The system keeps working, CNN-only, if the LLM runtime is unavailable.

## Phase-gated workflow

Work proceeds one phase at a time. Each phase ends with a summary, verification
results, open risks and the next phase, then stops for explicit approval before
continuing. Installing software, pulling images or models, running full training
and starting persistent services all require approval.

Current state: **Phase 3 complete** (Python, Docker and CUDA environment). See
[docs/STATUS.md](docs/STATUS.md) and [docs/PHASES.md](docs/PHASES.md).

## Documentation

| Document | Contents |
| --- | --- |
| [STATUS.md](docs/STATUS.md) | Current phase, verified invariants, open risks |
| [PHASES.md](docs/PHASES.md) | Phase plan, approval gates, test-set discipline |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, request flow, failure modes |
| [DATASET.md](docs/DATASET.md) | IP102 audit, scopes, class distributions |
| [TRAINING.md](docs/TRAINING.md) | Training protocol and model constraints |
| [EVALUATION.md](docs/EVALUATION.md) | Metrics, manual review, safety cases |
| [KNOWLEDGE_BASE.md](docs/KNOWLEDGE_BASE.md) | Verified knowledge schema and provenance |
| [OFFLINE_DEPLOYMENT.md](docs/OFFLINE_DEPLOYMENT.md) | Containers and offline validation |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | Honest limitations, updated as found |
| [INTEGRATION_HANDOFF.md](docs/INTEGRATION_HANDOFF.md) | Files suitable for transfer |
