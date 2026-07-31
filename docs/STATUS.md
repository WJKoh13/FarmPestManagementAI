# Project status

Experimental branch: `zy_CNN`. Updated at the end of every phase.

| Field | Value |
| --- | --- |
| Current phase completed | **Phase 2 — Project harness** |
| Next phase | Phase 3 — Python, Docker and CUDA environment |
| Branch | `zy_CNN` |
| Active default scope | `rice10` (switchable to `full102`) |
| Dependencies installed | **No** — deferred to Phase 3 by design |
| Test suite | 117 passed |
| Source data | Unmodified, read-only |

## Phase log

### Phase 1 — Read-only discovery (complete)

Read-only inspection. No files created or modified.

Every previously reported dataset fact was independently reverified and
confirmed exactly:

| Fact | Value |
| --- | --- |
| Classification images | 75,222, all real JPEG |
| Classes | 102 |
| Manifest labels | 0–101 (zero-based) |
| `classes.txt` numbering | 1–102, so `label = id - 1` |
| train / val / test | 45,095 / 7,508 / 22,619 |
| All 102 classes in every split | Yes |
| Smallest / largest train class | 42 (label 72) / 3,444 (label 101) |
| rice10 train / val / test / total | 4,318 / 721 / 2,166 / 7,205 |

Integrity was perfect: zero missing images, zero unreferenced images, zero
cross-split filename overlap, zero conflicting labels, and a bijection between
the 75,222 manifest records and the 75,222 files on disk. The ten rice10 class
names matched `classes.txt` exactly.

Hardware reverified: RTX 4070 Laptop (8,188 MiB, compute 8.9, driver 591.44),
Ryzen 7 8845HS (8C/16T), 63.3 GB RAM, 1,410 GB free on `D:`. Docker 29.6.1 with
Compose v5.3.0 and a registered `nvidia` runtime; WSL2 2.7.11.

See [DATASET.md](DATASET.md) for the full findings.

### Phase 2 — Project harness (complete)

Created the root-level harness: `src` layout, layered YAML configuration,
dataset-scope selection with automatic `num_classes` derivation, environment
overrides, structured logging, reproducibility utilities, the first CLI entry
point, unit tests and this documentation set. No dependencies were installed.

Both exit criteria were verified by execution, not assertion:

- **Both scopes resolve through configuration** via six independent paths
  (default, `--scope`, `--config`, `FPA__DATASET__SCOPE`, `--set`, and
  CLI-beats-environment precedence), each deriving `num_classes` correctly
  (10 for `rice10`, 102 for `full102`).
- **Inconsistent scope/`num_classes` combinations are rejected** with a clear
  `ConfigError`, as are unknown and missing scopes.

## Verified invariants

These are enforced by tests and re-checked every phase.

- `num_classes` is derived from `dataset.scope` and is never hard-coded.
  `tests/test_shipped_configs.py` fails if any config states it.
- Source dataset paths never appear among the project's writable directories.
- Configuration files contain no developer-specific absolute paths.
- Ollama model tags remain marked `verified: false` until Phase 11 confirms
  them against the library.
- `inference.checkpoint` stays `null` until a model is frozen in Phase 9.

## Open risks

Carried forward from Phase 1, plus items raised in Phase 2.

| # | Risk | Phase to resolve |
| --- | --- | --- |
| 1 | MSYS2 Python shadows official CPython on `PATH`; PyTorch wheels do not support MSYS2 | 3 |
| 2 | Global `site-packages` is shared and polluted; a dedicated venv is required | 3 |
| 3 | Only ~4.1 GB of 8 GB VRAM free with the desktop loaded; training and Ollama must not share the GPU | 3, 8, 14 |
| 4 | Docker `nvidia` runtime is registered but GPU passthrough is unverified (needs an image pull) | 3 |
| 5 | 7.5% of rice10 images are under 160 px on the short side and will be upscaled | 4, 5 |
| 6 | full102 imbalance is 82x; validation has classes with only 7 images, so macro F1 will be noisy | 4, 8 |
| 7 | Content-hash duplicates and cross-split leakage are not yet measured | 4 |
| 8 | Ollama is not installed | 11 |
| 9 | `classes.txt` mixes common names and Latin binomials; taxonomy is preserved, not corrected | 4, 10 |

## Rules in force

- One phase at a time; stop and wait for `CONTINUE PHASE <n>`.
- The `ip102_v1.1` directory is read-only. Never rename, move, delete,
  overwrite, re-encode or resplit source data.
- No test set is touched before the model is frozen in Phase 9.
- Approval is required before installing software, pulling images or models,
  running full training, or starting persistent services.
- No Git operations beyond `branch`/`status` without an explicit request.
