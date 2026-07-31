# Project status

Experimental branch: `zy_CNN`. Updated at the end of every phase.

| Field | Value |
| --- | --- |
| Current phase completed | **Phase 3 — Python, Docker and CUDA environment** |
| Next phase | Phase 4 — Full dataset audit and derived manifests |
| Branch | `zy_CNN` |
| Active default scope | `rice10` (switchable to `full102`) |
| Dependencies installed | **Yes** — `.venv`, base + `train` + `dev`; `app` deferred to Phase 12 |
| Interpreter | Official CPython 3.12.5 (`win-amd64`) in `.venv` |
| PyTorch | `2.13.0+cu126`, CUDA available, cuDNN 91002 |
| Test suite | 126 passed (117 from Phase 2, plus 9 environment guards) |
| Lint / types | `ruff` clean, `mypy` clean |
| Source data | Unmodified, read-only (reverified: 75,222 images) |

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

### Phase 3 — Python, Docker and CUDA environment (complete)

Provisioned the training environment and closed three of the four environment
risks carried from Phase 1. No source data was touched.

**Interpreter (risk 1 closed).** The MSYS2 Python that shadows CPython on `PATH`
was confirmed to report platform tag `mingw_x86_64_msvcrt_gnu`, for which no
PyTorch wheel exists. `.venv` was therefore created from the official CPython
3.12.5 (`win-amd64`) via `py -3.12`. **Every project command must use
`.venv\Scripts\python.exe`**; a bare `python` still resolves to MSYS2.

**Isolation (risk 2 closed).** The global interpreter carries 163 packages,
including an unrelated `pytest-django` and an older `numpy`. The venv resolves
exactly one `site-packages` directory — its own — so none of that is visible.

**CUDA.** `torch 2.13.0+cu126` and `torchvision 0.28.0+cu126` were installed from
the cu126 index. GPU execution was verified by running real work, not just an
`is_available()` call: a 512x512 matmul, an fp16 `autocast` block, and a
`Conv2d` at the project's actual 160x160 input size, all on device.

**Docker GPU passthrough (risk 4 closed).** `nvidia/cuda:12.6.3-base-ubuntu22.04`
was pulled and the container saw the RTX 4070 with driver 591.44 and sm_89,
under both `--gpus all` and `--runtime=nvidia --gpus all`. The driver advertises
CUDA 13.1, so the cu126 build has ample headroom.

**Dependency lock.** `requirements-lock.txt` pins all 45 resolved packages and
carries the `--extra-index-url` for cu126, since those wheels are not on PyPI. A
`pip install --dry-run -r` confirmed it resolves. The project package itself is
installed separately with `pip install -e . --no-deps`, which retires the
`add_bootstrap_path` workaround noted in `cli.py`.

**Linting.** `ruff` and `mypy` ran for the first time, since Phase 2 could not
install them. 24 findings surfaced in existing Phase 2 code: 22 were autofixed
and 5 fixed by hand. All were cosmetic — `typing` to `collections.abc` imports,
`__all__` ordering, a redundant dict comprehension, an over-long line, ambiguous
`l` loop variables, and three `pytest.raises(match=...)` patterns whose
unescaped `.` matched any character. The last group makes those assertions
strictly tighter. No behaviour changed, and all 117 tests still pass.

Installed versions are newer than the `pyproject.toml` floors in several cases
(`numpy 2.4.4`, `pandas 3.0.5`, `mypy 2.3.0`). The floors were left as-is and the
exact versions recorded in the lock file.

## Verified invariants

These are enforced by tests and re-checked every phase.

- `num_classes` is derived from `dataset.scope` and is never hard-coded.
  `tests/test_shipped_configs.py` fails if any config states it.
- Source dataset paths never appear among the project's writable directories.
- Configuration files contain no developer-specific absolute paths.
- Ollama model tags remain marked `verified: false` until Phase 11 confirms
  them against the library.
- `inference.checkpoint` stays `null` until a model is frozen in Phase 9.
- `requirements-lock.txt` pins every dependency exactly and carries the cu126
  extra index; `tests/test_environment.py` fails if a pin loosens, if the CUDA
  build silently becomes a CPU wheel, or if the index disappears.
- The active interpreter is never MSYS2/MinGW, which cannot run PyTorch.

## Open risks

Carried forward from Phase 1, plus items raised in Phase 2.

| # | Risk | Phase to resolve |
| --- | --- | --- |
| 1 | ~~MSYS2 Python shadows official CPython~~ **Closed in Phase 3.** `.venv` built from official CPython 3.12.5. A bare `python` still hits MSYS2, so always invoke `.venv\Scripts\python.exe` | done |
| 2 | ~~Global `site-packages` polluted~~ **Closed in Phase 3.** venv resolves only its own `site-packages` | done |
| 3 | VRAM is contended: 4,091 MiB free measured under desktop load versus 7,054 MiB when idle, of 8,188 MiB total. Training and Ollama must not share the GPU | 8, 14 |
| 4 | ~~Docker GPU passthrough unverified~~ **Closed in Phase 3.** Verified with `nvidia/cuda:12.6.3-base-ubuntu22.04` under both `--gpus all` and `--runtime=nvidia` | done |
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
