# Project status

Experimental branch: `zy_CNN`. Updated at the end of every phase.

| Field | Value |
| --- | --- |
| Current phase completed | **Phase 5 — Data loader and preprocessing** |
| Next phase | Phase 6 — Custom CNN and smoke training |
| Branch | `zy_CNN` |
| Active default scope | `rice10` (switchable to `full102`) |
| Dependencies installed | **Yes** — `.venv`, base + `train` + `dev`; `app` deferred to Phase 12 |
| Interpreter | Official CPython 3.12.5 (`win-amd64`) in `.venv` |
| PyTorch | `2.13.0+cu126`, CUDA available, cuDNN 91002 |
| Test suite | 428 passed (246 through Phase 4, plus 182 loader tests) |
| Lint / types | `ruff` clean, `mypy` clean |
| Source data | Unmodified, read-only (reverified: 75,222 images, 2020 timestamps) |
| Derived manifests | Built for both scopes, idempotent, verified against source |
| Preprocessing version | `1.0.0`, fingerprint `9e75177ab60f96e0` (identical for both scopes) |

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

### Phase 4 — Full dataset audit and derived manifests (complete)

Built scope-aware derived manifests for both scopes and completed the three
checks Phase 1 deferred. **Every image in both scopes was decoded in full and
hashed** — 75,222 for `full102`, of which 7,205 form `rice10`. No source file was
modified; the manifests' 2020 timestamps and the 75,222-file count are unchanged.

**Derived manifests.** `data/processed/<scope>/{train,validation,test}.csv` plus
a `.metadata.json` sidecar per split and one `class_mapping.json` per scope. Each
record stores both the IP102 label and the project label, so nothing downstream
re-derives the mapping. Every count reproduced Phase 1 exactly (rice10
4,318/721/2,166; full102 45,095/7,508/22,619). The build is **idempotent**:
`build_manifests.py --check` re-renders and compares bytes, and passes for both
scopes. Writes are atomic, and CSVs use explicit LF so Windows and container
output are byte-identical.

**Full decode: 0 failures, 0 truncated files**, in both scopes.

**New finding — ten PNG files with a `.jpg` extension.** Phase 1's 2,000-image
`full102` sample reported 100% JPEG; the exhaustive decode showed that was a
sampling artefact. Ten files are really PNG and **seven of those are RGBA**. All
ten sit in IP102 label 56 and all decode cleanly, since Pillow dispatches on
content, not extension. Phase 5's loader must therefore convert to RGB
explicitly — an untouched RGBA image would give the CNN a fourth input channel —
and must not switch to an extension-based reader. The files are not renamed or
re-encoded; the filenames are pinned by tests.

**New finding — exact-content cross-split leakage in `full102`.** Measured by
SHA-256 over file bytes.

| Scope | Duplicate groups | Within-split | Cross-split | Label conflicts |
| --- | --- | --- | --- | --- |
| rice10 | 1 | 1 | **0** | 0 |
| full102 | 5 | 3 | **2** | 0 |

`rice10` has **zero** cross-split leakage, so its validation figures are
uncontaminated — a point in favour of it as the development scope. `full102` has
two byte-identical train/test pairs (40410/40432 at label 56, 65553/66152 at
label 92). Two images out of 22,619 is ~0.009% of the test set, too small to move
a headline metric, and it is recorded rather than corrected because the official
splits are never modified. Phase 9 reports test metrics with and without them.
No duplicate group carries conflicting labels, so there is no annotation
contradiction among identical files.

This does **not** rule out near-duplicate leakage: byte hashing misses two
visually identical images saved at different JPEG qualities. Perceptual hashing
was not run.

**Dimensions re-measured exhaustively.** `full102` short side below 160 px is
4.0/5.3/5.8% for train/validation/test, and below 224 px is 28.9/31.0/31.3% —
close to Phase 1's sampled estimates. `rice10` is 6.3/8.3/9.6% below 160 px.

**Distributions reconfirmed.** `full102` train imbalance 82.0x (label 72 has 42
training and 7 validation images; label 101 has 3,444). `rice10` imbalance 2.8x.
All classes present in all splits for both scopes, and no filename spans splits.

Three scripts were added: `build_manifests.py` (build, `--check`),
`verify_dataset.py` (fast pre-training gate, no decoding) and `audit_dataset.py`
(the content-level audit). 120 tests were added; the suite is 246 passing, with
`ruff` and `mypy` clean.

### Phase 5 — Data loader and preprocessing (complete)

Built the path from a derived manifest to the tensor the CNN will consume:
`transforms.py` (pixel decisions), `dataset.py` (manifest to tensors),
`loaders.py` (DataLoader assembly and the rules it enforces), plus
`scripts/verify_loader.py` as the Phase 5 gate. No training was run and no
source file was touched.

**Preprocessing decisions**, recorded in [TRAINING.md](TRAINING.md) with the
reasoning: direct 160x160 resize (keeps the whole frame, since aspect ratios
span 0.24-6.04), **bilinear** with antialiasing (most images are *downscaled* —
median short side 250-320 px — so antialiasing matters more than the filter),
ImageNet mean/std as fixed constants rather than pretrained weights, and
unconditional RGB conversion.

**The Phase 4 RGBA risk is closed.** RGB conversion is applied in two
independent places — `load_image` at the decode boundary and the first step of
every pipeline — so bypassing one still cannot produce a four-channel tensor.
All ten real PNG-as-`.jpg` files were decoded and confirmed to yield exactly
`(3, 160, 160)`, under both scopes, and tests pin them by filename.

**Evaluation determinism is proven, not asserted.** `validation` and `test`
share one pipeline containing no random step, and the verification script
applies it twice to real images and compares tensors bit for bit. Two full
passes over the validation loader also produce identical batches. Conversely,
the training pipeline is checked to *actually vary* across eight draws — an
augmentation config that silently stopped randomising would fail.

**Training-only rules are enforced in code, not by convention.** Augmenting an
evaluation split raises; deriving sampler or class weights from an evaluation
split raises; `build_loaders` refuses class weighting when the training split
was not requested, rather than back-filling from validation data. Evaluation
loaders keep official manifest order with `drop_last=False`, so Phase 9 can join
per-image predictions to the manifest by position. `build_loaders` omits `test`
by default — a caller must name it explicitly.

**Loader throughput measured.** Peak ~1,367 img/s on `rice10` train at 8
workers, scaling monotonically from 2 and degrading at 12, which confirms the
configured default. A first measurement suggested 4 workers beat 8; that was an
artefact of a 5-batch warmup too short to amortise Windows spawn startup, and it
reversed cleanly at 10 batches. Data loading remains far from the bottleneck.

**New finding — an explicit-CUDA run now refuses to fall back to CPU.**
`resolve_device(..., allow_cpu_fallback=False)` raises rather than silently
turning an approved GPU run into a multi-day CPU one. This is exercised by
patching CUDA availability, so the branch is tested on this machine *because*
it has a GPU, not skipped for it.

182 tests were added across four files, including 29 that read the real dataset.
The suite is 428 passing, with `ruff` and `mypy` clean.

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
- The `classes.txt` off-by-one (`ip102_label = classes_txt_id - 1`) is applied in
  exactly one place, `farm_pest_ai.data.manifests.read_classes`, and is pinned by
  tests against both synthetic fixtures and the real file.
- Derived manifests carry their scope and class-mapping version; reading one
  under a different scope or an older mapping raises rather than misinterpreting
  the labels.
- `data/processed/<scope>/` keeps `rice10` and `full102` artifacts separate, so
  the two tasks can never overwrite one another.
- Derived manifests preserve official split order and agree with the source
  record for record; `verify_dataset.py` fails if they drift.
- Manifest CSVs use LF endings and are byte-identical on any platform, so the
  build is verifiably idempotent.
- Every decoded image reaches the model as exactly three channels. RGB
  conversion happens at the decode boundary *and* as the first pipeline step,
  and decoding dispatches on file content, never on the extension.
- Validation and test preprocessing contains no random step and is bit-identical
  across repeated application. Asking to augment an evaluation split raises.
- Class weights and sampler weights are derived from the training split only;
  requesting them from an evaluation split raises, and class weighting without
  the training split raises rather than falling back to other data.
- Evaluation loaders preserve official manifest order and never drop a batch, so
  predictions join to the manifest by position.
- `build_loaders` excludes the test split unless it is named explicitly.
- An explicitly requested CUDA device never degrades silently to CPU when
  `allow_cpu_fallback=False`.
- The resolved preprocessing is fingerprinted; any change to size,
  interpolation, normalisation or augmentation changes the fingerprint.

## Open risks

Carried forward from Phase 1, plus items raised in Phase 2.

| # | Risk | Phase to resolve |
| --- | --- | --- |
| 1 | ~~MSYS2 Python shadows official CPython~~ **Closed in Phase 3.** `.venv` built from official CPython 3.12.5. A bare `python` still hits MSYS2, so always invoke `.venv\Scripts\python.exe` | done |
| 2 | ~~Global `site-packages` polluted~~ **Closed in Phase 3.** venv resolves only its own `site-packages` | done |
| 3 | VRAM is contended: 4,091 MiB free measured under desktop load versus 7,054 MiB when idle, of 8,188 MiB total. Training and Ollama must not share the GPU | 8, 14 |
| 4 | ~~Docker GPU passthrough unverified~~ **Closed in Phase 3.** Verified with `nvidia/cuda:12.6.3-base-ubuntu22.04` under both `--gpus all` and `--runtime=nvidia` | done |
| 5 | Images under 160 px on the short side are upscaled. **Quantified in Phase 4**: rice10 6.3/8.3/9.6%, full102 4.0/5.3/5.8% by split. **Phase 5 fixed the policy** (bilinear, antialiased) and confirmed the cohort still yields correct tensors; whether errors concentrate there is still open | 9 |
| 6 | full102 imbalance is 82x; validation has classes with only 7 images, so macro F1 will be noisy. **Reconfirmed exhaustively in Phase 4** | 8 |
| 7 | ~~Content-hash duplicates and cross-split leakage unmeasured~~ **Closed in Phase 4.** rice10 has 0 cross-split groups; full102 has 2 (4 files, ~0.009% of test). Recorded, not corrected | done |
| 8 | Ollama is not installed | 11 |
| 9 | `classes.txt` mixes common names and Latin binomials; taxonomy is preserved, not corrected | 10 |
| 10 | ~~Ten `.jpg` files are really PNG and seven are RGBA~~ **Closed in Phase 5.** RGB conversion is applied at the decode boundary and again as the first pipeline step; all ten files verified to yield `(3, 160, 160)` under both scopes and pinned by tests | done |
| 11 | **New in Phase 4**: near-duplicate leakage is still unmeasured. Byte hashing catches only exact copies, not re-encodes of the same photo. Perceptual hashing was not run | 8, 9 |
| 12 | **New in Phase 5**: aspect ratio is not preserved — evaluation resizes directly to 160x160, distorting images far from 1:1 (source spans 0.24-6.04). The centre-crop alternative is configured but untested | 7 |
| 13 | **New in Phase 5**: augmentation magnitudes are untuned guesses. Phase 5 fixed the mechanism, not the strength | 7 |
| 14 | **New in Phase 5**: training-run reproducibility is conditional on a fixed `runtime.num_workers`. Changing the worker count changes how per-worker RNG streams interleave, so the exact augmentations drawn differ. Evaluation is unaffected | 7, 8 |
| 15 | **New in Phase 5**: normalisation uses ImageNet constants as fixed numbers rather than statistics measured on IP102. Changing them requires bumping `dataset.preprocessing_version` | 7 |

## Rules in force

- One phase at a time; stop and wait for `CONTINUE PHASE <n>`.
- The `ip102_v1.1` directory is read-only. Never rename, move, delete,
  overwrite, re-encode or resplit source data.
- No test set is touched before the model is frozen in Phase 9.
- Approval is required before installing software, pulling images or models,
  running full training, or starting persistent services.
- No Git operations beyond `branch`/`status` without an explicit request.
