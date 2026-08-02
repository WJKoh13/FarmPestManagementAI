# Project status

Experimental branch: `zy_CNN`. Updated at the end of every phase.

| Field | Value |
| --- | --- |
| Current phase completed | **Phase 7 — rice10 development experiments** |
| Next phase | Phase 8 — full102 experiment and scope selection |
| Phase 7 result | `custom_cnn` **0.5731** vs `baseline_cnn` 0.3837 validation macro F1 |
| Branch | `zy_CNN` |
| Active default scope | `rice10` (switchable to `full102`) |
| Dependencies installed | **Yes** — `.venv`, base + `train` + `dev`; `app` deferred to Phase 12 |
| Interpreter | Official CPython 3.12.5 (`win-amd64`) in `.venv` |
| PyTorch | `2.13.0+cu126`, CUDA available, cuDNN 91002 |
| Test suite | 636 passed (601 through Phase 6, plus 35 Phase 7 tests) |
| Lint / types | `ruff` clean, `mypy` clean |
| Models (as shipped) | `baseline_cnn` 1.15M params, `custom_cnn` 1.44M params (rice10) |
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

### Phase 6 — Custom CNN and smoke training (complete)

Built the vision layer — `blocks.py` (primitives), `models.py` (the two
architectures), `metrics.py` (scoring), `checkpoints.py` (provenance),
`training.py` (the engine) — plus `scripts/smoke_train.py` as the Phase 6 gate.
No real experiment was run, no hyperparameter was selected, and the test split
was never opened.

**Two architectures, both from primitive layers.** `baseline_cnn` is a plain
conv-BN-ReLU control; `custom_cnn` is residual depthwise-separable with
squeeze-and-excitation and stochastic depth at **1,435,242** parameters.
`torchvision.models` and pretrained weights are imported nowhere.

> **Corrected in Phase 7.** This entry originally reported `baseline_cnn` at
> 3,363,530 parameters and called `custom_cnn` "2.3x smaller". Both were wrong.
> That count came from `ModelConfig` field defaults, which are `custom_cnn`'s
> four-stage widths, not from `model_baseline.yaml`, which ships three stages.
> The shipped baseline is **1,148,874** parameters, so the control is the
> *smaller* of the two by 1.25x. See the Phase 7 entry.

**Gradients are proven to flow, not assumed.** The gate trains one 8-image batch
for 100 steps and requires the loss to collapse. This is the check that
separates "the loop ran" from "the model learned" — a detached tensor, a frozen
parameter or a mis-shaped loss all leave a one-epoch run looking entirely
normal. Both scopes reach 100% accuracy on the batch and land within 0.004 of
the theoretical loss floor, and every trainable parameter is confirmed to
receive a gradient.

**New finding — the label-smoothing loss floor is not zero.** With `eps=0.1`
over 10 classes the minimum achievable cross-entropy is **0.5003**, the entropy
of the smoothed target, rising to **0.7799** at 102 classes. A first version of
the gate compared a converged model against a near-zero target and failed it:
the model had reached 0.5038, essentially exactly the floor, with 100% accuracy
on the batch. `label_smoothing_loss_floor` now computes this, tests verify it
against a direct numerical minimisation of the real loss, and the threshold is
measured above the floor. A fixed threshold could not have served both scopes.

**New finding — AMP calibration silently consumed a short run's budget.**
torch's default `init_scale` of 65536 overflowed on this model's first few
steps; the scaler halves the scale and **skips the optimiser step** each time,
so roughly five of ten capped batches did no training at all. On a full run this
is negligible warmup, but it made the smoke epoch look like it never learned.
Fixed by starting the scaler at 2**12, and the engine now detects a skipped step
and holds the scheduler back, so the learning rate never advances past an
optimiser step that did not happen.

**New finding — capped validation must stride, not truncate.** Evaluation
loaders preserve official manifest order, which is grouped by class, so the
first five validation batches of `rice10` are **entirely class 0**. Macro F1 was
therefore 0.0000 for any model whatsoever, which made the epoch check incapable
of detecting a regression. Validation batches are now sampled with a stride
across the split: on the same 80-image budget, class coverage went from 1 of 10
to 6 of 10. The smoke budget was also raised from 10 to 60 training batches, so
one epoch now produces a measurable score (macro F1 0.032, accuracy 0.154), and
the gate fails outright if the epoch scores zero.

**Checkpoint provenance is enforced before any weight is copied.** Every
checkpoint embeds scope, class count, class-mapping version, manifest and
preprocessing versions, the preprocessing fingerprint, model configuration,
epoch, seed, environment and Git revision. Loading a `rice10` checkpoint under
`full102` raises, as does a stale class mapping, a manifest mismatch or — under
`strict_preprocessing`, which inference defaults to — a changed preprocessing
fingerprint. Writes are atomic with an `fsync`. The JSON sidecar is explicitly
**not** authoritative: a test rewrites one to claim the wrong scope and confirms
the embedded metadata still governs.

**Metrics verified against scikit-learn.** Accuracy, macro F1, weighted F1,
balanced accuracy and per-class F1 all match exactly, including the
zero-division convention. Smoke-run artifacts are marked `smoke: true` and the
run directory is deleted by default, so a meaningless number cannot later be
read as a result.

173 tests were added across five files, 11 of which read the real dataset. The
suite is 601 passing, with `ruff` and `mypy` clean.

### Phase 7 — rice10 development experiments (complete)

Built the real experiment entry point, then ran the controlled comparison it was
built for. Both arms completed from clean commit `5f169fc`. No test split was
built or read, and the source dataset is unchanged (2020 timestamps intact).

**Result: `custom_cnn` beats `baseline_cnn` by +0.1894 validation macro F1
(0.5731 vs 0.3837, a 1.49x improvement), winning on all ten classes.**

| | `baseline_cnn` | `custom_cnn` |
| --- | --- | --- |
| Parameters | 1,148,874 | 1,435,242 |
| **Validation macro F1** | 0.3837 | **0.5731** |
| Validation accuracy | 0.4771 | **0.6075** |
| Balanced accuracy | 0.4354 | **0.5930** |
| Top-5 accuracy | 0.8682 | **0.8849** |
| Best epoch | 58 / 60 | 58 / 60 |
| AMP skipped steps | 0 | 0 |
| Peak VRAM | 1,995 MiB | **858 MiB** |
| Median epoch | 11.1 s | **4.9 s** |

Per-class F1 is in [TRAINING.md](TRAINING.md). The largest gains are on the
classes the control handled worst — rice leafhopper +0.368, small brown plant
hopper +0.312, rice leaf caterpillar +0.239 — which is why the macro average
moves further than accuracy. Neither model left a class unpredicted.

**Risk 17 is resolved.** The custom architecture earns its complexity: better on
every class, 2.3x less peak VRAM, 2.3x faster per epoch, at 1.25x the
parameters.

**New finding — 60 epochs is undertrained for this protocol.** Both arms scored
their best at epoch 58 of 60 and neither approached patience 15, so the cosine
schedule drove the learning rate to zero while both were still improving. The
comparison is still valid — both were cut off identically — but the absolute
numbers are floors rather than ceilings. Extending the budget is a protocol
change that must apply to both arms equally, so it was not done unilaterally.

**New finding — a 51-minute stall that was not a training cost.** The baseline's
epoch 34 took 3,070 s against a median of 11.1 s. Every other epoch was normal,
AMP skipped zero steps, VRAM stayed flat and the loss fell smoothly across it,
so this was desktop GPU contention or a sleep rather than a code fault. It
matters only because it corrupts two derived figures: the 63.8 min wall clock
and the 371 img/s mean throughput both include it and **must not be quoted as
benchmarks**. Real baseline training was ~12 min at ~11.1 s/epoch, against the
11.4 min plan estimate. The custom run has no such outlier (6.3 min, median
4.9 s/epoch).

**The plan estimates held.** Predicted peak VRAM 1,991 / 852 MiB against measured
1,995 / 858 MiB — within 0.7% for both arms.

#### What was built before the runs

**`scripts/train.py` is the entry point for every real experiment.**
`smoke_train.py` is explicitly not an alternative, and the new script refuses to
become one: a configuration carrying a `smoke` section is rejected, and so is a
trainer holding either batch cap or the `smoke` marker. Three properties are
checked before the first batch, each against what the loaders actually produced
rather than against configuration — full split coverage against the manifest row
counts on disk, no test loader or dataset in the bundle, and no silent CPU
fallback. `--plan` resolves everything, measures a dozen real batches and exits
without training or writing a checkpoint.

**The test-split exclusion is now checked twice.** `build_loaders` omits it
unless named, and `assert_no_test_split` re-checks the resulting bundle. The
duplication is deliberate: a leaked test loader produces a perfectly plausible
number that would silently invalidate every decision made after it, and nothing
downstream would notice. The script also exposes no flag that could name the
test split, which a test pins.

**AMP skipped steps are now a logged quantity.** `EpochResult` carries
`optimizer_steps`, `amp_skipped_steps` and `amp_final_scale`; each appears in
`metrics.jsonl` and in the epoch log line, and the run summary keeps both the
run total and the per-epoch series. Keeping the series is what makes the
distinction usable: a handful of skips in epoch 1 is scale calibration, while a
total that keeps climbing means batches contributed no learning at all while the
loss curve still looked plausible. Phase 6 found this interaction by inspection;
it is no longer something that has to be noticed.

**New finding — the recorded `baseline_cnn` size was wrong, and the error
inverted the comparison.** `ModelConfig`'s field defaults are `custom_cnn`'s
four-stage widths `[64, 128, 256, 384]`. `smoke_train.py` built each
architecture as `ModelConfig(name=...)`, so it constructed a **four-stage**
`baseline_cnn` that no configuration file describes and reported it at 3,363,530
parameters. The shipped `model_baseline.yaml` has three stages and **1,148,874**
parameters.

| | Phase 6 reported | As shipped |
| --- | --- | --- |
| `baseline_cnn` rice10 | 3,363,530 (4 stages) | **1,148,874** (3 stages) |
| `baseline_cnn` full102 | 3,457,382 | **1,172,518** |
| `custom_cnn` rice10 | 1,435,242 | 1,435,242 (unchanged) |

So the control is not 2.3x larger than `custom_cnn` — it is **1.25x smaller**.
The Phase 6 full102 figure, 3,457,382, matches neither build and appears to have
been transcribed rather than measured. `smoke_train.py` now builds each
architecture from its own configuration file and records the `stage_channels` it
used, so the shape it reports is the shape that gets trained, and tests pin both
the three-stage shape and the exact count. This matters for Phase 7 rather than
being cosmetic: at 1.25x the two arms are close enough that a difference is
attributable to architecture, whereas at 2.3x it would have been partly a
capacity result.

**The shipped model configs are not a controlled comparison.** Layered on their
own they differ in learning rate (0.001 vs 0.002), epochs (60 vs 80), warmup
(3 vs 5), label smoothing (0.05 vs 0.1) and patience (12 vs 15), so a win could
be attributed to any of five differences. `configs/exp_rice10_protocol_a.yaml`
states the whole `training` section and is layered second, overriding both. The
resolved training configs for the two arms were verified equal field by field;
the only difference is `model.name`. Learning rate is the midpoint, 0.0015 —
neither arm gets its own tuned value, since tuning one and not the other
reintroduces the confound the file exists to remove.

**Both arms planned and measured on the real data** (RTX 4070 Laptop, batch 64,
AMP on, full 4,318 train / 721 validation, 67 + 12 batches, test never built):

| Arm | Parameters | s/step | img/s | Peak step VRAM | Estimated |
| --- | --- | --- | --- | --- | --- |
| `baseline_cnn` | 1,148,874 | 0.158 | 404 | 1,991 MiB | ~11.4 min / 60 epochs |
| `custom_cnn` | 1,435,242 | 0.053 | 1,195 | 852 MiB | ~3.8 min / 60 epochs |

The custom model is **3x faster per step and uses 2.3x less peak VRAM** despite
having 25% more parameters — factorised convolutions cost far fewer FLOPs and
activations than the baseline's dense `3x3` stack. Parameter count is a poor
proxy for either cost here.

35 tests were added across three files (`test_train_script.py`,
`test_train_script_integration.py`, plus additions to `test_shipped_configs.py`),
of which 5 read the real dataset. The suite is 636 passing, with `ruff` and
`mypy` clean.

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
- Models output `num_classes` **raw logits**; no softmax is applied inside a
  model, and tests fail if output rows ever sum to 1.
- `model.num_classes` may not be stated in configuration. It is derived from
  `dataset.scope`, and stating it is a hard error rather than a silent override,
  so the class count has exactly one source of truth.
- A model whose output width disagrees with the scope, or with the data bundle
  it would train on, is refused at construction.
- Every model input is checked to be a 4-D three-channel batch at the model
  boundary, so a four-channel tensor fails loudly rather than deep inside a
  convolution.
- Checkpoints carry scope, class count, class-mapping version, manifest and
  preprocessing versions and fingerprint. Loading one under a different scope or
  a stale class mapping raises, and the check runs before any weight is copied.
- A checkpoint's JSON sidecar is never authoritative; the embedded metadata
  governs, so editing or deleting a sidecar cannot make a bad checkpoint load.
- Checkpoint writes are atomic and fsynced, so an interrupted save leaves the
  previous checkpoint intact rather than a truncated file.
- Class weights reach the loss only from the training split, pre-computed by
  `build_loaders`; the engine never derives them itself.
- Weight decay is never applied to normalisation parameters or biases.
- The learning-rate schedule never advances past an optimiser step that AMP
  skipped.
- Metric aggregation is verified against scikit-learn, and a class the model
  never predicts scores zero in the macro average rather than being excluded.
- Smoke-run checkpoints and metrics are marked `smoke: true`, and the run
  directory is deleted unless `--keep-run` is passed.
- A real experiment runs only through `scripts/train.py`, which refuses a
  configuration carrying a `smoke` section and refuses a trainer holding either
  batch cap or the `smoke` marker.
- A training run uses the entire train and validation splits: each dataset's
  length is compared against its derived manifest's row count on disk, and a
  subset aborts the run.
- A training run never builds a test loader. `build_loaders` is called with
  exactly `("train", "validation")`, the resulting bundle is re-checked to carry
  nothing else, and no CLI flag can name the test split.
- Every epoch records `optimizer_steps`, `amp_skipped_steps` and
  `amp_final_scale`, and the run summary keeps both the total and the per-epoch
  series, so calibration skips are distinguishable from a persistent problem.
- The two architectures in a comparison resolve to identical training configs;
  a difference in any training field fails a test rather than producing an
  uninterpretable result.
- Each architecture's reported shape and parameter count come from its shipped
  configuration file, never from `ModelConfig` field defaults.
- Every completed run records the Git commit it ran from. Both Phase 7 runs
  recorded `5f169fc` with `dirty: false`, so the code that produced each
  checkpoint is recoverable.

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
| 16 | **New in Phase 6**: no architecture or hyperparameter has been tuned. Learning rate, batch size, epochs, augmentation strength and the two architectures' widths and depths are all untested defaults. The smoke figures are not evidence of anything | 7, 8 |
| 17 | ~~`custom_cnn` may not earn its complexity~~ **Closed in Phase 7.** It beats the control by +0.1894 macro F1 on all ten classes, at 2.3x less VRAM and 2.3x faster per epoch | done |
| 18 | **Partly addressed in Phase 7**: AMP skipped steps are now counted per epoch, logged and summarised; both real runs recorded 0 skips across 4,020 optimiser steps, so the Phase 6 calibration fix holds at full scale. There is still no test that forces an overflow and asserts the schedule holds back | 8 |
| 19 | **New in Phase 6**: full training reproducibility is untested end to end. Seeds, worker streams and RNG-state resumption are all implemented and unit-tested, but no two full runs have been compared for bit-identical results. Both Phase 7 runs recorded a clean commit, so a rerun is now at least *possible* | 8 |
| 20 | ~~The runtime estimate's 40% validation ratio is unmeasured~~ **Largely closed in Phase 7.** Peak VRAM predictions came within 0.7% and the per-epoch estimate matched the baseline's real ~11.1 s. Validation is far cheaper than 40% in practice (~0.9 s against ~11 s), so the estimate is conservative | done |
| 23 | **New in Phase 7**: 60 epochs is undertrained for protocol A. Both arms peaked at epoch 58 of 60 and neither triggered early stopping, so both absolute scores are floors. The comparison is unaffected, but any headline rice10 number quoted from these runs understates what the architecture can reach | 8 |
| 24 | **New in Phase 7**: a 51-minute stall in the baseline's epoch 34 (against an 11.1 s median) corrupts that run's wall-clock and mean-throughput figures. Cause is external — desktop GPU contention or sleep — and the model was unaffected. Long full102 runs are more exposed to this; consider running them with the desktop idle | 8 |
| 21 | **New in Phase 7**: free VRAM measured from inside a CUDA context (7,014 MiB) disagrees with `nvidia-smi` before the process starts (4,838 MiB), because Windows evicts idle desktop allocations under demand. Planning uses the conservative `nvidia-smi` figure; peak step VRAM of 1,991 MiB fits either way, but a larger batch size must be planned against the lower number | 8 |
| 22 | **New in Phase 7**: the shared protocol uses the midpoint learning rate 0.0015 for both arms, so neither is at its own optimum. This is deliberate — it is what makes the comparison controlled — but it means the comparison establishes which architecture is better *at a common setting*, not which has the higher achievable ceiling | 7, 8 |

## Rules in force

- One phase at a time; stop and wait for `CONTINUE PHASE <n>`.
- The `ip102_v1.1` directory is read-only. Never rename, move, delete,
  overwrite, re-encode or resplit source data.
- No test set is touched before the model is frozen in Phase 9.
- Approval is required before installing software, pulling images or models,
  running full training, or starting persistent services.
- No Git operations beyond `branch`/`status` without an explicit request.
