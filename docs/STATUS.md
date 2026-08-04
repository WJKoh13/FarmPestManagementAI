# Project status

Experimental branch: `zy_CNN`. Updated at the end of every phase.

| Field | Value |
| --- | --- |
| Current phase completed | **Phase 8.1 Stage 1 complete — E5–E9 all run; no arm beat its control** |
| Next phase | **Awaiting direction** — confirmation, combined recipes and Phase 9 all require explicit approval |
| Phase 8.1 verdict | **All 9 experiments negative.** E5 (TTA/ensembles), E6a/b (lr), E7a/b (MixUp/CutMix), E8 (SupCon), E9a/b (weighting). Best arm −0.0012. E0 and the Phase 8 protocol survive unchanged |
| Scope recommendation | **`rice10`** — on knowledge-base feasibility and rare-class measurement, not on any macro F1 ranking |
| Phase 8 result | `custom_cnn` **0.5443** vs `baseline_cnn` 0.4258 full102 validation macro F1 (+0.1185, single seed) |
| Image review | 5,039 rice10 images queued (train + validation); **0 human decisions entered** |
| Phase 7 result (corrected) | `custom_cnn` **0.5913** vs `baseline_cnn` 0.4314 validation macro F1 |
| Phase 7 result (as reported) | `custom_cnn` 0.5731 vs `baseline_cnn` 0.3837 — under-reported, see Phase 7.1 |
| Phase 7.2 best arm | **E2** (224x224) 0.6052, +0.0138 over E0 — inside run noise, unconfirmed |
| E4 verdict | 224 vs 160 = **+0.0079** over 3 seeds (was +0.0138 at one). Below the 0.01 noise threshold; **160x160 retained** |
| Branch | `zy_CNN` |
| Active default scope | `rice10` (switchable to `full102`) |
| Dependencies installed | **Yes** — `.venv`, base + `train` + `dev`; `app` deferred to Phase 12 |
| Interpreter | Official CPython 3.12.5 (`win-amd64`) in `.venv` |
| PyTorch | `2.13.0+cu126`, CUDA available, cuDNN 91002 |
| Test suite | **906 passed** (760 through Phase 8, plus 146 in Phase 8.1) |
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

### Phase 7.1 — Results integrity and visualisation (complete)

Found and fixed a defect in the metric that every Phase 7 decision was made on,
corrected the recorded results without retraining, and built the plotting entry
point. **No original artifact was modified and no model was retrained.**

**The defect.** `_safe_divide` in `vision/metrics.py` clamped its denominator to
`min=1`. For precision and recall this is a no-op — their denominators are
integer counts, so a positive one is already at least 1. **F1's denominator is
`precision + recall`, a fraction**, so whenever that sum fell strictly between 0
and 1 the clamp replaced it with 1 and the class's F1 was divided by too large a
number. The error is one-directional: it could only **under-report**, and it bit
hardest on the weakest classes, which is precisely where macro F1 is supposed to
be sensitive. A class at precision 0.10 / recall 0.20 scored 0.04 against a true
0.133 — a 3.3x under-report.

**Why the existing tests missed it.** `tests/test_metrics.py` already compared
macro, weighted and per-class F1 against scikit-learn and passed. Every case in
its parameter list happened to produce classes whose precision and recall summed
either to 0 or to at least 1, so the clamp never engaged. This is the useful
lesson: the tests were real, the comparison was against the right oracle, and the
defect still survived because no case exercised the one interval where the two
implementations differ.

**Corrected results.** The correction needed no GPU: every run recorded per-class
precision, recall and support beside the F1 it derived, so each corrected value
is exact arithmetic over `metrics.jsonl`.

| | reported | corrected | Δ | best epoch |
| --- | --- | --- | --- | --- |
| `baseline_cnn` | 0.3837 | **0.4314** | +0.0476 | 58 (unchanged) |
| `custom_cnn` | 0.5731 | **0.5913** | +0.0182 | 58 → **60** |

**The Phase 7 verdict survives; the margin does not.** The baseline gained 2.6x
more than the custom model, because the defect punished weak classes hardest and
the baseline had more of them. The headline gap falls from **+0.1894 to +0.1600**
(1.49x to 1.37x). `custom_cnn` still wins overall and still wins on all ten
classes.

**A checkpoint was deliberately left stale.** Under the corrected metric the
custom run's best epoch moves to 60, so its `best.pt` holds epoch 58 — the epoch
the defective metric chose. Rewriting it would fabricate a selection that never
happened, so it was not rewritten; `correct_metrics.py --verify-checkpoints`
reports the discrepancy instead, and both runs' checkpoints were verified to
carry the right scope, class count and epoch.

**New — `scripts/correct_metrics.py`** writes
`data/reports/phase7_metric_correction.json` with reported and corrected values
side by side, per epoch and per class, plus checkpoint verification.

**New — `scripts/plot_results.py`** renders accuracy, corrected macro F1, loss,
learning rate, per-class F1 and cross-run comparison figures as PNG **and** SVG
under `artifacts/plots/`. Values are plotted raw, never smoothed; the best epoch
and warm-up boundary are marked; accuracy and F1 axes are percentage-formatted
and zero-based; and every figure uses a single y-axis. Where the correction
changed a curve, the reported series is drawn behind it as a dashed ghost, so a
plot shows what was claimed next to what is true rather than quietly replacing
it.

51 tests were added across `test_metrics.py`, `test_results.py` and
`test_plots.py`, including one that reproduces the real epoch-1 confusion pattern
from the custom run and pins both the corrected value and the value it replaces.

### Phase 7.2 — Controlled rice10 experiments (E0–E3 complete)

Four one-variable-at-a-time experiments, run one at a time, seed 1337. No arm
built, inspected or evaluated the test split — verified from each run's summary
by a test, not merely by intent.

| | Variable changed | Best macro F1 | Δ vs E0 | Best epoch | Peak VRAM |
| --- | --- | --- | --- | --- | --- |
| E0 | *control*, corrected | 0.5913 | — | 60 / 60 | 858 MiB |
| E1 | longer budget, stretched cosine | 0.5978 | +0.0065 *ns* | 54 / 69 | 858 MiB |
| E2 | image size 160 → 224 | **0.6052** | **+0.0138** | 59 / 60 | 1,529 MiB |
| E3 | crop scale floor 0.6 → 0.8 | 0.5760 | −0.0153 | 44 / 60 | 858 MiB |

**E0 reproduced the corrected Phase 7 result bit-identically** — 0.591340 at
epoch 60, max per-epoch delta **0.00000000** across all 60 epochs, identical
per-class F1. The gating condition was met, and this also closes the
never-demonstrated end-to-end reproducibility question in risk 19.

**The one-variable property was enforced twice**: a test resolves each config
against E0 and fails if more than one field differs, and each completed run's
summary was re-checked to confirm only the intended field changed. Every VRAM
prediction landed within 0.7% of measurement.

**The differences are small relative to within-run noise.** Each run's last ten
epochs span 0.008–0.021 in macro F1, which brackets E2's +0.0138 margin. Since
"best epoch" is the maximum of a noisy series, the late-run mean is the more
conservative comparison:

| | best epoch | last-10 mean |
| --- | --- | --- |
| E0 | 0.5913 | 0.5832 |
| E1 | 0.5978 | **0.5708** — inverts |
| E2 | **0.6052** | **0.5993** |
| E3 | 0.5760 | 0.5686 |

**Only E2 survives both readings**, winning 7 of 10 classes. E1's advantage
inverts entirely under the mean, and it stopped early at epoch 69 having peaked
at 54 — so with a stretched cosine the model converges and then declines rather
than being starved of budget. E3 was worse on every reading, answering its open
question in the negative: aggressive cropping is doing useful regularisation on
rice10.

**Recommended E4 = E2 alone** (224x224, 60 epochs, crop 0.6–1.0). There is
nothing to combine — only one variable helped. **The three-seed confirmation is
required before this counts as a result**, because E2's margin sits inside the
noise of the runs it is compared against. 256x256 remains unjustified until then.

**A real bug was caught while plotting.** The confusion matrix was first computed
by scoring each checkpoint through the *ambient* configuration, so E2 — trained
at 224x224 — was scored through a 160x160 pipeline. It loaded without complaint,
because `strict_preprocessing` defaults off, and produced a plausible but wrong
matrix. `RunResults.preprocessing_config()` now rebuilds each run's own
preprocessing and passes `strict_preprocessing=True`, so a mismatch raises
instead. All five runs' rebuilt fingerprints were verified against their
checkpoints, and a test pins the round-trip.

**New — `scripts/compare_experiments.py`** ranks the arms on corrected macro F1,
marks differences below 0.01 as noise, and writes
`data/reports/phase72_experiment_comparison.json` plus the comparison figures.

Confusion matrices are now produced for every run (`--confusion`), and figures
are duplicated into each run directory (`--in-run-dir`). The E0 matrix shows the
residual errors are structured rather than diffuse: the three plant hoppers
confuse one another (16–25% each way), the two borers swap (12–16%), and rice
leaf caterpillar leaks 21% into rice leaf roller — all genuinely similar-looking
taxa.

### Phase 7.3 — Image-quality review (complete)

Built a read-only audit that **proposes and never decides**. `ip102_v1.1` is
opened read-only; two tests verify that measuring an image and building a contact
sheet leave the source file byte-identical. The test split cannot be reviewed —
`--split` does not offer it and the script re-checks.

**Only `blurry` and `low_resolution` are asserted automatically**, since only
those are measurable from pixels. The other eight categories need human
judgement and are at most *suggested*; `reviewer_decision` and `reviewer_notes`
ship empty, and reading back a manifest rejects any decision outside the ten
categories.

**Complete coverage of both reviewable splits** — 5,039 images (4,318 train +
721 validation), with E0 `custom_cnn` predictions. The test split was not
reviewed and cannot be.

| | train | validation |
| --- | --- | --- |
| `low_resolution` | 273 — 6.3% | 60 — 8.3% |
| `blurry` | 96 — 2.2% | 24 — 3.3% |
| `ambiguous` | 38 — 0.9% | 46 — 6.4% |
| `suspected_mislabel` | 238 — 5.5% | 220 — **30.5%** |

Both `low_resolution` figures **independently reproduce Phase 4's exhaustive
measurements exactly** (6.3% / 8.3%).

**The `suspected_mislabel` split difference proves the flag tracks the model, not
the labels**: 5.5% on train against 30.5% on validation, differing only in that
the model was fitted on one. Both are essentially the model's error rate on that
split, which is exactly why the category is a queue and never an action.

**New finding — the quality flags identify *easy* images, not hard ones.** On
held-out validation, blur-flagged images score 0.708 against 0.604 for the rest,
and low-resolution images 0.700 against 0.599. The contact sheets explain it:
many blur-flagged images are perfectly sharp but have a smooth, low-texture
subject on a plain background — the classic variance-of-Laplacian false positive
— and that same plain-close-up cohort is easy to classify. Two consequences:
the blur threshold is not validated and the flag reads closer to "low texture"
than "out of focus" (risk 29), and **risk 5's question about the sub-160 px
cohort is answered negatively for rice10** — errors do not concentrate there —
though the result is confounded and covers one scope.

Contact sheets confirm the taxonomy is needed: the splits visibly contain
illustration plates, multi-panel composites, watermarks and QR codes,
symptom-only frames and tiny subjects.

**No curated manifest was created and no review decision was made.** Any curated
split would go to a new versioned directory under
`data/processed/<scope>/curated/<version>/`, leaving the official manifests
byte-identical.

**New — review manifests are protected from casual overwrite.** The manifest is
the one artifact a human writes into by hand, so the script refuses to replace
one that carries reviewer decisions, or that holds more rows than the current run
would write. Both guards come from real incidents in this phase: a `--limit 40`
pass silently replaced a complete 721-row review, and the same path would have
destroyed decisions had any been entered.

**Fixed — the train split could not be scored at all at first.** `build_loaders`
gives the train split training semantics: it shuffles, augments, and drops the
short final batch, so 30 of 4,318 images went missing and predictions would not
have lined up with manifest rows. The coverage guard caught it rather than
letting a misaligned join through; the review now builds every split with
evaluation semantics.

50 tests were added across `test_review.py` and `test_shipped_configs.py`.

### E4 — three-seed confirmation of the 224x224 recipe (complete)

Six `custom_cnn` runs, one at a time: the E0 protocol (160x160) and the E2
protocol (224x224) at seeds 1337, 2024 and 7. **No new code, config or test was
needed** — `cli.py` already exposes `--seed`, so E4 is a protocol over the
existing configs rather than an implementation task. No source file was modified.

**Result: 224x224 is NOT confirmed, and 160x160 is retained.**

| size | seed | best epoch | best macro F1 | last-10 mean |
| --- | --- | --- | --- | --- |
| 160 | 1337 | 60 | 0.5913 | 0.5832 |
| 160 | 2024 | 40 (stopped 55) | 0.5916 | 0.5790 |
| 160 | 7 | 57 | 0.5980 | 0.5892 |
| 224 | 1337 | 59 | 0.6052 | 0.5993 |
| 224 | 2024 | 50 | **0.6126** | 0.6018 |
| 224 | 7 | 48 | **0.5869** | 0.5814 |

| arm | best: mean ± sd | range (spread) | last-10: mean ± sd |
| --- | --- | --- | --- |
| 160 | 0.5936 ± 0.0038 | 0.5913–0.5980 (0.0067) | 0.5838 ± 0.0052 |
| 224 | 0.6015 ± 0.0132 | 0.5869–0.6126 (**0.0257**) | 0.5942 ± 0.0112 |
| **224 − 160** | **+0.0079** | — | **+0.0104** |

**Seed 1337 reproduced Phase 7.2 exactly** — 0.5913 at 160 and 0.6052 at 224,
matching E0 and E2 to four decimals — which re-confirms end-to-end
reproducibility independently of risk 19's original demonstration.

**But the sign is not stable across seeds.** Paired 224 − 160 on best macro F1:
seed 1337 **+0.0138**, seed 2024 **+0.0210**, seed 7 **−0.0112**. Seed 7 at 224
scores below all three 160 runs. The mean advantage drops from the single-seed
+0.0138 to **+0.0079** — below the 0.01 threshold set for "distinguishable from
seed noise" — and is smaller than the 224 arm's own 0.0257 seed spread. The two
arms' ranges overlap heavily.

**The 224 arm is also 3.5x less stable** (sd 0.0132 vs 0.0038), with a wandering
best epoch (48, 50, 59) against the 160 arm's late cluster (40, 57, 60). It is
both unconfirmed on the mean and noisier per seed, so it is the weaker default.
Retaining 160 additionally costs 1.8x less VRAM and ~36% less wall clock, which
carries directly into full102.

**This is what the confirmation was for.** Phase 7.2 explicitly declined to call
E2 a result on one seed, and that caution was correct: the effect shrank by 43%
under replication and reversed on one of three seeds. **256x256 is now firmly
excluded**, since the rule was to consider it only on a meaningful 224 gain.

**Run hygiene verified, not assumed.** All six summaries were re-checked: correct
per-arm preprocessing fingerprint (`9e75177ab60f96e0` at 160,
`3378a6f0570336b3` at 224), splits exactly `['train', 'validation']` with no test
loader, correct seed recorded, identical peak VRAM within each arm (857.6 /
1,529.2 MiB), and **0 AMP skipped steps across all six runs**. Each run's
recorded `macro_f1` was independently recomputed from its per-class precision and
recall: max deviation **0.000000000000**, confirming all six use the corrected
Phase 7.1 formula.

**No GPU contention.** The worst epoch-time outlier across all six runs was 1.8x
the median, against the 277x stall that corrupted Phase 7's baseline timings
(risk 24). Median epoch 5.2 s at 160 and 7.2 s at 224; runs took 5.3–5.9 min and
7.8 min respectively.

Seed 2024 at 160 stopped early at epoch 55 having peaked at 40 — patience 15
firing correctly, the only early stop among the six.

The test split was never built. The suite is 752 passing, unchanged.

### Phase 8 — planning (complete; no training run)

Froze the full102 experiment protocol and ran preflight only. **Nothing was
trained, no checkpoint was written and no full102 run directory exists.**

**New — `configs/exp_full102_protocol_a.yaml`.** The full102 counterpart of the
rice10 E0 protocol, existing for the same reason: layering a model config alone
would let a result be attributed to any of five training differences. It states
the whole `training` section and is layered last.

**The recipe is E0, carried across unchanged.** Resolved field by field, the only
difference from `exp_rice10_protocol_a.yaml` is **`dataset.scope`** — a genuine
one-variable scope change, so any rice10-vs-full102 difference is attributable to
the task rather than to the protocol. Against the earlier ad-hoc full102 probe,
two fields differ: `training.epochs` 80 → 60 and `training.learning_rate`
0.002 → 0.0015, both reverting the probe's inherited `model_custom.yaml` values
to the E0 protocol.

| | value | source |
| --- | --- | --- |
| optimizer | `adamw`, weight decay 0.05 | E0 |
| learning rate | 0.0015 | E0 midpoint |
| scheduler | cosine, warmup 5 | E0 |
| epochs | 60 | E0 |
| batch size | 64 | E0 |
| image size | **160x160** | **E4-retained** — 224's gain was not confirmed, which is not the same as 160 being superior |
| label smoothing | 0.1 | E0 |
| grad clip | 1.0 | E0 |
| early stopping | macro F1 max, patience 15, min delta 0.001 | E0 |
| checkpoint monitor | macro F1, best + last | E0 |
| class weighting | **none** | E0 — see below |
| augmentation | crop 0.6–1.0, flip 0.5, rot 15°, jitter 0.2/0.2/0.2/0.02 | E0 |
| seed | 1337 | stated explicitly |

**No imbalance mitigation was introduced, deliberately.** full102's train
imbalance is 82x against rice10's 2.8x, but `class_weighting` stays `none`
because no mitigation is specified in the Phase 8 protocol and adding one would
confound the scope comparison with a second change. Imbalance is answered where
the documents say it is answered — in the *metric*: macro F1 averages over every
class including ones the model never predicts, so ignoring the tail is already
penalised. Weighting, resampling and focal loss are each a separate
one-variable experiment for after this control. A test pins the choice as
deliberate rather than defaulted.

**Preflight, both arms** (seed 1337, AMP on, batch 64, 160x160):

| Arm | Parameters | s/step | img/s | Peak step VRAM | Estimated |
| --- | --- | --- | --- | --- | --- |
| `custom_cnn` | 1,470,662 | 0.052 | 1,223 | 852 MiB | ~39 min / 60 epochs |
| `baseline_cnn` | 1,172,518 | 0.157 | 408 | 1,992 MiB | ~118 min / 60 epochs |

Both peak VRAM figures match the rice10 measurements almost exactly (852 and
1,992 MiB), which is expected: peak step VRAM depends on batch shape and model,
not on the number of classes. The full102 epoch is 10.4x larger than rice10's —
704 steps against 67, ~42,240 optimiser steps over 60 epochs.

**Verified, not assumed.** `verify_dataset.py --scope full102` passed
(45,095 / 7,508 / 22,619, class mapping 1.0.0). Both arms resolve to **identical
training configs** and differ only in `model.name`. Both plan reports record
`splits built = ['train', 'validation']` — **no test loader was constructed** —
with full manifest coverage, and both carry preprocessing fingerprint
`9e75177ab60f96e0`, matching the rice10 160x160 runs.

Derived manifest SHA-256 (first 32 hex): train `c6ce8e400e1a6bcc8f3af212db6e15b5`,
validation `d8c00ea2d3ae1f5bfd38d93663ac53e6`, test
`0519659ddc6f1662b850eddea37dbec8` (recorded for provenance; the test manifest
was hashed, never loaded).

8 tests were added to `test_shipped_configs.py`. The suite is **760 passing**,
with `ruff` and `mypy` clean.

### Phase 8 — full102 training (complete)

Both arms ran under the frozen `exp_full102_protocol_a.yaml`, seed 1337, one at a
time. The protocol was not changed between arms. **No test split was built.**

**Result: `custom_cnn` beats `baseline_cnn` by +0.1185 validation macro F1
(0.5443 vs 0.4258) on full102, winning 86 of 102 classes.**

| metric | `baseline_cnn` | `custom_cnn` | Δ |
| --- | --- | --- | --- |
| Parameters | 1,172,518 | 1,470,662 | |
| **Best macro F1** | 0.4258 | **0.5443** | **+0.1185** |
| Last-10 mean | 0.4224 | **0.5410** | +0.1186 |
| Best epoch | 60 / 60 | 54 / 60 | |
| Accuracy | 0.5436 | **0.5976** | +0.0541 |
| Balanced accuracy | 0.3889 | **0.5231** | +0.1342 |
| Weighted F1 | 0.5203 | **0.5935** | +0.0732 |
| Top-5 accuracy | 0.7997 | **0.8201** | +0.0204 |
| Wall clock | 118.6 min | **49.4 min** | 2.40x faster |
| Peak VRAM | 1,995.7 MiB | **858.3 MiB** | 2.33x less |
| Classes never predicted | **4** | **0** | |
| Zero-F1 classes | 7 | 1 | |

**Both readings agree** — peak and last-10 mean differ by 0.0001 — so unlike the
E4 comparison this verdict does not depend on which is used. At +0.1185 the gap
is **5.9x the 0.02 provisional threshold**, but it remains **single-seed
evidence**: no full102 seed replication was run, and E4 showed a single-seed
margin shrinking under replication. The direction is not in doubt; the exact
magnitude is.

**The margin is larger on the harder task.** rice10 gave +0.1600 corrected; here
+0.1185 on a 102-way problem where both models score far lower absolutely. The
gap widens most on the metrics that reward handling the tail: balanced accuracy
+0.1342 against accuracy +0.0541.

**Imbalance is where the architectures separate**, by validation support quartile:

| quartile | support | `baseline_cnn` | `custom_cnn` | Δ |
| --- | --- | --- | --- | --- |
| Q1 rarest | 7–26 | 0.2681 | **0.4767** | **+0.2086** |
| Q2 | 26–47 | 0.4842 | 0.5859 | +0.1017 |
| Q3 | 48–80 | 0.4375 | 0.5544 | +0.1169 |
| Q4 largest | 82–573 | 0.5037 | 0.5589 | +0.0552 |

The custom model's advantage on the rarest quartile is **3.8x its advantage on
the largest**. The baseline left **4 classes entirely unpredicted** and 7 at
F1 0; the custom model left **0 unpredicted** and 1 at F1 0.

**Risk 6 is milder than feared, for `custom_cnn`.** An 82x imbalance with
`class_weighting: none` might have collapsed the tail. Instead label 72 — the
rarest training class at 42 images, 7 in validation — scored **F1 0.7692**,
essentially matching label 101 (3,444 training images) at 0.7677. The rare
quartile is weaker but by ~0.08, not catastrophically. This does **not** retire
the risk: 7-image validation classes make those per-class figures very noisy, and
the baseline *did* fail on the tail.

**Convergence.** `custom_cnn` reached macro F1 0.40 at **epoch 21**;
`baseline_cnn` needed **epoch 44**. The custom arm peaked at 54 and plateaued
(last-10 range 0.0112); the baseline was still improving at the 60-epoch cap
(range 0.0068, best at epoch 60), so **its figure is a floor**. Extending the
budget is a protocol change applying to both arms, so it was not done — the same
reasoning as Phase 7.

#### AMP skip histories (the shared dynamic-scaling policy)

Both arms recorded **exactly 16 skips out of 42,240 attempted steps —
0.0379%** — under the identical frozen AMP policy. Step accounting is exact for
both: 42,224 taken + 16 skipped = 704 x 60.

| | `custom_cnn` | `baseline_cnn` |
| --- | --- | --- |
| Skip epochs | 15, 18, 21, 24, 27, 30, 33, 35, 38, 39, 45, 48, 50, 53, 56, 59 | 13, 16, 19, 22, 23, 29, 32, 35, 37, 38, 41, 45, 51, 53, 59 |
| Max in one epoch | 1 | 2 |
| Scale min / final | 4,096 / 65,536 | 4,096 / 16,384 |

The near-identical totals across two different architectures are the evidence
that these skips are a property of the **shared scaling policy under 704-step
epochs**, not of either model. They are not treated as a protocol difference.

**All five user-defined stop thresholds were clear on both runs**: no non-finite
loss, gradient or metric; no three consecutive skip epochs (worst was two); never
5+ skips in one epoch; cumulative rate 2.6x under the 0.1% limit; no scale
collapse. The scheduler tracked the theoretical cosine to within **5.7e-07**
across all 55 post-warmup epochs of the custom run, so no skip advanced the
learning rate past a step that did not happen.

**Why rice10 saw zero skips and full102 sees 16**: the full102 epoch is 704 steps
against rice10's 67, so ~10.5x the opportunities to overflow per epoch, at a
102-class loss with larger gradients. The rates are consistent, not divergent.

#### The interrupted first attempt

The first `baseline_cnn` run was killed at epoch 39/60 by an **unexpected laptop
restart**. It was **restarted from scratch rather than resumed**, so both arms
are uninterrupted and strictly comparable; resuming would have restarted the
dataloader RNG stream mid-run (risk 14) and made the two arms asymmetric.

**The atomic-write design held perfectly**: 39 epochs of `metrics.jsonl` with
**zero corrupt lines**, contiguous numbering, all values finite, and both
`best.pt` and `last.pt` loading cleanly with optimizer state. The partial run is
preserved at `artifacts/checkpoints/full102_baseline_protocolA_interrupted_ep39/`
and excluded from the comparison.

**It also produced an unplanned reproducibility result.** The restarted run
reproduced the interrupted one **bit-identically** through the first 30+ epochs —
macro F1 0.1484 / 0.2522 / 0.3215 at epochs 10 / 20 / 30, matching to four
decimals, with identical AMP skip counts and timings. This extends the risk-19
reproducibility guarantee, previously demonstrated only on rice10, to `full102`
from a cold start.

#### Recording limitation

`metrics.jsonl` omits per-class arrays for `full102` by design
([`training.py:927`](../src/farm_pest_ai/vision/training.py) — 102 classes x 4
arrays x 60 epochs would make the log unreadable). The per-class breakdown is
preserved in the `best.json` checkpoint sidecar, so it is available **for the
best epoch only**, not per epoch as on rice10. Every per-class figure above comes
from that sidecar.

Likewise, AMP skips are recorded per **epoch**, not per step, so a skip's global
step is known only to a 704-step range and the loss scale only at epoch
boundaries. Capturing exact per-step values would require an engine change, which
the frozen protocol forbids mid-phase.

### Phase 8 — validation figures and scope selection (complete)

**New — `scripts/plot_phase8.py`** renders ten figures as PNG **and** SVG under
`artifacts/plots/phase8/` (20 files), plus
`data/reports/phase8_validation_figures.json`. Every figure carries the footer
*"full102 validation split only — the test split is unused (Phase 9)"*. No
training artifact was overwritten and neither model was retrained.

| Figure | File stem |
| --- | --- |
| Train + validation loss curves | `phase8_loss_curves` |
| Validation macro F1 and accuracy | `phase8_validation_metrics` |
| Baseline vs custom metric comparison | `phase8_metric_comparison` |
| Per-class F1 ordered by support | `phase8_per_class_f1_by_support` |
| Support vs F1 scatter (log x) | `phase8_support_vs_f1` |
| Performance by support quartile | `phase8_support_quartiles` |
| Normalised confusion, each arm | `phase8_confusion_full102_{baseline,custom}_protocolA` |
| Most frequent confusion pairs, each arm | `phase8_top_confusions_full102_{baseline,custom}_protocolA` |

**No rice10-vs-full102 macro F1 ranking chart was produced**, per
[EVALUATION.md](EVALUATION.md): they are different classification tasks.

**Rescoring was verified against the recorded metrics.** Each checkpoint was
scored through **its own** recorded preprocessing with `strict_preprocessing=True`
via `confusion_matrix_for_run`, which also refuses any split but train or
validation. Both matrices summed to exactly **7,508** images, and rescored
accuracy reproduced the recorded figure — baseline exactly (0.5436), custom
within 0.000133 (one image, cuDNN forward-pass non-determinism).

**Fixed while building the figures — class names were silently dropped.** A
`from scripts.plot_results import ...` inside a `try/except` resolved only when
the repository root happened to be on `sys.path`, so the first render labelled
confusions `24 → 70` instead of `aphids → miridae`. The import now resolves by
file path and failures raise rather than degrading quietly.

#### Scope decision: **rice10 is the recommended final scope**

Applying the [EVALUATION.md](EVALUATION.md) criteria. The two scopes' macro F1
are **not** compared; each criterion is judged on its own terms.

**1. Application coverage — favours full102.** `rice10` covers ten rice pests;
`full102` covers all 102 IP102 classes across many crops. For an assistant whose
users photograph whatever is eating their crop, 102 classes is the more honest
input domain: a rice10 model has no way to say "this is not a rice pest" and will
map any beetle onto one of ten rice labels. This is the strongest argument
against the recommendation and is recorded as such.

**2. Validation reliability — favours rice10, but less than expected.** Both
tasks are around 60% top-1 (`rice10` 0.6075, `full102` 0.5978 — reported
side by side as separate measurements, not as a ranking). The revealing figures
are per-class: of 102 full102 classes only **16 reach F1 0.70** and **5 reach
0.80**, with 62 above 0.50. Errors are **diffuse, not fixable**: the top 15
confusion pairs account for only **11.3%** of the ~3,021 validation errors.
`rice10` also has **zero** cross-split leakage (Phase 4) where full102 has two
contaminated test pairs.

**3. Rare-class behaviour — favours rice10 structurally.** full102's rarest
support quartile (7–26 validation images) averages **F1 0.4767**, and per-class
figures over 7-image classes are extremely noisy — a single image moves recall by
14 points. `rice10`'s 2.8x imbalance produces no comparable tail. Notably
`custom_cnn` left **0 classes unpredicted** on full102 where `baseline_cnn` left
4, so the architecture handles the tail as well as can be expected; the problem
is measurement precision, not model collapse.

**4. Runtime and hardware feasibility — favours neither decisively.** full102
`custom_cnn` trains in 49.4 min at 858 MiB peak, well inside the RTX 4070
Laptop's budget, and inference cost is identical at 160x160 — a 102-way head adds
~35k parameters. Both are deployable offline.

**5. Verified knowledge-base effort — strongly favours rice10.** This is the
decisive practical criterion. [KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md) requires
every record to carry source organization, title, date/version, jurisdiction,
verification status and a local source reference, with **no** guidance inferred
from model memory. That is ~10 curated records for rice10 against **~102** for
full102 — a tenfold research burden on a project that also has to build the API,
knowledge base, LLM layer, Streamlit interface and offline container. A CNN that
identifies a pest the knowledge base cannot advise on returns identification plus
"verified treatment guidance is unavailable", which is safe but not useful.

**6. Uncertainty/abstention — full102's ~59.8% is workable, and this criterion
does *not* favour rice10.** Measured on validation, confidence separates correct
from incorrect predictions well (mean 0.772 vs 0.465):

| threshold | full102 coverage | full102 accuracy | rice10 coverage | rice10 accuracy |
| --- | --- | --- | --- | --- |
| none | 100% | 59.8% | 100% | 60.7% |
| 0.5 | 67.0% | 76.3% | 78.9% | 70.7% |
| 0.7 | 50.3% | **84.6%** | 58.5% | 77.5% |
| 0.9 | 24.8% | **92.7%** | 24.7% | 88.8% |

**full102's abstention curve is better than rice10's at every threshold.** So
59.8% headline accuracy *is* suitable with an abstention policy — at 0.7 the
system answers half the time at ~85% accuracy and defers the rest, which is a
defensible product. This criterion argues **for** full102, and the recommendation
does not rest on it.

**The recommendation.** `rice10`, on criteria 3 and 5 — the rare-class
measurement problem and the ~10x knowledge-curation burden — with criterion 2
supporting. It is the scope this project can actually finish to the standard its
own safety documents demand: ten classes with verified, provenance-carrying
treatment records, honest uncertainty, and no unadvisable predictions.

**What this costs, stated plainly.** rice10 cannot recognise 92 of the 102 pests
in the source data and will confidently mis-map an out-of-domain insect onto a
rice label. That limitation belongs in
[LIMITATIONS.md](LIMITATIONS.md) and in the user interface, not buried here. The
full102 checkpoint is **not discarded**: it is a completed, documented artifact
that supports a later scope widening once knowledge coverage exists, and Phase 9
freezes only the selected scope.

**This is a recommendation, not a decision** — scope selection is the user's
call, and the evidence for full102 (criteria 1 and 6) is real.

**Fixed — a test pinned a scope that Phase 8 legitimately widened.**
`test_compare_runs_reports_both_figures` asserted `row["scope"] == "rice10"` for
every discovered run, contradicting its own comment that later phases would add
run directories. It now asserts the scope is a *known* one; the two scopes' metrics
are still never combined. The suite is **760 passing**, `ruff` and `mypy` clean.

### Phase 8.1 — accuracy and generalization improvements (Stage 1 complete; **all nine experiments negative**)

Authorized by the user between Phases 8 and 9. **Phase 9 remains pending.** No
training run was launched: the phase brief authorizes infrastructure, tests,
validation-only inference and planning, and requires explicit approval before any
full training. The test split was not accessed, constructed, inspected or scored
at any point.

#### E5 — ensembling and test-time augmentation (complete; **negative**)

Inference only, over 15 rice10 arms and 3 full102 arms on the validation split.
No checkpoint was modified and no model was retrained.

**Result: neither TTA nor uniform ensembling is adopted.** Both were measured
properly and both failed to earn their cost.

**Deterministic horizontal-flip TTA does not help, and mostly hurts.** Paired
against the same six rice10 checkpoints:

| arm | single | +hflip | Δ |
| --- | --- | --- | --- |
| 160 seed 1337 | 0.5904 | 0.5980 | **+0.0076** |
| 160 seed 2024 | 0.5916 | 0.5820 | −0.0096 |
| 160 seed 7 | 0.5979 | 0.5914 | −0.0065 |
| 224 seed 1337 | 0.6052 | 0.5980 | −0.0071 |
| 224 seed 2024 | 0.6121 | 0.6125 | +0.0004 |
| 224 seed 7 | 0.5895 | 0.5789 | −0.0106 |
| **mean** | | | **−0.0043 ± 0.0070** |

Positive on **2 of 6**, mean negative, and every individual delta is inside the
±0.01 noise threshold. On full102 the same picture: macro F1 −0.0021 and balanced
accuracy −0.0039, against accuracy +0.0035 — the metric that matters moves the
wrong way. This is a real result rather than a null one: horizontal flip is
usually a safe TTA, and it failing here suggests the training-time flip
augmentation (p=0.5) has already extracted that invariance, leaving nothing for
the inference-time average to add.

**Uniform ensembling does not beat the best single member.**

| ensemble | member mean | best member | ensemble | vs mean | vs best |
| --- | --- | --- | --- | --- | --- |
| 160px x3 | 0.5933 | 0.5979 | 0.6011 | +0.0078 | **+0.0032** |
| 224px x3 | 0.6023 | 0.6121 | 0.5971 | −0.0052 | **−0.0150** |
| 160+224 x6 | 0.5978 | 0.6121 | 0.5929 | −0.0049 | **−0.0192** |

Only the 160px group beat its own member mean, and even there it beat the *best*
member by +0.0032 — a third of the noise threshold. The 224px and combined
ensembles were **worse than their best member**, which is the expected outcome
when uniform weights average a strong member with weaker ones. Tuning weights to
fix this was deliberately not done: fitting weights on the same validation split
that then judges the ensemble measures the split, not the method.

**The combined 160+224 ensemble was built correctly and still lost.** Each member
was scored through its own recorded preprocessing — fingerprints `9e75177ab60f96e0`
at 160 and `3378a6f0570336b3` at 224, verified against each checkpoint under
`strict_preprocessing=True` — and alignment was proven by comparing target
vectors, not assumed from loader order. So its −0.0192 is a genuine result about
uniform ensembling, not an artifact of a broken pipeline.

**`best.pt` is not the corrected-metric optimum, confirmed independently.**
Scoring `rice10_custom_e4_s160_seed1337/best.pt` gave **0.5904** against the
0.5913 recorded at epoch 60, because that checkpoint holds **epoch 58** — the
epoch the defective Phase 7.1 metric selected. This reproduces risk 25 from a
completely different direction and is exactly why the E5 report records each
member's epoch and refuses to assume `best.pt` holds the numerically best epoch.

**The full102 baseline was not ensembled with the custom model.** It is reported
as a standalone reference (macro F1 0.4244) only. Uniformly averaging a 0.4244
member into a 0.5444 one is as likely to hurt as help, and the phase requires a
validation-based reason before evaluating such a pairing at all.

Selective accuracy was recorded at every threshold for every arm and is kept in a
block separate from full-coverage metrics, so the two can never be conflated.

Reports: `data/reports/phase81_e5_rice10.json`, `phase81_e5_full102.json`, each
carrying every member's checkpoint SHA-256, epoch, fingerprint and image size.

#### Stage 1 rice10 screening — E6a, E6b, E7a, E7b, E8 (complete; **all negative**)

Five arms, seed 1337, one conceptual variable each, `custom_cnn` under the E0
protocol as control. Every arm ran to completion with 0 AMP skipped steps. **No
arm beat the control on either reading.**

| arm | variable | macro F1 | Δ | last-10 mean ± sd | accuracy | balanced acc | best ep | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **E0** | *control* | **0.5913** | — | 0.5832 ± 0.0049 | **0.6103** | 0.5885 | 60 | control |
| E6a | lr → 0.0008 | 0.5747 | −0.0166 | 0.5711 ± 0.0024 | 0.5964 | 0.5740 | 57 | unresolved |
| E6b | lr → 0.0030 | 0.5874 | −0.0039 | 0.5792 ± 0.0043 | 0.6047 | 0.5878 | 56 | indistinguishable |
| E7a | MixUp α 0.2 | 0.5734 | −0.0180 | 0.5612 ± 0.0053 | 0.5908 | 0.5669 | 36 | unresolved |
| E7b | CutMix α 1.0 | 0.5144 | **−0.0769** | 0.5050 ± 0.0046 | 0.5437 | 0.5107 | 48 | **worse** |
| E8 | SupCon w 0.1 | 0.5731 | −0.0183 | 0.5634 ± 0.0043 | 0.5895 | 0.5693 | 40 | unresolved |

Both readings agree in every case — no arm's peak and late-run mean disagree in
sign — so unlike E4 these verdicts do not depend on which is used. Only E7b
exceeds the 0.02 seed-noise threshold, and it does so in the **wrong direction**.
The other four sit inside it, which under the phase's own rule means *unresolved*,
not "slightly worse". No candidate advances to confirmation.

**E6: the shared midpoint was already near-optimal.** Both directions lost —
0.0008 by −0.0166 and 0.0030 by −0.0039 — so 0.0015 sits near the top of a flat
region rather than on a slope. **Risk 22 is substantially answered**: the
midpoint chosen to make the Phase 7 architecture comparison controlled was not
costing `custom_cnn` measurable accuracy. E6b is the closest arm to the control
of the five, and its best epoch (56) and last-10 sd match E0's closely.

**Stage 2 (E6c, weight decay) is therefore NOT triggered.** The gate was a
meaningful stage-1 winner; there is none. Running it would test a second knob
against a control whose first knob is already at its optimum.

**E7a MixUp: worse top-1, better calibration.** Macro F1 −0.0180, but it posts
the **lowest validation loss of any arm** (1.5135 against E0's 1.5989) and the
**highest top-5** (0.8946 against 0.8821). It is not failing to learn — it is
learning a smoother, better-calibrated posterior that is less sharp at rank 1.
Its selective curve is the most aggressive of the six: at threshold 0.7 it
answers 37.4% of the split at **87.0%** accuracy, against E0's 58.5% at 77.5%.
For a system whose product answer is an abstention policy, that trade is worth
recording even though the arm loses on the primary metric.

**E7b CutMix is the only clearly harmful arm** at −0.0769. On a dataset where
Phase 7.3's contact sheets showed many images are already tiny-subject or
partial-view frames, pasting a rectangle over the subject appears to destroy
evidence rather than teach robustness to occlusion. The hypothesis that
occlusion-style augmentation suits this data is answered negatively.

**E8 failed on the exact groups it was built for**, which is the most decisive
result of the five. The auxiliary objective exists to separate the documented
confusion groups, and it made **all three worse**:

| confusion group | E0 mean F1 | E8 mean F1 | Δ |
| --- | --- | --- | --- |
| plant hoppers (brown / white-backed / small brown) | 0.4706 | 0.4666 | −0.0040 |
| borers (asiatic / yellow rice) | 0.6152 | 0.5849 | −0.0303 |
| leaf roller vs leaf caterpillar | 0.5793 | 0.5371 | −0.0422 |

A macro-average loss could have hidden a real gain concentrated in six classes;
this shows there was none to hide. Per risk 40 this is evidence about **this
setting** — weight 0.1, temperature 0.07, both published defaults — not about
contrastive learning on this task, and the batch-composition argument that
motivated the method (~7 positives per anchor) remains sound. But the direction
gives no reason to spend further compute tuning it before Phase 9.

**Reading the train-validation gaps.** E0's gap is 0.2738 (train 0.8841 against
validation 0.6103). E6a narrowed it to 0.2187 and E8 to 0.2212 — both by fitting
the training set *less* rather than generalising better, since validation fell in
both cases. **E7a and E7b's gaps are not comparable** and are flagged as such in
the report: their training accuracy is measured on blended images against hard
labels, so E7a's −0.2144 describes the difficulty of the augmented images, not a
model that generalises better than it fits.

| arm | train acc | val acc | gap | comparable? |
| --- | --- | --- | --- | --- |
| E0 | 0.8841 | 0.6103 | 0.2738 | yes |
| E6a | 0.8151 | 0.5964 | 0.2187 | yes |
| E6b | 0.8785 | 0.6047 | 0.2738 | yes |
| E7a | 0.3764 | 0.5908 | −0.2144 | **no — mixed images** |
| E7b | 0.4585 | 0.5437 | −0.0852 | **no — mixed images** |
| E8 | 0.8106 | 0.5895 | 0.2212 | yes |

**Nothing closed the gap while raising validation accuracy**, which was the
phase's actual objective.

**Selective accuracy, all six arms** (coverage / accuracy among answered):

| arm | @0.5 | @0.7 | @0.9 |
| --- | --- | --- | --- |
| E0 | 78.9% / 70.7% | 58.5% / 77.5% | 24.7% / 88.8% |
| E6a | 73.6% / 70.2% | 52.4% / 79.4% | 26.5% / 89.0% |
| E6b | 81.6% / 67.2% | 61.6% / 74.5% | 28.6% / 89.3% |
| E7a | 59.4% / **76.4%** | 37.4% / **87.0%** | 11.2% / **91.4%** |
| E7b | 43.8% / **81.6%** | 23.2% / **92.2%** | 3.2% / 95.7% |
| E8 | 72.4% / 70.5% | 50.5% / 79.9% | 24.3% / 90.9% |

The mixed arms buy selective accuracy with coverage, exactly as expected from a
smoother posterior. E7b at threshold 0.7 answers under a quarter of the split.
**None of these is full-coverage accuracy**, and no arm improved that.

Every arm: 0 classes never predicted, peak VRAM 857.6–876.1 MiB, runtime
5.1–5.9 min, 844–958 img/s. Runs are `rice10_custom_e6a_lr0008`,
`e6b_lr0030`, `e7a_mixup`, `e7b_cutmix`, `e8_supcon`. Report:
`data/reports/phase81_stage1_rice10.json`.

#### Stage 1 full102 screening — E9a, E9b (complete; **no gain, and the trade-off is real**)

Two loss-weighting arms against the unchanged Phase 8 control, seed 1337,
sampling unchanged, weights derived from the training split only.

| arm | weighting | ratio | macro F1 | Δ | last-10 mean ± sd | accuracy | balanced acc | best ep | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **E0-102** | none | — | **0.5443** | — | 0.5410 ± 0.0031 | **0.5976** | 0.5231 | 54 | control |
| E9a | `inverse_sqrt` | 9.06x | 0.5431 | −0.0012 | 0.5384 ± 0.0043 | 0.5840 | **0.5435** | 60 | indistinguishable |
| E9b | `effective` β 0.999 | 23.53x | 0.5286 | −0.0157 | 0.5258 ± 0.0152 | 0.5706 | **0.5444** | 58 | unresolved |

**Neither arm improved macro F1**, the selection metric. E9a is within 0.0012 —
indistinguishable — and E9b is 0.0157 below, inside the 0.02 seed-noise band and
therefore unresolved rather than proven worse.

**But both raised balanced accuracy while lowering raw accuracy, which is exactly
the trade-off the phase said to report as a trade-off rather than an improvement:**

| arm | balanced accuracy | raw accuracy |
| --- | --- | --- |
| E0-102 | 0.5231 | 0.5976 |
| E9a | **+0.0204** (0.5435) | **−0.0136** (0.5840) |
| E9b | **+0.0213** (0.5444) | **−0.0270** (0.5706) |

Weighting is doing what it is supposed to do — spreading performance more evenly
across classes — but macro F1 does not reward it, because the recall gained on
rare classes is paid for in precision lost when the model over-predicts them.

**By validation-support quartile** (Phase 8 grouping; the control's figures
reproduce Phase 8's published 0.4767 / 0.5859 / 0.5544 / 0.5589 exactly, which
confirms the methodology matches):

| arm | Q1 rarest (7–26) | Q2 (26–47) | Q3 (48–80) | Q4 largest (82–573) |
| --- | --- | --- | --- | --- |
| E0-102 | 0.4767 | 0.5859 | 0.5544 | 0.5589 |
| E9a | **+0.0141** | −0.0035 | −0.0003 | **−0.0140** |
| E9b | −0.0094 | −0.0144 | −0.0080 | **−0.0298** |

**E9a is the clean demonstration**: it gains on the rarest quartile and loses an
almost identical amount on the largest, with the middle two unchanged. That is a
redistribution, not an improvement — the model is trading common-class accuracy
for rare-class accuracy at roughly 1:1.

**E9b's stronger 23.53x correction over-corrects**: it loses on *every* quartile
including the rarest, while still posting the highest balanced accuracy. Pushing
past ~9x buys nothing on this task. Since E9a (9.06x) is the gentler arm and the
only one to help the tail at all, the evidence points **away** from stronger
weighting, not toward it — so full inverse-frequency weighting at 82x is now
firmly excluded rather than merely deferred, and no third arm is proposed.

**Neither arm left any class unpredicted** (control also 0), so the risk-6 tail
collapse remains absent with or without weighting.

**Selective accuracy** (coverage / accuracy among answered):

| arm | @0.5 | @0.7 | @0.9 |
| --- | --- | --- | --- |
| E0-102 | 67.0% / 76.3% | 50.3% / **84.6%** | 24.8% / **92.7%** |
| E9a | 57.0% / 79.3% | 36.6% / 85.8% | 13.0% / 89.6% |
| E9b | 47.4% / **81.6%** | 25.8% / 86.1% | 8.8% / 91.0% |

The weighted arms answer less often for a similar answered-accuracy, and at
threshold 0.9 both are *worse* than the control on both axes. **The control
remains the best abstention policy**, which matters because that policy is the
project's realistic product answer.

Both arms: peak VRAM 858.3 MiB, ~49–50 min, AMP skipped steps 17 (E9a) and 14
(E9b) out of 42,240 — consistent with the control's 16 and with the shared
scaling policy rather than with weighting. E9a peaked at epoch 60, still
improving at the cap, so its figure is a floor.

#### The E9b crash and restart

E9b's first attempt was lost at **epoch 28/60** to a machine freeze and crash —
the **third** long-run loss on this hardware (risk 24). The mitigation held
again: 28 epochs of `metrics.jsonl` with **zero corrupt lines**, and both
`best.pt` and `last.pt` loading cleanly with correct provenance.

It was **restarted from scratch rather than resumed**, following the Phase 8
precedent: resuming restarts the dataloader RNG stream mid-run (risk 14), which
would have made E9b non-comparable with E9a and the control.

**The restart reproduced the lost run bit-identically** — max |delta| of
**0.000000000000** in validation macro F1 across the overlapping epochs. The
partial run is preserved at
`artifacts/checkpoints/full102_custom_e9b_effective_interrupted_ep28/` and
excluded from the comparison. This is the third independent confirmation of
end-to-end reproducibility, and the second on `full102`.

#### Stage 1 verdict

**Seven arms, zero improvements.** Nothing advances to confirmation, and no
combined recipe is proposed — combining arms that individually failed has no
evidential basis.

| scope | arms | best result |
| --- | --- | --- |
| rice10 | E6a, E6b, E7a, E7b, E8 | E6b at −0.0039 (indistinguishable) |
| full102 | E9a, E9b | E9a at −0.0012 (indistinguishable) |

Combined with E5's negative inference-time result, **every one of the nine
Phase 8.1 experiments failed to beat its control.** The E0 recipe and the Phase 8
full102 protocol both survive unchanged, which is itself a substantive finding:
the Phase 7/8 configurations are not obviously improvable by any of the five
standard techniques tried, and the ~0.60 validation accuracy plateau appears to
be a property of the architecture and data rather than of an untuned knob.

**Stage 2 (E6c, weight decay) is not triggered** — its gate was a meaningful
stage-1 learning-rate winner, and both directions lost.

**Run hygiene verified for all seven arms**: splits exactly
`['train', 'validation']` with no test loader, seed 1337, preprocessing
fingerprint `9e75177ab60f96e0`, and exactly the intended variable enabled in
each. Reports: `data/reports/phase81_stage1_rice10.json`,
`phase81_stage1_full102.json`.

#### E6–E9 — the infrastructure these arms run on

**New — `training.mixing`** (E7). Configuration-controlled MixUp and CutMix,
`method: none` by default so every historical config resolves to the pre-E7 path
exactly. Training-only: applying it outside a training pass raises. Metrics are
accumulated against the **original hard labels**, so a mixed run's curves stay
comparable with E0's. **CutMix corrects lambda from the actual clipped box area**
— the box centre is uniform so boxes routinely clip at an edge, and the
uncorrected lambda would supervise a blend the pixels do not show. Draws come
from a generator seeded off the run seed, which a test proves does not disturb
the global RNG stream the dataloader consumes.

**New — `training.fine_grained`** (E8). Supervised contrastive loss on a
projected embedding, `method: none` by default. **Chosen over triplet margin loss
on measured batch composition**: at batch size 64 the real rice10 training
distribution yields ~9.94 of 10 classes per batch and **~7 same-class partners
per anchor**, so SupCon consumes every positive without any sampler change —
which matters, because adding balanced sampling would make E8 a two-variable
experiment. On full102 the same batch gives only ~1.68 positives per anchor,
which is why E8 screens on rice10 first.

**The inference contract is unchanged.** `forward()` still returns raw class
logits. `forward_features()` and `forward_logits_and_features()` are separate
paths, and the projection head is **not part of the model's `state_dict`**, so an
E8 checkpoint loads into the ordinary inference path with no special case. A test
pins that.

**E9 needs no new loss.** The existing training-derived `class_weights` already
implements both required schemes. Verified on the real training split:

| scheme | min weight | max weight | ratio | mean |
| --- | --- | --- | --- | --- |
| `inverse_sqrt` (E9a) | 0.2597 | 2.3518 | **9.06x** | 1.0000 |
| `effective` beta 0.999 (E9b) | 0.1814 | 4.2668 | **23.53x** | 1.0000 |
| `inverse` (not screened) | 0.0562 | 4.6120 | 82.00x | 1.0000 |

**`training.class_weighting_beta` was made configurable** at the user's
direction before E9b ran. At the library default of 0.9999 the effective scheme
reaches 69.52x — nearly as aggressive as the full inverse weighting E9 excludes
for being too aggressive — which would have left both arms clustered at one end
with nothing between 9.1x and 82x. At beta 0.999 the arm sits at 23.53x and the
pair genuinely brackets the space. Beta defaults to 0.9999 so configs naming
only the scheme are unchanged, and it is recorded in every run summary.

**Every arm's one-variable property is pinned by a test**, and a test confirms
that all five historical configs still resolve to `mixing: none` and
`fine_grained: none`, so no earlier experiment changes meaning.

**Verification.** 906 tests pass (146 added: 29 E5, 38 E7, 37 E8, 12 analysis,
30 config/loader/dataset), `ruff` and `mypy` clean, and `smoke_train.py` still
passes on the disabled path. All four rice10 recipes were confirmed to actually
train on real data — finite loss, optimiser steps taken, weights updated —
before any plan was quoted.

**Plan measurements**, taken before the runs (RTX 4070 Laptop, batch 64, AMP on,
seed 1337, full splits, **no test loader built**). Measured runtimes came in at
5.1–5.9 min for the rice10 arms and 49.0 / 49.8 min for the full102 arms, against
these estimates:

| arm | scope | s/step | peak step VRAM | estimated |
| --- | --- | --- | --- | --- |
| E6a lr 0.0008 | rice10 | 0.065 | 852.1 MiB | ~4.6 min |
| E6b lr 0.0030 | rice10 | 0.059 | 852.1 MiB | ~4.3 min |
| E6c wd 0.10 | rice10 | 0.056 | 852.1 MiB | ~4.0 min |
| E7a MixUp | rice10 | 0.060 | 852.1 MiB | ~4.3 min |
| E7b CutMix | rice10 | 0.059 | 852.1 MiB | ~4.2 min |
| E8 SupCon | rice10 | 0.055 | 852.1 MiB | ~4.0 min |
| E9a inverse_sqrt | full102 | 0.058 | 852.5 MiB | ~43.3 min |
| E9b effective (beta 0.999) | full102 | 0.055 | 852.5 MiB | ~41.4 min |

Peak VRAM is unchanged at ~852 MiB for every arm, including E8 — the projection
head adds ~50k parameters and the contrastive similarity matrix is 64x64. The
auxiliary objective costs essentially nothing in memory or time.

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
- The F1 denominator `precision + recall` is never clamped. Only a zero
  denominator falls back to zero; a positive fractional denominator divides
  as-is. A regression test exercises precision and recall summing to below 1 —
  the one interval where the old and new implementations disagree — and compares
  macro, weighted and per-class F1 against scikit-learn there.
- Corrected metrics are recomputed from each run's recorded per-class precision
  and recall; the original artifacts are never rewritten, and a run whose best
  epoch moves under correction is flagged rather than having its `best.pt`
  silently re-pointed.
- Smoke-run artifacts are refused by the results reader, so a meaningless number
  cannot be corrected into a result.
- Each Phase 7.2 screening config differs from the E0 control in **exactly one**
  field; a test resolves both and fails if a second variable appears, and each
  completed run's summary is re-checked to have built only train and validation.
- A checkpoint is scored through **its own** recorded preprocessing, rebuilt from
  the run summary, with `strict_preprocessing=True`. Scoring a 224x224 model
  through a 160x160 pipeline raises rather than producing a plausible wrong
  figure, and a test pins each run's rebuilt fingerprint to its checkpoint's.
- The image-quality review opens `ip102_v1.1` read-only. Tests verify that
  measuring an image and building a contact sheet leave the source file's size
  and mtime unchanged.
- The review asserts only the two pixel-measurable categories. Every
  judgement-based category, including `suspected_mislabel`, is a suggestion, and
  `reviewer_decision` ships empty; an unrecognised decision is rejected on read.
- No review path can name the test split, and a curated manifest version
  containing a path separator is refused, so a curated write cannot escape
  `data/processed/<scope>/curated/`.
- The Phase 8 full102 protocol resolves to a training config **identical** to the
  rice10 E0 protocol, differing only in `dataset.scope`. A retuned training field
  fails a test rather than producing a scope result that is partly a protocol
  result, and the confirmed 160x160 input size is pinned.
- The Phase 8 protocol applies **no imbalance mitigation**:
  `class_weighting` is `none` while selection and early stopping run on macro F1.
  The choice is pinned by a test so it reads as deliberate rather than defaulted.
- An ensemble averages **raw logits**, never predicted labels or probabilities,
  and exposes no per-member weight argument, so weights cannot be tuned on the
  validation split that judges the ensemble.
- Ensemble members are refused unless their scope, class count, sample count
  **and target vectors** all agree. Members may differ in preprocessing — a
  160x160 and a 224x224 model combine legitimately — because each is scored
  through its own recorded pipeline; alignment is proven by the shared target
  vector rather than assumed from loader order.
- `rice10` and `full102` members can never be ensembled together; the scope check
  rejects it before any logit is touched.
- E5 records each member's **epoch** and never assumes `best.pt` holds the
  numerically best epoch under the corrected metric.
- Selective (post-abstention) accuracy is reported in a block separate from
  full-coverage metrics, and its entries carry no bare `accuracy` key, so the two
  cannot be conflated by a report reader.
- Batch mixing is **training-only**: applying it outside a training pass raises,
  and evaluation preprocessing is untouched. Metrics are always accumulated
  against the original hard labels.
- CutMix lambda is derived from the **actual clipped box area**, verified by
  measuring the changed pixels, not from the lambda that sized the box.
- Mixing draws come from a generator seeded off the run seed and do not consume
  the global RNG stream, so the dataloader's augmentation draws are unaffected.
- `forward()` returns raw class logits regardless of the auxiliary objective, and
  the projection head is absent from the model's `state_dict`, so a checkpoint
  trained with E8 loads into the ordinary inference path unchanged.
- `training.mixing` and `training.fine_grained` default to `none`, and a test
  confirms every pre-Phase-8.1 config still resolves to the disabled path.
- Every Phase 8.1 arm differs from its control in exactly one conceptual field,
  pinned by tests; the E7 arms are additionally pinned to keep label smoothing at
  the control value, and the E9 arms to leave sampling unchanged.

## Open risks

Carried forward from Phase 1, plus items raised in Phase 2.

| # | Risk | Phase to resolve |
| --- | --- | --- |
| 1 | ~~MSYS2 Python shadows official CPython~~ **Closed in Phase 3.** `.venv` built from official CPython 3.12.5. A bare `python` still hits MSYS2, so always invoke `.venv\Scripts\python.exe` | done |
| 2 | ~~Global `site-packages` polluted~~ **Closed in Phase 3.** venv resolves only its own `site-packages` | done |
| 3 | VRAM is contended: 4,091 MiB free measured under desktop load versus 7,054 MiB when idle, of 8,188 MiB total. Training and Ollama must not share the GPU | 8, 14 |
| 4 | ~~Docker GPU passthrough unverified~~ **Closed in Phase 3.** Verified with `nvidia/cuda:12.6.3-base-ubuntu22.04` under both `--gpus all` and `--runtime=nvidia` | done |
| 5 | Images under 160 px on the short side are upscaled. **Quantified in Phase 4**: rice10 6.3/8.3/9.6%, full102 4.0/5.3/5.8% by split. **Phase 5 fixed the policy** (bilinear, antialiased). **Phase 7.3 measured the consequence on rice10 validation: errors do NOT concentrate there** — that cohort scores 0.700 against 0.599 for normal-resolution images. Confounded, though: low-resolution images here are largely plain close-ups, which are easy for reasons unrelated to size. Open for `full102`, where the task is far harder | 8, 9 |
| 6 | full102 imbalance is 82x; validation has classes with only 7 images, so macro F1 will be noisy. **Reconfirmed exhaustively in Phase 4. Measured in Phase 8 and milder than feared for `custom_cnn`**: rarest-quartile mean F1 0.4767 against 0.5589 for the largest, and label 72 (42 train / 7 validation images) scored 0.7692, matching label 101 (3,444 train) at 0.7677. Zero classes unpredicted. But `baseline_cnn` *did* fail on the tail — 4 classes unpredicted, rarest quartile 0.2681 — so the risk is real and architecture-dependent, and 7-image per-class figures remain very noisy | 9 |
| 7 | ~~Content-hash duplicates and cross-split leakage unmeasured~~ **Closed in Phase 4.** rice10 has 0 cross-split groups; full102 has 2 (4 files, ~0.009% of test). Recorded, not corrected | done |
| 8 | Ollama is not installed | 11 |
| 9 | `classes.txt` mixes common names and Latin binomials; taxonomy is preserved, not corrected | 10 |
| 10 | ~~Ten `.jpg` files are really PNG and seven are RGBA~~ **Closed in Phase 5.** RGB conversion is applied at the decode boundary and again as the first pipeline step; all ten files verified to yield `(3, 160, 160)` under both scopes and pinned by tests | done |
| 11 | **New in Phase 4**: near-duplicate leakage is still unmeasured. Byte hashing catches only exact copies, not re-encodes of the same photo. Perceptual hashing was not run | 8, 9 |
| 12 | **New in Phase 5**: aspect ratio is not preserved — evaluation resizes directly to 160x160, distorting images far from 1:1 (source spans 0.24-6.04). The centre-crop alternative is configured but untested | 7 |
| 13 | **New in Phase 5, partly addressed in Phase 7.2**: augmentation magnitudes are untuned guesses. E3 tested the RandomResizedCrop floor and moving it 0.6 → 0.8 made results *worse* (−0.0153), so the shipped 0.6 is not obviously too aggressive. Rotation, jitter and flip magnitudes remain untuned | 8 |
| 14 | **New in Phase 5**: training-run reproducibility is conditional on a fixed `runtime.num_workers`. Changing the worker count changes how per-worker RNG streams interleave, so the exact augmentations drawn differ. Evaluation is unaffected | 7, 8 |
| 15 | **New in Phase 5**: normalisation uses ImageNet constants as fixed numbers rather than statistics measured on IP102. Changing them requires bumping `dataset.preprocessing_version` | 7 |
| 16 | **New in Phase 6**: no architecture or hyperparameter has been tuned. Learning rate, batch size, epochs, augmentation strength and the two architectures' widths and depths are all untested defaults. The smoke figures are not evidence of anything | 7, 8 |
| 17 | ~~`custom_cnn` may not earn its complexity~~ **Closed in Phase 7.** It beats the control by +0.1894 macro F1 on all ten classes, at 2.3x less VRAM and 2.3x faster per epoch | done |
| 18 | **Partly addressed in Phase 7, exercised for real in Phase 8**: rice10 recorded 0 skips across 4,020 steps, but full102's 704-step epochs produced **16 skips in each arm out of 42,240 attempted steps (0.0379%)**. The guard was verified against real data: the schedule tracked the theoretical cosine to within 5.7e-07 across all 55 post-warmup epochs, and step accounting balanced exactly (42,224 taken + 16 skipped). Skips are recorded per epoch, not per step, so a skip's global step is known only to a 704-step range. There is still no test that *forces* an overflow and asserts the schedule holds back | 9 |
| 19 | ~~Full training reproducibility untested end to end~~ **Closed in Phase 7.2.** E0 rebuilt the corrected Phase 7 custom run bit-identically: max per-epoch macro F1 delta 0.00000000 across all 60 epochs, and identical per-class F1, at fixed `num_workers`. Changing the worker count still changes augmentation draws (risk 14) | done |
| 20 | ~~The runtime estimate's 40% validation ratio is unmeasured~~ **Largely closed in Phase 7.** Peak VRAM predictions came within 0.7% and the per-epoch estimate matched the baseline's real ~11.1 s. Validation is far cheaper than 40% in practice (~0.9 s against ~11 s), so the estimate is conservative | done |
| 23 | **New in Phase 7, refined in Phase 7.2**: both Phase 7 arms peaked at or beside the 60-epoch cap, which looked like undertraining. E1 tested it: given 100 epochs and a stretched cosine the model stopped early at 69 having peaked at 54, and did *not* beat E0 on the late-run mean. So the cap was not the binding constraint — the 60-epoch cosine anneals to zero by epoch 60, which is what made both arms look still-climbing | done |
| 24 | **New in Phase 7; recurred in Phase 8 and AGAIN in Phase 8.1 — three long-run losses on this machine**: Phase 7 saw a 51-minute stall in the baseline's epoch 34; Phase 8's first `baseline_cnn` attempt was killed outright at epoch 39/60 by an unexpected laptop restart. Long runs on this machine are genuinely exposed. **Mitigation proved adequate**: atomic writes left zero corrupt records, and a from-scratch restart reproduced the lost run bit-identically. Phase 8.1's E9b was lost to a freeze/crash at epoch 28/60. The mitigation held a third time: **28 epochs, zero corrupt lines**, both checkpoints loading cleanly, and the from-scratch restart reproduced the lost run **bit-identically** (max |delta| 0.000000000000 over the overlapping epochs). Prefer an idle desktop for multi-hour runs, and expect to restart rather than resume, since resuming breaks RNG-stream continuity (risk 14) | 9 |
| 35 | **New in Phase 8**: the `custom_cnn` vs `baseline_cnn` full102 result is **single-seed**. The +0.1185 margin is 5.9x the 0.02 provisional threshold and both readings agree to 0.0001, so the direction is not in doubt — but E4 demonstrated that a single-seed margin can shrink substantially under replication, so the exact magnitude is provisional. No full102 seed replication has been run (~2.8 h per additional seed pair) | 9 |
| 36 | **New in Phase 8**: `baseline_cnn` was still improving at the 60-epoch cap (best at epoch 60, last-10 range 0.0068), so **its 0.4258 is a floor, not a ceiling**. `custom_cnn` peaked at 54 and plateaued. The comparison is still valid — both arms were cut off identically — but the true gap at convergence is unknown and could be smaller. Extending the budget is a protocol change that must apply to both arms | 9 |
| 21 | **New in Phase 7**: free VRAM measured from inside a CUDA context (7,014 MiB) disagrees with `nvidia-smi` before the process starts (4,838 MiB), because Windows evicts idle desktop allocations under demand. Planning uses the conservative `nvidia-smi` figure; peak step VRAM of 1,991 MiB fits either way, but a larger batch size must be planned against the lower number | 8 |
| 22 | ~~The midpoint learning rate 0.0015 leaves neither arm at its own optimum~~ **Largely closed in Phase 8.1.** E6 screened both directions on `custom_cnn`: 0.0008 lost by 0.0166 and 0.0030 by 0.0039, so 0.0015 sits near the top of a flat region and the midpoint was not costing measurable accuracy. Measured for `custom_cnn` only, at one seed; `baseline_cnn` was not rescreened, but it is not the selected architecture | done |
| 25 | **New in Phase 7.1**: `custom_cnn`'s `best.pt` holds epoch 58, which the defective metric selected; the corrected best is epoch 60. The checkpoint was deliberately not rewritten. Any statement about "the best model" from that run must name the selecting metric, and Phase 9 must not freeze that checkpoint assuming it is the corrected optimum | 8, 9 |
| 26 | **New in Phase 7.1**: the metric defect survived a passing scikit-learn comparison because no test case exercised precision + recall in (0, 1). Other metrics verified the same way — balanced accuracy, top-5, and anything Phase 9 adds — carry the same exposure unless their tests cover the intervals where implementations can diverge | 8, 9 |
| 27 | **New in Phase 7.2**: E1 changes both budget and cosine schedule shape, since the schedule is defined over `training.epochs`. The two are inseparable for a cosine, so an E1 gain cannot be attributed to length alone. **Moot in practice** — E1 did not improve on the late-run mean, so there is no gain to attribute | done |
| 30 | ~~Single-seed screening cannot establish an arm~~ **Closed by E4, and the concern was justified.** Across three seeds the 224 advantage fell from +0.0138 to +0.0079 (below the 0.01 threshold) and *reversed* on seed 7. 224x224 was not adopted. The general lesson stands for every future arm: a single-seed margin under ~0.02 on this 721-image validation split is not a result | done |
| 34 | **New in E4**: three seeds per arm is itself underpowered for the difference being tested — a ~0.008 mean gap against a ~0.013 sd needs roughly 20+ seeds per arm to resolve. E4 can say "not confirmed" but cannot say "no effect"; the mean does favour 224 on both readings and 2 of 3 seeds favour it. Any future close comparison faces the same limit, so prefer variables expected to move the metric by more than a point | 8, 9 |
| 31 | **New in Phase 7.2, reinforced by E4**: "best epoch" is the maximum of a noisy series and systematically favours the luckiest epoch. E1 ranked second on peak but *last* on late-run mean. E4 adds that the 224 arm's best epoch wanders (48, 50, 59) where 160's clusters late (40, 57, 60) — the peak is sampling a flatter, noisier region there. Both readings agreed in E4, which is why its verdict is stable. Any future selection — including Phase 9's freeze — should look at both rather than the peak alone | 8, 9 |
| 32 | **New in Phase 7.2**: scoring a checkpoint through the ambient configuration rather than its own recorded preprocessing produced a wrong E2 confusion matrix that loaded without error. Fixed and tested for this path, but `strict_preprocessing` still defaults to `False` elsewhere, so any other post-hoc evaluation added later carries the same exposure | 8, 9 |
| 28 | **New in Phase 7.3**: no human review pass has been completed. The audit built the queue; the real rates of `diagram_text`, `symptom_only`, `tiny_subject`, `unrelated` and genuine mislabelling are all still unknown, and the 30.5% `suspected_mislabel` figure is a model error rate, not a defect count | 8, 9 |
| 29 | **New in Phase 7.3, confirmed**: the `blurry` flag is a variance-of-Laplacian focus measure at an unvalidated threshold of 100. The false positive is now *demonstrated*, not merely predicted — contact sheets show sharp moth and hopper photographs flagged because the subject is smooth against a plain background, and the flagged cohort scores **better** than average (0.708 vs 0.604 on held-out validation). Read the flag as "low texture", not "out of focus"; its 2.2–3.3% is a queue size, not a blur rate | 8, 9 |
| 33 | **New in Phase 7.3**: `full102` has not been reviewed at all. At 52,603 reviewable images it is ~10x the rice10 work, so every review finding above describes rice10 only | 8 |
| 37 | **New in Phase 8.1 (E5), and CONFIRMED by Stage 1**: the cheap inference-time options were exhausted without a gain, and the training options have now failed too — all seven training arms landed at or below their control, best −0.0012. The plateau is not an untuned knob. **The realistic remaining levers are the abstention policy (already good: full102 answers ~50% at ~85%), more or better data, or a different architecture** — none of which is a Phase 8.1 experiment. Original E5 evidence: Flip-TTA averaged −0.0043 macro F1 across six paired rice10 runs and −0.0021 on full102; no uniform ensemble beat its own best member except by +0.0032, a third of the noise threshold. Every remaining route to higher accuracy therefore requires **training**, which is more expensive and less certain. If E6–E9 also come back inside noise, the honest conclusion is that ~0.60 validation accuracy is close to this architecture's ceiling on this data, and the product answer is the abstention policy rather than a better headline number | 8.1, 9 |
| 38 | **New in Phase 8.1**: E6–E9 are **screened at a single seed (1337)**, and E4 established that a single-seed rice10 margin under ~0.02 can shrink or reverse. Risk 34 additionally showed that three seeds cannot resolve a ~0.008 gap. So the screen can promote a candidate but cannot confirm one, and any arm that lands inside ±0.02 must be treated as unresolved rather than as a small win | 8.1 |
| 39 | ~~E7 and E8 add training code paths that have never completed a full run~~ **Closed in Phase 8.1.** All three completed 60-epoch runs with 0 AMP skipped steps, finite losses throughout and no late divergence; E9a/E9b likewise on full102 (17 and 14 skips, matching the control's 16). The paths are exercised. Original concern: They are unit-tested, and all four recipes were verified to train on real batches with finite loss and moving weights, but a defect that only appears over 60 epochs — a slow divergence, a late NaN, an interaction with the AMP scaler — would not have been caught yet. The first full run of each is also its first integration test | 8.1 |
| 40 | **New in Phase 8.1**: the auxiliary objective's weight (0.1) and temperature (0.07) are **published defaults, not tuned values**, exactly the situation risk 22 flags for the learning rate. A negative E8 result is therefore evidence about *this setting*, not about contrastive learning on this task, and must be reported that way | 8.1 |
| 41 | ~~E9's two schemes differ far less than intended~~ **Closed, then answered.** Beta was made configurable and E9b ran at 23.53x rather than 69.5x. The measurement then made the concern moot in the other direction: the *gentler* arm (E9a, 9.06x) was the only one to help the rare quartile at all, and E9b's stronger correction lost on every quartile. Stronger weighting is not the direction, so full inverse at 82x is now firmly excluded | done |
| 42 | **New in Phase 8.1 Stage 1**: every arm is **single-seed (1337)**, and four of the seven landed inside the ±0.02 band that E4 showed can reverse across seeds. Those four (E6a, E6b, E7a, E8) are recorded as *unresolved*, not as "slightly worse" — the screen can rule out large effects but cannot distinguish a small real loss from noise. Only E7b (−0.0769) is outside the band. Confirming any of them would cost 3 seeds x ~6 min on rice10, which is cheap, but nothing there is promising enough to justify it | 9 |
| 43 | **New in Phase 8.1 Stage 1**: E7a MixUp **improved validation loss (1.5135 vs 1.5989), top-5 (0.8946 vs 0.8821) and selective accuracy (87.0% at threshold 0.7 vs 77.5%) while losing macro F1**. The primary metric and the calibration metrics disagree, so "MixUp did not help" is true only of top-1 discrimination. If the deployed policy is abstention-based — which the scope decision suggests it should be — MixUp may be the better recipe on the metric that actually matters to users, at the cost of coverage (59.4% vs 78.9% at 0.5). This was not the phase's selection criterion and is **not** proposed as a change; it is flagged because Phase 9's uncertainty policy should weigh it | 9 |
| 44 | **New in Phase 8.1 Stage 1**: E9 established that full102 weighting **redistributes rather than improves** — E9a gained +0.0141 on the rarest quartile and lost −0.0140 on the largest, at roughly 1:1, with balanced accuracy +0.0204 and raw accuracy −0.0136. There is no setting that avoids the trade, and the stronger arm was worse everywhere. Whether to take it is a **product decision** about whether rare-pest recall is worth common-pest accuracy, not a metric decision — and macro F1 will not make it | 9 |

## Rules in force

- One phase at a time; stop and wait for `CONTINUE PHASE <n>`.
- The `ip102_v1.1` directory is read-only. Never rename, move, delete,
  overwrite, re-encode or resplit source data.
- No test set is touched before the model is frozen in Phase 9.
- Approval is required before installing software, pulling images or models,
  running full training, or starting persistent services.
- No Git operations beyond `branch`/`status` without an explicit request.
