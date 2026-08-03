# Training protocol

Populated with results from Phase 6 onward. The protocol below is fixed now so
that experiments stay comparable.

## Fixed rules

- Official IP102 train/validation/test assignments; never resplit.
- Fixed, recorded random seeds (`reproducibility.seed`, default 1337).
- Augmentation applied to the **training split only**.
- Validation and test preprocessing is deterministic: no random augmentation.
- Class weights, when used, are computed from the **training split only**.
- Optimizer: AdamW, unless an alternative is explicitly justified in writing.
- Mixed precision where supported.
- Checkpoint resume, plus separate best and last checkpoints.
- Early stopping on **validation macro F1**.
- The fully resolved configuration is stored with every run.
- Structured JSON Lines metrics and logs per run.

## Preprocessing constraints from the Phase 4 audit

- Images must be **converted to RGB explicitly**. Ten `.jpg` files are really
  PNG and seven of those are RGBA; left alone they would present a fourth input
  channel. See [DATASET.md](DATASET.md).
- The loader must dispatch on image **content**, not on the file extension.
- ~4-10% of images are below 160 px on the short side and get upscaled. The
  interpolation choice is a Phase 5 decision, recorded with the preprocessing
  version.

## Preprocessing decisions (Phase 5)

Implemented in `farm_pest_ai.data.transforms`, configured under `preprocessing`
in `configs/base.yaml`, and fingerprinted so a silent change is detectable.

| Decision | Value | Why |
| --- | --- | --- |
| Resize target | 160x160 direct | Keeps the whole frame. Aspect ratios span 0.24-6.04, so a centre crop would discard edges where a small insect may sit |
| Interpolation | `bilinear`, antialiased | Median short side is 250-320 px against a 160 px input, so most images are **downscaled**, where antialiasing matters more than the filter. Bicubic would sharpen JPEG artefacts in the 4-10% that upscale |
| Normalisation | ImageNet mean/std | Fixed constants, **not** pretrained weights. Training-split statistics are a Phase 7 option |
| RGB conversion | Unconditional, at decode | The ten PNG-as-`.jpg` files, seven RGBA. Applied in both `load_image` and the pipeline, so bypassing one still cannot yield four channels |
| Evaluation | One shared deterministic pipeline | `validation` and `test` are built separately but produce identical tensors, so they can never drift apart |

Augmentation is **training-only** and deliberately conservative: random resized
crop (scale 0.6-1.0), horizontal flip 0.5, rotation ±15°, mild colour jitter.
Vertical flip is off — photographs are ground-referenced, so an inverted insect
is not a realistic input. Hue jitter is capped at 0.02 because pest
identification leans on colour. Magnitudes are untuned; Phase 7 fits them
against validation macro F1.

`preprocessing_version` (currently `1.0.0`) plus a 16-hex-character fingerprint
of the resolved pipeline are recorded with every run, so two runs can be proven
to have preprocessed identically.

## Measured loader throughput (Phase 5)

`rice10` training split, batch size 64 at 160x160, RTX 4070 Laptop / Ryzen 7
8845HS, measured after a 10-batch warmup over 50 batches, twice per setting:

| `num_workers` | Throughput |
| --- | --- |
| 2 | ~372 img/s |
| 4 | ~685 img/s |
| 6 | ~1,043 img/s |
| **8 (configured)** | **~1,367 img/s** |
| 12 | ~980 img/s |

Throughput scales to a peak at 8 workers and degrades at 12, confirming the
configured default. **Warmup matters on Windows**: with only a 5-batch warmup,
8 workers appeared *slower* than 4, because spawn startup had not amortised.
Any future benchmark must discard at least ten batches before timing.

This is well above what one GPU consumes at this input size, so Phase 1's
conclusion holds: data loading is not the bottleneck.

## Selection metric

**Primary: validation macro F1.** Chosen over accuracy because both scopes are
imbalanced — severely so for `full102` (82x) — and accuracy would be dominated
by the largest classes.

Secondary metrics recorded every epoch: balanced accuracy, top-1 accuracy,
top-5 accuracy, weighted F1, per-class precision and recall, learning rate,
epoch time, throughput and peak VRAM.

Reported at evaluation time: confusion matrix, calibration, model size,
parameter count, CPU and GPU latency, and peak memory.

## Test-set discipline

The test set is **not** used for architecture, hyperparameter, epoch,
augmentation, threshold or scope selection. It is evaluated once, in Phase 9,
after everything is frozen. No retuning follows.

## Model constraints

Architectures are implemented from primitive PyTorch layers.

Allowed: `nn.Conv2d`, `nn.BatchNorm2d`, pooling, activations, dropout, linear
layers, and manually implemented residual, depthwise-separable and multi-scale
blocks; torchvision transforms and general utilities.

Prohibited: `torchvision.models`, any prebuilt CNN architecture, pretrained
weights, and downloaded checkpoints.

Models output raw logits with no softmax applied inside the model, because the
training losses expect logits.

## Recorded per run

Dataset scope, number of classes, class mapping, class-mapping version,
manifest version, preprocessing version, model configuration, training seed,
environment snapshot (Python, PyTorch, CUDA, cuDNN, GPU) and Git revision.

## Runtime notes

Phase 1 measured only ~4.1 GB of the RTX 4070 Laptop's 8 GB free with a normal
desktop session loaded. Batch size 64 at 160x160 is the starting point; raise it
only after confirming free VRAM. Training and Ollama must never occupy the GPU
simultaneously.

Estimated wall-clock from Phase 1 measurements:

| Scope | Images/epoch | Estimated epoch | Typical run |
| --- | --- | --- | --- |
| rice10 | 4,318 | ~8-20 s | ~15-35 min |
| full102 | 45,095 | ~80-170 s | ~2.5-5.5 h |

**Superseded for full102 by Phase 8 measurement.** The figures above are Phase 1
hardware extrapolations. Two complete 60-epoch runs on the real loader measured:

| Scope | Model | Measured epoch | Measured run |
| --- | --- | --- | --- |
| full102 | `custom_cnn` | ~49 s | **49.4 min** |
| full102 | `baseline_cnn` | ~119 s | **118.6 min** |

So the custom model is ~3x faster than the original estimate's midpoint, while
the baseline lands inside it. Parameter count is a poor proxy for cost here: the
baseline has 20% *fewer* parameters and takes 2.4x longer, because factorised
convolutions cost far fewer FLOPs than a dense `3x3` stack.

Data loading is not the bottleneck: 449.7 img/s single-process warm, well above
what one GPU consumes at this input size.

## Architectures (Phase 6)

Both are built from primitive PyTorch layers in `farm_pest_ai.vision.blocks`. No
`torchvision.models`, prebuilt architecture, pretrained weight or downloaded
checkpoint is imported anywhere.

As shipped in `configs/model_baseline.yaml` and `configs/model_custom.yaml`:

| | `baseline_cnn` (Model A) | `custom_cnn` (Model B) |
| --- | --- | --- |
| Structure | 3 stages of conv-BN-ReLU x2 + max pool | Strided stem + 4 stages of residual separable blocks |
| `stage_channels` | `[64, 128, 256]` | `[64, 128, 256, 384]` |
| Parameters (rice10) | 1,148,874 | 1,435,242 |
| Parameters (full102) | 1,172,518 | 1,470,662 |
| Parameter memory (rice10) | 4.39 MiB | 5.54 MiB |

**Corrected in Phase 7.** Phase 6 recorded the baseline at 3,363,530 parameters
and called Model B "2.3x smaller". Both figures were wrong, and for one reason:
`ModelConfig`'s field defaults are Model B's four-stage widths, so
`ModelConfig(name="baseline_cnn")` — which is how `smoke_train.py` built it —
produces a **four-stage** baseline that no configuration file describes. The
shipped baseline has three stages and 1.15M parameters. Measured against the
architecture an experiment actually trains, the control is the **smaller** of
the two, by 1.25x, not larger by 2.3x.

The gate now builds each architecture from its own configuration file
(`MODEL_CONFIG_FILES` in `scripts/smoke_train.py`) and records the
`stage_channels` it used, so the shape it reports is the shape that gets
trained. `tests/test_shipped_configs.py` pins both the three-stage shape and the
1,148,874 count.

The two are within 1.25x of each other, which is what makes the Phase 7
comparison an architecture result rather than a capacity one. Model B is still
the more parameter-efficient design per unit of depth, since its `3x3`
convolutions are factorised into depthwise plus pointwise pairs — it is deeper
than the control at only 1.25x the parameters. Whether that translates into a
better validation macro F1 is the Phase 7 question.

Model B's blocks add squeeze-and-excitation channel gating and stochastic depth
that ramps linearly from 0 at the first block to `drop_path` at the last. The
ramp matters: dropping early blocks as often as late ones removes the low-level
features the whole network depends on.

The second separable convolution in each residual block ends **linear**, so the
shortcut is added before the final activation and the identity path carries an
unmodified signal.

## Selection metric implementation

Metrics accumulate into one on-device confusion matrix, so accuracy, macro F1,
balanced accuracy and weighted F1 are all guaranteed to describe the same
predictions. Every headline figure is verified against scikit-learn.

**Macro-averaging convention.** A class with no predictions contributes **zero**
to the macro average rather than being dropped. This is the stricter reading: a
model that abandons a rare class should not be rewarded by having that class
silently excluded. `full102` validation has classes with as few as seven images,
so the choice is not hypothetical. A worked case in the tests scores 87.2%
accuracy against 0.45 macro F1, which is exactly the gap the metric exists to
expose.

Balanced accuracy is the one exception: it averages recall over classes actually
**present** in the split, since a class absent from a split has no recall to
average and counting it as zero would penalise the model for the split's
composition rather than its predictions.

## Label-smoothing loss floor

With smoothing `eps` over `C` classes the minimum achievable cross-entropy is
the **entropy of the smoothed target**, not zero:

| `eps` | classes | floor |
| --- | --- | --- |
| 0.1 | 10 | 0.5003 |
| 0.1 | 102 | 0.7799 |
| 0.05 | 10 | 0.2824 |

`label_smoothing_loss_floor` computes this and the tests verify it against a
direct numerical minimisation of the real loss. It matters for the Phase 6
overfit gate: a converged model reached 0.5038 against a floor of 0.5003, and a
"near zero" target would have reported that healthy model as broken.

## Checkpoint provenance

Every checkpoint embeds its scope, class count, class-mapping version, manifest
version, preprocessing version and fingerprint, model configuration, epoch,
seed, environment and Git revision. `load_checkpoint` verifies all of it
**before** copying any weight, so a mismatch cannot leave a half-populated
network behind.

Loading a `rice10` checkpoint under `full102` raises. This is the single most
important guard in the vision layer: a silently mismatched checkpoint does not
crash, it produces confident and wrong pest identifications. Writes are atomic
with an `fsync`, since a checkpoint can represent hours of GPU time.

## Phase 7 experiment entry point

`scripts/train.py` runs every real experiment. `scripts/smoke_train.py` is not
an alternative: it caps batches, runs one epoch and marks its artifacts
`smoke: true` so its numbers can never be read as results. The real entry point
refuses a configuration carrying a `smoke` section, and refuses a trainer that
was handed either batch cap.

Three properties are checked before the first batch, each against the state the
loaders actually produced rather than against configuration:

- **Full splits.** Each dataset's length is compared against the row count of
  its derived manifest on disk. A subset — an accidental `Subset`, a rebuilt
  smaller manifest — aborts the run instead of producing a full-looking
  experiment over a fraction of the data.
- **No test split.** `build_loaders` is called with exactly
  `("train", "validation")`, and the resulting bundle is then asserted to carry
  no other loader or dataset. The script exposes no flag that could name the
  test split. It is evaluated once, in Phase 9.
- **No silent CPU fallback.** An explicit CUDA request that cannot be satisfied
  aborts; `--allow-cpu` is required to opt in.

`--plan` resolves everything above, measures a dozen real batches for a runtime
estimate, prints free VRAM, and exits **without training or writing a
checkpoint**. That is the form to run before requesting approval.

### AMP skipped-step accounting

Every epoch records `optimizer_steps`, `amp_skipped_steps` and
`amp_final_scale` in `metrics.jsonl`, and the run summary carries both the total
and the per-epoch series. The two together are what distinguishes the expected
case from the failure: a handful of skips in the first epoch is scale
calibration, while a total that keeps climbing means batches are contributing no
learning at all while the loss curve still looks plausible. Phase 6 found this
interaction by inspection; it is now a logged quantity rather than something
that has to be noticed.

### Experiment 1: the controlled rice10 architecture comparison

`configs/exp_rice10_protocol_a.yaml` holds one training protocol so the two
architectures can be compared. It is layered **after** a model config and states
the entire `training` section, which overrides whatever that file set:

```
scripts/train.py --config model_baseline.yaml --config exp_rice10_protocol_a.yaml
scripts/train.py --config model_custom.yaml   --config exp_rice10_protocol_a.yaml
```

This is necessary because the shipped model configs are not comparable as
written — `model_baseline.yaml` trains at lr 0.001 for 60 epochs with warmup 3,
smoothing 0.05 and patience 12, while `model_custom.yaml` trains at lr 0.002 for
80 epochs with warmup 5, smoothing 0.1 and patience 15. A win under those
settings could be attributed to any of five differences. Under the shared
protocol the only difference between the two runs is `model.name`, which
`tests/test_shipped_configs.py` verifies by comparing the fully resolved
training configs field by field.

The learning rate is the midpoint of the two shipped values, 0.0015. Neither
architecture is handed its own tuned rate: per-architecture tuning is a separate
experiment, run only after the control is established, because tuning one arm
and not the other reintroduces exactly the confound this file exists to remove.

## Results

### Experiment 1 — rice10 architecture comparison (Phase 7, complete)

Both arms ran the full 4,318 train / 721 validation splits under
`exp_rice10_protocol_a.yaml`, seed 1337, preprocessing fingerprint
`9e75177ab60f96e0`, from clean commit `5f169fc`. The test split was never built.
**All figures below are validation figures.**

> **Corrected in Phase 7.1.** Every F1 in this section was recomputed after a
> defect was found in the shared safe-division helper, which clamped the F1
> denominator `precision + recall` to a minimum of 1 and so under-reported F1
> for every class whose precision and recall summed to less than 1. Both
> **reported** and **corrected** figures are shown. Precision, recall, accuracy,
> balanced accuracy and top-5 were never affected. See
> [the correction](#the-phase-71-macro-f1-correction) below.

| | `baseline_cnn` | `custom_cnn` |
| --- | --- | --- |
| Parameters | 1,148,874 | 1,435,242 |
| **Validation macro F1 (corrected)** | 0.4314 | **0.5913** |
| Validation macro F1 (as reported) | 0.3837 | 0.5731 |
| Validation accuracy | 0.4771 | **0.6103** |
| Weighted F1 (corrected) | 0.4573 | **0.6095** |
| Balanced accuracy | 0.4354 | **0.5885** |
| Top-5 accuracy | 0.8682 | **0.8821** |
| Validation loss | 1.7427 | **1.5989** |
| Best epoch (corrected metric) | 58 | 60 |
| Best epoch (as reported) | 58 | 58 |
| AMP skipped steps | 0 | 0 |
| Optimiser steps | 4,020 | 4,020 |
| Peak VRAM | 1,995 MiB | **858 MiB** |
| Median epoch | 11.1 s | **4.9 s** |

**`custom_cnn` wins by +0.1600 corrected macro F1, a 1.37x improvement**, and it
still wins on **every one of the ten classes** — there is no class where the
control is better, so this is not a trade-off between common and rare pests:

| Class | Support | `baseline_cnn` | `custom_cnn` | Δ |
| --- | --- | --- | --- | --- |
| rice leaf roller | 111 | 0.652 | 0.793 | +0.141 |
| rice leaf caterpillar | 48 | 0.171 | 0.366 | +0.194 |
| asiatic rice borer | 106 | 0.462 | 0.650 | +0.188 |
| yellow rice borer | 50 | 0.506 | 0.580 | +0.074 |
| rice gall midge | 51 | 0.558 | 0.708 | +0.151 |
| brown plant hopper | 83 | 0.378 | 0.414 | +0.036 |
| white backed plant hopper | 90 | 0.407 | 0.516 | +0.109 |
| small brown plant hopper | 56 | 0.225 | 0.481 | +0.256 |
| rice water weevil | 86 | 0.600 | 0.779 | +0.179 |
| rice leafhopper | 40 | 0.354 | 0.625 | +0.271 |

All figures are corrected. Neither model left a class unpredicted. The largest
gains are on the classes the control handled worst — rice leafhopper (+0.271),
small brown plant hopper (+0.256) and rice leaf caterpillar (+0.194) — which is
why the macro average moves further than accuracy does.

**The correction narrowed the margin without changing the verdict.** The
baseline gained +0.0476 macro F1 and the custom model only +0.0182, because the
defect punished weak classes hardest and the baseline had more of them. The
headline gap therefore falls from +0.1894 to +0.1600 (1.49x to 1.37x). Every
qualitative Phase 7 conclusion survives: `custom_cnn` wins overall, wins on all
ten classes, and does so at 2.3x less VRAM.

Plots: `artifacts/plots/rice10_baseline_protocolA/`,
`artifacts/plots/rice10_custom_protocolA/`,
`artifacts/plots/comparison_macro_f1.png` and
`artifacts/plots/comparison_per_class_f1.png`, regenerated with
`python scripts/plot_results.py`.

#### The Phase 7.1 macro F1 correction

The shared `_safe_divide` helper in `farm_pest_ai.vision.metrics` clamped its
denominator to `min=1` before dividing. For **precision and recall** that is a
no-op — their denominators are integer counts, so a positive one is already at
least 1 — which is why those two were always right, and why the existing
scikit-learn agreement tests passed. **F1's denominator is `precision + recall`,
a fraction.** Whenever that sum fell strictly between 0 and 1, the clamp replaced
it with 1 and the class's F1 was divided by too large a number.

The error is therefore one-directional: it could only ever **under-report**, and
it bit hardest exactly where macro F1 is meant to be sensitive — the weak
classes. A class with precision 0.10 and recall 0.20 scored 0.04 instead of
0.133, a 3.3x under-report.

The fix replaces only zero denominators, so the zero-division convention is
unchanged and a positive denominator of any magnitude is divided by as-is.

**The correction required no retraining.** Every run recorded per-class
precision, recall and support alongside the F1 it derived, so each corrected
value is an exact arithmetic recomputation from `metrics.jsonl`:

```bash
python scripts/correct_metrics.py --verify-checkpoints
```

The original artifacts are never modified; the report is written to
`data/reports/phase7_metric_correction.json` with reported and corrected values
side by side.

**One consequence is not cosmetic.** Under the corrected metric `custom_cnn`'s
best epoch moves from 58 to 60, so its saved `best.pt` holds **epoch 58** — the
epoch the defective metric selected. That checkpoint was deliberately **not**
rewritten: it is what the run actually chose, and re-pointing it would fabricate
a selection that never happened. Any claim about "the best model" from that run
must say which metric selected it. The baseline's best epoch is unchanged, so
its `best.pt` is unaffected.

**Risk 17 is resolved: the extra architectural complexity earns its place.**
`custom_cnn` is better on every class, uses 2.3x less peak VRAM and trains 2.3x
faster per epoch, at 1.25x the parameters. Squeeze-and-excitation, residual
connections and stochastic depth are carrying real weight here, not decoration.

**Neither arm early-stopped, and both were still improving at the cap.** Best
corrected macro F1 landed at epoch 58 of 60 for the baseline and epoch **60 of
60** for the custom model, with patience 15 never approached by either. The
cosine schedule drove the learning rate to zero while both models were still
gaining, so **60 epochs is undertrained for this protocol**. The comparison
remains valid — both arms were cut off at the same point under the same schedule
— but the absolute numbers are floors, not ceilings. Extending the budget is a
protocol change that must apply to both arms equally, which is what E1 below
does. That the custom model's corrected best is the *very last* epoch makes the
point more sharply than the reported figures did.

**A 51-minute stall in the baseline run is not a training cost.** Epoch 34 took
3,070 s against a median of 11.1 s; every other epoch was normal, AMP skipped
zero steps, VRAM stayed flat and the loss fell smoothly across it. The signature
is desktop GPU contention or a sleep, not a code fault. Real training time was
~12 min against the 11.4 min plan estimate. **The 63.8 min wall clock and the
371 img/s mean throughput are both distorted by this single epoch and must not be
quoted as benchmarks**; the honest baseline figures are ~11.1 s/epoch and the
plan-measured 404 img/s. The custom run has no such outlier: 6.3 min wall clock,
median 4.9 s/epoch.

The plan estimates were otherwise accurate. Predicted peak VRAM 1,991 / 852 MiB
against measured 1,995 / 858 MiB, within 0.7%.

### Experiment 2 — rice10 screening (Phase 7.2, complete)

Four one-variable-at-a-time experiments on `custom_cnn`, all on `rice10`, all
selected on **corrected** validation macro F1, all at seed 1337. No arm built,
inspected or evaluated the test split.

| | Variable changed | Best macro F1 | Δ vs E0 | Best epoch | Wall clock |
| --- | --- | --- | --- | --- | --- |
| **E0** | *control* — 160px, 60 ep, crop 0.6–1.0 | 0.5913 | — | 60 / 60 | 5.7 min |
| **E1** | longer budget, stretched cosine (100 ep) | 0.5978 | +0.0065 *ns* | 54 / 69 | 6.6 min |
| **E2** | `dataset.image_size` 160 → 224 | **0.6052** | **+0.0138** | 59 / 60 | 7.9 min |
| **E3** | `augmentation.scale` floor 0.6 → 0.8 | 0.5760 | −0.0153 | 44 / 60 | 5.6 min |

Ranking **E2 > E1 > E0 > E3**. *ns* marks a difference below 0.01, which a single
seed cannot separate from noise on a 721-image validation split.

**E0 reproduced the corrected Phase 7 result exactly** — macro F1 0.591340 at
epoch 60, **bit-identical across all 60 epochs** (max per-epoch delta 0.00000000)
and identical per-class F1. This validates both the correction and the pipeline's
end-to-end reproducibility at a fixed worker count, which had never been
demonstrated before (risk 19).

**The differences are small relative to within-run noise.** Epoch-to-epoch range
over each run's last ten epochs is 0.008–0.021, which brackets E2's +0.0138
advantage. Because "best epoch" is the maximum of a noisy series, it flatters
whichever run had the luckiest epoch, so the late-run **mean** is the more
conservative reading:

| | best epoch | last-10 mean | last-10 sd |
| --- | --- | --- | --- |
| E0 | 0.5913 | 0.5832 | 0.0047 |
| E1 | 0.5978 | 0.5708 | 0.0065 |
| E2 | **0.6052** | **0.5993** | 0.0042 |
| E3 | 0.5760 | 0.5686 | 0.0031 |

**E2 is the only change that survives both readings.** Its late-run mean beats E0
by +0.016 and it wins 7 of 10 classes. E1's peak advantage **inverts** under the
mean (0.5708 against E0's 0.5832): its single good epoch at 54 was not
representative. E3 is worse on both readings.

**E1 is not evidence that 60 epochs was too few.** Given 100 epochs it stopped at
**69** on patience 15, having peaked at 54 — so with a stretched cosine the model
converges and then declines rather than being starved of budget. Note this does
not contradict the Phase 7 finding: under the *60-epoch* cosine both arms were
still climbing at the cap, because that schedule anneals the learning rate to
zero by epoch 60. Stretching the schedule changes when the model settles, and
once it can settle, it does so before epoch 60's equivalent point.

**E3 answered its open question in the negative.** Raising the crop floor to 0.8
*hurt* (−0.0153, and worst on both readings), so on rice10 the regularisation
value of aggressive cropping outweighs the risk of cropping the subject out of
frame. Risk 13 is partly addressed: the 0.6 floor is not obviously too
aggressive, and moving it in this direction is not the improvement it looked
like.

Plots for each run, including confusion matrices, are in
`artifacts/plots/<run_id>/` and duplicated into each run directory under
`<run_dir>/plots/`. Comparison figures:
`artifacts/plots/phase72_comparison_macro_f1.{png,svg}` and
`phase72_comparison_per_class_f1.{png,svg}`. Report:
`data/reports/phase72_experiment_comparison.json`.

**Structured, explicable errors.** The E0 confusion matrix shows the residual
errors are not diffuse: the three plant hoppers confuse one another (brown ↔
white-backed ↔ small-brown, 16–25% leakage each way), the two borers swap
(asiatic ↔ yellow, 12–16%), and rice leaf caterpillar leaks 21% into rice leaf
roller. Each pair is a genuinely similar-looking taxon, so this is a model
confusing lookalikes rather than a broken pipeline.

#### Planning figures (measured before the runs)

| | Fingerprint | Predicted peak VRAM | Measured peak VRAM |
| --- | --- | --- | --- |
| E0 | `9e75177ab60f96e0` | 852 MiB | 858 MiB |
| E1 | `9e75177ab60f96e0` | 852 MiB | 858 MiB |
| E2 | `3378a6f0570336b3` | 1,524 MiB | 1,529 MiB |
| E3 | `e07f829a792b4962` | 852 MiB | 858 MiB |

Every prediction landed within 0.7% of the measurement.

Configs: `exp_rice10_protocol_a.yaml` (E0), `exp_rice10_e1_epochs100.yaml`,
`exp_rice10_e2_224.yaml`, `exp_rice10_e3_crop08.yaml`. Each extends the E0
protocol and overrides exactly one field; `tests/test_shipped_configs.py`
resolves each config against E0 and **fails if more than one field differs**, so
the one-variable property is enforced rather than asserted.

**Why each experiment exists**

- **E1 (100 epochs)** — both Phase 7 arms peaked at or beside the cap and neither
  triggered early stopping, so both scores are floors. The cosine schedule is
  defined over `training.epochs`, so raising the cap stretches the decay across
  the whole run rather than appending a flat tail at lr ≈ 0. Length and schedule
  shape are inseparable for a cosine; that is a property of the schedule, not a
  confound being hidden.
- **E2 (224x224)** — Phase 4 measured 28.9% of images below 224 px on the short
  side against 4.0% below 160 px, so a larger input upscales far more data; set
  against that, small subjects may simply not resolve at 160. Batch size stays at
  64 because the measured 1,524 MiB fits well inside the conservative ~4.8 GiB
  free figure (risk 21) — had it not fitted, lowering the batch size would have
  made E2 a two-variable experiment, which is a result to report rather than
  route around.
- **E3 (crop 0.8–1.0)** — risk 13 records that augmentation magnitudes are
  untuned guesses, and a 0.6 area floor can crop a small pest out of frame
  entirely, leaving a label with no evidence. The direction is genuinely open:
  weaker augmentation keeps the subject but regularises less, and the E0 train
  curve ran well above validation.

**Screening protocol.** Each of E1–E3 was compared against E0 on corrected
validation macro F1. A combined **E4** recipe is proposed from whichever changes
helped, and the final recipe is confirmed across **three seeds** — a single-seed
difference on a 721-image validation split is not a result. 256x256 is considered
**only** if 224 shows a meaningful gain over 160; without that evidence it is a
cost with no argued benefit.

The test split was never built by any of these runs.

#### E4 recommendation (proposed before the confirmation ran)

_Superseded by the three-seed result below, which did not confirm it._

**Recommended E4 = E2 alone: 224x224, 60 epochs, crop 0.6–1.0.**

There is nothing to combine. Of the three variables screened, only the input size
helped, so the "combined" recipe is the control plus that single change:

- **Include 224x224.** The only change that improves both the peak (+0.0138) and
  the more conservative late-run mean (+0.016), while winning 7 of 10 classes. It
  costs ~1.8x the VRAM (1,529 vs 858 MiB) and ~39% more wall clock, both of which
  fit comfortably.
- **Exclude the 100-epoch budget.** E1's peak advantage inverts under the
  late-run mean, and it stopped early at 69 epochs having peaked at 54 — evidence
  that the extra budget is not being used, not that it helps.
- **Exclude the 0.8 crop floor.** E3 was worse on every reading.

**The three-seed confirmation is required before this is called a result.** E2's
margin (+0.0138) sits inside the epoch-to-epoch range of the runs it is compared
against (0.008–0.021), so a single seed cannot establish it. The confirmation
should compare E0 and E2 at three seeds each and report the seed spread, not one
number per arm.

**256x256 is not yet justified.** The rule set before screening was to consider
it only on a meaningful 224 gain. +0.0138 clears the stated 0.01 threshold but is
within run noise, so the honest position is that the 160 → 224 gain is
*suggestive but unconfirmed*. Deciding 256 before the three-seed confirmation
would be building on the weaker of two available readings.

### E4 three-seed confirmation (complete) — 224 is NOT confirmed

Six runs, `custom_cnn`, one at a time, seeds 1337 / 2024 / 7. The two arms are
the E0 protocol (160x160) and the E2 protocol (224x224); nothing else differs.
No run built the test split, and all six recorded `amp_skipped_steps 0`.

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

**The single-seed E2 result did not survive.** Seed 1337 reproduced Phase 7.2
exactly — 0.5913 at 160 and 0.6052 at 224, matching E0 and E2 to four decimals,
which independently re-confirms end-to-end reproducibility (risk 19). But the
other two seeds disagree with each other about the sign:

| paired 224 − 160 | best | last-10 mean |
| --- | --- | --- |
| seed 1337 | +0.0138 | +0.0161 |
| seed 2024 | +0.0210 | +0.0229 |
| seed 7 | **−0.0112** | **−0.0079** |

**Seed 7 at 224 (0.5869) scores below all three 160 runs.** The mean advantage
falls from the single-seed +0.0138 to **+0.0079**, which is *below* the 0.01
threshold this project set for "distinguishable from seed noise", and it is
smaller than the 224 arm's own seed spread of 0.0257. Two arms whose ranges
overlap this heavily — 160 spans 0.5913–0.5980, 224 spans 0.5869–0.6126 — are
not separated by three samples each.

**The 224 arm is also markedly less stable**: sd 0.0132 against 0.0038, a 3.5x
larger seed-to-seed variance, and its best epoch wanders (48, 50, 59) where the
160 arm's clusters late (40, 57, 60). A setting that is both unconfirmed on the
mean and noisier per seed is the weaker default, not the stronger one.

**Verdict: 224x224 is not confirmed and is not adopted as the Phase 8 input
size.** The evidence is genuinely ambiguous rather than negative — the mean does
favour 224 on both readings, and 2 of 3 seeds favour it — but "favoured by an
underpowered test" is not the standard this project set before running it.
Retaining 160x160 also costs 1.8x less VRAM (858 vs 1,529 MiB) and ~36% less
wall clock per run, which matters directly for full102.

**256x256 is now firmly excluded.** The rule was to consider it only on a
meaningful 224 gain, and the confirmation removed rather than strengthened that
gain.

**What would settle it.** Three seeds per arm cannot resolve a ~0.008 difference
against a ~0.013 sd; a rough power estimate puts the requirement near 20+ seeds
per arm, which is ~4 hours of rice10 compute for a question whose answer is worth
under one macro-F1 point. The better use of that budget is full102, where the
task is harder and the input-size question may behave differently — 28.9% of
full102 training images have a short side below 224 px against 4.0% below 160 px,
so 224 upscales far more of that data than it does here.

The test split was never built by any of these runs.

### Phase 8 — full102 results (complete)

Both arms under the frozen `exp_full102_protocol_a.yaml`, seed 1337, 160x160,
`class_weighting: none`, 60 epochs. No test split was built.

| | `baseline_cnn` | `custom_cnn` |
| --- | --- | --- |
| Parameters | 1,172,518 | 1,470,662 |
| **Validation macro F1** | 0.4258 | **0.5443** |
| Last-10 mean | 0.4224 | **0.5410** |
| Balanced accuracy | 0.3889 | **0.5231** |
| Accuracy | 0.5436 | **0.5976** |
| Top-5 accuracy | 0.7997 | **0.8201** |
| Best epoch | 60 / 60 (still climbing) | 54 / 60 (plateaued) |
| Classes never predicted | 4 | **0** |
| Peak VRAM | 1,995.7 MiB | **858.3 MiB** |
| Wall clock | 118.6 min | **49.4 min** |

`custom_cnn` wins **86 of 102 classes** by +0.1185 macro F1. Both readings agree
to 0.0001, so the verdict does not depend on peak-versus-mean. Still single-seed
(risk 35), and the baseline's figure is a floor since it had not converged at the
cap (risk 36).

**The advantage concentrates on the rare tail** — mean per-class F1 by validation
support quartile:

| quartile | support | `baseline_cnn` | `custom_cnn` | Δ |
| --- | --- | --- | --- | --- |
| Q1 rarest | 7–26 | 0.2681 | **0.4767** | **+0.2086** |
| Q2 | 26–47 | 0.4842 | 0.5859 | +0.1017 |
| Q3 | 48–80 | 0.4375 | 0.5544 | +0.1169 |
| Q4 largest | 82–573 | 0.5037 | 0.5589 | +0.0552 |

The gap on the rarest quartile is 3.8x the gap on the largest, which is why
balanced accuracy separates the arms (+0.1342) far more than raw accuracy
(+0.0541). Under an unweighted loss the baseline abandoned 4 classes entirely;
the custom model abandoned none.

`custom_cnn` also converges much faster: macro F1 0.40 at epoch **21** against
the baseline's **44**.

**AMP under 704-step epochs.** Both arms recorded exactly **16 skipped steps out
of 42,240 (0.0379%)** under the identical dynamic-scaling policy — near-identical
totals across two different architectures, which is what identifies the skips as
a property of the shared policy rather than of either model. rice10's 67-step
epochs produced zero skips over 4,020 steps; the rates are consistent. The
schedule tracked the theoretical cosine to within 5.7e-07 throughout, so no skip
advanced the learning rate past a step that did not happen.

### Smoke figures (Phase 6, not results)

_The figures below are from capped runs and are **meaningless as
measurements** — they exist only to prove the machinery works._

| Scope | Model | Overfit check (8 images, 100 steps) | One capped epoch |
| --- | --- | --- | --- |
| rice10 | custom_cnn | loss 2.3565 -> 0.5038 (floor 0.5003), batch accuracy 1.00 | val macro F1 0.032, acc 0.154 |
| full102 | custom_cnn | loss 4.6200 -> 0.7937 (floor 0.7799), batch accuracy 1.00 | val macro F1 0.005, acc 0.113 |
| rice10 | baseline_cnn | — | val macro F1 0.078, acc 0.242 |

Both scopes drive a small batch to within 0.004 of the theoretical loss floor
and reach 100% accuracy on it, which is what proves gradients reach every
parameter.

Measured training throughput was ~90 img/s on `rice10` at batch size 16 on the
RTX 4070 Laptop, against the ~1,367 img/s the loader sustains — confirming again
that data loading is not the bottleneck.
