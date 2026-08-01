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

Data loading is not the bottleneck: 449.7 img/s single-process warm, well above
what one GPU consumes at this input size.

## Architectures (Phase 6)

Both are built from primitive PyTorch layers in `farm_pest_ai.vision.blocks`. No
`torchvision.models`, prebuilt architecture, pretrained weight or downloaded
checkpoint is imported anywhere.

| | `baseline_cnn` (Model A) | `custom_cnn` (Model B) |
| --- | --- | --- |
| Structure | 3 stages of conv-BN-ReLU x2 + max pool | Strided stem + 4 stages of residual separable blocks |
| Parameters (rice10) | 3,363,530 | **1,435,242** |
| Parameters (full102) | 3,457,382 | 1,470,662 |
| Parameter memory | 12.84 MiB | **5.54 MiB** |

Model B is **2.3x smaller** than the control despite being deeper, because its
`3x3` convolutions are factorised into depthwise plus pointwise pairs. Whether
that translates into a better validation macro F1 is a Phase 7 question; Phase 6
only establishes that both build, train and checkpoint correctly.

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

## Results

_No real experiment has been run. Phase 6 established the pipeline; Phases 7-9
produce the results. The smoke figures below are from capped runs and are
**meaningless as measurements** — they exist only to prove the machinery works._

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
