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

## Results

_No training has been run. This section is populated in Phases 6-9._
