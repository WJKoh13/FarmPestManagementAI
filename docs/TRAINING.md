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
