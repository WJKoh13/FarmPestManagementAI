# Interim Technical Report — Custom CNN Pest Classification

**Status:** Interim. Development and validation results only. The test split is sealed and has not been evaluated.
**Branch:** `zy_CNN` (experimental; not merged to `main`)
**Latest experiments covered:** E4 (`det_top10`) and E5 (`det_top15`) bounding-box cropping, seed 1337
**Evidence date:** run artifacts timestamped 2026-08-04, git commit `e2d8d25`

---

## 1. Executive summary

This project is building an entirely offline organic farm pest-management assistant. A custom convolutional neural network (CNN) — a network that learns spatial filters over image pixels — is the only pest-identification component; a local language model may later explain retrieved treatment evidence but must never classify an image. The CNN is written from primitive PyTorch layers: `torchvision.models`, prebuilt architectures and pretrained weights are prohibited project-wide and are imported nowhere in the vision package.

The most recent controlled experiment asked one narrow question: **does supplying the network a padded bounding-box crop, instead of the full camera frame, improve validation generalisation?** Four runs form two paired arms — E4A/E4B on the ten-class detection scope `det_top10`, E5A/E5B on the fifteen-class scope `det_top15` — with cropping as the single differing configuration field.

**Cropping improved aggregate validation generalisation in both pairs.** On `det_top10`, validation macro F1 rose from 0.7199 to 0.7627 (**+4.28 percentage points**); on `det_top15`, from 0.6028 to 0.6603 (**+5.75 percentage points**), with balanced accuracy up 3.60 and 4.40 points. Both pairs exceed the project's 0.01 seed-noise threshold on macro F1 and balanced accuracy simultaneously, and the train–validation gap narrowed in both. All four results come from a **single seed**, and the project's own history shows a single-seed margin shrinking 43% and reversing on one of three seeds under replication.

The limitations are material. All four runs overfit, with train-minus-validation macro F1 gaps of 0.2004 to 0.3039. `det_top15` carries a 62.9:1 validation support imbalance with **no class weighting applied**, its rarest class having seven validation images — too few for a stable estimate. The `det_top10` and `det_top15` class sets are **not nested**: only six of ten classes overlap, so E4 and E5 are two valid within-pair crop tests, not a ten-versus-fifteen-class progression. Finally, a crop-trained classifier needs a crop at inference and cannot itself locate a pest.

The most defensible next step is one controlled experiment on `det_top15`: the current cropped arm against one conservative class-weighted loss, all else fixed, replicated across seeds.

---

## 2. Project objective and experiment lineage

### 2.1 Objective

The system identifies an insect pest from a farmer's photograph and returns verified organic or integrated pest management (IPM) guidance, fully offline. Component boundaries are strict: the CNN identifies; a SQLite knowledge base supplies provenance-carrying treatment records; the language model explains retrieved evidence only. When knowledge is missing, the system returns identification and states that verified guidance is unavailable; when the CNN is uncertain, it shows alternatives and withholds class-specific advice.

### 2.2 Four distinct scopes

The project defines four classification scopes in [`src/farm_pest_ai/scopes.py`](../src/farm_pest_ai/scopes.py). They are **separate tasks whose metrics must never be pooled or ranked against one another**:

| Scope | Classes | Image population | Role |
|---|---:|---|---|
| `rice10` | 10 | IP102 Classification split | Development scope; recommended final scope |
| `full102` | 102 | IP102 Classification split | Broad experiment scope |
| `det_top10` | 10 | IP102 **Detection** subset (VOC2007), all boxed | Crop experiment E4 |
| `det_top15` | 15 | IP102 **Detection** subset (VOC2007), all boxed | Crop experiment E5 |

The two detection scopes draw from a different image population with its own official train/validation/test assignment and **its own zero-based label numbering**. A detection project label is not interchangeable with an IP102 classification label; no mapping back to classification labels is exposed, precisely so a detection result cannot be silently joined to a `full102` one.

### 2.3 Lineage

Earlier phases established the harness, the audited dataset, the loader, both architectures and the training engine, then ran controlled classification experiments. Two results carry into the present work. First, the custom architecture beat the plain convolutional control on both classification scopes. Second — methodologically more important — a promising single-seed image-size gain (+0.0138 macro F1 at 224×224) **failed three-seed confirmation**, falling to +0.0079 and reversing sign on one seed. That episode is why every claim in Section 7 is qualified as single-seed. A lineage table appears in [Appendix E](#appendix-e--experiment-lineage).

---

## 3. Data and preprocessing

### 3.1 Detection subset and splits

Split assignments come from [`splits_top10.json`](../ip102_v1.1/Detection/VOC2007/splits_top10.json) and [`splits_top15.json`](../ip102_v1.1/Detection/VOC2007/splits_top15.json), which store `[filename, label]` pairs under `train`/`val`/`test`. No split is invented. Verified counts:

| Scope | Train | Validation | Test (sealed) | Total |
|---|---:|---:|---:|---:|
| `det_top10` | 6,395 | 1,370 | 1,370 | 9,135 |
| `det_top15` | 6,748 | 1,446 | 1,447 | 9,641 |

**Filename disjointness verified.** Across all three pairings in both scopes, the intersection of split filename sets is exactly zero, and no filename repeats within a split. `load_splits` additionally raises on any cross-split filename rather than letting a caller opt in.

> **Leakage caveat.** This establishes *filename*-level disjointness only. No content-hash audit exists for the detection subset — `data/reports/` holds `dataset_audit_rice10.json` and `dataset_audit_full102.json` but no detection equivalent, and the crop audits record no SHA-256 values. Two byte-identical images under different filenames would go undetected. **It cannot be claimed that content-level leakage is absent.** Establishing that needs a SHA-256 audit over the detection JPEGs, of the kind already run for the classification scopes.

### 3.2 Class imbalance

Validation support per class, read from the split files:

| Scope | Minimum support | Maximum support | Ratio |
|---|---:|---:|---:|
| `det_top10` | 64 | 440 | 6.9 : 1 |
| `det_top15` | **7** | 440 | **62.9 : 1** |

`det_top15` contains classes with 49 and 153 images in total across all splits. A class with seven validation images cannot yield a stable estimate: one image moves its recall by roughly 14 percentage points. **No class weighting was applied in any of the four runs** — `class_weighting: none`, with each run's data bundle confirming `class_weights: null`.

### 3.3 The non-nested "top N" class definitions

The scope descriptions read "Ten/Fifteen most frequent IP102 detection classes". Matching classes across scopes by filename-set overlap shows this does not hold:

| Relationship | Result |
|---|---|
| `det_top10` classes with an identical counterpart in `det_top15` (Jaccard = 1.000) | **6 of 10** |
| `det_top10` classes absent from `det_top15` entirely | 4 (labels 0, 3, 5, 6; 425–533 images each) |
| Images in `det_top10` but not `det_top15` | 1,829 |
| Images in `det_top15` but not `det_top10` | 2,335 |

Under a genuine frequency ranking, the top-15 set would be a strict superset of the top-10 set. It is not: `det_top15` *includes* classes with 49 and 153 total images while *excluding* four `det_top10` classes of 425–533 images each. **This is flagged as a dataset-definition issue.** Its consequence is specific and bounded: E4 and E5 each remain a valid, properly paired within-scope test of cropping, because within each pair the class set, manifest and seed are identical. What they are **not** is a clean ten-versus-fifteen-class progression, and the difference between the E4 and E5 headline numbers must not be read as a class-count effect. Whether the mismatch originates upstream in the IP102 detection annotations or in how these split files were generated is **not yet established**; resolving it needs the provenance of the two JSON files.

### 3.4 Bounding boxes, padding and clamping

Boxes are stored in [`boxes_top10.json`](../ip102_v1.1/Detection/VOC2007/boxes_top10.json) and [`boxes_top15.json`](../ip102_v1.1/Detection/VOC2007/boxes_top15.json) as `[x1, y1, x2, y2]` in absolute pixels — a reading established by measurement: over a 500-box sample it produced zero violations, while an `[x, y, w, h]` reading produced 408.

`pad_and_clamp` in [`detection.py:411`](../src/farm_pest_ai/data/detection.py#L411) grows each box by **15% of the box's own width and height on every side**, then clamps to `[0, width] × [0, height]`. Padding is relative to the box, so a small box grows by few pixels; each side grows independently, so a box against an image edge still gains full padding on the opposite side. A guard forces at least one pixel per dimension, so a crop is never empty.

Box area as a fraction of source image area (from the crop audits):

| Scope | p25 | **Median** | p75 | Mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| `det_top10` | 0.2947 | **0.4306** | 0.5867 | 0.4443 | 0.0053 | 0.9757 |
| `det_top15` | 0.2965 | **0.4338** | 0.5926 | 0.4459 | 0.0053 | 0.9757 |

The median box covers about **43%** of its frame, with an interquartile range of roughly **29%–59%**. Cropping therefore discards a little over half the pixels in the median case, while raising the subject's effective resolution correspondingly at a fixed 160×160 output. After padding, the crop covers the full frame for 801 `det_top10` images (10.3%), for which both arms see nearly identical pixels.

Exactly one `det_top10` image (`IP022000163.jpg`, train) has no box. `partition_records` drops it from **both** arms, so the arms consume identical sample sets.

### 3.5 Preprocessing pipeline

Built by `build_transform` in [`transforms.py:424`](../src/farm_pest_ai/data/transforms.py#L424). Input is **160×160 RGB**, normalised with ImageNet constants (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) used as fixed numbers, not pretrained weights. RGB conversion is unconditional and first — the classification audit found ten PNG files behind `.jpg` extensions, seven of them RGBA, which would otherwise present a fourth channel.

| Stage | Training | Validation / test |
|---|---|---|
| RGB conversion | Yes (unconditional) | Yes (unconditional) |
| Geometry | `RandomResizedCrop(160)`, area scale 0.6–1.0, aspect 0.75–1.333, bilinear + antialias | `Resize((160,160))`, bilinear + antialias |
| Horizontal flip | p = 0.5 | none |
| Rotation | ±15° | none |
| Colour jitter | brightness/contrast/saturation 0.2, hue 0.02 | none |
| Normalisation | ImageNet mean/std | ImageNet mean/std |

Randomness appears **only** in the training branch; validation and test share one deterministic pipeline. All four runs record preprocessing fingerprint `9e75177ab60f96e0`, a hash of the resolved pipeline that lets a checkpoint refuse to load under different preprocessing.

Cropping composes *outside* this pipeline: `BoxCropTransform` crops to the padded box, then calls the ordinary transform on the crop, so the crop arm inherits every preprocessing decision rather than reimplementing it. A missing filename or box **raises** rather than falling back to the full frame, since a silent fallback would turn a crop-arm sample into a full-frame one and corrupt the comparison ([`dataset.py:295`](../src/farm_pest_ai/data/dataset.py#L295)).

### 3.6 Why identical manifests and seeds are necessary

`build_detection_records` deliberately takes no cropping argument, so both arms receive byte-identical records, and images without a usable box are dropped from both. With manifest, seed (1337), initialisation, augmentation stream and every hyperparameter shared, the only quantity differing between E4A and E4B is *which pixels the model sees* — so any difference is attributable to cropping rather than to sample composition, initialisation luck or a co-varying hyperparameter. Each arm's configuration states exactly one field beyond its scope: `dataset.use_bbox_crop`.

**The test split remains sealed.** All four runs record splits built as train and validation only; no test loader was constructed, and `scripts/train.py` exposes no flag naming the test split.

---

## 4. CNN architecture

### 4.1 Provenance and naming

The architecture is `custom_cnn`, implemented in [`models.py`](../src/farm_pest_ai/vision/models.py) from primitives in [`blocks.py`](../src/farm_pest_ai/vision/blocks.py). It borrows three ideas associated with EfficientNet — depthwise-separable convolutions, squeeze-and-excitation and stochastic depth — but **is not EfficientNet** and is not called one here: there is no compound width/depth/resolution scaling rule, no inverted-residual expansion ratio, no searched stage layout and no pretrained weights. It is a hand-specified four-stage residual separable network.

### 4.2 Verified layer table

Derived by constructing the model from the active configuration and recording tensor shapes at every block for a 160×160 input.

| Stage | Block type | × | In → Out ch. | Stride | Spatial | Norm / Act. | SE | Skip | Drop-path |
|---|---|---:|---|---:|---|---|---|---|---:|
| Stem | `ConvBNAct` 3×3 | 1 | 3 → 32 | 2 | 160² → 80² | BN / SiLU | — | — | — |
| 1 | `ResidualSeparable` | 2 | 32 → 64; 64 → 64 | 2, 1 | 80² → 40² | BN / SiLU | r = 0.25 | 1×1 proj.; identity | 0.0000, 0.0125 |
| 2 | `ResidualSeparable` | 2 | 64 → 128; 128 → 128 | 2, 1 | 40² → 20² | BN / SiLU | r = 0.25 | 1×1 proj.; identity | 0.0250, 0.0375 |
| 3 | `ResidualSeparable` | 3 | 128 → 256; 256 → 256 | 2, 1, 1 | 20² → 10² | BN / SiLU | r = 0.25 | 1×1 proj.; identity ×2 | 0.0500, 0.0625, 0.0750 |
| 4 | `ResidualSeparable` | 2 | 256 → 384; 384 → 384 | 2, 1 | 10² → 5² | BN / SiLU | r = 0.25 | 1×1 proj.; identity | 0.0875, **0.1000** |
| Head act. | SiLU | 1 | 384 → 384 | — | 5² | — / SiLU | — | — | — |
| Head | `AdaptiveAvgPool2d(1)` → `Flatten` → `Dropout(0.3)` → `Linear` | 1 | 384 → `num_classes` | — | 5² → 1² | — | — | — | — |

Every expected configuration value is confirmed against both [`configs/model_custom.yaml`](../configs/model_custom.yaml) and each run's resolved `run.json`: stem 32; stage channels `[64, 128, 256, 384]`; repeats `[2, 2, 3, 2]`; strides `[2, 2, 2, 2]`; block `residual_separable`; SiLU; batch normalisation; SE ratio 0.25; drop-path 0.10; global-average-pooling head; dropout 0.30. Stem plus four stride-2 stages reduce 160×160 to 5×5, a total downsampling factor of 32.

### 4.3 Verified parameter counts

| Scope | Total | **Trainable** | Buffers (not optimised) | Param. memory |
|---|---:|---:|---:|---:|
| `det_top10` (10-way) | 1,435,242 | 1,435,242 | 16,425 | 5.54 MiB |
| `det_top15` (15-way) | 1,437,167 | 1,437,167 | 16,425 | 5.55 MiB |

Every parameter is trainable; nothing is frozen. The 16,425 **buffers** are BatchNorm running means and variances — persistent state written to the checkpoint but updated by observation rather than gradient descent, which is why `count_parameters` reports them separately. The five extra classes cost 1,925 parameters (`384 × 5 + 5`): the backbone holds 1,430,464 against 3,850 in the 10-way classifier, so class count is a negligible fraction of capacity.

### 4.4 What each mechanism does in this model

**Depthwise-separable convolution.** A dense `k×k` convolution is factorised into a depthwise convolution (one `3×3` filter per input channel, no cross-channel mixing) followed by a pointwise `1×1` convolution (cross-channel mixing, no spatial extent). Cost falls to roughly `1/out_channels + 1/k²` of the dense equivalent — about an eighth here — which is what lets four stages reaching 384 channels fit in 1.44M parameters and 858 MiB peak VRAM.

**Residual (skip) connections.** Each block computes `act(drop_path(branch(x)) + shortcut(x))`. The second separable convolution's pointwise projection is deliberately **linear**, so the shortcut is added *before* the final activation, letting the identity path carry an unmodified signal. The shortcut is `nn.Identity` when shape is unchanged, and a strided 1×1 `ConvBNAct` at each stage's first block, where stride and channels both change.

**Squeeze-and-excitation (SE).** Channel attention: the feature map is averaged over height and width to one value per channel ("squeeze"), passed through a bottleneck (e.g. 384 → 96 → 384 at ratio 0.25) and a sigmoid, then used to rescale each channel. The gate lies in (0, 1), so it can attenuate or preserve a channel but never invert it — a cheap global way to suppress channels responding to background rather than the insect. **No SE ablation has been run**, so its contribution on this data is not established.

**Batch normalisation.** Normalises each channel's activations across the batch, then applies a learned scale and shift, stabilising optimisation. Convolutions preceding a norm carry no bias, since BatchNorm's shift makes it redundant. At evaluation it uses running statistics, part of why evaluation is deterministic.

**SiLU.** `x · sigmoid(x)`, a smooth non-monotonic activation that passes a small negative signal rather than zeroing it as ReLU does.

**Stochastic depth (drop-path).** During training each sample independently drops the block's *entire residual branch* with probability `p`, survivors scaled by `1/(1-p)` to keep the expected activation unchanged. The rate ramps **linearly from 0.0000 at the first block to 0.1000 at the last**, since dropping early blocks as often as late ones would remove low-level features the whole network depends on. At evaluation the branch is always kept.

**Global average pooling and dropout.** The 384×5×5 map is averaged to a 384-vector, discarding spatial position — where the pest sits in the frame is not evidence of its species — and making the head independent of input resolution. Dropout (p = 0.30) then zeroes 30% of that vector, during training only.

**Raw logits.** The classifier emits `num_classes` unnormalised scores; **no softmax exists inside the model**, since the losses expect logits and the inference policy owns the conversion. `num_classes` is derived from `dataset.scope` and never hard-coded — stating `model.num_classes` in configuration is a hard error — so a 10-way head cannot be read as a 15-way one.

---

## 5. Training and optimisation

All four runs used the shared protocol in [`configs/exp_detection_protocol.yaml`](../configs/exp_detection_protocol.yaml), which states the whole `training` section and is layered last. Every value below is verified against each run's resolved `run.json`.

| Setting | Value | Purpose |
|---|---|---|
| Loss | Cross-entropy, label smoothing 0.10 | Regularisation |
| Class weighting | **none** (`class_weights: null`) | — |
| Optimiser | AdamW | Optimisation |
| Learning rate | 0.0015 (peak, post-warm-up) | Optimisation |
| Weight decay | 0.05 | Regularisation |
| Schedule | Cosine, 5 warm-up epochs, stepped **per batch** | Optimisation |
| Epochs | 60 | Budget |
| Batch size | 64 | Optimisation |
| Gradient clipping | Global norm 1.0 | Stability |
| Mixed precision | Enabled (CUDA) | Speed / memory |
| Early stopping | macro F1, mode max, patience 15, min delta 0.001 | Budget |
| Checkpoints | best + last, monitor macro F1 | Selection |
| Seed | 1337 | Reproducibility |
| Steps/epoch | 99 (`det_top10`), 105 (`det_top15`) | — |

Learning rate 0.0015 is the midpoint originally chosen so that neither arm of an earlier architecture comparison received its own tuned value; a later two-sided probe on `rice10` found both 0.0008 and 0.0030 worse, so it sits near the top of a flat region rather than on a slope.

### 5.1 Separating optimisation from regularisation

**Devices for faster or more stable training**, which aim to reach a given fit sooner or more reliably rather than to generalise better:

- **AdamW** adapts a per-parameter step size and applies weight decay decoupled from the gradient update.
- **Cosine schedule with linear warm-up**: the rate ramps over five epochs, then decays to zero at epoch 60. Stepping *per batch* rather than per epoch matters at these split sizes — with 99–105 batches per epoch, an epoch-level warm-up would be a coarse five-step staircase.
- **Gradient clipping** at global norm 1.0 bounds one outlier batch's influence.
- **Automatic mixed precision (AMP)** runs the forward pass largely in fp16 with a dynamically scaled loss to prevent gradient underflow. The engine holds the scheduler back on a skipped step, so the learning rate never advances past an optimiser step that did not happen. **All four runs recorded 0 AMP skipped steps.**

**Devices intended to improve generalisation:** label smoothing (0.10), which softens the one-hot target and discourages over-confident logits — note this raises the achievable loss floor above zero, so these loss values are not comparable to an unsmoothed run; weight decay (0.05), dropout (0.30) and stochastic depth (→ 0.10) per Section 4.4; and training-only augmentation per Section 3.5.

**Not used in these four runs:** class weighting, balanced sampling, MixUp, CutMix, random erasing (0.0) and test-time augmentation.

### 5.2 Determinism and selection

`seed_everything` seeds Python's `random`, `PYTHONHASHSEED`, NumPy and torch (all CUDA devices included), requests deterministic kernels and disables cuDNN autotuning; per-worker seeds are derived so changing one does not perturb the others. Reproducibility has been demonstrated end-to-end three times on other scopes, twice bit-identically after an interrupted run was restarted from scratch.

**Validation macro F1 is the selection metric**, chosen because the data are imbalanced and accuracy would be dominated by the largest class. `best.pt` is written whenever it improves.

---

## 6. Checkpoints and inference operation

### 6.1 `best.pt` versus `last.pt`

`best.pt` holds the epoch with the highest validation macro F1; `last.pt` holds the final epoch. Both are written atomically with an `fsync`, and `best.pt` is the artifact for evaluation and deployment. One caution the project has already met: `best.pt` holds *the epoch the monitor selected at the time*, which need not be the numerically best epoch under a later-corrected metric. For these four runs the monitored and recorded metrics agree.

### 6.2 Checkpoint payload

From [`checkpoints.py:273`](../src/farm_pest_ai/vision/checkpoints.py#L273): `model_state`; embedded `metadata`; and, when supplied, `optimizer_state`, `scheduler_state`, `scaler_state` and `rng_state` (Python, NumPy, torch CPU and all CUDA generators). The last four exist so a run can resume and are not needed for inference.

The embedded metadata carries scope, class count, class-mapping version, manifest version, preprocessing version and fingerprint, the full model configuration, epoch, seed, environment and git revision. The JSON sidecar (`best.json`) is explicitly **not** authoritative — a test rewrites one to claim the wrong scope and confirms the embedded metadata still governs.

### 6.3 Inference path

1. **Read metadata** and reconstruct the architecture from the embedded `model` configuration — the checkpoint describes its own shape, so no ambient configuration can silently build a different network.
2. **Verify provenance before any weight is copied**: scope, class count, class-mapping version and manifest version must match. `load_model_for_inference` defaults `strict_preprocessing=True`, so a fingerprint mismatch **raises**. The guard is not hypothetical — a 224×224 model scored through a 160×160 pipeline once loaded cleanly and produced a plausible but wrong confusion matrix.
3. **Load weights** with `strict=True`.
4. **`model.eval()`** — BatchNorm switches to running statistics; dropout and drop-path become identities. Evaluation is deterministic.
5. **Forward pass → raw logits**, shape `(N, num_classes)`.
6. **Softmax → probabilities**, outside the model.
7. **Argmax → predicted class**, the maximum probability serving as the confidence score for the abstention policy.

### 6.4 Deployment consequence of crop training

A crop-trained classifier saw padded box crops in training and **must receive a comparable crop at inference**. Presenting a full frame is a mismatch the fingerprint check does *not* catch, because the fingerprint covers the resize/normalise pipeline, not whether a crop was applied upstream.

**The classifier does not localise.** It has no detection head and emits no box. A deployed crop-based pipeline therefore needs a trained detector, a user-drawn region, an automatic proposal mechanism, or images that arrive already cropped. **No such component exists in this repository** — object detection is out of scope unless a detection phase is authorised. This is the largest gap between the E4/E5 result and a deployable system, and a genuine cost to weigh against cropping's measured gain.

---

## 7. Results

### 7.1 Main table — best validation checkpoint

All figures read from each run's `best.json`. Validation split only; **the test split has not been evaluated**.

| Run | Scope | Best epoch | Accuracy | Balanced acc. | **Macro F1** | Weighted F1 | Top-5 acc. | Val. loss | Train macro F1 | **Gap** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E4A full frame | `det_top10` | 50 / 60 | 0.7620 | 0.7166 | 0.7199 | 0.7626 | 0.9650 | 1.1441 | 0.9410 | 0.2211 |
| **E4B crop +15%** | `det_top10` | 53 / 60 | **0.8015** | **0.7526** | **0.7627** | **0.7992** | **0.9723** | **1.0448** | 0.9631 | **0.2004** |
| E5A full frame | `det_top15` | 45 / 60 | 0.7372 | 0.5999 | 0.6028 | 0.7358 | 0.9364 | 1.3237 | 0.9067 | 0.3039 |
| **E5B crop +15%** | `det_top15` | 48 / 60 | **0.7676** | **0.6439** | **0.6603** | **0.7618** | **0.9530** | **1.2132** | 0.9348 | **0.2745** |

Every value reconciles exactly with the artifacts, including E5A balanced accuracy, flagged for confirmation in the brief: the recorded value is **0.5999406651629007**. Arms are reported side by side within each pair; **`det_top10` and `det_top15` results are not compared with each other, nor with `rice10` or `full102`.**

Gap is train-minus-validation macro F1 at the best epoch. All four runs completed 60 epochs with 0 AMP skipped steps, peak VRAM 857.6–857.7 MiB and 7.8–8.0 minutes wall clock.

### 7.2 Paired improvements from cropping

Absolute differences, treatment minus control. Because both quantities are proportions, these are **percentage points (pp)**, not relative percentages; relative changes are shown separately.

| Metric | E4 (`det_top10`) | | E5 (`det_top15`) | |
|---|---:|---:|---:|---:|
| | **Absolute (pp)** | Relative | **Absolute (pp)** | Relative |
| Accuracy | **+3.94** | +5.17% | **+3.04** | +4.13% |
| Balanced accuracy | **+3.60** | +5.03% | **+4.40** | +7.33% |
| Macro F1 | **+4.28** | +5.94% | **+5.75** | +9.54% |
| Weighted F1 | +3.66 | — | +2.60 | — |
| Top-5 accuracy | +0.73 | — | +1.66 | — |
| Validation loss | −0.0994 | −8.68% | −0.1105 | −8.35% |
| Last-10-epoch mean macro F1 | +4.17 | +5.83% | +7.26 | +12.56% |
| Final train–val accuracy gap | −2.26 | −11.49% | −1.76 | −7.85% |

Two properties strengthen the reading. First, **the best-epoch and late-run-mean readings agree in sign and rough magnitude for both pairs**, so the verdict does not depend on which is used — unlike the earlier image-size experiment, where the two disagreed. Second, the train–validation gap moved *down* while validation moved *up*: the signature of genuine generalisation improvement rather than of a task merely becoming easier to memorise.

Per-sample prediction flips at the best checkpoints, both arms rescored through their own recorded preprocessing:

| Pair | Corrected by cropping | Broken by cropping | Net | Samples |
|---|---:|---:|---:|---:|
| E4 | 145 | 91 | **+54** | 1,370 |
| E5 | 157 | 113 | **+44** | 1,446 |

Cropping is not uniformly beneficial at the sample level: it **breaks 91 and 113 previously correct predictions**. The aggregate gain is a net of two substantial opposing flows, which is a materially weaker claim than "cropping helps".

### 7.3 Learning curves

![E4 learning curves](../artifacts/plots/crop_experiments/e4_crop_vs_fullframe.png)

**Figure 1 — E4 (`det_top10`), crop versus full frame.** Three panels (accuracy, loss, corrected macro F1) over 60 epochs; solid lines validation, dashed training. The crop arm (orange) sits above the full-frame arm (blue) on validation accuracy and macro F1 from roughly epoch 15 onward, and below it on validation loss throughout. Training curves rise to ~0.95–0.97 while validation plateaus near 0.76–0.80, and validation loss flattens after roughly epoch 30 while training loss keeps falling — the visual signature of the gaps quantified in Section 8.

**Figure 2 — E5 (`det_top15`), crop versus full frame.** The same three-panel layout for the fifteen-class scope: [`e5_crop_vs_fullframe.png`](../artifacts/plots/crop_experiments/e5_crop_vs_fullframe.png).

### 7.4 Per-class behaviour

**Aggregates conceal substantial disagreement.** Cropping improved F1 on **9 of 10** `det_top10` classes and **10 of 15** `det_top15` classes.

`det_top10`, per class at the best checkpoint:

| Label | Support | F1 full | F1 crop | ΔF1 (pp) | Recall full | Recall crop | ΔRecall (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 65 | 0.8661 | 0.9008 | +3.46 | 0.8462 | 0.9077 | +6.15 |
| 1 | 130 | 0.9057 | 0.9127 | +0.70 | 0.9231 | 0.8846 | −3.85 |
| 2 | 64 | 0.7244 | 0.7611 | +3.67 | 0.7188 | 0.6719 | −4.69 |
| 3 | 64 | 0.6897 | 0.7899 | +10.03 | 0.6250 | 0.7344 | +10.94 |
| 4 | 132 | 0.6861 | 0.7664 | +8.03 | 0.7121 | 0.7955 | +8.33 |
| **5** | 80 | 0.4500 | 0.4320 | **−1.80** | **0.4500** | **0.3375** | **−11.25** |
| 6 | 65 | 0.6190 | 0.7101 | +9.11 | 0.6000 | 0.7538 | +15.38 |
| 7 | 140 | 0.6957 | 0.7321 | +3.64 | 0.7429 | 0.6929 | −5.00 |
| 8 | 190 | 0.6860 | 0.7184 | +3.24 | 0.6842 | 0.8526 | +16.84 |
| 9 | 440 | 0.8766 | 0.9037 | +2.71 | 0.8636 | 0.8955 | +3.18 |

**Label 5 is the sole regression**, and it carries the second-largest recall movement in either direction: **−11.25 pp**, from 0.4500 to 0.3375. Three explanations fit the evidence and are not separable from these artifacts alone: genuine loss of contextual information, if the species is identified partly by substrate or group behaviour the crop excludes; box inconsistency for this class; or sampling noise — at support 80 the change is 9 images. It was already the weakest class in both arms (F1 ≈ 0.45), so it dominates neither macro F1 nor accuracy, but the direction is a real counter-example to any blanket claim.

> **Class naming not established.** The brief refers to this class as *Locustoidea*. That name appears in the repository **only** in `data/processed/full102/class_mapping.json`, the **classification** mapping. The detection split files carry bare integer labels, no detection class-name mapping exists, and detection labels are explicitly not interchangeable with classification labels. The identity of `det_top10` label 5 is therefore **not yet established** — it would need the VOC2007 annotation class names or the split files' provenance — so it is reported here as label 5.

`det_top15` shows both larger gains and larger regressions, concentrated at low support:

| Label | Support | F1 full | F1 crop | ΔF1 (pp) | Note |
|---:|---:|---:|---:|---:|---|
| 9 | 52 | 0.4697 | 0.7379 | **+26.82** | Largest gain |
| 6 | 30 | 0.5079 | 0.7719 | **+26.40** | Low support |
| **2** | **7** | 0.1818 | 0.4000 | **+21.82** | **Support 7 — not a stable estimate** |
| 13 | 62 | 0.4853 | 0.5410 | +5.57 | |
| 11 | 31 | 0.6562 | 0.7042 | +4.80 | |
| 12 | 190 | 0.6839 | 0.7255 | +4.16 | |
| 7 | 52 | 0.7308 | 0.7692 | +3.85 | |
| 8 | 62 | 0.3725 | 0.3962 | +2.37 | Weakest class both arms |
| 5 | 132 | 0.7538 | 0.7758 | +2.20 | |
| 14 | 440 | 0.8644 | 0.8702 | +0.58 | Largest class |
| 10 | 140 | 0.7918 | 0.7899 | −0.20 | |
| 0 | 130 | 0.8923 | 0.8722 | −2.01 | |
| 3 | 23 | 0.4000 | 0.3784 | −2.16 | Low support |
| 4 | 31 | 0.4348 | 0.4091 | −2.57 | Low support |
| 1 | 64 | 0.8160 | 0.7627 | −5.33 | Largest regression |

**Label 2's apparent +21.82 pp gain must not be read as evidence.** With **seven** validation images, F1 takes only a small set of discrete values, and the move from 0.1818 to 0.4000 corresponds to correctly classifying **one additional image** (recall 1/7 → 2/7). No conclusion about that class is supportable at this support. The same caution applies with decreasing force to labels 3 (23), 6 (30), 4 (31) and 11 (31). Because macro F1 weights every class equally, these tiny classes influence the headline metric out of all proportion to their reliability — the strongest single argument for the seed replication recommended in Section 10.

**No class was left unpredicted** in any of the four runs (`classes_never_predicted: []`), so there is no complete class collapse despite the 62.9:1 imbalance and the absence of class weighting.

---

## 8. Overfitting and undesirable behaviour

### 8.1 Train–validation gaps

| Run | Train macro F1 | Val. macro F1 | **Gap (macro F1)** | Train acc. | Val. acc. | Gap (acc.) |
|---|---:|---:|---:|---:|---:|---:|
| E4A | 0.9410 | 0.7199 | 0.2211 | 0.9500 | 0.7620 | 0.1879 |
| E4B | 0.9631 | 0.7627 | **0.2004** | 0.9691 | 0.8015 | 0.1676 |
| E5A | 0.9067 | 0.6028 | **0.3039** | 0.9362 | 0.7372 | 0.1990 |
| E5B | 0.9348 | 0.6603 | 0.2745 | 0.9563 | 0.7676 | 0.1886 |

All four fit the training split far better than the validation split. The gap is markedly larger on macro F1 than on accuracy in the `det_top15` runs (0.3039 versus 0.1990 for E5A), placing the excess in the minority classes, exactly where support is thinnest.

### 8.2 Best-versus-final metric change

| Run | Best epoch | Val. F1 at best | Val. F1 at epoch 60 | **Change** |
|---|---:|---:|---:|---:|
| E4A | 50 | 0.7199 | 0.7179 | **−0.0020** |
| E4B | 53 | 0.7627 | 0.7611 | **−0.0016** |
| E5A | 45 | 0.6028 | 0.5837 | **−0.0190** |
| E5B | 48 | 0.6603 | 0.6471 | **−0.0132** |

The `det_top10` runs are essentially flat after their peak (−0.002); the `det_top15` runs decline roughly ten times as much. None triggered early stopping — patience 15 was never exhausted and all four ran the full 60 epochs — so the decline is mild rather than a collapse.

### 8.3 Validation loss versus macro F1

Minimum validation loss fell at epochs 39, 43, 56 and 50 for E4A, E4B, E5A and E5B, while best macro F1 fell at 50, 53, 45 and 48. **The two disagree in every run**, and in opposite directions between scopes. In E4A loss began deteriorating 11 epochs before macro F1 peaked — the classic pattern of a network growing over-confident on its errors while its ranking of the correct class still improves. Selecting on loss would have chosen a different checkpoint in all four runs. This vindicates monitoring macro F1, but also means validation loss is not a reliable early-stopping signal here.

### 8.4 Verdict per run

Criterion applied: **mild** = gap < 0.25 *and* best-to-final decline < 0.005; **moderate** = gap 0.25–0.32 *or* decline 0.005–0.02; **strong** = gap > 0.32 or decline > 0.02 or a collapsing validation metric.

| Run | Gap | Decline | **Verdict** |
|---|---:|---:|---|
| E4A | 0.2211 | −0.0020 | **Mild** |
| E4B | 0.2004 | −0.0016 | **Mild** |
| E5A | 0.3039 | −0.0190 | **Moderate** |
| E5B | 0.2745 | −0.0132 | **Moderate** |

Overfitting is present in all four runs and worse on `det_top15`, consistent with its more severe imbalance and thinner per-class support. **`best.pt` limits the *consequence* of late-epoch deterioration** — it captures the peak, so the −0.019 late decline never reaches a deployed model — **but it does not reduce the underlying generalisation gap**, which is a property of the fitted function, not of which epoch is saved. Selecting the best of 60 noisy validation evaluations also imparts a small optimistic bias.

### 8.5 Other undesirable behaviour and risks

**Minority-class instability on `det_top15`.** Classes at support 7–31 produce per-class F1 values that move in large discrete jumps. Because macro F1 averages classes equally, roughly a third of the selection metric rests on individually unreliable estimates.

**Imbalance without weighting.** All four runs used `class_weighting: none`, so rare classes receive gradient signal roughly in proportion to their frequency. The tail did not collapse — no class went unpredicted — but rare-class F1 is visibly depressed (labels 2, 3, 4 and 8 all below 0.44 in E5B).

**Context lost to cropping.** With a median box covering ~43% of the frame, cropping discards most surrounding pixels. Label 5 in `det_top10` and the 91 and 113 broken predictions show this costs something real even where the aggregate improves.

**Non-nested class definitions.** Per Section 3.3, E4 and E5 are two independent within-pair tests. Any reading of "the effect grows with class count" from these two numbers is unsupported.

**Repeated validation-set tuning.** A cumulative, structural risk: the validation split has now informed architecture, image size, learning rate, augmentation, checkpoint selection and cropping across many experiments. Each decision leaks a little information into the model, so validation figures increasingly overstate held-out performance. The sealed test split is the only remaining unbiased estimate, and it can be spent **once**.

**Single seed.** All four runs used seed 1337, and the project has already seen a single-seed margin shrink 43% and reverse sign under three-seed replication. The *direction* of the cropping result is supported by four internally consistent readings; the *magnitude* is not established.

---

## 9. How challenges are currently handled

**Implemented** = demonstrably in effect for these four runs. **Proposed** = not implemented; not to be read as completed work.

| Challenge | Mitigation | Status | Residual risk |
|---|---|---|---|
| Class imbalance | Macro F1 as selection metric; balanced accuracy reported alongside | **Implemented** | Metric-level only. No weighting, sampling or focal loss. `det_top15` rare-class F1 stays low |
| | Conservative class-weighted loss | **Proposed** | Untested on detection scopes; on `full102` weighting redistributed accuracy without improving macro F1 |
| Overfitting | Label smoothing 0.10, weight decay 0.05, dropout 0.30, stochastic depth → 0.10, training-only augmentation, `best.pt` selection, early stopping (patience 15) | **Implemented** | Gaps of 0.20–0.30 persist. Early stopping never fired |
| | MixUp / CutMix / higher dropout | **Proposed** | Tested only on `rice10`, where both were negative (CutMix −0.0769) |
| Small pests, background clutter | Padded bounding-box crop (E4B/E5B); SE channel attention | **Implemented** | Requires a box at inference. SE contribution unablated |
| Context lost after cropping | Fixed 15% padding on every side | **Implemented** | Padding not tuned. Label 5 regressed; 91/113 predictions broken |
| | Crop-plus-larger-context arm | **Proposed** | Not run |
| Reproducibility | Full seeding (Python/hash/NumPy/torch/CUDA), deterministic kernels, cuDNN autotune off, derived worker seeds, resolved config + environment + git revision recorded per run, preprocessing fingerprint | **Implemented** | Verified bit-identical three times on other scopes; **not separately re-verified for E4/E5** |
| Checkpoint corruption / wrong-scope loading | Embedded metadata (authoritative over sidecar); scope, class-count, mapping and manifest verification before weights are copied; `strict_preprocessing=True` for inference; atomic write + `fsync` | **Implemented** | Fingerprint does not capture whether an upstream crop was applied |
| Interrupted training | Atomic per-epoch `metrics.jsonl`; optimiser/scheduler/scaler/RNG state in checkpoint; restart-from-scratch convention | **Implemented** | Resuming restarts the dataloader RNG stream, so restart-from-scratch is preferred; three long runs lost to hardware on other phases |
| Metric selection | Macro F1 primary; accuracy, balanced accuracy, weighted F1, top-5 and loss all recorded; metrics verified against scikit-learn | **Implemented** | Best-of-60 selection is optimistically biased. Loss and macro F1 disagree in all four runs |
| Train/validation leakage | Official splits used unmodified; cross-split filename overlap is a hard error; verified zero overlap | **Implemented** | **Filename level only — no content-hash audit exists for the detection subset** |
| Test-split discipline | Test loader never built; `assert_no_test_split` re-checks the bundle; no CLI flag can name it | **Implemented** | Cumulative validation-tuning bias is unquantified |
| Inference-time localisation | — | **None** | **Unsolved.** Crop-trained models need a detector, user region, proposal mechanism or pre-cropped image. No such component exists |

---

## 10. Limitations and recommended next experiments

### 10.1 Principal limitations

1. **Single seed** — the direction of the cropping result is well supported; the magnitude is not.
2. **No inference-time localisation** (Section 6.4), which gates any deployment claim for a crop-based pipeline.
3. **`det_top15` imbalance is unmitigated**, and the selection metric is disproportionately influenced by classes of support 7–31.
4. **Non-nested class definitions** prevent reading E4 and E5 as a progression.
5. **Content-level leakage unaudited** for the detection subset.
6. **Cumulative validation-tuning bias** across many experiments.
7. **Detection class names unknown**, limiting agronomic interpretation.

### 10.2 Priority 1 — conservative class weighting on `det_top15`

Two arms: E5B (cropped, unchanged) as control against an identical run differing **only** in `training.class_weighting`, using the gentler of the two schemes already implemented (`inverse_sqrt`, roughly a 9× weight ratio on `full102`). This addresses the largest identified weakness on the scope where it bites hardest, as a genuine one-variable change.

**Expect validation accuracy to fall while balanced accuracy, and possibly macro F1, rise.** Up-weighting rare classes makes the model predict them more often, so recall on them rises; but because they are rare, many of the additional predictions are wrong, so *precision* on them falls and some previously correct majority-class predictions become errors. Raw accuracy, dominated by majority classes, therefore falls, while balanced accuracy — the unweighted mean of per-class recall — rises almost mechanically. Macro F1 sits between, rewarding recall gains but charging for precision losses, so it can move either way. Exactly this was observed on `full102`: both weighting arms raised balanced accuracy (+0.0204, +0.0213) while lowering raw accuracy (−0.0136, −0.0270), and neither improved macro F1. **A rise in balanced accuracy alone is a redistribution, not an improvement**, and must be reported as a trade-off.

Because that prior evidence is unpromising, the reason to run this arm is that `det_top15`'s imbalance (62.9:1) is far more severe than `rice10`'s and differently structured from `full102`'s, making the outcome genuinely uncertain rather than foregone.

### 10.3 Priority 2 — seed replication

Repeat E5A/E5B, ideally also E4A/E4B, at two further seeds, reporting the paired per-seed difference, its mean and spread. **This is arguably more valuable than any new technique**: the project's own history shows a single-seed effect reversing under replication, and `det_top15`'s tiny minority classes make macro F1 unusually seed-sensitive. Where per-class conclusions on classes of support < 30 matter, stratified cross-validation over combined train+validation data would give a usable estimate — no amount of replicating a 7-image evaluation will make that class's figure reliable.

### 10.4 Priority 3 — regularisation, strictly one at a time

To attack the gaps in Section 8 directly, test **one** of: modestly higher dropout (0.30 → 0.40), MixUp, or CutMix, changing nothing else.

**No promise is made that any will improve validation scores.** On `rice10`, MixUp lost 0.0180 macro F1 and CutMix lost 0.0769 — the latter plausibly because occluding a rectangle destroys evidence when the subject is already small. On crops, where the insect fills much more of the frame, occlusion-style augmentation may behave differently; that is a hypothesis, not a prediction.

### 10.5 A note on returns

Nine consecutive experiments in an earlier phase failed to beat their controls, suggesting the accuracy plateau reflects properties of the architecture and data rather than an untuned hyperparameter. Cropping worked because it changed *what information reaches the model*, not how the model is regularised. **Collecting more, better-labelled, more representative images — particularly for the classes at support 7–31 — is therefore likely to be more robust than continued regularisation tuning.** Regularisation redistributes a fixed information budget; data increases it. This is slower and less convenient than another training run, which is precisely why it is easy to defer indefinitely.

---

## 11. Interim conclusion

**1. How does the CNN operate?** A 160×160 RGB image is normalised and passed through a stride-2 stem (3 → 32 channels) and four stages of residual depthwise-separable blocks, repeats `[2, 2, 3, 2]` and widths `[64, 128, 256, 384]`, each stage halving resolution to a final 5×5 map. Every block applies squeeze-and-excitation gating, batch normalisation, SiLU, a residual shortcut and stochastic depth ramping to 0.10. Global average pooling reduces the map to a 384-vector; dropout (0.30) and one linear layer emit `num_classes` **raw logits**, with softmax and argmax applied outside the model. It holds 1,435,242 parameters at 10 classes and 1,437,167 at 15 — all trainable — plus 16,425 BatchNorm buffers, and is built from primitive PyTorch layers with no pretrained weights.

**2. Did bounding-box cropping improve validation generalisation?** **Yes, in both pairs, on a single seed.** Validation macro F1 rose +4.28 pp on `det_top10` and +5.75 pp on `det_top15`, balanced accuracy +3.60 pp and +4.40 pp. Both pairs clear the 0.01 noise threshold on macro F1 and balanced accuracy together, agree between the best-epoch and late-run-mean readings, and narrowed the train–validation gap. The gain is not uniform: 91 and 113 previously correct predictions were broken, one `det_top10` class lost 11.25 pp of recall, and five of fifteen `det_top15` classes regressed. Cropping also requires a box at inference that the system cannot currently produce.

**3. Is overfitting present?** **Yes, in all four runs** — gaps of 0.2211, 0.2004, 0.3039 and 0.2745, mild on `det_top10` and moderate on `det_top15`, with validation macro F1 declining after its peak in every run (−0.002 to −0.019). `best.pt` keeps that decline out of a deployed model but does not close the gap.

**4. What prevents a deployment claim today?** Five things, any one sufficient. (i) The test split has never been evaluated; every figure here is validation-only and partly biased by repeated validation-based tuning. (ii) All results are single-seed. (iii) The models cannot localise a pest, and no detector or region-proposal component exists. (iv) `det_top15` minority classes are measured on as few as seven images. (v) The knowledge base, API, LLM layer, interface and offline container are not built, and identification without verified treatment guidance is not the product.

**5. What is the single highest-priority next experiment?** The `det_top15` class-weighting comparison of Section 10.2 — cropped control against one conservative `inverse_sqrt`-weighted arm, all else fixed — **run across at least three seeds**, so it also delivers the replication of Section 10.3. If only one is affordable, **replicate the existing E5 pair first**: an unreplicated baseline makes every later comparison against it unreliable.

---

## Unresolved questions

1. **Are the `det_top10`/`det_top15` class sets intended to be non-nested?** Only 6 of 10 classes overlap, and `det_top15` omits four `det_top10` classes of 425–533 images while including classes of 49 and 153. Requires the provenance of `splits_top10.json` and `splits_top15.json`.
2. **What are the detection class names?** No detection class-name mapping exists in the repository, so no per-class result can be given an agronomic interpretation — including `det_top10` label 5, the sole E4 regression.
3. **Is there content-level (near-duplicate) leakage in the detection splits?** Only filename disjointness is established. Requires a SHA-256 audit like those run for the classification scopes.
4. **Does the cropping effect survive seed replication?** Unknown; single seed only.
5. **Is 15% the right padding?** Untested. No crop-plus-larger-context arm has been run.
6. **Why did `det_top10` label 5 lose 11.25 pp of recall under cropping?** Lost context, box inconsistency and sampling noise are all consistent with the artifacts and are not separable from them.
7. **Does squeeze-and-excitation earn its cost on this data?** No ablation has been run.
8. **How much optimistic bias has accumulated in validation figures** from repeated validation-based selection? Unquantified, and only measurable by spending the sealed test split.
9. **Is the crop result specific to the detection image population?** The detection subset differs from the classification splits; whether cropping would help `rice10` or `full102` is untested, and boxes are not available for those scopes.

---

# Appendices

## Appendix A — Evidence and artifact index

**Architecture and model behaviour**
- [`src/farm_pest_ai/vision/models.py`](../src/farm_pest_ai/vision/models.py) — `CustomCNN`, `ModelConfig`, `build_model`, `count_parameters`
- [`src/farm_pest_ai/vision/blocks.py`](../src/farm_pest_ai/vision/blocks.py) — `ResidualSeparableBlock`, `DepthwiseSeparableConv`, `SqueezeExcite`, `DropPath`, `ConvBNAct`
- [`configs/model_custom.yaml`](../configs/model_custom.yaml)
- [`src/farm_pest_ai/scopes.py`](../src/farm_pest_ai/scopes.py) — the four scope definitions

**Training, metrics, reproducibility**
- [`src/farm_pest_ai/vision/training.py`](../src/farm_pest_ai/vision/training.py)
- [`src/farm_pest_ai/vision/metrics.py`](../src/farm_pest_ai/vision/metrics.py)
- [`src/farm_pest_ai/vision/checkpoints.py`](../src/farm_pest_ai/vision/checkpoints.py)
- [`src/farm_pest_ai/reproducibility.py`](../src/farm_pest_ai/reproducibility.py)
- [`src/farm_pest_ai/data/transforms.py`](../src/farm_pest_ai/data/transforms.py)
- [`configs/exp_detection_protocol.yaml`](../configs/exp_detection_protocol.yaml)

**Detection cropping and paired definitions**
- [`src/farm_pest_ai/data/detection.py`](../src/farm_pest_ai/data/detection.py)
- [`src/farm_pest_ai/data/dataset.py`](../src/farm_pest_ai/data/dataset.py)
- [`configs/exp_det_top10_e4a_fullframe.yaml`](../configs/exp_det_top10_e4a_fullframe.yaml) · [`e4b_crop15`](../configs/exp_det_top10_e4b_crop15.yaml) · [`e5a_fullframe`](../configs/exp_det_top15_e5a_fullframe.yaml) · [`e5b_crop15`](../configs/exp_det_top15_e5b_crop15.yaml)
- [`splits_top10.json`](../ip102_v1.1/Detection/VOC2007/splits_top10.json) · [`splits_top15.json`](../ip102_v1.1/Detection/VOC2007/splits_top15.json) · [`boxes_top10.json`](../ip102_v1.1/Detection/VOC2007/boxes_top10.json) · [`boxes_top15.json`](../ip102_v1.1/Detection/VOC2007/boxes_top15.json)

**Run artifacts** — each directory holds `best.json`, `best.pt`, `last.pt`, `metrics.jsonl`, `run.json`, `summary.json`:
- [`artifacts/checkpoints/det_top10_e4a_fullframe/`](../artifacts/checkpoints/det_top10_e4a_fullframe/) · [`det_top10_e4b_crop15/`](../artifacts/checkpoints/det_top10_e4b_crop15/) · [`det_top15_e5a_fullframe/`](../artifacts/checkpoints/det_top15_e5a_fullframe/) · [`det_top15_e5b_crop15/`](../artifacts/checkpoints/det_top15_e5b_crop15/)

**Comparison and audits**
- [`scripts/compare_crop_experiments.py`](../scripts/compare_crop_experiments.py)
- [`data/reports/crop_experiment_comparison.json`](../data/reports/crop_experiment_comparison.json)
- [`data/reports/crop_audit_det_top10.json`](../data/reports/crop_audit_det_top10.json) · [`crop_audit_det_top15.json`](../data/reports/crop_audit_det_top15.json)
- [`artifacts/plots/crop_experiments/e4_crop_vs_fullframe.png`](../artifacts/plots/crop_experiments/e4_crop_vs_fullframe.png) (+ `.svg`) · [`e5_crop_vs_fullframe.png`](../artifacts/plots/crop_experiments/e5_crop_vs_fullframe.png) (+ `.svg`)
- [`docs/STATUS.md`](STATUS.md) — phase log and risk register

**Which artifact is authoritative for what:** `best.json` for best-epoch metrics; `metrics.jsonl` for per-epoch series and train/validation splits; `run.json` for resolved configuration, parameter counts and environment; `crop_experiment_comparison.json` for derived paired differences and prediction flips; embedded checkpoint metadata over any JSON sidecar.

## Appendix B — Active configuration (E4/E5)

```yaml
# Composed: base.yaml → model_custom.yaml → exp_detection_protocol.yaml → <arm>.yaml
dataset:
  scope: det_top10 | det_top15   # per arm
  use_bbox_crop: false | true    # THE single experimental variable
  bbox_padding: 0.15
  image_size: [160, 160]
  # num_classes is DERIVED from scope (10 / 15); stating it is a hard error

model:
  name: custom_cnn
  stem_channels: 32
  stage_channels: [64, 128, 256, 384]
  stage_blocks:   [2, 2, 3, 2]
  stage_strides:  [2, 2, 2, 2]
  block: residual_separable
  se_ratio: 0.25
  dropout: 0.3
  drop_path: 0.1
  activation: silu
  norm: batchnorm
  head: global_avg_pool

preprocessing:
  interpolation: bilinear
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]
  augmentation:                    # TRAINING SPLIT ONLY
    enabled: true
    random_resized_crop: true
    scale: [0.6, 1.0]
    ratio: [0.75, 1.3333]
    horizontal_flip: 0.5
    vertical_flip: 0.0
    rotation_degrees: 15.0
    color_jitter_brightness: 0.2
    color_jitter_contrast: 0.2
    color_jitter_saturation: 0.2
    color_jitter_hue: 0.02
    random_erasing: 0.0

training:
  optimizer: adamw
  learning_rate: 0.0015
  weight_decay: 0.05
  batch_size: 64
  epochs: 60
  scheduler: cosine
  warmup_epochs: 5
  label_smoothing: 0.1
  class_weighting: none
  grad_clip_norm: 1.0
  early_stopping: {metric: macro_f1, mode: max, patience: 15, min_delta: 0.001}
  checkpoint:     {save_best: true, save_last: true, monitor: macro_f1}

reproducibility: {seed: 1337, deterministic: true, cudnn_benchmark: false}
runtime:         {device: auto, amp: true, num_workers: 8, drop_last: true}
```

**Recorded run facts.** Preprocessing fingerprint `9e75177ab60f96e0` (all four); device `cuda`; AMP enabled; 99 steps/epoch (`det_top10`) and 105 (`det_top15`); peak VRAM 857.6–857.7 MiB; 7.8–8.0 min per run; 0 AMP skipped steps; torch 2.13.0+cu126, CUDA 12.6, cuDNN 91002, Python 3.12.5, RTX 4070 Laptop (8,188 MiB); git `e2d8d25` on `zy_CNN`.

## Appendix C — Checkpoint usage notes

- **`best.pt`** — highest validation macro F1; use for evaluation and deployment. **`last.pt`** — epoch 60; use only to resume.
- **Payload:** `model_state`, `metadata`, and (when saved) `optimizer_state`, `scheduler_state`, `scaler_state`, `rng_state`.
- **Embedded metadata is authoritative**; `best.json` is a convenience sidecar and is not trusted for scope or class count.
- **Provenance is verified before weights are copied.** Scope, class count, class-mapping version and manifest version must match; `load_model_for_inference` defaults to `strict_preprocessing=True`, so a preprocessing-fingerprint mismatch raises.
- **Scope pinning:** a `det_top10` checkpoint cannot be loaded as `det_top15`, `rice10` or `full102`.
- **Crop arms need a crop.** E4B/E5B expect a 15%-padded box crop at inference; the fingerprint check does **not** detect a missing upstream crop.
- Always `model.eval()` before inference. Model output is raw logits; apply softmax externally.

## Appendix D — Reproducing the analysis without retraining

All commands are read-only with respect to checkpoints, metrics and source data, and none touches the test split. Use the project interpreter (`.venv\Scripts\python.exe` on this machine).

```bash
# Regenerate the paired comparison report and both figures
python scripts/compare_crop_experiments.py
python scripts/compare_crop_experiments.py --pair E4      # one pair only

# Re-verify the crop audit (box geometry, drops, contact sheets)
python scripts/audit_crops.py --scope det_top10
python scripts/audit_crops.py --scope det_top15 --contact-sheets

# Environment and dataset gates
python scripts/verify_environment.py
python -m pytest -q
```

Outputs: [`data/reports/crop_experiment_comparison.json`](../data/reports/crop_experiment_comparison.json), per-arm prediction CSVs under `data/reports/crop_experiments/`, and PNG + SVG figures under [`artifacts/plots/crop_experiments/`](../artifacts/plots/crop_experiments/).

The comparison script re-scores each checkpoint through **its own** recorded preprocessing under strict verification, so a pipeline mismatch raises rather than producing a plausible but wrong result. It is inference-only.

> Commands are taken from each script's own documented examples; the scripts themselves were not re-executed for this report, since the existing artifacts they produce were read directly.

## Appendix E — Experiment lineage

Compact context only; **these scopes are separate tasks and their metrics are not comparable across rows.**

| Phase | Scope | Question | Outcome |
|---|---|---|---|
| 7 | `rice10` | `custom_cnn` vs `baseline_cnn` | Custom wins (+0.1600 macro F1, corrected) |
| 7.1 | — | Macro-F1 formula defect | Corrected without retraining; results restated |
| 7.2 | `rice10` | Budget / image size / crop floor | Only 224×224 promising (+0.0138, single seed) |
| E4 (rice10) | `rice10` | 224 vs 160 over 3 seeds | **Not confirmed** (+0.0079, reversed on 1 seed); 160 retained |
| 8 | `full102` | Same protocol at 102 classes | Custom wins (+0.1185, single seed) |
| 8.1 | `rice10`, `full102` | 9 improvement experiments | **All negative**; controls survive |
| **E4** | **`det_top10`** | **Crop vs full frame** | **Crop wins, +4.28 pp macro F1, single seed** |
| **E5** | **`det_top15`** | **Crop vs full frame** | **Crop wins, +5.75 pp macro F1, single seed** |

## Appendix F — Metric definitions

Let TP, FP, FN be true positives, false positives and false negatives for a class; *support* is that class's number of ground-truth instances.

| Term | Definition | Note in this project |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of predictions for this class, the fraction correct |
| **Recall** | TP / (TP + FN) | Of true instances, the fraction found |
| **F1** | 2·P·R / (P + R) | Harmonic mean; zero when either is zero |
| **Accuracy** | Correct / total | **Dominated by the largest class.** At 440/1,446 support for one class, not a safe headline |
| **Balanced accuracy** | Unweighted mean of per-class **recall** | Every class counts equally; insensitive to precision |
| **Macro F1** | Unweighted mean of per-class **F1** | **The selection metric.** Counts every class equally, so a never-predicted class contributes 0 |
| **Weighted F1** | Per-class F1 weighted by support | Tracks accuracy closely; hides tail behaviour |
| **Top-5 accuracy** | True class within the 5 highest logits | Weak signal at 10–15 classes |
| **Loss** | Cross-entropy with label smoothing 0.10 | **Floor is above zero.** Not comparable to unsmoothed runs |
| **Train–validation gap** | Train metric − validation metric, same epoch | Larger = more overfitting. Compare only between runs whose training metric is measured the same way — a MixUp/CutMix arm's training accuracy is measured on blended images and is **not** comparable |
| **Percentage point (pp)** | Arithmetic difference of two percentages | +4.28 pp = 0.7199 → 0.7627 |
| **Relative percent** | (new − old) / old × 100 | The same change is +5.94% relative |
| **Noise threshold** | 0.01 absolute on macro F1 | Below this, treated as indistinguishable from seed noise |

---

*Interim report. Validation results only; the test split remains sealed and unevaluated. All quantitative claims are traceable to the artifacts listed in Appendix A.*
