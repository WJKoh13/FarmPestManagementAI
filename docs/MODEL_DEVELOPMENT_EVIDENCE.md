# Unified CNN model development evidence: Rice10 to Broad15 Deep V2

This is the single authoritative model-development document. Part I preserves
the historical Rice10 experiments that motivated Deep V2. Part II records the
active Broad15 dataset, controlled protocol, interrupted training and warm
restart, frozen model selection, validation analysis, and one-time final test.
Rice10 and Broad15 metrics must never be compared as if they used the same task.

## Purpose

This document records how the custom IP102 CNN progressed from the initial
baseline to Deep V2, which settings changed, why they were changed, and what the
saved results demonstrate on both the historical Rice10 and final Broad15
experiments. It is intended to support the project report, code review, and
explanation to the lecturer.

The central academic constraint is that the submitted CNN must be constructed
layer by layer and trained entirely from scratch. No pretrained weights,
transfer learning, or imported prebuilt CNN architecture is used by either the
baseline or Deep V2.

## Evidence sources

The statements and values below come from the repository's current artifacts:

- Baseline model: [`src/models/justin_baseline_cnn.py`](../src/models/justin_baseline_cnn.py)
- Deep model: [`src/models/justin_deep_cnn.py`](../src/models/justin_deep_cnn.py)
- Baseline configuration: [`configs/justin_baseline.yaml`](../configs/justin_baseline.yaml)
- Deep V2 configuration: [`configs/justin_deep_v2.yaml`](../configs/justin_deep_v2.yaml)
- Baseline notebook: [`notebooks/IP102_Justin_Baseline_CNN.ipynb`](../notebooks/IP102_Justin_Baseline_CNN.ipynb)
- Deep V2 notebook: [`notebooks/IP102_Justin_Deep_CNN.ipynb`](../notebooks/IP102_Justin_Deep_CNN.ipynb)
- Broad15 baseline configuration: [`configs/broad15_baseline.yaml`](../configs/broad15_baseline.yaml)
- Broad15 Deep V2 configuration: [`configs/broad15_deep_v2.yaml`](../configs/broad15_deep_v2.yaml)
- Broad15 baseline notebook: [`notebooks/IP102_Broad15_Baseline_CNN.ipynb`](../notebooks/IP102_Broad15_Baseline_CNN.ipynb)
- Broad15 Deep V2 notebook: [`notebooks/IP102_Broad15_Deep_CNN_V2.ipynb`](../notebooks/IP102_Broad15_Deep_CNN_V2.ipynb)
- Broad15 class definition: [`data_manifests/broad15_classes.json`](../data_manifests/broad15_classes.json)
- Model tests: [`tests/test_justin_baseline_cnn.py`](../tests/test_justin_baseline_cnn.py) and
  [`tests/test_justin_deep_cnn.py`](../tests/test_justin_deep_cnn.py)
- Local histories: `runs/justin_baseline_notebook/training_history.csv`,
  `runs/justin_deep_cnn/training_history.csv`, and
  `runs/justin_deep_cnn_v2/training_history.csv`

The `runs/` directory is intentionally ignored by Git. The numerical results
are recorded in this document so that the evidence remains visible in the
repository, but the original local CSV files and checkpoints should still be
retained as primary experiment artifacts.

## Part I: historical Rice10 development

All results from this point through “Historical recommendations” describe the
older ten-class Rice10 task. They explain why Deep V2 was designed but are not
Broad15 results.

## Dataset and settings retained across the experiments

The experiments use the same ten selected rice-pest classes from the official
IP102 classification split:

- 4,318 training images
- 721 validation images
- 2,166 untouched test images
- original IP102 class IDs: `0, 1, 3, 4, 5, 7, 8, 9, 10, 11`

The following core choices were retained:

| Setting | Value | Reason |
|---|---:|---|
| Input size | 160 x 160 RGB | Keeps computation practical while retaining useful visual detail. |
| Batch size | 32 | Fits the available hardware and gives stable batch updates. |
| Random seed | 42 | Makes data order and initialization more reproducible. |
| Optimizer | AdamW | Updates all randomly initialized parameters and applies decoupled weight decay. |
| Weight decay | 0.0001 | Mild regularization against excessively large weights. |
| Loss | Class-weighted cross-entropy | Reduces bias toward the larger IP102 classes. |
| Input normalization | mean `(0.5, 0.5, 0.5)`, standard deviation `(0.5, 0.5, 0.5)` | Maps image channels approximately from `[0, 1]` to `[-1, 1]`. |
| Model-selection metric | Validation macro-F1 | Gives every class equal importance despite class imbalance. |
| Test usage | Not used during tuning | Prevents test-set leakage and optimistic final reporting. |

## Stage 1: custom baseline CNN

### Architecture

The baseline was deliberately small so it could verify the complete training
pipeline and establish a reference result before adding complexity:

```text
RGB input
  -> Conv 3x3 (32)  -> BatchNorm -> ReLU -> MaxPool
  -> Conv 3x3 (64)  -> BatchNorm -> ReLU -> MaxPool
  -> Conv 3x3 (128) -> BatchNorm -> ReLU -> MaxPool
  -> Adaptive average pooling
  -> Linear 128 -> ReLU -> Dropout 0.30 -> Linear 10
```

- Trainable parameters: **111,274**
- Feature extraction: one convolution per resolution stage
- Classifier dropout: **0.30**
- Initial learning rate: **0.001**
- Training performed: **15 epochs**
- Training crop scale: **0.75 to 1.00**

The baseline was not intended to be a published architecture. It was assembled
from primitive PyTorch layers as the team's minimum viable CNN and pipeline
check.

### Baseline evidence

The checkpoint selected by validation macro-F1 was epoch 13:

| Metric at selected checkpoint | Value |
|---|---:|
| Training accuracy | 27.19% |
| Validation accuracy | 29.54% |
| Training macro-F1 | 24.51% |
| Validation macro-F1 | **27.03%** |

The highest observed validation accuracy was **31.90%** at epoch 15, but epoch
13 is the correct model-selection checkpoint because macro-F1 was defined as
the primary metric.

Training and validation performance were both low and close together. This did
not show the classic pattern of severe overfitting, where training performance
is high but validation performance is much lower. The result suggested that the
baseline had limited feature-extraction capacity and/or had not trained long
enough.

## Stage 2: first deeper experiment

The next experiment increased representational capacity:

```text
Four resolution stages
  -> two Conv-BatchNorm-ReLU operations per stage
  -> channels 32, 64, 128, 256
  -> pooling after each stage
  -> adaptive average pooling
  -> Linear 256 -> ReLU -> Dropout -> Linear 10
```

- Trainable parameters: **1,241,578**
- Eight convolutional layers instead of three
- Four feature stages instead of three
- Original classifier dropout: **0.40**
- Original stage dropouts: **0.00, 0.10, 0.20, 0.30**
- Initial learning rate: **0.001**
- Training performed: **15 epochs**

The checkpoint tensor shapes confirm that the saved first-deep run used the
same four-stage, eight-convolution topology later retained by V2. Its best
result occurred at epoch 15:

| Metric at selected checkpoint | Value |
|---|---:|
| Training accuracy | 23.85% |
| Validation accuracy | 25.66% |
| Training macro-F1 | 20.92% |
| Validation macro-F1 | **21.24%** |

This was worse than the baseline. Importantly, both its training and validation
scores remained low, so the evidence did not indicate conventional overfitting.
It was more consistent with an optimization-limited or over-regularized run:

- the model had roughly eleven times more parameters than the baseline;
- it was given only the same 15-epoch training duration;
- deeper-stage dropout reached 0.30 and classifier dropout reached 0.40;
- the initial learning rate remained at 0.001 despite the larger network.

These observations motivated tuning the training configuration while retaining
the deeper topology.

## Stage 3: Deep V2

Deep V2 kept the four-stage, eight-convolution topology. This was deliberate:
the next question was whether the deeper network could learn effectively with
less aggressive regularization and a more conservative optimization schedule.

### Changes from the baseline and first deep run

| Component | Baseline | First deep run | Deep V2 | Reason for V2 change |
|---|---:|---:|---:|---|
| Convolutions | 3 | 8 | 8 | Preserve the deeper model's ability to learn more detailed textures and pest structures. |
| Feature stages | 3 | 4 | 4 | Retain a higher-level 256-channel representation. |
| Trainable parameters | 111,274 | 1,241,578 | 1,241,578 | Test whether the deeper capacity can be optimized before redesigning it again. |
| Classifier dropout | 0.30 | 0.40 | **0.20** | Reduce possible underfitting from excessive regularization. |
| Stage dropouts | none | 0.00/0.10/0.20/0.30 | **0.00/0.00/0.05/0.10** | Preserve more feature information, especially in early stages. |
| Initial learning rate | 0.001 | 0.001 | **0.0005** | Use smaller updates for the larger network and reduce unstable movement around useful weights. |
| Maximum epochs actually run | 15 | 15 | **50** | Give the larger network enough time to converge. |
| Scheduler patience | 3 | 3 | **3** | Retained; halve the learning rate when validation macro-F1 stalls. |
| Early-stopping patience | 10 | 10 | **10** | Retained; stop only after a sustained lack of validation improvement. |
| Training crop scale | 0.75-1.00 | 0.75-1.00 | **0.90-1.00** | Reduce the chance that an aggressive crop removes a small or poorly localized pest. |
| Horizontal flip/rotation/colour jitter | enabled | enabled | enabled | Retain moderate visual variation without changing the pest class. |

The explanations in the last column are experiment hypotheses grounded in the
observed learning behaviour. They are not yet individual causal proofs; that
requires one-change-at-a-time ablation experiments.

### Deep V2 evidence

The checkpoint selected by validation macro-F1 was epoch 47:

| Metric at selected checkpoint | Value |
|---|---:|
| Training accuracy | 49.56% |
| Validation accuracy | **47.43%** |
| Training macro-F1 | 48.69% |
| Validation macro-F1 | **45.96%** |
| Learning rate | 0.00003125 |

Compared with the baseline's selected checkpoint, V2 improved validation
accuracy by **17.89 percentage points** and macro-F1 by **18.93 percentage
points**. Compared using each run's peak validation accuracy, it improved from
31.90% to 47.43%, a gain of **15.53 percentage points**.

At the selected V2 checkpoint, the train-validation gap was approximately 2.13
percentage points for accuracy and 2.73 points for macro-F1. This relatively
small gap is evidence against severe overfitting at that checkpoint. It does not
prove that the model will generalize to the test set, but it is a healthy
validation result.

The learning-rate history also shows that performance continued improving after
the scheduler reduced the rate. Validation macro-F1 rose from 30.31% at epoch 12
to 45.96% at epoch 47. Stopping every experiment at 15 epochs would therefore
have hidden much of V2's learning progress.

## What the current evidence supports

The saved evidence supports the following conclusions:

1. The shallow baseline provided a working reference but had limited fitting
   performance.
2. Simply making the network deeper did not automatically improve performance.
3. The first deep run was not showing high-training/low-validation overfitting;
   both scores were low.
4. The complete V2 combination of lower dropout, lower initial learning rate,
   less aggressive cropping, learning-rate reductions, and longer training was
   substantially better on the validation split.
5. Deep V2 reached its best validation result without a large
   train-validation gap.

## What the current evidence does not prove

The baseline and V2 runs are **not a controlled architecture-only comparison**.
V2 changed the architecture relative to the baseline and also changed the
learning rate, dropout, crop scale, and number of epochs. Consequently, the
current evidence shows that the complete V2 experimental configuration is
better; it cannot determine exactly how much improvement came from each change.

The current values are also validation results. The official test split must
remain untouched until the team freezes its final architecture and
hyperparameters. A final test result is required before claiming deployment
performance.

## Scratch-built and scratch-trained compliance

The baseline and Deep V2 meet the technical definition of training from scratch:

- every layer is explicitly constructed from primitive `torch.nn` components;
- no `torchvision.models` architecture is instantiated;
- no external checkpoint or ImageNet weight file is loaded;
- convolution and linear weights receive new Kaiming-random initialization;
- linear biases are initialized to zero;
- BatchNorm scale and offset start at one and zero respectively;
- AdamW learns all trainable weights from the selected IP102 training split;
- checkpoints loaded during evaluation contain weights produced by these local
  training runs.

The network uses standard mathematical components such as convolution,
BatchNorm, ReLU, pooling, and dropout. The custom contribution is the team's
layer configuration and the decisions about depth, channels, regularization,
optimization, and preprocessing. The team should describe it as a **custom deep
sequential CNN**, not as VGG, ResNet, AlexNet, GoogLeNet, or a novel research
architecture.

## Historical recommendations that led to Broad15

Before final test evaluation, run controlled ablations using the same seed and
official validation split. Change only one factor at a time:

1. Train the baseline for 50 epochs with V2's learning-rate schedule and
   preprocessing. This better isolates the effect of architecture depth.
2. Keep the Deep V2 architecture and restore the first deep run's stronger
   dropout. This tests the over-regularization hypothesis.
3. Keep all V2 settings and compare crop scales `0.75-1.00` and `0.90-1.00`.
4. Keep all V2 settings and compare initial learning rates `0.001` and `0.0005`.
5. If time permits, repeat the final configuration with three seeds and report
   mean and standard deviation.
6. Freeze the selected configuration, evaluate it once on the official test
   split, and report accuracy, macro-F1, per-class F1, and the confusion matrix.

Each future run should save its exact configuration, seed, history, best
checkpoint epoch, and final metrics under a unique run name. Do not overwrite
the existing evidence.

## Part II: active Broad15 experiment

### Project goal

The active task is to classify one field or reference image as one of 15
selected farm-pest categories using a CNN designed layer by layer by the team
and trained entirely from random initialization on IP102.

The application may attach management guidance after classification, but the
classifier and guidance are separate responsibilities. An LLM must not replace
the CNN or be treated as an authoritative pesticide adviser.

### Frozen Broad15 v1 dataset

The shortlist was selected using measurable dataset-quality proxies: available
classification images, annotation coverage, sampled resolution, object
visibility, and representation across broad pest categories. It is a practical
benchmark subset rather than an expert-certified species taxonomy.

| Project label | Original IP102 label | IP102 class | Train | Validation | Test |
|---:|---:|---|---:|---:|---:|
| 0 | 14 | grub | 516 | 86 | 258 |
| 1 | 15 | mole_cricket | 989 | 165 | 495 |
| 2 | 16 | wireworm | 532 | 88 | 267 |
| 3 | 18 | black_cutworm | 512 | 85 | 257 |
| 4 | 22 | corn_borer | 1,018 | 170 | 510 |
| 5 | 23 | army_worm | 642 | 107 | 322 |
| 6 | 24 | aphids | 2,456 | 409 | 1,229 |
| 7 | 37 | flea_beetle | 473 | 79 | 237 |
| 8 | 45 | flax_budworm | 639 | 107 | 320 |
| 9 | 47 | tarnished_plant_bug | 492 | 82 | 246 |
| 10 | 51 | blister_beetle | 1,138 | 189 | 570 |
| 11 | 69 | cicadella_viridis | 767 | 128 | 384 |
| 12 | 70 | miridae | 3,048 | 508 | 1,525 |
| 13 | 86 | prodenia_litura | 782 | 130 | 392 |
| 14 | 101 | cicadellidae | 3,444 | 573 | 1,723 |
| | | **Total** | **17,448** | **2,906** | **8,735** |

Original labels are zero-based and retain IP102's official train, validation,
and test partitions. The tracked source of truth is
[`data_manifests/broad15_classes.json`](../data_manifests/broad15_classes.json).
Regenerate the ignored manifests without random resplitting:

```bash
python scripts/build_subset_manifests.py --definition data_manifests/broad15_classes.json
```

#### Dataset limitations

- Classification images mix adults, larvae, pupae, eggs, damage-only scenes,
  diagrams, composites, specimen plates, and field photographs.
- Class sizes remain imbalanced: training counts range from 473 to 3,444.
- Similar-looking insects and inconsistent object scale create genuine visual
  ambiguity.
- A high-resolution image may still contain a very small pest.
- Labels mix taxonomic ranks. For example, `cicadellidae` is a family while
  `cicadella_viridis` is a more specific member-level category.
- Dataset-quality proxies do not replace expert image-by-image label review.

These limitations are why macro-F1 and per-class behavior are reported with
accuracy. Broad15 performance is benchmark evidence, not biological certainty.

### Locked controlled protocol

Baseline and Deep V2 use the same:

- Broad15 manifests and label mapping;
- 160 x 160 RGB input and batch size 32;
- seed 42;
- AdamW with learning rate 0.0005 and weight decay 0.0001;
- class-weighted cross-entropy;
- training augmentation and deterministic evaluation preprocessing;
- `ReduceLROnPlateau` scheduler;
- validation macro-F1 checkpoint selection;
- maximum 50 epochs and early-stopping patience 10.

Shared settings live in
[`configs/_broad15_base.yaml`](../configs/_broad15_base.yaml). The architecture
configuration files contain only model-specific choices. This makes the
baseline-versus-Deep-V2 comparison a controlled architecture comparison.

| Model | Trainable parameters | Relative size |
|---|---:|---:|
| Broad15 baseline | 111,919 | 1.0x |
| Broad15 Deep V2 | 1,242,863 | 11.1x |

### Broad15 baseline evidence

The baseline completed 50 epochs. Its best validation checkpoint was epoch 50:

| Metric | Value |
|---|---:|
| Training loss | 1.9131 |
| Validation loss | 1.8680 |
| Training accuracy | 35.05% |
| Validation accuracy | **38.82%** |
| Training macro-F1 | 32.33% |
| Validation macro-F1 | **35.25%** |

The baseline learning rate reduced from 0.0005 to 0.00025 at epoch 12,
0.000125 at epoch 37, and 0.0000625 at epoch 46. Performance continued to
improve late in training, showing that the shallow model benefited from the
full schedule but remained capacity-limited.

### Deep V2 interruption and warm restart

The initial Deep V2 process was interrupted when VS Code closed on 2026-08-04.
The best-only checkpoint survived:

| Phase-one field | Saved value |
|---|---:|
| Best global epoch | 25 |
| Validation macro-F1 | **48.59%** |

That checkpoint did not contain optimizer, scheduler, early-stopping, RNG, or
history state. Training therefore resumed as a **warm restart** from the best
weights with a new optimizer and scheduler. It was not an exact continuation.

The second phase ran 25 local epochs corresponding to global epochs 26-50. The
current checkpoint uses the correct global best epoch, 48. The phase-two CSV
uses local labels 1-25; this labeling limitation is retained for provenance.

| Metric | Selected global epoch 48 | Final global epoch 50 |
|---|---:|---:|
| Training loss | 0.8986 | **0.8635** |
| Validation loss | **1.0751** | 1.1137 |
| Training accuracy | 65.66% | **66.32%** |
| Validation accuracy | **65.31%** | 63.21% |
| Training macro-F1 | 64.76% | **65.80%** |
| Validation macro-F1 | **62.48%** | 61.13% |
| Learning rate | 0.00025 | 0.00025 |

The commonly quoted 66% is final **training** accuracy, not held-out accuracy.
Epoch 48 is correctly selected because validation macro-F1 is the declared
selection metric. From epoch 48 to 50, training metrics improved while
validation metrics declined slightly, so retaining epoch 48 prevented the
weaker final model from replacing the better checkpoint.

A separate clean reproducibility run was started after crash-safe checkpointing
was implemented. It was stopped at the user's request after one completed epoch
because of the available time. Its partial artifacts are excluded from model
selection, and no further clean epochs were run.

### Controlled Broad15 comparison

| Comparison | Baseline | Deep V2 | Absolute gain |
|---|---:|---:|---:|
| Validation macro-F1 at epoch 25 | 30.68% | **48.59%** | **+17.91 points** |
| Best validation macro-F1 by epoch 50 | 35.25% | **62.48%** | **+27.23 points** |
| Best validation accuracy by epoch 50 | 38.82% | **65.31%** | **+26.50 points** |

The evidence supports Deep V2 as the stronger Broad15 architecture candidate.
It does not isolate the contribution of any individual layer, and one seed does
not establish run-to-run variance.

### Frozen model and one-time test

The epoch-48 warm-restart checkpoint was frozen before the official test split
was evaluated. Test data was not used to select the architecture or epoch. The
test split was evaluated once on 2026-08-04.

| Split | Images | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Validation | 2,906 | **65.31%** | 61.26% | 65.83% | **62.48%** |
| Test | 8,735 | **64.41%** | 59.98% | 64.59% | **61.23%** |

Test accuracy is 0.91 percentage points below validation accuracy, and test
macro-F1 is 1.24 points below validation macro-F1. The small differences support
similar behavior across IP102's held-out splits for this fixed checkpoint. They
do not estimate performance on farms, new regions, or out-of-distribution data.

### Final per-class evidence

| Class | Validation F1 | Test precision | Test recall | Test F1 | Test support |
|---|---:|---:|---:|---:|---:|
| grub | 77.84% | 80.49% | 76.74% | 78.57% | 258 |
| mole_cricket | 83.13% | 78.72% | 84.44% | **81.48%** | 495 |
| wireworm | 64.84% | 64.86% | 62.92% | 63.88% | 267 |
| black_cutworm | 67.02% | 53.80% | 66.15% | 59.34% | 257 |
| corn_borer | 64.79% | 57.36% | 66.47% | 61.58% | 510 |
| army_worm | 45.45% | 43.62% | 38.20% | 40.73% | 322 |
| aphids | 66.41% | 72.34% | 66.40% | 69.24% | 1,229 |
| flea_beetle | 65.67% | 48.77% | 75.53% | 59.27% | 237 |
| flax_budworm | 43.24% | 39.08% | 42.50% | 40.72% | 320 |
| tarnished_plant_bug | 36.14% | 28.06% | 56.91% | **37.58%** | 246 |
| blister_beetle | 67.26% | 59.51% | 76.84% | 67.08% | 570 |
| cicadella_viridis | 60.82% | 51.80% | 78.65% | 62.46% | 384 |
| miridae | 65.26% | 71.49% | 54.75% | 62.01% | 1,525 |
| prodenia_litura | 53.78% | 64.41% | 55.87% | 59.84% | 392 |
| cicadellidae | 75.49% | 85.38% | 66.45% | 74.74% | 1,723 |

The weakest classes are consistently `tarnished_plant_bug`, `flax_budworm`,
and `army_worm` on both validation and test. This consistency suggests a real
class-level limitation rather than an isolated validation fluctuation.

### Validation confusion and image inspection

| Actual class | Predicted class | Validation images |
|---|---|---:|
| miridae | tarnished_plant_bug | 60 |
| cicadellidae | cicadella_viridis | 56 |
| miridae | blister_beetle | 47 |
| miridae | aphids | 47 |
| aphids | miridae | 44 |
| prodenia_litura | flax_budworm | 18 |

Manual inspection of high-confidence mistakes found:

- `53914.jpg` is a clean true-bug specimen plate labeled `miridae` and predicted
  as `tarnished_plant_bug`; the fine-grained visual distinction is difficult.
- `72223.jpg` is a watermarked leafhopper stock photograph labeled
  `cicadellidae` and predicted as `cicadella_viridis`; the two labels are
  hierarchically related rather than cleanly mutually exclusive.
- `55873.jpg` is a sparse, partial line drawing labeled `miridae` and predicted
  as `aphids`, unlike a normal field photograph.
- `19925.jpg` is a solitary aphid on a blank background predicted as `miridae`,
  illustrating true-bug shape similarity.
- `62767.jpg` is an adult moth labeled `prodenia_litura` and predicted as
  `flax_budworm`; both classes contain Lepidoptera and mixed life stages.

The errors therefore reflect both model limitations and dataset heterogeneity:
fine-grained morphology, overlapping taxonomy, life-stage variation, drawings,
specimen plates, watermarks, backgrounds, and object scale.

### Engineering evidence and crash safety

The shared trainer now:

- supports Intel Arc through PyTorch XPU;
- writes history after each completed epoch;
- atomically writes `last_checkpoint.pt` after each epoch;
- stores model, optimizer, scheduler, global epoch, best metric,
  early-stopping counter, full history, and RNG state;
- uses explicit global epoch labels on resumed command-line runs;
- keeps validation and test artifacts separate;
- refuses to replace an existing split evaluation without `--force`;
- produces reproducible weak-class, confusion, and high-confidence error lists.

Sixteen unit tests pass, including checkpoint reload and history-schema tests.

### Frozen artifacts and reproducibility

The final local run is:

`runs/broad15/final/deep_v2_seed42_selected/`

It contains the frozen checkpoint, selection provenance, validation and test
result JSON files, prediction CSVs, confusion matrices, validation error
analysis, and a generated final report. The directory is ignored by Git and
must be backed up separately.

The final model has approximately 4.77 MB of checkpoint storage and measured
CPU inference latency of approximately 19.37 ms per image on the evaluation
laptop.

### Final conclusions and limitations

The current evidence supports these conclusions:

1. The Broad15 shallow baseline is a valid reference but is substantially
   weaker than Deep V2.
2. Deep V2's additional feature capacity is useful under the locked Broad15
   protocol.
3. Validation and test results are close, providing credible held-out evidence
   within IP102.
4. Class performance remains uneven, with the three weakest categories near
   38-41% test F1.
5. Benchmark taxonomy and image heterogeneity materially contribute to errors.

The evidence does not support claims of exact uninterrupted optimization,
run-to-run stability, guaranteed real-farm performance, biological certainty,
or causal attribution of the gain to an individual Deep V2 layer.

The frozen test result is final evidence, not a new tuning target. Future work
should prioritize label/taxonomy curation, weak-class image review, additional
seeds, and external field-image validation rather than repeatedly tuning against
the official test set.
