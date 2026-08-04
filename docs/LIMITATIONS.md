# Limitations

Recorded honestly as they are discovered, not only at the end.

## Known now (Phases 1-4)

### Dataset

- **Severe class imbalance in `full102`**: 82x between the largest (3,444) and
  smallest (42) training classes. Rare classes will be recognised poorly.
- **Very small validation classes**: `full102` validation has classes with only
  7 images. Per-class recall there is statistically noisy, and macro F1
  inherits that variance.
- **Low-resolution images**: measured exhaustively in Phase 4. Below 160 px on
  the short side: `rice10` 6.3/8.3/9.6% and `full102` 4.0/5.3/5.8% for
  train/validation/test. These are upscaled, losing genuine detail. Roughly
  29-44% of images are below 224 px.
- **Exact-content cross-split leakage in `full102`** (measured in Phase 4): two
  byte-identical train/test pairs, 4 files in total, ~0.009% of the test set.
  Recorded rather than corrected, since official splits are never modified;
  Phase 9 reports test metrics with and without them. `rice10` has **zero**
  cross-split leakage.
- **Near-duplicate leakage remains unmeasured.** Phase 4 hashed file bytes,
  which catches only exact copies. The same photograph re-encoded at a different
  JPEG quality would not be detected, so neither scope can be called
  leakage-free. Perceptual hashing was not run.
- **Ten `.jpg` files are actually PNG, seven of them RGBA** (all IP102 label
  56), found by the Phase 4 full decode; Phase 1's 2,000-image sample missed
  them. They decode correctly, but the loader must convert to RGB explicitly or
  the CNN receives a fourth input channel.
- **Taxonomy is inconsistent in the source**: `classes.txt` mixes common names
  with Latin binomials and includes at least one family-level name
  (`Cicadellidae`). Raw names are preserved rather than silently corrected.
- **IP102 is a research dataset**, not a Vietnam-specific field survey. Its
  image distribution may not match photographs taken by farmers on a phone in
  the target region.

### Scope

- `rice10` covers ten rice pests only. Anything outside those classes will
  still be forced into one of the ten, which is why the uncertainty policy
  matters more than the headline accuracy.
- A model cannot say "this is not a pest" unless that behaviour is explicitly
  designed. Out-of-distribution inputs (a healthy leaf, a non-agricultural
  photo) are handled by the uncertainty policy, not by the classifier itself.

### Hardware

- Only ~4.1 GB of the RTX 4070 Laptop's 8 GB VRAM was free during Phase 1 with
  an ordinary desktop session running. Batch sizes and concurrent GPU use are
  constrained accordingly.
- Docker GPU passthrough was verified in Phase 3 under both `--gpus all` and
  `--runtime=nvidia`.

### Environment

- MSYS2 Python shadows the official CPython on `PATH`; official PyTorch wheels
  do not support MSYS2. Phase 3 must pin the interpreter explicitly.
- The global `site-packages` is shared with unrelated projects, so a dedicated
  environment is required for reproducibility.

## Safety limitations

- Confidence is never presented as certainty.
- The system does not diagnose plant disease, only the pest classes in scope.
- Treatment guidance is limited to classes with verified knowledge records.
- Organic approval status is jurisdiction-dependent and is stated as such.
- Dosage is provided only when an explicit verified source supplies it.
- Severe or uncertain outbreaks warrant expert confirmation; the system says so.

## Loader and preprocessing limitations (Phase 5)

- **Aspect ratio is not preserved.** Evaluation resizes directly to 160x160,
  distorting images whose aspect ratio is far from 1:1 — and the source spans
  0.24 to 6.04. The alternative, resizing the shorter side and centre-cropping,
  discards the frame edges where a small insect may sit. Keeping the whole frame
  was judged the lesser loss, but it is a real distortion and
  `preprocessing.resize_shorter_side` exists to test the other choice in
  Phase 7.
- **Upscaling cannot recover detail.** 6.3% of `rice10` training images and 4.0%
  of `full102` are below 160 px on the short side and are enlarged. Bilinear
  interpolation invents no information; it only avoids sharpening JPEG
  artefacts the way bicubic would. Whether errors concentrate in this cohort is
  a Phase 9 question.
- **Alpha is discarded, not composited against a known background.** The seven
  RGBA files are converted with Pillow's default, so a transparent region
  resolves against black. All seven are ordinary photographs with an unused
  alpha plane, so this is not expected to matter, but it is not verified
  pixel by pixel.
- **Normalisation uses the standard ImageNet constants**, chosen as fixed
  numbers rather than measured on IP102. No pretrained weights are involved.
  Statistics measured on the training split may fit the data better; changing
  them requires bumping `dataset.preprocessing_version`.
- **Augmentation strength is untuned.** Phase 5 fixed the mechanism, not the
  magnitudes. The defaults are conservative guesses; Phase 7 tunes them against
  validation macro F1.
- **Determinism is guaranteed for evaluation, not for training.** The training
  loader is reproducible given a fixed seed and worker count, but changing
  `runtime.num_workers` changes how the per-worker RNG streams interleave and
  therefore the exact augmentations drawn.

## Accuracy limitations (Phase 8.1)

**Full-coverage accuracy is ~60% on both scopes, and nine experiments failed to
move it.** Phase 8.1 tried five standard techniques across nine arms — test-time
augmentation, ensembling, learning-rate retuning, MixUp, CutMix, supervised
contrastive learning and two class-weighting schemes. **Not one beat its
control**, the best landing at −0.0012. The plateau is therefore a property of
the architecture and data rather than an untuned hyperparameter, and the
remaining levers are the abstention policy, more or better data, or a different
architecture — none of which Phase 8.1 was scoped to try.

**The original inference-time finding, which the training arms then confirmed:** rice10 validation accuracy is 0.6103 and full102 is ~0.5976. E5
tested the two inference-time options that cost no training at all — test-time
augmentation and ensembling — and **both failed**:

- Deterministic horizontal-flip TTA: **−0.0043** mean macro F1 across six paired
  rice10 checkpoints, positive on only 2 of 6, and −0.0021 on full102.
- Uniform ensembling: no ensemble beat its own best member by more than +0.0032,
  and the 224px and mixed-resolution ensembles were **worse** than their best
  member by −0.0150 and −0.0192.

Every remaining route to higher accuracy therefore requires retraining, which is
both more expensive and less certain. If E6–E9 also land inside noise, the honest
conclusion is that ~0.60 is close to this architecture's ceiling on this data,
and the product answer is the abstention policy rather than a better headline
number.

**70% accuracy has not been achieved at full coverage, and must not be claimed
from the selective figures.** Both models already exceed 70% accuracy *among
answered predictions* at confidence 0.5 — rice10 70.7% at 78.9% coverage,
full102 76.3% at 67.0% coverage — but that is a different quantity. A selective
figure without its coverage is not a meaningful accuracy claim.

**The accuracy/coverage trade-off is real and unavoidable.** At threshold 0.7
full102 answers half the time at ~84.6% accuracy and defers the rest. That is a
defensible product, but it means roughly half of user queries return "uncertain"
rather than an identification, and the deferred half is exactly the harder,
more ambiguous images a user is most likely to need help with.

**Four of the seven training arms are unresolved, not refuted.** E6a, E6b, E7a
and E8 all landed inside the ±0.02 band that E4 showed can reverse across seeds,
so the screen rules out a large effect but cannot distinguish a small real loss
from noise. Only E7b (−0.0769) is unambiguously worse. Screening was seed 1337
only. E4 established that a rice10 margin under ~0.02 can shrink by
43% and reverse across seeds, and risk 34 established that three seeds cannot
resolve a ~0.008 gap at all. The screen can promote a candidate for confirmation;
it cannot confirm one. Any arm landing inside ±0.02 is unresolved, not a small
win.

**The accuracy the metric measures and the accuracy users experience may differ.**
E7a MixUp lost macro F1 (−0.0180) while improving validation loss, top-5 and
selective accuracy — 87.0% among answered predictions at threshold 0.7, against
the control's 77.5%. For an abstention-based product that is the metric that
matters, and the phase's primary metric did not reward it. This is recorded as an
open question for Phase 9's uncertainty policy, not as a proposed change.

**The auxiliary objective's hyperparameters are published defaults, not tuned
values.** Weight 0.1 and temperature 0.07 were not selected on this data, which
is the same situation risk 22 records for the learning rate. A negative E8 result
is evidence about that specific setting, not about contrastive learning on this
task.

**full102 imbalance mitigation is a trade, not a fix.** E9a moved +0.0141 on the
rarest support quartile and −0.0140 on the largest, at roughly 1:1, with balanced
accuracy +0.0204 against raw accuracy −0.0136. No setting avoids the trade, and
the stronger E9b arm was worse on every quartile. Whether to accept it is a
product decision about rare-pest recall versus common-pest accuracy; macro F1
will not decide it.

**E9 screened two points on a wide continuum.** `inverse_sqrt` gives a 9.06x
weight ratio and `effective` at beta 0.999 gives 23.53x, against full inverse
weighting's 82x. Beta was made configurable and set to 0.999 precisely so the
second arm brackets rather than duplicates the excluded extreme — at the library
default it would have been 69.5x. Two points still cannot map the curve: if both
arms help, the optimum between or beyond them is unmeasured, and a third arm
would be needed to locate it.

## To be added

Populated in later phases: architecture limitations (6-8), final model error
analysis and calibration quality (9), knowledge coverage gaps (10), LLM
hallucination and unsupported-claim rates (11), and deployment constraints (14).
