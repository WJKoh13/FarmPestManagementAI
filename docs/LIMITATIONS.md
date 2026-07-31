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

## To be added

Populated in later phases: loader and augmentation caveats (5), architecture
limitations (6-8), final model error analysis and calibration quality (9),
knowledge coverage gaps (10), LLM hallucination and unsupported-claim rates
(11), and deployment constraints (14).
