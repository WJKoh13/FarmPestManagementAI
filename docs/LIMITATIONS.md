# Limitations

Recorded honestly as they are discovered, not only at the end.

## Known now (Phases 1-2)

### Dataset

- **Severe class imbalance in `full102`**: 82x between the largest (3,444) and
  smallest (42) training classes. Rare classes will be recognised poorly.
- **Very small validation classes**: `full102` validation has classes with only
  7 images. Per-class recall there is statistically noisy, and macro F1
  inherits that variance.
- **Low-resolution images**: 7.5% of `rice10` and 4.5% of `full102` images have
  a short side below 160 px and must be upscaled, losing genuine detail.
  41.7% of `rice10` images are below 224 px.
- **Content duplicates and cross-split leakage are not yet measured.**
  Filename-level checks are clean, but that does not rule out identical image
  content appearing in more than one split, which would inflate results. This
  is measured in Phase 4.
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
- Docker GPU passthrough is registered but unverified until Phase 3.

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

Populated in later phases: measured duplicate and leakage rates (4), loader and
augmentation caveats (5), architecture limitations (6-8), final model error
analysis and calibration quality (9), knowledge coverage gaps (10), LLM
hallucination and unsupported-claim rates (11), and deployment constraints (14).
