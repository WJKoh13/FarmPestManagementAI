# Evaluation

Populated from Phase 7 onward.

## Automatic evaluation

Selection uses **validation macro F1**. Reported alongside it: balanced
accuracy, top-1 and top-5 accuracy, weighted F1, per-class precision and recall,
confusion matrix, calibration, model size, parameter count, CPU and GPU latency,
peak memory and throughput.

### Comparing scopes

`rice10` and `full102` are **different classification tasks**. Their macro F1
values are never compared directly as though they measured the same thing. A
10-way problem with 2.8x imbalance is not comparable to a 102-way problem with
82x imbalance.

Scope selection (Phase 8) weighs intended application coverage, validation
quality, per-class behaviour, runtime, model size, hardware feasibility and
knowledge-base feasibility — not a single headline number.

### Known measurement caveats

`full102` validation contains classes with as few as 7 images (labels 72 and
80). Per-class recall for the rare tail is therefore very noisy, and macro F1
inherits that variance. This is reported explicitly rather than smoothed over.

**Exact-content test leakage in `full102`.** The Phase 4 audit found two
byte-identical train/test pairs: 40410/40432 (label 56) and 65553/66152
(label 92). Two contaminated images out of 22,619 is ~0.009% of the test set,
far too small to move a headline metric, but Phase 9 reports test results **with
and without** them rather than assuming the effect is negligible. The official
splits are not modified. `rice10` has **zero** cross-split leakage.

Byte hashing detects exact copies only. Near-duplicate leakage — the same photo
re-encoded at a different JPEG quality — remains unmeasured, so no claim of
"leakage-free" is made for either scope.

## Manual evaluation

Two Streamlit review workflows (Phase 13), persisted to SQLite with CSV export.

**CNN review**: registered-model selection, scope display, blind review with
ground truth revealed only after submission, a rating of correct / incorrect /
partially acceptable / cannot determine, expected-class selection, image-quality
notes, and reviewer or anonymous session identity.

**LLM review**: candidate selection, groundedness rubric, safety rubric,
unsupported-claim reporting, notes and export.

Manually reviewed test images are **never** used to tune the frozen model.

## Safety cases

Exercised in Phase 14 against the running system: healthy leaf, no visible
insect, unknown insect, blurry image, multiple insects, non-agricultural image,
low-confidence result, missing knowledge record, Ollama unavailable, invalid
LLM JSON, unsupported treatment request, dosage request without a source, and a
certification question without jurisdiction.

Each must produce a safe, honest response rather than a confident guess.

## Results

_No evaluation has been run. Populated in Phases 7-9 and 13-14._
