# Evaluation

Populated from Phase 7 onward.

## Automatic evaluation

Selection uses **validation macro F1**. Reported alongside it: balanced
accuracy, top-1 and top-5 accuracy, weighted F1, per-class precision and recall,
confusion matrix, calibration, model size, parameter count, CPU and GPU latency,
peak memory and throughput.

### The F1 definition (corrected in Phase 7.1)

Per-class F1 is `2 * precision * recall / (precision + recall)`, with the
denominator used **as-is** whenever it is positive. Only an exactly-zero
denominator falls back to zero, matching `sklearn`'s `zero_division=0`. Macro F1
averages over every class including ones the model never predicted; weighted F1
averages by ground-truth support.

Phase 7's implementation clamped that denominator to a minimum of 1, which is
correct for precision and recall — their denominators are integer counts — but
wrong for F1, whose denominator is a fraction. Every class with
`0 < precision + recall < 1` was therefore under-reported. All Phase 7 figures
were recomputed; see [TRAINING.md](TRAINING.md).

**The lesson for later phases.** That defect coexisted with a passing
scikit-learn comparison, because no test case exercised the interval where the
two implementations differ. Any metric added in Phase 9 must be tested across the
ranges where a plausible implementation error would actually show, not only on
convenient inputs.

Corrected metrics can be recomputed from completed runs without retraining, since
per-class precision, recall and support are recorded beside every F1:

```bash
python scripts/correct_metrics.py --verify-checkpoints
```

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

### Phase 8 scope assessment (complete)

Applied to the two completed full102 runs against the rice10 development results.
No rice10-vs-full102 macro F1 ranking was produced, per the rule above.

**Recommended scope: `rice10`**, on rare-class measurement reliability and
verified-knowledge feasibility (~10 records against ~102). Full reasoning,
including the two criteria that favour `full102`, is in
[STATUS.md](STATUS.md).

**Abstention is viable at full102's ~59.8% top-1.** Measured on validation,
softmax confidence separates correct from incorrect predictions (mean 0.772 vs
0.465), and thresholding trades coverage for accuracy predictably:

| threshold | coverage | accuracy on answered |
| --- | --- | --- |
| none | 100% | 59.8% |
| 0.5 | 67.0% | 76.3% |
| 0.7 | 50.3% | 84.6% |
| 0.9 | 24.8% | 92.7% |

These are **validation** figures used to characterise the policy shape, not to
select a threshold for deployment. Phase 9 freezes the uncertainty policy before
the test split is opened, and any threshold must be chosen on validation data
only.

**Errors are diffuse rather than concentrated.** The 15 most frequent confusion
pairs account for only 11.3% of full102's ~3,021 validation errors, so there is
no small set of fixable confusions. The pairs that do recur are taxonomically
coherent — `aphids → miridae`, `blister beetle ↔ legume blister beetle`,
`white backed plant hopper → brown plant hopper` — which is a property of the
label space, not a defect in the model.

## Results

_Test-set evaluation has not been run; it happens once, in Phase 9. Validation
results for Phases 7-8 are in [TRAINING.md](TRAINING.md) and [STATUS.md](STATUS.md)._
