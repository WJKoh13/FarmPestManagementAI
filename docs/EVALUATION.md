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

## Ensemble and TTA evaluation rules (Phase 8.1)

These apply to any evaluation that combines more than one forward pass, whether
across checkpoints (ensembling) or across transformed views of one image (TTA).
Implemented in `farm_pest_ai.vision.ensemble` and exercised by
`scripts/evaluate_ensemble.py`.

**Combine raw logits, before softmax.** Never combine predicted labels, and never
combine probabilities. Voting throws away the confidence the combination exists
to pool, and probability averaging saturates a confident member's contribution at
~1.0 so it can be outvoted by several mild members — the two methods can select
different labels on the same inputs. Averaging is done in float64 so member order
cannot change the result.

**Score every member through its own recorded preprocessing.** Rebuild it from
the member's run summary and verify the resulting fingerprint against the
fingerprint embedded in its checkpoint, with `strict_preprocessing=True`. A model
trained at 224x224 loads cleanly into a 160x160 pipeline and produces a
plausible, wrong result; Phase 7.2 hit exactly that. Members with *different*
fingerprints may still be combined — that is what makes a mixed-resolution
ensemble legitimate — because each was scored correctly for itself.

**Prove alignment; do not assume it.** Evaluation loaders preserve official
manifest order with `drop_last=False`, so row *i* should be the same image for
every member — but a misaligned ensemble produces a perfectly plausible accuracy
and nothing downstream would notice. Every member therefore carries its own
target vector, and combination is refused unless scope, class count, sample count
and targets all agree.

**Never mix scopes.** `rice10` and `full102` labels do not denote the same
categories; the scope check rejects the combination before any logit is touched.

**Uniform weights in a first experiment.** Fitting per-member weights on the same
validation split that then judges the ensemble measures the split rather than the
method. The API exposes no weight argument.

**Name the checkpoint and record the epoch.** State whether `best.pt` or
`last.pt` was used and why. Do not assume `best.pt` holds the numerically best
epoch under the corrected metric — Phase 7.1 deliberately left one stale, and E5
reproduced that discrepancy independently.

**Record reproducibility metadata**: ensemble membership, each member's
checkpoint SHA-256, epoch, scope, architecture, preprocessing fingerprint and
image size.

**Do not ensemble a substantially weaker model into a stronger one by default.**
Uniform averaging is as likely to drag the strong member down as to help. Such a
pairing is evaluated only with a validation-based reason, and both individual
components are reported alongside it.

## Full-coverage versus selective accuracy

These are **different claims** and are reported in separate blocks. Conflating
them overstates the system substantially.

**Full-coverage top-1 accuracy** is accuracy over every validation image, with no
abstention. This is the headline number and the one a "70% accuracy" target
refers to.

**Selective accuracy** is accuracy among *answered* predictions only, after the
model abstains on anything below a confidence threshold. It rises with the
threshold precisely because the model declines the cases it is unsure about, and
must always be quoted together with its **coverage**.

Reported at thresholds 0.5, 0.7 and 0.9. Measured on validation:

| threshold | rice10 coverage | rice10 selective acc | full102 coverage | full102 selective acc |
| --- | --- | --- | --- | --- |
| none | 100% | 60.7% | 100% | 59.8% |
| 0.5 | 78.9% | 70.7% | 67.0% | 76.3% |
| 0.7 | 58.5% | 77.5% | 50.3% | 84.6% |
| 0.9 | 24.7% | 88.8% | 24.8% | 92.7% |

**The existing models already exceed 70% selective accuracy at confidence 0.5.
They do not reach 70% full-coverage accuracy.** Any statement that the system is
"over 70% accurate" must name which of the two it means and, for the selective
figure, state the coverage it was achieved at.

In the report format, selective entries carry a `selective_accuracy` key and
deliberately no bare `accuracy` key, so a reader cannot mistake one for the
other; a test pins that.

## Phase 8.1 selective-accuracy measurements

Measured on validation for every Stage 1 arm. The control remains the best
abstention policy on both scopes; no arm improved coverage and answered accuracy
together.

**rice10** (coverage / accuracy among answered):

| arm | @0.5 | @0.7 | @0.9 |
| --- | --- | --- | --- |
| E0 *(control)* | 78.9% / 70.7% | 58.5% / 77.5% | 24.7% / 88.8% |
| E6a | 73.6% / 70.2% | 52.4% / 79.4% | 26.5% / 89.0% |
| E6b | 81.6% / 67.2% | 61.6% / 74.5% | 28.6% / 89.3% |
| E7a MixUp | 59.4% / **76.4%** | 37.4% / **87.0%** | 11.2% / **91.4%** |
| E7b CutMix | 43.8% / **81.6%** | 23.2% / **92.2%** | 3.2% / 95.7% |
| E8 | 72.4% / 70.5% | 50.5% / 79.9% | 24.3% / 90.9% |

**full102**:

| arm | @0.5 | @0.7 | @0.9 |
| --- | --- | --- | --- |
| E0-102 *(control)* | 67.0% / 76.3% | 50.3% / **84.6%** | 24.8% / **92.7%** |
| E9a | 57.0% / 79.3% | 36.6% / 85.8% | 13.0% / 89.6% |
| E9b | 47.4% / **81.6%** | 25.8% / 86.1% | 8.8% / 91.0% |

**The mixed and weighted arms buy answered-accuracy with coverage.** E7b answers
under a quarter of the rice10 split at threshold 0.7. Reading any of these
selective figures as a headline accuracy would overstate the system severely —
E7b's 92.2% describes 23.2% of the images.

**E7a is the one genuine tension in the phase.** It lost on the primary metric
(macro F1 −0.0180) while beating the control substantially on selective accuracy
at every threshold. For an abstention-based product that is arguably the relevant
comparison, and Phase 9's uncertainty policy should weigh it explicitly rather
than inheriting the macro-F1 ranking unexamined. See risk 43.

## Results

_Test-set evaluation has not been run; it happens once, in Phase 9. Validation
results for Phases 7-8 are in [TRAINING.md](TRAINING.md) and [STATUS.md](STATUS.md)._
