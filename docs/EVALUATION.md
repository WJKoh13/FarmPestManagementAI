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

### Known measurement caveat

`full102` validation contains classes with as few as 7 images (labels 72 and
80). Per-class recall for the rare tail is therefore very noisy, and macro F1
inherits that variance. This is reported explicitly rather than smoothed over.

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
