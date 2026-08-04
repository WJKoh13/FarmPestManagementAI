# Phase plan

Work proceeds one phase at a time. Each phase ends with a summary, a file list,
the commands run, verification results, open risks and the next phase, then
stops and waits for `CONTINUE PHASE <n>`.

| # | Phase | State |
| --- | --- | --- |
| 1 | Read-only discovery | **Complete** |
| 2 | Project harness | **Complete** |
| 3 | Python, Docker and CUDA environment | **Complete** |
| 4 | Full dataset audit and derived manifests | **Complete** |
| 5 | Data loader and preprocessing | **Complete** |
| 6 | Custom CNN and smoke training | **Complete** |
| 7 | rice10 development experiments | **Complete** |
| 8 | full102 experiment and scope selection | **Training complete; scope selection pending** |
| 8.1 | Accuracy and generalization improvements (E5–E9) | **Stage 1 complete — all nine experiments negative** |
| 9 | Freeze and final CNN evaluation | Pending |
| 10 | Verified offline knowledge base | Pending |
| 11 | Ollama and local LLM evaluation | Pending |
| 12 | FastAPI integration | Pending |
| 13 | Streamlit and manual evaluation | Pending |
| 14 | Containerization and offline validation | Pending |
| 15 | Final report and integration handoff | Pending |

## Phase 8.1 — accuracy and generalization improvements

Inserted between Phases 8 and 9. **Phase 9 stays pending** until Phase 8.1 is
complete and the user explicitly approves the final scope, checkpoint,
preprocessing and uncertainty policy.

### Why this phase exists

Both scopes show a train–validation gap that points at generalization rather
than at capacity or budget:

| scope | validation accuracy | corrected macro F1 | train accuracy | val samples |
| --- | --- | --- | --- | --- |
| `rice10` `custom_cnn` | 0.6103 | 0.5913 | ~0.8841 | 721 |
| `full102` `custom_cnn` | ~0.5976 | 0.5443 | ~0.8389 | 7,508 |

The two are **separate tasks** and are never ranked against one another. To
reach 70% *full-coverage* top-1 accuracy, `rice10` needs ~65 more correct
predictions and `full102` ~769.

"Train longer" is **not** repeated as an isolated experiment: E1 already tested
a 100-epoch stretched cosine on rice10 and it did not improve the late-run mean.

### Objectives

Primary selection metric remains **validation macro F1**. Every arm also reports
full-coverage top-1 accuracy, balanced accuracy, weighted F1, top-5, validation
loss, train-versus-validation accuracy and loss gaps, per-class
precision/recall/F1/support, classes never predicted, best epoch, last-10 mean
and standard deviation, runtime, throughput, peak VRAM, and confidence/abstention
coverage and answered-accuracy at thresholds 0.5, 0.7 and 0.9.

An improvement must raise macro F1 and/or balanced accuracy without an
unacceptable loss in raw accuracy.

**70% full-coverage accuracy and 70% selective accuracy are different claims.**
The existing validation results already exceed 70% selective accuracy at
confidence 0.5 (rice10 70.7% at 78.9% coverage; full102 76.3% at 67.0%). That is
never reported as full-coverage accuracy.

### Experiment series

| id | Variable | Scope | Kind |
| --- | --- | --- | --- |
| E5 | Ensembling and test-time augmentation | rice10 + full102 | **inference only** |
| E6 | Optimizer and regularization tuning for `custom_cnn` | rice10 first | training |
| E7 | MixUp (E7a) and CutMix (E7b) | rice10 first | training |
| E8 | Fine-grained class-separation auxiliary objective | rice10 | training |
| E9 | full102 imbalance mitigation via loss weighting | full102 | training |

**E5** is inference-only and runs before any training approval is requested. It
averages **raw logits**, never predicted labels; preserves each checkpoint's
recorded scope, mapping, architecture, epoch and preprocessing under strict
verification; states explicitly whether `best.pt` or `last.pt` is used and why;
and uses **uniform weights only** — tuning ensemble weights on the same
validation split is not permitted in the first experiment.

**E6** is a staged, one-variable-at-a-time screen (learning rate, then weight
decay, then dropout/drop-path only if the gap justifies it), never a Cartesian
grid. The shared 0.0015 / 0.05 midpoint was chosen for a controlled architecture
comparison, not established as optimal for `custom_cnn`.

**E7** adds configuration-controlled, **training-only** MixUp and CutMix,
disabled by default. Validation preprocessing stays deterministic and metrics
still compare against the original hard labels. CutMix corrects lambda with the
actual clipped box area. The interaction with existing label smoothing is stated
explicitly rather than changed silently.

**E8** adds one auxiliary embedding objective. `forward()` keeps returning raw
class logits; any embedding path is separate and separately tested. Total
objective is `cross_entropy + auxiliary_weight * fine_grained_loss`. Reported
against the documented rice10 confusion groups: the three plant hoppers, the two
borers, and rice leaf caterpillar versus rice leaf roller.

**E9** uses the **existing** training-derived class-weighting capability before
any new loss is considered: `none` (control) versus inverse-square-root (9.06x)
versus effective-number at **beta 0.999** (23.53x). Full inverse-frequency
(82.00x) is avoided in the first screen because the imbalance is 82x; the
effective scheme's beta was made configurable so that arm sits between the two
rather than at 69.5x, which is where the library default would have put it.
Sampling stays unchanged so this remains a loss-weighting experiment. Results are reported by validation-support quartile using the Phase 8
grouping, and a tail-F1 gain that costs raw accuracy is reported as a trade-off.

### Experimental discipline

1. One conceptual variable per experiment; the existing `custom_cnn` is control.
2. Seed 1337 for initial screening.
3. Compare best macro F1, last-10 mean, epoch-to-epoch variability, raw accuracy,
   balanced accuracy and per-class behaviour.
4. A small single-seed difference is **not** conclusive. E4 established that
   differences below roughly 0.02 on rice10 can vanish or reverse across seeds.
5. Only meaningful or consistently favourable candidates advance.
6. The final rice10 candidate is confirmed across paired seeds 1337, 2024 and 7.
7. A proportionate full102 confirmation policy is proposed before spending
   additional expensive seeds.
8. No model is ever selected using the test set.

### Approval gate inside this phase

Authorized without further approval: documentation, infrastructure, configs,
scripts, reports, tests, unit tests, lint, type checking, validation-only
inference, short planning/profiling commands, and the E5 inference-only
ensemble/TTA experiments.

**Stage 1 was authorized and has been run** (E6a, E6b, E7a, E7b, E8, E9a, E9b),
after `--plan` measurements were reported and approved. **All seven arms landed
at or below their control**, so nothing advanced.

Still requiring explicit approval: multi-seed confirmation, combined recipes,
any Stage 2 arm, and Phase 9. Stage 2 (E6c, weight decay) was **not** triggered —
its gate was a meaningful stage-1 learning-rate winner and both directions lost.

### Test-split rule for this phase

The test split is not accessed, constructed, inspected or scored at any point in
Phase 8.1. Every decision uses training and validation data only.

## Approval gates

Explicit approval is required before:

- Installing system software or Python dependencies
- Pulling Docker images or Ollama models
- Copying large amounts of dataset data
- Running a full CNN training experiment or a long benchmark
- Starting persistent application services
- Changing Windows, WSL2, Docker, NVIDIA or firewall settings
- Deleting files, containers, images or volumes

## Test-set discipline

The test set is not touched for architecture, hyperparameter, epoch,
augmentation, threshold or scope selection. Every such decision uses validation
data. The official test set for the selected scope is evaluated **once**, in
Phase 9, after the scope, checkpoint, preprocessing, class mapping and
uncertainty policy are frozen. No retuning happens after test results are seen.

## Git discipline

Only `git branch --show-current` and `git status --short` are run without an
explicit request. No switching, merging, rebasing, cherry-picking, pulling,
committing, pushing or history rewriting. Commands that remove ignored files
(`git clean -x`, `git clean -fdx`) are forbidden, since the dataset is ignored
and would be destroyed.

This branch will not be merged into `main`. Selected finished files will later
be transferred to a new integration branch created from `main`; see
[INTEGRATION_HANDOFF.md](INTEGRATION_HANDOFF.md).
