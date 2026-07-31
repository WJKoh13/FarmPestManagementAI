# Phase plan

Work proceeds one phase at a time. Each phase ends with a summary, a file list,
the commands run, verification results, open risks and the next phase, then
stops and waits for `CONTINUE PHASE <n>`.

| # | Phase | State |
| --- | --- | --- |
| 1 | Read-only discovery | **Complete** |
| 2 | Project harness | **Complete** |
| 3 | Python, Docker and CUDA environment | Next |
| 4 | Full dataset audit and derived manifests | Pending |
| 5 | Data loader and preprocessing | Pending |
| 6 | Custom CNN and smoke training | Pending |
| 7 | rice10 development experiments | Pending |
| 8 | full102 experiment and scope selection | Pending |
| 9 | Freeze and final CNN evaluation | Pending |
| 10 | Verified offline knowledge base | Pending |
| 11 | Ollama and local LLM evaluation | Pending |
| 12 | FastAPI integration | Pending |
| 13 | Streamlit and manual evaluation | Pending |
| 14 | Containerization and offline validation | Pending |
| 15 | Final report and integration handoff | Pending |

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
