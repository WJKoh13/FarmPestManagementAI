# FarmPestManagementAI

Scratch-built CNN classification for a broad set of 15 farm pests from IP102.
The primary experiment compares the team's own shallow baseline against its own
Deep V2 model. Both networks are defined layer by layer and trained from random
initialization: no pretrained weights, transfer learning, `torchvision.models`,
or copied named architecture is used.

The earlier ten-class rice-pest experiment is retained as historical evidence.
Its results must not be compared numerically with Broad15 results because the
class set and dataset size are different.

## Broad15 goal

The application goal is to classify one image into one of these 15 selected
farm-pest categories.

| Project label | IP102 ID | Class | Train | Validation | Test |
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

These are zero-based original labels in the IP102 split files. The tracked
source of truth is
[`data_manifests/broad15_classes.json`](data_manifests/broad15_classes.json).
The selection is an evidence-based quality shortlist, not an entomologist's
certification of every image. The dataset limitations, development history,
frozen Deep V2 validation result, and one-time test result are consolidated in
[`docs/MODEL_DEVELOPMENT_EVIDENCE.md`](docs/MODEL_DEVELOPMENT_EVIDENCE.md).

## Team rules

- Do not import a pretrained or prebuilt CNN and do not download weights.
- Do not commit the IP102 images or model checkpoints; both are ignored.
- Preserve IP102's official train, validation, and test splits.
- Use validation macro-F1 for model selection. Do not tune against the test set.
- Use the same Broad15 protocol for baseline and Deep V2 so the architecture is
  the controlled difference.
- Keep seed 42 for reproducibility. A later robustness study may repeat the
  chosen model with several seeds.

## Prepare the data

On Windows, create or refresh the project environment first:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then extract IP102 so this folder exists:

```text
IP102_v1.1/Classification/ip102_v1.1/
```

Generate the ignored Broad15 CSV manifests from the official split files:

```bash
python scripts/build_subset_manifests.py --definition data_manifests/broad15_classes.json
```

The command validates the class names and exact split totals. It does not create
a new random split. The older `scripts/setup_data.py` command creates the legacy
Rice10 manifests and is not the Broad15 preparation command.

## Retrain the two models

Use either the notebooks or the shared command-line trainer. Do not launch both
routes for the same run.

### Notebook route

1. Open [`notebooks/IP102_Broad15_Baseline_CNN.ipynb`](notebooks/IP102_Broad15_Baseline_CNN.ipynb).
2. Restart the kernel, confirm `QUICK_RUN = True`, and use **Run All** for a
   three-epoch pipeline check.
3. Restart the kernel again, set `QUICK_RUN = False`, and use **Run All** for the
   real baseline run.
4. Repeat the two checks with
   [`notebooks/IP102_Broad15_Deep_CNN_V2.ipynb`](notebooks/IP102_Broad15_Deep_CNN_V2.ipynb).

The notebooks report validation results only. That is intentional: do not use
the test results to decide between the two models.

### Command-line route

```bash
python -m src.train --config configs/broad15_baseline.yaml
python -m src.train --config configs/broad15_deep_v2.yaml
```

On a supported Intel Arc environment, `device: auto` selects PyTorch XPU. It
can also be requested explicitly with `--device xpu`. The trainer appends the
history and atomically writes `last_checkpoint.pt` after every completed epoch.
Resume the same run directory with:

```bash
python -m src.train --config configs/broad15_deep_v2.yaml \
  --resume runs/broad15/justin_deep_v2/<run_id>/last_checkpoint.pt
```

The resume checkpoint includes the model, optimizer, scheduler, global epoch,
early-stopping state, history, and main-process random-number states. On
Windows, `--num-workers 0` gives the most reliable stochastic continuation
because no persistent data-loader worker state exists outside the checkpoint.

Both configurations inherit
[`configs/_broad15_base.yaml`](configs/_broad15_base.yaml), which locks the
same manifests, image size, augmentation, normalization, optimizer settings,
learning rate, scheduler, epoch limit, early stopping, loss, and seed.

## Fair comparison and final test

Compare the best validation macro-F1, per-class F1, confusion matrix, parameter
count, checkpoint size, and CPU latency. Accuracy is useful but is not the main
score because Broad15 is imbalanced.

Only after the team freezes the winning architecture and settings, evaluate the
single selected run on the untouched test set:

```bash
python -m src.evaluate --run runs/broad15/<model>/<run_id>
```

Report both test accuracy and macro-F1, plus per-class results. Do not repeatedly
evaluate different variants on the test set. Evaluation artifacts are prefixed
with `validation_` or `test_`, and the command refuses to replace an existing
result unless `--force` is explicitly supplied.

## Models

| Experiment | Configuration | Scratch-built implementation |
|---|---|---|
| Broad15 baseline | `configs/broad15_baseline.yaml` | `src/models/justin_baseline_cnn.py` |
| Broad15 Deep V2 | `configs/broad15_deep_v2.yaml` | `src/models/justin_deep_cnn.py` |

Both must accept `[batch, 3, 160, 160]` and return `[batch, 15]` raw logits.
Softmax remains outside the model because cross-entropy expects logits.

Other model files in `src/models/` are team experiments or historical stubs;
they are not part of this two-model Broad15 rerun unless the team explicitly
adds them to the experiment plan.

## Run artifacts

The command-line trainer writes a timestamped directory under `runs/broad15/`.
The notebooks write to their named directories in the same Broad15 folder.
Typical artifacts include `best_model.pt`, `last_checkpoint.pt`,
`training_history.csv`, `train_summary.json`, learning curves, split-specific
confusion matrices, result JSON, and prediction CSV files. `runs/` is ignored by
Git, so back up important final artifacts separately.

## Unified model-development evidence

The complete evidence file preserves the historical Rice10 experiments while
clearly separating them from the active Broad15 dataset, controlled comparison,
warm-restart provenance, final validation/test metrics, and failure analysis:
[`docs/MODEL_DEVELOPMENT_EVIDENCE.md`](docs/MODEL_DEVELOPMENT_EVIDENCE.md).
