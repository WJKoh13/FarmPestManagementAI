# FarmPestManagementAI

Ten-class rice-pest classification on the IP102 dataset, using six CNN
architectures written from scratch in PyTorch. Intended for offline use, so model
size and CPU latency count alongside accuracy.

## Team rules

- Never import a pretrained or prebuilt CNN. No `torchvision.models`, no
  downloaded weights. The classic architectures are design references only.
- Never commit images or checkpoints. Both are already in `.gitignore`.
- Never change the official test split, and never use test performance to pick an
  epoch or tune a hyperparameter. The validation split decides everything.
- Never silently change anything in `configs/_base.yaml`. That file is the
  controlled protocol; if the team agrees to change it, record the change and
  re-run every model.
- Ask for a code review before starting a long training run.

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Extracts the dataset and builds the manifests. Asserts 4318/721/2166.
.venv/bin/python scripts/setup_data.py --tar ~/Downloads/ip102_v1.1.tar

# One-time, already committed: RGB mean/std from the training split only.
.venv/bin/python scripts/compute_norm_stats.py

# 14 validation checks + the augmentation preview image.
.venv/bin/python scripts/check_data.py
```

Runs on CUDA, Apple MPS or CPU without any code change (`device: auto`).

## The ten classes

| Project label | Original IP102 ID | Pest | Train | Val | Test |
|---:|---:|---|---:|---:|---:|
| 0 | 0 | rice_leaf_roller | 669 | 111 | 335 |
| 1 | 1 | rice_leaf_caterpillar | 292 | 48 | 147 |
| 2 | 3 | asiatic_rice_borer | 631 | 106 | 316 |
| 3 | 4 | yellow_rice_borer | 302 | 50 | 152 |
| 4 | 5 | rice_gall_midge | 303 | 51 | 152 |
| 5 | 7 | brown_plant_hopper | 500 | 83 | 251 |
| 6 | 8 | white_backed_plant_hopper | 535 | 90 | 268 |
| 7 | 9 | small_brown_plant_hopper | 331 | 56 | 166 |
| 8 | 10 | rice_water_weevil | 513 | 86 | 257 |
| 9 | 11 | rice_leafhopper | 242 | 40 | 122 |
| | | **total** | **4318** | **721** | **2166** |

Splits come straight from the official IP102 `train.txt` / `val.txt` / `test.txt`.
No new random split is ever created.

## Who owns what

Four models, one member each:

| Model | Config | File | Status |
|---|---|---|---|
| AlexNet-style | `configs/alexnet.yaml` | `src/models/alexnet_cnn.py` | done, 1,795,018 params |
| VGG16-style (config D) | `configs/vgg16.yaml` | `src/models/vgg_cnn.py` | stub |
| VGG19-style (config E) | `configs/vgg19.yaml` | `src/models/vgg_cnn.py` | stub |
| Own shallow baseline | `configs/baseline.yaml` | `src/models/baseline_cnn.py` | stub |

Unassigned spares, only if the group wants more comparison rows:

| Model | Config | File |
|---|---|---|
| GoogLeNet/Inception-style | `configs/googlenet.yaml` | `src/models/googlenet_cnn.py` |
| Custom residual | `configs/residual.yaml` | `src/models/residual_cnn.py` |
| Lightweight separable | `configs/lightweight.yaml` | `src/models/lightweight_cnn.py` |

Each stub file contains the full architecture spec, the suggested layer layout,
the target parameter count and the acceptance checks. Use
`src/models/alexnet_cnn.py` as the worked example for conventions.

VGG16 and VGG19 differ only in their layer configuration (13 vs 16 conv layers),
which is what makes the pair a clean depth ablation.

## Writing your model

1. Fill in your file. It must accept `[B, 3, 160, 160]` and return `[B, 10]` raw
   logits. No softmax inside the model - `CrossEntropyLoss` expects logits.
2. Check the shape, parameter count and that gradients flow:
   ```bash
   .venv/bin/python -m src.summarize --check
   ```
   Stay inside the agreed 0.5M-5M parameter budget and record your exact count.
3. Prove the wiring works before burning hours on a real run:
   ```bash
   .venv/bin/python scripts/overfit_test.py --model residual
   ```
   It must reach ~100% training accuracy on 64 images. If it cannot, you have a
   bug - the script prints the list of usual suspects.
4. Train and evaluate:
   ```bash
   .venv/bin/python -m src.train --config configs/residual.yaml
   .venv/bin/python -m src.evaluate --run runs/residual/<run_id>
   ```

Roughly 20-25 s per epoch for the baseline on an M3 (4,318 images, batch 32), so
a full 60-epoch run is about 20-30 minutes. Deeper models take proportionally
longer.

## Run artifacts

Every run writes `runs/<model>/<run_id>/`:

```
config.yaml             the exact config used - the run is reproducible from this
best_model.pt           checkpoint at the best validation macro F1
training_history.csv    per-epoch losses and metrics, written as it goes
loss_curve.png          train vs validation loss
metric_curve.png        validation accuracy and macro F1
results.json            test metrics, params, size, CPU latency
confusion_matrix.png    counts, shaded by row fraction
predictions.csv         per-image prediction with filenames and class names
```

`predictions.csv` is what you use for the required error analysis - filter on
`correct == 0` and inspect at least ten failures.

## Final comparison

```bash
.venv/bin/python -m src.summarize --compare
```

Collects every `results.json` into the comparison table. Macro F1 is the primary
metric, but pick the final model on application suitability: it runs offline, so
CPU latency and file size matter too.

## Layout

```
configs/          _base.yaml holds the locked protocol; model YAMLs extend it
data_manifests/   generated CSVs (git-ignored) + selected_classes.json
scripts/          setup_data, compute_norm_stats, check_data, overfit_test,
                  make_smoke_manifests
src/data/         dataset.py, transforms.py
src/models/       one file per architecture + the build_model registry
src/utils/        seed, device, metrics, plots
src/              train.py, evaluate.py, predict.py, summarize.py, config.py
runs/             checkpoints and results (git-ignored)
```
