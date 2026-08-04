# IP102 pest classification — model benchmark

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
A shared harness for comparing pest-classification models on the same data, under
the same training protocol, scored the same way.

**Everyone owns one notebook.** You define your architecture in it; the harness
gives you the data, the training loop and the metrics. Because we all import the
same loaders and call the same `save_run`, the numbers in the comparison table
differ only by the thing we are actually comparing.

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
---

## Quick start

```bash
git clone <repo-url> && cd FarmPestManagementAI
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# One-time: build the manifests and normalization stats. Takes a few minutes.
.venv/bin/python scripts/setup_data.py

# 16 checks: split overlap, missing classes, unreadable images, box bounds,
# normalization. Run this before you train anything.
.venv/bin/python scripts/check_data.py
```

Then copy `notebooks/_TEMPLATE.ipynb` to `notebooks/<model>_<yourname>.ipynb` and
fill in the two cells marked **YOUR MODEL**. That is the whole workflow.

You need the IP102 images at the path in `protocol.yaml`
(`IP102_v1.1/Detection/VOC2007/JPEGImages`). They are not in the repo — 4.8 GB.
The template's first cell handles Colab, including mounting the images from Drive.

Runs on CUDA, Apple MPS or CPU with no code change (`device: auto`).

---

## The dataset

Fifteen classes of the **IP102 detection subset**, cropped to the annotated
bounding box with a 25% margin. Classes were screened on image quality, not raw
count: annotation coverage, how many images fall under 224px on the shorter
side, and the median fraction of the frame the pest occupies.

| Label | Class | Original IP102 | Train | Val | Test |
|---:|---|---:|---:|---:|---:|
| 0 | grub | 14 | 608 | 130 | 130 |
| 1 | mole_cricket | 15 | 298 | 64 | 63 |
| 2 | wireworm ⚠ | 16 | 34 | 7 | 8 |
| 3 | black_cutworm | 18 | 107 | 23 | 23 |
| 4 | corn_borer | 22 | 144 | 31 | 31 |
| 5 | army_worm | 23 | 615 | 132 | 132 |
| 6 | aphids | 24 | 142 | 30 | 31 |
| 7 | flea_beetle | 37 | 241 | 52 | 52 |
| 8 | flax_budworm | 45 | 287 | 62 | 61 |
| 9 | tarnished_plant_bug | 47 | 244 | 52 | 53 |
| 10 | blister_beetle | 51 | 654 | 140 | 141 |
| 11 | cicadella_viridis | 69 | 144 | 31 | 31 |
| 12 | miridae | 70 | 886 | 190 | 189 |
| 13 | prodenia_litura | 86 | 290 | 62 | 62 |
| 14 | cicadellidae | 101 | 2054 | 440 | 440 |
| | **total** | | **6748** | **1446** | **1447** |

Cropping to the box is what makes this subset learnable. The IP102 classification
images are web-scraped and roughly a third of them show crop damage, a lifecycle
diagram or a screenshot rather than the insect. The detection boxes cut straight
to the animal — worth roughly **+8 points** of accuracy on its own.

### Two things to say in the report

**Class 2 is too small to learn.** `wireworm` has 49 images in total, 34 of them
for training. Its per-class scores are noise, and because macro F1 weights every
class equally it drags the headline metric down by itself. It is kept for
taxonomic coverage — report that cost openly rather than quietly dropping the
class.

**The split is ours, not IP102's.** `data_manifests/splits_top15.json` is a
stratified 70/15/15 split, not the official IP102 split, so our numbers are not
directly comparable to published IP102 results. It is committed to git so
everyone trains on the identical split — do not regenerate it.

Also: the classes are imbalanced 60× (class 14 has 2054 training images, class 2
has 34). Hence weighted cross-entropy, and hence **macro F1 rather than
accuracy** as the primary metric — a model that ignores the small classes
entirely still scores well on accuracy. The majority class is 30.4% of the test
split: the floor any useful model must clear.

---

## The rules

1. **Never touch the test split** except through `save_run`. The best epoch is
   chosen by *validation* macro F1. Tuning anything against test invalidates
   every number in the table.
2. **Never edit `protocol.yaml` for your own run.** It is the controlled
   protocol. If the team agrees to change it, bump `protocol_version` and re-run
   every model — `compare.py` refuses to mix versions in one table.
3. **Set `pretrained=` honestly** in `save_run`. It is the one field that
   silently corrupts the comparison if it is wrong.
4. **Never commit images or checkpoints.** Both are already in `.gitignore`.
5. Work on a branch and open a PR. `main` should always be runnable.

---

## Writing your model

The architecture is yours. The contract is small:

- accepts `[B, 3, 160, 160]`
- returns `[B, 10]` **raw logits** — no softmax inside the model,
  `CrossEntropyLoss` expects logits

```python
from ip102_bench import load_protocol, build_dataloaders, train_model, save_run

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
protocol = load_protocol()
loaders  = build_dataloaders(protocol)

model  = MyNet(num_classes=protocol.num_classes)
result = train_model(model, protocol, loaders)

save_run(model=model, model_name='mynet', protocol=protocol, result=result,
         test_loader=loaders['test'], pretrained=False, author='your name')
```

Before a real run, use the template's overfit probe: a correct model reaches
~100% training accuracy on 64 images within a couple of hundred steps. If it
cannot, a full run will only spend hours proving the same thing.

`train_model` applies the protocol for you — AdamW, weighted cross-entropy,
ReduceLROnPlateau on validation macro F1, early stopping, best-epoch checkpoint.
Write your own loop if you prefer, but then matching `protocol.yaml` exactly is
on you.

### Pretrained models are allowed, and flagged

```python
from ip102_bench.models import build_pretrained
model = build_pretrained('resnet18', num_classes=protocol.num_classes)
# ... and pass pretrained=True to save_run
```

The comparison table keeps scratch and pretrained models in separate groups. A
fine-tuned ResNet is the external benchmark that tells us how much headroom our
own architectures are leaving on the table — it is not competing with them.

### Bringing a model you trained elsewhere

If you already trained something in another repo or Colab session and do not want
to retrain it:

```python
from ip102_bench import result_from_external_run, save_run

result = result_from_external_run(model, history_df)   # model has its trained weights
save_run(..., result=result, notes='trained externally, protocol not verified')
```

Nothing can verify the external run used our augmentation, schedule or split — so
say so in `notes`, and don't let it be read as a clean comparison.

---

## Comparing

```bash
.venv/bin/python -m ip102_bench.compare
.venv/bin/python -m ip102_bench.compare --csv comparison.csv
```

Collects every `results.json` under `runs/`, sorted by macro F1. Runs recorded
under a different protocol version or subset are listed separately rather than
mixed in.

Macro F1 is the primary metric, but pick the final model on application fit: this
runs offline, so CPU latency and model size are real costs. All three are
recorded for every run.

---

## What each run writes

`runs/<model_name>/<run_id>/`:

```
results.json          metrics, params, latency, protocol snapshot, environment
training_history.csv  per-epoch losses and metrics
best_model.pt         weights at the best validation macro F1
predictions.csv       per-image prediction — this is your error analysis input
curves.png            train vs validation accuracy and loss
confusion_matrix.png  counts, shaded by row fraction
per_class_f1.png      sorted worst-first
```

## Using the selected CNN in the Streamlit app

After training, choose the run with the strongest **macro F1**. Do not rename or move individual
files: keep the complete artifact pair in this location:

```text
runs/<model_name>/<run_id>/
  best_model.pt
  results.json
```

The app reads every `results.json`, selects the run with the highest `macro_f1`,
then loads its neighbouring `best_model.pt` automatically when it starts.
`save_run(...)` already produces this layout. For the completed in-repository
model, train and save it with `model_name="alexnet"`; the app can then load it
without any configuration change.

If the selected model uses a new architecture, keep its Python architecture
definition in `ip102_bench/models/`, register it in `SCRATCH_REGISTRY`, and use
the same `model_name` in `save_run`. This is required because a PyTorch state
dictionary contains weights, not the model's layer definitions.

For the error analysis, filter `predictions.csv` on `correct == 0` and sort by
confidence descending. The confident mistakes are where the model learned
something wrong; the low-confidence ones are just hard images.

---

## Layout

```
protocol.yaml         the locked protocol — image size, splits, optimizer, seed

ip102_bench/          the shared harness
  protocol.py         loads and validates protocol.yaml
  data.py             dataset, box cropping, dataloaders, class weights
  transforms.py       the locked augmentation and eval pipelines
  training.py         train_model — the locked training loop
  artifacts.py        save_run — the run artifact contract
  metrics.py          macro F1, per-class, params, CPU latency
  plots.py            the three standard figures
  compare.py          the comparison table
  models/             optional reference architectures + build_pretrained

app/                  the offline pest-assistant chatbot — see below
notebooks/            one per person; _TEMPLATE.ipynb is the starting point
scripts/              setup_data.py, check_data.py, import_propestnet_run.py
docs/propestnet.md    ProPestNet: architecture, results, ablation, dataset
data_manifests/       the committed dataset definition + generated CSVs
  splits_top15.json   the train/val/test split everyone shares
  boxes_top15.json    one box per image: the largest annotated object
  classes_top15.json  label -> slug and farmer-facing display name
  *.csv, norm_stats.json   generated by setup_data.py, git-ignored
runs/                 checkpoints and results (git-ignored)
tests/                pytest suite — architecture, checkpoint, assistant
archive/              superseded rice-10 handoff notes and benchmark script
```

## The chatbot

`app/` is the farmer-facing side: upload a photo, get an identification and
organic treatment guidance, fully offline. It serves **ProPestNet**, the model
documented in `docs/propestnet.md`, at its own preprocessing (128px, ImageNet
normalization) rather than the harness defaults — the app reads those settings
from the checkpoint, so the two can never drift apart silently.

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py    # chat UI
.venv/bin/python -m uvicorn app.main:app --port 8000      # /analyze, /chat, /health
```

The app loads the highest-scoring run from `runs/`. To make a trained ProPestNet
checkpoint available to it, see `scripts/import_propestnet_run.py --help`. With
no usable checkpoint the app says so in a banner rather than serving noise.

### The language model

The conversation runs on a **local** model through Ollama. It is optional, and
the app tells you in the sidebar whether it found one:

```bash
brew install ollama && ollama serve
ollama pull phi3            # or llama3.2:3b / qwen2.5:3b, then export OLLAMA_MODEL
```

Without it the assistant still identifies pests and still answers, straight from
the guides in `app/treatment_guides.py` — the classifier and the written advice
are the product; the language model only rephrases them around the farmer's own
question. Nothing leaves the machine either way.

`OLLAMA_BASE_URL`, `OLLAMA_MODEL` and `OLLAMA_TIMEOUT` override the defaults.

### Context

Three things make it a conversation rather than a photo endpoint, all in
`app/conversation.py`:

- **History** — the last few turns go to the model, so follow-up questions work.
- **The pest in hand** — `PestContext` records what the last photo was identified
  as, from the classifier's own top-1, so "how often do I spray it?" needs no
  second upload.
- **Grounding** — `app/retrieval.py` scores the fifteen treatment guides against
  the question and attaches the best ones to the system prompt, under an
  instruction not to contradict them. Keyword scoring, not embeddings: fifteen
  short documents do not need a vector index, and it keeps the app dependency-free.

Conversations and their photos are saved under `.chats/` (git-ignored), so
closing the laptop does not lose an identification.

## Switching datasets

`protocol.yaml` defines three subsets: `detection_top15` (default), `rice10` (the
old ten-class rice subset from the classification images) and `all` (full 102
classes). Switching is one line plus a rebuild:

```bash
.venv/bin/python scripts/setup_data.py --subset rice10
```

Bump `protocol_version` if the team switches for real, so old runs don't quietly
end up in the same table.
