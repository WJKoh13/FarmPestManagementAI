# IP102 pest classification — model benchmark

A shared harness for comparing pest-classification models on the same data, under
the same training protocol, scored the same way.

**Everyone owns one notebook.** You define your architecture in it; the harness
gives you the data, the training loop and the metrics. Because we all import the
same loaders and call the same `save_run`, the numbers in the comparison table
differ only by the thing we are actually comparing.

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

Top ten classes of the **IP102 detection subset**, cropped to the annotated
bounding box with a 15% margin.

| Label | Class | Original IP102 | Train | Val | Test |
|---:|---|---:|---:|---:|---:|
| 0 | grub | 14 | 305 | 65 | 66 |
| 1 | mole_cricket | 15 | 608 | 130 | 130 |
| 2 | wireworm | 16 | 298 | 64 | 63 |
| 3 | corn_borer | 22 | 298 | 64 | 63 |
| 4 | aphids | 24 | 615 | 132 | 132 |
| 5 | locustoidea ⚠ | 48 | 373 | 80 | 80 |
| 6 | legume_blister_beetle | 50 | 304 | 65 | 66 |
| 7 | blister_beetle | 51 | 654 | 140 | 141 |
| 8 | miridae | 70 | 886 | 190 | 189 |
| 9 | cicadellidae | 101 | 2054 | 440 | 440 |
| | **total** | | **6395** | **1370** | **1370** |

Cropping to the box is what makes this subset learnable. The IP102 classification
images are web-scraped and roughly a third of them show crop damage, a lifecycle
diagram or a screenshot rather than the insect. The detection boxes cut straight
to the animal.

### Two things to say in the report

**Class 5 is mislabeled upstream.** `Locustoidea` should be locusts and
grasshoppers. Every sample inspected is a mirid plant bug, which makes it a
near-duplicate of class 8 (`miridae`). Expect 5↔8 confusion in every model's
matrix — it is a dataset fault, not a modelling failure. Do not let anyone
present it as one.

**The split is ours, not IP102's.** `image_refactor/splits_top10.json` is a stratified 70/15/15
split, not the official IP102 split, so our numbers are not directly comparable
to published IP102 results. It is committed to git so everyone trains on the
identical split — do not regenerate it.

Also: the classes are imbalanced 6.9× (class 9 has 2054 training images, classes
2 and 3 have 298). Hence weighted cross-entropy, and hence **macro F1 rather than
accuracy** as the primary metric — a model that ignores the small classes
entirely still scores well on accuracy.

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

For the error analysis, filter `predictions.csv` on `correct == 0` and sort by
confidence descending. The confident mistakes are where the model learned
something wrong; the low-confidence ones are just hard images.

---

## Layout

```
protocol.yaml         the locked protocol — image size, splits, optimizer, seed
image_refactor/       the committed dataset definition — do not regenerate these
  splits_top10.json   the train/val/test split everyone shares
  boxes_top10.json    one box per image: the largest annotated object

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

notebooks/            one per person; _TEMPLATE.ipynb is the starting point
scripts/              setup_data.py, check_data.py
data_manifests/       generated CSVs, classes.json, norm_stats.json
runs/                 checkpoints and results (git-ignored)
archive/              superseded rice-10 handoff notes and benchmark script
```

## Switching datasets

`protocol.yaml` defines three subsets: `detection_top10` (default), `rice10` (the
old ten-class rice subset from the classification images) and `all` (full 102
classes). Switching is one line plus a rebuild:

```bash
.venv/bin/python scripts/setup_data.py --subset rice10
```

Bump `protocol_version` if the team switches for real, so old runs don't quietly
end up in the same table.
