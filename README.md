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

## The app

`app/` is the farmer-facing side: upload a photo, get an identification and
organic treatment guidance, fully offline. **`docs/architecture.md` explains the
whole thing in plain language** and is the place to start.

Three components, three processes' worth of separation:

| | | |
|---|---|---|
| **Front end** | `app/streamlit_app.py` | The screen. Holds no model and makes no decisions. |
| **Back end** | `app/main.py` | FastAPI. Runs the agent, owns the photo store, keeps no session. |
| **AI engine** | ProPestNet + qwen2.5:3b + `docs/knowledge/` | Everything that knows something. |

```bash
.venv/bin/python scripts/run_app.py     # starts both, waits for the backend, opens :8501
```

Or separately:

```bash
.venv/bin/python -m uvicorn app.main:app --port 8000
.venv/bin/python -m streamlit run app/streamlit_app.py
```

It serves **ProPestNet** (`docs/propestnet.md`) at its own preprocessing — 128px,
ImageNet normalization — read from the checkpoint, so the two cannot drift apart
silently. The app loads the highest-scoring run in `runs/`; see
`scripts/import_propestnet_run.py --help` to add one. With no usable checkpoint
it says so in a banner rather than serving noise.

### Function calling

The language model decides for itself when to run the CNN. It is given three
tools and picks:

| Tool | Returns |
|---|---|
| `classify_pest_image` | The CNN's verdict on an attached photo |
| `lookup_treatment_guide` | The vetted guide — the only source of a product or dose |
| `search_knowledge_base` | Background passages from the reference library |

No branch in the code decides to classify. The message merely says a photo exists
and where; the model asks for the classifier. The tool trace above each answer
shows what it chose, and marks anything the app added rather than the model.

`app/agent.py` runs the loop, `app/agent_tools.py` holds the schemas and their
implementations side by side, and the model never receives a confidence number —
the certainty decision is made in Python against `CONFIDENCE_FLOOR`.

To watch it without a browser:

```bash
.venv/bin/python scripts/try_agent.py --image sample_images/aphids__IP025000243.jpg "on my kale"
```

### The language model

Local, through Ollama, and it must support tool calling:

```bash
brew install ollama && ollama serve
ollama pull qwen2.5:3b        # primary, sub-3B, instruction-tuned
ollama pull qwen2.5:1.5b      # faster fallback, used automatically if 3b is absent
ollama pull nomic-embed-text  # embeddings for the reference library
```

Without it the app still identifies pests and still answers, from the guides in
`app/treatment_guides.py`. Nothing leaves the machine either way.
`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_KEEP_ALIVE` and
`OLLAMA_TIMEOUT` override the defaults; `PEST_API_URL` points the front end at
the back end.

### The reference library (RAG)

`docs/knowledge/` is markdown prose about organic pest management — how the
inputs work, how to scout, how to read damage. `scripts/build_knowledge_index.py`
chunks it by heading, embeds each passage with `nomic-embed-text`, and writes
`data_manifests/knowledge_index.json`, which is committed so a bare clone has
working search with no Ollama and no build step.

Retrieval is cosine similarity over normalised vectors — one dot product, no
vector database, because forty passages do not need one. If the embedding model
is missing it falls back to keyword scoring.

**The library never states a dose or an interval.** Those live only in
`treatment_guides.py`, which is short and hand-checked;
`tests/test_knowledge.py` greps the corpus and fails if one appears.

### Context

Three things make it a conversation rather than a photo endpoint:

- **History** — the last few turns go to the model, so follow-up questions work.
- **The pest in hand** — `PestContext` records what the last photo was identified
  as, from the classifier's own top-1, so "how often do I spray it?" needs no
  second upload.
- **Grounding** — the model must fetch a guide before advising. If it tries to
  answer without one, the app fetches it and marks the step as its own.

A pest the classifier was *not* confident about is refused by the guide tool, so
an uncertain guess can never be laundered into confident instructions — and the
refusal survives into later turns.

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
