# VGG19 (XML-cropped) — Beatrice

A from-scratch VGG19 trained on per-object crops taken straight from the VOC2007
annotation XML. Notebook: [`notebooks/Beatrice_vgg19_xml_cropped.ipynb`](../notebooks/Beatrice_vgg19_xml_cropped.ipynb).
Architecture module: [`ip102_bench/models/vgg_cnn.py`](../ip102_bench/models/vgg_cnn.py),
registered as `vgg19_beatrice`.

No weight is pretrained — this is configuration E of Simonyan & Zisserman built from
PyTorch primitives, not `torchvision.models.vgg19`.

## Architecture

The 2014 network as published: sixteen 3x3 convolutions in five blocks, no BatchNorm, and
the three fully connected layers rather than a global-average-pool head.

```text
128x128x3  -> block 1: 2x conv 64,  maxpool  ->  64x64x64
           -> block 2: 2x conv 128, maxpool  ->  32x32x128
           -> block 3: 4x conv 256, maxpool  ->  16x16x256
           -> block 4: 4x conv 512, maxpool  ->   8x8x512
           -> block 5: 4x conv 512, maxpool  ->   4x4x512
           -> adaptive average pool to 7x7   ->   7x7x512
           -> Linear(25088, 4096) -> Linear(4096, 4096) -> Linear(4096, 15)
```

**139,631,695 parameters** — and roughly 102M of them are in the first `Linear(25088, 4096)`
alone. That single layer holds nine times the entire ProPestNet network. It is the cost of
keeping the classifier faithful to the paper instead of pooling globally, and it is the
number to quote when the report asks what the FC head is worth.

Two consequences worth knowing before serving it:

- **532 MB on disk, ~28 ms per image on CPU** for a single pass — about 110 ms through the
  app's four-pass TTA. ProPestNet is 11M parameters by comparison. This app runs offline on
  modest hardware, so size and latency are real costs, not footnotes.
- **It cannot run on Apple MPS at 128px.** Block 5 emits a 4x4 feature map and the adaptive
  pool asks for 7x7; Metal has no kernel for a non-divisible adaptive pool
  ([pytorch#96056](https://github.com/pytorch/pytorch/issues/96056)). Harmless in production
  — the app only ever uses CUDA or CPU — but it means no GPU acceleration on a Mac, and
  `scripts/import_vgg19_run.py` falls back to CPU automatically rather than dying halfway
  through an evaluation.

## How it was trained

Her notebook does not use the shared harness. It parses the annotation XML itself, emits
**one crop per annotated object** rather than one per image, and runs its own training loop:

| | Beatrice's run | The locked protocol |
|---|---|---|
| Input | 128px, plain square resize | 160px, resize 1.14x then centre crop |
| Crop padding | 0.05 | 0.25 |
| Normalization | ImageNet statistics | `data_manifests/norm_stats.json` |
| Loss | Cross-entropy, unweighted | Inverse-frequency weighted |
| Optimizer | Adam, lr 1e-4, wd 5e-4 | AdamW, lr 1e-3, wd 1e-4 |
| Schedule | ReduceLROnPlateau on val **loss** | ReduceLROnPlateau on val **macro F1** |
| Epochs | 10, batch 8 | 60, batch 32, early stopping |
| Best epoch chosen by | Validation **accuracy** | Validation **macro F1** |

The split is the shared one — she reads the same `data_manifests/splits_top15.json`, and her
label ids are the project labels in the same order as `classes_top15.json`. That agreement is
what makes her weights servable under the app's class names at all, so
`scripts/import_vgg19_run.py` asserts it rather than assuming it.

Because the classes are imbalanced 60:1, choosing the best epoch on **accuracy** rather than
macro F1 systematically favours the large classes. That is the single biggest reason her run
is not a like-for-like entry in the benchmark table.

## Results

<!-- Filled from runs/vgg19_beatrice/<run_id>/results.json -->

_Pending: written when her checkpoint is imported._

Her notebook records only a best **validation accuracy** — it never computes macro F1, the
project's primary metric. So the importer scores the project's test split itself, under her
preprocessing, and writes a real `macro_f1` into `results.json`. Without that the run would
carry no comparable number at all and the app would rank it last by default.

## Getting it into the app

Checkpoints are git-ignored, so her `xml_crop_vgg19_best.pth` has to arrive as a file rather
than through git:

```bash
python scripts/import_vgg19_run.py --checkpoint /path/to/xml_crop_vgg19_best.pth
```

That writes `runs/vgg19_beatrice/<run_id>/` with the weights plus the preprocessing contract
— `image_size`, `mean`, `std`, `crop_mode`, `crop_margin` — read back by
[`app/cnn_model.py`](../app/cnn_model.py) when the app rebuilds her transform. Nothing about
her pipeline is inferred from the protocol; a model trained on 0.05-padding 128px crops and
served through the protocol's 0.25-margin 160px pipeline would not error, it would just be
quietly worse at every prediction.

Optionally, `python scripts/evaluate_tta.py runs/vgg19_beatrice/<run_id>` records what she
scores through the app's own test-time augmentation, which is the setting the app serves
every model under.

The run is flagged `comparable_to_main: false` with a `protocol_version` that
`ip102_bench.compare` filters out of the protocol-v1 table. The measured test macro F1 is a
real, comparable score; the *run* is not a protocol entry. Both statements are true and the
bundle records both.
