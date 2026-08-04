# ProPest — Organic Farm Pest Management AI System

The vision component of the offline pest-management consultant: a farmer photographs a
pest, the model names it, and the advisor layer maps that name onto certification-safe
organic treatments.

## The submitted model — `ProPestNet.ipynb`

**ProPestNet** is an original convolutional architecture, designed for this problem and
implemented from scratch in PyTorch. Nothing comes from `torchvision.models`, no weights
are loaded, and no published architecture is reproduced. The only third-party code is
`torchvision.transforms` for augmentation and PyTorch's standard layer primitives
(`Conv2d`, `BatchNorm2d`, `Linear`) — the building blocks, not the building.

The notebook runs end to end: dataset construction, the architecture and the evidence
behind every design decision, the formulas for each layer type, model summary, training,
accuracy and loss curves, held-out test evaluation, single-image inference, and an
ablation that retrains the network with each design decision removed.

### Why this architecture is original

Rather than reproduce a known network, each part of ProPestNet answers something we
measured about IP102 photographs:

| What we measured | Design decision it forced |
|---|---|
| Insect size in frame varies enormously — median 43%, under a quarter in 18% of images; test accuracy runs 58.5% → 70.9% across that range | **Multi-scale stem**: 3x3, 5x5 and 7x7 convolutions in *parallel* on the raw pixels, concatenated. One kernel size is the wrong bet on data this varied |
| Errors concentrate in two near-identical pairs — the two blister beetles, and Miridae vs Locustoidea. They share colour and silhouette; a few fine cues separate them | **Squeeze-and-excitation channel gate** in every block, so the network can amplify the handful of channels that separate two similar beetles and mute the ones they share |
| Small insects cannot afford lossy downsampling — max-pooling discards three of every four activations by a fixed rule | **Learned stride-2 convolutions** for every downsampling step; the network decides what to discard |
| 6,395 training images. A flatten-then-`Linear(8192, 512)` head would hold 4.2M parameters on its own — the classic overfitting culprit | **Two-scale global-average-pooling head**: pool stages 3 and 4, concatenate, `Linear(640, 15)` = 9,615 parameters |
| Nine convolutional blocks on 6,395 images means gradients travel a long way | **Identity skip in every block**, with the second BatchNorm initialised to gamma = 0.1 so blocks start near pass-through and the network deepens itself |

```text
128x128x3   -> MultiScaleStem (3x3|5x5|7x7, s2)      ->  64x64x96
            -> stage 1: transition 96->64,     2x PestBlock ->  64x64x64
            -> stage 2: transition 64->128 s2, 2x PestBlock ->  32x32x128
            -> stage 3: transition 128->256 s2, 3x PestBlock ->  16x16x256 --.
            -> stage 4: transition 256->384 s2, 2x PestBlock ->   8x8x384 --.|
                                    global average pool both -> 640 features <'
                                    dropout(0.4) -> Linear(640, 15)
```

`PestBlock` = `Conv3x3-BN-ReLU -> Conv3x3-BN -> ChannelGate -> + input -> ReLU`. Channel
count is preserved inside a block, so the skip is a pure identity with no projection
anywhere in the network.

**10,988,015 parameters**, of which 99.9% sit in the convolutional stages and 6,410 in
the classifier head. Capacity goes into learning features, not into memorising the
training set.

### Results

<!-- From the executed notebook: 15 classes, 60 epochs, seed 42. -->

| Metric | ProPestNet | + test-time augmentation |
|---|---:|---:|
| Test accuracy | 69.18% | **72.63%** |
| Test macro-F1 | 0.6077 | **0.6378** |
| Test macro-precision | 0.5867 | 0.6138 |
| Test macro-recall | 0.6725 | 0.6849 |
| Top-3 accuracy | 86.52% | — |
| Majority-class baseline | 30.41% | 30.41% |
| Parameters | 10,988,015 | 10,988,015 |
| Input resolution | 128 x 128 | 128 x 128 |
| Epochs | 60 of 60 (no early stop) | — |
| Training time | 123 min on Apple-Silicon MPS | — |

Against a 30.4% majority-class baseline on 15 classes, 72.6% is **2.4x baseline**. The
**top-3 accuracy of 86.5%** matters more than top-1 for the intended use: showing a farmer
three candidates with confidences is more useful than one confident-looking guess.

#### Test-time augmentation was worth more than expected

Averaging the softmax over deterministic views costs nothing at training time and gained
**+3.5 points of accuracy, +0.030 macro-F1**. The setting was chosen on validation and
applied to the test split exactly once:

| Setting | Passes | Val accuracy | Val macro-F1 |
|---|---:|---:|---:|
| centre only (no TTA) | 1 | 0.7047 | 0.6084 |
| centre + mirror | 2 | 0.6999 | 0.6043 |
| centre + whole | 2 | 0.7372 | 0.6446 |
| **centre + whole + mirror** | 4 | **0.7407** | **0.6447** |

Note *which* view did the work. Mirroring alone made things slightly **worse**. The whole
gain comes from the `whole` view — squeezing the entire bounding-box crop into the input
instead of centre-cropping it. That is the framing hypothesis confirming itself: framing
is this dataset's dominant nuisance variable, the same finding that made bounding-box
cropping worth +8 points.

#### The schedule fix, and the evidence it was needed

An earlier run set `CosineAnnealingLR(T_max=60)` with `PATIENCE = 8`, and early-stopped at
epoch 44 with the learning rate still at ~17% of peak — the low-learning-rate endgame
never happened. Raising patience to 20 let the schedule finish, and **the best validation
score then landed at epoch 57 of 60**, inside the window the old setting cut off. The
defect was real, not theoretical.

#### Where the accuracy actually goes — read the per-class table

| Pest | Images | Test F1 |
|---|---:|---:|
| grub | 868 | 0.854 |
| aphids | 203 | 0.788 |
| flea beetle | 345 | 0.784 |
| blister beetle | 935 | 0.784 |
| Cicadellidae | 2,934 | 0.783 |
| mole cricket | 425 | 0.773 |
| army worm | 879 | 0.748 |
| Miridae | 1,265 | 0.648 |
| Cicadella viridis | 206 | 0.598 |
| Prodenia litura | 414 | 0.569 |
| corn borer | 206 | 0.452 |
| tarnished plant bug | 349 | 0.441 |
| flax budworm | 410 | 0.427 |
| **black cutworm** | **153** | **0.367** |
| **wireworm** | **49** | **0.100** |

Macro-F1 weights every class equally, so **wireworm's 49 images count exactly as much as
Cicadellidae's 2,934**. Those two smallest classes drag macro-F1 down by roughly 0.06 on
their own — more than test-time augmentation gained. That is a property of the class list,
not a failure of the model: with 34 training images, no architecture learns wireworm.
Accuracy (69.2%) is far less sensitive to this than macro-F1 (0.608), which is why both
are reported.

### The ablation — evidence that each decision earns its place

Section 11 retrains the network three times, adding one group of decisions at a time, and
reports **validation** macro-F1 (the test split is never used for a design decision). All
three variants are the same `ProPestNet` class with different constructor flags, so
nothing but the ablated decision differs.

| Variant | Stem | Skips | SE gate | Downsampling | Head | Params |
|---|---|---|---|---|---|---:|
| A — plain control | single 3x3 | no | no | max-pool | flatten + `Linear(6144, 512)` | 14,056,810 |
| B — + skips, learned downsampling | single 3x3 | yes | no | stride-2 conv | flatten + `Linear(6144, 512)` | 14,056,810 |
| C — full ProPestNet | 3x3 / 5x5 / 7x7 | yes | yes | stride-2 conv | two-scale GAP | 10,988,015 |

**Results after 12 epochs each — and they do not all support the design:**

| Variant | Params | Val accuracy | Val macro-F1 | vs previous |
|---|---:|---:|---:|---:|
| A — plain control | 14,059,375 | 0.325 | 0.264 | — |
| B — + skips, learned downsampling | 14,059,375 | **0.453** | **0.392** | **+0.128** |
| C — full ProPestNet | 10,988,015 | 0.384 | 0.335 | −0.057 |

Two honest conclusions:

1. **Skip connections and learned downsampling are decisively worth it**: +0.128 macro-F1,
   by far the largest single effect measured, and they cost no parameters at all.
2. **The multi-scale stem, channel gate and pooling head did not improve accuracy.** They
   *lost* 0.057 macro-F1 against variant B. What they deliver instead is efficiency: the
   same job with **3.07M fewer parameters**, a 22% reduction.

The design rationale above predicted those parts would improve accuracy, and they did not.
This is reported as measured rather than argued away. The same comparison run on a
different ten-class subset gave the same ordering (B ahead by 0.010), so it replicates
rather than being a one-off.

Two caveats, neither established: the ablation trains 12 epochs while the deliverable
trains 60, and channel gates are known to converge slowly; and variant C has 22% less
capacity to work with. Settling it would mean running both at full length, which was not
done.

It costs about an hour, so it is opt-in:

```bash
PROPEST_ABLATION=1 jupyter execute project/ProPestNet.ipynb
```

Results cache to `project/runs/ablation/results.json` and the table reprints instantly on
later runs.

### Getting it to work — bugs worth knowing about

These were found while building the pipeline, and all four still apply:

1. **Learning rate 3x too high.** Early attempts sat at chance (train accuracy 13% on a
   10-class problem). Diagnosed by testing whether the network could overfit an 800-image
   subset with augmentation off: SGD lr 0.01 reached 0.27, AdamW 1e-3 reached 0.39, AdamW
   **3e-4** reached 0.78. That is why the notebook uses 3e-4 and does not re-tune it.
2. **Evaluation preprocessing did not match training.** The eval transform resized to
   256px then centre-cropped 128px, discarding 75% of every validation and test image by
   area, while training used `RandomResizedCrop` over nearly the whole image. Fixing it to
   `Resize(146) -> CenterCrop(128)` moved validation macro-F1 at epoch 2 from 0.17 to 0.32
   — the score the broken version needed twelve epochs to reach.
3. **Too much background, not enough insect.** The pest fills a median 43% of an IP102
   frame, but in 18% of images it is under a quarter and grass or soil dominates.
   Bucketing test accuracy by insect size showed a clean 12-point spread (58.5% under 15%
   of frame, 70.9% over half). Cropping every image to its annotated bounding box plus a
   25% margin was worth roughly **+8 points** — the single biggest improvement, and the
   measurement that later motivated ProPestNet's multi-scale stem.
4. **Inference did not crop the way training did.** `predict_pest()` opened images raw, so
   it fed uncropped photos to a model trained on crops. Caught because top-1 accuracy
   disagreed with the confusion matrix on identical images. They now agree exactly.

Section 13 of the notebook writes this up as a reusable debugging playbook.

### Dataset

Only the *Detection* half of IP102 is on disk
(`data/IP102_v1.1/Detection/VOC2007/JPEGImages.tar`, 18,981 images). The *Classification*
half with its official `train/val/test.txt` splits is not present. IP102 detection
filenames encode the class (`IP<class><id>.jpg`), so the notebook recovers labels from
filenames, extracts the archive into `data/IP102_v1.1/Classification/images_det/`, and
builds a seeded stratified 70/15/15 split cached in `project/splits_top15.json`.

The fifteen classes selected by the team, with their IP102 ids (1-based, indexing
`Classification/classes.txt`):

| Label | IP102 class | Pest | Train | Val | Test | Total |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 15 | grub | 608 | 130 | 130 | 868 |
| 1 | 16 | mole cricket | 298 | 64 | 63 | 425 |
| 2 | 17 | wireworm | 34 | 7 | 8 | 49 |
| 3 | 19 | black cutworm | 107 | 23 | 23 | 153 |
| 4 | 23 | corn borer | 144 | 31 | 31 | 206 |
| 5 | 24 | army worm | 615 | 132 | 132 | 879 |
| 6 | 25 | aphids | 142 | 30 | 31 | 203 |
| 7 | 38 | flea beetle | 241 | 52 | 52 | 345 |
| 8 | 46 | flax budworm | 287 | 62 | 61 | 410 |
| 9 | 48 | tarnished plant bug | 244 | 52 | 53 | 349 |
| 10 | 52 | blister beetle | 654 | 140 | 141 | 935 |
| 11 | 70 | Cicadella viridis | 144 | 31 | 31 | 206 |
| 12 | 71 | Miridae | 886 | 190 | 189 | 1,265 |
| 13 | 87 | Prodenia litura | 290 | 62 | 62 | 414 |
| 14 | 102 | Cicadellidae | 2054 | 440 | 440 | 2,934 |
| | | **total** | **6,748** | **1,446** | **1,447** | **9,641** |

Split sizes: **6,748 train / 1,446 validation / 1,447 test**.
The majority class (Cicadellidae) is 30.4% of the test split — the floor any
useful model must clear.

Two classes are far too small to learn: **wireworm has 49 images total (34 for
training)** and black cutworm 153. They are kept because the class list is shared
across the team, and their cost is reported openly in the per-class table above rather
than hidden by dropping them.

Every image is cropped to its annotated bounding box plus a 25% margin before any other
transform; boxes are available for 99.99% of these images and cached in
`project/boxes_top15.json`.

### Running it

Set up the environment once (PyTorch does not support Python 3.14, so use 3.13):

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib numpy pillow ipykernel torchinfo
python -m ipykernel install --user --name propest --display-name "ProPest (torch)"
```

Then open `ProPestNet.ipynb` with the *ProPest (torch)* kernel and run all cells. The
first run extracts the image archive (~1 minute) and writes the split file; later runs
reuse both. Training picks up CUDA or Apple-Silicon MPS automatically.

Environment overrides:

| Variable | Default | Effect |
|---|---|---|
| `PROPEST_EPOCHS` | 60 | Epoch count. Set to `1` for a fast plumbing check |
| `PROPEST_ABLATION` | 0 | Set to `1` to run the section-11 ablation |
| `PROPEST_ABLATION_EPOCHS` | 12 | Epochs per ablation variant |

### Outputs

```text
project/runs/propestnet/best_model.pt      best-validation-F1 checkpoint (~42 MB)
project/runs/propestnet/history.csv        per-epoch loss / accuracy / macro-F1
project/runs/propestnet/results.json       test metrics + confusion matrix
project/runs/ablation/results.json         ablation table (opt-in)
assets/propestnet_curves.png               accuracy, loss, F1, learning-rate curves
assets/propestnet_test.png                 confusion matrix + per-class accuracy
assets/propestnet_per_class.png            one unseen photograph of each pest
assets/propestnet_predictions.png          predictions on 24 unseen test photographs
assets/propestnet_mistakes.png             the most confident errors
```

All of these are gitignored (`data/`, `runs/`, `*.pt`) except the `assets/` figures.

---

# Appendix — Pretrained IP102 Benchmark (teammate handoff)

## Purpose

`experiments/pretrained_ip102_benchmark.py` measures how a pretrained ResNet-18 performs
on the same fifteen-class subset.

> **This is an external benchmark, not the submitted model.** It loads ImageNet pretrained
> weights, which the course does not permit for the deliverable. The submitted model is
> ProPestNet. State this explicitly in the presentation.

It reads the **same** dataset definition ProPestNet trains on —
`project/splits_top15.json` and `project/boxes_top15.json` — so the two numbers are
directly comparable. Only three things differ, and all three are inherent to using a
pretrained network: ImageNet weights, the ResNet-18 architecture, and the 224px input
those weights expect (ProPestNet uses 128px).

## Exact dataset subset

The same ten classes, the same seeded stratified 70/15/15 split, and the same
bounding-box crop with a 25% margin:

| Split | Images |
|---|---:|
| Train | 6,395 |
| Validation | 1,370 |
| Test | 1,370 |

Nothing is rebuilt — the script loads both JSON files and fails loudly if the counts,
the filename-encoded classes, or the on-disk images disagree with them.

## 1. Copy the required files

Send four things to the other machine:

```text
experiments/pretrained_ip102_benchmark.py   the script
splits_top15.json                           the split
boxes_top15.json                            the boxes
images_det/                                 the IP102 detection JPEGs
```

`images_det/` is `data/IP102_v1.1/Classification/images_det` here. Only the 9,135
files the split names are read, so a subset of the archive is enough.

## 2. Create a Python environment

Python 3.10-3.13 (PyTorch does not support 3.14).

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision pillow numpy
```

For an NVIDIA GPU, install the appropriate PyTorch build using the command
provided by <https://pytorch.org/get-started/locally/> instead of guessing a
CUDA package version.

## 3. Verify the dataset without training

```bash
python project/experiments/pretrained_ip102_benchmark.py \
  --image-dir data/IP102_v1.1/Classification/images_det \
  --check-data
```

Expected output:

```text
train: 6395
val: 1370
test: 1370
boxes: 9134/9135 images (99.99%)
majority-class baseline on test: 30.4%
Dataset check passed.
```

Do not continue if these numbers differ. `--splits` and `--boxes` default to the JSON
files in `project/`, so only `--image-dir` is needed here; a teammate with the files
somewhere else passes all three.

Then smoke-test the training loop — three batches per epoch, about a minute, and
`results.json` records `"smoke_run": true` so it can never be mistaken for a real
result:

```bash
python project/experiments/pretrained_ip102_benchmark.py \
  --image-dir data/IP102_v1.1/Classification/images_det \
  --output-dir /tmp/smoke --smoke
```

## 4. Run the benchmark

```powershell
python pretrained_ip102_benchmark.py `
  --image-dir "C:\path\to\images_det" `
  --output-dir "runs\pretrained_resnet18" `
  --head-epochs 5 `
  --finetune-epochs 10 `
  --batch-size 32 `
  --seed 42
```

On the first run, torchvision downloads the official ImageNet ResNet-18
weights. Internet access is needed once unless those weights are already in the
local PyTorch cache.

Training has two phases:

1. Freeze the pretrained feature extractor and train the new fifteen-class head.
2. Unfreeze only `layer4` and the head for low-learning-rate fine-tuning.

The test set is evaluated only after both phases and after the best validation
checkpoint has been selected.

## 5. Outputs to return to the team

The output directory contains:

```text
best_model.pt
history.csv
results.json
```

Send all three files back to the team. The most important comparison values
are:

- Test accuracy
- Test macro F1
- Per-class F1
- Best validation macro F1
- Training time
- Trainable and total parameter counts

## Fair-comparison rules

- Do not edit `splits_top15.json` or `boxes_top15.json`, and do not create a new
  random split. The whole point is that both models see the same images.
- Do not tune using the test set.
- Keep seed `42` for the first comparison.
- Record any changed learning rate, augmentation, batch size, or epoch count.
- Compare this model with ProPestNet using macro F1, not accuracy alone.
- State clearly in the presentation that pretrained ResNet-18 is an external
  benchmark and is not the submitted course model.
