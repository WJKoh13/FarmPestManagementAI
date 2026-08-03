# Pretrained IP102 Benchmark - Teammate Handoff

## Purpose

This experiment measures how a pretrained ResNet-18 performs on the exact same
ten-class IP102 rice-pest subset used by the team's scratch-built CNN.

This is a **benchmark only**. Because the course requires an original CNN with
no pretrained or prebuilt model, do not submit this ResNet-18 as the final
course model.

## Exact dataset subset

The script preserves the official IP102 `train.txt`, `val.txt`, and `test.txt`
splits and remaps these labels:

| Project label | IP102 label | Pest |
|---:|---:|---|
| 0 | 0 | Rice leaf roller |
| 1 | 1 | Rice leaf caterpillar |
| 2 | 3 | Asiatic rice borer |
| 3 | 4 | Yellow rice borer |
| 4 | 5 | Rice gall midge |
| 5 | 7 | Brown planthopper |
| 6 | 8 | White-backed planthopper |
| 7 | 9 | Small brown planthopper |
| 8 | 10 | Rice water weevil |
| 9 | 11 | Rice leafhopper |

Expected filtered totals:

| Split | Images |
|---|---:|
| Train | 4,318 |
| Validation | 721 |
| Test | 2,166 |

## 1. Copy the required files

Copy the entire classification dataset to the other machine. The directory
passed to `--data-root` must contain:

```text
ip102_v1.1/
├── images/
│   ├── 00000.jpg
│   └── ...
├── train.txt
├── val.txt
└── test.txt
```

Also send `pretrained_ip102_benchmark.py` from this folder.

## 2. Create a Python environment

Python 3.10-3.12 is recommended.

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

```powershell
python pretrained_ip102_benchmark.py `
  --data-root "C:\path\to\ip102_v1.1" `
  --check-data
```

Expected output:

```text
train: 4318
val: 721
test: 2166
Dataset check passed.
```

Do not continue if these counts differ.

## 4. Run the benchmark

```powershell
python pretrained_ip102_benchmark.py `
  --data-root "C:\path\to\ip102_v1.1" `
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

1. Freeze the pretrained feature extractor and train the new ten-class head.
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

- Do not change the selected labels.
- Do not create a new random split.
- Do not tune using the test set.
- Keep seed `42` for the first comparison.
- Record any changed learning rate, augmentation, batch size, or epoch count.
- Compare this model with the scratch CNN using macro F1, not accuracy alone.
- State clearly in the presentation that pretrained ResNet-18 is an external
  benchmark and is not the submitted course model.

