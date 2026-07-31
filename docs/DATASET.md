# Dataset

Source: **IP102 v1.1**, classification subset. Treated as strictly read-only.

Location (configurable; never hard-coded in the package):

```
<project_root>/ip102_v1.1/Classification/
    images/          75,222 .jpg files, flat, ~2.97 GB
    classes.txt      102 class names, numbered 1-102
    classes.docx     same content, not parsed
    train.txt        45,095 records
    val.txt           7,508 records
    test.txt         22,619 records
```

`ip102_v1.1/Detection/` (VOC2007 layout) exists but object detection is **out of
scope**. It is never read or processed.

## Manifest format

Each line is `<filename> <zero-based-label>`, for example:

```
00002.jpg 0
```

Verified in Phase 1: no malformed lines, no blank lines, LF endings, trailing
newline present, all filenames `.jpg`, no path separators, no duplicate
filenames within a split.

## The off-by-one between `classes.txt` and the manifests

`classes.txt` numbers classes **1 through 102**, while manifests use labels
**0 through 101**. The relationship is:

```
ip102_label = classes_txt_id - 1
```

This offset is applied once, in `farm_pest_ai.data.manifests`, and is the single
most likely source of a silent labelling bug. It is covered by tests.

## Phase 1 verification results

Every previously reported fact was independently reverified and confirmed.

### Split counts

| Split | Records |
| --- | --- |
| train | 45,095 |
| validation (`val.txt`) | 7,508 |
| test | 22,619 |
| **Total** | **75,222** |

The total equals the number of files on disk exactly, and the mapping between
manifest records and files is a bijection.

### Integrity

| Check | Result |
| --- | --- |
| Referenced but missing from disk | 0 |
| On disk but unreferenced | 0 |
| Filename overlap across splits | 0 |
| Filenames with conflicting labels | 0 |
| Distinct labels overall | 102 (0–101, none missing) |
| All 102 classes present in every split | Yes |

### full102 class distribution

| Split | Classes | Total | Min | Max | Median | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| train | 102 | 45,095 | 42 (label 72) | 3,444 (label 101) | 289.0 | 442.1 |
| validation | 102 | 7,508 | 7 (labels 72, 80) | 573 (label 101) | 48.0 | 73.6 |
| test | 102 | 22,619 | 22 (label 72) | 1,723 (label 101) | 145.0 | 221.8 |
| all | 102 | 75,222 | 71 (label 72) | 5,740 (label 101) | 482.0 | 737.5 |

**Train imbalance ratio: 82.0x.** Validation classes with as few as 7 images
mean per-class validation recall is extremely noisy for the rare tail; macro F1
on `full102` must be read with that in mind.

### rice10 class distribution

Counts confirmed exactly against the previously reported values.

| Project label | IP102 label | Class name | Train | Val | Test |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | rice leaf roller | 669 | 111 | 335 |
| 1 | 1 | rice leaf caterpillar | 292 | 48 | 147 |
| 2 | 3 | asiatic rice borer | 631 | 106 | 316 |
| 3 | 4 | yellow rice borer | 302 | 50 | 152 |
| 4 | 5 | rice gall midge | 303 | 51 | 152 |
| 5 | 7 | brown plant hopper | 500 | 83 | 251 |
| 6 | 8 | white backed plant hopper | 535 | 90 | 268 |
| 7 | 9 | small brown plant hopper | 331 | 56 | 166 |
| 8 | 10 | rice water weevil | 513 | 86 | 257 |
| 9 | 11 | rice leafhopper | 242 | 40 | 122 |
| | | **Total** | **4,318** | **721** | **2,166** |

Total: **7,205**. All ten classes appear in all three splits. Imbalance is only
**2.8x**, far gentler than `full102` — one reason `rice10` is the development
scope.

Note that IP102 labels **2** (paddy stem maggot) and **6** fall inside the same
numeric neighbourhood but are deliberately **not** part of `rice10`. The mapping
is defined once in `farm_pest_ai.scopes` and is covered by an exact test.

### Image formats

| Sample | Result |
| --- | --- |
| rice10, all 7,205 images | 100% real JPEG by header, PIL mode RGB, **0 decode failures**, 0 truncated, 0 extension/content mismatches |
| full102, random 2,000 | Identical: 100% JPEG, RGB, 0 failures |

### Image dimensions

| Scope | Width min/median/max | Height min/median/max | Aspect min/median/max |
| --- | --- | --- | --- |
| rice10 (all) | 64 / 342 / 8,113 | 63 / 270 / 4,303 | 0.37 / 1.33 / 5.23 |
| full102 (sample) | 98 / 430 / 5,184 | 86 / 328 / 3,456 | 0.40 / 1.33 / 4.41 |

**Images smaller than the 160x160 model input:**

| Scope | Short side < 160 px | Short side < 224 px |
| --- | --- | --- |
| rice10 | 7.5% | 41.7% |
| full102 | 4.5% | 29.1% |

A meaningful minority of images must be **upscaled** to reach 160x160, which
introduces interpolation blur. Phase 4 records the source dimensions in the
derived manifests so this cohort can be analysed separately, and Phase 9 checks
whether errors concentrate in it. The fact that ~30-40% of images are already
below 224 px supports 160x160 as the input size rather than a larger one.

### Measured decode throughput

Single process, decode to RGB and resize to 160x160: **100.6 img/s** cold,
**449.7 img/s** with a warm OS cache. With 8 workers, data loading will not
bottleneck the GPU for either scope.

## Classification scopes

Defined in `src/farm_pest_ai/scopes.py`, the single source of truth.

| Scope | Classes | Project labels | Purpose |
| --- | --- | --- | --- |
| `rice10` | 10 | 0–9, remapped from IP102 labels 0,1,3,4,5,7,8,9,10,11 | Development, fast iteration, focused deployment |
| `full102` | 102 | 0–101, identical to IP102 | Broad coverage experiment |

`num_classes` is always derived from the scope. Configuration that states a
contradicting `num_classes` is rejected with an error.

## Rules

- The `ip102_v1.1` tree is **read-only**. Never rename, move, delete, overwrite,
  re-encode or reorganise source images, and never modify the original
  manifests.
- Never randomly resplit. The official train/validation/test assignments are
  preserved exactly.
- Derived manifests, reports and any converted copies are written only to
  project-controlled directories (`data/processed/`, `data/reports/`).
- The dataset is Git-ignored and must never be committed.
- Raw class names are preserved verbatim. `classes.txt` mixes common names with
  Latin binomials (for example `Sternochetus frigidus`) and includes at least
  one family-level name (`Cicadellidae`); taxonomy is **not** silently
  corrected. A separate canonical name field is recorded alongside the raw name.

## Not yet measured

Deferred to Phase 4:

- Exact-content duplicate detection by hash.
- Exact-content cross-split leakage (a real risk to headline metrics for IP102,
  since filename-level checks being clean does not rule it out).
- Full decode of all 75,222 images (only rice10 was exhaustively decoded).
