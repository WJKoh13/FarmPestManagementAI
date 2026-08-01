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

> **Superseded for `full102` by Phase 4.** The exhaustive decode of all 75,222
> images found ten PNG files carrying a `.jpg` extension, seven of them RGBA.
> The 2,000-image sample happened to miss all ten. See
> [Phase 4 verification results](#phase-4-verification-results). The `rice10`
> row stands: it was already exhaustive.

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
introduces interpolation blur. Phase 4 measured this exhaustively for both
scopes; see [Phase 4 verification results](#phase-4-verification-results) for the
final figures. Phase 9 checks whether errors concentrate in this cohort. The fact
that ~30-40% of images are already below 224 px supports 160x160 as the input
size rather than a larger one.

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

## Phase 4 verification results

Phase 4 built the derived manifests and completed the checks Phase 1 deferred.
Every image in **both** scopes was decoded in full and hashed: 75,222 images for
`full102`, of which 7,205 form `rice10`.

Reports: `data/reports/dataset_audit_rice10.json`,
`data/reports/dataset_audit_full102.json`.

### Derived manifests

`data/processed/<scope>/{train,validation,test}.csv`, each with a
`.metadata.json` sidecar and a `class_mapping.json` per scope. Columns:
`filename, relative_path, ip102_label, project_label, class_name, split`.

Both the IP102 label and the project label are stored, so nothing downstream
re-derives the mapping. Every record count reproduced Phase 1 exactly, and the
build is idempotent: `scripts/build_manifests.py --check` passes for both scopes.

### Full decode

| Scope | Images | Decode failures | Truncated |
| --- | --- | --- | --- |
| rice10 | 7,205 | **0** | 0 |
| full102 | 75,222 | **0** | 0 |

### Format anomaly: ten PNG files with a `.jpg` extension

Phase 1 sampled 2,000 `full102` images and saw 100% JPEG. The exhaustive decode
found that this was a sampling artefact. **Ten files are actually PNG**, and
**seven of those carry an alpha channel (RGBA)**:

| Split | Files | Mode |
| --- | --- | --- |
| train | 40256, 40557 | RGB |
| train | 40549, 40563, 40577 | RGBA |
| test | 40630 | RGB |
| test | 40314, 40574, 40591, 40601 | RGBA |

All ten belong to **IP102 label 56**, and all ten decode without error, because
Pillow dispatches on content rather than on the extension.

Two consequences for Phase 5, both now **implemented and verified**:

- The loader must not switch to an extension-based reader. Decoding goes through
  Pillow's content dispatch in `farm_pest_ai.data.dataset.load_image`.
- It must convert to RGB explicitly. An RGBA image left alone would hand the CNN
  a fourth input channel. Conversion happens both at the decode boundary and as
  the first step of every transform pipeline, so bypassing one still cannot
  produce four channels. All ten files are confirmed to yield `(3, 160, 160)`
  and are pinned by name in `tests/test_loader_integration.py`.

These files are **not** renamed or re-encoded: the source tree is read-only. The
filenames are pinned by tests in `tests/test_dataset_integration.py`.

### Exact-content duplicates and cross-split leakage

Measured by SHA-256 over file bytes. This detects exact duplicates only; two
visually identical images saved at different JPEG qualities hash differently, so
near-duplicate leakage is **not** ruled out by this check.

| Scope | Duplicate groups | Within-split | **Cross-split** | Label conflicts |
| --- | --- | --- | --- | --- |
| rice10 | 1 | 1 | **0** | 0 |
| full102 | 5 | 3 | **2** | 0 |

**`rice10` has zero cross-split leakage.** Its validation figures are
uncontaminated, which reinforces it as the development scope.

`full102` carries two byte-identical train/test pairs, both within a single
class:

| Train | Test | IP102 label |
| --- | --- | --- |
| 40410.jpg | 40432.jpg | 56 |
| 65553.jpg | 66152.jpg | 92 |

Two contaminated images out of 22,619 is roughly 0.009% of the test set, far too
small to move a headline metric. It is recorded rather than corrected: the
official splits are never modified. Phase 9 reports test metrics both with and
without these two images.

No duplicate group carries conflicting labels in either scope, so there is no
annotation contradiction among identical files.

### Dimensions, re-measured exhaustively

Phase 1's `full102` figures came from a 2,000-image sample; these cover every
image.

| Scope / split | Short side min/median/max | < 160 px | < 224 px |
| --- | --- | --- | --- |
| rice10 train | 63 / 256 / 4,303 | 6.3% | 40.7% |
| rice10 validation | 85 / 250 / 3,240 | 8.3% | 42.6% |
| rice10 test | 64 / 247 / 3,456 | 9.6% | 43.5% |
| full102 train | 52 / 320 / 4,303 | 4.0% | 28.9% |
| full102 validation | 72 / 320 / 3,456 | 5.3% | 31.0% |
| full102 test | 59 / 320 / 6,034 | 5.8% | 31.3% |

The sub-160px cohort is real but modest, and the ~29-44% below 224 px continues
to support 160x160 as the input size rather than something larger.

### Distributions, reconfirmed

`full102` train imbalance is **82.0x**, with label 72 the smallest at 42 training
images and label 101 the largest at 3,444. Validation still has classes with only
7 images (label 72), so macro F1 on `full102` remains noisy for the rare tail.

`rice10` imbalance is **2.8x** across all three splits.

Both scopes have all classes present in all three splits, no filename appears in
more than one split, and every derived record traces back to the source manifest
in order.

## Still not measured

- **Near-duplicate** detection by perceptual hashing. Byte-level hashing catches
  only exact copies, so visually redundant images remain possible.
- Whether classification errors concentrate in the sub-160px upscale cohort.
  Recorded now; analysed in Phase 9.
