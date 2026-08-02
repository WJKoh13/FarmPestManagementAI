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

## Image-quality review policy (Phase 7.3)

A read-only audit that **proposes and never decides**. It measures objective
image properties, records what a trained checkpoint predicted, and writes a
review manifest for a human to complete:

```bash
python scripts/review_images.py --split validation --contact-sheets
python scripts/review_images.py --split validation \
    --checkpoint artifacts/checkpoints/rice10_custom_protocolA/best.pt
```

**Hard rules.** `ip102_v1.1` is opened read-only: no source image is renamed,
moved, deleted, re-encoded or relabelled, and two tests verify that measuring an
image and building a contact sheet leave it byte-identical. Official derived
manifests are never edited. **The test split cannot be reviewed** — `--split`
does not offer it, and the reviewable splits are checked again inside the script,
because reviewing the test set would let its contents shape a data decision.

**The LLM never deletes or relabels an image.** The audit's most confident output
is the word *suspected*.

### Categories

| Category | Meaning |
| --- | --- |
| `valid_close_up` | Keep. Clear, well-framed subject |
| `difficult_but_valid` | Keep. Correctly labelled but genuinely hard |
| `blurry` | Out of focus or motion-blurred |
| `low_resolution` | Short side below the model input; upscaled by preprocessing |
| `tiny_subject` | Pest present but occupies a very small part of the frame |
| `symptom_only` | Crop damage visible, no identifiable insect |
| `diagram_text` | Illustration, plate or text rather than a photograph |
| `unrelated` | Not a pest image at all |
| `ambiguous` | Cannot be assigned confidently from the image alone |
| `suspected_mislabel` | Appears to show a different class than its label |

**Only `blurry` and `low_resolution` are asserted automatically**, because only
those are measurable from pixels — a short side below the input size, and a
variance-of-Laplacian focus measure below threshold. Every other category needs
human judgement and is at most *suggested* in the `suspected_issue` column.

The distinction is deliberate. A low-confidence prediction is evidence about the
**model**, not proof about the **label**: a blurry photograph of the right pest
is still labelled correctly, and a model can be confidently wrong on a perfectly
good image. So a confident model/label disagreement is queued as
`suspected_mislabel` because it deserves a person's attention, never because it
establishes anything.

### Manifest and review workflow

The manifest carries filename, split, current label and class name, dimensions,
short side, aspect ratio, model prediction and confidence, whether the prediction
matched, measured quality flags, the suspected issue, and two columns —
`reviewer_decision` and `reviewer_notes` — that are **written empty by design**.
Reading a completed manifest back rejects any `reviewer_decision` outside the ten
categories, so a typo cannot propagate downstream.

Contact sheets are grouped by suspected issue, 42 thumbnails per sheet, because
no one reviews thousands of spreadsheet rows but anyone can scan a page of
images.

### First pass over rice10 (both reviewable splits)

5,039 images — the complete train and validation splits — at thresholds short
side < 160 px and focus < 100, with predictions from the E0 `custom_cnn`
`best.pt`. The test split was not reviewed and cannot be.

| Measurement | train (4,318) | validation (721) |
| --- | --- | --- |
| `low_resolution` (measured) | 273 — 6.3% | 60 — 8.3% |
| `blurry` (measured) | 96 — 2.2% | 24 — 3.3% |
| `ambiguous` (queued) | 38 — 0.9% | 46 — 6.4% |
| `suspected_mislabel` (queued) | 238 — 5.5% | 220 — 30.5% |

Both `low_resolution` figures **independently reproduce** the Phase 4 exhaustive
measurements exactly — 6.3% for train and 8.3% for validation — which is a useful
check that the audit measures what it claims to.

**The `suspected_mislabel` split difference proves the flag tracks the model, not
the labels.** It is 5.5% on train and 30.5% on validation, and the only thing
that differs is that the model was fitted on one and not the other. Both numbers
are essentially the model's error rate on that split. Reading the validation
figure as a defect count would propose discarding a third of the split on the
say-so of a model that is itself only ~61% accurate.

#### The quality flags do not identify hard images

Held-out validation accuracy, split by flag:

| Cohort | Images | Accuracy | Mean confidence |
| --- | --- | --- | --- |
| flagged `blurry` | 24 | **0.708** | 0.767 |
| not flagged | 697 | 0.604 | 0.710 |
| flagged `low_resolution` | 60 | **0.700** | 0.774 |
| normal resolution | 661 | 0.599 | 0.706 |

Both flagged cohorts are *easier* for the model, not harder. On the train split
the blur cohort is fit to 0.990 against 0.930 for everything else — the opposite
of what a genuine blur flag would produce.

The contact sheets explain why. Many blur-flagged images are **perfectly sharp**
(`02073.jpg`, `02702.jpg`, `03888.jpg`, `04401.jpg`): what they share is a
smooth, low-texture subject on a plain background, which is exactly the false
positive a variance-of-Laplacian focus measure produces. The same plain-close-up
cohort is also easy to classify, which confounds both flags.

Two consequences, both recorded rather than acted on:

- **The `blurry` threshold of 100 is not validated** and the flag is better read
  as "low texture" than "out of focus" (risk 29).
- **Risk 5 asked whether errors concentrate in the sub-160 px upscale cohort. On
  rice10 validation they do not** — that cohort scores *above* average. This is
  one scope on one model and is confounded as described, so it narrows the risk
  rather than closing it; `full102`, where 4.0–5.8% fall below 160 px across a
  much harder task, is the real test.

Scanning the contact sheets also confirms the taxonomy is needed: the splits
visibly contain botanical **illustration plates**, **multi-panel composites**
tiling several photographs into one file, **watermarks and QR codes**,
**symptom-only** frames showing damaged leaves with no insect, and **tiny
subjects** in wide field shots.

Artifacts: `data/reports/image_review_rice10_{train,validation}.csv` and 28
contact sheets under `data/reports/contact_sheets/<split>/`.

#### Review manifests are not overwritten casually

The manifest is the one artifact a human writes into by hand, so
`review_images.py` refuses to replace an existing one that either carries
reviewer decisions or holds more rows than the current run would write. Both
guards come from real incidents: a `--limit 40` pass silently replaced a complete
721-row review during Phase 7.3, and the same path would have destroyed reviewer
decisions had any been entered. `--force` overrides deliberately.

### Curated manifests

If a completed human review ever justifies a curated split, it goes to a **new
versioned directory**, `data/processed/<scope>/curated/<version>/`. The official
derived manifests stay byte-identical, so the benchmark remains reproducible and
any curated experiment must state which version it used. `curated_manifest_dir`
rejects a version containing a path separator, so a curated write cannot escape
its directory.

**No curated manifest has been created.** No review decision has been made.

## Still not measured

- **Near-duplicate** detection by perceptual hashing. Byte-level hashing catches
  only exact copies, so visually redundant images remain possible.
- Whether classification errors concentrate in the sub-160px upscale cohort **on
  `full102`**. Measured for `rice10` in Phase 7.3 and the answer was no — that
  cohort scores above average — but the result is confounded by the
  plain-close-up effect described above and covers only one scope.
- Every judgement-based review category. The Phase 7.3 audit built the queue and
  the contact sheets over both reviewable splits; **no human review pass has been
  completed**, so the true rates of `diagram_text`, `symptom_only`,
  `tiny_subject`, `unrelated` and real mislabelling are all still unknown.
- A validated blur threshold. The current focus measure demonstrably flags sharp
  low-texture images, so its 2.2–3.3% is a queue size, not a blur rate.
- The `full102` review. Phase 7.3 covered `rice10` only; at 52,603 reviewable
  images `full102` is roughly ten times the work and was not attempted.
