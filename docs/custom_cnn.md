# custom_cnn — a residual depthwise-separable CNN

The second from-scratch entry in the benchmark, by Zi Yang, trained under the locked
protocol in `protocol.yaml`. Notebook: [`notebooks/custom_cnn_ziyang.ipynb`](../notebooks/custom_cnn_ziyang.ipynb).
Architecture module: [`ip102_bench/models/custom_cnn.py`](../ip102_bench/models/custom_cnn.py),
registered as `custom_cnn_ziyang`.

Nothing here comes from `torchvision.models` and no weight is pretrained. The only
third-party pieces are `torchvision.transforms` and PyTorch's layer primitives.

## The design in one line

A strided stem, then four stages of residual blocks whose 3x3 convolutions are factorized
into depthwise + pointwise pairs, each block gated by squeeze-and-excitation and
regularized by stochastic depth that ramps with depth.

```text
160x160x3  -> stem: ConvBNAct 3x3 s2, 32ch            ->  80x80x32
           -> stage 1: 2x ResidualSeparableBlock  s2  ->  40x40x64
           -> stage 2: 2x ResidualSeparableBlock  s2  ->  20x20x128
           -> stage 3: 3x ResidualSeparableBlock  s2  ->  10x10x256
           -> stage 4: 2x ResidualSeparableBlock  s2  ->   5x5x384
           -> SiLU -> global average pool -> dropout(0.3) -> Linear(384, 15)
```

`ResidualSeparableBlock` = `DWSep(3x3, stride) -> DWSep(3x3, linear) -> SE -> DropPath -> + shortcut -> SiLU`.
The second separable convolution's pointwise projection is left **linear** so the shortcut
is added before the final activation — that ordering is what lets the identity path carry
an unmodified signal. The shortcut is a true identity when the shape is unchanged and a
1x1 strided projection otherwise.

## Why these choices

| Decision | What it buys |
|---|---|
| **Depthwise-separable 3x3s** instead of dense ones | Roughly `1/out_channels + 1/9` of the parameters. That factorization is what pays for the depth: nine residual blocks in **1,437,167 parameters** — an order of magnitude under ProPestNet's 10,988,015 |
| **Squeeze-and-excitation in every block** (268,640 params, 19% of the network) | Global average pooling collapses each channel to one number, so the gate decides from the whole feature map rather than a 3x3 neighbourhood. The gate is a sigmoid: a channel can be attenuated or preserved, never inverted |
| **Stochastic depth ramping 0 → 0.1 with depth** | Drops a residual branch for a whole sample, survivors scaled by `1/keep_prob` so the expectation is unchanged. Ramped rather than uniform, because dropping early blocks as often as late ones removes low-level features the whole network depends on |
| **SiLU over ReLU** | Non-monotonic and smooth; it does not zero the gradient for every negative pre-activation, which matters in a deep stack of thin depthwise layers |
| **Adaptive average pooling head**, `Linear(384, 15)` = 5,775 params | No flatten-then-`Linear` blowup, and the network is not tied to one input resolution — 160px under the current protocol, 128px or 224px without a code change |
| **No bias where a norm follows** | BatchNorm subtracts a per-channel mean immediately after, so a bias term would be cancelled and only waste parameters |

99.5% of the parameters sit in the convolutional stages. Capacity goes into learning
features, not into a fat classifier head.

## Protocol

Trained under `protocol.yaml` v1, unmodified — the architecture is the only thing that
differs from any other entry in the table:

| | |
|---|---|
| Subset | `detection_top15` — 15 classes, 6,748 / 1,446 / 1,447 train / val / test |
| Input | 160 x 160, box crop with 0.25 margin |
| Normalization | `data_manifests/norm_stats.json`, measured on the training split only |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Loss | Inverse-frequency weighted cross-entropy |
| Schedule | ReduceLROnPlateau on validation macro F1, factor 0.5, patience 5 |
| Stopping | 60 epochs, early stopping patience 10; weights kept from the best **validation** macro F1 |
| Seed | 42 |

The test split is touched exactly once, by `save_run`, after the best epoch is already
fixed by validation. Primary metric is macro F1, not accuracy — the classes are heavily
imbalanced (2,054 `cicadellidae` training images against 34 `wireworm`).

## Results

<!-- Filled from runs/custom_cnn_ziyang/<run_id>/results.json -->

_Pending: written when the training run completes._

## How the app serves it

`save_run` writes `runs/custom_cnn_ziyang/<run_id>/` and the app discovers it with no
configuration change — [`app/cnn_model.py`](../app/cnn_model.py) scans `runs/` and loads
the highest-scoring run whose class count matches the 15 the app serves.

Two things make that safe, and both are worth knowing if you train another model:

- **The checkpoint carries its own preprocessing.** `image_size`, `mean`, `std`,
  `crop_mode` and `crop_margin` are written into `best_model.pt` by `save_run`, and
  [`app/propest_inference.py`](../app/propest_inference.py) rebuilds the transform from
  those values rather than from a constant. A state dict holds weights, not the pipeline
  that produced them; this model trains at 160px with repo-measured normalization, while
  the app's fallbacks are ProPestNet's 128px and ImageNet statistics. Serving one through
  the other does not raise — it just quietly makes every prediction worse.
- **Runs are ranked by the setting they will be served under.** `app.cnn_model._score`
  prefers a bundle's prior-corrected score, then its TTA score, then its single pass —
  because that corrected number is what a farmer actually gets. `save_run` only records a
  single pass, so run `scripts/evaluate_tta.py runs/custom_cnn_ziyang/<run_id>` after
  training. Without it this run is ranked on its raw score against ProPestNet's
  TTA-corrected one, which compares two different things.

The architecture module must stay in `ip102_bench/models/` and registered in
`SCRATCH_REGISTRY` under the same `model_name` used at save time — the app rebuilds the
network from that name alone. `tests/test_custom_cnn.py` pins the module to the notebook
cell it was copied from, so the two cannot drift apart and silently invalidate the weights.

## Not to be confused with the legacy `det_top15` checkpoint

An earlier checkpoint of this same architecture exists from an experimental branch,
imported by [`scripts/import_custom_cnn_run.py`](../scripts/import_custom_cnn_run.py). It
trained outside this protocol — 0.15-margin crops, ImageNet normalization, its own
`det_top15` scope — so its validation score is not comparable with anything in the table.
The importer records a legacy `protocol_version` that `ip102_bench.compare` filters out,
and marks the bundle ineligible for automatic selection so it can never take the served
slot by scoring well under different rules. It remains loadable by explicit path from the
model picker.
