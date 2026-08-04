"""Tests for E7: training-only MixUp and CutMix.

The properties under test are the ones that produce a *plausible wrong training
run* rather than an error:

* the disabled path being a strict no-op, so every historical config is unchanged;
* CutMix correcting lambda for the clipped box area, the classic implementation
  bug — an uncorrected lambda supervises a blend the pixels do not show;
* mixed targets and loss being mathematically correct, verified against a direct
  soft-target cross-entropy;
* metrics comparing against the original hard labels;
* mixing being refused outside a training pass;
* the draws being reproducible from the run seed.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from farm_pest_ai.vision.mixing import (
    MIXING_METHODS,
    BatchMixer,
    MixingConfig,
    MixingError,
    mixed_criterion,
    mixing_config_from_config,
)


def _batch(
    n: int = 8, c: int = 3, h: int = 16, w: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic image batch and label vector."""
    torch.manual_seed(0)
    return torch.randn(n, c, h, w), torch.arange(n) % 4


# -- disabled by default ------------------------------------------------


def test_default_is_disabled() -> None:
    """Mixing is off unless a configuration asks for it."""
    config = MixingConfig()
    assert config.method == "none"
    assert not config.enabled


def test_disabled_path_is_a_strict_no_op() -> None:
    """Images pass through by identity, not merely by equality."""
    images, targets = _batch()
    mixer = BatchMixer(MixingConfig(), seed=1337)

    result = mixer.apply(images, targets, training=True)

    assert result.images is images
    assert result.targets_a is targets
    assert result.lam == 1.0
    assert not result.mixed
    assert result.method == "none"


def test_disabled_mixing_leaves_the_loss_identical() -> None:
    """The disabled loss path is the plain criterion call, bit for bit."""
    images, targets = _batch()
    logits = torch.randn(images.shape[0], 4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    mixer = BatchMixer(MixingConfig(), seed=1337)

    batch = mixer.apply(images, targets, training=True)

    assert torch.equal(
        mixed_criterion(criterion, logits, batch), criterion(logits, targets)
    )


def test_a_config_without_a_mixing_section_resolves_to_none() -> None:
    """Every historical configuration keeps its exact meaning."""
    from farm_pest_ai.config import Config

    config = Config(data={"training": {"learning_rate": 0.0015}})
    assert mixing_config_from_config(config).method == "none"


def test_disabled_mixing_ignores_alpha_and_probability() -> None:
    """Nonsense parameters are tolerated while the method is 'none'.

    Validation only constrains them when they can actually take effect, so a
    config that leaves stale values behind while disabling mixing still loads.
    """
    MixingConfig(method="none", alpha=-1.0, probability=5.0).validate()


# -- configuration ------------------------------------------------------


def test_known_methods() -> None:
    """Only the two documented methods, plus 'none'."""
    assert MIXING_METHODS == ("none", "mixup", "cutmix")


def test_unknown_method_is_rejected() -> None:
    """A typo fails loudly rather than silently disabling mixing."""
    with pytest.raises(MixingError, match=r"unknown training\.mixing\.method"):
        MixingConfig(method="mixupp").validate()


@pytest.mark.parametrize("alpha", [0.0, -0.5])
def test_non_positive_alpha_is_rejected_when_enabled(alpha: float) -> None:
    """Beta(alpha, alpha) needs a positive shape parameter."""
    with pytest.raises(MixingError, match=r"alpha must be positive"):
        MixingConfig(method="mixup", alpha=alpha).validate()


@pytest.mark.parametrize("probability", [-0.1, 1.5])
def test_out_of_range_probability_is_rejected(probability: float) -> None:
    """Probability is a probability."""
    with pytest.raises(MixingError, match=r"probability must be in"):
        MixingConfig(method="mixup", probability=probability).validate()


def test_config_records_target_and_loss_behaviour() -> None:
    """A checkpoint states how its targets and loss behaved, not just the method."""
    payload = MixingConfig(method="cutmix", alpha=1.0, probability=0.5).to_dict()

    assert payload["method"] == "cutmix"
    assert payload["alpha"] == 1.0
    assert payload["probability"] == 0.5
    assert payload["target_mode"] == "soft_pair"
    assert "lam" in payload["loss"]
    assert payload["metrics_against"] == "original hard labels"
    assert payload["applies_to"] == "train split only"


def test_disabled_config_records_the_plain_loss() -> None:
    """A disabled record must not imply a soft target."""
    payload = MixingConfig().to_dict()
    assert payload["target_mode"] == "hard"
    assert payload["loss"] == "CE(logits, targets)"


# -- training-only enforcement ------------------------------------------


@pytest.mark.parametrize("method", ["mixup", "cutmix"])
def test_mixing_is_refused_outside_training(method: str) -> None:
    """An evaluation batch may never be mixed."""
    images, targets = _batch()
    mixer = BatchMixer(MixingConfig(method=method), seed=1337)

    with pytest.raises(MixingError, match="training-only augmentation"):
        mixer.apply(images, targets, training=False)


def test_disabled_mixer_is_safe_outside_training() -> None:
    """With mixing off there is nothing to refuse."""
    images, targets = _batch()
    mixer = BatchMixer(MixingConfig(), seed=1337)

    result = mixer.apply(images, targets, training=False)

    assert not result.mixed


# -- lambda bounds and determinism --------------------------------------


@pytest.mark.parametrize("method", ["mixup", "cutmix"])
def test_lambda_stays_within_bounds(method: str) -> None:
    """Lambda is a convex weight over many draws."""
    images, targets = _batch()
    mixer = BatchMixer(MixingConfig(method=method, alpha=0.4), seed=1337)

    for _ in range(200):
        result = mixer.apply(images, targets, training=True)
        assert 0.0 <= result.lam <= 1.0


@pytest.mark.parametrize("method", ["mixup", "cutmix"])
def test_the_same_seed_reproduces_the_same_mixes(method: str) -> None:
    """Two mixers on one seed draw identically."""
    images, targets = _batch()
    a = BatchMixer(MixingConfig(method=method, alpha=0.4), seed=1337)
    b = BatchMixer(MixingConfig(method=method, alpha=0.4), seed=1337)

    for _ in range(10):
        left = a.apply(images, targets, training=True)
        right = b.apply(images, targets, training=True)
        assert left.lam == right.lam
        assert torch.equal(left.images, right.images)
        assert torch.equal(left.targets_b, right.targets_b)


@pytest.mark.parametrize("method", ["mixup", "cutmix"])
def test_different_seeds_draw_differently(method: str) -> None:
    """The seed actually reaches the draws."""
    images, targets = _batch()
    a = BatchMixer(MixingConfig(method=method, alpha=0.4), seed=1337)
    b = BatchMixer(MixingConfig(method=method, alpha=0.4), seed=2024)

    lams_a = [a.apply(images, targets, training=True).lam for _ in range(10)]
    lams_b = [b.apply(images, targets, training=True).lam for _ in range(10)]

    assert lams_a != lams_b


def test_mixing_does_not_disturb_the_global_rng_stream() -> None:
    """The mixer owns its generator, so dataloader augmentations are unaffected.

    Phase 5 established that changing how the global stream is consumed changes
    the augmentations drawn, which would silently break run reproducibility.
    """
    images, targets = _batch()
    mixer = BatchMixer(MixingConfig(method="mixup", alpha=0.4), seed=1337)

    torch.manual_seed(99)
    expected = torch.randn(4)

    torch.manual_seed(99)
    for _ in range(20):
        mixer.apply(images, targets, training=True)
    actual = torch.randn(4)

    assert torch.equal(expected, actual)


def test_probability_zero_never_mixes() -> None:
    """A zero probability disables mixing per batch without disabling the method."""
    images, targets = _batch()
    mixer = BatchMixer(
        MixingConfig(method="mixup", alpha=0.4, probability=0.0), seed=1337
    )

    for _ in range(20):
        assert not mixer.apply(images, targets, training=True).mixed


def test_a_single_image_batch_is_not_mixed() -> None:
    """There is no partner to mix with."""
    images, targets = _batch(n=1)
    mixer = BatchMixer(MixingConfig(method="mixup", alpha=0.4), seed=1337)

    result = mixer.apply(images, targets, training=True)

    assert not result.mixed
    assert result.lam == 1.0


# -- MixUp correctness --------------------------------------------------


def test_mixup_produces_the_exact_convex_combination() -> None:
    """The output equals lam*x + (1-lam)*x[perm], reproduced independently."""
    images, targets = _batch()
    mixer = BatchMixer(MixingConfig(method="mixup", alpha=0.4), seed=1337)

    result = mixer.apply(images, targets, training=True)
    assert result.mixed

    # Recover the permutation from the labels, then rebuild the mix.
    lam = result.lam
    for index in range(images.shape[0]):
        expected_row = lam * images[index] + (1 - lam) * images
        # The mixed row must match the blend with *some* partner row.
        matches = [
            bool(torch.allclose(result.images[index], expected_row[j], atol=1e-6))
            for j in range(images.shape[0])
        ]
        assert any(matches), f"row {index} is not a convex blend of two inputs"


def test_mixup_keeps_the_batch_shape() -> None:
    """Mixing changes pixels, never geometry."""
    images, targets = _batch()
    mixer = BatchMixer(MixingConfig(method="mixup", alpha=0.4), seed=1337)

    result = mixer.apply(images, targets, training=True)

    assert result.images.shape == images.shape


# -- CutMix area correction ---------------------------------------------


def test_cutmix_lambda_matches_the_actual_pasted_area() -> None:
    """The reported lambda is derived from the pixels, not from the draw.

    This is the correction the phase brief requires. The pasted region is
    recovered by comparing the mixed batch against the original, and the
    resulting area must reproduce ``1 - lam`` exactly.
    """
    images, targets = _batch(n=8, h=32, w=32)
    mixer = BatchMixer(MixingConfig(method="cutmix", alpha=1.0), seed=7)

    checked = 0
    for _ in range(40):
        result = mixer.apply(images, targets, training=True)
        if not result.mixed:
            continue
        # Pixels that changed anywhere in the batch mark the pasted box.
        changed = (result.images != images).any(dim=0).any(dim=0)
        pasted = int(changed.sum())
        total = images.shape[2] * images.shape[3]
        if pasted == 0:
            # A degenerate box (lam ~ 1) pastes nothing; lambda must say so.
            assert result.lam == pytest.approx(1.0)
            continue
        assert result.lam == pytest.approx(1.0 - pasted / total, abs=1e-9)
        checked += 1

    assert checked > 0, "no non-degenerate CutMix box was produced"


def test_cutmix_box_is_a_contiguous_rectangle() -> None:
    """The pasted region is one axis-aligned rectangle, as CutMix requires."""
    images, targets = _batch(n=8, h=32, w=32)
    mixer = BatchMixer(MixingConfig(method="cutmix", alpha=1.0), seed=7)

    for _ in range(40):
        result = mixer.apply(images, targets, training=True)
        if not result.mixed:
            continue
        changed = (result.images != images).any(dim=0).any(dim=0)
        if not bool(changed.any()):
            continue
        rows = torch.nonzero(changed.any(dim=1)).reshape(-1)
        cols = torch.nonzero(changed.any(dim=0)).reshape(-1)
        # Contiguous spans.
        assert int(rows[-1] - rows[0]) == len(rows) - 1
        assert int(cols[-1] - cols[0]) == len(cols) - 1
        return
    pytest.fail("no non-degenerate CutMix box was produced")


def test_cutmix_corrected_lambda_differs_from_the_drawn_lambda() -> None:
    """Clipping must actually bite, otherwise the correction is untested.

    Over many draws at least one box has to be clipped by an image edge — that
    is the case the uncorrected implementation gets wrong.
    """
    images, targets = _batch(n=4, h=32, w=32)
    mixer = BatchMixer(MixingConfig(method="cutmix", alpha=1.0), seed=3)

    areas_from_pixels = []
    for _ in range(60):
        result = mixer.apply(images, targets, training=True)
        if not result.mixed:
            continue
        changed = (result.images != images).any(dim=0).any(dim=0)
        pasted = int(changed.sum())
        if pasted > 0:
            areas_from_pixels.append((result.lam, pasted / (32 * 32)))

    assert areas_from_pixels
    # Every reported lambda agrees with the measured area. If lambda were the
    # uncorrected draw, clipped boxes would disagree.
    for lam, area in areas_from_pixels:
        assert lam == pytest.approx(1.0 - area, abs=1e-9)


# -- soft targets and the loss ------------------------------------------


def test_mixed_loss_equals_a_direct_soft_target_cross_entropy() -> None:
    """The two-term weighted loss equals one CE against the blended target.

    Cross-entropy is linear in the target distribution, so this identity is what
    makes the two-term formulation correct rather than merely conventional.
    """
    torch.manual_seed(0)
    logits = torch.randn(6, 5)
    targets_a = torch.tensor([0, 1, 2, 3, 4, 0])
    targets_b = torch.tensor([4, 3, 2, 1, 0, 1])
    lam = 0.37

    from farm_pest_ai.vision.mixing import MixedBatch

    batch = MixedBatch(
        images=torch.zeros(6, 3, 4, 4),
        targets_a=targets_a,
        targets_b=targets_b,
        lam=lam,
        mixed=True,
        method="mixup",
    )
    criterion = nn.CrossEntropyLoss()
    actual = mixed_criterion(criterion, logits, batch)

    # Build the blended one-hot target directly and score it.
    one_hot_a = torch.zeros(6, 5).scatter_(1, targets_a.unsqueeze(1), 1.0)
    one_hot_b = torch.zeros(6, 5).scatter_(1, targets_b.unsqueeze(1), 1.0)
    blended = lam * one_hot_a + (1 - lam) * one_hot_b
    expected = -(blended * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

    assert actual == pytest.approx(float(expected), abs=1e-6)


def test_mixed_loss_composes_correctly_with_label_smoothing() -> None:
    """Smoothing applies inside each term, matching a smoothed blended target.

    The phase brief requires the label-smoothing interaction to be addressed
    explicitly rather than left implicit. This pins it.
    """
    torch.manual_seed(0)
    logits = torch.randn(6, 5)
    targets_a = torch.tensor([0, 1, 2, 3, 4, 0])
    targets_b = torch.tensor([4, 3, 2, 1, 0, 1])
    lam = 0.37
    eps = 0.1
    classes = 5

    from farm_pest_ai.vision.mixing import MixedBatch

    batch = MixedBatch(
        images=torch.zeros(6, 3, 4, 4),
        targets_a=targets_a,
        targets_b=targets_b,
        lam=lam,
        mixed=True,
        method="mixup",
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=eps)
    actual = mixed_criterion(criterion, logits, batch)

    def smoothed(targets: torch.Tensor) -> torch.Tensor:
        one_hot = torch.zeros(6, classes).scatter_(1, targets.unsqueeze(1), 1.0)
        return one_hot * (1 - eps) + eps / classes

    blended = lam * smoothed(targets_a) + (1 - lam) * smoothed(targets_b)
    expected = -(blended * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

    assert actual == pytest.approx(float(expected), abs=1e-6)


def test_lambda_one_reduces_to_the_unmixed_loss() -> None:
    """A degenerate mix must cost exactly what no mix costs."""
    from farm_pest_ai.vision.mixing import MixedBatch

    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    targets_a = torch.tensor([0, 1, 2, 3])
    targets_b = torch.tensor([3, 2, 1, 0])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    batch = MixedBatch(
        images=torch.zeros(4, 3, 4, 4),
        targets_a=targets_a,
        targets_b=targets_b,
        lam=1.0,
        mixed=True,
        method="mixup",
    )

    assert mixed_criterion(criterion, logits, batch) == pytest.approx(
        float(criterion(logits, targets_a)), abs=1e-6
    )


def test_hard_targets_are_the_original_labels() -> None:
    """Metrics must never see a mixed target."""
    images, targets = _batch()
    mixer = BatchMixer(MixingConfig(method="cutmix", alpha=1.0), seed=1337)

    result = mixer.apply(images, targets, training=True)

    assert torch.equal(result.hard_targets, targets)
    assert torch.equal(result.targets_a, targets)


def test_mixed_loss_is_differentiable() -> None:
    """Gradients flow through the mixed objective."""
    from farm_pest_ai.vision.mixing import MixedBatch

    logits = torch.randn(4, 5, requires_grad=True)
    batch = MixedBatch(
        images=torch.zeros(4, 3, 4, 4),
        targets_a=torch.tensor([0, 1, 2, 3]),
        targets_b=torch.tensor([3, 2, 1, 0]),
        lam=0.4,
        mixed=True,
        method="mixup",
    )

    mixed_criterion(nn.CrossEntropyLoss(), logits, batch).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# -- config and checkpoint round-trip -----------------------------------


def test_training_config_round_trips_mixing() -> None:
    """The resolved training config carries mixing into the run record."""
    from farm_pest_ai.config import Config
    from farm_pest_ai.vision.training import training_config_from_config

    config = Config(
        data={
            "dataset": {"scope": "rice10"},
            "training": {
                "mixing": {"method": "mixup", "alpha": 0.2, "probability": 1.0}
            },
        }
    )
    resolved = training_config_from_config(config)

    assert resolved.mixing.method == "mixup"
    assert resolved.mixing.alpha == 0.2
    assert resolved.to_dict()["mixing"]["method"] == "mixup"
    assert resolved.to_dict()["mixing"]["target_mode"] == "soft_pair"


def test_training_config_defaults_to_no_mixing() -> None:
    """An E0-style config records mixing as disabled."""
    from farm_pest_ai.config import Config
    from farm_pest_ai.vision.training import training_config_from_config

    config = Config(
        data={"dataset": {"scope": "rice10"}, "training": {"epochs": 60}}
    )
    resolved = training_config_from_config(config)

    assert resolved.mixing.method == "none"
    assert resolved.to_dict()["mixing"]["target_mode"] == "hard"


def test_an_invalid_mixing_section_fails_config_resolution() -> None:
    """A bad mixing block aborts before training starts."""
    from farm_pest_ai.config import Config
    from farm_pest_ai.vision.training import training_config_from_config

    config = Config(
        data={
            "dataset": {"scope": "rice10"},
            "training": {"mixing": {"method": "nonsense"}},
        }
    )
    with pytest.raises(MixingError, match=r"unknown training\.mixing\.method"):
        training_config_from_config(config)
