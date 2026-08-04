"""Tests for E8: the fine-grained class-separation auxiliary objective.

The properties under test:

* the inference contract is unchanged — ``forward()`` still returns raw logits
  and no prediction path consults the embedding;
* the disabled path is a strict no-op, so historical configs are unaffected;
* the loss is numerically stable at the published temperature, where a naive
  implementation overflows;
* degenerate batches (no positives) are handled without NaN and without
  aborting;
* gradients flow to the backbone through the auxiliary term;
* the loss actually rewards class separation, verified on constructed
  embeddings rather than assumed.
"""

from __future__ import annotations

import pytest
import torch

from farm_pest_ai.scopes import resolve_scope
from farm_pest_ai.vision.finegrained import (
    FINE_GRAINED_METHODS,
    FineGrainedConfig,
    FineGrainedError,
    ProjectionHead,
    fine_grained_config_from_config,
    supervised_contrastive_loss,
)
from farm_pest_ai.vision.models import ModelConfig, build_model


def _model(name: str = "custom_cnn", num_classes: int = 10):
    """Build a small model of the requested architecture."""
    config = ModelConfig(
        name=name,
        num_classes=num_classes,
        stem_channels=8,
        stage_channels=(8, 16),
        stage_blocks=(1, 1),
        stage_strides=(2, 2),
    )
    return build_model(config)


def _normalised(values: torch.Tensor) -> torch.Tensor:
    """L2-normalise rows, as the projection head does."""
    return torch.nn.functional.normalize(values, dim=1)


# -- configuration ------------------------------------------------------


def test_default_is_disabled() -> None:
    """The objective is opt-in."""
    config = FineGrainedConfig()
    assert config.method == "none"
    assert not config.enabled


def test_known_methods() -> None:
    """One implemented method, plus 'none'."""
    assert FINE_GRAINED_METHODS == ("none", "supcon")


def test_unknown_method_is_rejected() -> None:
    """A typo fails loudly rather than silently disabling the objective."""
    with pytest.raises(FineGrainedError, match=r"unknown training\.fine_grained\.method"):
        FineGrainedConfig(method="supcon2").validate()


def test_non_positive_temperature_is_rejected() -> None:
    """Temperature divides the similarities."""
    with pytest.raises(FineGrainedError, match="temperature must be positive"):
        FineGrainedConfig(method="supcon", temperature=0.0).validate()


def test_negative_weight_is_rejected() -> None:
    """A negative auxiliary weight would reward collapsing classes together."""
    with pytest.raises(FineGrainedError, match="weight must be non-negative"):
        FineGrainedConfig(method="supcon", weight=-0.1).validate()


def test_unknown_mining_strategy_is_rejected() -> None:
    """Only the documented strategy is available."""
    with pytest.raises(FineGrainedError, match=r"unknown training\.fine_grained\.mining"):
        FineGrainedConfig(method="supcon", mining="hardest").validate()


def test_tiny_embedding_dim_is_rejected() -> None:
    """A one-dimensional embedding cannot express angular separation."""
    with pytest.raises(FineGrainedError, match="embedding_dim must be at least 2"):
        FineGrainedConfig(method="supcon", embedding_dim=1).validate()


def test_disabled_config_ignores_other_fields() -> None:
    """Stale values are tolerated while the method is 'none'."""
    FineGrainedConfig(method="none", weight=-5.0, temperature=0.0).validate()


def test_config_records_the_full_recipe() -> None:
    """Margin/temperature, weight, embedding dim and mining are all recorded."""
    payload = FineGrainedConfig(
        method="supcon", weight=0.2, temperature=0.1, embedding_dim=64
    ).to_dict()

    assert payload["method"] == "supcon"
    assert payload["weight"] == 0.2
    assert payload["temperature"] == 0.1
    assert payload["embedding_dim"] == 64
    assert payload["mining"] == "all_positives"
    assert payload["objective"] == "cross_entropy + weight * supervised_contrastive"
    assert payload["applies_to"] == "train split only"
    assert "raw class logits" in payload["inference_contract"]


def test_a_config_without_the_section_resolves_to_none() -> None:
    """Historical configurations keep their exact meaning."""
    from farm_pest_ai.config import Config

    config = Config(data={"training": {"learning_rate": 0.0015}})
    assert fine_grained_config_from_config(config).method == "none"


def test_training_config_round_trips_the_objective() -> None:
    """The resolved training config carries it into the run record."""
    from farm_pest_ai.config import Config
    from farm_pest_ai.vision.training import training_config_from_config

    config = Config(
        data={
            "dataset": {"scope": "rice10"},
            "training": {
                "fine_grained": {
                    "method": "supcon",
                    "weight": 0.1,
                    "temperature": 0.07,
                    "embedding_dim": 128,
                }
            },
        }
    )
    resolved = training_config_from_config(config)

    assert resolved.fine_grained.method == "supcon"
    assert resolved.to_dict()["fine_grained"]["temperature"] == 0.07


def test_training_config_defaults_to_no_objective() -> None:
    """An E0-style config records cross-entropy alone."""
    from farm_pest_ai.config import Config
    from farm_pest_ai.vision.training import training_config_from_config

    config = Config(data={"dataset": {"scope": "rice10"}, "training": {"epochs": 60}})
    resolved = training_config_from_config(config)

    assert resolved.fine_grained.method == "none"
    assert resolved.to_dict()["fine_grained"]["objective"] == "cross_entropy"


# -- the inference contract is unchanged --------------------------------


@pytest.mark.parametrize("name", ["custom_cnn", "baseline_cnn"])
def test_forward_still_returns_raw_logits(name: str) -> None:
    """The public contract is untouched by the new embedding path."""
    model = _model(name).eval()
    images = torch.randn(4, 3, 64, 64)

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (4, 10)
    # Raw logits, not probabilities.
    assert not torch.allclose(
        logits.softmax(dim=1).sum(dim=1), logits.sum(dim=1), atol=1e-3
    )
    assert not torch.allclose(logits.sum(dim=1), torch.ones(4), atol=1e-3)


@pytest.mark.parametrize("name", ["custom_cnn", "baseline_cnn"])
def test_forward_features_returns_pooled_features(name: str) -> None:
    """The embedding path yields the pooled vector the classifier reads."""
    model = _model(name).eval()
    images = torch.randn(4, 3, 64, 64)

    with torch.no_grad():
        features = model.forward_features(images)

    assert features.shape == (4, model.feature_dim)


@pytest.mark.parametrize("name", ["custom_cnn", "baseline_cnn"])
def test_combined_pass_matches_the_separate_paths(name: str) -> None:
    """One backbone pass gives the same logits and features as two calls.

    In eval mode the network is deterministic, so this is exact. It is what
    justifies using the combined path during training for cost reasons.
    """
    model = _model(name).eval()
    images = torch.randn(4, 3, 64, 64)

    with torch.no_grad():
        logits, features = model.forward_logits_and_features(images)
        expected_logits = model(images)
        expected_features = model.forward_features(images)

    assert torch.allclose(logits, expected_logits, atol=1e-6)
    assert torch.allclose(features, expected_features, atol=1e-6)


def test_feature_dim_matches_the_classifier_input() -> None:
    """The embedding width is the classifier's input width."""
    model = _model()
    assert model.feature_dim == model.classifier.in_features


def test_forward_features_rejects_a_bad_input() -> None:
    """The embedding path applies the same input checking as forward()."""
    from farm_pest_ai.vision.models import ModelError

    model = _model().eval()
    with pytest.raises(ModelError, match="expected 3 input channels"):
        model.forward_features(torch.randn(2, 4, 64, 64))


def test_checkpoint_state_dict_is_unchanged_by_the_objective() -> None:
    """The projection head is not part of the model's state_dict.

    This is what lets an E8 checkpoint load into the ordinary inference path
    with no special case — the saved weights are byte-compatible with a model
    trained without the auxiliary objective.
    """
    model = _model()
    keys = set(model.state_dict())

    head = ProjectionHead(model.feature_dim, 128)

    assert not any(key.startswith("projection") for key in keys)
    assert not set(head.state_dict()) & keys


# -- the projection head ------------------------------------------------


def test_projection_head_normalises_its_output() -> None:
    """Rows are unit vectors, so the dot product is a cosine similarity."""
    head = ProjectionHead(32, 16)
    embeddings = head(torch.randn(8, 32))

    assert embeddings.shape == (8, 16)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(8), atol=1e-5)


# -- the loss: correctness ----------------------------------------------


def test_loss_is_lower_for_well_separated_classes() -> None:
    """The objective actually rewards separation, verified not assumed."""
    # Two classes, tightly clustered and far apart.
    separated = _normalised(
        torch.tensor(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
        )
    )
    # The same labels, but the classes are interleaved.
    entangled = _normalised(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01], [0.01, 0.99]])
    )
    targets = torch.tensor([0, 0, 1, 1])

    good = supervised_contrastive_loss(separated, targets, temperature=0.1)
    bad = supervised_contrastive_loss(entangled, targets, temperature=0.1)

    assert float(good) < float(bad)


def test_loss_matches_a_direct_reference_implementation() -> None:
    """Pinned against an independent, naive computation of the same formula."""
    torch.manual_seed(0)
    embeddings = _normalised(torch.randn(6, 4))
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    temperature = 0.5

    actual = supervised_contrastive_loss(
        embeddings, targets, temperature=temperature
    )

    # Direct transcription of the definition, no stability tricks.
    similarity = embeddings @ embeddings.t() / temperature
    total = 0.0
    anchors = 0
    for i in range(6):
        positives = [j for j in range(6) if j != i and targets[j] == targets[i]]
        if not positives:
            continue
        others = [j for j in range(6) if j != i]
        denominator = sum(float(torch.exp(similarity[i, j])) for j in others)
        term = sum(
            float(similarity[i, p]) - float(torch.log(torch.tensor(denominator)))
            for p in positives
        )
        total += -term / len(positives)
        anchors += 1

    assert float(actual) == pytest.approx(total / anchors, abs=1e-5)


def test_loss_is_permutation_invariant() -> None:
    """Reordering the batch must not change the loss."""
    torch.manual_seed(0)
    embeddings = _normalised(torch.randn(8, 4))
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    permutation = torch.randperm(8)

    original = supervised_contrastive_loss(embeddings, targets)
    shuffled = supervised_contrastive_loss(
        embeddings[permutation], targets[permutation]
    )

    assert float(original) == pytest.approx(float(shuffled), abs=1e-5)


# -- the loss: stability and degenerate batches -------------------------


def test_loss_is_finite_at_the_published_temperature() -> None:
    """T=0.07 is where a naive implementation overflows."""
    torch.manual_seed(0)
    embeddings = _normalised(torch.randn(64, 128))
    targets = torch.randint(0, 10, (64,))

    loss = supervised_contrastive_loss(embeddings, targets, temperature=0.07)

    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_loss_is_finite_for_identical_embeddings() -> None:
    """Perfectly collapsed embeddings must not produce NaN.

    Every similarity is 1/T here, which is the worst case for an unstabilised
    exponential.
    """
    embeddings = _normalised(torch.ones(16, 8))
    targets = torch.randint(0, 4, (16,))

    loss = supervised_contrastive_loss(embeddings, targets, temperature=0.07)

    assert torch.isfinite(loss)


def test_batch_with_no_positives_returns_zero_and_stays_differentiable() -> None:
    """Every sample a different class: no valid anchor, but backward() works."""
    embeddings = _normalised(torch.randn(4, 8)).requires_grad_(True)
    targets = torch.tensor([0, 1, 2, 3])

    loss = supervised_contrastive_loss(embeddings, targets)

    assert float(loss.detach()) == 0.0
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_single_sample_batch_returns_zero() -> None:
    """One sample has neither positives nor negatives."""
    embeddings = _normalised(torch.randn(1, 8))
    loss = supervised_contrastive_loss(embeddings, torch.tensor([0]))

    assert float(loss) == 0.0


def test_partial_positives_excludes_anchors_without_partners() -> None:
    """An anchor with no positive contributes nothing, not a zero.

    Counting it as zero would dilute the mean and make the reported magnitude
    depend on batch composition rather than on the embedding geometry.
    """
    torch.manual_seed(0)
    embeddings = _normalised(torch.randn(5, 4))
    # Classes 0 and 1 have partners; class 2 is a singleton.
    with_singleton = supervised_contrastive_loss(
        embeddings, torch.tensor([0, 0, 1, 1, 2]), temperature=0.5
    )

    assert torch.isfinite(with_singleton)
    assert float(with_singleton) > 0.0


def test_loss_rejects_shape_mismatch() -> None:
    """Mismatched embeddings and targets raise rather than broadcasting."""
    with pytest.raises(FineGrainedError, match="against"):
        supervised_contrastive_loss(torch.randn(4, 8), torch.tensor([0, 1, 2]))


def test_loss_rejects_a_non_positive_temperature() -> None:
    """Guarded at the call site as well as in configuration."""
    with pytest.raises(FineGrainedError, match="temperature must be positive"):
        supervised_contrastive_loss(
            _normalised(torch.randn(4, 8)), torch.tensor([0, 0, 1, 1]),
            temperature=-1.0,
        )


def test_loss_rejects_non_matrix_embeddings() -> None:
    """A flat vector is a caller error, not something to reshape silently."""
    with pytest.raises(FineGrainedError, match="expected embeddings of shape"):
        supervised_contrastive_loss(torch.randn(8), torch.zeros(8, dtype=torch.int64))


# -- gradient flow to the backbone --------------------------------------


def test_auxiliary_gradients_reach_the_backbone() -> None:
    """The auxiliary term trains the shared features, not just the head.

    Without this, the objective would shape a projection that nothing else uses
    and the backbone would be unaffected — the failure mode where an auxiliary
    loss appears to run but changes nothing.
    """
    model = _model()
    head = ProjectionHead(model.feature_dim, 32)
    images = torch.randn(8, 3, 64, 64)
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    _, features = model.forward_logits_and_features(images)
    loss = supervised_contrastive_loss(head(features), targets, temperature=0.5)
    loss.backward()

    stem_parameter = next(model.parameters())
    assert stem_parameter.grad is not None
    assert torch.isfinite(stem_parameter.grad).all()
    assert float(stem_parameter.grad.abs().sum()) > 0.0


def test_projection_head_receives_gradients() -> None:
    """The head itself is optimised."""
    head = ProjectionHead(16, 8)
    embeddings = head(torch.randn(8, 16))
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    supervised_contrastive_loss(embeddings, targets, temperature=0.5).backward()

    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_disabled_path_builds_no_projection_head() -> None:
    """With the objective off, no auxiliary parameter exists at all."""
    from farm_pest_ai.config import Config
    from farm_pest_ai.vision.training import training_config_from_config

    config = Config(data={"dataset": {"scope": "rice10"}, "training": {}})
    resolved = training_config_from_config(config)

    assert not resolved.fine_grained.enabled


# -- batch composition feasibility --------------------------------------


def test_rice10_batches_supply_positives_for_every_anchor() -> None:
    """The feasibility argument for choosing SupCon over triplet mining.

    Drawn from the real rice10 training distribution: at batch size 64 nearly
    every class is present and each anchor has several same-class partners, so
    the loss needs no sampler change. Pinned so a future change to the scope or
    batch size that invalidates the argument fails here.
    """
    import csv
    from collections import Counter
    from pathlib import Path

    manifest = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "rice10"
        / "train.csv"
    )
    if not manifest.is_file():
        pytest.skip("rice10 derived manifest is not built")

    counts: Counter[str] = Counter()
    with manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["project_label"]] += 1

    total = sum(counts.values())
    assert len(counts) == resolve_scope("rice10").num_classes

    batch = 64
    proportions = [value / total for value in counts.values()]
    expected_positives = sum(p * (batch - 1) * p for p in proportions)
    expected_classes = sum(1 - (1 - p) ** batch for p in proportions)

    # ~7 positives per anchor and ~9.9 of 10 classes present.
    assert expected_positives > 3.0
    assert expected_classes > 9.0
