"""Training tests that run against the real IP102 data and derived manifests.

These cover what the synthetic engine tests cannot: that a model built from the
shipped configurations consumes real 160x160 batches, that the class count
derived from the scope matches the labels the manifests actually carry, and that
a checkpoint written from a real run round-trips with its provenance intact.

Every test skips when the dataset or the built manifests are absent. Nothing
here writes to ``ip102_v1.1``, and nothing reads the test split — Phase 9 is the
only phase permitted to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from farm_pest_ai.config import load_config
from farm_pest_ai.data.loaders import build_loaders

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from farm_pest_ai.vision.checkpoints import (  # noqa: E402
    CheckpointError,
    load_checkpoint,
    read_metadata,
)
from farm_pest_ai.vision.metrics import MetricsAccumulator  # noqa: E402
from farm_pest_ai.vision.models import (  # noqa: E402
    ModelConfig,
    build_model,
    model_config_from_config,
)
from farm_pest_ai.vision.training import (  # noqa: E402
    Trainer,
    build_trainer,
    training_config_from_config,
)

pytestmark = pytest.mark.dataset


@pytest.fixture(scope="module")
def smoke_config():
    """The shipped smoke configuration, or a skip when the dataset is absent."""
    config = load_config("smoke_test.yaml")
    if not config.paths.images_dir.is_dir():
        pytest.skip(f"IP102 images not found at {config.paths.images_dir}")
    if not (config.paths.processed_dir / "rice10" / "train.csv").is_file():
        pytest.skip("derived manifests not built; run scripts/build_manifests.py")
    return config


@pytest.fixture(scope="module")
def bundle(smoke_config):
    """Real train and validation loaders. Never the test split."""
    return build_loaders(smoke_config, ("train", "validation"))


# -- the shipped configurations -----------------------------------------


@pytest.mark.parametrize(("scope", "expected"), [("rice10", 10), ("full102", 102)])
def test_shipped_model_config_derives_the_right_class_count(
    scope: str, expected: int
) -> None:
    """The shipped model config yields the active scope's width, unchanged.

    ``model_custom.yaml`` extends ``base.yaml``, so the scope is overridden
    rather than layered in from a second file: a data config listed after it
    would have its scope replaced by the base default. The scope therefore
    arrives as an override, which is also how the CLI ``--scope`` flag works.
    """
    config = load_config("model_custom.yaml", overrides={"dataset": {"scope": scope}})
    assert config.num_classes == expected
    assert model_config_from_config(config).num_classes == expected


def test_smoke_config_builds_a_trainable_model(smoke_config, bundle, tmp_path: Path) -> None:
    trainer = build_trainer(
        smoke_config, bundle, run_dir=tmp_path / "run", run_id="test", smoke=True
    )
    assert trainer.model.num_classes == bundle.num_classes == 10


# -- real batches -------------------------------------------------------


def test_model_consumes_a_real_batch(smoke_config, bundle) -> None:
    """The loader's output shape and the model's input contract must agree."""
    model = build_model(
        model_config_from_config(smoke_config), scope=smoke_config.dataset.scope
    ).eval()
    images, _targets = next(iter(bundle.loaders["validation"]))
    assert images.shape[1:] == (3, 160, 160)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (images.shape[0], 10)
    assert bool(torch.isfinite(logits).all())


def test_real_labels_lie_within_the_scope(bundle) -> None:
    """Manifest labels must fit the head the scope derives."""
    _, targets = next(iter(bundle.loaders["train"]))
    assert int(targets.min()) >= 0
    assert int(targets.max()) < bundle.num_classes


def test_metrics_accept_real_predictions(smoke_config, bundle) -> None:
    model = build_model(
        model_config_from_config(smoke_config), scope=smoke_config.dataset.scope
    ).eval()
    accumulator = MetricsAccumulator(bundle.num_classes)
    with torch.no_grad():
        for index, (images, targets) in enumerate(bundle.loaders["validation"]):
            accumulator.update(model(images), targets)
            if index >= 1:
                break
    metrics = accumulator.compute()
    assert metrics.samples > 0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert 0.0 <= metrics.accuracy <= 1.0


# -- a real short run ---------------------------------------------------


@pytest.mark.slow
def test_short_real_run_writes_a_verifiable_checkpoint(
    smoke_config, bundle, tmp_path: Path
) -> None:
    """One capped epoch on real data, then verify the checkpoint's provenance."""
    trainer = build_trainer(
        smoke_config,
        bundle,
        run_dir=tmp_path / "run",
        run_id="integration",
        smoke=True,
        max_train_batches=3,
        max_validation_batches=2,
    )
    history = trainer.fit(epochs=1)
    assert len(history) == 1

    checkpoint = tmp_path / "run" / "last.pt"
    assert checkpoint.is_file()

    metadata = read_metadata(checkpoint)
    assert metadata.scope == "rice10"
    assert metadata.num_classes == 10
    assert metadata.smoke is True
    # The fingerprint must match the pipeline that actually produced the batches.
    assert metadata.preprocessing_fingerprint == bundle.preprocessing.fingerprint

    reloaded, _, _ = load_checkpoint(
        checkpoint,
        scope="rice10",
        preprocessing_fingerprint=bundle.preprocessing.fingerprint,
        strict_preprocessing=True,
    )
    assert reloaded.num_classes == 10


@pytest.mark.slow
def test_real_checkpoint_is_refused_under_the_other_scope(
    smoke_config, bundle, tmp_path: Path
) -> None:
    """The safety property, exercised end to end on real data."""
    trainer = build_trainer(
        smoke_config,
        bundle,
        run_dir=tmp_path / "run",
        run_id="integration",
        smoke=True,
        max_train_batches=2,
        max_validation_batches=1,
    )
    trainer.fit(epochs=1)
    with pytest.raises(CheckpointError, match="never be used under a different scope"):
        load_checkpoint(tmp_path / "run" / "last.pt", scope="full102")


@pytest.mark.slow
def test_training_never_builds_a_test_loader(smoke_config) -> None:
    """Nothing before Phase 9 may touch the test split."""
    built = build_loaders(smoke_config, ("train", "validation"))
    assert "test" not in built.loaders
    assert "test" not in built.datasets


@pytest.mark.slow
def test_class_weights_come_from_training_data_only(smoke_config, tmp_path: Path) -> None:
    """Weighted loss must be derived from the training split alone."""
    config = load_config(
        "smoke_test.yaml", overrides={"training": {"class_weighting": "inverse_sqrt"}}
    )
    if not config.paths.images_dir.is_dir():
        pytest.skip("IP102 images not found")

    weighted = build_loaders(config, ("train", "validation"))
    assert weighted.class_weights is not None
    assert len(weighted.class_weights) == 10

    trainer = Trainer(
        build_model(model_config_from_config(config), scope=config.dataset.scope),
        weighted,
        training_config_from_config(config),
        run_dir=tmp_path / "run",
        model_config=model_config_from_config(config),
    )
    assert trainer.criterion.weight is not None
    # The loss weight tensor lives on the training device, so it comes back to
    # the host before being compared.
    assert trainer.criterion.weight.cpu().tolist() == pytest.approx(
        list(weighted.class_weights)
    )


# -- scope isolation ----------------------------------------------------


def test_a_full102_model_cannot_train_on_rice10_data(bundle, tmp_path: Path) -> None:
    """Three sources of the class count must agree: config, model and data."""
    from farm_pest_ai.vision.training import TrainingConfig, TrainingError

    wrong = build_model(ModelConfig(name="custom_cnn", num_classes=102))
    with pytest.raises(TrainingError, match="refusing to train"):
        Trainer(wrong, bundle, TrainingConfig(), run_dir=tmp_path / "run")
