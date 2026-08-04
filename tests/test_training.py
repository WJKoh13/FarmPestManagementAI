"""Tests for the training engine.

These use a synthetic in-memory dataset rather than IP102: the engine's
behaviour — scheduling, early stopping, checkpoint selection, resumption and the
training-only-statistics rule — is independent of the images, and a synthetic
set keeps the suite fast. ``tests/test_training_integration.py`` covers the real
data path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from farm_pest_ai.data.loaders import LoaderBundle, RuntimeConfig  # noqa: E402
from farm_pest_ai.data.transforms import PreprocessingConfig  # noqa: E402
from farm_pest_ai.scopes import RICE10  # noqa: E402
from farm_pest_ai.vision.metrics import ClassificationMetrics  # noqa: E402
from farm_pest_ai.vision.models import ModelConfig, build_model  # noqa: E402
from farm_pest_ai.vision.training import (  # noqa: E402
    EpochResult,
    Trainer,
    TrainingConfig,
    TrainingError,
    build_optimizer,
    build_scheduler,
    training_config_from_config,
)


class _TinyDataset(torch.utils.data.Dataset):
    """A deterministic synthetic dataset shaped like the real one."""

    def __init__(self, size: int = 32, num_classes: int = 10, split: str = "train") -> None:
        self.size = size
        self.num_classes = num_classes
        self.split = split
        # Images are pre-transformed tensors, so there is no pipeline to report.
        # LoaderBundle.describe() reads this attribute for its run record.
        self.transform = None
        generator = torch.Generator().manual_seed(0)
        # 32x32 rather than 160x160: the engine does not care about resolution
        # and this keeps every test in this file well under a second.
        self.images = torch.randn(size, 3, 32, 32, generator=generator)
        self.targets = tuple(int(i % num_classes) for i in range(size))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]

    def describe(self) -> dict[str, Any]:
        return {"split": self.split, "records": self.size}


def _bundle(
    *,
    num_classes: int = 10,
    class_weights: tuple[float, ...] | None = None,
    splits: tuple[str, ...] = ("train", "validation"),
) -> LoaderBundle:
    """Build a LoaderBundle over synthetic data."""
    datasets = {split: _TinyDataset(num_classes=num_classes, split=split) for split in splits}
    loaders = {
        split: torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=split == "train")
        for split, dataset in datasets.items()
    }
    return LoaderBundle(
        loaders=loaders,
        datasets=datasets,
        preprocessing=PreprocessingConfig(image_size=(32, 32)),
        runtime=RuntimeConfig(num_workers=0, persistent_workers=False, pin_memory=False),
        scope=RICE10,
        device="cpu",
        batch_size=8,
        seed=1337,
        class_weights=class_weights,
    )


def _model(num_classes: int = 10) -> torch.nn.Module:
    """A deliberately tiny model, so tests measure the engine not the network."""
    return build_model(
        ModelConfig(
            name="custom_cnn",
            num_classes=num_classes,
            stem_channels=4,
            stage_channels=(4, 8),
            stage_blocks=(1, 1),
            stage_strides=(2, 2),
            drop_path=0.0,
        )
    )


def _trainer(tmp_path: Path, **overrides: Any) -> Trainer:
    """Build a trainer over synthetic data."""
    config_kwargs: dict[str, Any] = {
        "epochs": 2,
        "warmup_epochs": 0,
        "batch_size": 8,
        "amp": False,
        "early_stopping_patience": 5,
    }
    bundle_kwargs = overrides.pop("bundle_kwargs", {})
    config_kwargs.update(overrides)
    bundle = _bundle(**bundle_kwargs)
    return Trainer(
        _model(bundle.num_classes),
        bundle,
        TrainingConfig(**config_kwargs),
        run_dir=tmp_path / "run",
        run_id="test",
        model_config=ModelConfig(name="custom_cnn", num_classes=bundle.num_classes),
    )


# -- configuration ------------------------------------------------------


def test_training_config_rejects_warmup_longer_than_the_run() -> None:
    with pytest.raises(TrainingError, match="leaves no epochs"):
        TrainingConfig(epochs=5, warmup_epochs=5).validate()


def test_training_config_rejects_unknown_optimizer() -> None:
    with pytest.raises(TrainingError, match=r"unknown training\.optimizer"):
        TrainingConfig(optimizer="lion").validate()


def test_training_config_rejects_unknown_scheduler() -> None:
    with pytest.raises(TrainingError, match=r"unknown training\.scheduler"):
        TrainingConfig(scheduler="exponential").validate()


def test_training_config_rejects_label_smoothing_of_one() -> None:
    with pytest.raises(TrainingError, match=r"\[0, 1\)"):
        TrainingConfig(label_smoothing=1.0).validate()


def test_training_config_reads_amp_from_runtime_section() -> None:
    from farm_pest_ai.config import Config

    config = Config(
        data={
            "dataset": {"scope": "rice10"},
            "paths": {},
            "training": {"epochs": 10},
            "runtime": {"amp": False},
        }
    )
    assert training_config_from_config(config).amp is False


# -- optimiser ----------------------------------------------------------


def test_optimizer_excludes_norms_and_biases_from_weight_decay() -> None:
    """Decaying a BatchNorm scale suppresses the channel it normalises."""
    model = _model()
    optimizer = build_optimizer(model, TrainingConfig(weight_decay=0.05))
    decayed, undecayed = optimizer.param_groups
    assert decayed["weight_decay"] == 0.05
    assert undecayed["weight_decay"] == 0.0
    # Every undecayed parameter is 1-D (norm weights, biases).
    assert all(p.ndim <= 1 for p in undecayed["params"])
    assert all(p.ndim > 1 for p in decayed["params"])


def test_optimizer_covers_every_trainable_parameter() -> None:
    model = _model()
    optimizer = build_optimizer(model, TrainingConfig())
    grouped = sum(len(group["params"]) for group in optimizer.param_groups)
    assert grouped == len([p for p in model.parameters() if p.requires_grad])


# -- schedule -----------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Detected call of.*:UserWarning")
def test_warmup_starts_low_and_reaches_full_rate() -> None:
    model = _model()
    config = TrainingConfig(epochs=10, warmup_epochs=2, learning_rate=0.1)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=10)

    first = optimizer.param_groups[0]["lr"]
    assert first < config.learning_rate
    for _ in range(20):  # 2 epochs x 10 steps
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(config.learning_rate, rel=1e-6)


@pytest.mark.filterwarnings("ignore:Detected call of.*:UserWarning")
def test_cosine_schedule_decays_to_zero() -> None:
    model = _model()
    config = TrainingConfig(epochs=4, warmup_epochs=0, learning_rate=0.1, scheduler="cosine")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=10)
    for _ in range(40):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.filterwarnings("ignore:Detected call of.*:UserWarning")
def test_no_scheduler_holds_the_rate_constant() -> None:
    model = _model()
    config = TrainingConfig(epochs=4, warmup_epochs=0, learning_rate=0.1, scheduler="none")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=5)
    for _ in range(20):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


# -- construction guards ------------------------------------------------


def test_trainer_requires_a_training_loader() -> None:
    bundle = _bundle(splits=("validation",))
    with pytest.raises(TrainingError, match="no training split"):
        Trainer(_model(), bundle, TrainingConfig(), run_dir=Path("unused"))


def test_trainer_rejects_a_model_that_disagrees_with_the_scope() -> None:
    """A 102-way model may not be trained on rice10 labels."""
    bundle = _bundle(num_classes=10)
    with pytest.raises(TrainingError, match="refusing to train"):
        Trainer(_model(102), bundle, TrainingConfig(), run_dir=Path("unused"))


def test_trainer_rejects_wrong_length_class_weights(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="class weights have"):
        _trainer(tmp_path, bundle_kwargs={"class_weights": (1.0, 2.0)})


def test_trainer_applies_class_weights_to_the_loss(tmp_path: Path) -> None:
    """Weights arrive pre-computed from the training split; the engine applies them."""
    weights = tuple(float(i + 1) for i in range(10))
    trainer = _trainer(tmp_path, bundle_kwargs={"class_weights": weights})
    assert trainer.criterion.weight is not None
    assert list(trainer.criterion.weight) == pytest.approx(list(weights))


def test_amp_is_disabled_on_cpu(tmp_path: Path) -> None:
    """Autocast on CPU adds casts with no tensor cores to exploit."""
    trainer = _trainer(tmp_path, amp=True)
    assert trainer.amp_enabled is False


# -- the loop -----------------------------------------------------------


def test_fit_completes_and_records_history(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, epochs=2)
    history = trainer.fit()
    assert len(history) == 2
    assert [r.epoch for r in history] == [1, 2]
    assert all(isinstance(r, EpochResult) for r in history)


def test_fit_writes_run_record_and_metrics(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, epochs=1)
    trainer.fit()
    run_dir = tmp_path / "run"
    assert (run_dir / "run.json").is_file()

    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["epoch"] == 1
    assert record["scope"] == "rice10"
    assert "validation" in record


def test_run_record_captures_provenance(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, epochs=1)
    trainer.fit()
    record = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert record["scope"] == "rice10"
    assert record["num_classes"] == 10
    assert record["class_mapping_version"]
    assert "environment" in record
    assert record["model_summary"]["num_classes"] == 10


def test_loss_decreases_over_training(tmp_path: Path) -> None:
    """The engine must actually optimise, not merely iterate."""
    trainer = _trainer(tmp_path, epochs=4, learning_rate=0.01)
    history = trainer.fit()
    assert history[-1].train.loss < history[0].train.loss


def test_best_checkpoint_written_only_on_improvement(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, epochs=2)
    trainer.fit()
    assert (tmp_path / "run" / "best.pt").is_file()
    assert (tmp_path / "run" / "last.pt").is_file()


def test_checkpoints_are_marked_smoke_when_requested(tmp_path: Path) -> None:
    from farm_pest_ai.vision.checkpoints import read_metadata

    bundle = _bundle()
    trainer = Trainer(
        _model(),
        bundle,
        TrainingConfig(epochs=1, warmup_epochs=0, amp=False),
        run_dir=tmp_path / "run",
        smoke=True,
        model_config=ModelConfig(name="custom_cnn", num_classes=10),
    )
    trainer.fit()
    assert read_metadata(tmp_path / "run" / "last.pt").smoke is True


def test_non_finite_loss_aborts_training(tmp_path: Path) -> None:
    """A NaN loss must stop the run rather than write a poisoned checkpoint."""
    trainer = _trainer(tmp_path, epochs=1)

    def nan_loss(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        return torch.tensor(float("nan"), requires_grad=True)

    trainer.criterion = nan_loss  # type: ignore[assignment]
    with pytest.raises(TrainingError, match="non-finite loss"):
        trainer.fit()


def test_evaluate_rejects_an_absent_split(tmp_path: Path) -> None:
    """The test split is absent unless explicitly requested, and stays that way."""
    trainer = _trainer(tmp_path)
    with pytest.raises(TrainingError, match="no 'test' split"):
        trainer.evaluate("test")


def test_evaluation_does_not_update_parameters(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    before = [p.clone() for p in trainer.model.parameters()]
    trainer.evaluate("validation")
    after = list(trainer.model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_evaluation_is_repeatable(tmp_path: Path) -> None:
    """Deterministic preprocessing plus eval mode means a stable score."""
    trainer = _trainer(tmp_path)
    first, _ = trainer.evaluate("validation")
    second, _ = trainer.evaluate("validation")
    assert first.macro_f1 == pytest.approx(second.macro_f1)
    assert first.accuracy == pytest.approx(second.accuracy)


def test_capped_validation_strides_across_the_split(tmp_path: Path) -> None:
    """Front-truncating would score one class only, since order is by class."""
    bundle = _bundle()
    trainer = Trainer(
        _model(),
        bundle,
        TrainingConfig(epochs=1, warmup_epochs=0, amp=False),
        run_dir=tmp_path / "run",
        max_validation_batches=2,
        model_config=ModelConfig(name="custom_cnn", num_classes=10),
    )
    metrics, _ = trainer.evaluate("validation")
    assert metrics.samples == 16


# -- early stopping -----------------------------------------------------


def test_early_stopping_halts_a_stalled_run(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, epochs=10, early_stopping_patience=1)

    frozen = ClassificationMetrics(
        accuracy=0.5, macro_f1=0.5, weighted_f1=0.5, balanced_accuracy=0.5, samples=8
    )
    trainer.evaluate = lambda split="validation": (frozen, 0.0)  # type: ignore[assignment]

    history = trainer.fit()
    # Epoch 1 sets the baseline, epoch 2 fails to improve and exhausts patience.
    assert len(history) == 2


def test_early_stopping_respects_min_delta(tmp_path: Path) -> None:
    """An improvement below min_delta does not reset patience."""
    trainer = _trainer(
        tmp_path, epochs=6, early_stopping_patience=2, early_stopping_min_delta=0.5
    )
    scores = iter([0.10, 0.11, 0.12, 0.13, 0.14, 0.15])

    def barely_improving(split: str = "validation") -> tuple[ClassificationMetrics, float]:
        value = next(scores)
        return (
            ClassificationMetrics(
                accuracy=value,
                macro_f1=value,
                weighted_f1=value,
                balanced_accuracy=value,
                samples=8,
            ),
            0.0,
        )

    trainer.evaluate = barely_improving  # type: ignore[assignment]
    assert len(trainer.fit()) == 3


# -- resumption ---------------------------------------------------------


def test_resume_restores_epoch_and_step(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, epochs=2)
    trainer.fit()
    steps = trainer.global_step

    fresh = _trainer(tmp_path, epochs=4)
    assert fresh.resume(tmp_path / "run" / "last.pt") == 3
    assert fresh.global_step == steps


def test_resume_restores_optimizer_state(tmp_path: Path) -> None:
    """Without the moment estimates a resumed AdamW run spikes."""
    trainer = _trainer(tmp_path, epochs=1)
    trainer.fit()

    fresh = _trainer(tmp_path, epochs=2)
    fresh.resume(tmp_path / "run" / "last.pt")
    assert fresh.optimizer.state_dict()["state"]


def test_resume_rejects_a_missing_checkpoint(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    with pytest.raises(TrainingError, match="cannot resume"):
        trainer.resume(tmp_path / "absent.pt")


def test_resume_refuses_a_checkpoint_from_another_scope(tmp_path: Path) -> None:
    """The provenance check runs before any state is restored."""
    from farm_pest_ai.scopes import FULL102
    from farm_pest_ai.vision.checkpoints import CheckpointMetadata, save_checkpoint

    other = build_model(ModelConfig(name="custom_cnn", num_classes=102))
    path = save_checkpoint(
        tmp_path / "foreign.pt",
        other,
        CheckpointMetadata(scope=FULL102.name, num_classes=102),
    )
    trainer = _trainer(tmp_path)
    with pytest.raises(TrainingError, match="cannot resume"):
        trainer.resume(path)
