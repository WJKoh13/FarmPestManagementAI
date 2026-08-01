"""Tests for ``scripts/train.py``, the Phase 7 experiment entry point.

The properties that matter here are safety properties, not numerical ones. A
training script that quietly trains on a subset, or that builds a test loader,
produces artifacts indistinguishable from a correct run — so each guard is
tested against a deliberately broken bundle rather than only against a healthy
one. The AMP step accounting is tested here too, since it is what makes a run
whose gradients keep overflowing visible in the log.

Nothing here trains a real model or touches the dataset; the integration
counterpart is ``test_train_script_integration.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from farm_pest_ai.data.loaders import LoaderBundle, RuntimeConfig  # noqa: E402
from farm_pest_ai.data.transforms import preprocessing_config_from_config  # noqa: E402
from farm_pest_ai.scopes import get_scope  # noqa: E402
from farm_pest_ai.vision.training import (  # noqa: E402
    EpochResult,
    TrainingConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_train_module() -> Any:
    """Import ``scripts/train.py`` as a module.

    The scripts directory is not a package, so the file is loaded by path. This
    is how the script's own functions get tested rather than only its
    command-line behaviour.
    """
    path = PROJECT_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("_train_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = _load_train_module()


# -- the test split is never built ---------------------------------------


class _FakeLoader:
    """A loader stand-in that reports a batch count and nothing else."""

    def __init__(self, batches: int) -> None:
        self._batches = batches

    def __len__(self) -> int:
        return self._batches


class _FakeDataset:
    """A dataset stand-in that reports a length."""

    def __init__(self, images: int) -> None:
        self._images = images

    def __len__(self) -> int:
        return self._images


def _bundle(splits: dict[str, tuple[int, int]]) -> LoaderBundle:
    """Build a LoaderBundle carrying only the named splits.

    Args:
        splits: Split name to ``(images, batches)``.
    """
    from farm_pest_ai.config import Config

    config = Config(
        data={
            "dataset": {"scope": "rice10", "image_size": [160, 160]},
            "preprocessing": {},
            "reproducibility": {"seed": 1337},
        }
    )
    return LoaderBundle(
        loaders={name: _FakeLoader(batches) for name, (_, batches) in splits.items()},
        datasets={name: _FakeDataset(images) for name, (images, _) in splits.items()},
        preprocessing=preprocessing_config_from_config(config),
        runtime=RuntimeConfig(),
        scope=get_scope("rice10"),
        device="cpu",
        batch_size=64,
        seed=1337,
    )


def test_train_splits_constant_excludes_test() -> None:
    """The only splits a run may build are train and validation."""
    assert train.TRAINING_SPLITS == ("train", "validation")
    assert "test" not in train.TRAINING_SPLITS


def test_assert_no_test_split_accepts_a_clean_bundle() -> None:
    bundle = _bundle({"train": (4318, 67), "validation": (721, 12)})
    train.assert_no_test_split(bundle)


def test_assert_no_test_split_rejects_a_leaked_test_loader() -> None:
    """A test loader that reached the bundle aborts the run.

    This is the single most consequential guard in the script: a test loader
    that slips through produces a number that looks entirely normal and
    silently invalidates every decision made after it.
    """
    bundle = _bundle(
        {"train": (4318, 67), "validation": (721, 12), "test": (2166, 34)}
    )
    with pytest.raises(train.TrainingRunError, match=r"carries \['test'\]"):
        train.assert_no_test_split(bundle)


# -- caps belong to the smoke gate, not to an experiment -----------------


class _FakeTrainer:
    """A trainer stand-in exposing only what the cap check reads."""

    def __init__(
        self,
        *,
        max_train_batches: int | None = None,
        max_validation_batches: int | None = None,
        smoke: bool = False,
    ) -> None:
        self.max_train_batches = max_train_batches
        self.max_validation_batches = max_validation_batches
        self.smoke = smoke


def test_assert_no_caps_accepts_an_uncapped_trainer() -> None:
    train.assert_no_caps(_FakeTrainer())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_train_batches": 60},
        {"max_validation_batches": 15},
        {"max_train_batches": 60, "max_validation_batches": 15},
    ],
)
def test_assert_no_caps_rejects_a_capped_trainer(kwargs: dict[str, int]) -> None:
    """A capped run would report a metric computed over a slice."""
    with pytest.raises(train.TrainingRunError, match="batch caps are set"):
        train.assert_no_caps(_FakeTrainer(**kwargs))


def test_assert_no_caps_rejects_a_smoke_trainer() -> None:
    with pytest.raises(train.TrainingRunError, match="smoke"):
        train.assert_no_caps(_FakeTrainer(smoke=True))


# -- the invocation guards -----------------------------------------------


def test_resume_without_a_run_name_is_refused() -> None:
    """Resuming needs to know which run directory to continue."""
    from farm_pest_ai.config import Config

    args = train.build_arg_parser().parse_args(["--resume"])
    message = train._refuse_unusable_invocation(args, Config(data={}))
    assert message is not None
    assert "--run-name" in message


def test_a_smoke_configuration_is_refused() -> None:
    """The smoke config caps batches, so it cannot produce an experiment."""
    from farm_pest_ai.config import Config

    args = train.build_arg_parser().parse_args([])
    config = Config(data={"smoke": {"max_train_batches": 60}})
    message = train._refuse_unusable_invocation(args, config)
    assert message is not None
    assert "smoke" in message


def test_a_normal_invocation_is_not_refused() -> None:
    from farm_pest_ai.config import Config

    args = train.build_arg_parser().parse_args(["--run-name", "rice10_custom_e1"])
    assert train._refuse_unusable_invocation(args, Config(data={})) is None


def test_shorthand_flags_fold_into_the_override_list() -> None:
    """One override list means one precedence rule."""
    args = train.build_arg_parser().parse_args(
        ["--model", "baseline_cnn", "--device", "cuda", "--epochs", "40"]
    )
    train._fold_shorthand_overrides(args)
    assert "model.name=baseline_cnn" in args.overrides
    assert "runtime.device=cuda" in args.overrides
    assert "training.epochs=40" in args.overrides


def test_the_script_never_offers_a_way_to_name_the_test_split() -> None:
    """No flag exposes the test split, and --split does not exist."""
    parser = train.build_arg_parser()
    flags = {action.dest for action in parser._actions}
    assert "split" not in flags
    assert "splits" not in flags
    assert "include_test" not in flags


# -- AMP skipped-step accounting -----------------------------------------


def _epoch(
    epoch: int, *, macro_f1: float, skipped: int, steps: int = 67
) -> EpochResult:
    """Build an EpochResult with the fields the summary reads."""
    from farm_pest_ai.vision.metrics import ClassificationMetrics

    metrics = ClassificationMetrics(
        accuracy=macro_f1,
        macro_f1=macro_f1,
        weighted_f1=macro_f1,
        balanced_accuracy=macro_f1,
        loss=1.0,
        samples=721,
    )
    return EpochResult(
        epoch=epoch,
        train=metrics,
        validation=metrics,
        learning_rate=1e-3,
        train_seconds=10.0,
        validation_seconds=2.0,
        images_per_second=430.0,
        peak_vram_mib=1500.0,
        improved=True,
        best_metric=macro_f1,
        optimizer_steps=steps,
        amp_skipped_steps=skipped,
        amp_final_scale=4096.0,
    )


def test_epoch_result_records_amp_skipped_steps() -> None:
    """The per-epoch JSON Lines record carries the skip count."""
    payload = _epoch(1, macro_f1=0.4, skipped=3).to_dict()
    assert payload["amp_skipped_steps"] == 3
    assert payload["optimizer_steps"] == 67
    assert payload["amp_final_scale"] == 4096.0


def test_summary_totals_amp_skipped_steps_across_the_run() -> None:
    """Calibration skips at the start are expected; a climbing total is not.

    The per-epoch series is kept alongside the total precisely so the two cases
    can be told apart after the fact.
    """
    history = [
        _epoch(1, macro_f1=0.30, skipped=4),
        _epoch(2, macro_f1=0.45, skipped=0),
        _epoch(3, macro_f1=0.41, skipped=1),
    ]
    summary = train.summarize_history(history)
    assert summary["amp_skipped_steps_total"] == 5
    assert summary["amp_skipped_steps_by_epoch"] == [4, 0, 1]
    assert summary["optimizer_steps_total"] == 201


def test_summary_selects_the_best_epoch_by_validation_macro_f1() -> None:
    """Macro F1 is the selection metric, not accuracy and not the last epoch."""
    history = [
        _epoch(1, macro_f1=0.30, skipped=0),
        _epoch(2, macro_f1=0.52, skipped=0),
        _epoch(3, macro_f1=0.48, skipped=0),
    ]
    summary = train.summarize_history(history)
    assert summary["best_epoch"] == 2
    assert summary["best_validation_macro_f1"] == pytest.approx(0.52)
    assert summary["final_epoch"] == 3
    assert summary["epochs_completed"] == 3


def test_summary_of_an_empty_history_is_not_an_error() -> None:
    """An interrupted run before the first epoch still summarises."""
    assert train.summarize_history([])["epochs_completed"] == 0


# -- runtime estimation ---------------------------------------------------


class _TimingTrainer:
    """A trainer stand-in exposing the loaders the estimate reads."""

    def __init__(self, train_batches: int, validation_batches: int) -> None:
        self.bundle = type(
            "_B",
            (),
            {
                "loaders": {
                    "train": _FakeLoader(train_batches),
                    "validation": _FakeLoader(validation_batches),
                },
                "batch_size": 64,
            },
        )()


def test_runtime_estimate_scales_with_epochs_and_batches() -> None:
    """A doubled epoch count doubles the estimate."""
    trainer = _TimingTrainer(67, 12)
    step = {"measured": True, "mean_step_seconds": 0.1}

    one = train.estimate_runtime(trainer, step, 1)
    ten = train.estimate_runtime(trainer, step, 10)

    assert one["train_batches_per_epoch"] == 67
    assert one["validation_batches_per_epoch"] == 12
    # The reported totals are rounded to 0.1 s, so the ten-epoch figure can sit
    # up to 10 x 0.05 s away from ten times the one-epoch figure.
    assert ten["total_seconds"] == pytest.approx(one["total_seconds"] * 10, abs=0.5)
    assert ten["epoch_seconds"] == pytest.approx(one["epoch_seconds"])


def test_runtime_estimate_reports_when_it_could_not_measure() -> None:
    """An unmeasured step never produces a fabricated estimate."""
    result = train.estimate_runtime(_TimingTrainer(67, 12), {"measured": False}, 80)
    assert result == {"estimated": False}


# -- VRAM reporting -------------------------------------------------------


def test_vram_snapshot_reports_no_cuda_for_a_cpu_device() -> None:
    snapshot = train.vram_snapshot("cpu")
    assert snapshot["cuda"] is False
    assert snapshot["device"] == "cpu"


@pytest.mark.gpu
def test_vram_snapshot_reports_free_and_total_on_cuda() -> None:
    """Free VRAM is a real constraint: it is measured, not assumed."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    snapshot = train.vram_snapshot("cuda")
    assert snapshot["cuda"] is True
    assert snapshot["total_mib"] > 0
    assert 0 < snapshot["free_mib"] <= snapshot["total_mib"]


# -- the training config the comparison depends on ------------------------


def test_a_controlled_comparison_needs_identical_training_settings() -> None:
    """Two models compared under one protocol must differ only in the model.

    This is what makes the Phase 7 baseline-versus-custom result mean anything:
    if the two runs also differ in learning rate or epochs, a win cannot be
    attributed to the architecture.
    """
    shared = {
        "learning_rate": 0.002,
        "weight_decay": 0.05,
        "batch_size": 64,
        "epochs": 60,
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
    }
    first = TrainingConfig(**shared).validate()
    second = TrainingConfig(**shared).validate()
    assert first.to_dict() == second.to_dict()
