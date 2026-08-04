"""Tests for DataLoader construction and the rules it enforces.

A complete synthetic project tree is built in ``tmp_path``: derived manifests,
images and a configuration pointing at both. That keeps these tests independent
of ``ip102_v1.1`` while still exercising the real code path a training run takes,
including reading a manifest back off disk and validating its provenance.

The rules under test are the ones that would silently corrupt an experiment:
augmentation must never touch evaluation, class statistics must never come from
evaluation, evaluation order must be preserved, and worker seeding must be
reproducible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from farm_pest_ai.config import Config, load_config
from farm_pest_ai.data.dataset import DatasetError
from farm_pest_ai.data.loaders import (
    DEFAULT_BATCH_SIZE,
    LoaderError,
    RuntimeConfig,
    build_dataset,
    build_loader,
    build_loaders,
    resolve_device,
    runtime_config_from_config,
    sampler_weights,
)
from farm_pest_ai.data.manifests import (
    ClassInfo,
    build_derived_manifest,
    write_derived_manifest,
)
from farm_pest_ai.data.transforms import PreprocessingConfig, describe_transform
from farm_pest_ai.scopes import RICE10

# Loader construction needs torch, torchvision and Pillow; the modules under
# test import them lazily, so the whole file skips rather than each test.
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
PIL_Image = pytest.importorskip("PIL.Image")

#: Records per split in the synthetic dataset. Small, but enough that the
#: training split is imbalanced and multi-batch.
SPLIT_SIZES = {"train": 40, "validation": 20, "test": 20}


@pytest.fixture()
def project(tmp_path: Path) -> Config:
    """Build a synthetic project tree and return a config pointing at it."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = tmp_path / "processed"

    classes = tuple(
        ClassInfo(
            classes_txt_id=original + 1,
            ip102_label=original,
            raw_name=f"pest {original}",
            canonical_name=f"pest {original}",
        )
        for original in range(12)
    )

    for split, size in SPLIT_SIZES.items():
        source: list[tuple[str, int]] = []
        for index in range(size):
            # Cycle through the ten rice10 labels so every class is present,
            # then add extras to label 0 so the training split is imbalanced.
            project_label = index % RICE10.num_classes
            original = RICE10.to_original_label(project_label)
            filename = f"{split}_{index:04d}.jpg"
            PIL_Image.new("RGB", (70, 50), (index * 3 % 256, 90, 60)).save(
                images_dir / filename
            )
            source.append((filename, original))
        if split == "train":
            for extra in range(10):
                filename = f"{split}_extra_{extra:04d}.jpg"
                PIL_Image.new("RGB", (70, 50), (10, 10, 10)).save(images_dir / filename)
                source.append((filename, RICE10.to_original_label(0)))

        manifest = build_derived_manifest(split, source, RICE10, classes)
        write_derived_manifest(manifest, processed_dir)

    return load_config(
        "base.yaml",
        overrides={
            "paths": {
                "processed_dir": str(processed_dir),
                "images_dir": str(images_dir),
            },
            "runtime": {"num_workers": 0, "persistent_workers": False},
            "training": {"batch_size": 8},
        },
    )


# -- runtime configuration ----------------------------------------------


def test_runtime_config_comes_from_the_section(project: Config) -> None:
    runtime = runtime_config_from_config(project)
    assert runtime.num_workers == 0
    assert runtime.persistent_workers is False


def test_persistent_workers_without_workers_is_rejected() -> None:
    with pytest.raises(LoaderError, match=r"persistent_workers requires"):
        RuntimeConfig(num_workers=0, persistent_workers=True).validate()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_workers": -1}, r"num_workers must be >= 0"),
        ({"prefetch_factor": 0}, r"prefetch_factor must be >= 1"),
        ({"device": "tpu"}, r"device must be"),
    ],
)
def test_invalid_runtime_is_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(LoaderError, match=match):
        RuntimeConfig(**{"persistent_workers": False, **kwargs}).validate()


def test_cpu_is_always_resolvable() -> None:
    assert resolve_device("cpu") == "cpu"


def test_auto_resolves_to_an_available_device() -> None:
    assert resolve_device("auto") in ("cpu", "cuda")


def test_explicit_cuda_refuses_a_silent_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An approved training run must not quietly become a CPU run.

    CUDA availability is patched rather than skipped on: this branch matters
    most on a machine that *has* a GPU, where a driver fault at 3 a.m. is
    exactly what would otherwise silently start a days-long CPU run.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(LoaderError, match=r"refusing to fall back to CPU"):
        resolve_device("cuda", allow_cpu_fallback=False)


def test_explicit_cuda_may_fall_back_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda", allow_cpu_fallback=True) == "cpu"


def test_auto_resolves_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_build_loaders_can_refuse_the_cpu_fallback(
    project: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal must survive the whole build path, not just the helper."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config(
        "base.yaml",
        overrides={
            **project.to_dict(),
            "runtime": {**project.section("runtime"), "device": "cuda"},
        },
    )
    with pytest.raises(LoaderError, match=r"refusing to fall back to CPU"):
        build_loaders(config, ("train",), allow_cpu_fallback=False)


# -- datasets from configuration ----------------------------------------


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_dataset_builds_for_every_split(project: Config, split: str) -> None:
    dataset = build_dataset(project, split)
    assert len(dataset) == SPLIT_SIZES[split] + (10 if split == "train" else 0)
    assert dataset.num_classes == 10


def test_dataset_rejects_an_unknown_split(project: Config) -> None:
    with pytest.raises(LoaderError, match=r"unknown split"):
        build_dataset(project, "holdout")


@pytest.mark.parametrize("split", ["validation", "test"])
def test_refusing_to_augment_evaluation(project: Config, split: str) -> None:
    """The single most important guard in this module."""
    with pytest.raises(LoaderError, match=r"refusing to augment"):
        build_dataset(project, split, augment=True)


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    config = load_config(
        "base.yaml",
        overrides={"paths": {"processed_dir": str(tmp_path / "absent")}},
    )
    with pytest.raises(DatasetError, match=r"derived manifest not found"):
        build_dataset(config, "train")


def test_manifest_version_mismatch_is_rejected(project: Config) -> None:
    config = load_config(
        "base.yaml",
        overrides={
            **project.to_dict(),
            "dataset": {**project.section("dataset"), "manifest_version": "9.9.9"},
        },
    )
    with pytest.raises(DatasetError, match=r"manifest version"):
        build_dataset(config, "train")


def test_scope_mismatch_is_rejected(project: Config) -> None:
    """A rice10 manifest must never be read as full102."""
    config = load_config(
        "base.yaml",
        overrides={**project.to_dict(), "dataset": {"scope": "full102"}},
    )
    with pytest.raises(DatasetError):
        build_dataset(config, "train")


# -- pipelines per split ------------------------------------------------


def test_training_is_augmented_and_evaluation_is_not(project: Config) -> None:
    bundle = build_loaders(project, ("train", "validation", "test"))
    train_steps = describe_transform(bundle.datasets["train"].transform)
    assert any("Random" in step for step in train_steps)
    for split in ("validation", "test"):
        steps = describe_transform(bundle.datasets[split].transform)
        assert not [s for s in steps if "Random" in s or "Jitter" in s], steps


def test_augment_false_makes_training_deterministic(project: Config) -> None:
    bundle = build_loaders(project, ("train",), augment=False)
    steps = describe_transform(bundle.datasets["train"].transform)
    assert not [s for s in steps if "Random" in s], steps


# -- batches ------------------------------------------------------------


def test_batches_have_the_expected_shape_and_dtype(project: Config) -> None:
    bundle = build_loaders(project, ("train", "validation"))
    images, labels = next(iter(bundle.loaders["validation"]))
    assert images.shape == (8, 3, 160, 160)
    assert images.dtype is torch.float32
    assert labels.dtype is torch.int64
    assert int(labels.min()) >= 0
    assert int(labels.max()) < bundle.num_classes


def test_evaluation_covers_every_image_exactly_once(project: Config) -> None:
    """drop_last must never apply to evaluation, or images go unscored."""
    bundle = build_loaders(project, ("train", "validation"))
    loader = bundle.loaders["validation"]
    assert loader.drop_last is False
    seen = sum(int(images.shape[0]) for images, _ in loader)
    assert seen == len(bundle.datasets["validation"])


def test_training_drops_a_short_final_batch(project: Config) -> None:
    """BatchNorm must not see a one-sample batch."""
    bundle = build_loaders(project, ("train",), batch_size=16)
    loader = bundle.loaders["train"]
    assert loader.drop_last is True
    for images, _ in loader:
        assert images.shape[0] == 16


def test_evaluation_preserves_manifest_order(project: Config) -> None:
    """Phase 9 joins predictions to the manifest by position."""
    bundle = build_loaders(project, ("train", "validation"))
    dataset = bundle.datasets["validation"]
    collected: list[int] = []
    for _, labels in bundle.loaders["validation"]:
        collected.extend(int(label) for label in labels)
    assert tuple(collected) == dataset.targets


def test_training_is_shuffled(project: Config) -> None:
    """Training order must differ from manifest order, with the same content."""
    bundle = build_loaders(project, ("train",), batch_size=8)
    dataset = bundle.datasets["train"]
    collected: list[int] = []
    for _, labels in bundle.loaders["train"]:
        collected.extend(int(label) for label in labels)

    manifest_order = list(dataset.targets)
    # drop_last removes a partial batch, so compare against the same length.
    assert collected != manifest_order[: len(collected)]
    # Shuffling reorders; it must not invent or lose labels.
    assert set(collected) == set(manifest_order)


def test_the_same_seed_reproduces_the_training_order(project: Config) -> None:
    def order(seed: int) -> list[int]:
        bundle = build_loaders(project, ("train",), batch_size=8, seed=seed)
        return [
            int(label) for _, labels in bundle.loaders["train"] for label in labels
        ]

    # Two independently constructed loaders, same seed: the identical
    # expressions are the point, not a tautology.
    first_run = order(1337)
    second_run = order(1337)
    assert first_run == second_run


def test_a_different_seed_changes_the_training_order(project: Config) -> None:
    def order(seed: int) -> list[int]:
        bundle = build_loaders(project, ("train",), batch_size=8, seed=seed)
        return [
            int(label) for _, labels in bundle.loaders["train"] for label in labels
        ]

    assert order(1337) != order(20260801)


def test_evaluation_batches_are_bit_identical_across_passes(project: Config) -> None:
    """The determinism guarantee, measured end to end through the loader."""
    bundle = build_loaders(project, ("train", "validation"))
    loader = bundle.loaders["validation"]
    first = [images.clone() for images, _ in loader]
    second = [images.clone() for images, _ in loader]
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


# -- loader construction guards -----------------------------------------


def test_batch_size_must_be_positive(project: Config) -> None:
    dataset = build_dataset(project, "train")
    with pytest.raises(LoaderError, match=r"batch_size must be positive"):
        build_loader(dataset, RuntimeConfig(num_workers=0, persistent_workers=False),
                     batch_size=0)


def test_sampler_and_shuffle_are_mutually_exclusive(project: Config) -> None:
    from torch.utils.data import WeightedRandomSampler

    dataset = build_dataset(project, "train")
    sampler = WeightedRandomSampler([1.0] * len(dataset), len(dataset))
    with pytest.raises(LoaderError, match=r"mutually exclusive"):
        build_loader(
            dataset,
            RuntimeConfig(num_workers=0, persistent_workers=False),
            sampler=sampler,
            shuffle=True,
        )


def test_unknown_split_in_build_loaders_is_rejected(project: Config) -> None:
    with pytest.raises(LoaderError, match=r"unknown split"):
        build_loaders(project, ("train", "holdout"))


def test_no_splits_is_rejected(project: Config) -> None:
    with pytest.raises(LoaderError, match=r"at least one split"):
        build_loaders(project, ())


def test_build_loaders_omits_test_by_default(project: Config) -> None:
    """Nothing before Phase 9 may read the test split without asking."""
    bundle = build_loaders(project)
    assert set(bundle.loaders) == {"train", "validation"}


# -- class statistics come from training only ---------------------------


def test_sampler_weights_refuse_evaluation_splits(project: Config) -> None:
    for split in ("validation", "test"):
        with pytest.raises(LoaderError, match=r"only be derived from the training"):
            sampler_weights(build_dataset(project, split))


def test_sampler_weights_favour_the_rare_class(project: Config) -> None:
    dataset = build_dataset(project, "train")
    weights = sampler_weights(dataset)
    assert len(weights) == len(dataset)
    # Label 0 has the extra records, so its samples must weigh least.
    by_label = dict(zip(dataset.targets, weights, strict=True))
    assert by_label[0] == min(by_label.values())


def test_class_weights_are_derived_when_configured(project: Config) -> None:
    config = load_config(
        "base.yaml",
        overrides={
            **project.to_dict(),
            "training": {**project.section("training"), "class_weighting": "inverse"},
        },
    )
    bundle = build_loaders(config, ("train", "validation"))
    assert bundle.class_weights is not None
    assert len(bundle.class_weights) == 10
    # Label 0 is the most common, so it earns the smallest weight.
    assert bundle.class_weights[0] == min(bundle.class_weights)


def test_class_weights_are_absent_by_default(project: Config) -> None:
    assert build_loaders(project, ("train", "validation")).class_weights is None


def test_class_weighting_beta_reaches_the_weight_vector(project: Config) -> None:
    """Beta is plumbed from configuration through to the derived weights.

    Phase 8.1's E9b depends on this: without the passthrough, `effective` would
    silently always use the library default of 0.9999 no matter what the config
    said, and the arm would run at 69.5x instead of the intended 23.5x.
    """
    bundles = {}
    for beta in (0.999, 0.9999):
        config = load_config(
            "base.yaml",
            overrides={
                **project.to_dict(),
                "training": {
                    **project.section("training"),
                    "class_weighting": "effective",
                    "class_weighting_beta": beta,
                },
            },
        )
        bundle = build_loaders(config, ("train", "validation"))
        assert bundle.class_weights is not None
        bundles[beta] = bundle

    gentle = bundles[0.999].class_weights
    strong = bundles[0.9999].class_weights
    assert gentle is not None and strong is not None
    # The two betas must produce genuinely different corrections.
    assert gentle != pytest.approx(strong)
    assert max(strong) / min(strong) > max(gentle) / min(gentle)


def test_class_weighting_beta_is_recorded_in_the_bundle(project: Config) -> None:
    """A run's summary states the beta its weights came from."""
    config = load_config(
        "base.yaml",
        overrides={
            **project.to_dict(),
            "training": {
                **project.section("training"),
                "class_weighting": "effective",
                "class_weighting_beta": 0.999,
            },
        },
    )
    described = build_loaders(config, ("train", "validation")).describe()

    assert described["class_weighting"] == "effective"
    assert described["class_weighting_beta"] == pytest.approx(0.999)


def test_class_weighting_beta_defaults_to_the_library_value(project: Config) -> None:
    """A config naming only the scheme behaves exactly as it did before."""
    config = load_config(
        "base.yaml",
        overrides={
            **project.to_dict(),
            "training": {**project.section("training"), "class_weighting": "effective"},
        },
    )
    bundle = build_loaders(config, ("train", "validation"))
    assert bundle.describe()["class_weighting_beta"] == pytest.approx(0.9999)


def test_class_weighting_without_the_training_split_is_refused(
    project: Config,
) -> None:
    """Weights must never be back-filled from validation data."""
    config = load_config(
        "base.yaml",
        overrides={
            **project.to_dict(),
            "training": {**project.section("training"), "class_weighting": "inverse"},
        },
    )
    with pytest.raises(LoaderError, match=r"needs the training split"):
        build_loaders(config, ("validation",))


# -- the bundle record --------------------------------------------------


def test_describe_records_the_run_provenance(project: Config) -> None:
    import json

    described = build_loaders(project, ("train", "validation")).describe()
    assert described["scope"] == "rice10"
    assert described["num_classes"] == 10
    assert described["class_mapping_version"]
    assert described["preprocessing_fingerprint"]
    assert described["splits"]["train"]["augmented"] is True
    assert described["splits"]["validation"]["augmented"] is False
    assert described["splits"]["train"]["shuffled"] is True
    assert described["splits"]["validation"]["shuffled"] is False
    json.dumps(described)


def test_bundle_reports_the_configured_batch_size(project: Config) -> None:
    assert build_loaders(project, ("train",)).batch_size == 8


def test_default_batch_size_is_used_when_unconfigured(project: Config) -> None:
    """With no training.batch_size, the documented default must apply."""
    data = project.to_dict()
    data.pop("training", None)
    config = load_config("base.yaml", overrides=data)
    assert "batch_size" not in config.section("training")
    assert build_loaders(config, ("validation",)).batch_size == DEFAULT_BATCH_SIZE


def test_preprocessing_override_reaches_the_bundle(project: Config) -> None:
    preprocessing = PreprocessingConfig(image_size=(96, 96)).validate()
    bundle = build_loaders(project, ("validation",), preprocessing=preprocessing)
    images, _ = next(iter(bundle.loaders["validation"]))
    assert images.shape[1:] == (3, 96, 96)
