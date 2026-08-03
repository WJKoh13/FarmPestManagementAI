"""Validate the configuration files that actually ship with the project.

These tests parse every file in ``configs/`` so that a typo, a broken
``extends`` chain or a scope/num_classes contradiction fails immediately rather
than at the start of a long training run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from farm_pest_ai.config import load_config
from farm_pest_ai.scopes import num_classes_for

#: Configs that resolve to a complete, valid configuration on their own.
STANDALONE = [
    "base.yaml",
    "data_rice10.yaml",
    "data_full102.yaml",
    "model_baseline.yaml",
    "model_custom.yaml",
    "smoke_test.yaml",
    "exp_rice10_protocol_a.yaml",
    "exp_rice10_e1_epochs100.yaml",
    "exp_rice10_e2_224.yaml",
    "exp_rice10_e3_crop08.yaml",
    "exp_full102_protocol_a.yaml",
    "app.yaml",
    "llm.yaml",
]

#: Phase 1 measurements. The scope configs must keep agreeing with these.
PHASE1_COUNTS = {
    "rice10": {"train": 4318, "validation": 721, "test": 2166, "total": 7205},
    "full102": {"train": 45095, "validation": 7508, "test": 22619, "total": 75222},
}


def config_path(configs_dir: Path, name: str) -> Path:
    path = configs_dir / name
    assert path.is_file(), f"missing shipped config: {path}"
    return path


@pytest.mark.parametrize("name", STANDALONE)
def test_shipped_config_is_valid_yaml(configs_dir: Path, name: str) -> None:
    data = yaml.safe_load(config_path(configs_dir, name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)


@pytest.mark.parametrize("name", STANDALONE)
def test_shipped_config_resolves_and_validates(configs_dir: Path, name: str) -> None:
    config = load_config(config_path(configs_dir, name))
    assert config.dataset.num_classes == num_classes_for(config.dataset.scope_name)
    assert config.paths.project_root.is_dir()


def test_base_defaults_to_rice10(configs_dir: Path) -> None:
    assert load_config(config_path(configs_dir, "base.yaml")).dataset.scope_name == "rice10"


@pytest.mark.parametrize(
    ("name", "scope", "expected"), [("data_rice10.yaml", "rice10", 10),
                                    ("data_full102.yaml", "full102", 102)]
)
def test_scope_configs_select_the_right_scope(
    configs_dir: Path, name: str, scope: str, expected: int
) -> None:
    config = load_config(config_path(configs_dir, name))
    assert config.dataset.scope_name == scope
    assert config.num_classes == expected


@pytest.mark.parametrize("name", ["data_rice10.yaml", "data_full102.yaml"])
def test_expected_counts_match_phase1(configs_dir: Path, name: str) -> None:
    """Guard the audited dataset counts against accidental edits."""
    config = load_config(config_path(configs_dir, name))
    counts = config.section("expected_counts")
    expected = PHASE1_COUNTS[config.dataset.scope_name]
    assert counts == expected
    assert counts["train"] + counts["validation"] + counts["test"] == counts["total"]


@pytest.mark.parametrize("name", STANDALONE)
def test_no_config_hardcodes_num_classes(configs_dir: Path, name: str) -> None:
    """num_classes must always be derived from the scope, never written down."""
    text = config_path(configs_dir, name).read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "num_classes:" not in stripped, (
            f"{name} states num_classes; it must be derived from dataset.scope"
        )


@pytest.mark.parametrize("name", STANDALONE)
def test_no_developer_absolute_paths(configs_dir: Path, name: str) -> None:
    """Configs must stay portable between Windows and Linux containers."""
    text = config_path(configs_dir, name).read_text(encoding="utf-8")
    for marker in ("C:\\", "D:\\", "/home/", "/Users/"):
        assert marker not in text, f"{name} contains a developer-specific path: {marker}"


def test_model_configs_share_the_dataset_scope(configs_dir: Path) -> None:
    """A model config must work unchanged for either scope."""
    for name in ("model_baseline.yaml", "model_custom.yaml"):
        path = config_path(configs_dir, name)
        assert load_config(path).num_classes == 10
        overridden = load_config(path, cli_overrides=["dataset.scope=full102"])
        assert overridden.num_classes == 102


def test_smoke_config_is_bounded(configs_dir: Path) -> None:
    config = load_config(config_path(configs_dir, "smoke_test.yaml"))
    assert config.get("training.epochs") == 1
    assert config.get("smoke.max_train_batches", 0) > 0
    assert config.get("smoke.overfit_steps", 0) > 0


@pytest.mark.parametrize("name", STANDALONE)
def test_augmentation_is_never_promised_for_evaluation(
    configs_dir: Path, name: str
) -> None:
    """Augmentation lives under `preprocessing.augmentation` and is train-only.

    A config growing a `validation`/`test` augmentation key would be a silent
    correctness bug, since evaluation preprocessing must stay deterministic.
    """
    data = yaml.safe_load(config_path(configs_dir, name).read_text(encoding="utf-8"))
    preprocessing = data.get("preprocessing") or {}
    assert "validation" not in preprocessing
    assert "test" not in preprocessing


def test_base_preprocessing_defaults(configs_dir: Path) -> None:
    """Pin the Phase 5 preprocessing decisions against accidental edits."""
    config = load_config(config_path(configs_dir, "base.yaml"))
    assert config.get("preprocessing.interpolation") == "bilinear"
    # null keeps the whole frame; a value here would centre-crop instead.
    assert config.get("preprocessing.resize_shorter_side") is None
    assert config.get("preprocessing.augmentation.enabled") is True
    # Ground-referenced photographs: an inverted insect is not a real input.
    assert config.get("preprocessing.augmentation.vertical_flip") == 0.0
    # Pest identification leans on colour, so hue shifts stay small.
    assert config.get("preprocessing.augmentation.color_jitter_hue") <= 0.05


def test_evaluation_never_drops_a_batch(configs_dir: Path) -> None:
    """drop_last is a training-only setting; the loader enforces the rest."""
    config = load_config(config_path(configs_dir, "base.yaml"))
    assert config.get("runtime.drop_last") is True


def test_app_config_defaults_are_safe(configs_dir: Path) -> None:
    """Safety-relevant defaults must not silently drift."""
    config = load_config(config_path(configs_dir, "app.yaml"))
    assert config.get("safety.suppress_treatment_when_uncertain") is True
    assert config.get("knowledge.allow_unverified_guidance") is False
    assert config.get("safety.degrade_gracefully_without_llm") is True
    assert config.get("safety.require_source_for_dosage") is True
    assert config.get("inference.checkpoint") is None  # set only after Phase 9


def test_the_rice10_comparison_holds_the_protocol_identical(configs_dir: Path) -> None:
    """The two architectures must differ only in the architecture.

    This is what makes the Phase 7 result interpretable. Layered on their own,
    ``model_baseline.yaml`` and ``model_custom.yaml`` differ in learning rate,
    epochs, warmup, label smoothing and patience, so a win could be attributed
    to any of five things. ``exp_rice10_protocol_a.yaml`` is layered second and
    states the whole ``training`` section, which overrides both. If a future
    edit reintroduces a difference, this fails rather than producing a
    comparison that quietly means nothing.
    """
    from farm_pest_ai.vision.models import model_config_from_config
    from farm_pest_ai.vision.training import training_config_from_config

    experiment = config_path(configs_dir, "exp_rice10_protocol_a.yaml")
    resolved = {
        name: load_config([config_path(configs_dir, name), experiment])
        for name in ("model_baseline.yaml", "model_custom.yaml")
    }

    baseline, custom = resolved["model_baseline.yaml"], resolved["model_custom.yaml"]

    # Identical protocol, down to every field the trainer reads.
    assert (
        training_config_from_config(baseline).to_dict()
        == training_config_from_config(custom).to_dict()
    )
    # Identical data handling: same scope, seed, image size and preprocessing.
    for key in (
        "dataset.scope",
        "dataset.image_size",
        "reproducibility.seed",
        "preprocessing.augmentation",
        "runtime.amp",
    ):
        assert baseline.get(key) == custom.get(key), f"{key} differs between the arms"

    # And the one thing that must differ.
    assert model_config_from_config(baseline).name == "baseline_cnn"
    assert model_config_from_config(custom).name == "custom_cnn"
    assert baseline.num_classes == custom.num_classes == 10


def test_full102_protocol_differs_from_rice10_e0_only_in_scope(
    configs_dir: Path,
) -> None:
    """Phase 8 carries the confirmed rice10 recipe across unchanged.

    The scope comparison is only interpretable if the *task* is the one thing
    that changed. E4 did not confirm 224x224's gain over 160x160 — which is not
    the same as showing 160 superior — so 160 is *retained* and the Phase 8
    protocol reuses the E0 recipe verbatim; if a future edit retunes a field
    here, this fails rather than silently producing a scope result that is
    partly a protocol result.
    """
    from farm_pest_ai.vision.training import training_config_from_config

    rice10 = load_config(
        [
            config_path(configs_dir, "model_custom.yaml"),
            config_path(configs_dir, "exp_rice10_protocol_a.yaml"),
        ]
    )
    full102 = load_config(
        [
            config_path(configs_dir, "model_custom.yaml"),
            config_path(configs_dir, "exp_full102_protocol_a.yaml"),
        ]
    )

    # Every field the trainer reads is identical across the two scopes.
    assert (
        training_config_from_config(rice10).to_dict()
        == training_config_from_config(full102).to_dict()
    )
    # As is the preprocessing, including the input size E4 retained.
    assert rice10.get("dataset.image_size") == full102.get("dataset.image_size")
    assert list(full102.get("dataset.image_size")) == [160, 160]
    assert rice10.get("preprocessing.augmentation") == full102.get(
        "preprocessing.augmentation"
    )
    assert rice10.get("reproducibility.seed") == full102.get("reproducibility.seed")

    # And the one thing that must differ.
    assert rice10.get("dataset.scope") == "rice10"
    assert full102.get("dataset.scope") == "full102"
    assert rice10.num_classes == 10
    assert full102.num_classes == 102


def test_full102_protocol_holds_the_protocol_identical_across_arms(
    configs_dir: Path,
) -> None:
    """Both architectures resolve to the same protocol under Phase 8, as in Phase 7."""
    from farm_pest_ai.vision.models import model_config_from_config
    from farm_pest_ai.vision.training import training_config_from_config

    experiment = config_path(configs_dir, "exp_full102_protocol_a.yaml")
    baseline = load_config([config_path(configs_dir, "model_baseline.yaml"), experiment])
    custom = load_config([config_path(configs_dir, "model_custom.yaml"), experiment])

    assert (
        training_config_from_config(baseline).to_dict()
        == training_config_from_config(custom).to_dict()
    )
    for key in (
        "dataset.scope",
        "dataset.image_size",
        "reproducibility.seed",
        "preprocessing.augmentation",
        "runtime.amp",
    ):
        assert baseline.get(key) == custom.get(key), f"{key} differs between the arms"

    assert model_config_from_config(baseline).name == "baseline_cnn"
    assert model_config_from_config(custom).name == "custom_cnn"
    assert baseline.num_classes == custom.num_classes == 102


def test_full102_protocol_applies_no_imbalance_mitigation(configs_dir: Path) -> None:
    """82x imbalance is answered by the metric, not by reweighting the loss.

    No imbalance mitigation is specified in the Phase 8 protocol, so introducing
    one would confound the scope comparison with a second change. Weighting or
    resampling is a separate one-variable experiment to run after this control.
    This pins the choice as deliberate rather than defaulted.
    """
    config = load_config(
        [
            config_path(configs_dir, "model_custom.yaml"),
            config_path(configs_dir, "exp_full102_protocol_a.yaml"),
        ]
    )
    assert config.get("training.class_weighting") == "none"
    # Selection and early stopping still run on macro F1, which is what makes
    # ignoring the rare tail costly.
    assert config.get("training.early_stopping.metric") == "macro_f1"
    assert config.get("training.checkpoint.monitor") == "macro_f1"


#: The Phase 7.2 screening experiments, each with the single field it is
#: allowed to change relative to the E0 control. A one-variable-at-a-time
#: experiment whose config drifts to two variables produces a number that
#: cannot be attributed to anything, so the constraint is pinned here.
PHASE72_EXPERIMENTS = {
    "exp_rice10_e1_epochs100.yaml": "training.epochs",
    "exp_rice10_e2_224.yaml": "dataset.image_size",
    "exp_rice10_e3_crop08.yaml": "preprocessing.augmentation.scale",
}


def _flatten(payload: dict[str, object], prefix: str = "") -> dict[str, object]:
    """Flatten a nested config mapping to dotted keys."""
    flat: dict[str, object] = {}
    for key, value in payload.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, dotted))
        else:
            flat[dotted] = value
    return flat


@pytest.mark.parametrize(
    ("experiment", "expected_key"), sorted(PHASE72_EXPERIMENTS.items())
)
def test_phase72_experiments_change_exactly_one_variable(
    configs_dir: Path, experiment: str, expected_key: str
) -> None:
    """Each screening experiment differs from the E0 control in one field only."""
    control = load_config(
        [
            config_path(configs_dir, "model_custom.yaml"),
            config_path(configs_dir, "exp_rice10_protocol_a.yaml"),
        ]
    )
    variant = load_config(
        [
            config_path(configs_dir, "model_custom.yaml"),
            config_path(configs_dir, experiment),
        ]
    )

    control_flat = _flatten(control.to_dict())
    variant_flat = _flatten(variant.to_dict())
    differences = {
        key
        for key in set(control_flat) | set(variant_flat)
        if control_flat.get(key) != variant_flat.get(key)
        # The provenance list naturally names a different file.
        and "config_sources" not in key
    }

    assert differences == {expected_key}, (
        f"{experiment} changes {sorted(differences)}, not just {expected_key!r}"
    )
    # The scope, and therefore the class count, is never a screening variable.
    assert variant.num_classes == control.num_classes == 10


def test_phase72_experiments_never_touch_the_test_split(configs_dir: Path) -> None:
    """No screening config may name the test split in any form."""
    for experiment in PHASE72_EXPERIMENTS:
        text = config_path(configs_dir, experiment).read_text(encoding="utf-8")
        body = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("#")
        )
        assert "test" not in body, f"{experiment} references the test split"


def test_the_shipped_baseline_is_not_the_model_config_default(configs_dir: Path) -> None:
    """The shipped baseline is three stages, not the four-stage default.

    ``ModelConfig`` defaults to ``custom_cnn``'s four-stage widths, so building
    ``ModelConfig(name="baseline_cnn")`` produces a 3.36M-parameter model that
    no configuration file describes. ``model_baseline.yaml`` ships three stages
    and 1.15M parameters, and that is what an experiment trains. Phase 6's
    reported 3.36M came from the defaults; this test pins the distinction so
    the two cannot be conflated again.
    """
    torch = pytest.importorskip("torch")
    assert torch is not None

    from farm_pest_ai.vision.models import (
        ModelConfig,
        build_model,
        count_parameters,
        model_config_from_config,
    )

    default = ModelConfig(name="baseline_cnn", num_classes=10)
    shipped = model_config_from_config(
        load_config(config_path(configs_dir, "model_baseline.yaml"))
    )

    assert len(shipped.stage_channels) == 3
    assert list(shipped.stage_channels) == [64, 128, 256]
    assert len(default.stage_channels) == 4
    assert shipped.stage_channels != default.stage_channels

    shipped_params = count_parameters(build_model(shipped, scope="rice10"))["total"]
    assert shipped_params == 1_148_874


def test_the_shipped_baseline_is_a_credible_control(configs_dir: Path) -> None:
    """The control must be comparable in size to the model it is controlling.

    A control an order of magnitude smaller or larger would make the Phase 7
    comparison a capacity result rather than an architecture one. As shipped the
    two are within 1.3x, with the baseline the *smaller* of the pair — which is
    the opposite of the Phase 6 note that custom_cnn is 2.3x smaller, since that
    compared against the four-stage default.
    """
    pytest.importorskip("torch")
    from farm_pest_ai.vision.models import (
        build_model,
        count_parameters,
        model_config_from_config,
    )

    counts = {
        name: count_parameters(
            build_model(
                model_config_from_config(load_config(config_path(configs_dir, name))),
                scope="rice10",
            )
        )["total"]
        for name in ("model_baseline.yaml", "model_custom.yaml")
    }
    ratio = max(counts.values()) / min(counts.values())
    assert ratio < 2.0, f"the two arms differ {ratio:.1f}x in size: {counts}"


def test_llm_candidates_are_marked_unverified(configs_dir: Path) -> None:
    """Model tags must not be presented as verified before Phase 11 checks them."""
    config = load_config(config_path(configs_dir, "llm.yaml"))
    candidates = config.get("llm.candidates")
    assert isinstance(candidates, list) and candidates
    for candidate in candidates:
        assert candidate["verified"] is False, (
            "Ollama tags must be confirmed against the library in Phase 11 "
            "before being marked verified"
        )
    assert config.get("llm.selected") is None
    assert config.get("llm.grounding.require_evidence") is True
    assert config.get("llm.grounding.refuse_without_evidence") is True
