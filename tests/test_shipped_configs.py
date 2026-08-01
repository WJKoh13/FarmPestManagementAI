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
