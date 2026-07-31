"""Tests for configuration loading, layering and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from farm_pest_ai.config import (
    Config,
    ConfigError,
    deep_merge,
    env_overrides,
    load_config,
    parse_override,
)


# --------------------------------------------------------------------------
# scope / num_classes consistency (Phase 2 exit criterion)
# --------------------------------------------------------------------------
def test_num_classes_derived_from_scope(minimal_config: dict[str, Any]) -> None:
    config = Config(data=minimal_config)
    assert config.dataset.scope_name == "rice10"
    assert config.dataset.num_classes == 10
    assert config.num_classes == 10


def test_full102_derives_102_classes(minimal_config: dict[str, Any]) -> None:
    minimal_config["dataset"]["scope"] = "full102"
    assert Config(data=minimal_config).num_classes == 102


def test_matching_num_classes_is_accepted(minimal_config: dict[str, Any]) -> None:
    """Stating the correct value is redundant but not an error."""
    minimal_config["dataset"]["num_classes"] = 10
    assert Config(data=minimal_config).num_classes == 10


@pytest.mark.parametrize(
    ("scope", "stated"),
    [("rice10", 102), ("rice10", 9), ("full102", 10), ("full102", 101)],
)
def test_inconsistent_scope_and_num_classes_rejected(
    minimal_config: dict[str, Any], scope: str, stated: int
) -> None:
    """A num_classes that contradicts the scope must be a hard error."""
    minimal_config["dataset"]["scope"] = scope
    minimal_config["dataset"]["num_classes"] = stated
    config = Config(data=minimal_config)
    with pytest.raises(ConfigError, match="contradicts dataset.scope"):
        config.validate()


def test_non_integer_num_classes_rejected(minimal_config: dict[str, Any]) -> None:
    minimal_config["dataset"]["num_classes"] = "ten"
    config = Config(data=minimal_config)
    with pytest.raises(ConfigError, match="must be an integer"):
        config.validate()


def test_unknown_scope_rejected(minimal_config: dict[str, Any]) -> None:
    minimal_config["dataset"]["scope"] = "rice20"
    config = Config(data=minimal_config)
    with pytest.raises(ConfigError, match="unknown dataset scope"):
        config.validate()


def test_missing_scope_rejected() -> None:
    config = Config(data={"dataset": {}, "paths": {}})
    with pytest.raises(ConfigError, match="dataset.scope is required"):
        config.validate()


def test_missing_dataset_section_rejected() -> None:
    config = Config(data={"paths": {}})
    with pytest.raises(ConfigError, match="missing a 'dataset' section"):
        config.validate()


# --------------------------------------------------------------------------
# image_size
# --------------------------------------------------------------------------
def test_image_size_scalar_expands(minimal_config: dict[str, Any]) -> None:
    minimal_config["dataset"]["image_size"] = 160
    assert Config(data=minimal_config).dataset.image_size == (160, 160)


def test_image_size_defaults_to_160(minimal_config: dict[str, Any]) -> None:
    del minimal_config["dataset"]["image_size"]
    assert Config(data=minimal_config).dataset.image_size == (160, 160)


@pytest.mark.parametrize("bad", [[160, 160, 3], [0, 160], [-1, -1], "160x160", [1.5, 2.0]])
def test_invalid_image_size_rejected(minimal_config: dict[str, Any], bad: Any) -> None:
    minimal_config["dataset"]["image_size"] = bad
    config = Config(data=minimal_config)
    with pytest.raises(ConfigError):
        config.validate()


# --------------------------------------------------------------------------
# merging and overrides
# --------------------------------------------------------------------------
def test_deep_merge_is_recursive_and_pure() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"c": 99}, "e": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 1, "c": 99}, "d": 3, "e": 4}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}  # unchanged


def test_deep_merge_replaces_lists_wholesale() -> None:
    assert deep_merge({"x": [1, 2, 3]}, {"x": [9]}) == {"x": [9]}


def test_parse_override_coerces_types() -> None:
    assert parse_override("training.epochs=40") == (["training", "epochs"], 40)
    assert parse_override("runtime.amp=true") == (["runtime", "amp"], True)
    assert parse_override("training.lr=0.002") == (["training", "lr"], 0.002)
    assert parse_override("dataset.scope=rice10") == (["dataset", "scope"], "rice10")


def test_parse_override_rejects_malformed() -> None:
    with pytest.raises(ConfigError):
        parse_override("no-equals-sign")
    with pytest.raises(ConfigError):
        parse_override("=value")


def test_env_overrides_build_nested_mapping() -> None:
    result = env_overrides(
        {"FPA__DATASET__SCOPE": "full102", "FPA__TRAINING__EPOCHS": "40", "OTHER": "x"}
    )
    assert result == {"dataset": {"scope": "full102"}, "training": {"epochs": 40}}


def test_env_override_applies_to_loaded_config(
    write_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config("c.yaml", {"dataset": {"scope": "rice10"}, "paths": {}})
    monkeypatch.setenv("FPA__DATASET__SCOPE", "full102")
    assert load_config(path).num_classes == 102


def test_cli_override_beats_environment(
    write_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config("c.yaml", {"dataset": {"scope": "rice10"}, "paths": {}})
    monkeypatch.setenv("FPA__DATASET__SCOPE", "full102")
    config = load_config(path, cli_overrides=["dataset.scope=rice10"])
    assert config.dataset.scope_name == "rice10"
    assert config.num_classes == 10


def test_env_override_can_still_conflict(
    write_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scope switched by environment must not silently keep a stale count."""
    path = write_config(
        "c.yaml", {"dataset": {"scope": "rice10", "num_classes": 10}, "paths": {}}
    )
    monkeypatch.setenv("FPA__DATASET__SCOPE", "full102")
    with pytest.raises(ConfigError, match="contradicts dataset.scope"):
        load_config(path)


# --------------------------------------------------------------------------
# extends
# --------------------------------------------------------------------------
def test_extends_merges_parent(write_config) -> None:
    write_config("parent.yaml", {"dataset": {"scope": "rice10"}, "paths": {}, "x": 1})
    child = write_config("child.yaml", {"extends": "parent.yaml", "x": 2})
    config = load_config(child)
    assert config.dataset.scope_name == "rice10"
    assert config.get("x") == 2


def test_extends_detects_cycle(write_config) -> None:
    a = write_config("a.yaml", {"extends": "b.yaml"})
    write_config("b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ConfigError, match="circular"):
        load_config(a)


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("does-not-exist.yaml")


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def test_relative_paths_anchor_to_project_root(
    minimal_config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPA_PROJECT_ROOT", str(tmp_path))
    paths = Config(data=minimal_config).paths
    assert paths.project_root == tmp_path.resolve()
    assert paths.dataset_root == (tmp_path / "ip102_v1.1").resolve()
    assert paths.artifacts_dir == (tmp_path / "artifacts").resolve()
    assert paths.checkpoints_dir == (tmp_path / "artifacts" / "checkpoints").resolve()


def test_absolute_paths_are_preserved(
    minimal_config: dict[str, Any], tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    minimal_config["paths"]["dataset_root"] = str(elsewhere)
    assert Config(data=minimal_config).paths.dataset_root == elsewhere.resolve()


def test_writable_dirs_exclude_dataset(
    minimal_config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source data must never appear among directories the project writes to."""
    monkeypatch.setenv("FPA_PROJECT_ROOT", str(tmp_path))
    paths = Config(data=minimal_config).paths
    writable = set(paths.writable_dirs())
    assert paths.dataset_root not in writable
    assert paths.images_dir not in writable
    assert paths.classification_root not in writable


def test_ensure_writable_dirs_creates_them(
    minimal_config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPA_PROJECT_ROOT", str(tmp_path))
    paths = Config(data=minimal_config).paths
    paths.ensure_writable_dirs()
    for directory in paths.writable_dirs():
        assert directory.is_dir()
    assert not paths.dataset_root.exists()


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------
def test_dotted_get_and_require(minimal_config: dict[str, Any]) -> None:
    config = Config(data=minimal_config)
    assert config.get("dataset.scope") == "rice10"
    assert config.get("missing.key", "fallback") == "fallback"
    assert config.require("dataset.scope") == "rice10"
    with pytest.raises(ConfigError, match="required configuration key"):
        config.require("missing.key")


def test_seed_validation(minimal_config: dict[str, Any]) -> None:
    assert Config(data=minimal_config).seed == 1337
    minimal_config["reproducibility"]["seed"] = "abc"
    config = Config(data=minimal_config)
    with pytest.raises(ConfigError, match="must be an integer"):
        _ = config.seed


def test_serialisation_roundtrip(minimal_config: dict[str, Any]) -> None:
    import json

    import yaml

    config = Config(data=minimal_config)
    assert yaml.safe_load(config.to_yaml())["dataset"]["scope"] == "rice10"
    assert json.loads(config.to_json())["dataset"]["scope"] == "rice10"


def test_section_returns_copy(minimal_config: dict[str, Any]) -> None:
    config = Config(data=minimal_config)
    section = config.section("dataset")
    section["scope"] = "full102"
    assert config.dataset.scope_name == "rice10"
