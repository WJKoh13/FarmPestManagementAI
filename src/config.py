"""Config loading. A model YAML inherits everything from configs/_base.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path, overrides: dict | None = None) -> dict:
    """Load a YAML config, resolving a single ``extends:`` parent.

    Keys in the child override the parent. Paths in the returned dict stay
    relative; use ``resolve_path`` to make them absolute.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    parent_name = config.pop("extends", None)
    if parent_name:
        parent = load_config(config_path.parent / parent_name)
        parent.pop("_config_path", None)
        parent.update(config)
        config = parent

    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})

    config["_config_path"] = str(config_path)
    return config


def resolve_path(path: str | Path) -> Path:
    """Interpret a config path relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
