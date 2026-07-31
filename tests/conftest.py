"""Shared pytest fixtures and import bootstrapping.

Adds ``src`` to ``sys.path`` so the suite runs from a bare checkout before the
package is installed in Phase 3.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def project_root() -> Path:
    """The repository root."""
    return PROJECT_ROOT


@pytest.fixture()
def configs_dir() -> Path:
    """The shipped ``configs`` directory."""
    return PROJECT_ROOT / "configs"


@pytest.fixture()
def minimal_config() -> dict[str, Any]:
    """A minimal valid configuration mapping."""
    return {
        "dataset": {"scope": "rice10", "image_size": [160, 160]},
        "paths": {"dataset_root": "ip102_v1.1"},
        "reproducibility": {"seed": 1337},
    }


@pytest.fixture()
def write_config(tmp_path: Path):
    """Return a helper that writes a YAML file into ``tmp_path``."""
    import yaml

    def _write(name: str, data: dict[str, Any]) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write


@pytest.fixture(autouse=True)
def _clear_fpa_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent ambient ``FPA__*`` variables from leaking into tests."""
    import os

    for key in list(os.environ):
        if key.startswith("FPA__") or key == "FPA_PROJECT_ROOT":
            monkeypatch.delenv(key, raising=False)
