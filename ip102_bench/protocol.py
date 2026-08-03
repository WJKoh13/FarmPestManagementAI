"""Loading the locked protocol.

Everything that must be identical across models lives in ``protocol.yaml``.
Notebooks read it through :func:`load_protocol` and never hardcode a value, so
there is exactly one place to look when someone asks "what settings was this
trained with?".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = PROJECT_ROOT / "protocol.yaml"


def resolve_path(value: str | Path) -> Path:
    """Interpret a relative path as relative to the project root, not the cwd.

    Notebooks run from ``notebooks/``, scripts from the repo root, and Colab from
    somewhere else entirely. Anchoring on the project root makes every config
    path work from all three without change.
    """
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Protocol:
    """The locked settings, parsed and validated."""

    version: int
    dataset: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    runtime: dict[str, Any]
    source_path: Path = field(compare=False, default=DEFAULT_PROTOCOL_PATH)

    # Convenience accessors for the values notebooks touch constantly.
    @property
    def image_size(self) -> int:
        return int(self.training["image_size"])

    @property
    def batch_size(self) -> int:
        return int(self.training["batch_size"])

    @property
    def epochs(self) -> int:
        return int(self.training["epochs"])

    @property
    def seed(self) -> int:
        return int(self.training["seed"])

    @property
    def subset_name(self) -> str:
        return str(self.dataset["subset"])

    @property
    def subset(self) -> dict[str, Any]:
        """The active entry from ``dataset.subsets``."""
        name = self.subset_name
        subsets = self.dataset["subsets"]
        if name not in subsets:
            raise KeyError(
                f"protocol.yaml selects subset '{name}', which is not defined under "
                f"dataset.subsets. Defined: {sorted(subsets)}"
            )
        return subsets[name]

    @property
    def image_root(self) -> Path:
        """Directory the manifest's ``image_path`` values are relative to."""
        return resolve_path(self.subset["images"])

    @property
    def crop_mode(self) -> str:
        return str(self.dataset.get("crop", {}).get("mode", "none"))

    @property
    def crop_margin(self) -> float:
        return float(self.dataset.get("crop", {}).get("margin", 0.0))

    @property
    def manifest_dir(self) -> Path:
        return resolve_path(self.dataset["manifest_dir"])

    def manifest(self, split: str) -> Path:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split '{split}'. Use train, validation or test.")
        return self.manifest_dir / f"{split}.csv"

    @property
    def norm_stats_path(self) -> Path:
        return resolve_path(self.dataset["norm_stats"])

    def norm_stats(self) -> tuple[list[float], list[float]]:
        """RGB (mean, std) measured on the training split only."""
        path = self.norm_stats_path
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/setup_data.py` to build the "
                "manifests and normalization statistics."
            )
        stats = json.loads(path.read_text(encoding="utf-8"))
        return stats["mean"], stats["std"]

    @property
    def class_names(self) -> list[str]:
        """Class names ordered so that ``class_names[i]`` is project label ``i``."""
        meta_path = self.manifest_dir / "classes.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"{meta_path} not found. Run `python scripts/setup_data.py` first."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return [entry["class_name"] for entry in meta["classes"]]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, embedded verbatim into every ``results.json``."""
        return {
            "protocol_version": self.version,
            "dataset": self.dataset,
            "training": self.training,
            "evaluation": self.evaluation,
        }


def load_protocol(path: str | Path | None = None) -> Protocol:
    """Read and validate ``protocol.yaml``."""
    protocol_path = Path(path) if path else DEFAULT_PROTOCOL_PATH
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    raw = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))

    missing = [k for k in ("protocol_version", "dataset", "training", "evaluation")
               if k not in raw]
    if missing:
        raise ValueError(f"{protocol_path} is missing required sections: {missing}")

    protocol = Protocol(
        version=int(raw["protocol_version"]),
        dataset=raw["dataset"],
        training=raw["training"],
        evaluation=raw["evaluation"],
        runtime=raw.get("runtime", {}),
        source_path=protocol_path,
    )
    protocol.subset  # raises now, rather than midway through a training run
    return protocol
