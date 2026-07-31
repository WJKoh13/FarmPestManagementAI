"""Configuration loading, layering and validation.

Configuration is resolved from three layers, lowest precedence first:

1. YAML files, composed via an optional top-level ``extends`` key.
2. Environment variables prefixed with ``FPA__`` (double underscore separates
   nesting levels, e.g. ``FPA__DATASET__SCOPE=full102``).
3. Explicit overrides passed programmatically or from a CLI ``--set`` flag.

No developer-specific absolute path is ever baked into the package: dataset and
artifact locations come from YAML, environment variables or CLI arguments, and
relative paths are anchored to the project root.

The resolved configuration is a plain nested mapping wrapped in :class:`Config`,
which exposes dotted-key access plus a validated, typed view of the sections the
project depends on.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import yaml

from .scopes import ScopeSpec, num_classes_for, resolve_scope, scope_names

__all__ = [
    "ENV_PREFIX",
    "ENV_NESTING_SEPARATOR",
    "ConfigError",
    "Config",
    "DatasetConfig",
    "PathsConfig",
    "load_config",
    "project_root",
    "deep_merge",
    "parse_override",
]

#: Prefix identifying environment variables that override configuration.
ENV_PREFIX = "FPA__"

#: Separator used inside environment variable names to express nesting.
ENV_NESTING_SEPARATOR = "__"

#: Environment variable that, when set, overrides the detected project root.
ENV_PROJECT_ROOT = "FPA_PROJECT_ROOT"

#: Maximum number of ``extends`` hops, to catch accidental cycles cheaply.
_MAX_EXTENDS_DEPTH = 16


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed or self-inconsistent."""


def project_root() -> Path:
    """Locate the project root directory.

    Uses ``FPA_PROJECT_ROOT`` when set, otherwise walks upward from this file
    until a directory containing both ``configs`` and ``src`` is found. Falls
    back to the third parent of this module (``src/farm_pest_ai/config.py`` ->
    repository root) so the package still imports from unusual layouts.
    """
    env = os.environ.get(ENV_PROJECT_ROOT)
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    return here.parents[2]


def deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either.

    Nested mappings are merged key by key; every other value, including lists,
    is replaced wholesale.
    """
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce_scalar(raw: str) -> Any:
    """Interpret an environment/CLI string using YAML scalar rules.

    Produces ``bool``, ``int``, ``float``, ``None`` or ``str`` as appropriate so
    that ``FPA__TRAINING__EPOCHS=40`` yields an integer rather than a string.
    """
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    # A bare string such as "rice10" round-trips fine; anything unparseable
    # (e.g. "{unbalanced") falls back to the literal text.
    return raw if value is None and raw.strip().lower() not in {"null", "~", ""} else value


def _assign(target: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    """Assign ``value`` at the nested ``path`` inside ``target``."""
    cursor: MutableMapping[str, Any] = target
    for part in path[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[path[-1]] = value


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse a ``dotted.key=value`` override string.

    Args:
        item: For example ``"training.epochs=40"``.

    Returns:
        The key path and the coerced value.

    Raises:
        ConfigError: If the string has no ``=`` or an empty key.
    """
    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise ConfigError(
            f"invalid override {item!r}; expected the form dotted.key=value"
        )
    path = [p for p in key.strip().split(".") if p]
    if not path:
        raise ConfigError(f"invalid override key in {item!r}")
    return path, _coerce_scalar(raw)


def env_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Collect configuration overrides from the environment.

    ``FPA__DATASET__SCOPE=full102`` becomes ``{"dataset": {"scope": "full102"}}``.
    Variable name casing is normalised to lower case.
    """
    env = os.environ if environ is None else environ
    out: dict[str, Any] = {}
    for name, raw in env.items():
        if not name.startswith(ENV_PREFIX) or name == ENV_PROJECT_ROOT:
            continue
        remainder = name[len(ENV_PREFIX) :]
        if not remainder:
            continue
        path = [p.lower() for p in remainder.split(ENV_NESTING_SEPARATOR) if p]
        if not path:
            continue
        _assign(out, path, _coerce_scalar(raw))
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path``."""
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return dict(data)


def _load_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML file, resolving its ``extends`` chain depth-first."""
    resolved = path.resolve()
    if resolved in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, resolved))
        raise ConfigError(f"circular 'extends' in configuration: {chain}")
    if len(_seen) >= _MAX_EXTENDS_DEPTH:
        raise ConfigError(f"'extends' chain deeper than {_MAX_EXTENDS_DEPTH} files")

    data = _read_yaml(resolved)
    parents = data.pop("extends", None)
    if parents is None:
        return data

    if isinstance(parents, (str, Path)):
        parents = [parents]
    if not isinstance(parents, Sequence):
        raise ConfigError(f"{resolved}: 'extends' must be a string or list of strings")

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(str(parent)).expanduser()
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        merged = deep_merge(merged, _load_with_extends(parent_path, (*_seen, resolved)))
    return deep_merge(merged, data)


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations, all resolved to absolute paths."""

    project_root: Path
    dataset_root: Path
    classification_root: Path
    images_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    metrics_dir: Path
    plots_dir: Path
    predictions_dir: Path
    exports_dir: Path
    model_registry_dir: Path
    knowledge_dir: Path
    reports_dir: Path
    manual_evaluation_dir: Path

    def writable_dirs(self) -> tuple[Path, ...]:
        """Directories the project may create and write into.

        Deliberately excludes every dataset path: source data is read-only.
        """
        return (
            self.processed_dir,
            self.artifacts_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.metrics_dir,
            self.plots_dir,
            self.predictions_dir,
            self.exports_dir,
            self.model_registry_dir,
            self.knowledge_dir,
            self.reports_dir,
            self.manual_evaluation_dir,
        )

    def ensure_writable_dirs(self) -> None:
        """Create the writable directories if they do not yet exist."""
        for directory in self.writable_dirs():
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    """Validated dataset section, including the resolved scope."""

    scope: ScopeSpec
    manifest_version: str
    preprocessing_version: str
    image_size: tuple[int, int]

    @property
    def scope_name(self) -> str:
        """Name of the active scope."""
        return self.scope.name

    @property
    def num_classes(self) -> int:
        """Number of output classes, always derived from the scope."""
        return self.scope.num_classes


@dataclass
class Config:
    """A resolved configuration.

    Wraps the merged mapping and exposes validated views of the sections the
    project relies on. Unknown sections are preserved untouched so that phases
    added later can read their own keys without changing this class.
    """

    data: dict[str, Any]
    sources: tuple[Path, ...] = field(default_factory=tuple)

    # -- generic access -------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Read a value by dotted key, returning ``default`` when absent."""
        cursor: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def require(self, dotted_key: str) -> Any:
        """Read a value by dotted key, raising when it is missing."""
        sentinel = object()
        value = self.get(dotted_key, sentinel)
        if value is sentinel:
            raise ConfigError(f"required configuration key is missing: {dotted_key}")
        return value

    def section(self, name: str) -> dict[str, Any]:
        """Return a copy of a top-level section, or an empty mapping."""
        value = self.data.get(name, {})
        if not isinstance(value, Mapping):
            raise ConfigError(f"configuration section {name!r} must be a mapping")
        return copy.deepcopy(dict(value))

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the merged configuration mapping."""
        return copy.deepcopy(self.data)

    def to_yaml(self) -> str:
        """Serialise the resolved configuration as YAML."""
        return yaml.safe_dump(self.data, sort_keys=True, allow_unicode=True)

    def to_json(self) -> str:
        """Serialise the resolved configuration as indented JSON."""
        return json.dumps(self.data, indent=2, sort_keys=True, default=str)

    # -- validated views ------------------------------------------------
    @property
    def scope(self) -> ScopeSpec:
        """The active dataset scope."""
        return self.dataset.scope

    @property
    def num_classes(self) -> int:
        """Number of output classes derived from the active scope."""
        return self.dataset.num_classes

    @property
    def dataset(self) -> DatasetConfig:
        """Validated dataset configuration.

        Raises:
            ConfigError: If the scope is missing or unknown, if ``num_classes``
                is stated but contradicts the scope, or if ``image_size`` is
                malformed.
        """
        section = self.data.get("dataset")
        if not isinstance(section, Mapping):
            raise ConfigError("configuration is missing a 'dataset' section")

        raw_scope = section.get("scope")
        if raw_scope is None:
            raise ConfigError(
                "dataset.scope is required; expected one of " f"{list(scope_names())}"
            )
        try:
            scope = resolve_scope(str(raw_scope))
        except ValueError as exc:
            raise ConfigError(str(exc)) from None

        # num_classes is derived, never authoritative. If a config states it,
        # it must agree with the scope; a mismatch is a hard error rather than
        # a silently ignored value.
        stated = section.get("num_classes")
        if stated is not None:
            if isinstance(stated, bool) or not isinstance(stated, int):
                raise ConfigError(
                    f"dataset.num_classes must be an integer, got {stated!r}"
                )
            expected = num_classes_for(scope)
            if stated != expected:
                raise ConfigError(
                    f"dataset.num_classes={stated} contradicts dataset.scope="
                    f"{scope.name!r}, which defines {expected} classes; remove the "
                    "key or correct it, as num_classes is always derived from the scope"
                )

        size = section.get("image_size", [160, 160])
        if isinstance(size, int) and not isinstance(size, bool):
            image_size = (size, size)
        elif isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
            values = list(size)
            if len(values) != 2 or not all(
                isinstance(v, int) and not isinstance(v, bool) for v in values
            ):
                raise ConfigError(
                    f"dataset.image_size must be an int or two ints, got {size!r}"
                )
            image_size = (int(values[0]), int(values[1]))
        else:
            raise ConfigError(
                f"dataset.image_size must be an int or two ints, got {size!r}"
            )
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ConfigError(f"dataset.image_size must be positive, got {image_size}")

        return DatasetConfig(
            scope=scope,
            manifest_version=str(section.get("manifest_version", "1.0.0")),
            preprocessing_version=str(section.get("preprocessing_version", "1.0.0")),
            image_size=image_size,
        )

    @property
    def paths(self) -> PathsConfig:
        """Validated, absolute filesystem paths.

        Relative paths in the ``paths`` section are anchored to the project
        root so that the same configuration works on Windows and inside a
        Linux container.
        """
        section = self.data.get("paths")
        if not isinstance(section, Mapping):
            raise ConfigError("configuration is missing a 'paths' section")

        root_raw = section.get("project_root")
        root = (
            Path(str(root_raw)).expanduser().resolve()
            if root_raw
            else project_root()
        )

        def resolve(key: str, default: str) -> Path:
            raw = section.get(key, default)
            path = Path(str(raw)).expanduser()
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        dataset_root = resolve("dataset_root", "ip102_v1.1")
        classification_root = resolve(
            "classification_root", str(dataset_root / "Classification")
        )
        artifacts_dir = resolve("artifacts_dir", "artifacts")
        processed_dir = resolve("processed_dir", "data/processed")

        def under_artifacts(key: str, name: str) -> Path:
            raw = section.get(key)
            if raw is None:
                return artifacts_dir / name
            path = Path(str(raw)).expanduser()
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        return PathsConfig(
            project_root=root,
            dataset_root=dataset_root,
            classification_root=classification_root,
            images_dir=resolve("images_dir", str(classification_root / "images")),
            processed_dir=processed_dir,
            artifacts_dir=artifacts_dir,
            checkpoints_dir=under_artifacts("checkpoints_dir", "checkpoints"),
            logs_dir=under_artifacts("logs_dir", "logs"),
            metrics_dir=under_artifacts("metrics_dir", "metrics"),
            plots_dir=under_artifacts("plots_dir", "plots"),
            predictions_dir=under_artifacts("predictions_dir", "predictions"),
            exports_dir=under_artifacts("exports_dir", "exports"),
            model_registry_dir=under_artifacts("model_registry_dir", "model_registry"),
            knowledge_dir=resolve("knowledge_dir", "data/knowledge"),
            reports_dir=resolve("reports_dir", "data/reports"),
            manual_evaluation_dir=resolve(
                "manual_evaluation_dir", "data/manual_evaluation"
            ),
        )

    @property
    def seed(self) -> int:
        """The configured global random seed."""
        value = self.get("reproducibility.seed", 1337)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"reproducibility.seed must be an integer, got {value!r}")
        return value

    def validate(self) -> Config:
        """Eagerly validate every known section.

        Returns:
            ``self``, so the call can be chained after :func:`load_config`.

        Raises:
            ConfigError: On the first inconsistency found.
        """
        _ = self.dataset
        _ = self.paths
        _ = self.seed
        return self


def load_config(
    paths: Path | str | Iterable[Path | str] | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    cli_overrides: Sequence[str] | None = None,
    use_env: bool = True,
    environ: Mapping[str, str] | None = None,
    validate: bool = True,
) -> Config:
    """Load and merge configuration from files, environment and overrides.

    Args:
        paths: One or more YAML files, merged left to right. Relative paths are
            resolved against the current directory, then the project root, then
            the project ``configs`` directory, so ``load_config("base.yaml")``
            works from anywhere.
        overrides: A nested mapping applied after the environment.
        cli_overrides: ``dotted.key=value`` strings applied last.
        use_env: Whether to apply ``FPA__``-prefixed environment variables.
        environ: Environment mapping to read instead of :data:`os.environ`.
        validate: Whether to validate known sections before returning.

    Returns:
        The resolved :class:`Config`.

    Raises:
        ConfigError: If a file is missing, malformed, or the result is invalid.
    """
    if paths is None:
        candidates: list[Path | str] = []
    elif isinstance(paths, (str, Path)):
        candidates = [paths]
    else:
        candidates = list(paths)

    root = project_root()
    merged: dict[str, Any] = {}
    used: list[Path] = []
    for candidate in candidates:
        path = Path(str(candidate)).expanduser()
        if not path.is_absolute():
            for base in (Path.cwd(), root, root / "configs"):
                trial = base / path
                if trial.is_file():
                    path = trial
                    break
        merged = deep_merge(merged, _load_with_extends(path))
        used.append(path.resolve())

    if use_env:
        merged = deep_merge(merged, env_overrides(environ))
    if overrides:
        merged = deep_merge(merged, overrides)
    if cli_overrides:
        applied: dict[str, Any] = {}
        for item in cli_overrides:
            key_path, value = parse_override(item)
            _assign(applied, key_path, value)
        merged = deep_merge(merged, applied)

    config = Config(data=merged, sources=tuple(used))
    return config.validate() if validate else config
