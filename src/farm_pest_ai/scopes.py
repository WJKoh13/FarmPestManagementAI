"""Dataset scope definitions.

This module is the single source of truth for classification scopes. Nothing
else in the project may hard-code the number of classes or the class mapping:
loaders, models, losses, metrics, checkpoints, API schemas and the frontend all
derive them from here.

Two scopes are supported:

``rice10``
    A ten-class rice-pest subset of IP102, used for rapid development and as a
    focused deployment option. Project labels 0-9 map onto a fixed, ordered
    selection of IP102 labels.

``full102``
    The complete IP102 classification task. Project labels are identical to the
    original IP102 labels, 0-101.

The IP102 manifests use zero-based labels while ``classes.txt`` numbers classes
from 1, so ``classes.txt`` line ``id`` describes IP102 label ``id - 1``. That
offset is applied in :mod:`farm_pest_ai.data.manifests`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

__all__ = [
    "CLASS_MAPPING_VERSION",
    "DatasetScope",
    "ScopeSpec",
    "SCOPES",
    "RICE10",
    "FULL102",
    "get_scope",
    "resolve_scope",
    "num_classes_for",
    "is_valid_scope",
    "scope_names",
]

#: Bumped whenever the meaning of a project label changes for any scope.
#: Checkpoints record this so that a model trained under an older mapping is
#: rejected rather than silently misinterpreted.
CLASS_MAPPING_VERSION: Final[str] = "1.0.0"

#: Total number of classes in the IP102 classification task.
IP102_NUM_CLASSES: Final[int] = 102

DatasetScope = str
"""Type alias documenting that a value is a scope name such as ``"rice10"``."""


@dataclass(frozen=True)
class ScopeSpec:
    """An immutable description of a classification scope.

    Attributes:
        name: Scope identifier used in configuration and checkpoints.
        description: Human-readable summary shown in the UI and reports.
        original_labels: IP102 labels included in this scope, in project-label
            order. Index ``i`` of this sequence is project label ``i``.
    """

    name: str
    description: str
    original_labels: tuple[int, ...]

    @property
    def num_classes(self) -> int:
        """Number of output classes, derived from the mapping."""
        return len(self.original_labels)

    @property
    def project_to_original(self) -> Mapping[int, int]:
        """Map project label -> original IP102 label."""
        return {p: o for p, o in enumerate(self.original_labels)}

    @property
    def original_to_project(self) -> Mapping[int, int]:
        """Map original IP102 label -> project label."""
        return {o: p for p, o in enumerate(self.original_labels)}

    @property
    def is_identity(self) -> bool:
        """True when project labels equal original IP102 labels."""
        return self.original_labels == tuple(range(len(self.original_labels)))

    def includes_original(self, original_label: int) -> bool:
        """Return whether ``original_label`` belongs to this scope."""
        return original_label in self.original_to_project

    def to_project_label(self, original_label: int) -> int:
        """Convert an IP102 label to this scope's project label.

        Raises:
            KeyError: If ``original_label`` is not part of this scope.
        """
        try:
            return self.original_to_project[original_label]
        except KeyError:
            raise KeyError(
                f"IP102 label {original_label} is not part of scope "
                f"{self.name!r} (valid: {sorted(self.original_to_project)})"
            ) from None

    def to_original_label(self, project_label: int) -> int:
        """Convert a project label back to its IP102 label.

        Raises:
            KeyError: If ``project_label`` is out of range for this scope.
        """
        try:
            return self.project_to_original[project_label]
        except KeyError:
            raise KeyError(
                f"project label {project_label} is out of range for scope "
                f"{self.name!r} (0..{self.num_classes - 1})"
            ) from None

    def validate(self) -> None:
        """Check internal consistency of the mapping.

        Raises:
            ValueError: If labels are duplicated or outside the IP102 range.
        """
        if not self.original_labels:
            raise ValueError(f"scope {self.name!r} has no classes")
        if len(set(self.original_labels)) != len(self.original_labels):
            raise ValueError(f"scope {self.name!r} has duplicate original labels")
        bad = [x for x in self.original_labels if not 0 <= x < IP102_NUM_CLASSES]
        if bad:
            raise ValueError(
                f"scope {self.name!r} has labels outside 0..{IP102_NUM_CLASSES - 1}: {bad}"
            )


#: Ten-class rice-pest subset. The order defines project labels 0-9 and must
#: never be reordered without bumping ``CLASS_MAPPING_VERSION``.
RICE10: Final[ScopeSpec] = ScopeSpec(
    name="rice10",
    description="Ten-class rice pest subset of IP102",
    original_labels=(0, 1, 3, 4, 5, 7, 8, 9, 10, 11),
)

#: Complete IP102 classification task; project labels equal original labels.
FULL102: Final[ScopeSpec] = ScopeSpec(
    name="full102",
    description="Complete IP102 classification task",
    original_labels=tuple(range(IP102_NUM_CLASSES)),
)

SCOPES: Final[Mapping[str, ScopeSpec]] = {
    RICE10.name: RICE10,
    FULL102.name: FULL102,
}

for _spec in SCOPES.values():
    _spec.validate()
del _spec


def scope_names() -> tuple[str, ...]:
    """Return the supported scope names in a stable order."""
    return tuple(SCOPES)


def is_valid_scope(name: object) -> bool:
    """Return whether ``name`` identifies a known scope."""
    return isinstance(name, str) and name in SCOPES


def get_scope(name: str) -> ScopeSpec:
    """Look up a scope by name.

    Args:
        name: Scope identifier, for example ``"rice10"``.

    Returns:
        The matching :class:`ScopeSpec`.

    Raises:
        ValueError: If ``name`` is not a supported scope.
    """
    try:
        return SCOPES[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset scope {name!r}; expected one of {list(SCOPES)}"
        ) from None


def resolve_scope(scope: str | ScopeSpec) -> ScopeSpec:
    """Accept either a scope name or an already-resolved spec."""
    if isinstance(scope, ScopeSpec):
        return scope
    return get_scope(scope)


def num_classes_for(scope: str | ScopeSpec) -> int:
    """Derive the number of output classes for a scope.

    This is the only sanctioned way to obtain ``num_classes``.
    """
    return resolve_scope(scope).num_classes
