"""Tests for dataset scope definitions and label mapping."""

from __future__ import annotations

import pytest

from farm_pest_ai.scopes import (
    CLASS_MAPPING_VERSION,
    FULL102,
    RICE10,
    SCOPES,
    ScopeSpec,
    get_scope,
    is_valid_scope,
    num_classes_for,
    resolve_scope,
    scope_names,
)

#: The rice10 mapping verified against classes.txt in Phase 1. Encoded here so
#: an accidental reordering of the scope definition fails loudly.
EXPECTED_RICE10 = {
    0: (0, "rice leaf roller"),
    1: (1, "rice leaf caterpillar"),
    2: (3, "asiatic rice borer"),
    3: (4, "yellow rice borer"),
    4: (5, "rice gall midge"),
    5: (7, "brown plant hopper"),
    6: (8, "white backed plant hopper"),
    7: (9, "small brown plant hopper"),
    8: (10, "rice water weevil"),
    9: (11, "rice leafhopper"),
}


def test_supported_scopes() -> None:
    assert set(scope_names()) == {"rice10", "full102"}


def test_num_classes_are_derived() -> None:
    assert num_classes_for("rice10") == 10
    assert num_classes_for("full102") == 102
    assert RICE10.num_classes == 10
    assert FULL102.num_classes == 102


def test_num_classes_matches_mapping_length() -> None:
    for spec in SCOPES.values():
        assert spec.num_classes == len(spec.original_labels)
        assert spec.num_classes == len(spec.project_to_original)
        assert spec.num_classes == len(spec.original_to_project)


@pytest.mark.parametrize(("project", "expected"), sorted(EXPECTED_RICE10.items()))
def test_rice10_mapping_is_exact(project: int, expected: tuple[int, str]) -> None:
    """Project labels must map onto the IP102 labels verified in Phase 1."""
    original, _name = expected
    assert RICE10.to_original_label(project) == original
    assert RICE10.to_project_label(original) == project


def test_rice10_project_labels_are_contiguous() -> None:
    assert sorted(RICE10.project_to_original) == list(range(10))


def test_rice10_excludes_non_rice_labels() -> None:
    """IP102 labels 2 and 6 sit inside the rice range but are not in scope."""
    assert not RICE10.includes_original(2)
    assert not RICE10.includes_original(6)
    assert RICE10.includes_original(11)


def test_full102_is_identity_mapping() -> None:
    assert FULL102.is_identity
    assert not RICE10.is_identity
    for label in range(102):
        assert FULL102.to_project_label(label) == label
        assert FULL102.to_original_label(label) == label


def test_out_of_scope_label_raises() -> None:
    with pytest.raises(KeyError, match="not part of scope"):
        RICE10.to_project_label(50)
    with pytest.raises(KeyError, match="out of range"):
        RICE10.to_original_label(10)
    with pytest.raises(KeyError):
        FULL102.to_project_label(102)


def test_unknown_scope_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dataset scope"):
        get_scope("rice20")
    assert not is_valid_scope("rice20")
    assert not is_valid_scope(None)
    assert is_valid_scope("full102")


def test_resolve_scope_accepts_spec_or_name() -> None:
    assert resolve_scope("rice10") is RICE10
    assert resolve_scope(RICE10) is RICE10


def test_scopes_are_immutable() -> None:
    with pytest.raises(AttributeError):
        RICE10.name = "other"  # type: ignore[misc]


def test_validate_rejects_duplicate_labels() -> None:
    bad = ScopeSpec("bad", "duplicate labels", (0, 1, 1))
    with pytest.raises(ValueError, match="duplicate"):
        bad.validate()


def test_validate_rejects_out_of_range_labels() -> None:
    bad = ScopeSpec("bad", "out of range", (0, 102))
    with pytest.raises(ValueError, match="outside"):
        bad.validate()


def test_validate_rejects_empty_scope() -> None:
    with pytest.raises(ValueError, match="no classes"):
        ScopeSpec("bad", "empty", ()).validate()


def test_class_mapping_version_is_set() -> None:
    assert CLASS_MAPPING_VERSION
