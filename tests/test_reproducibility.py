"""Tests for seeding and environment capture."""

from __future__ import annotations

import random

import pytest

from farm_pest_ai.reproducibility import (
    DEFAULT_SEED,
    derive_seed,
    environment_snapshot,
    seed_everything,
    worker_init_fn,
)


def test_seed_everything_reports_state() -> None:
    state = seed_everything(42)
    assert state.seed == 42
    assert state.deterministic is True
    assert "python" in state.seeded


def test_seeding_makes_python_rng_reproducible() -> None:
    seed_everything(123)
    first = [random.random() for _ in range(5)]
    seed_everything(123)
    assert [random.random() for _ in range(5)] == first


def test_different_seeds_diverge() -> None:
    seed_everything(1)
    first = [random.random() for _ in range(5)]
    seed_everything(2)
    assert [random.random() for _ in range(5)] != first


def test_seed_everything_rejects_non_integer() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        seed_everything("abc")  # type: ignore[arg-type]


def test_derive_seed_is_deterministic_and_distinct() -> None:
    assert derive_seed(1337, "loader", 0) == derive_seed(1337, "loader", 0)
    assert derive_seed(1337, "loader", 0) != derive_seed(1337, "loader", 1)
    assert derive_seed(1337, "loader", 0) != derive_seed(1338, "loader", 0)


def test_derive_seed_is_in_uint32_range() -> None:
    for worker in range(16):
        assert 0 <= derive_seed(DEFAULT_SEED, "loader_worker", worker) < 2**32


def test_worker_init_is_reproducible_per_worker() -> None:
    worker_init_fn(3, base_seed=99)
    first = [random.random() for _ in range(3)]
    worker_init_fn(3, base_seed=99)
    assert [random.random() for _ in range(3)] == first

    worker_init_fn(4, base_seed=99)
    assert [random.random() for _ in range(3)] != first


def test_environment_snapshot_has_expected_keys() -> None:
    snapshot = environment_snapshot(include_git=False)
    for key in ("python_version", "platform", "cpu_count", "torch_version", "cuda_available"):
        assert key in snapshot
    assert isinstance(snapshot["cuda_available"], bool)


def test_environment_snapshot_survives_missing_torch() -> None:
    """The harness must describe the environment before Phase 3 installs torch."""
    snapshot = environment_snapshot(include_git=False)
    if snapshot["torch_version"] is None:
        assert snapshot["cuda_available"] is False
        assert snapshot["gpu_count"] == 0
