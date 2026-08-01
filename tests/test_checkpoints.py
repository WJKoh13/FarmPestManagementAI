"""Tests for checkpoint provenance enforcement.

The central property: a checkpoint carries the scope, class-mapping version and
preprocessing fingerprint it was trained under, and loading it under different
ones **raises**. A silently mismatched checkpoint does not crash — it produces
confident, wrong pest identifications, which is the worst failure this project
can have.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from farm_pest_ai.scopes import CLASS_MAPPING_VERSION, FULL102, RICE10  # noqa: E402
from farm_pest_ai.vision.checkpoints import (  # noqa: E402
    CheckpointError,
    CheckpointMetadata,
    best_checkpoint_path,
    capture_rng_state,
    last_checkpoint_path,
    load_checkpoint,
    load_model_for_inference,
    read_metadata,
    restore_rng_state,
    save_checkpoint,
    write_metadata_sidecar,
)
from farm_pest_ai.vision.models import ModelConfig, build_model  # noqa: E402


@pytest.fixture()
def rice10_model() -> torch.nn.Module:
    """A small rice10 model, kept narrow so tests stay fast."""
    return build_model(
        ModelConfig(
            name="custom_cnn",
            num_classes=10,
            stem_channels=8,
            stage_channels=(8, 16),
            stage_blocks=(1, 1),
            stage_strides=(2, 2),
        )
    )


def _metadata(scope=RICE10, **overrides) -> CheckpointMetadata:
    """Build metadata for a checkpoint under ``scope``."""
    defaults = {
        "scope": scope.name,
        "num_classes": scope.num_classes,
        "preprocessing_fingerprint": "abc123",
        "model": ModelConfig(
            name="custom_cnn",
            num_classes=scope.num_classes,
            stem_channels=8,
            stage_channels=(8, 16),
            stage_blocks=(1, 1),
            stage_strides=(2, 2),
        ).to_dict(),
        "epoch": 3,
        "global_step": 120,
    }
    defaults.update(overrides)
    return CheckpointMetadata(**defaults)


# -- round-trip ---------------------------------------------------------


def test_checkpoint_roundtrips_weights_exactly(
    tmp_path: Path, rice10_model: torch.nn.Module
) -> None:
    path = save_checkpoint(tmp_path / "best.pt", rice10_model, _metadata())
    reloaded, _, _ = load_checkpoint(path, scope=RICE10)

    original_state = rice10_model.state_dict()
    for key, value in reloaded.state_dict().items():
        assert torch.equal(value, original_state[key])


def test_checkpoint_records_provenance(
    tmp_path: Path, rice10_model: torch.nn.Module
) -> None:
    path = save_checkpoint(tmp_path / "best.pt", rice10_model, _metadata())
    metadata = read_metadata(path)
    assert metadata.scope == "rice10"
    assert metadata.num_classes == 10
    assert metadata.class_mapping_version == CLASS_MAPPING_VERSION
    assert metadata.preprocessing_fingerprint == "abc123"
    assert metadata.epoch == 3
    assert metadata.global_step == 120


def test_checkpoint_rebuilds_architecture_without_a_template(
    tmp_path: Path, rice10_model: torch.nn.Module
) -> None:
    """A checkpoint alone is enough to reconstruct its model."""
    path = save_checkpoint(tmp_path / "best.pt", rice10_model, _metadata())
    reloaded, metadata, _ = load_checkpoint(path, scope=RICE10)
    assert reloaded.num_classes == metadata.num_classes == 10


def test_checkpoint_write_is_atomic(
    tmp_path: Path, rice10_model: torch.nn.Module
) -> None:
    """No temporary file may survive a successful write."""
    save_checkpoint(tmp_path / "best.pt", rice10_model, _metadata())
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == []


def test_checkpoint_preserves_optimizer_and_scheduler_state(
    tmp_path: Path, rice10_model: torch.nn.Module
) -> None:
    optimizer = torch.optim.AdamW(rice10_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.5)
    path = save_checkpoint(
        tmp_path / "last.pt",
        rice10_model,
        _metadata(),
        optimizer=optimizer,
        scheduler=scheduler,
    )
    _, _, extras = load_checkpoint(path, scope=RICE10)
    assert "optimizer_state" in extras
    assert "scheduler_state" in extras


# -- the core safety property -------------------------------------------


def test_loading_under_the_wrong_scope_raises(
    tmp_path: Path, rice10_model: torch.nn.Module
) -> None:
    """A rice10 checkpoint must never load as full102."""
    path = save_checkpoint(tmp_path / "best.pt", rice10_model, _metadata())
    with pytest.raises(CheckpointError, match="never be used under a different scope"):
        load_checkpoint(path, scope=FULL102)


def test_metadata_verification_rejects_class_count_mismatch() -> None:
    """Scope name right, class count wrong: still refused."""
    metadata = _metadata(num_classes=7)
    with pytest.raises(CheckpointError, match="7 output classes"):
        metadata.verify_against(RICE10)


def test_metadata_verification_rejects_stale_class_mapping() -> None:
    """A superseded mapping means project labels no longer mean the same thing."""
    metadata = _metadata(class_mapping_version="0.9.0")
    with pytest.raises(CheckpointError, match="class mapping"):
        metadata.verify_against(RICE10)


def test_metadata_verification_rejects_manifest_mismatch() -> None:
    metadata = _metadata(manifest_version="1.0.0")
    with pytest.raises(CheckpointError, match="manifest version"):
        metadata.verify_against(RICE10, manifest_version="2.0.0")


def test_strict_preprocessing_rejects_a_changed_pipeline() -> None:
    """Serving a model under preprocessing it was not trained on is silent damage."""
    metadata = _metadata(preprocessing_fingerprint="aaaa")
    with pytest.raises(CheckpointError, match="different pixels"):
        metadata.verify_against(
            RICE10, preprocessing_fingerprint="bbbb", strict_preprocessing=True
        )


def test_non_strict_preprocessing_allows_a_changed_pipeline() -> None:
    """Mid-development comparisons may tolerate it; Phase 9 will not."""
    metadata = _metadata(preprocessing_fingerprint="aaaa")
    metadata.verify_against(
        RICE10, preprocessing_fingerprint="bbbb", strict_preprocessing=False
    )


def test_matching_provenance_passes_verification() -> None:
    metadata = _metadata(preprocessing_fingerprint="abc123")
    metadata.verify_against(
        RICE10,
        preprocessing_fingerprint="abc123",
        manifest_version="1.0.0",
        strict_preprocessing=True,
    )


def test_inference_loader_requires_scope_and_is_strict_by_default(
    tmp_path: Path, rice10_model: torch.nn.Module
) -> None:
    path = save_checkpoint(tmp_path / "best.pt", rice10_model, _metadata())
    with pytest.raises(CheckpointError, match="different pixels"):
        load_model_for_inference(path, RICE10, preprocessing_fingerprint="different")


def test_inference_loader_returns_model_in_eval_mode(
    tmp_path: Path, rice10_model: torch.nn.Module
) -> None:
    path = save_checkpoint(tmp_path / "best.pt", rice10_model, _metadata())
    model, metadata = load_model_for_inference(
        path, RICE10, preprocessing_fingerprint="abc123"
    )
    assert not model.training
    assert metadata.scope == "rice10"


def test_loading_into_a_mismatched_model_raises(
    tmp_path: Path, rice10_model: torch.nn.Module
) -> None:
    path = save_checkpoint(tmp_path / "best.pt", rice10_model, _metadata())
    wrong = build_model(ModelConfig(name="custom_cnn", num_classes=102))
    with pytest.raises(CheckpointError, match="output classes"):
        load_checkpoint(path, model=wrong)


# -- malformed input ----------------------------------------------------


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "absent.pt", scope=RICE10)


def test_a_file_without_metadata_is_not_a_project_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "foreign.pt"
    torch.save({"model_state": {}}, path)
    with pytest.raises(CheckpointError, match="carries no metadata"):
        load_checkpoint(path, scope=RICE10)


def test_metadata_without_scope_is_rejected() -> None:
    with pytest.raises(CheckpointError, match="missing 'scope'"):
        CheckpointMetadata.from_dict({"num_classes": 10})


def test_metadata_without_num_classes_is_rejected() -> None:
    with pytest.raises(CheckpointError, match="missing 'num_classes'"):
        CheckpointMetadata.from_dict({"scope": "rice10"})


def test_internally_inconsistent_metadata_is_rejected() -> None:
    """The metadata and the model section must agree on the class count."""
    metadata = _metadata()
    metadata.model["num_classes"] = 99
    with pytest.raises(CheckpointError, match="internally inconsistent"):
        metadata.model_config()


def test_unknown_model_field_is_rejected() -> None:
    """A checkpoint from an incompatible version must not be half-loaded."""
    metadata = _metadata()
    metadata.model["mystery_option"] = True
    with pytest.raises(CheckpointError, match="unknown field"):
        metadata.model_config()


def test_metadata_without_model_section_cannot_rebuild() -> None:
    with pytest.raises(CheckpointError, match="carries no model configuration"):
        _metadata(model={}).model_config()


# -- sidecar ------------------------------------------------------------


def test_sidecar_is_written_as_readable_json(tmp_path: Path) -> None:
    import json

    path = write_metadata_sidecar(tmp_path / "best.json", _metadata())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["scope"] == "rice10"
    assert payload["num_classes"] == 10


def test_sidecar_is_not_the_authority(
    tmp_path: Path, rice10_model: torch.nn.Module
) -> None:
    """Deleting or editing a sidecar cannot make a bad checkpoint loadable."""
    path = save_checkpoint(tmp_path / "best.pt", rice10_model, _metadata())
    sidecar = write_metadata_sidecar(tmp_path / "best.json", _metadata())
    sidecar.write_text('{"scope": "full102", "num_classes": 102}', encoding="utf-8")
    # The embedded metadata still governs.
    with pytest.raises(CheckpointError, match="never be used under a different scope"):
        load_checkpoint(path, scope=FULL102)


# -- RNG state ----------------------------------------------------------


def test_rng_state_roundtrip_reproduces_the_same_draws() -> None:
    torch.manual_seed(0)
    state = capture_rng_state()
    first = torch.rand(5)
    restore_rng_state(state)
    assert torch.equal(torch.rand(5), first)


def test_restore_rng_state_tolerates_missing_entries() -> None:
    """A CUDA-written checkpoint resumed on CPU must still restore CPU streams."""
    torch.manual_seed(0)
    state = {"torch": torch.get_rng_state()}
    restore_rng_state(state)


# -- paths --------------------------------------------------------------


def test_checkpoint_paths_are_named_consistently(tmp_path: Path) -> None:
    assert best_checkpoint_path(tmp_path).name == "best.pt"
    assert last_checkpoint_path(tmp_path).name == "last.pt"


def test_smoke_flag_is_preserved(tmp_path: Path, rice10_model: torch.nn.Module) -> None:
    """A smoke checkpoint must be identifiable as one."""
    path = save_checkpoint(tmp_path / "best.pt", rice10_model, _metadata(smoke=True))
    assert read_metadata(path).smoke is True
