"""``scripts/train.py`` against the real IP102 data and derived manifests.

The unit tests prove each guard fires on a synthetic bundle. These prove the
guards see the real thing: that the loaders the script actually builds cover
every row of the rice10 train and validation manifests, that no test loader is
constructed, and that ``--plan`` resolves a complete plan without writing a
checkpoint or training a single step.

No test here runs a real experiment, and nothing reads the test split.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from farm_pest_ai.config import load_config
from farm_pest_ai.data.loaders import build_loaders

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.dataset


def _load_train_module() -> Any:
    """Import ``scripts/train.py`` by path; scripts is not a package."""
    path = PROJECT_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("_train_script_it", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = _load_train_module()


@pytest.fixture(scope="module")
def experiment_config():
    """The shipped custom-model config, or a skip when data is absent.

    Deliberately not ``smoke_test.yaml``: this exercises the configuration a
    real experiment would use, which is the one whose split coverage matters.
    """
    config = load_config("model_custom.yaml")
    if not config.paths.images_dir.is_dir():
        pytest.skip(f"IP102 images not found at {config.paths.images_dir}")
    if not (config.paths.processed_dir / "rice10" / "train.csv").is_file():
        pytest.skip("derived manifests not built; run scripts/build_manifests.py")
    return config


@pytest.fixture(scope="module")
def real_bundle(experiment_config):
    """Real rice10 train and validation loaders. Never the test split."""
    return build_loaders(experiment_config, train.TRAINING_SPLITS)


def test_the_bundle_carries_no_test_loader(real_bundle) -> None:
    """The property Phase 9 depends on, checked against real loaders."""
    assert "test" not in real_bundle.loaders
    assert "test" not in real_bundle.datasets
    train.assert_no_test_split(real_bundle)


def test_the_run_covers_every_manifest_row(experiment_config, real_bundle) -> None:
    """Both splits are used whole, verified against the manifests on disk.

    The Phase 1 counts (4,318 train and 721 validation for rice10) are asserted
    directly as well, so a manifest that was rebuilt smaller would fail here
    rather than producing a quietly smaller experiment.
    """
    coverage = train.assert_full_splits(experiment_config, real_bundle)
    assert coverage["train"]["images"] == 4318
    assert coverage["validation"]["images"] == 721
    assert set(coverage) == {"train", "validation"}


def test_a_truncated_split_is_refused(experiment_config, real_bundle) -> None:
    """A dataset short of its manifest aborts rather than training on a subset.

    Constructed by substituting a torch ``Subset`` for the real training
    dataset: that is exactly what an accidental slice would look like, and
    without this check it would produce a full-looking run over 100 images.
    """
    from torch.utils.data import Subset

    truncated = type(real_bundle)(
        loaders=dict(real_bundle.loaders),
        datasets={
            **real_bundle.datasets,
            "train": Subset(real_bundle.datasets["train"], list(range(100))),
        },
        preprocessing=real_bundle.preprocessing,
        runtime=real_bundle.runtime,
        scope=real_bundle.scope,
        device=real_bundle.device,
        batch_size=real_bundle.batch_size,
        seed=real_bundle.seed,
    )
    with pytest.raises(train.TrainingRunError, match="entire split"):
        train.assert_full_splits(experiment_config, truncated)


def test_plan_resolves_without_training_or_writing(
    experiment_config, tmp_path: Path, capsys
) -> None:
    """``--plan`` prints a complete plan and leaves no artifact behind.

    Run against a temporary checkpoints directory so a stray write would be
    detectable; the assertion is that the run directory is never created.
    """
    checkpoints = tmp_path / "checkpoints"
    reports = tmp_path / "reports"
    exit_code = train.main(
        [
            "--config",
            "model_custom.yaml",
            "--plan",
            "--run-name",
            "plan_probe",
            "--set",
            f"paths.artifacts_dir={tmp_path.as_posix()}",
            "--set",
            f"paths.checkpoints_dir={checkpoints.as_posix()}",
            "--set",
            f"paths.reports_dir={reports.as_posix()}",
            # Two workers keeps the Windows spawn cost of this test small; it
            # does not affect what the plan resolves.
            "--set",
            "runtime.num_workers=2",
            "--set",
            "runtime.persistent_workers=false",
        ]
    )
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Training plan: scope rice10" in output
    assert "4,318 images" in output
    assert "test        not built" in output
    assert "nothing was trained" in output

    assert not (checkpoints / "plan_probe").exists()


def test_a_smoke_config_is_refused_by_the_real_entry_point(capsys) -> None:
    """``smoke_test.yaml`` caps batches, so the script must not accept it."""
    exit_code = train.main(["--config", "smoke_test.yaml", "--run-name", "nope"])
    assert exit_code == 2
    assert "smoke" in capsys.readouterr().out
