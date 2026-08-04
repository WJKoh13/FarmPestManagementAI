"""A run written by a notebook must be servable by the app, unchanged.

``save_run`` is the only interface between a notebook and the app: whatever it
writes into ``runs/<model>/<run_id>/`` is all the app will ever know about those
weights. The failure this file guards against is not a crash -- it is a bundle
that loads perfectly and is then fed through the wrong transform, because a
state dict carries weights but not the preprocessing that produced them.

That already happened once in this project (see the module docstring of
app/propest_inference.py) and it was caught only because two metrics disagreed
on identical images. A protocol run trains at 160px with the normalization in
data_manifests/norm_stats.json; the app's fallbacks are 128px and ImageNet
statistics. Both numbers are reasonable, neither errors, and serving one model
through the other's pipeline just quietly makes it worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.cnn_model import (  # noqa: E402
    DEFAULT_IMAGE_SIZE, DEFAULT_MEAN, _score, describe_runs, load_best_model,
)
from ip102_bench.artifacts import save_run  # noqa: E402
from ip102_bench.models import build_model  # noqa: E402
from ip102_bench.protocol import load_protocol  # noqa: E402
from ip102_bench.training import TrainingResult  # noqa: E402

# The smallest architecture the registry actually builds, for the same reason
# tests/run_bundles.py picks it: these fixtures write real weights to disk.
MODEL_NAME = "custom_cnn_ziyang"


def stub_test_loader(num_classes: int, image_size: int):
    """The two batches ``predict_split`` needs, without touching the dataset.

    ``predict_split`` only iterates and reads ``batch[0], batch[1], batch[2]``,
    so a list of tuples is a loader as far as it is concerned. Nothing here
    asserts on the metrics that come out -- they are noise by construction. The
    subject is the bundle, not the numbers in it.
    """
    return [
        (torch.randn(2, 3, image_size, image_size), torch.tensor([0, 1 % num_classes]),
         ["a.jpg", "b.jpg"]),
    ]


def one_epoch_history() -> pd.DataFrame:
    """The columns plot_curves reads. One row is enough to draw a figure."""
    return pd.DataFrame(
        [{"epoch": 1, "train_accuracy": 0.4, "val_accuracy": 0.3,
          "train_loss": 1.9, "val_loss": 2.1, "val_macro_f1": 0.25}]
    )


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory) -> tuple[Path, object]:
    """A real ``save_run`` bundle in a temporary ``runs/``. Never the repo's."""
    protocol = load_protocol()
    runs_dir = tmp_path_factory.mktemp("runs")
    # `output_root` is the documented per-machine runtime knob, and redirecting
    # it is what keeps this test off the developer's real runs/ directory.
    protocol.runtime["output_root"] = str(runs_dir)
    model = build_model(MODEL_NAME, num_classes=protocol.num_classes)

    result = TrainingResult(
        history=one_epoch_history(),
        best_epoch=1,
        best_val_metric=0.25,
        best_state_dict=model.state_dict(),
        epochs_trained=1,
        training_seconds=1.0,
        stopped_early=False,
    )

    save_run(
        model=model,
        model_name=MODEL_NAME,
        protocol=protocol,
        result=result,
        test_loader=stub_test_loader(protocol.num_classes, protocol.image_size),
        pretrained=False,
        author="test",
        device="cpu",
        run_id="test_run",
    )
    return runs_dir, protocol


def test_the_app_serves_a_notebook_run_through_its_training_transform(saved_run):
    """The regression: 160px protocol weights must not be served at 128px."""
    runs_dir, protocol = saved_run
    loaded = load_best_model(num_classes=protocol.num_classes, runs_dir=runs_dir)

    assert loaded.model is not None, loaded.reason
    assert loaded.image_size == protocol.image_size
    assert loaded.mean == pytest.approx(protocol.norm_stats()[0])
    assert loaded.std == pytest.approx(protocol.norm_stats()[1])
    assert loaded.crop_mode == protocol.crop_mode
    assert loaded.crop_margin == pytest.approx(protocol.crop_margin)

    # The fallbacks are what the app uses when a bundle says nothing, and they
    # are ProPestNet's, not the protocol's. If a protocol run ever comes back
    # holding these, the metadata stopped being written and the test above is
    # passing by coincidence.
    assert loaded.image_size != DEFAULT_IMAGE_SIZE
    assert loaded.mean != pytest.approx(DEFAULT_MEAN)


def test_the_bundle_carries_the_class_list_and_display_names(saved_run):
    runs_dir, protocol = saved_run
    loaded = load_best_model(num_classes=protocol.num_classes, runs_dir=runs_dir)

    assert loaded.class_names == protocol.class_names
    # The farmer sees these; slugs like `black_cutworm` are not an answer.
    assert loaded.display_names == protocol.display_names


def test_the_model_picker_can_read_the_class_count(saved_run):
    """describe_runs reads results.json only -- and save_run names it differently.

    The importers write `classes`, save_run writes `class_names`. Reading one key
    made every notebook run report an unknown class count, which silently
    disabled the compatibility check the picker exists to show.
    """
    runs_dir, protocol = saved_run
    described = describe_runs(num_classes=protocol.num_classes, runs_dir=runs_dir)

    assert len(described) == 1
    assert described[0]["num_classes"] == protocol.num_classes
    assert described[0]["usable"], described[0]["problem"]
    assert described[0]["model"] == MODEL_NAME


def test_ranking_uses_the_setting_the_run_will_be_served_under():
    """A run is ranked by the most corrected score it records, not its raw one.

    That is the right rule -- the app serves each model through its own TTA and
    its own recorded prior, so the corrected number is what a farmer actually
    gets. The catch is that it only compares fairly if every bundle carries the
    corrected figure, and ``save_run`` records a single pass. Closing that gap
    is what ``scripts/evaluate_tta.py`` exists for.
    """
    served = {
        "test_with_tta_and_prior": {"macro_f1": 0.90},
        "test_with_tta": {"macro_f1": 0.80},
        "test": {"macro_f1": 0.50},
    }
    assert _score(served) == pytest.approx(0.90)
    assert _score({"test_with_tta": {"macro_f1": 0.80}, "test": {"macro_f1": 0.50}}) == (
        pytest.approx(0.80)
    )

    # What save_run writes: flat, single-pass, no nesting.
    assert _score({"macro_f1": 0.61}) == pytest.approx(0.61)
    # And a run with a real score beats one that can only evidence validation.
    assert _score({"macro_f1": 0.61}) > _score({"best_val_macro_f1": 0.60})
    assert _score({}) == -1.0
