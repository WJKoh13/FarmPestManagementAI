"""The module, the notebook and Beatrice's checkpoint must stay one network.

``ip102_bench/models/vgg_cnn.py`` carries a verbatim copy of section 9 of
``notebooks/Beatrice_vgg19_xml_cropped.ipynb``. The notebook is the graded
artifact and has to show the architecture inline, so the duplication is
deliberate -- but a copy that silently drifts would load her weights into a
different network and quietly change what the app predicts. These tests are what
make the duplication safe.

They also pin what ``scripts/import_vgg19_run.py`` depends on: strict weight
loading, the class count, and the registry key the app rebuilds the architecture
from. Her weights only exist as a file on her machine; if any of this drifts,
that file becomes unloadable and there is no way to regenerate it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ip102_bench.models import SCRATCH_REGISTRY, build_model  # noqa: E402
from ip102_bench.models.vgg_cnn import VGG19  # noqa: E402
from ip102_bench.protocol import load_protocol  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "Beatrice_vgg19_xml_cropped.ipynb"

MODEL_NAME = "vgg19_beatrice"

# Her notebook trains at this resolution with ImageNet statistics and a 0.05 box
# padding -- none of which are the protocol's. The importer records them on the
# bundle so the app serves her model the way she trained it.
HER_IMAGE_SIZE = 128
HER_CROP_MARGIN = 0.05


def notebook_vgg19_class():
    """Build the VGG19 class defined inside the notebook itself."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = [
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and "class VGG19" in "".join(cell["source"])
    ]
    assert sources, "no cell in the notebook defines VGG19"

    import torch.nn as nn

    namespace = {"nn": nn, "torch": torch}
    # Everything up to the cell's own instantiation, which needs NUM_CLASSES.
    exec(sources[0].split("model = VGG19(")[0], namespace)  # noqa: S102
    return namespace["VGG19"]


# -- the notebook and the module are the same network --------------------


def test_module_matches_notebook_state_dict():
    from_module = VGG19(num_classes=15)
    from_notebook = notebook_vgg19_class()(num_classes=15)

    assert list(from_module.state_dict()) == list(from_notebook.state_dict()), (
        "the module and the notebook define different networks"
    )
    for key, tensor in from_module.state_dict().items():
        assert tensor.shape == from_notebook.state_dict()[key].shape, f"{key} changed shape"


def test_parameter_count_matches_the_notebook():
    """Her checkpoint fits exactly this many parameters, and no other number."""
    from_module = sum(p.numel() for p in VGG19(num_classes=15).parameters())
    from_notebook = sum(p.numel() for p in notebook_vgg19_class()(num_classes=15).parameters())
    assert from_module == from_notebook


def test_the_small_classifier_variant_also_matches():
    """Her config exposes a 512-unit head; the port must carry that faithfully."""
    module = VGG19(num_classes=15, strict_classifier=False, small_classifier_units=512)
    notebook = notebook_vgg19_class()(
        num_classes=15, strict_classifier=False, small_classifier_units=512
    )
    assert list(module.state_dict()) == list(notebook.state_dict())
    assert module.state_dict()["classifier.0.weight"].shape == (
        notebook.state_dict()["classifier.0.weight"].shape
    )


# -- the registry contract -----------------------------------------------


def test_registered_under_a_stable_key():
    assert MODEL_NAME in SCRATCH_REGISTRY


def test_does_not_redefine_the_half_width_vgg_keys():
    """`vgg16`/`vgg19` belong to the other spec; hers must not squat on them.

    Those keys point at the ``VGGCNN`` stub, a half-width BatchNorm variant on a
    0.5-5M parameter budget. Pointing them at this full-width network instead
    would orphan any checkpoint saved against the stub's spec.
    """
    assert SCRATCH_REGISTRY["vgg19"] is not SCRATCH_REGISTRY[MODEL_NAME]


def test_returns_raw_logits_at_her_resolution():
    model = build_model(MODEL_NAME, num_classes=15).eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, HER_IMAGE_SIZE, HER_IMAGE_SIZE))

    assert logits.shape == (2, 15)
    # Raw logits, not probabilities: a softmax would make each row sum to 1.
    assert not torch.allclose(logits.exp().sum(dim=1), torch.ones(2))


@pytest.mark.parametrize("image_size", [128, 160, 224])
def test_adaptive_pooling_accepts_any_resolution(image_size):
    model = build_model(MODEL_NAME, num_classes=15).eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, image_size, image_size))
    assert logits.shape == (1, 15)


def test_class_count_comes_from_the_caller():
    protocol = load_protocol()
    model = build_model(MODEL_NAME, num_classes=protocol.num_classes).eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, HER_IMAGE_SIZE, HER_IMAGE_SIZE))
    assert logits.shape == (1, protocol.num_classes)


def test_uses_no_pretrained_weights():
    """Two fresh builds must differ -- identical weights would mean a fixed checkpoint."""
    first = build_model(MODEL_NAME, num_classes=15).state_dict()
    second = build_model(MODEL_NAME, num_classes=15).state_dict()
    # classifier.6 is the final Linear: 0 Linear, 1 ReLU, 2 Dropout, 3 Linear,
    # 4 ReLU, 5 Dropout, 6 Linear.
    assert not torch.allclose(first["classifier.6.weight"], second["classifier.6.weight"])


# -- strict weight loading, which is how her checkpoint is imported -------


def test_a_matching_state_dict_loads_strictly():
    source = build_model(MODEL_NAME, num_classes=15)
    target = build_model(MODEL_NAME, num_classes=15)
    target.load_state_dict(source.state_dict(), strict=True)

    for key, tensor in source.state_dict().items():
        assert torch.equal(tensor, target.state_dict()[key])


def test_rejects_weights_for_a_different_class_count():
    ten_class = build_model(MODEL_NAME, num_classes=10).state_dict()
    with pytest.raises(RuntimeError):
        build_model(MODEL_NAME, num_classes=15).load_state_dict(ten_class, strict=True)


def test_rejects_another_architectures_weights():
    other = build_model("custom_cnn_ziyang", num_classes=15).state_dict()
    with pytest.raises(RuntimeError):
        build_model(MODEL_NAME, num_classes=15).load_state_dict(other, strict=True)


# -- the import contract --------------------------------------------------


def _importer():
    """Load the import script as a module without running its CLI."""
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "import_vgg19_run.py"
    spec = importlib.util.spec_from_file_location("import_vgg19_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importer_records_her_preprocessing_not_the_protocols():
    """The bundle must never silently fall back to this repository's pipeline."""
    importer = _importer()
    protocol = load_protocol()

    assert importer.HER_IMAGE_SIZE == HER_IMAGE_SIZE
    assert importer.HER_CROP_MARGIN == HER_CROP_MARGIN
    assert importer.HER_MEAN == [0.485, 0.456, 0.406]      # ImageNet
    assert importer.HER_CROP_MODE == "box"
    assert importer.MODEL_NAME == MODEL_NAME

    # The point of recording them: they differ from the locked protocol on both
    # axes, so a fallback would be wrong twice over.
    assert importer.HER_CROP_MARGIN != protocol.crop_margin
    assert importer.HER_IMAGE_SIZE != protocol.image_size
    assert importer.HER_MEAN != protocol.norm_stats()[0]


def test_importer_uses_a_protocol_version_compare_cannot_mix():
    """compare.py filters on protocol_version; hers must not match v1."""
    importer = _importer()
    assert importer.HER_PROTOCOL_VERSION != load_protocol().version
    assert isinstance(importer.HER_PROTOCOL_VERSION, str)


def test_importer_rejects_a_payload_in_another_format(tmp_path):
    """Her notebook writes 'model_state_dict', not 'state_dict' or 'model_state'."""
    importer = _importer()
    path = tmp_path / "wrong_format.pth"
    torch.save({"state_dict": build_model(MODEL_NAME, num_classes=15).state_dict()}, path)

    with pytest.raises(SystemExit):
        importer.load_her_state(path)


def test_importer_reads_weights_and_config_from_her_format(tmp_path):
    importer = _importer()
    weights = build_model(MODEL_NAME, num_classes=15).state_dict()
    path = tmp_path / "xml_crop_vgg19_best.pth"
    torch.save(
        {
            "model_state_dict": weights,
            "epoch": 7,
            "best_val_accuracy": 41.2,
            "config": {"image_size": 128, "num_classes": 15, "box_padding": 0.05},
        },
        path,
    )

    state_dict, config = importer.load_her_state(path)
    assert list(state_dict) == list(weights)
    assert config["image_size"] == HER_IMAGE_SIZE
    assert config["box_padding"] == HER_CROP_MARGIN
    assert config["epoch"] == 7

    # And what it returns must load strictly, which is the whole contract.
    build_model(MODEL_NAME, num_classes=15).load_state_dict(state_dict, strict=True)


def test_importer_verifies_label_ordering_across_both_manifests():
    """Her labels come from splits_top15.json; the app names them from classes_top15.json.

    If those two ever disagree the importer must refuse rather than write a
    bundle whose every prediction carries the wrong pest's name.
    """
    importer = _importer()
    importer.verify_label_ordering(load_protocol())  # must not raise today
