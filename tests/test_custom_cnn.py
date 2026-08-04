"""The module, the notebook and the legacy checkpoint must stay one network.

``ip102_bench/models/custom_cnn.py`` is a verbatim copy of section 2 of
``notebooks/custom_cnn_ziyang.ipynb``. The notebook is the graded artifact and
has to show the architecture inline, so the duplication is deliberate -- but a
copy that silently drifts would load old weights into a new network and quietly
change what the app predicts. These tests are what make the duplication safe.

They also pin the contract ``scripts/import_custom_cnn_run.py`` depends on:
strict weight loading, the class count and ordering, and the legacy
preprocessing metadata that the app must not replace with the protocol's.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ip102_bench.models import SCRATCH_REGISTRY, build_model  # noqa: E402
from ip102_bench.models.custom_cnn import CustomCNN  # noqa: E402
from ip102_bench.protocol import load_protocol  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "custom_cnn_ziyang.ipynb"

MODEL_NAME = "custom_cnn_ziyang"

# The legacy det_top15 checkpoint contains exactly this many parameters, so a
# change here means those weights no longer fit.
EXPECTED_PARAMETERS = 1_437_167

# What the legacy run was trained with. import_custom_cnn_run.py writes these
# into the bundle verbatim; the protocol's own values are deliberately different.
LEGACY_IMAGE_SIZE = 160
LEGACY_MEAN = [0.485, 0.456, 0.406]
LEGACY_STD = [0.229, 0.224, 0.225]
LEGACY_CROP_MARGIN = 0.15


def notebook_custom_cnn_class():
    """Build the CustomCNN class defined inside the notebook itself."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = [
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and "class CustomCNN" in "".join(cell["source"])
    ]
    assert sources, "no cell in the notebook defines CustomCNN"

    import torch.nn as nn

    namespace = {"nn": nn, "torch": torch}
    # Everything up to the cell's own instantiation, which needs `protocol`.
    exec(sources[0].split("MODEL_NAME")[0], namespace)  # noqa: S102
    return namespace["CustomCNN"]


# -- the notebook and the module are the same network --------------------


def test_module_matches_notebook_state_dict():
    from_module = CustomCNN(num_classes=15)
    from_notebook = notebook_custom_cnn_class()(num_classes=15)

    module_keys = list(from_module.state_dict())
    notebook_keys = list(from_notebook.state_dict())
    assert module_keys == notebook_keys, "the module and the notebook define different networks"

    for key, tensor in from_module.state_dict().items():
        assert tensor.shape == from_notebook.state_dict()[key].shape, f"{key} changed shape"


def test_parameter_count_is_the_one_the_checkpoint_expects():
    model = build_model(MODEL_NAME, num_classes=15)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS


# -- the registry contract -----------------------------------------------


def test_registered_under_a_stable_key():
    assert MODEL_NAME in SCRATCH_REGISTRY


def test_returns_raw_logits_for_the_protocols_class_count():
    protocol = load_protocol()
    model = build_model(MODEL_NAME, num_classes=protocol.num_classes).eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, protocol.image_size, protocol.image_size))

    assert logits.shape == (2, protocol.num_classes)
    # Raw logits, not probabilities: a softmax would make each row sum to 1.
    assert not torch.allclose(logits.exp().sum(dim=1), torch.ones(2))


@pytest.mark.parametrize("num_classes", [2, 10, 15, 102])
def test_class_count_is_never_hard_coded(num_classes):
    model = build_model(MODEL_NAME, num_classes=num_classes).eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 160, 160))
    assert logits.shape == (1, num_classes)


@pytest.mark.parametrize("image_size", [128, 160, 224])
def test_adaptive_pooling_accepts_any_resolution(image_size):
    """Adaptive pooling is what keeps the architecture off a single resolution."""
    model = build_model(MODEL_NAME, num_classes=15).eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, image_size, image_size))
    assert logits.shape == (1, 15)


def test_rejects_a_non_rgb_batch():
    model = build_model(MODEL_NAME, num_classes=15).eval()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 4, 160, 160))


def test_uses_no_pretrained_weights():
    """Two fresh builds must differ -- identical weights would mean a fixed checkpoint."""
    first = build_model(MODEL_NAME, num_classes=15).state_dict()
    second = build_model(MODEL_NAME, num_classes=15).state_dict()
    assert not torch.allclose(first["classifier.weight"], second["classifier.weight"])


# -- strict weight loading ------------------------------------------------


def test_a_matching_state_dict_loads_strictly():
    source = build_model(MODEL_NAME, num_classes=15)
    target = build_model(MODEL_NAME, num_classes=15)
    target.load_state_dict(source.state_dict(), strict=True)

    for key, tensor in source.state_dict().items():
        assert torch.equal(tensor, target.state_dict()[key])


def test_rejects_weights_for_a_different_class_count():
    """A 15-way head must never be reinterpreted under another class list."""
    fifteen = build_model(MODEL_NAME, num_classes=15).state_dict()
    ten = build_model(MODEL_NAME, num_classes=10)
    with pytest.raises(RuntimeError):
        ten.load_state_dict(fifteen, strict=True)


def test_rejects_a_state_dict_with_a_missing_key():
    state_dict = build_model(MODEL_NAME, num_classes=15).state_dict()
    del state_dict["stem.conv.weight"]
    with pytest.raises(RuntimeError):
        build_model(MODEL_NAME, num_classes=15).load_state_dict(state_dict, strict=True)


def test_rejects_a_state_dict_with_an_unexpected_key():
    state_dict = build_model(MODEL_NAME, num_classes=15).state_dict()
    state_dict["not_a_real_layer.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        build_model(MODEL_NAME, num_classes=15).load_state_dict(state_dict, strict=True)


def test_rejects_a_state_dict_with_a_mis_shaped_key():
    state_dict = build_model(MODEL_NAME, num_classes=15).state_dict()
    state_dict["stem.conv.weight"] = torch.zeros(8, 3, 3, 3)
    with pytest.raises(RuntimeError):
        build_model(MODEL_NAME, num_classes=15).load_state_dict(state_dict, strict=True)


def test_rejects_another_architectures_weights():
    propestnet = build_model("propestnet", num_classes=15).state_dict()
    with pytest.raises(RuntimeError):
        build_model(MODEL_NAME, num_classes=15).load_state_dict(propestnet, strict=True)


# -- the legacy import contract -------------------------------------------


def _importer():
    """Load the import script as a module without running its CLI."""
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "import_custom_cnn_run.py"
    spec = importlib.util.spec_from_file_location("import_custom_cnn_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importer_records_the_legacy_preprocessing_not_the_protocols():
    """The bundle must never silently fall back to this repository's pipeline."""
    importer = _importer()
    protocol = load_protocol()

    assert importer.LEGACY_IMAGE_SIZE == LEGACY_IMAGE_SIZE
    assert importer.LEGACY_MEAN == LEGACY_MEAN
    assert importer.LEGACY_STD == LEGACY_STD
    assert importer.LEGACY_CROP_MODE == "box"
    assert importer.LEGACY_CROP_MARGIN == LEGACY_CROP_MARGIN

    # The point of recording them: they differ from the locked protocol.
    assert importer.LEGACY_CROP_MARGIN != protocol.crop_margin
    assert importer.MODEL_NAME == MODEL_NAME


def test_importer_uses_a_legacy_protocol_version_compare_cannot_mix():
    """compare.py filters on protocol_version; the legacy tag must not match v1."""
    importer = _importer()
    assert importer.LEGACY_PROTOCOL_VERSION != load_protocol().version
    assert isinstance(importer.LEGACY_PROTOCOL_VERSION, str)


def test_importer_rejects_a_payload_without_model_state(tmp_path):
    """The legacy format stores weights under 'model_state', not 'state_dict'."""
    importer = _importer()
    path = tmp_path / "wrong_format.pt"
    torch.save({"state_dict": build_model(MODEL_NAME, num_classes=15).state_dict()}, path)

    with pytest.raises(SystemExit):
        importer.load_legacy_state(path)


def test_importer_reads_weights_from_model_state(tmp_path):
    importer = _importer()
    weights = build_model(MODEL_NAME, num_classes=15).state_dict()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state": weights,
            "optimizer_state": {},
            "metadata": {"epoch": 48, "num_classes": 15},
        },
        path,
    )

    state_dict, metadata = importer.load_legacy_state(path)
    assert list(state_dict) == list(weights)
    assert metadata["epoch"] == 48

    # And what it returns must load strictly, which is the whole contract.
    build_model(MODEL_NAME, num_classes=15).load_state_dict(state_dict, strict=True)


def test_class_names_are_the_protocols_ordered_fifteen():
    """The legacy class order matches detection_top15; ordering is load-bearing."""
    protocol = load_protocol()
    assert protocol.num_classes == 15
    assert protocol.class_names[0] == "grub"
    assert protocol.class_names[-1] == "cicadellidae"
    assert len(protocol.display_names) == 15
