"""The module and the notebook must stay the same network.

``ip102_bench/models/propestnet.py`` is a verbatim copy of section 5 of
``notebooks/WJ_ProPestNet.ipynb``. The notebook is the graded artifact and has
to show the architecture inline, so the duplication is deliberate -- but a copy
that silently drifts would load old weights into a new network and quietly
change what the app predicts. This test is what makes the duplication safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ip102_bench.models import build_model  # noqa: E402
from ip102_bench.models.propestnet import ProPestNet  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "WJ_ProPestNet.ipynb"

# docs/propestnet.md reports this exact count; it is also what the trained
# checkpoint contains, so a change here means old weights no longer fit.
EXPECTED_PARAMETERS = 10_988_015


def notebook_propestnet_class(num_classes: int = 15):
    """Build the ProPestNet class defined inside the notebook itself."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = [
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and "class ProPestNet" in "".join(cell["source"])
    ]
    assert sources, "no cell in the notebook defines ProPestNet"

    import torch.nn as nn

    namespace = {"nn": nn, "torch": torch, "NUM_CLASSES": num_classes}
    # Everything up to the cell's own instantiation, which needs a `device`.
    exec(sources[0].split("model = ProPestNet()")[0], namespace)  # noqa: S102
    return namespace["ProPestNet"]


def test_module_matches_notebook_state_dict():
    from_module = ProPestNet(num_classes=15)
    from_notebook = notebook_propestnet_class()(num_classes=15)

    module_keys = list(from_module.state_dict())
    notebook_keys = list(from_notebook.state_dict())
    assert module_keys == notebook_keys, "the module and the notebook define different networks"

    for key, tensor in from_module.state_dict().items():
        assert tensor.shape == from_notebook.state_dict()[key].shape, f"{key} changed shape"


def test_parameter_count_is_the_documented_one():
    model = build_model("propestnet", num_classes=15)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS


def test_registry_default_is_fifteen_classes():
    model = build_model("propestnet")
    assert model(torch.zeros(1, 3, 128, 128)).shape == (1, 15)


def test_ablation_flags_still_build():
    """Section 11 needs these; a refactor that breaks them breaks the writeup."""
    plain = ProPestNet(num_classes=15, multiscale_stem=False, use_residual=False,
                       use_se=False, learned_downsample=False, head="flatten")
    assert plain(torch.zeros(1, 3, 128, 128)).shape == (1, 15)
