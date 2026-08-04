"""Tests for the CNN architectures and their configuration contract."""

from __future__ import annotations

import pytest

# The vision package imports torch at module scope, unlike the data layer, so
# the whole file skips when the training extras are absent.
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from farm_pest_ai.config import Config  # noqa: E402
from farm_pest_ai.scopes import FULL102, RICE10  # noqa: E402
from farm_pest_ai.vision.blocks import (  # noqa: E402
    ConvBNAct,
    DepthwiseSeparableConv,
    DropPath,
    ResidualSeparableBlock,
    SqueezeExcite,
    build_activation,
    build_norm,
)
from farm_pest_ai.vision.models import (  # noqa: E402
    MODEL_NAMES,
    BaselineCNN,
    CustomCNN,
    ModelConfig,
    ModelError,
    build_model,
    count_parameters,
    model_config_from_config,
    summarize_model,
)

# -- blocks -------------------------------------------------------------


def test_conv_bn_act_omits_bias_when_normalised() -> None:
    block = ConvBNAct(3, 8, 3, norm="batchnorm")
    assert block.conv.bias is None


def test_conv_bn_act_keeps_bias_without_norm() -> None:
    block = ConvBNAct(3, 8, 3, norm="none")
    assert block.conv.bias is not None


def test_conv_bn_act_preserves_spatial_size_at_stride_one() -> None:
    block = ConvBNAct(3, 8, 3)
    assert block(torch.randn(2, 3, 32, 32)).shape == (2, 8, 32, 32)


def test_conv_bn_act_halves_spatial_size_at_stride_two() -> None:
    block = ConvBNAct(3, 8, 3, stride=2)
    assert block(torch.randn(2, 3, 32, 32)).shape == (2, 8, 16, 16)


def test_depthwise_separable_has_fewer_parameters_than_dense() -> None:
    separable = DepthwiseSeparableConv(64, 64, 3)
    dense = ConvBNAct(64, 64, 3)
    separable_count = sum(p.numel() for p in separable.parameters())
    dense_count = sum(p.numel() for p in dense.parameters())
    assert separable_count < dense_count


def test_squeeze_excite_preserves_shape() -> None:
    block = SqueezeExcite(16, 0.25)
    x = torch.randn(2, 16, 8, 8)
    assert block(x).shape == x.shape


def test_squeeze_excite_gate_only_attenuates() -> None:
    """The sigmoid gate can scale a channel down but never invert its sign."""
    block = SqueezeExcite(8, 0.25)
    x = torch.abs(torch.randn(4, 8, 6, 6)) + 0.1
    out = block(x)
    assert bool((out >= 0).all())
    assert bool((out <= x + 1e-5).all())


def test_squeeze_excite_bottleneck_is_at_least_one_channel() -> None:
    """A tiny ratio must not produce a zero-width bottleneck."""
    block = SqueezeExcite(4, 0.01)
    assert block.reduce.out_channels >= 1
    assert block(torch.randn(1, 4, 4, 4)).shape == (1, 4, 4, 4)


def test_drop_path_is_identity_in_eval_mode() -> None:
    block = DropPath(0.5).eval()
    x = torch.randn(8, 4, 2, 2)
    assert torch.equal(block(x), x)


def test_drop_path_is_identity_when_probability_is_zero() -> None:
    block = DropPath(0.0).train()
    x = torch.randn(8, 4, 2, 2)
    assert torch.equal(block(x), x)


def test_drop_path_drops_whole_samples_in_training() -> None:
    """Dropping must be per-sample, not per-element."""
    torch.manual_seed(0)
    block = DropPath(0.5).train()
    out = block(torch.ones(64, 4, 2, 2))
    per_sample = out.flatten(1)
    # Each sample is either entirely zero or entirely scaled; never mixed.
    for row in per_sample:
        assert bool((row == 0).all()) or bool((row != 0).all())
    assert bool((per_sample == 0).any())


def test_drop_path_preserves_expected_value() -> None:
    torch.manual_seed(0)
    block = DropPath(0.25).train()
    out = block(torch.ones(4096, 2, 1, 1))
    assert out.mean().item() == pytest.approx(1.0, abs=0.05)


def test_drop_path_rejects_probability_of_one() -> None:
    with pytest.raises(ValueError, match=r"drop_prob must be in \[0, 1\)"):
        DropPath(1.0)


def test_residual_block_identity_shortcut_when_shape_unchanged() -> None:
    block = ResidualSeparableBlock(32, 32, stride=1)
    assert isinstance(block.shortcut, torch.nn.Identity)


def test_residual_block_projects_when_shape_changes() -> None:
    block = ResidualSeparableBlock(32, 64, stride=2)
    assert not isinstance(block.shortcut, torch.nn.Identity)
    assert block(torch.randn(2, 32, 16, 16)).shape == (2, 64, 8, 8)


def test_build_activation_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        build_activation("mish")


def test_build_norm_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown norm"):
        build_norm("layernorm", 8)


def test_group_norm_adapts_when_groups_do_not_divide_channels() -> None:
    norm = build_norm("groupnorm", 12, groups=8)
    assert 12 % norm.num_groups == 0


# -- model contract -----------------------------------------------------


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("num_classes", [10, 102])
def test_models_output_expected_logit_count(name: str, num_classes: int) -> None:
    model = build_model(ModelConfig(name=name, num_classes=num_classes)).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 160, 160))
    assert logits.shape == (2, num_classes)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_emit_raw_logits_not_probabilities(name: str) -> None:
    """No softmax inside the model: the losses expect logits."""
    model = build_model(ModelConfig(name=name, num_classes=10)).eval()
    with torch.no_grad():
        logits = model(torch.randn(8, 3, 160, 160))
    row_sums = logits.sum(dim=1)
    assert not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3))
    assert bool((logits < 0).any())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_reject_non_rgb_input(name: str) -> None:
    """A four-channel tensor must fail at the model boundary, not deep inside."""
    model = build_model(ModelConfig(name=name, num_classes=10))
    with pytest.raises(ModelError, match="expected 3 input channels"):
        model(torch.randn(2, 4, 160, 160))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_reject_non_4d_input(name: str) -> None:
    model = build_model(ModelConfig(name=name, num_classes=10))
    with pytest.raises(ModelError, match="expected a 4-D"):
        model(torch.randn(3, 160, 160))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_gradients_reach_every_parameter(name: str) -> None:
    """Every trainable parameter must be connected to the loss."""
    model = build_model(ModelConfig(name=name, num_classes=10))
    model.train()
    logits = model(torch.randn(4, 3, 160, 160))
    logits.sum().backward()
    missing = [
        param_name
        for param_name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []


def test_build_model_selects_the_configured_architecture() -> None:
    assert isinstance(build_model(ModelConfig(name="custom_cnn", num_classes=10)), CustomCNN)
    assert isinstance(
        build_model(ModelConfig(name="baseline_cnn", num_classes=10)), BaselineCNN
    )


def test_build_model_rejects_scope_mismatch() -> None:
    """A 10-class model may not be built under a 102-class scope."""
    with pytest.raises(ModelError, match="may never be reinterpreted"):
        build_model(ModelConfig(name="custom_cnn", num_classes=10), scope=FULL102)


def test_build_model_accepts_matching_scope() -> None:
    model = build_model(ModelConfig(name="custom_cnn", num_classes=10), scope=RICE10)
    assert model.num_classes == RICE10.num_classes


def test_custom_cnn_is_smaller_than_baseline() -> None:
    """The separable design must justify itself on parameter count."""
    baseline = count_parameters(build_model(ModelConfig(name="baseline_cnn", num_classes=10)))
    custom = count_parameters(build_model(ModelConfig(name="custom_cnn", num_classes=10)))
    assert custom["total"] < baseline["total"]


def test_drop_path_ramps_linearly_with_depth() -> None:
    """Early blocks must be dropped less often than late ones."""
    model = build_model(
        ModelConfig(
            name="custom_cnn",
            num_classes=10,
            stage_channels=(16, 32),
            stage_blocks=(2, 2),
            stage_strides=(2, 2),
            drop_path=0.4,
        )
    )
    probabilities = [
        module.drop_prob for module in model.modules() if isinstance(module, DropPath)
    ]
    assert probabilities[0] == 0.0
    assert probabilities == sorted(probabilities)
    assert probabilities[-1] == pytest.approx(0.4)


# -- configuration ------------------------------------------------------


def test_model_config_rejects_stated_num_classes() -> None:
    """num_classes has exactly one source of truth: dataset.scope."""
    config = Config(
        data={
            "dataset": {"scope": "rice10"},
            "paths": {},
            "model": {"name": "custom_cnn", "num_classes": 10},
        }
    )
    with pytest.raises(ModelError, match="must not be stated in configuration"):
        model_config_from_config(config)


@pytest.mark.parametrize(
    ("scope", "expected"), [("rice10", 10), ("full102", 102)]
)
def test_model_config_derives_num_classes_from_scope(scope: str, expected: int) -> None:
    config = Config(data={"dataset": {"scope": scope}, "paths": {}, "model": {}})
    assert model_config_from_config(config).num_classes == expected


def test_model_config_rejects_mismatched_stage_lengths() -> None:
    with pytest.raises(ModelError, match="stage_blocks"):
        ModelConfig(
            name="custom_cnn",
            num_classes=10,
            stage_channels=(32, 64),
            stage_blocks=(1, 1, 1),
        ).validate()


def test_model_config_rejects_unknown_architecture() -> None:
    with pytest.raises(ModelError, match=r"unknown model\.name"):
        ModelConfig(name="resnet50", num_classes=10).validate()


def test_model_config_rejects_dropout_of_one() -> None:
    with pytest.raises(ModelError, match=r"model.dropout must be in \[0, 1\)"):
        ModelConfig(num_classes=10, dropout=1.0).validate()


def test_model_config_rejects_single_class() -> None:
    with pytest.raises(ModelError, match="at least two classes"):
        ModelConfig(num_classes=1).validate()


def test_model_config_roundtrips_through_dict() -> None:
    original = ModelConfig(name="custom_cnn", num_classes=102, dropout=0.25)
    payload = original.to_dict()
    assert payload["num_classes"] == 102
    assert payload["dropout"] == 0.25


# -- summary ------------------------------------------------------------


def test_summarize_model_reports_output_width() -> None:
    model = build_model(ModelConfig(name="custom_cnn", num_classes=102))
    summary = summarize_model(model)
    assert summary["num_classes"] == 102
    assert summary["output_shape"] == [102]
    assert summary["parameters"]["total"] > 0


def test_summarize_model_restores_training_mode() -> None:
    """The probe must not leave the model in eval mode."""
    model = build_model(ModelConfig(name="custom_cnn", num_classes=10))
    model.train()
    summarize_model(model)
    assert model.training


def test_summarize_model_does_not_perturb_batchnorm_statistics() -> None:
    """Running the probe must not change BatchNorm running statistics."""
    model = build_model(ModelConfig(name="custom_cnn", num_classes=10))
    model.train()
    before = [b.clone() for b in model.buffers()]
    summarize_model(model)
    after = list(model.buffers())
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_count_parameters_separates_buffers() -> None:
    model = build_model(ModelConfig(name="custom_cnn", num_classes=10))
    counts = count_parameters(model)
    assert counts["buffers"] > 0
    assert counts["trainable"] <= counts["total"]
