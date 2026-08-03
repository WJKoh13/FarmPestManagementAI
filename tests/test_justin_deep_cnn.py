"""Unit tests for the scratch-built deeper sequential CNN."""

import unittest

import torch

from src.models import build_model
from src.models.justin_deep_cnn import ConvStage, JustinDeepCNN


class DeepCNNTests(unittest.TestCase):
    def test_default_model_output_shape_and_parameter_count(self) -> None:
        model = build_model("justin_deep_v2")
        model.eval()

        with torch.inference_mode():
            logits = model(torch.randn(2, 3, 160, 160))

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(trainable_parameters, 1_241_578)

    def test_model_supports_a_different_class_count(self) -> None:
        model = build_model("justin_deep_v2", num_classes=102)
        model.eval()

        with torch.inference_mode():
            logits = model(torch.randn(1, 3, 160, 160))

        self.assertEqual(logits.shape, (1, 102))

    def test_feature_vector_shape(self) -> None:
        model = build_model("justin_deep_v2")
        model.eval()

        with torch.inference_mode():
            features = model.forward_features(torch.randn(2, 3, 160, 160))

        self.assertEqual(features.shape, (2, 256))

    def test_stage_dropouts_are_configurable(self) -> None:
        model = build_model(
            "justin_deep_v2",
            classifier_dropout=0.20,
            stage_dropouts=(0.00, 0.00, 0.05, 0.10),
        )

        dropout_probabilities = [
            stage[-1].p if isinstance(stage[-1], torch.nn.Dropout2d) else 0.0
            for stage in model.features
        ]

        self.assertEqual(dropout_probabilities, [0.0, 0.0, 0.05, 0.10])
        self.assertEqual(model.classifier[-2].p, 0.20)

    def test_backward_pass_produces_finite_gradients(self) -> None:
        model = build_model("justin_deep_v2")
        images = torch.randn(2, 3, 160, 160)
        labels = torch.tensor([0, 9])

        loss = torch.nn.CrossEntropyLoss()(model(images), labels)
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients)
        )

    def test_new_models_receive_independent_random_weights(self) -> None:
        first_model = build_model("justin_deep_v2")
        second_model = build_model("justin_deep_v2")

        first_weight = next(first_model.parameters()).detach()
        second_weight = next(second_model.parameters()).detach()

        self.assertFalse(torch.equal(first_weight, second_weight))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            JustinDeepCNN(num_classes=1)
        with self.assertRaises(ValueError):
            JustinDeepCNN(classifier_dropout=1.0)
        with self.assertRaises(ValueError):
            ConvStage(3, 32, dropout=-0.1)
        with self.assertRaises(ValueError):
            JustinDeepCNN(stage_dropouts=(0.0, 0.1, 0.2))
        with self.assertRaises(ValueError):
            JustinDeepCNN(stage_dropouts=(0.0, 0.1, 0.2, 1.0))


if __name__ == "__main__":
    unittest.main()
