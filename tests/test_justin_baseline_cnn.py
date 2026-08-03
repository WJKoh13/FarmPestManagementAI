"""Unit tests for Justin's scratch-built baseline CNN."""

import unittest

import torch

from src.models import build_model
from src.models.justin_baseline_cnn import JustinBaselineCNN


class JustinBaselineCNNTests(unittest.TestCase):
    def test_default_model_output_shape_and_parameter_count(self) -> None:
        model = build_model("justin_baseline")
        model.eval()

        with torch.inference_mode():
            logits = model(torch.randn(2, 3, 160, 160))

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(trainable_parameters, 111_274)

    def test_model_supports_a_different_class_count(self) -> None:
        model = build_model("justin_baseline", num_classes=5)
        model.eval()

        with torch.inference_mode():
            logits = model(torch.randn(3, 3, 160, 160))

        self.assertEqual(logits.shape, (3, 5))

    def test_backward_pass_produces_gradients(self) -> None:
        model = build_model("justin_baseline")
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
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            JustinBaselineCNN(num_classes=1)
        with self.assertRaises(ValueError):
            JustinBaselineCNN(dropout=1.0)
        with self.assertRaises(ValueError):
            JustinBaselineCNN(channels=[])


if __name__ == "__main__":
    unittest.main()

