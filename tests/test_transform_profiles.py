"""Tests for named preprocessing profiles used by migrated experiments."""

import unittest

from torchvision import transforms

from src.data.transforms import build_eval_transform, build_train_transform


class TransformProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def test_deep_v2_training_profile_matches_notebook(self) -> None:
        pipeline = build_train_transform(
            160,
            self.mean,
            self.std,
            profile="deep_v2",
        )

        self.assertEqual(
            [type(step) for step in pipeline.transforms],
            [
                transforms.RandomResizedCrop,
                transforms.RandomHorizontalFlip,
                transforms.RandomRotation,
                transforms.ColorJitter,
                transforms.ToTensor,
                transforms.Normalize,
            ],
        )
        self.assertEqual(pipeline.transforms[0].size, (160, 160))
        self.assertEqual(pipeline.transforms[0].scale, (0.9, 1.0))

    def test_stretch_evaluation_profile_is_deterministic(self) -> None:
        pipeline = build_eval_transform(
            160,
            self.mean,
            self.std,
            profile="stretch",
        )

        self.assertEqual(
            [type(step) for step in pipeline.transforms],
            [
                transforms.Resize,
                transforms.ToTensor,
                transforms.Normalize,
            ],
        )
        self.assertEqual(pipeline.transforms[0].size, (160, 160))

    def test_unknown_profiles_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_train_transform(160, self.mean, self.std, profile="unknown")
        with self.assertRaises(ValueError):
            build_eval_transform(160, self.mean, self.std, profile="unknown")


if __name__ == "__main__":
    unittest.main()
