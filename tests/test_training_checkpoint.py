"""Tests for crash-safe training artifacts."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.train import HISTORY_FIELDS, atomic_torch_save


class TrainingCheckpointTests(unittest.TestCase):
    def test_atomic_save_produces_reloadable_checkpoint_without_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "last_checkpoint.pt"
            payload = {
                "epoch": 7,
                "best_epoch": 5,
                "state_dict": {"weight": torch.tensor([1.0, 2.0])},
            }

            atomic_torch_save(payload, checkpoint_path)

            self.assertTrue(checkpoint_path.is_file())
            self.assertFalse(checkpoint_path.with_suffix(".pt.tmp").exists())
            restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(restored["epoch"], 7)
            self.assertTrue(
                torch.equal(restored["state_dict"]["weight"], torch.tensor([1.0, 2.0]))
            )

    def test_history_records_global_epoch_and_training_macro_f1(self) -> None:
        self.assertEqual(HISTORY_FIELDS[0], "epoch")
        self.assertIn("train_macro_f1", HISTORY_FIELDS)


if __name__ == "__main__":
    unittest.main()
