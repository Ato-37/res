"""CPU contract tests for STEAR-style Qwen attention selection smoke."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_qwen_stear_attention_feasibility import (  # noqa: E402
    select_top_visual_positions,
)


class AttentionSelectorTests(unittest.TestCase):
    def test_ranks_only_visual_positions_and_breaks_ties_by_position(self):
        import torch
        layer0 = torch.zeros((1, 2, 8, 8))
        layer1 = torch.zeros((1, 2, 8, 8))
        layer0[0, :, -1, 2] = 0.8
        layer0[0, :, -1, 4] = 0.2
        layer1[0, :, -1, 2] = 0.2
        layer1[0, :, -1, 4] = 0.8
        layer0[0, :, -1, 6] = 1.0  # High text-key score must be ignored.
        layer1[0, :, -1, 6] = 1.0
        chosen, scores = select_top_visual_positions(
            (layer0, layer1), (2, 4), layer_indices=(0, 1), ratio=0.5)
        self.assertEqual(chosen, (2,))
        self.assertEqual(scores, (0.5, 0.5))

    def test_ratio_selects_at_least_one_and_rejects_bad_attention(self):
        import torch
        attention = torch.zeros((1, 1, 5, 5))
        attention[0, 0, -1, 1] = 0.4
        selected, _ = select_top_visual_positions(
            (attention,), (1, 3, 4), layer_indices=(0,), ratio=0.1)
        self.assertEqual(len(selected), 1)
        bad = attention.clone()
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "attention_tensor_invalid"):
            select_top_visual_positions((bad,), (1, 3), layer_indices=(0,))

    def test_rejects_unsorted_positions_and_invalid_layer_indices(self):
        import torch
        attention = torch.zeros((1, 1, 5, 5))
        with self.assertRaisesRegex(ValueError, "visual_positions_invalid"):
            select_top_visual_positions((attention,), (3, 1), layer_indices=(0,))
        with self.assertRaisesRegex(ValueError, "attention_layer_indices_invalid"):
            select_top_visual_positions((attention,), (1, 3), layer_indices=(1,))
        with self.assertRaisesRegex(ValueError, "attention_layer_indices_invalid"):
            select_top_visual_positions((attention,), (1, 3), layer_indices=(0, 0))

    def test_rejects_batches_other_than_one(self):
        import torch
        attention = torch.zeros((2, 1, 5, 5))
        with self.assertRaisesRegex(ValueError, "attention_sequence_shape_invalid"):
            select_top_visual_positions((attention,), (1, 3), layer_indices=(0,))


if __name__ == "__main__":
    unittest.main()
