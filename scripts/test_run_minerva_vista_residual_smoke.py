from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_minerva_vista_residual_smoke import (
    _choose_sham_group, _dose_vector, _pair_start, _score_name_tokens,
    _target_name_token_positions,
)


class FakeFastTokenizer:
    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is True and clean_up_tokenization_spaces is False
        mapping = {(): "", (4,): "A ", (4, 5): "A cup"}
        return mapping.get(tuple(token_ids), "A xyz")


class VistaResidualHelpersTests(unittest.TestCase):
    def test_pair_mapping_for_even_and_odd_anchor(self):
        self.assertEqual(_pair_start(0), 0)
        self.assertEqual(_pair_start(1), 0)
        self.assertEqual(_pair_start(62), 62)
        self.assertEqual(_pair_start(63), 62)
        with self.assertRaises(ValueError):
            _pair_start(64)

    def test_farthest_temporal_sham_with_low_index_tie_break(self):
        self.assertEqual(_choose_sham_group(0), 31)
        self.assertEqual(_choose_sham_group(31), 0)
        # A midpoint anchor has equal distance to both ends; choose lower group.
        self.assertEqual(_choose_sham_group(2, group_count=5), 0)
        with self.assertRaises(ValueError):
            _choose_sham_group(32)

    def test_dose_vector_is_signed_full_minus_filled_at_ten_percent_rms(self):
        full = torch.tensor([2.0, 0.0, -2.0, 0.0])
        filled = torch.tensor([0.0, 0.0, 0.0, 0.0])
        delta, audit = _dose_vector(full, filled, torch=torch)
        self.assertGreater(float(torch.dot(delta, full)), 0.0)
        self.assertAlmostEqual(audit["normalized_delta_rms_fp32"],
                               audit["full_residual_rms_fp32"] * 0.10, places=6)
        self.assertAlmostEqual(float(delta[0]), 0.2, places=6)

    def test_zero_and_nonfinite_vectors_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "zero_or_nonfinite"):
            _dose_vector(torch.ones(4), torch.ones(4), torch=torch)
        with self.assertRaisesRegex(ValueError, "zero_or_nonfinite"):
            _dose_vector(torch.tensor([1.0, float("nan")]), torch.zeros(2), torch=torch)

    def test_target_name_span_requires_exact_generation_roundtrip(self):
        self.assertEqual(_target_name_token_positions("A cup", [4, 5], "cup",
            FakeFastTokenizer()), (1,))
        with self.assertRaisesRegex(ValueError, "nonmonotonic|final_text_mismatch"):
            _target_name_token_positions("A cup", [4, 7], "cup", FakeFastTokenizer())

    def test_score_uses_absolute_causal_query_positions_for_selected_target_tokens(self):
        # P=5, L=3; logits window rows correspond to P-1, P, P+1, P+2.
        logits = torch.zeros((1, 4, 12), dtype=torch.float32)
        result = _score_name_tokens(logits=logits, target_ids=[2, 3, 4],
            prompt_len=5, target_positions=(1, 2), torch=torch)
        self.assertEqual(result["absolute_query_positions"], [5, 6])
        self.assertEqual(result["target_token_indices"], [1, 2])
        self.assertEqual(result["object_name_token_count"], 2)


if __name__ == "__main__":
    unittest.main()
