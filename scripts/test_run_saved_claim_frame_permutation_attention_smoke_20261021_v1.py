from __future__ import annotations

import unittest

from run_saved_claim_frame_permutation_attention_smoke import (
    compute_bias_frame_order_interaction,
    deterministic_reverse_block_indices,
    reorder_frames,
    validate_biased_mask_rows,
)
from run_saved_claim_frame_permutation_attention_smoke_20261021_v1 import (
    _total_frame_count_matches,
)
import torch


class SavedClaimFramePermutationSmokeTests(unittest.TestCase):
    def test_video_total_frame_metadata_allows_float_roundoff_only(self):
        self.assertTrue(_total_frame_count_matches(15.999999999999998, 16))
        self.assertFalse(_total_frame_count_matches(15.9, 16))
        self.assertFalse(_total_frame_count_matches(float("nan"), 16))
        self.assertFalse(_total_frame_count_matches("invalid", 16))

    def test_reverse_blocks_preserves_within_block_temporal_patch_order(self):
        self.assertEqual(deterministic_reverse_block_indices(8, 2),
                         (6, 7, 4, 5, 2, 3, 0, 1))
        self.assertEqual(deterministic_reverse_block_indices(6, 3),
                         (3, 4, 5, 0, 1, 2))

    def test_rejects_invalid_block_geometry_and_identity_permutation(self):
        for frame_count, patch in ((0, 1), (1, 1), (8, 0), (7, 2), (4, 4)):
            with self.subTest(frame_count=frame_count, patch=patch), self.assertRaises(ValueError):
                deterministic_reverse_block_indices(frame_count, patch)

    def test_reorder_copies_objects_and_does_not_mutate_source(self):
        class Frame:
            def __init__(self, value):
                self.value = value
            def copy(self):
                return Frame(self.value)

        source = [Frame(i) for i in range(4)]
        output = reorder_frames(source, (2, 3, 0, 1))
        self.assertEqual([x.value for x in output], [2, 3, 0, 1])
        self.assertTrue(all(a is not b for a, b in zip(source, output, strict=True)))
        self.assertEqual([x.value for x in source], [0, 1, 2, 3])

    def test_reorder_fails_closed_for_duplicate_or_missing_indices(self):
        source = [object(), object(), object()]
        for indices in ((0, 0, 2), (0, 1)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                reorder_frames(source, indices)

    def test_interaction_calculation(self):
        # Baseline chronology sensitivity is .4; biased sensitivity is .9.
        result = compute_bias_frame_order_interaction(
            original_baseline=-2.0, permuted_baseline=-2.4,
            original_biased=-1.2, permuted_biased=-2.1)
        self.assertAlmostEqual(
            result["original_minus_permuted_baseline_nats_per_token"], .4)
        self.assertAlmostEqual(
            result["original_minus_permuted_biased_nats_per_token"], .9)
        self.assertAlmostEqual(
            result["bias_by_frame_order_interaction_nats_per_token"], .5)

    def test_null_interaction_when_bias_changes_both_orders_equally(self):
        result = compute_bias_frame_order_interaction(
            original_baseline=-2.0, permuted_baseline=-2.5,
            original_biased=-1.0, permuted_biased=-1.5)
        self.assertAlmostEqual(
            result["bias_by_frame_order_interaction_nats_per_token"], 0.0)

    def test_interaction_rejects_nonfinite_and_bool_scores(self):
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                compute_bias_frame_order_interaction(
                    original_baseline=value, permuted_baseline=0,
                    original_biased=0, permuted_biased=0)

    def test_biased_mask_audit_requires_bias_on_all_selected_cells(self):
        mask = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        mask[0, 0][torch.tensor([3, 4])[:, None], torch.tensor([0, 1])[None, :]] = 0.69314718
        self.assertTrue(validate_biased_mask_rows(mask,
            key_positions=(0, 1), query_positions=(3, 4),
            bias=0.69314718, torch=torch))
        mask[0, 0, 4, 1] = 0.0
        with self.assertRaisesRegex(ValueError, "value_mismatch"):
            validate_biased_mask_rows(mask,
                key_positions=(0, 1), query_positions=(3, 4),
                bias=0.69314718, torch=torch)


if __name__ == "__main__":
    unittest.main()
