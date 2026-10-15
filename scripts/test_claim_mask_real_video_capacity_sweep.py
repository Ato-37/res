"""Unit checks for the fixed real-video capacity-sweep contract."""
from fractions import Fraction
import argparse
import unittest

from run_claim_mask_real_video_capacity_sweep import (
    UNIFORM_TOLERANCE, expected_video_tokens, parse_frame_counts,
    validate_fixed_spatial_grid,
)


class RealVideoCapacitySweepTests(unittest.TestCase):
    def test_frozen_sampling_tolerance_is_half_original_cfr_frame(self):
        self.assertEqual(UNIFORM_TOLERANCE, Fraction(1001, 60000))

    def test_fixed_spatial_grid_video_token_counts_scale_with_temporal_grid(self):
        self.assertEqual(expected_video_tokens([2, 24, 44], 2), 528)
        self.assertEqual(expected_video_tokens([4, 24, 44], 2), 1056)
        self.assertEqual(expected_video_tokens([8, 24, 44], 2), 2112)

    def test_frame_count_cli_allows_only_ordered_prespecified_subsets(self):
        self.assertEqual(parse_frame_counts("4,8,16"), (4, 8, 16))
        self.assertEqual(parse_frame_counts("4,8"), (4, 8))
        for invalid in ("8,16", "4,16", "4,16,8", "4,4", "3,8", "4,32", ""):
            with self.subTest(value=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_frame_counts(invalid)

    def test_spatial_grid_must_match_across_temporal_conditions(self):
        self.assertEqual(validate_fixed_spatial_grid([[2, 24, 44], [4, 24, 44]]), [24, 44])
        with self.assertRaisesRegex(ValueError, "spatial video grid changed"):
            validate_fixed_spatial_grid([[2, 24, 44], [4, 32, 44]])


if __name__ == "__main__":
    unittest.main()
