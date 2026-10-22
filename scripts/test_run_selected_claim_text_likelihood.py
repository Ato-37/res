import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_selected_claim_text_likelihood import (
    CLAIM_FRAME_COUNT, CONDITIONS, _build_sampled_conditions,
    claim_window_contrasts, validate_saved_generation_pts,
)
from presentation_time_utils import uniform_sample_indices
from fractions import Fraction


class ClaimTextLikelihoodTests(unittest.TestCase):
    def test_positive_means_selected_window_mask_hurts_more_than_control(self):
        result = claim_window_contrasts({
            "full": -1.0,
            "claim_window_black": -2.0,
            "temporal_control_black": -1.5,
        })
        self.assertEqual(result["control_minus_selected_mask_nats_per_token"], 0.5)
        self.assertEqual(result["full_minus_selected_mask_nats_per_token"], 1.0)

    def test_requires_exact_three_conditions(self):
        with self.assertRaisesRegex(ValueError, "conditions_invalid"):
            claim_window_contrasts({"full": -1.0})

    def test_rejects_boolean_and_nonfinite_scores(self):
        for value in (True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonfinite"):
                claim_window_contrasts({name: value for name in CONDITIONS})

    def test_all_equal_scores_produce_zero_contrasts(self):
        result = claim_window_contrasts({name: -3.25 for name in CONDITIONS})
        self.assertEqual(result["control_minus_selected_mask_nats_per_token"], 0.0)
        self.assertEqual(result["full_minus_selected_mask_nats_per_token"], 0.0)

    def test_selected_interval_maps_to_generation_grid_with_matched_control(self):
        from PIL import Image
        times = [Fraction(index, 1) for index in range(64)]
        sampled = uniform_sample_indices(len(times), 16)
        frames = [Image.new("RGB", (4, 4), (index, index, index)) for index in range(16)]
        conditions, metadata = _build_sampled_conditions(
            frames, sampled, times, [(Fraction(12, 1), Fraction(20, 1))])
        self.assertEqual(len(conditions["full"]), 16)
        self.assertEqual(set(metadata["condition_frame_sha256"]), set(CONDITIONS))
        self.assertEqual(len(metadata["evidence_ranks"]), len(metadata["control_ranks"]))
        self.assertFalse(set(metadata["evidence_ranks"]) & set(metadata["control_ranks"]))
        self.assertEqual(conditions["full"][0].getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(sum(a.tobytes() != b.tobytes() for a, b in zip(
            conditions["full"], conditions["claim_window_black"], strict=True)),
            len(metadata["evidence_ranks"]))
        self.assertEqual(sum(a.tobytes() != b.tobytes() for a, b in zip(
            conditions["full"], conditions["temporal_control_black"], strict=True)),
            len(metadata["control_ranks"]))

    def test_saved_pts_require_finite_non_boolean_numbers(self):
        actual = [float(i) for i in range(CLAIM_FRAME_COUNT)]
        self.assertTrue(validate_saved_generation_pts(actual, actual))
        for bad in (float("nan"), float("inf"), True, "1.0"):
            saved = list(actual); saved[3] = bad
            with self.subTest(bad=bad):
                self.assertFalse(validate_saved_generation_pts(saved, actual))
        self.assertFalse(validate_saved_generation_pts(actual[:-1], actual))


if __name__ == "__main__":
    unittest.main()
