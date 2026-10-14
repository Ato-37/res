"""Synthetic-only tests for the prospective claim-grounding metric contract."""
import unittest

from temporal_claim_grounding_utils import score_claim_interval


class TemporalClaimGroundingTests(unittest.TestCase):
    def test_multiple_alternative_gold_spans_use_maximum_single_span_iou(self):
        result = score_claim_interval(
            [8, 10], [[0, 2], [8, 10]], 12, "matched")
        self.assertEqual(result, {
            "temporal_iou": 1.0, "status": "scored", "failure_code": None})

    def test_no_overlap_is_a_zero_score_not_an_exclusion(self):
        result = score_claim_interval([4, 5], [[0, 2], [8, 10]], 12, "matched")
        self.assertEqual(result["temporal_iou"], 0.0)
        self.assertEqual(result["status"], "scored")

    def test_large_finite_intervals_do_not_overflow_union(self):
        result = score_claim_interval(
            [0.0, 1.2e308], [[0.5e308, 1.7e308]], 1.7e308, "matched")
        self.assertAlmostEqual(result["temporal_iou"], 0.7 / 1.7, places=14)

    def test_parse_and_invalid_interval_failures_receive_zero(self):
        cases = [
            (None, "timestamp_parse_failure"),
            ([], "prediction_interval_invalid"),
            ([2, 2], "prediction_interval_invalid"),
            ([3, 2], "prediction_interval_invalid"),
            ([-1, 2], "prediction_interval_invalid"),
            ([0, 13], "prediction_interval_invalid"),
            ([0, float("nan")], "prediction_interval_invalid"),
            ([False, 2], "prediction_interval_invalid"),
            ([0, 10**1000], "prediction_interval_invalid"),
        ]
        for prediction, code in cases:
            with self.subTest(prediction=prediction):
                result = score_claim_interval(
                    prediction, [[0, 2]], 12, "matched")
                self.assertEqual(result["temporal_iou"], 0.0)
                self.assertEqual(result["failure_code"], code)

    def test_claim_mismatch_and_unverifiable_remain_zero_scored(self):
        for status, code in (
            ("mismatch", "claim_alignment_mismatch"),
            ("unverifiable", "claim_alignment_unverifiable"),
        ):
            with self.subTest(status=status):
                result = score_claim_interval([1, 2], [], 12, status)
                self.assertEqual(result["temporal_iou"], 0.0)
                self.assertEqual(result["failure_code"], code)

    def test_unresolved_review_and_malformed_gold_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "review must be resolved"):
            score_claim_interval([1, 2], [[0, 2]], 12, "unresolved")
        with self.assertRaisesRegex(ValueError, "at least one"):
            score_claim_interval([1, 2], [], 12, "matched")
        with self.assertRaisesRegex(ValueError, "gold interval"):
            score_claim_interval([1, 2], [[2, 2]], 12, "matched")
        with self.assertRaisesRegex(ValueError, "finite positive"):
            score_claim_interval([1, 2], [[0, 2]], 10**1000, "matched")


if __name__ == "__main__":
    unittest.main()
