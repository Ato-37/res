from __future__ import annotations

import unittest

from analyze_frozen_frame_mean_attention_bias_interaction import summarize


class CohortSummaryTests(unittest.TestCase):
    def test_failed_row_disables_full_cohort_estimate_without_survivor_mean(self):
        result = summarize([
            {"row_ordinal": 0, "score_status": "completed", "interaction_nats_per_token": -0.01},
            {"row_ordinal": 1, "score_status": "failed", "failure_code": "selected_claim_invalid"},
            {"row_ordinal": 2, "score_status": "completed", "interaction_nats_per_token": 0.04},
        ], 3)
        self.assertEqual(result["requested_rows"], 3)
        self.assertEqual(result["effective_score_rows"], 2)
        self.assertEqual(result["failed_rows"], 1)
        self.assertFalse(result["full_cohort_estimate_available"])
        self.assertNotIn("mean", result)
        self.assertNotIn("mean_interaction", result)

    def test_rejects_missing_duplicate_or_reordered_ordinals(self):
        for rows in (
            [{"row_ordinal": 0, "score_status": "completed"}],
            [{"row_ordinal": 0, "score_status": "completed"},
             {"row_ordinal": 0, "score_status": "failed"}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize(rows, 2)


if __name__ == "__main__":
    unittest.main()
