from __future__ import annotations

import unittest

from qta_question_swap_metrics import (
    forward_accounting,
    group_jaccard,
    question_repeat_schedule,
    summarize_video_contrasts,
    video_contrast,
)


class QuestionSwapMetricTests(unittest.TestCase):
    def test_identical_repeat_and_cross_sets_yield_zero(self):
        result = video_contrast({"A1": [0, 2], "A2": [0, 2],
                                 "B1": [0, 2], "B2": [0, 2]},
                                group_count=5)
        self.assertEqual(result, {"j_repeat": 1.0, "j_cross": 1.0,
                                  "d_repeat_minus_cross": 0.0})

    def test_cross_question_change_over_repeat_noise_is_positive(self):
        result = video_contrast({"A1": [0, 1], "A2": [0, 1],
                                 "B1": [3, 4], "B2": [3, 4]},
                                group_count=5)
        self.assertEqual(result["j_repeat"], 1.0)
        self.assertEqual(result["j_cross"], 0.0)
        self.assertEqual(result["d_repeat_minus_cross"], 1.0)

    def test_group_jaccard_checks_bounds_and_duplicate_indices(self):
        with self.assertRaisesRegex(ValueError, "selected_group_set_invalid"):
            group_jaccard([0, 0], [1], group_count=3)
        with self.assertRaisesRegex(ValueError, "selected_group_set_invalid"):
            group_jaccard([0], [3], group_count=3)

    def test_summary_reproduces_mean_spread_and_sign_counts(self):
        rows = [
            {"j_repeat": 1.0, "j_cross": 0.5, "d_repeat_minus_cross": 0.5},
            {"j_repeat": 0.5, "j_cross": 1.0, "d_repeat_minus_cross": -0.5},
            {"j_repeat": 0.75, "j_cross": 0.75, "d_repeat_minus_cross": 0.0},
        ]
        summary = summarize_video_contrasts(rows)
        self.assertEqual(summary["mean_d_repeat_minus_cross"], 0.0)
        self.assertEqual(summary["positive_count"], 1)
        self.assertEqual(summary["negative_count"], 1)
        self.assertEqual(summary["zero_within_tolerance_count"], 1)
        self.assertFalse(summary["inferential_test_performed"])

    def test_summary_rejects_wrong_arithmetic(self):
        with self.assertRaisesRegex(ValueError, "range_or_arithmetic_invalid"):
            summarize_video_contrasts([{"j_repeat": 1.0, "j_cross": 0.5,
                                         "d_repeat_minus_cross": 0.0}])

    def test_repeat_schedule_reuses_exact_batch_objects_in_fixed_order(self):
        batch_a, batch_b = object(), object()
        schedule = question_repeat_schedule(batch_a, batch_b, "question A", "question B")
        self.assertEqual([row[0] for row in schedule], ["A1", "A2", "B1", "B2"])
        self.assertIs(schedule[0][1], batch_a)
        self.assertIs(schedule[1][1], batch_a)
        self.assertIs(schedule[2][1], batch_b)
        self.assertIs(schedule[3][1], batch_b)

    def test_forward_accounting_keeps_failures_and_unattempted_separate(self):
        self.assertEqual(forward_accounting(320, 9, 8), {
            "requested_forwards": 320, "attempted_forwards": 9,
            "completed_forwards": 8, "failed_forwards": 1,
            "not_attempted_forwards": 311,
        })
        with self.assertRaisesRegex(ValueError, "accounting_invalid"):
            forward_accounting(4, 5, 4)


if __name__ == "__main__":
    unittest.main()
