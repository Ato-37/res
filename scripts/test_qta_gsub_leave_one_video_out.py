from __future__ import annotations

import math
import unittest

from analyze_qta_question_swap_gsub_alignment import leave_one_video_out_summary


class LeaveOneVideoOutSummaryTests(unittest.TestCase):
    def test_finite_cohort_influence_is_aggregate_only(self):
        result = leave_one_video_out_summary([-1.0, 0.0, 1.0])
        self.assertEqual(result["n_video_pairs"], 3)
        self.assertEqual(result["full_mean"], 0.0)
        self.assertEqual(result["leave_one_out_mean_min"], -0.5)
        self.assertEqual(result["leave_one_out_mean_max"], 0.5)
        self.assertEqual(result["max_absolute_leave_one_out_shift"], 0.5)
        self.assertFalse(result["inference_performed"])
        self.assertEqual(set(result), {
            "n_video_pairs", "full_mean", "leave_one_out_mean_min",
            "leave_one_out_mean_max", "max_absolute_leave_one_out_shift",
            "inference_performed",
        })

    def test_rejects_insufficient_or_nonfinite_input(self):
        for values in ([], [1.0], [math.nan, 0.0], [math.inf, 0.0], [True, 0.0]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "video_pair_values_invalid"):
                    leave_one_video_out_summary(values)


if __name__ == "__main__":
    unittest.main()
