import sys
import unittest

import numpy as np

sys.path.insert(0, "scripts")
from analyze_head_query_temporal_resolution import compare_four_strata


class TemporalResolutionAnalysisTests(unittest.TestCase):
    def _metrics(self, seed):
        rng = np.random.default_rng(seed)
        return (rng.random((4, 28, 28)), rng.random((4, 28, 28)),
                rng.integers(0, 4, size=(4, 28, 28), dtype=np.uint8))

    def test_aligns_only_shared_sample_indices_and_reports_four_conditions(self):
        m8 = {"generation_prefix": self._metrics(1),
              "question_body_mean": self._metrics(2)}
        m16 = {"generation_prefix": self._metrics(3),
               "question_body_mean": self._metrics(4)}
        result = compare_four_strata(m8, [0, 1, 3, 4], m16, [1, 2, 3, 5])
        self.assertEqual(result["matched_sample_count"], 2)
        self.assertEqual(set(result["ranked_conditions"]), {
            "8_frame_generation_prefix", "8_frame_question_body_mean",
            "16_frame_generation_prefix", "16_frame_question_body_mean"})
        self.assertIn("query_position_jaccard_difference_16_minus_8",
                      result["contrasts"])

    def test_rejects_empty_or_duplicate_indices(self):
        metrics = {"generation_prefix": self._metrics(1),
                   "question_body_mean": self._metrics(2)}
        with self.assertRaisesRegex(ValueError, "no_common_successful_samples"):
            compare_four_strata(metrics, [0, 1, 2, 3], metrics, [4, 5, 6, 7])
        with self.assertRaisesRegex(ValueError, "sample_indices_must_be_unique_vectors"):
            compare_four_strata(metrics, [0, 0, 2, 3], metrics, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
