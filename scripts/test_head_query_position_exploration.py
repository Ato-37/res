import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, "scripts")
from run_head_query_position_exploration import (
    compare_rankings,
    find_unique_question_span,
    spearman_average_ties,
    summarize_all_heads,
    temporal_summary,
)


class HeadQueryPositionTests(unittest.TestCase):
    def test_exact_question_span_after_video(self):
        self.assertEqual(find_unique_question_span(
            [1, 9, 9, 9, 3, 4, 5, 6], [4, 5], 1, 4), (5, 7))

    def test_missing_or_duplicate_question_span_rejected(self):
        with self.assertRaisesRegex(ValueError, "question_span_not_unique"):
            find_unique_question_span([1, 2, 3], [4], 0, 1)
        with self.assertRaisesRegex(ValueError, "question_span_not_unique"):
            find_unique_question_span([9, 9, 4, 5, 4, 5], [4, 5], 0, 2)

    def test_question_span_before_or_overlapping_video_rejected(self):
        with self.assertRaisesRegex(ValueError, "question_span_outside_text_after_video"):
            find_unique_question_span([4, 5, 9, 9], [4, 5], 2, 4)

    def test_temporal_summary_averages_query_tokens_after_bin_sums(self):
        raw = torch.tensor([[[[0., 0., .1, .1, .2, .2, .1, .1],
                              [0., 0., .3, .1, .1, .1, .1, .1],
                              [0., 0., .1, .1, .1, .3, .1, .1]]]])
        mass, entropy, peak = temporal_summary(
            raw, video_start=2, video_end=8, temporal_bins=3,
            query_start=0, query_end=2)
        self.assertEqual(mass.shape, (1,))
        self.assertEqual(entropy.shape, (1,))
        self.assertEqual(peak.shape, (1,))
        # First two query rows yield bin means [0.3, 0.3, 0.2].
        self.assertAlmostEqual(float(mass[0]), 0.8, places=6)
        self.assertEqual(int(peak[0]), 0)

    def test_spearman_average_ties(self):
        self.assertAlmostEqual(spearman_average_ties([1, 2, 2], [3, 1, 1]), -1.0)
        self.assertIsNone(spearman_average_ties([1], [1]))

    def test_summary_and_paired_comparison_are_descriptive(self):
        rng = np.random.default_rng(20261019)
        mass = rng.random((4, 28, 28), dtype=np.float32)
        entropy = rng.random((4, 28, 28), dtype=np.float32)
        peaks = rng.integers(0, 8, size=(4, 28, 28), dtype=np.uint8)
        ranked, elbow = summarize_all_heads(mass, entropy, peaks)
        self.assertGreaterEqual(elbow, 10)
        self.assertLessEqual(len(ranked), 784)
        result = compare_rankings((mass, entropy, peaks), (mass, entropy, peaks))
        self.assertEqual(result["comparison"]["top20_intersection"], 20)
        self.assertEqual(result["comparison"]["top20_jaccard"], 1.0)
        self.assertAlmostEqual(
            result["comparison"]["spearman_average_ties_on_shared_survivor_entropy"],
            1.0)

    def test_four_temporal_bins_are_supported_with_correct_entropy_bound(self):
        rng = np.random.default_rng(20261019)
        mass = rng.random((3, 28, 28), dtype=np.float32)
        entropy = rng.random((3, 28, 28), dtype=np.float32)
        peaks = rng.integers(0, 4, size=(3, 28, 28), dtype=np.uint8)
        result = compare_rankings((mass, entropy, peaks), (mass, entropy, peaks),
                                  temporal_bins=4)
        self.assertEqual(result["comparison"]["top20_intersection"], 20)
        self.assertEqual(result["comparison"]["top20_jaccard"], 1.0)

    def test_summary_rejects_entropy_outside_temporal_support(self):
        mass = np.full((2, 28, 28), 0.5)
        entropy = np.full((2, 28, 28), 1.5)
        peaks = np.zeros((2, 28, 28), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "metric_value_range_invalid"):
            summarize_all_heads(mass, entropy, peaks, temporal_bins=4)


if __name__ == "__main__":
    unittest.main()
