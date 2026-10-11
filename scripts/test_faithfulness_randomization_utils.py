"""CPU tests for attribution-ranking randomization diagnostics."""

import unittest

from faithfulness_randomization_utils import compare_attribution_rankings


class RandomizationUtilsTests(unittest.TestCase):
    def test_identical_rankings_have_full_agreement(self):
        summary = compare_attribution_rankings([4, 3, 2, 1], [4, 3, 2, 1], top_k=2)
        self.assertAlmostEqual(summary['spearman_rho'], 1.0)
        self.assertEqual(summary['top_k_overlap_count'], 2)

    def test_reversed_ranking_has_negative_rank_correlation(self):
        summary = compare_attribution_rankings([4, 3, 2, 1], [1, 2, 3, 4], top_k=1)
        self.assertAlmostEqual(summary['spearman_rho'], -1.0)
        self.assertEqual(summary['top_k_overlap_count'], 0)

    def test_rejects_invalid_rankings(self):
        for first, second in (([], []), ([1], [1, 2]), ([float('nan')], [0])):
            with self.subTest(first=first, second=second), self.assertRaises(ValueError):
                compare_attribution_rankings(first, second)


if __name__ == '__main__':
    unittest.main()
