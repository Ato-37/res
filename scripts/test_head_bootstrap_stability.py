import sys
import unittest

import numpy as np

sys.path.insert(0, "scripts")
from analyze_head_bootstrap_stability import bootstrap_summary, sink_fractions
from run_head_bootstrap_exploration import summarize


class HeadBootstrapStabilityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.mass = rng.random((5, 28, 28), dtype=np.float32)
        self.entropy = rng.random((5, 28, 28), dtype=np.float32)
        self.peaks = rng.integers(0, 8, size=(5, 28, 28), dtype=np.uint8)

    def test_summary_returns_unique_top20_pairs(self):
        ranked, elbow, eligible = summarize(self.mass, self.entropy, self.peaks)
        pairs = [(row[0], row[1]) for row in ranked]
        self.assertLessEqual(len(pairs), 20)
        self.assertEqual(len(set(pairs)), len(pairs))
        self.assertGreaterEqual(elbow, 10)
        self.assertGreaterEqual(eligible, len(pairs))

    def test_bootstrap_is_reproducible_for_fixed_seed(self):
        groups = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
        a = bootstrap_summary(self.mass, self.entropy, self.peaks, groups, replicates=12, seed=123)
        b = bootstrap_summary(self.mass, self.entropy, self.peaks, groups, replicates=12, seed=123)
        self.assertEqual(a, b)
        self.assertEqual(a["sample_count"], 5)
        for row in a["top20_selection_frequency"]:
            self.assertGreaterEqual(row["top20_selection_frequency"], 0.0)
            self.assertLessEqual(row["top20_selection_frequency"], 1.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "metric_array_shape_mismatch"):
            bootstrap_summary(self.mass, self.entropy[:-1], self.peaks,
                              np.arange(5, dtype=np.int32), replicates=2)

    def test_rejects_advertised_but_unsupported_four_dimensional_input(self):
        with self.assertRaisesRegex(ValueError, r"expected_\[n,28,28\]_arrays"):
            bootstrap_summary(self.mass[..., None], self.entropy[..., None], self.peaks[..., None],
                              np.arange(5, dtype=np.int32), replicates=2)

    def test_duplicate_media_rows_share_a_bootstrap_cluster(self):
        result = bootstrap_summary(self.mass, self.entropy, self.peaks,
                                   np.asarray([4, 4, 7, 8, 9], dtype=np.int32),
                                   replicates=12, seed=123)
        self.assertEqual(result["unique_media_clusters"], 4)

    def test_sink_fraction_uses_resampled_row_count(self):
        resampled = np.zeros((2, 28, 28), dtype=np.uint8)
        self.assertTrue(np.all(sink_fractions(resampled) == 1.0))


if __name__ == "__main__":
    unittest.main()
