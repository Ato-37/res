"""CPU regression tests: python -m unittest discover -s scripts -p 'test_analyze_activation_patching.py'."""

import json
import unittest

from analyze_activation_patching import analyze, summarize


class AnalysisTests(unittest.TestCase):
    def test_ratio_does_not_average_tiny_denominators(self):
        stats = summarize(["a", "b"], [1.0, -0.1], [2.0, 0.001], n_boot=100)
        self.assertAlmostEqual(stats["aggregate_recovery"], 0.9 / 2.001)
        self.assertGreater(stats["aggregate_recovery"], 0)
        self.assertLess((1.0 / 2.0 - 0.1 / 0.001) / 2, 0)

    def test_zero_singleton_constant(self):
        self.assertIsNone(summarize([], [])["mean"])
        self.assertIsNone(summarize(["a"], [0])["mean_ci95"])
        stats = summarize(["a", "b"], [0, 0], n_boot=100)
        self.assertEqual(stats["mean_ci95"], [0, 0])
        json.dumps(stats, allow_nan=False)

    def test_influence_and_reproducibility(self):
        stats = summarize(["a", "b", "c"], [1, 2, 9], n_boot=100)
        self.assertEqual(stats["largest_abs_sample_id"], "c")
        self.assertEqual(stats["mean_without_largest_abs"], 1.5)
        self.assertEqual(stats["leave_one_out_mean_range"], [1.5, 5.5])
        self.assertEqual(stats, summarize(["a", "b", "c"], [1, 2, 9], n_boot=100))

    def test_no_harmed_subset_and_outside_grounding(self):
        record = dict(sample_id="a", swap_drop=1,
                      layers={"27": {"patch_rescue": 0.2}},
                      sampled_indices=[0, 10], sampled_timestamps=[0, 1],
                      grounding=[[0.4, 0.6]], evidence_indices=[0])
        report = analyze(dict(records=[record], n_effective=1), n_boot=100)
        self.assertEqual(report["n_harmed"], 0)
        self.assertEqual(report["outside_grounding_evidence_ids"], ["a"])
        self.assertIsNone(report["rows"][1]["mean"])
        json.dumps(report, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyze(dict(records=[record, record]), n_boot=100)

    def test_incomplete_or_nonfinite_results_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            analyze(dict(layers=[6, 27], records=[dict(
                sample_id="a", swap_drop=-1, layers={"27": {"patch_rescue": 0}}
            )]), n_boot=100)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            summarize(["a"], [float("nan")])
        with self.assertRaisesRegex(ValueError, "positive damage"):
            summarize(["a"], [1], damage=[0])


if __name__ == "__main__":
    unittest.main()
