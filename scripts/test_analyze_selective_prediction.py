import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_selective_prediction import acc_at_coverage, aurc, auroc, paired_bootstrap


class SelectivePredictionTests(unittest.TestCase):
    def test_accuracy_at_coverage_prefers_confident(self):
        correct = np.array([1, 1, 0, 0])
        conf = np.array([0.9, 0.8, 0.7, 0.6])
        self.assertEqual(acc_at_coverage(correct, conf, 0.5), 1.0)
        self.assertEqual(acc_at_coverage(correct, conf, 1.0), 0.5)
        self.assertEqual(acc_at_coverage(correct, conf[::-1], 0.5), 0.0)

    def test_aurc_and_auroc_extremes(self):
        correct = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(aurc(correct, np.array([4, 3, 2, 1.0])), (0 + 0 + 1 / 3 + 0.5) / 4)
        self.assertEqual(auroc(np.array([4, 3, 2, 1.0]), correct), 1.0)
        self.assertEqual(auroc(np.array([1, 2, 3, 4.0]), correct), 0.0)

    def test_bootstrap_detects_clear_gain_and_not_identical_models(self):
        rng = np.random.default_rng(0)
        n = 300
        gold = rng.integers(0, 5, n)
        base = rng.dirichlet(np.ones(5), n)
        same = {"base": base, "ens": base.copy()}
        r = paired_bootstrap(gold, same, 0.8, draws=500)
        self.assertEqual(r["difference"], 0.0)
        self.assertFalse(r["supported"])
        good = np.zeros((n, 5)); good[np.arange(n), gold] = 0.9; good += 0.025
        better = {"base": base, "ens": good}
        self.assertTrue(paired_bootstrap(gold, better, 0.8, draws=500)["supported"])


if __name__ == "__main__":
    unittest.main()
