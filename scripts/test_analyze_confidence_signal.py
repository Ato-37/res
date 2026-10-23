import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_confidence_signal as acs


def synthetic(n, informative, seed):
    rng = random.Random(seed)
    recs = []
    for _ in range(n):
        conc = rng.random()                      # 0 = diffuse, 1 = concentrated attention
        p_correct = 0.35 + 0.55 * conc if informative else 0.65
        correct = rng.random() < p_correct
        lp = [rng.gauss(0, 1) for _ in range(5)]
        lp[0] += 1.5                              # confidence unrelated to correctness
        gs = [0.05 + (conc * 2.0 if i == 0 else 0.0) + rng.random() * 0.05 for i in range(16)]
        recs.append({"baseline_choice_logprobs": lp, "group_scores": gs, "baseline_correct": correct})
    return recs


class ConfidenceSignalTests(unittest.TestCase):
    def test_informative_concentration_improves_brier(self):
        X, y, lp = acs.build(synthetic(300, True, 1))
        oof = acs.cv_oof(X, y, lp, repeats=3)
        d, ci = acs.paired_bootstrap((oof["logistic_conf_plus_concentration"] - y) ** 2, (oof["logistic_conf"] - y) ** 2, reps=500)
        self.assertLess(ci[1], 0)

    def test_uninformative_concentration_gives_no_reliable_gain(self):
        X, y, lp = acs.build(synthetic(300, False, 2))
        oof = acs.cv_oof(X, y, lp, repeats=3)
        d, ci = acs.paired_bootstrap((oof["logistic_conf_plus_concentration"] - y) ** 2, (oof["logistic_conf"] - y) ** 2, reps=500)
        self.assertGreaterEqual(ci[1], 0)

    def test_metrics(self):
        y = np.array([1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(acs.auroc(np.array([0.9, 0.1, 0.8, 0.2]), y), 1.0)
        self.assertAlmostEqual(acs.ece(np.array([1.0, 0.0, 1.0, 0.0]), y), 0.0)


if __name__ == "__main__":
    unittest.main()
