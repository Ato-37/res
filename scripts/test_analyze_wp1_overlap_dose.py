import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_wp1_overlap_dose import decide, dose_points, slope


def rec(base, pairs):
    return {"baseline_margin": base, "knockouts": [{"arm": f"d{i}", "overlap_fraction": f, "margin": m} for i, (f, m) in enumerate(pairs)]}


class WP1DTests(unittest.TestCase):
    def test_slope_recovers_linear_dose(self):
        r = rec(1.0, [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)])
        self.assertAlmostEqual(slope(dose_points(r)), -1.0)

    def test_slope_undefined_with_one_level_and_flat_is_zero(self):
        self.assertIsNone(slope([(0.5, 1.0)]))
        self.assertAlmostEqual(slope([(0.0, -0.2), (1.0, -0.2)]), 0.0)

    def test_decide_rules(self):
        res = {"p_two_sided_t": 0.01, "p_sign_flip": 0.01, "p_wilcoxon": 0.02, "mean_nats": 0.5, "tost_p": 0.9}
        self.assertEqual(decide(res), "supported")
        self.assertEqual(decide(dict(res, mean_nats=0.2)), "inconclusive")
        self.assertEqual(decide(dict(res, mean_nats=0.1, p_two_sided_t=0.6, tost_p=0.01)), "equivalent_within_delta_min")


if __name__ == "__main__":
    unittest.main()
