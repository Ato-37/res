import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_annotation_key_bias_confirmatory import analyze


class ConfirmatoryAnalysisTests(unittest.TestCase):
    def test_large_effect_is_meaningful(self):
        rng = random.Random(1)
        v = [0.1 + rng.gauss(0, 0.05) for _ in range(100)]
        r = analyze(v, 0.025, permutations=2000)
        self.assertEqual(r["verdict"], "practically_meaningful_effect")
        self.assertLess(r["p_two_sided_t"], 1e-6)
        self.assertLess(r["p_sign_flip"], 0.01)

    def test_tight_null_is_equivalent(self):
        rng = random.Random(2)
        v = [rng.gauss(0, 0.05) for _ in range(300)]
        r = analyze(v, 0.025, permutations=2000)
        self.assertEqual(r["verdict"], "no_meaningful_effect_equivalence")
        self.assertLess(r["tost_p"], 0.05)

    def test_wide_null_is_inconclusive(self):
        rng = random.Random(3)
        v = [rng.gauss(0, 0.3) for _ in range(20)]
        r = analyze(v, 0.025, permutations=2000)
        self.assertEqual(r["verdict"], "inconclusive")

    def test_small_significant_effect_is_below_threshold(self):
        rng = random.Random(4)
        v = [0.01 + rng.gauss(0, 0.005) for _ in range(200)]
        r = analyze(v, 0.025, permutations=2000)
        self.assertEqual(r["verdict"], "statistically_significant_below_threshold")

    def test_too_small_rejected(self):
        with self.assertRaises(ValueError):
            analyze([0.1, 0.2], 0.025)


if __name__ == "__main__":
    unittest.main()
