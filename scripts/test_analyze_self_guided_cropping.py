import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_self_guided_cropping import contrast, decide, versus_baseline


def rec(base, **margins):
    return {"baseline_margin": base, "arms": {k: {"margin": v} for k, v in margins.items()}}


class SgcTests(unittest.TestCase):
    def test_contrast_and_baseline_difference(self):
        records = [rec(1.0, selfguided=2.0, random=0.5), rec(0.0, selfguided=-1.0, random=0.5)]
        self.assertEqual(contrast(records, "selfguided", "random"), [1.5, -1.5])
        self.assertEqual(versus_baseline(records, "selfguided"), [1.0, -1.0])

    def test_decide_requires_threshold_all_tests_and_holm(self):
        res = {"p_two_sided_t": 0.01, "p_sign_flip": 0.01, "p_wilcoxon": 0.02, "mean_nats": 0.5, "tost_p": 0.9}
        self.assertEqual(decide(res, 0.02), "supported")
        self.assertEqual(decide(res, 0.08), "inconclusive")
        res2 = dict(res, mean_nats=0.2)
        self.assertEqual(decide(res2, 0.02), "inconclusive")
        self.assertEqual(decide(dict(res2, tost_p=0.01, p_two_sided_t=0.6), 0.9), "equivalent_within_delta_min")


if __name__ == "__main__":
    unittest.main()
