import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_wp2_cited_evidence import arm_change, decide


class WP2Tests(unittest.TestCase):
    def test_arm_change_uses_self_margin(self):
        r = {"baseline_self_margin": 2.0, "knockouts": [{"arm": "a", "self_margin": 0.5}, {"arm": "b", "self_margin": 1.5}]}
        self.assertEqual(arm_change(r, "a"), -1.5)
        self.assertEqual(arm_change(r, "b"), -0.5)

    def test_decide(self):
        res = {"p_two_sided_t": 0.01, "p_sign_flip": 0.01, "p_wilcoxon": 0.02, "mean_nats": 0.4, "tost_p": 0.9}
        self.assertEqual(decide(res), "supported")
        self.assertEqual(decide(dict(res, mean_nats=0.1, p_two_sided_t=0.5, tost_p=0.02)), "equivalent_within_delta_min")


if __name__ == "__main__":
    unittest.main()
