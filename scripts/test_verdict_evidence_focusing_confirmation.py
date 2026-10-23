import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verdict_evidence_focusing_confirmation import verdict


def row(mean, pt, pf, pw, tost):
    return {"mean_nats": mean, "p_two_sided_t": pt, "p_sign_flip": pf, "p_wilcoxon": pw, "tost_p": tost}


class VerdictTests(unittest.TestCase):
    def test_confirmed_needs_all_three_tests_and_threshold(self):
        self.assertEqual(verdict(row(0.5, 0.01, 0.01, 0.02, 0.9)), "confirmed")
        self.assertEqual(verdict(row(0.5, 0.01, 0.01, 0.08, 0.9)), "not_confirmed_inconclusive")
        self.assertEqual(verdict(row(0.2, 0.001, 0.001, 0.001, 0.9)), "not_confirmed_inconclusive")

    def test_equivalence_when_tost_rejects(self):
        self.assertEqual(verdict(row(0.02, 0.8, 0.8, 0.7, 0.001)), "equivalent_within_delta_min")


if __name__ == "__main__":
    unittest.main()
