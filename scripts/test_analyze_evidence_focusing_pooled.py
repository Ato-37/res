import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_evidence_focusing_pooled import decide, values
from analyze_annotation_key_bias_confirmatory import analyze


class PooledTests(unittest.TestCase):
    def test_values_reads_records_or_rows(self):
        self.assertEqual(values({"records": [{"primary_annotation_minus_sham_nats": 1.5}]}), [1.5])
        self.assertEqual(values({"rows": [{"primary_annotation_minus_sham_nats": 2.5}]}), [2.5])

    def test_decide_confirmed_for_large_consistent_effect(self):
        rng = random.Random(7)
        res = analyze([0.8 + rng.gauss(0, 1.0) for _ in range(300)], 0.3, permutations=2000)
        self.assertEqual(decide(res), "confirmed")

    def test_decide_equivalent_for_tight_small_effect(self):
        rng = random.Random(8)
        res = analyze([0.05 + rng.gauss(0, 0.4) for _ in range(100)], 0.3, permutations=2000)
        self.assertEqual(decide(res), "equivalent_within_delta_min")

    def test_decide_inconclusive_for_noisy_small_sample(self):
        rng = random.Random(9)
        res = analyze([0.1 + rng.gauss(0, 3.0) for _ in range(30)], 0.3, permutations=2000)
        self.assertEqual(decide(res), "not_confirmed_inconclusive")


if __name__ == "__main__":
    unittest.main()
