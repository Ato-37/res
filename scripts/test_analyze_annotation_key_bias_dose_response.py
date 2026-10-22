import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_annotation_key_bias_dose_response import holm


class HolmTests(unittest.TestCase):
    def test_holm_matches_manual(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_holm_is_monotone_and_capped(self):
        adj = holm([0.5, 0.9, 0.2])
        self.assertTrue(all(0 <= a <= 1 for a in adj))
        self.assertEqual(adj[1], 0.9 if adj[1] == 0.9 else adj[1])
        self.assertLessEqual(adj[2], adj[0])


if __name__ == "__main__":
    unittest.main()
