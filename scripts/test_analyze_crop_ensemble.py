import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_crop_ensemble import compare, mcnemar_exact


class CropEnsembleTests(unittest.TestCase):
    def test_mcnemar_exact_known_values(self):
        self.assertAlmostEqual(mcnemar_exact(19, 7), 0.0290, places=3)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertEqual(mcnemar_exact(5, 5), 1.0)

    def test_compare_support_rule(self):
        base = np.array([1] * 60 + [0] * 40, dtype=bool)
        ens = base.copy(); ens[60:75] = True  # +15 videos fixed, none broken
        r = compare(ens, base)
        self.assertTrue(r["supported"])
        self.assertEqual((r["wrong_to_correct"], r["correct_to_wrong"]), (15, 0))
        ens2 = base.copy(); ens2[60] = True
        self.assertFalse(compare(ens2, base)["supported"])


if __name__ == "__main__":
    unittest.main()
