import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_progressive_knockout import criteria, curve


def rec(vals):
    return {"baseline_correct": True, "knockouts": [{"edge": "video_to_text", "direction": d, "n": n, "relative_change_percent": v} for (d, n), v in vals.items()]}


class ProgressiveTests(unittest.TestCase):
    def test_curve_mean_and_criteria(self):
        recs = [rec({("forward", 28): -80.0, ("forward", 20): -78.0, ("reverse", 24): -2.0, ("forward", 4): -1.0}),
                rec({("forward", 28): -60.0, ("forward", 20): -58.0, ("reverse", 24): -4.0, ("forward", 4): -3.0}),
                {"baseline_correct": False, "knockouts": []}]
        c = curve(recs, draws=200)
        self.assertAlmostEqual(c["video_to_text|forward|28"]["mean_percent"], -70.0)
        self.assertEqual(c["video_to_text|forward|28"]["n"], 2)
        self.assertEqual(set(criteria(c).values()), {True})

    def test_criteria_flags_failures(self):
        recs = [rec({("forward", 28): -10.0, ("forward", 20): -40.0, ("reverse", 24): -30.0, ("forward", 4): -20.0})]
        self.assertEqual(set(criteria(curve(recs, draws=50)).values()), {False})


if __name__ == "__main__":
    unittest.main()
