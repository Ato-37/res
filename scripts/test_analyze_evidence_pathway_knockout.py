import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_evidence_pathway_knockout import contrast_values, margin_delta


def rec(base_margin, evid, sham, excluded=False):
    if excluded:
        return {"excluded": True, "knockouts": []}
    arms = [{"arm": "v2t_evidence_band", "margin": base_margin + evid}, {"arm": "v2t_sham_band", "margin": base_margin + sham}]
    return {"excluded": False, "baseline_margin": base_margin, "baseline_correct": True, "knockouts": arms}


class ContrastTests(unittest.TestCase):
    def test_margin_delta_is_knockout_minus_baseline(self):
        self.assertAlmostEqual(margin_delta(rec(2.0, -1.5, -0.25), "v2t_evidence_band"), -1.5)

    def test_contrast_is_evidence_minus_sham_and_skips_excluded(self):
        vals = contrast_values([rec(2.0, -1.5, -0.25), rec(0, 0, 0, excluded=True), rec(1.0, -0.5, 0.5)], "v2t", "band")
        self.assertEqual([round(v, 6) for v in vals], [-1.25, -1.0])


if __name__ == "__main__":
    unittest.main()
