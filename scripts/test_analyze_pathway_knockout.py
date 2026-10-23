import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pathway_knockout import EDGES, criteria, profile

CENTERS = (2, 5, 8, 11, 14, 17, 20, 23, 26)


def record(correct, shape):
    ks = []
    for edge in EDGES:
        for c in CENTERS:
            ks.append({"edge": edge, "center": c, "relative_change_percent": shape(edge, c), "predicted_after": 0})
    return {"baseline_correct": correct, "predicted": 0, "knockouts": ks if correct else []}


def paper_like(edge, c):
    return {"cross_frame": -40.0 if c <= 14 else -1.0, "video_to_text": -60.0 if 8 <= c <= 17 else -2.0,
            "video_to_last": -1.0, "text_to_last": -70.0 if c >= 17 else -2.0}[edge]


class CriteriaTests(unittest.TestCase):
    def _records(self, shape, n=6):
        # tiny noise so the t interval is defined
        return [record(True, lambda e, c, i=i: shape(e, c) + 0.1 * i) for i in range(n)] + [record(False, shape)]

    def test_paper_like_profile_replicates(self):
        prof = profile(self._records(paper_like))
        crit = criteria(prof)
        self.assertEqual(crit["outcome"], "replicates")
        self.assertEqual(crit["criteria_met"], 4)

    def test_flat_profile_does_not_replicate(self):
        prof = profile(self._records(lambda e, c: -0.5))
        self.assertEqual(criteria(prof)["outcome"], "does_not_replicate")

    def test_incorrect_baselines_are_excluded(self):
        prof = profile(self._records(paper_like))
        self.assertEqual(prof["video_to_text"][11]["n"], 6)


if __name__ == "__main__":
    unittest.main()
