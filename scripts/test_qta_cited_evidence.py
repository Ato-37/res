import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qta_cited_evidence import cited_groups, parse_citation


class CitedEvidenceTests(unittest.TestCase):
    def test_parse_variants(self):
        p = parse_citation("Answer: C. Time: 3.0-7.5 seconds")
        self.assertEqual((p["ok"], p["letter"], p["start"], p["end"]), (True, "C", 3.0, 7.5))
        p = parse_citation("Answer: B\nEvidence: 12 to 15")
        self.assertEqual((p["ok"], p["letter"], p["start"], p["end"]), (True, "B", 12.0, 15.0))
        self.assertFalse(parse_citation("I think it is A")["ok"])
        self.assertEqual(parse_citation("Answer: A. Time: 9-2")["reason"], "invalid_span")
        self.assertFalse(parse_citation("")["ok"])

    def test_cited_groups_overlap_rule(self):
        times = [i * 0.5 for i in range(32)]  # 0..15.5 s, groups of 2 frames: group g spans [g, g+0.5]
        self.assertEqual(cited_groups(times, 2.0, 4.2), (2, 3, 4))
        self.assertEqual(cited_groups(times, 0.0, 0.0), (0,))
        self.assertEqual(cited_groups(times, 100.0, 120.0), ())
        self.assertEqual(cited_groups(times, 15.0, 99.0), (15,))
        with self.assertRaises(ValueError):
            cited_groups([0.0, 1.0, 2.0], 0, 1)


if __name__ == "__main__":
    unittest.main()
