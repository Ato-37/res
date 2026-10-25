import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qta_overlap_dose import overlap_sets


class OverlapDoseTests(unittest.TestCase):
    def test_sets_have_fixed_size_and_graded_overlap(self):
        ann, sham = (4, 5, 6, 7), (0, 1, 12, 15)
        sets = overlap_sets(ann, sham, 16, "tok")
        self.assertEqual([s["overlap"] for s in sets], [0, 1, 2, 3, 4])
        for s in sets:
            self.assertEqual(len(s["groups"]), 4)
            self.assertEqual(len(set(s["groups"]) & set(ann)), s["overlap"])
        self.assertEqual(sets[0]["groups"], tuple(sorted(sham)))
        self.assertEqual(sets[-1]["groups"], ann)

    def test_small_budget_dedups_levels(self):
        sets = overlap_sets((3,), (9,), 16, "tok")
        self.assertEqual([s["overlap"] for s in sets], [0, 1])
        sets2 = overlap_sets((3, 4), (9, 10), 16, "tok")
        self.assertEqual([s["overlap"] for s in sets2], [0, 1, 2])

    def test_deterministic_and_invalid_geometry(self):
        a = overlap_sets((4, 5, 6, 7), (0, 1, 12, 15), 16, "tok")
        b = overlap_sets((4, 5, 6, 7), (0, 1, 12, 15), 16, "tok")
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            overlap_sets(tuple(range(9)), tuple(range(9, 16)), 16, "tok")
        with self.assertRaises(ValueError):
            overlap_sets((4, 5), (5, 6), 16, "tok")


if __name__ == "__main__":
    unittest.main()
