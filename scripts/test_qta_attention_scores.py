import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qta_attention_scores as sc


class ScoreTests(unittest.TestCase):
    def test_group_mass_sums_attention_over_group_keys(self):
        attn = torch.zeros(1, 2, 5, 6)
        attn[0, :, 3, 0:2] = 0.25   # row 3 puts 0.5 on keys 0,1
        attn[0, :, 4, 2:4] = 0.5    # row 4 puts 1.0 on keys 2,3
        mass = sc.group_mass(attn, rows=[3, 4], groups=[(0, 1), (2, 3), (4, 5)], torch_module=torch)
        self.assertTrue(torch.allclose(mass, torch.tensor([0.25, 0.5, 0.0])))

    def test_group_mass_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            sc.group_mass(torch.zeros(2, 2, 3, 3), [0], [(0,)], torch)

    def test_top_groups_is_sorted_and_breaks_ties_low(self):
        self.assertEqual(sc.top_groups([0.1, 0.9, 0.9, 0.3], 2), (1, 2))
        self.assertEqual(sc.top_groups([0.5, 0.5, 0.5], 2), (0, 1))
        with self.assertRaises(ValueError):
            sc.top_groups([1, 2], 3)

    def test_hash_random_groups_deterministic_and_token_dependent(self):
        a = sc.hash_random_groups(4, 16, "token-a")
        self.assertEqual(a, sc.hash_random_groups(4, 16, "token-a"))
        self.assertEqual(len(set(a)), 4)
        self.assertNotEqual(a, sc.hash_random_groups(4, 16, "token-b"))

    def test_overlap_with_chance(self):
        r = sc.overlap_with_chance((1, 2, 3), (2, 3, 4, 5), 16)
        self.assertEqual(r["overlap"], 2)
        self.assertAlmostEqual(r["expected_by_chance"], 3 * 4 / 16)


if __name__ == "__main__":
    unittest.main()
