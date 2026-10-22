import unittest

from scripts.audit_qta_v4_union_arithmetic_properties import audit


class QtaV4UnionArithmeticPropertyTests(unittest.TestCase):
    def test_seeded_property_audit_matches_independent_reference(self):
        result = audit(draws=1000, seed=20261018)
        self.assertEqual(result["status"], "synthetic_arithmetic_properties_passed")
        self.assertEqual(result["synthetic_draws"], 1000)
        self.assertLessEqual(result["max_abs_difference_from_independent_reference_nat"], 1e-12)
        self.assertGreater(result["draws_where_treatment_top_two_would_change_if_recomputed"], 0)
        self.assertFalse(result["faithfulness_result"])
        self.assertEqual(result["model_forwards"], 0)

    def test_invalid_draw_counts_fail_closed(self):
        for draws in (0, -1, True, 1.5):
            with self.subTest(draws=draws), self.assertRaisesRegex(
                    ValueError, "draws_must_be_positive_integer"):
                audit(draws=draws)


if __name__ == "__main__":
    unittest.main()
