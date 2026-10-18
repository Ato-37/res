from __future__ import annotations

import unittest

from plan_qta_paired_sample_size import (
    exact_sign_p_value,
    exact_sign_test_power,
    paired_t_power,
    required_n,
    required_sign_test_n,
    sensitivity_grid,
    sign_test_sensitivity,
)


class QtaPairedSampleSizeTests(unittest.TestCase):
    def test_zero_effect_has_size_alpha(self):
        self.assertAlmostEqual(paired_t_power(12, 0.0, 0.4), 0.05, places=12)

    def test_power_increases_with_sample_size_and_effect(self):
        self.assertLess(paired_t_power(8, 0.2, 0.4), paired_t_power(20, 0.2, 0.4))
        self.assertLess(paired_t_power(20, 0.2, 0.4), paired_t_power(20, 0.4, 0.4))

    def test_required_n_is_minimal_and_hits_target(self):
        n = required_n(0.25, 0.4, 0.8)
        self.assertIsNotNone(n)
        assert n is not None
        self.assertGreaterEqual(paired_t_power(n, 0.25, 0.4), 0.8)
        self.assertLess(paired_t_power(n - 1, 0.25, 0.4), 0.8)

    def test_n3_sensitivity_is_explicitly_assumption_only(self):
        row = sensitivity_grid([0.25], [0.4])[0]
        self.assertEqual(row["unit"], "independent_source_video_cluster_mean_of_paired_deltaF")
        self.assertEqual(row["scope"], "assumption_sensitivity_not_empirical_or_preregistered")
        self.assertAlmostEqual(row["power_at_n3"], paired_t_power(3, 0.25, 0.4))

    def test_invalid_inputs_fail_closed(self):
        for args, code in [
            ((1, 0.1, 0.2), "n_must_be_integer_at_least_2"),
            ((3, 0.1, 0.0), "sd_must_be_positive_finite"),
            ((3, 0.1, 0.2, 1.0), "alpha_must_be_between_zero_and_one"),
        ]:
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, code):
                paired_t_power(*args)

    def test_required_power_must_exceed_alpha(self):
        with self.assertRaisesRegex(ValueError, "target_power_must_be_greater"):
            required_n(0.0, 0.4, 0.01, alpha=0.05)

    def test_scenario_generators_are_fully_consumed(self):
        rows = sensitivity_grid((x for x in (0.1, 0.2)),
                                (x for x in (0.2, 0.4)),
                                (x for x in (0.8, 0.9)))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(set(row["required_n_by_power"]) == {"0.8", "0.9"}
                            for row in rows))

    def test_empty_scenarios_are_rejected_even_without_calculation(self):
        with self.assertRaisesRegex(ValueError, "scenario_iterables_must_be_nonempty"):
            sensitivity_grid([], [0.4])
        with self.assertRaisesRegex(ValueError, "target_power_must_be_greater"):
            sensitivity_grid([0.2], [0.4], [0.01], alpha=0.05)

    def test_three_video_two_sided_sign_test_cannot_reject(self):
        self.assertEqual(exact_sign_p_value(3, 0), 0.25)
        self.assertEqual(exact_sign_test_power(3, 0.9), 0.0)
        self.assertEqual(exact_sign_p_value(6, 0), 0.03125)

    def test_sign_test_power_and_required_n_match_exact_enumeration(self):
        self.assertAlmostEqual(exact_sign_test_power(6, 0.8), 0.262208, places=12)
        n = required_sign_test_n(0.8, 0.8)
        self.assertEqual(n, 20)
        self.assertGreaterEqual(exact_sign_test_power(n, 0.8), 0.8)
        self.assertLess(exact_sign_test_power(n - 1, 0.8), 0.8)

    def test_sign_sensitivity_labels_scenarios_and_not_mean_deltaf(self):
        rows = sign_test_sensitivity((p for p in (0.7, 0.8)))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["unit"], "independent_source_video_level_deltaF_sign")
        self.assertEqual(rows[1]["scope"],
                         "directional_robustness_sensitivity_not_primary_mean_deltaF_test")
        self.assertTrue(rows[1]["assumptions_are_scenarios_not_empirical_estimates"])
        self.assertEqual(rows[1]["n_counts"], "nonzero_video_level_deltaF_signs")
        self.assertTrue(rows[1]["positive_sign_probability_is_conditional_on_nonzero_deltaF"])
        self.assertFalse(rows[1]["zero_effect_rate_modeled"])

    def test_sign_functions_reject_bad_counts_and_probabilities(self):
        for args, code in [
            ((0, 0), "nonzero_sign_count_must_be_positive"),
            ((-1, 2), "sign_counts_must_be_nonnegative_integers"),
            ((1.0, 2), "sign_counts_must_be_nonnegative_integers"),
        ]:
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, code):
                exact_sign_p_value(*args)
        with self.assertRaisesRegex(ValueError, "probability_positive_must_be_between"):
            exact_sign_test_power(3, 1.1)

    def test_sign_required_n_uses_smallest_attainable_sample_size(self):
        self.assertEqual(required_sign_test_n(0.9, 0.8), 12)
        self.assertEqual(required_sign_test_n(0.7, 0.8), 49)

    def test_sign_sensitivity_rejects_invalid_scenarios(self):
        with self.assertRaisesRegex(ValueError, "scenario_iterables_must_be_nonempty"):
            sign_test_sensitivity([], [0.8])
        with self.assertRaisesRegex(ValueError, "probability_positive_must_be_between"):
            sign_test_sensitivity([1.1])
        with self.assertRaisesRegex(ValueError, "target_power_must_be_greater"):
            sign_test_sensitivity([0.8], [0.05])


if __name__ == "__main__":
    unittest.main()
