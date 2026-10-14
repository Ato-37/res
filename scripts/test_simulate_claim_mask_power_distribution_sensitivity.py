import math
import unittest

from simulate_claim_mask_power_distribution_sensitivity import (
    build_results,
    simulate_condition,
)


class TestDistributionalPowerSensitivity(unittest.TestCase):
    def test_simulation_is_seed_reproducible_and_labels_hypotheses(self):
        args = ([20], [0.0, 0.2], 3.0, 20_000, 42, 0.95, 1000)
        first = build_results(*args)
        second = build_results(*args)
        self.assertEqual(first, second)
        self.assertFalse(first["empirical_data_used"])
        self.assertFalse(first["sesoi_selected"])
        self.assertEqual(len(first["rows"]), 2)
        self.assertGreaterEqual(first["rows"][0]["upper_tail_probability"], 0)
        self.assertLessEqual(first["rows"][0]["upper_tail_probability"], 1)

    def test_unit_variance_t_distribution_requires_df_above_two(self):
        for df in (2.0, 1.5, float("nan")):
            with self.subTest(df=df), self.assertRaises(ValueError):
                simulate_condition(20, 0.2, df, 100, 1)

    def test_zero_excess_null_calibration_and_reported_monte_carlo_errors(self):
        result = simulate_condition(20, 0.0, 3.0, 50_000, 20261014, batch_size=5000)
        self.assertLess(abs(result["upper_tail_probability"] - 0.025), 0.005)
        self.assertLess(abs(result["two_sided_rejection_probability"] - 0.05), 0.01)
        self.assertAlmostEqual(
            result["upper_tail_monte_carlo_se"],
            math.sqrt(result["upper_tail_probability"] * (1 - result["upper_tail_probability"]) / 50_000),
        )
        self.assertAlmostEqual(
            result["two_sided_monte_carlo_se"],
            math.sqrt(result["two_sided_rejection_probability"] * (1 - result["two_sided_rejection_probability"]) / 50_000),
        )

    def test_bad_repetition_and_sample_sizes_fail(self):
        for kwargs in (
            {"n": 1, "excess": 0.2, "df": 3, "repetitions": 100, "seed": 1},
            {"n": 20, "excess": -0.2, "df": 3, "repetitions": 100, "seed": 1},
            {"n": 20, "excess": 0.2, "df": 3, "repetitions": 0, "seed": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_condition(**kwargs)


if __name__ == "__main__":
    unittest.main()
