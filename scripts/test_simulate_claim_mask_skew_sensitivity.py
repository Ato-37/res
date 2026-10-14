import unittest
import warnings

from simulate_claim_mask_skew_sensitivity import build_results, simulate_condition


class TestClaimMaskSkewSensitivity(unittest.TestCase):
    def test_result_is_reproducible_and_explicitly_hypothetical(self):
        first = simulate_condition(20, 0.2, 5000, 123, batch_size=257)
        second = simulate_condition(20, 0.2, 5000, 123, batch_size=257)
        self.assertEqual(first, second)
        self.assertEqual(first["distribution"],
                         "centered_unit_variance_exponential_skewness_2")
        self.assertFalse(build_results([20], [0.0], 100, 2)["empirical_data_used"])

    def test_boundary_and_hypothetical_power_are_finite(self):
        result = simulate_condition(20, 0.0, 20000, 456)
        self.assertGreaterEqual(result["success_probability"], 0.0)
        self.assertLessEqual(result["success_probability"], 1.0)
        self.assertAlmostEqual(result["normal_noncentral_t_reference"], 0.025, places=10)
        self.assertGreater(result["monte_carlo_se"], 0.0)

    def test_invalid_inputs_fail_closed(self):
        for args in ((1, 0.2, 10, 1), (20, -0.1, 10, 1),
                     (20, float("nan"), 10, 1), (20, 0.2, 0, 1),
                     (20, 10**1000, 10, 1), (20, 0.2, 10, -1),
                     (20, 0.2, 10, 1, float("nan")),
                     (20, 0.2, 10, 1, 10**1000)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                simulate_condition(*args)

    def test_numerically_unstable_reference_fails_before_simulation_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "numerically unstable"):
                simulate_condition(20, 1e100, 1000, 1)


if __name__ == "__main__":
    unittest.main()
