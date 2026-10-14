import unittest

from analyze_claim_mask_power_sensitivity import (
    build_grid,
    minimum_n_for_target_power,
    paired_t_lower_ci_power,
)


class TestClaimMaskPowerSensitivity(unittest.TestCase):
    def test_boundary_power_is_one_sided_alpha_of_two_sided_interval(self):
        self.assertAlmostEqual(paired_t_lower_ci_power(40, 0.0, 0.95), 0.025, places=10)

    def test_power_increases_with_n_and_distance_above_sesoi(self):
        self.assertLess(paired_t_lower_ci_power(20, 0.2), paired_t_lower_ci_power(80, 0.2))
        self.assertLess(paired_t_lower_ci_power(40, 0.1), paired_t_lower_ci_power(40, 0.4))

    def test_grid_labels_assumptions_and_does_not_choose_sesoi(self):
        result = build_grid([20, 80], [0.1, 0.5])
        self.assertEqual(len(result["rows"]), 4)
        self.assertTrue(result["sesoi_selected"] is False)
        self.assertTrue(result["power_inputs_are_hypothetical"])
        self.assertTrue(result["empirical_data_used"] is False)
        self.assertIn("scipy", result["software"])
        self.assertIn("python", result["software"])
        self.assertEqual(result["minimum_n_by_hypothetical_excess"][1]["minimum_effective_video_n"],
                         minimum_n_for_target_power(0.5))

    def test_minimum_n_meets_target_and_zero_gap_does_not(self):
        n = minimum_n_for_target_power(0.5)
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(paired_t_lower_ci_power(n, 0.5), 0.8)
        self.assertLess(paired_t_lower_ci_power(n - 1, 0.5), 0.8)
        self.assertIsNone(minimum_n_for_target_power(0.0, max_n=100))

    def test_invalid_inputs_fail(self):
        for args in ((1, 0.2), (20, -0.1), (20, float("nan")), (20, 10**1000),
                     (20, "0.2"), (20, 0.2, 1.0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                paired_t_lower_ci_power(*args)


if __name__ == "__main__":
    unittest.main()
