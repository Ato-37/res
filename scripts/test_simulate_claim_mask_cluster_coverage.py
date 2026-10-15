import math
import unittest

from simulate_claim_mask_cluster_coverage import (
    LAYOUTS,
    simulate_condition,
    simulate_grid,
)


class ClusterCoverageSimulationTests(unittest.TestCase):
    def test_iid_baseline_has_nominal_coverage_with_mc_uncertainty(self):
        result = simulate_condition([8] * 10, 0.0, repetitions=20_000,
                                    seed=20261015, batch_size=2_000)
        self.assertLessEqual(abs(result["empirical_coverage"] - 0.95),
                             4 * result["coverage_monte_carlo_se"])
        self.assertAlmostEqual(result["analytic_design_effect"], 1.0)
        self.assertAlmostEqual(result["analytic_sampling_sd"], 1 / math.sqrt(80))
        low, high = result["coverage_wilson_95_interval"]
        self.assertLessEqual(low, result["empirical_coverage"])
        self.assertGreaterEqual(high, result["empirical_coverage"])

    def test_cluster_dependence_inflates_video_mean_variance_and_understates_sd(self):
        result = simulate_condition([8] * 10, 0.2, repetitions=20_000,
                                    seed=20261016, batch_size=2_000)
        self.assertAlmostEqual(result["analytic_design_effect"], 2.4)
        self.assertAlmostEqual(result["design_effect_effective_video_count"], 80 / 2.4)
        self.assertGreater(result["analytic_sampling_sd"], 1 / math.sqrt(80))
        self.assertLess(result["mean_naive_estimated_se"], result["analytic_sampling_sd"])
        self.assertAlmostEqual(result["empirical_sampling_sd"],
                               result["analytic_sampling_sd"], delta=0.015)

    def test_perfect_cluster_dependence_exposes_naive_interval_undercoverage(self):
        result = simulate_condition([8] * 10, 1.0, repetitions=20_000,
                                    seed=20261017, batch_size=2_000)
        self.assertLess(result["empirical_coverage"], 0.85)
        self.assertAlmostEqual(result["analytic_expected_naive_sample_variance"], 72 / 79)
        self.assertEqual(result["analytic_design_effect"], 8.0)

    def test_unequal_fixed_sizes_preserve_video_weighted_estimand_formula(self):
        sizes = LAYOUTS["unbalanced_10"]
        self.assertEqual(sum(sizes), 80)
        result = simulate_condition(sizes, 0.2, repetitions=2_000,
                                    seed=20261018, batch_size=250)
        expected_deff = 1 + 0.2 * (80 * sum(size * size for size in sizes) / 80**2 - 1)
        self.assertAlmostEqual(result["analytic_design_effect"], expected_deff)
        self.assertEqual(result["cluster_sizes"], sizes)
        self.assertAlmostEqual(result["analytic_sampling_sd"], math.sqrt(2.6 / 80))
        self.assertAlmostEqual(result["empirical_sampling_sd"], math.sqrt(2.6 / 80), delta=0.015)
        self.assertAlmostEqual(result["empirical_mean_naive_sample_variance"], 77.4 / 79,
                               delta=0.035)

    def test_fixed_seed_reproduces_and_grid_discloses_scope(self):
        left = simulate_condition([2] * 5, 0.1, repetitions=100,
                                  seed=20261019, batch_size=17)
        right = simulate_condition([2] * 5, 0.1, repetitions=100,
                                   seed=20261019, batch_size=17)
        self.assertEqual(left, right)
        result = simulate_grid(repetitions=100, seed=99, batch_size=20,
                               iccs=(0.0, 0.2), layout_names=("balanced_10",))
        self.assertEqual(len(result["conditions"]), 2)
        self.assertIn("not estimated from project data", result["limitations"][0])
        self.assertIn("No alternative interval is evaluated", result["limitations"][2])

    def test_named_cell_seed_is_invariant_to_grid_order(self):
        first = simulate_grid(repetitions=100, seed=200, batch_size=20,
                              iccs=(0.0, 0.2),
                              layout_names=("balanced_10", "unbalanced_10"))
        reversed_grid = simulate_grid(repetitions=100, seed=200, batch_size=20,
                                      iccs=(0.2, 0.0),
                                      layout_names=("unbalanced_10", "balanced_10"))
        index = lambda result: {(row["layout_name"], row["icc"]): row
                                for row in result["conditions"]}
        self.assertEqual(index(first), index(reversed_grid))
        self.assertEqual(first["grid_config"]["repetitions_per_cell"], 100)

    def test_grid_rejects_duplicate_or_malformed_cells(self):
        for kwargs in (
            {"layout_names": ("balanced_10", "balanced_10")},
            {"layout_names": ([])},
            {"iccs": (0.1, 0.1)},
            {"iccs": (10**400,)},
            {"seed": -1},
            {"seed": 2**63},
            {"seed": True},
            {"repetitions": 1},
            {"batch_size": 0},
        ):
            with self.subTest(kwargs=kwargs):
                options = {"repetitions": 10}
                options.update(kwargs)
                with self.assertRaises(ValueError):
                    simulate_grid(**options)

    def test_invalid_cluster_conditions_fail_closed(self):
        invalid = [
            (([80], 0.0), {}),
            (([0, 2], 0.0), {}),
            (([True, 2], 0.0), {}),
            (([2, 2], float("nan")), {}),
            (([2, 2], True), {}),
            (([2, 2], 0.5), {"repetitions": 1}),
            (([2, 2], 0.5), {"seed": -1}),
            (([2, 2], 0.5), {"batch_size": 0}),
        ]
        for (sizes, icc), kwargs in invalid:
            with self.subTest(sizes=sizes, icc=icc, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    simulate_condition(sizes, icc, **kwargs)


if __name__ == "__main__":
    unittest.main()
