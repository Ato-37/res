import math
import hashlib
import unittest
from pathlib import Path

import simulate_claim_mask_cluster_coverage as cluster_simulator
import simulate_claim_mask_tail_coverage as tail_simulator
from simulate_claim_mask_tail_coverage import (
    DISTRIBUTIONS,
    simulate_condition,
    simulate_grid,
)


class TailCoverageSimulationTests(unittest.TestCase):
    def test_iid_normal_is_calibrated_at_small_fixture_scale(self):
        result = simulate_condition([8] * 10, 0.0, "normal", "normal",
                                    repetitions=20_000, seed=20261015, batch_size=1000)
        self.assertLessEqual(abs(result["empirical_coverage"] - 0.95),
                             4 * result["coverage_monte_carlo_se"])
        self.assertAlmostEqual(result["analytic_sampling_sd"], 1 / math.sqrt(80))
        self.assertEqual(result["coverage_count"],
                         round(result["empirical_coverage"] * result["repetitions"]))
        low, high = result["coverage_wilson_95_interval"]
        self.assertLessEqual(low, result["empirical_coverage"])
        self.assertGreaterEqual(high, result["empirical_coverage"])

    def test_finite_variance_heavy_tails_keep_analytic_cluster_sd(self):
        for cluster_dist, residual_dist in (
            ("normal", "student_t3_unit_variance"),
            ("student_t3_unit_variance", "normal"),
            ("student_t5_unit_variance", "student_t3_unit_variance"),
        ):
            with self.subTest(cluster=cluster_dist, residual=residual_dist):
                result = simulate_condition(
                    [8] * 10, 0.2, cluster_dist, residual_dist,
                    repetitions=3000, seed=20261016, batch_size=500,
                )
                self.assertAlmostEqual(result["analytic_design_effect"], 2.4)
                self.assertAlmostEqual(result["analytic_sampling_sd"], math.sqrt(2.4 / 80))
                self.assertAlmostEqual(result["empirical_sampling_sd"],
                                       result["analytic_sampling_sd"], delta=0.035)

    def test_cell_seeds_and_grid_are_order_independent(self):
        a = simulate_grid(repetitions=100, seed=30, batch_size=25,
                          iccs=(0.0, 0.2), layout_names=("balanced_10",))
        b = simulate_grid(repetitions=100, seed=30, batch_size=25,
                          iccs=(0.2, 0.0), layout_names=("balanced_10",))
        key = lambda row: (row["icc"], row["cluster_component_distribution"],
                           row["residual_component_distribution"])
        self.assertEqual({key(r): r for r in a["conditions"]},
                         {key(r): r for r in b["conditions"]})
        self.assertEqual(len(a["conditions"]), 12)
        self.assertEqual(a["grid_config"]["total_simulated_datasets"], 1200)
        dependency = Path(cluster_simulator.__file__).resolve()
        self.assertEqual(a["source_dependencies_sha256"][dependency.name],
                         hashlib.sha256(dependency.read_bytes()).hexdigest())
        runner = Path(tail_simulator.__file__).resolve()
        self.assertEqual(a["source_sha256"], hashlib.sha256(runner.read_bytes()).hexdigest())
        self.assertIn("infinite fourth moment", a["data_generation"]["component_distributions"][
            "student_t3_unit_variance"])

    def test_frozen_full_grid_has_exact_unique_cell_count(self):
        result = simulate_grid(repetitions=2, seed=31, batch_size=1)
        self.assertEqual(result["grid_config"]["condition_count"], 288)
        self.assertEqual(result["grid_config"]["total_simulated_datasets"], 576)
        rows = result["conditions"]
        keys = {(row["layout_name"], row["icc"],
                 row["cluster_component_distribution"],
                 row["residual_component_distribution"]) for row in rows}
        self.assertEqual(len(keys), 288)
        zero_icc = [row for row in rows if row["icc"] == 0]
        positive_icc = [row for row in rows if row["icc"] > 0]
        self.assertEqual(len(zero_icc), 18)
        self.assertEqual(len(positive_icc), 270)

    def test_distribution_grid_is_explicit_and_validates_bad_inputs(self):
        self.assertEqual(len(DISTRIBUTIONS), 3)
        invalid = (
            (([8] * 10, 0.0, "student_t5_unit_variance", "normal"), {}),
            (([8] * 10, 0.2, "normal", "t4"), {}),
            (([8] * 10, True, "normal", "normal"), {}),
            (([8] * 10, 0.2, "normal", "normal"), {"repetitions": 1}),
            (([8] * 10, 0.2, "normal", "normal"), {"seed": -1}),
        )
        for args, kwargs in invalid:
            with self.subTest(args=args, kwargs=kwargs):
                options = {"repetitions": 100, "seed": 1, "batch_size": 20}
                options.update(kwargs)
                with self.assertRaises(ValueError):
                    simulate_condition(*args, **options)


if __name__ == "__main__":
    unittest.main()
