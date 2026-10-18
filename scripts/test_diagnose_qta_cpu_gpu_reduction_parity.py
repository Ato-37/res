from __future__ import annotations

import unittest

from diagnose_qta_cpu_gpu_reduction_parity import parity_passed


class QtaCpuGpuReductionParityTests(unittest.TestCase):
    @staticmethod
    def valid_result():
        return {
            "trials": 2,
            "temporal_group_count": 8,
            "per_layer_group_mass_comparisons": 12 * 8 * 2,
            "per_layer_mass_mismatch_count": 0,
            "aggregate_score_comparisons": 8 * 2,
            "aggregate_score_mismatch_count": 0,
            "topk_selection_mismatch_count": 0,
            "maximum_absolute_layer_mass_difference": 0.0,
            "maximum_absolute_aggregate_score_difference": 0.0,
        }

    def test_zero_mismatches_pass(self):
        self.assertTrue(parity_passed(self.valid_result()))

    def test_each_mismatch_class_fails(self):
        fields = (
            "per_layer_mass_mismatch_count",
            "aggregate_score_mismatch_count",
            "topk_selection_mismatch_count",
        )
        for field in fields:
            with self.subTest(field=field):
                result = self.valid_result()
                result[field] = 1
                self.assertFalse(parity_passed(result))

    def test_missing_comparison_class_fails_closed(self):
        result = self.valid_result()
        del result["topk_selection_mismatch_count"]
        self.assertFalse(parity_passed(result))

    def test_bool_counters_and_empty_comparison_denominators_fail(self):
        for field in ("trials", "per_layer_mass_mismatch_count",
                      "aggregate_score_mismatch_count",
                      "topk_selection_mismatch_count"):
            with self.subTest(field=field):
                result = self.valid_result()
                result[field] = False
                self.assertFalse(parity_passed(result))
        empty = self.valid_result()
        empty["per_layer_group_mass_comparisons"] = 0
        empty["aggregate_score_comparisons"] = 0
        self.assertFalse(parity_passed(empty))

    def test_nonfinite_or_nonzero_maximum_difference_fails(self):
        for field, value in (("maximum_absolute_layer_mass_difference", float("nan")),
                             ("maximum_absolute_aggregate_score_difference", 0.01),
                             ("maximum_absolute_layer_mass_difference", 10**1000)):
            with self.subTest(field=field):
                result = self.valid_result()
                result[field] = value
                self.assertFalse(parity_passed(result))


if __name__ == "__main__":
    unittest.main()
