import unittest

from analyze_qta_crosscohort_sensitivity import (
    _describe, _recompute_qta_delta, _recompute_sham_delta,
)


class CrossCohortSensitivityTests(unittest.TestCase):
    def test_description_counts_near_zero_explicitly(self):
        result = _describe([0.25, -0.125, 1e-8])
        self.assertAlmostEqual(result["mean_nats"], 0.04166667, places=7)
        self.assertEqual(result["positive_cohorts"], 1)
        self.assertEqual(result["negative_cohorts"], 1)
        self.assertEqual(result["near_zero_cohorts"], 1)

    def test_recomputes_qta_from_four_locked_margins(self):
        row = {"primary": {
            "margins_nats": {
                "baseline_control_ablation": 1.5,
                "baseline_evidence_ablation": 1.0,
                "qta_control_ablation": 1.75,
                "qta_evidence_ablation": 1.0,
            },
            "delta_f_qta_minus_baseline_nats": 0.25,
        }}
        self.assertAlmostEqual(_recompute_qta_delta(row), 0.25)

    def test_rejects_declared_qta_delta_mismatch(self):
        row = {"primary": {
            "margins_nats": {
                "baseline_control_ablation": 1.5,
                "baseline_evidence_ablation": 1.0,
                "qta_control_ablation": 1.75,
                "qta_evidence_ablation": 1.0,
            },
            "delta_f_qta_minus_baseline_nats": -0.25,
        }}
        with self.assertRaisesRegex(ValueError, "qta_delta_recomputation_mismatch"):
            _recompute_qta_delta(row)

    def test_recomputes_matched_sham_contrast(self):
        self.assertAlmostEqual(_recompute_sham_delta({
            "delta_f_qta_nats": 0.5,
            "delta_f_sham_nats": 0.25,
            "qta_minus_sham_specificity_nats": 0.25,
        }), 0.25)


if __name__ == "__main__":
    unittest.main()
