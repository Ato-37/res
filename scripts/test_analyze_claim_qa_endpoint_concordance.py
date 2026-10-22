import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_claim_qa_endpoint_concordance import (
    _fixed_full_prediction_dependence, _pearson, _ranks,
    _validate_unique_cohort_labels, _within_cohort_pearson,
)


class EndpointConcordanceTests(unittest.TestCase):
    def test_perfect_positive_and_negative_correlations(self):
        self.assertAlmostEqual(_pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(_pearson([1, 2, 3], [6, 4, 2]), -1.0)

    def test_average_ranks_for_ties(self):
        self.assertEqual(_ranks([3.0, 1.0, 1.0, 5.0]), [3.0, 1.5, 1.5, 4.0])

    def test_within_cohort_centering_removes_group_offsets(self):
        cohorts = [
            {"rows": [{"qa_margin_D_nats": x, "claim_text_D_nats_per_token": 2*x + 8}
                      for x in (1.0, 2.0, 4.0)]},
            {"rows": [{"qa_margin_D_nats": x + 100,
                       "claim_text_D_nats_per_token": 2*x - 50}
                      for x in (0.0, 3.0, 8.0)]},
        ]
        self.assertAlmostEqual(_within_cohort_pearson(cohorts), 1.0)

    def test_rejects_constant_or_mismatched_vectors(self):
        with self.assertRaisesRegex(ValueError, "constant_input"):
            _pearson([1, 1], [2, 3])
        with self.assertRaisesRegex(ValueError, "input_size_invalid"):
            _pearson([1], [1, 2])

    def test_cohort_labels_must_be_unique_and_nonempty(self):
        _validate_unique_cohort_labels([{"label": "06"}, {"label": "07"}])
        with self.assertRaisesRegex(ValueError, "cohort_labels_not_unique"):
            _validate_unique_cohort_labels([{"label": "06"}, {"label": "06"}])
        with self.assertRaisesRegex(ValueError, "cohort_label_invalid"):
            _validate_unique_cohort_labels([{"label": ""}])

    def test_frozen_prediction_and_foil_come_from_full_condition(self):
        result = _fixed_full_prediction_dependence({
            "full": [-0.2, -1.0, -2.0, -3.0, -4.0],
            "3b_likelihood_selected_window_black": [-0.5, -0.6, -2.0, -3.0, -4.0],
            "temporal_control_black": [-0.3, -1.2, -2.0, -3.0, -4.0],
        })
        self.assertEqual(result["frozen_full_prediction_index"], 0)
        self.assertEqual(result["frozen_full_foil_index"], 1)
        self.assertAlmostEqual(result["evidence_mask_margin_nats"], 0.1)
        self.assertAlmostEqual(result["control_mask_margin_nats"], 0.9)
        self.assertAlmostEqual(
            result["predicted_label_control_minus_evidence_logprob_nats"], 0.2)
        self.assertAlmostEqual(
            result["foil_label_control_minus_evidence_logprob_nats"], -0.6)
        self.assertAlmostEqual(result["control_minus_evidence_predicted_margin_nats"], 0.8)

    def test_full_condition_ties_break_by_label_order(self):
        result = _fixed_full_prediction_dependence({
            "full": [-1.0, -1.0, -2.0, -3.0, -4.0],
            "3b_likelihood_selected_window_black": [-1.0] * 5,
            "temporal_control_black": [-1.0] * 5,
        })
        self.assertEqual((result["frozen_full_prediction_index"],
                          result["frozen_full_foil_index"]), (0, 1))

    def test_predicted_margin_rejects_invalid_scores_or_conditions(self):
        for invalid in (
            {},
            {"full": [0.0] * 4, "3b_likelihood_selected_window_black": [0.0] * 5,
             "temporal_control_black": [0.0] * 5},
            {"full": [0.0, 0.0, 0.0, 0.0, float("nan")],
             "3b_likelihood_selected_window_black": [0.0] * 5,
             "temporal_control_black": [0.0] * 5},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _fixed_full_prediction_dependence(invalid)


if __name__ == "__main__":
    unittest.main()
