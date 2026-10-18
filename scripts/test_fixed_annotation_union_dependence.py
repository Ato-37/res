import unittest

from fixed_annotation_union_dependence import (
    contrast_from_choice_log_probabilities, score_four_condition_logits,
)


class AnnotationUnionDependenceTests(unittest.TestCase):
    def test_delta_uses_baseline_full_prediction_and_has_predeclared_positive_sign(self):
        values = {
            "baseline_full": [-0.1, -1.0, -2.0, -3.0, -4.0],
            "baseline_union_mask": [-0.4, -0.8, -1.5, -2.5, -3.5],
            "intervention_full": [-0.1, -1.0, -2.0, -3.0, -4.0],
            "intervention_union_mask": [-0.7, -0.5, -1.5, -2.5, -3.5],
        }
        result = contrast_from_choice_log_probabilities(values)
        self.assertEqual(result["answer_index"], 0)
        self.assertEqual(result["foil_index"], 1)
        self.assertEqual(result["baseline_union_dependence"], 0.5)
        self.assertEqual(result["intervention_union_dependence"], 1.1)
        self.assertAlmostEqual(result["delta_union_dependence"], 0.6)
        self.assertEqual(result["metric"], "delta_union_dependence")
        self.assertNotIn("delta_f", result)
        self.assertIn("no p-value", result["inferential_status"])

    def test_exact_four_condition_denominator_and_finite_log_probs_required(self):
        values = {name: [-0.1, -1.0, -2.0, -3.0, -4.0]
                  for name in ("baseline_full", "baseline_union_mask",
                               "intervention_full", "intervention_union_mask")}
        with self.assertRaisesRegex(ValueError, "exact_four_union_conditions_required"):
            contrast_from_choice_log_probabilities({**values, "control": values["baseline_full"]})
        values["intervention_full"] = [float("nan")] * 5
        with self.assertRaisesRegex(ValueError, "answer_log_probability_nonfinite"):
            contrast_from_choice_log_probabilities(values)

    def test_four_condition_logit_adapter_uses_fixed_baseline_full_answer_foil(self):
        import torch
        names = ("baseline_full", "baseline_union_mask",
                 "intervention_full", "intervention_union_mask")
        logits = {
            "baseline_full": torch.tensor([[[4.0, 3.0, 1.0, 0.0, -1.0, -2.0]]]),
            "baseline_union_mask": torch.tensor([[[3.0, 3.5, 1.0, 0.0, -1.0, -2.0]]]),
            "intervention_full": torch.tensor([[[4.0, 3.0, 1.0, 0.0, -1.0, -2.0]]]),
            "intervention_union_mask": torch.tensor([[[2.0, 3.5, 1.0, 0.0, -1.0, -2.0]]]),
        }
        outputs = {name: {"logits": value} for name, value in logits.items()}
        result = score_four_condition_logits(condition_outputs=outputs,
            answer_token_ids=[0, 1, 2, 3, 4],
            input_lengths={name: 1 for name in names}, torch_module=torch)
        self.assertEqual(set(result["contrast"]["margin_by_condition"]), set(names))
        self.assertEqual(result["contrast"]["answer_index"], 0)
        self.assertEqual(result["contrast"]["foil_index"], 1)
        with self.assertRaisesRegex(ValueError, "exact_four_union_conditions_required"):
            score_four_condition_logits(condition_outputs={"baseline_full": outputs["baseline_full"]},
                answer_token_ids=[0, 1, 2, 3, 4], input_lengths={}, torch_module=torch)

    def test_multitoken_logits_use_final_position_and_keep_baseline_pq_fixed(self):
        import torch
        names = ("baseline_full", "baseline_union_mask",
                 "intervention_full", "intervention_union_mask")
        baseline = torch.tensor([[[30., 0., 0., 0., 0., 0.],
                                  [0., 30., 0., 0., 0., 0.],
                                  [0., 9., 8., 1., 0., 0.]]])
        treatment = torch.tensor([[[30., 0., 0., 0., 0., 0.],
                                   [0., 30., 0., 0., 0., 0.],
                                   [9., 1., 0., 8., 0., 0.]]])
        outputs = {"baseline_full": {"logits": baseline},
                   "baseline_union_mask": {"logits": baseline},
                   "intervention_full": {"logits": treatment},
                   "intervention_union_mask": {"logits": treatment}}
        result = score_four_condition_logits(condition_outputs=outputs,
            answer_token_ids=[0, 1, 2, 3, 4],
            input_lengths={name: 3 for name in names}, torch_module=torch)
        self.assertEqual(result["score_position_by_condition"], {name: 2 for name in names})
        self.assertEqual(result["contrast"]["answer_index"], 1)
        self.assertEqual(result["contrast"]["foil_index"], 2)


if __name__ == "__main__":
    unittest.main()
