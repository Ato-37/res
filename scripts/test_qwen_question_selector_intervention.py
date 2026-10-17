from __future__ import annotations

import math
import unittest

import torch

from qwen_internal_attention_ablation import compose_video_key_bias_and_ablation
from smoke_qwen_question_selector_intervention import build_smoke_condition_plan


class QuestionSelectorInterventionCompositionTests(unittest.TestCase):
    def setUp(self):
        self.mask = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        upper = torch.triu(torch.ones((5, 5), dtype=torch.bool), diagonal=1)
        self.mask[0, 0][upper] = float("-inf")

    def test_bias_then_same_key_ablation_leaves_key_blocked(self):
        combined = compose_video_key_bias_and_ablation(
            attention_mask=self.mask, biased_positions=(1, 2),
            ablated_positions=(1, 2), query_position=4,
            logit_bias=math.log(2.0), torch_module=torch)
        self.assertTrue(torch.isneginf(combined[0, 0, 4, 1:3]).all())
        self.assertTrue(torch.equal(combined[0, 0, 4, [0, 3, 4]],
                                    self.mask[0, 0, 4, [0, 3, 4]]))

    def test_bias_is_retained_when_a_disjoint_key_is_ablated(self):
        combined = compose_video_key_bias_and_ablation(
            attention_mask=self.mask, biased_positions=(1,),
            ablated_positions=(2,), query_position=4,
            logit_bias=math.log(2.0), torch_module=torch)
        self.assertAlmostEqual(float(combined[0, 0, 4, 1]), math.log(2.0))
        self.assertTrue(torch.isneginf(combined[0, 0, 4, 2]))
        self.assertEqual(float(combined[0, 0, 4, 0]), 0.0)

    def test_partial_overlap_gives_ablation_precedence(self):
        combined = compose_video_key_bias_and_ablation(
            attention_mask=self.mask, biased_positions=(1, 2),
            ablated_positions=(2, 3), query_position=4,
            logit_bias=math.log(2.0), torch_module=torch)
        self.assertAlmostEqual(float(combined[0, 0, 4, 1]), math.log(2.0))
        self.assertTrue(torch.isneginf(combined[0, 0, 4, 2:4]).all())

    def test_condition_plan_has_nine_unique_reusable_branches(self):
        target, sham = (1, 2), (7, 8)
        evidence, control = (10, 11), (20, 21)
        plan = build_smoke_condition_plan(
            target_positions=target, sham_positions=sham,
            evidence_positions=evidence, control_positions=control)
        self.assertEqual(len(plan), 9)
        self.assertEqual(len({branch["name"] for branch in plan}), 9)
        self.assertEqual(plan[0], {"name": "baseline_full", "bias": (), "ablation": ()})
        self.assertEqual([branch["name"].split("_")[0] for branch in plan],
                         ["baseline"] * 3 + ["target"] * 3 + ["sham"] * 3)
        self.assertEqual(plan[3]["bias"], plan[4]["bias"])
        self.assertEqual(plan[4]["ablation"], evidence)
        self.assertEqual(plan[5]["ablation"], control)
        self.assertEqual(plan[6]["bias"], plan[7]["bias"])


if __name__ == "__main__":
    unittest.main()
