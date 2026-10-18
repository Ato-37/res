"""CPU integration fixture for QTA v4 query-hook and next-token scoring positions."""
import unittest
from unittest.mock import patch

import torch

import claim_mask_video_decoder
import run_nextgqa_qta_v4_union_scorer as scorer
import run_nextgqa_validation_qta_annotation_proxy as kernel


class _InputIds:
    def __init__(self, length):
        self.shape = (1, length)


class _HookHandle:
    def __init__(self, hooks, callback):
        self.hooks = hooks
        self.callback = callback

    def remove(self):
        self.hooks.remove(self.callback)


class _FakeAttention:
    def __init__(self):
        self.pre_hooks = []
        self.forward_hooks = []
        self.seen_masks = []

    def register_forward_pre_hook(self, callback, with_kwargs=False):
        assert with_kwargs
        self.pre_hooks.append(callback)
        return _HookHandle(self.pre_hooks, callback)

    def register_forward_hook(self, callback):
        self.forward_hooks.append(callback)
        return _HookHandle(self.forward_hooks, callback)

    def run(self, attention_mask):
        args, kwargs = (), {"attention_mask": attention_mask}
        for callback in tuple(self.pre_hooks):
            args, kwargs = callback(self, args, kwargs)
        mask = kwargs["attention_mask"]
        self.seen_masks.append(mask.detach().clone())
        attention = torch.full_like(mask, 1.0 / mask.shape[-1])
        output = (None, attention)
        for callback in tuple(self.forward_hooks):
            output = callback(self, args, output) or output
        return mask


class _FakeModel:
    def __init__(self, length):
        self.calls = []
        layers = [type("Layer", (), {"self_attn": _FakeAttention()})()
                  for _ in range(16)]
        self.model = type("Outer", (), {"language_model":
            type("LanguageModel", (), {"layers": layers})()})()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        length = kwargs["input_ids"].shape[1]
        base_mask = torch.zeros((1, 1, length, length), dtype=torch.float32)
        for layer in self.model.language_model.layers:
            last_mask = layer.self_attn.run(base_mask)
        bias = float(last_mask[0, 0, length - 1, 1])
        evidence_visible = bool(torch.isfinite(last_mask[0, 0, length - 1, 2]))
        # Make the synthetic answer distribution depend on the actual hooked mask.
        answer_logit = 4.0 + bias + (1.0 if evidence_visible else 0.0)
        foil_logit = 2.0
        logits = torch.tensor([[[answer_logit, foil_logit, 0., -1., -2., -3.]]])
        return type("Output", (), {"logits": logits})()


class QtaV4QueryScorePositionIntegrationTests(unittest.TestCase):
    def test_score_row_applies_hooks_to_last_prompt_query_and_scores_that_token(self):
        prompt_length = 7
        input_ids = _InputIds(prompt_length)
        batch = {"input_ids": input_ids}
        model = _FakeModel(prompt_length)
        groups = [[1], [2]]
        mapped = {"evidence_group_indices": [1],
                  "annotation_covered_group_indices": [1],
                  "control_group_indices": [],
                  "evidence_token_positions": (2,)}
        expected = {"geometry_sha256": "a" * 64, "group_count": 2}
        item = {"video_path": "/synthetic/fixture.mp4", "video_id": "v",
                "qid": "q", "gold": 0,
                "planned": {"selected_row_binding_sha256": "b" * 64}}
        counts = {name: {"requested": 1, "attempted": 0, "completed": 0}
                  for name in scorer.CHOICE_NAMES}

        with (patch.object(torch.cuda, "mem_get_info", return_value=(3 << 30, 16 << 30)),
              patch.object(torch.cuda, "empty_cache"),
              patch.object(claim_mask_video_decoder, "decode_sampled_video",
                           return_value=([], {"synthetic": True})),
              patch.object(kernel, "prepare_geometry",
                           return_value=(batch, mapped, groups, (4, 5),
                                         (0, 1, 2, 3, 4), "a" * 64))):
            result = scorer.score_row(model, object(), item, expected, counts)

        self.assertEqual(counts, {name: {"requested": 1, "attempted": 1, "completed": 1}
                                  for name in scorer.CHOICE_NAMES})
        self.assertEqual(len(model.calls), 4)
        self.assertTrue(all(call["input_ids"] is input_ids for call in model.calls))
        self.assertTrue(all(call["use_cache"] is False for call in model.calls))
        self.assertTrue(all(call["logits_to_keep"] == 1 for call in model.calls))

        # Production key-intervention pre-hooks touched only query L-1. Baseline-full
        # has no intervention; the other 3 branches each run the real hook helper.
        targeted_indices = {index - 1 for index in kernel.LAYERS_1_BASED}
        for branch_index, name in enumerate(scorer.CHOICE_NAMES):
            masks = [layer.self_attn.seen_masks[branch_index]
                     for layer in model.model.language_model.layers]
            expected_mask = torch.zeros_like(masks[0])
            if name in {"baseline_union_mask", "intervention_union_mask"}:
                expected_mask[0, 0, prompt_length - 1, 2] = float("-inf")
            if name in {"intervention_full", "intervention_union_mask"}:
                expected_mask[0, 0, prompt_length - 1, 1] = kernel.LOGIT_BIAS
            for layer_index, mask in enumerate(masks):
                if layer_index in targeted_indices:
                    self.assertTrue(torch.equal(mask, expected_mask))
                else:
                    self.assertTrue(torch.equal(mask, torch.zeros_like(mask)))

        self.assertEqual(result["baseline_full_correct"], 1)
        self.assertEqual(result["answer_index"], 0)
        self.assertEqual(result["qta_selected_group_count"], 1)
        self.assertEqual(set(result["margin_by_condition"]), set(scorer.CHOICE_NAMES))
        expected_margins = {"baseline_full": 3.0, "baseline_union_mask": 2.0,
                            "intervention_full": 3.0 + kernel.LOGIT_BIAS,
                            "intervention_union_mask": 2.0 + kernel.LOGIT_BIAS}
        for name, expected_margin in expected_margins.items():
            self.assertAlmostEqual(result["margin_by_condition"][name],
                                   expected_margin, places=6)
        self.assertAlmostEqual(result["baseline_union_dependence"], 1.0, places=6)
        self.assertAlmostEqual(result["intervention_union_dependence"], 1.0, places=6)
        self.assertAlmostEqual(result["delta_union_dependence"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
