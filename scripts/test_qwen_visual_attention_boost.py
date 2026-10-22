from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from qwen_visual_attention_boost import (
    bias_video_keys_during_generation,
    bias_selected_video_keys,
    install_generation_video_key_bias_hooks,
    install_video_key_bias_hooks,
    remove_hooks,
)


class _CaptureAttention(nn.Module):
    def forward(self, hidden_states=None, *, attention_mask=None):
        return attention_mask


class QwenVisualAttentionBoostTests(unittest.TestCase):
    def setUp(self):
        self.mask = torch.zeros((1, 1, 6, 6), dtype=torch.float32)

    def test_bias_changes_only_selected_keys_at_fixed_query(self):
        result = bias_selected_video_keys(
            attention_mask=self.mask, selected_positions=(1, 3),
            query_position=5, logit_bias=0.6931471805599453,
            torch_module=torch)
        self.assertIsNot(result, self.mask)
        self.assertTrue(torch.equal(self.mask, torch.zeros_like(self.mask)))
        expected = torch.zeros_like(self.mask)
        expected[0, 0, 5, 1] = 0.6931471805599453
        expected[0, 0, 5, 3] = 0.6931471805599453
        self.assertTrue(torch.equal(result, expected))

    def test_rejects_invalid_selector_query_and_mask(self):
        invalid = [
            ((1, 3, 2), 5, 0.5),
            ((1, 5), 5, 0.5),
            ((1,), 0, 0.5),
            ((1,), 5, 0.0),
            ((1,), 5, float("nan")),
        ]
        for positions, query, bias in invalid:
            with self.subTest(positions=positions, query=query, bias=bias):
                with self.assertRaises(ValueError):
                    bias_selected_video_keys(
                        attention_mask=self.mask, selected_positions=positions,
                        query_position=query, logit_bias=bias,
                        torch_module=torch)
        with self.assertRaisesRegex(ValueError, "attention_mask_shape_invalid"):
            bias_selected_video_keys(
                attention_mask=self.mask[:, 0], selected_positions=(1,),
                query_position=5, logit_bias=0.5, torch_module=torch)
        with self.assertRaisesRegex(ValueError, "attention_mask_must_be_additive_float"):
            bias_selected_video_keys(
                attention_mask=torch.zeros((1, 1, 6, 6), dtype=torch.bool),
                selected_positions=(1,), query_position=5, logit_bias=0.5,
                torch_module=torch)

    def test_allows_causal_negative_infinity_but_rejects_nan_or_positive_infinity_anywhere(self):
        causal = self.mask.clone()
        causal[0, 0, 4, 5] = float("-inf")
        result = bias_selected_video_keys(
            attention_mask=causal, selected_positions=(1,),
            query_position=5, logit_bias=0.5, torch_module=torch)
        self.assertTrue(torch.isneginf(result[0, 0, 4, 5]))
        for invalid_value in (float("nan"), float("inf")):
            invalid = self.mask.clone()
            invalid[0, 0, 0, 0] = invalid_value
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(
                        ValueError, "attention_mask_contains_invalid_nonfinite"):
                    bias_selected_video_keys(
                        attention_mask=invalid, selected_positions=(1,),
                        query_position=5, logit_bias=0.5, torch_module=torch)

    def test_hook_reuses_frozen_positions_and_can_be_removed(self):
        attention = _CaptureAttention()
        model = SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])))
        positions = [1, 3]
        handles = install_video_key_bias_hooks(
            model=model, layer_indices=(0,), selected_positions=positions,
            query_position=5, logit_bias=0.5, torch_module=torch)
        positions[:] = [2]
        try:
            observed = attention(attention_mask=self.mask)
            self.assertEqual(float(observed[0, 0, 5, 1]), 0.5)
            self.assertEqual(float(observed[0, 0, 5, 3]), 0.5)
            self.assertEqual(float(observed[0, 0, 5, 2]), 0.0)
            self.assertEqual(float(observed[0, 0, 4, 1]), 0.0)
        finally:
            remove_hooks(handles)
        untouched = attention(attention_mask=self.mask)
        self.assertTrue(torch.equal(untouched, self.mask))

    def test_hook_registration_failure_removes_prior_hooks(self):
        first = _CaptureAttention()
        model = SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[
                SimpleNamespace(self_attn=first), SimpleNamespace(self_attn=object())])))
        with self.assertRaisesRegex(ValueError, "self_attention_pre_hook_unavailable"):
            install_video_key_bias_hooks(
                model=model, layer_indices=(0, 1), selected_positions=(1,),
                query_position=5, logit_bias=0.5, torch_module=torch)
        self.assertEqual(len(first._forward_pre_hooks), 0)

    def test_generation_bias_targets_prefill_prediction_row_and_cached_decode(self):
        prefill = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
        for query in range(6):
            prefill[0, 0, query, query + 1:] = float("-inf")
        changed = bias_video_keys_during_generation(
            attention_mask=prefill, selected_positions=(1, 3),
            first_query_position=5, logit_bias=0.6931471805599453,
            torch_module=torch)
        expected = prefill.clone()
        expected[0, 0, 5, 1] = 0.6931471805599453
        expected[0, 0, 5, 3] = 0.6931471805599453
        self.assertTrue(torch.equal(changed, expected))

        # Cached decode: Q=1, K=7, so this row predicts the next token at abs pos 6.
        cached = torch.zeros((1, 1, 1, 7), dtype=torch.float32)
        cached[0, 0, 0, 6] = 0.0  # self key remains untouched; selected keys are strict predecessors.
        cached_changed = bias_video_keys_during_generation(
            attention_mask=cached, selected_positions=(1, 3),
            first_query_position=5, logit_bias=0.5, torch_module=torch)
        self.assertEqual(cached_changed[0, 0, 0].tolist(),
                         [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0])

    def test_generation_bias_leaves_prompt_rows_and_rejects_noncausal_cached_key(self):
        full = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
        changed = bias_video_keys_during_generation(
            attention_mask=full, selected_positions=(1,),
            first_query_position=4, logit_bias=0.5, torch_module=torch)
        self.assertTrue(torch.equal(changed[0, 0, :4], full[0, 0, :4]))
        self.assertTrue(torch.equal(changed[0, 0, 4:, 1], torch.tensor([0.5, 0.5])))
        with self.assertRaisesRegex(ValueError, "precede_generation_query"):
            bias_video_keys_during_generation(
                attention_mask=torch.zeros((1, 1, 1, 5)),
                selected_positions=(4,), first_query_position=3,
                logit_bias=0.5, torch_module=torch)

    def test_generation_hook_applies_across_cached_steps_and_cleans_up(self):
        attention = _CaptureAttention()
        model = SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])))
        observations = []
        handles = install_generation_video_key_bias_hooks(
            model=model, layer_indices=(0,), selected_positions=(1, 3),
            first_query_position=4, logit_bias=0.5, torch_module=torch,
            observations=observations)
        try:
            prefill = attention(attention_mask=torch.zeros((1, 1, 5, 5)))
            self.assertTrue(torch.equal(prefill[0, 0, :4], torch.zeros_like(prefill[0, 0, :4])))
            self.assertTrue(torch.equal(prefill[0, 0, 4, [1, 3]], torch.tensor([0.5, 0.5])))
            decode = attention(attention_mask=torch.zeros((1, 1, 1, 6)))
            self.assertTrue(torch.equal(decode[0, 0, 0, [1, 3]], torch.tensor([0.5, 0.5])))
            self.assertEqual(len(observations), 2)
            self.assertTrue(all(item["verified"] for item in observations))
            self.assertEqual([item["modified_cells"] for item in observations], [2, 2])
            self.assertEqual([item["max_abs_delta_error"] for item in observations], [0.0, 0.0])
        finally:
            remove_hooks(handles)
        self.assertTrue(torch.equal(attention(attention_mask=torch.zeros((1, 1, 1, 6))),
                                    torch.zeros((1, 1, 1, 6))))

    def test_generation_hook_audits_dtype_representable_bf16_bias(self):
        attention = _CaptureAttention()
        model = SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])))
        observations = []
        handles = install_generation_video_key_bias_hooks(
            model=model, layer_indices=(0,), selected_positions=(1,),
            first_query_position=3, logit_bias=0.6931471805599453,
            torch_module=torch, observations=observations)
        try:
            mask = torch.zeros((1, 1, 4, 4), dtype=torch.bfloat16)
            changed = attention(attention_mask=mask)
            expected = float(torch.tensor(0.6931471805599453, dtype=torch.bfloat16))
            self.assertEqual(float(changed[0, 0, 3, 1]), expected)
            self.assertEqual(observations[0]["effective_bias"], expected)
            self.assertTrue(observations[0]["verified"])
        finally:
            remove_hooks(handles)

    def test_generation_hook_fails_closed_on_static_or_nonincremental_cache(self):
        attention = _CaptureAttention()
        model = SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])))
        handles = install_generation_video_key_bias_hooks(
            model=model, layer_indices=(0,), selected_positions=(1,),
            first_query_position=4, logit_bias=0.5, torch_module=torch)
        try:
            with self.assertRaisesRegex(ValueError, "prefill_must_be_square"):
                attention(attention_mask=torch.zeros((1, 1, 5, 8)))
            attention(attention_mask=torch.zeros((1, 1, 5, 5)))
            with self.assertRaisesRegex(ValueError, "not_incremental"):
                attention(attention_mask=torch.zeros((1, 1, 1, 7)))
            # A failed generation is not reusable: the attempted decode never
            # advances state, so the expected key length remains six.
            attention(attention_mask=torch.zeros((1, 1, 1, 6)))
        finally:
            remove_hooks(handles)

    def test_generation_hook_tracks_distinct_layer_indices_across_steps(self):
        attentions = [_CaptureAttention(), _CaptureAttention()]
        model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention) for attention in attentions])))
        handles = install_generation_video_key_bias_hooks(
            model=model, layer_indices=(0, 1), selected_positions=(1,),
            first_query_position=4, logit_bias=0.5, torch_module=torch)
        try:
            prefill = [attention(attention_mask=torch.zeros((1, 1, 5, 5)))
                       for attention in attentions]
            self.assertTrue(all(float(mask[0, 0, 4, 1]) == 0.5 for mask in prefill))
            decode = [attention(attention_mask=torch.zeros((1, 1, 1, 6)))
                      for attention in attentions]
            self.assertTrue(all(float(mask[0, 0, 0, 1]) == 0.5 for mask in decode))
        finally:
            remove_hooks(handles)


if __name__ == "__main__":
    unittest.main()
