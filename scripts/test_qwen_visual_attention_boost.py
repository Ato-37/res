from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from qwen_visual_attention_boost import (
    bias_selected_video_keys,
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

    def test_hook_reuses_frozen_positions_and_can_be_removed(self):
        attention = _CaptureAttention()
        model = SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])))
        handles = install_video_key_bias_hooks(
            model=model, layer_indices=(0,), selected_positions=(1, 3),
            query_position=5, logit_bias=0.5, torch_module=torch)
        try:
            observed = attention(attention_mask=self.mask)
            self.assertEqual(float(observed[0, 0, 5, 1]), 0.5)
            self.assertEqual(float(observed[0, 0, 5, 3]), 0.5)
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


if __name__ == "__main__":
    unittest.main()
