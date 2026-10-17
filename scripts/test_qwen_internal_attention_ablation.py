"""CPU tests for fixed-query Qwen visual-key attention-access ablation."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from qwen_internal_attention_ablation import (  # noqa: E402
    ablate_selected_video_keys,
    install_video_key_ablation_hooks,
    remove_hooks,
)


class InternalAttentionAblationTests(unittest.TestCase):
    def setUp(self):
        mask = torch.full((1, 1, 5, 5), float("-inf"), dtype=torch.float32)
        for query in range(5):
            mask[0, 0, query, :query + 1] = 0
        self.mask = mask

    def test_only_selected_keys_at_fixed_query_are_masked(self):
        result = ablate_selected_video_keys(
            attention_mask=self.mask, selected_positions=[1, 2],
            query_position=4, torch_module=torch)
        expected = self.mask.clone()
        expected[0, 0, 4, 1:3] = float("-inf")
        self.assertTrue(torch.equal(torch.isneginf(result), torch.isneginf(expected)))
        self.assertTrue(torch.equal(result[0, 0, :4], self.mask[0, 0, :4]))
        self.assertTrue(torch.equal(result[0, 0, 4, [0, 3, 4]],
                                    self.mask[0, 0, 4, [0, 3, 4]]))
        self.assertFalse(torch.isnan(result).any())
        self.assertTrue(torch.isfinite(result[0, 0, 4, :5]).any())
        self.assertFalse(torch.isneginf(self.mask[0, 0, 4, 1:3]).any())

    def test_rejects_masked_selected_keys_and_all_key_ablation(self):
        padded = self.mask.clone()
        padded[0, 0, 4, 1] = float("-inf")
        with self.assertRaisesRegex(ValueError, "selected_key_not_currently_visible"):
            ablate_selected_video_keys(
                attention_mask=padded, selected_positions=[1],
                query_position=4, torch_module=torch)
        all_video_keys = ablate_selected_video_keys(
            attention_mask=self.mask, selected_positions=[0, 1, 2, 3],
            query_position=4, torch_module=torch)
        # The current answer-query token remains visible even when every prior
        # key is removed; caller mapping must keep this self/text escape path in mind.
        self.assertTrue(torch.isfinite(all_video_keys[0, 0, 4, 4]))

        min_mask = torch.full_like(self.mask, torch.finfo(self.mask.dtype).min)
        min_mask[0, 0, 4, 1] = 0
        with self.assertRaisesRegex(ValueError, "ablation_removed_all_causal_keys"):
            ablate_selected_video_keys(
                attention_mask=min_mask, selected_positions=[1],
                query_position=4, torch_module=torch)

    def test_rejects_invalid_shapes_dtypes_positions_and_nonzero_selected(self):
        with self.assertRaisesRegex(ValueError, "shape_or_dtype_invalid"):
            ablate_selected_video_keys(
                attention_mask=torch.zeros((1, 5, 5)), selected_positions=[1],
                query_position=4, torch_module=torch)
        with self.assertRaisesRegex(ValueError, "selected_causal_video_keys_invalid"):
            ablate_selected_video_keys(
                attention_mask=self.mask, selected_positions=[2, 1],
                query_position=4, torch_module=torch)
        biased = self.mask.clone()
        biased[0, 0, 4, 1] = 0.25
        with self.assertRaisesRegex(ValueError, "selected_key_mask_value_not_zero"):
            ablate_selected_video_keys(
                attention_mask=biased, selected_positions=[1],
                query_position=4, torch_module=torch)

        cached_shape = torch.zeros((1, 1, 1, 5), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "uncached_query_position_required"):
            ablate_selected_video_keys(
                attention_mask=cached_shape, selected_positions=[1],
                query_position=0, torch_module=torch)

    def test_rejects_nan_and_positive_infinity_anywhere(self):
        for value, error in ((float("nan"), "attention_mask_contains_nan"),
                             (float("inf"), "attention_mask_contains_positive_infinity")):
            malformed = self.mask.clone()
            malformed[0, 0, 0, 0] = value
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                ablate_selected_video_keys(
                    attention_mask=malformed, selected_positions=[1],
                    query_position=4, torch_module=torch)

    def test_hook_freezes_positions_and_removes_cleanly(self):
        class TinyAttention(torch.nn.Module):
            def forward(self, attention_mask):
                return attention_mask

        attention = TinyAttention()
        model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)])))
        positions = [1]
        hooks = install_video_key_ablation_hooks(
            model=model, layer_indices=[0], selected_positions=positions,
            query_position=4, torch_module=torch)
        positions[0] = 2
        result = attention(attention_mask=self.mask)
        self.assertTrue(torch.isneginf(result[0, 0, 4, 1]))
        self.assertFalse(torch.isneginf(result[0, 0, 4, 2]))
        remove_hooks(hooks)
        unchanged = attention(attention_mask=self.mask)
        self.assertTrue(torch.equal(unchanged, self.mask))


if __name__ == "__main__":
    unittest.main()
