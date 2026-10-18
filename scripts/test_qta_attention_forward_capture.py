from __future__ import annotations

import unittest

from qta_attention_forward_capture import (
    aggregate_captured_qta_rows, capture_qta_attention_forward_once,
)
from qwen_question_temporal_attention_selector import aggregate_temporal_group_attention


class FakeAttention:
    def __init__(self, torch_module, *, bad=None, registration_error=False):
        self.torch = torch_module
        self.bad = bad
        self.registration_error = registration_error
        self.force_dtype = None
        self.hooks = []

    def register_forward_hook(self, hook):
        if self.registration_error:
            raise RuntimeError("synthetic_hook_registration_failure")
        self.hooks.append(hook)
        owner = self

        class Handle:
            def remove(self):
                owner.hooks.remove(hook)

        return Handle()

    def __call__(self, hidden):
        attention = self.torch.full((1, 2, 4, 4), 0.25)
        if self.bad == "shape":
            attention = self.torch.full((1, 2, 3, 4), 1 / 3)
        elif self.bad == "not_normalized":
            attention = self.torch.full((1, 2, 4, 4), 0.2)
        elif self.bad == "bf16_rounding":
            distribution = self.torch.tensor([0.333, 0.333, 0.334, 0.0])
            attention = distribution.view(1, 1, 1, 4).expand(1, 2, 4, 4)
        elif self.bad == "nan":
            attention[0, 0, 1, 0] = self.torch.nan
        if self.force_dtype is not None:
            attention = attention.to(self.force_dtype)
        output = (hidden, attention)
        if self.bad == "missing":
            output = (hidden, None)
        for hook in tuple(self.hooks):
            hook(self, (hidden,), output)
        return output


class FakeLayer:
    def __init__(self, torch_module, *, bad=None):
        self.self_attn = FakeAttention(torch_module, bad=bad)


class FakeModel:
    def __init__(self, torch_module, *, bad_layer=None, skip_layer=None,
                 registration_error_layer=None):
        self.torch = torch_module
        self.model = type("ModelRoot", (), {})()
        self.model.language_model = type("LanguageModel", (), {})()
        self.model.language_model.layers = [
            FakeLayer(torch_module, bad=bad_layer if i == 4 else None)
            for i in range(16)
        ]
        if registration_error_layer is not None:
            self.model.language_model.layers[
                registration_error_layer].self_attn.registration_error = True
        self.skip_layer = skip_layer

    def __call__(self):
        hidden = self.torch.zeros((1, 4, 2))
        for index, layer in enumerate(self.model.language_model.layers):
            if index == self.skip_layer:
                continue
            hidden, _attention = layer.self_attn(hidden)
        return hidden


class QtaAttentionForwardCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def _capture(self, model, forward=None, *, positions=(1, 2), length=4):
        call = forward or model
        return capture_qta_attention_forward_once(
            model=model, forward=call, question_positions=positions,
            input_length=length, torch_module=self.torch)

    def test_captures_exact_layers_and_removes_hooks(self):
        model = FakeModel(self.torch)
        output, rows, report = self._capture(model)
        self.assertEqual(tuple(sorted(rows)), tuple(range(5, 17)))
        self.assertEqual(tuple(rows[5].shape), (2, 2, 4))
        self.assertEqual(report["attention_shape"], [12, 2, 2, 4])
        self.assertEqual(report["forward_calls"], 1)
        self.assertFalse(report["attention_capture_authenticated"])
        self.assertEqual(output.shape, (1, 4, 2))
        self.assertTrue(all(not layer.self_attn.hooks
                            for layer in model.model.language_model.layers))

    def test_captured_row_aggregation_matches_full_attention_selector(self):
        self.torch.manual_seed(20261018)
        full = {
            index: self.torch.softmax(self.torch.randn(1, 2, 4, 4), dim=-1)
            for index in range(12)
        }
        old = aggregate_temporal_group_attention(
            attention_by_layer=full, layer_indices=tuple(range(12)),
            question_positions=(2, 3), group_positions=((0,), (1,)),
            torch_module=self.torch)
        captured = {index + 5: tensor[0, :, (2, 3), :]
                    for index, tensor in full.items()}
        new = aggregate_captured_qta_rows(
            rows=captured, group_positions=((0,), (1,)),
            torch_module=self.torch)
        self.assertEqual(new, old)

    def test_rejects_missing_layer_capture(self):
        model = FakeModel(self.torch, skip_layer=15)
        with self.assertRaisesRegex(RuntimeError,
                                   "qta_forward_capture_layer_set_incomplete"):
            self._capture(model)

    def test_rejects_second_layer_invocation(self):
        model = FakeModel(self.torch)

        def two_forwards():
            model()
            return model()

        with self.assertRaisesRegex(RuntimeError,
                                   "qta_forward_capture_layer_called_multiple_times"):
            self._capture(model, two_forwards)

    def test_rejects_attention_shape_or_missing_matrix(self):
        for invalid in ("shape", "missing"):
            with self.subTest(invalid=invalid):
                model = FakeModel(self.torch, bad_layer=invalid)
                with self.assertRaisesRegex(RuntimeError,
                                           "qta_forward_capture_attention_output_invalid"):
                    self._capture(model)

    def test_rejects_nonfinite_attention_and_cleans_hooks(self):
        model = FakeModel(self.torch, bad_layer="nan")
        with self.assertRaisesRegex(RuntimeError,
                                   "qta_forward_capture_attention_value_invalid"):
            self._capture(model)
        self.assertTrue(all(not layer.self_attn.hooks
                            for layer in model.model.language_model.layers))

    def test_rejects_non_normalized_float32_attention_rows(self):
        model = FakeModel(self.torch, bad_layer="not_normalized")
        with self.assertRaisesRegex(RuntimeError,
                                   "qta_forward_capture_attention_row_not_normalized"):
            self._capture(model)

    def test_accepts_bfloat16_rows_with_dtype_specific_tolerance(self):
        model = FakeModel(self.torch)
        for layer in model.model.language_model.layers[4:16]:
            layer.self_attn.force_dtype = self.torch.bfloat16
        _output, _rows, report = self._capture(model)
        self.assertEqual(report["attention_source_dtype"], "bfloat16")
        self.assertEqual(report["attention_row_sum_tolerance"], 0.0041)
        self.assertTrue(report[
            "attention_rows_satisfy_source_dtype_normalization_tolerance"])

    def test_bfloat16_bound_accepts_rounding_error_that_float32_bound_rejects(self):
        model = FakeModel(self.torch, bad_layer="bf16_rounding")
        for layer in model.model.language_model.layers[4:16]:
            layer.self_attn.force_dtype = self.torch.bfloat16
        _output, _rows, report = self._capture(model)
        deviation = report["attention_row_sum_max_abs_deviation"]
        self.assertGreater(deviation, 2e-4)
        self.assertLessEqual(deviation, 0.0041)
        self.assertTrue(report[
            "attention_rows_satisfy_source_dtype_normalization_tolerance"])

    def test_rejects_attention_source_dtype_change_between_layers(self):
        model = FakeModel(self.torch)
        model.model.language_model.layers[5].self_attn.force_dtype = self.torch.bfloat16
        with self.assertRaisesRegex(
                RuntimeError,
                "qta_forward_capture_attention_source_dtype_mismatch"):
            self._capture(model)

    def test_rejects_invalid_query_positions_before_forward(self):
        model = FakeModel(self.torch)
        with self.assertRaisesRegex(ValueError, "qta_forward_capture_input_invalid"):
            self._capture(model, positions=(1, 3))
        self.assertTrue(all(not layer.self_attn.hooks
                            for layer in model.model.language_model.layers))

    def test_rejects_shared_attention_module_aliases(self):
        model = FakeModel(self.torch)
        model.model.language_model.layers[5].self_attn = (
            model.model.language_model.layers[4].self_attn)
        with self.assertRaisesRegex(ValueError,
                                   "qta_forward_capture_layer_module_alias"):
            self._capture(model)

    def test_cleans_up_if_hook_registration_fails_partway(self):
        model = FakeModel(self.torch, registration_error_layer=5)
        first = model.model.language_model.layers[4].self_attn
        with self.assertRaisesRegex(RuntimeError,
                                   "synthetic_hook_registration_failure"):
            self._capture(model)
        self.assertEqual(first.hooks, [])


if __name__ == "__main__":
    unittest.main()
