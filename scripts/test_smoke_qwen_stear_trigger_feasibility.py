"""CPU-only contract tests for the Qwen STEAR trigger feasibility runner."""
import unittest

from smoke_qwen_stear_trigger_feasibility import (
    _capture_projected_logits,
    _capture_attention_row,
    _layer_entropies,
    _select_engineering_kes,
)


class EngineeringKesTests(unittest.TestCase):
    def test_attention_capture_averages_heads_at_last_query(self):
        import torch

        weights = torch.tensor([[[[0.1, 0.9], [0.8, 0.2]],
                                 [[0.3, 0.7], [0.4, 0.6]]]])
        rows = {}
        _capture_attention_row(output=(None, weights), layer_index=4,
                               attention_rows=rows, torch_module=torch)
        self.assertTrue(torch.equal(rows[4], torch.tensor([0.6, 0.4])))
        self.assertEqual(rows[4].device.type, "cpu")

    def test_attention_capture_rejects_invalid_batch_contract(self):
        import torch

        with self.assertRaisesRegex(RuntimeError, "self_attention_weights_unavailable"):
            _capture_attention_row(
                output=(None, torch.zeros((2, 1, 2, 2))), layer_index=4,
                attention_rows={}, torch_module=torch)

    def test_engineering_kes_uses_selected_layer_neighborhood_and_top_ten_percent(self):
        import torch

        # Twelve compact rows, with deliberately highest attention on visual
        # positions 3 and 17 across the selected layer's radius-one window.
        rows = {}
        for layer in range(4, 16):
            row = torch.zeros((20,), dtype=torch.float32)
            if layer in (13, 14, 15):
                row[3] = 1.0
                row[17] = 0.9
            rows[layer] = row
        neighborhood, chosen = _select_engineering_kes(
            attention_rows=rows, visual_positions=tuple(range(20)),
            selected_layer=14, layer_range=(4, 15), radius=1,
            sequence_length=20)
        self.assertEqual(neighborhood, (13, 14, 15))
        self.assertEqual(chosen, (3, 17))

    def test_engineering_kes_clips_neighborhood_at_layer_boundary(self):
        import torch

        rows = {layer: torch.arange(20, dtype=torch.float32)
                for layer in range(4, 16)}
        neighborhood, chosen = _select_engineering_kes(
            attention_rows=rows, visual_positions=tuple(range(20)),
            selected_layer=4, layer_range=(4, 15), radius=1,
            sequence_length=20)
        self.assertEqual(neighborhood, (4, 5))
        self.assertEqual(chosen, (18, 19))


class LayerEntropySummaryTests(unittest.TestCase):
    def test_capture_projects_only_the_final_query_with_shared_norm_and_head(self):
        import torch

        class LanguageModel:
            norm = staticmethod(lambda value: value + 1.0)

        class Model:
            lm_head = torch.nn.Linear(3, 5, bias=False)

        model = Model()
        hidden = torch.tensor([[[9.0, 9.0, 9.0], [1.0, 2.0, 3.0]]])
        expected = model.lm_head(hidden[:, -1, :] + 1.0)[0]
        layer_logits = {}
        final = {}
        _capture_projected_logits(
            output=hidden, layer_index=4, expected_indices=(4,),
            final_layer_index=4, language_model=LanguageModel(), model=model,
            logits_by_layer=layer_logits, final_projection_check=final,
            torch_module=torch)
        self.assertTrue(torch.equal(layer_logits[4], expected))
        self.assertTrue(torch.equal(final["logits"], expected))
        self.assertEqual(layer_logits[4].device.type, "cpu")

    def test_capture_rejects_multiple_examples(self):
        import torch

        class LanguageModel:
            norm = staticmethod(lambda value: value)

        class Model:
            lm_head = torch.nn.Linear(3, 4, bias=False)

        with self.assertRaisesRegex(RuntimeError, "decoder_hidden_output_contract_invalid"):
            _capture_projected_logits(
                output=torch.zeros((2, 3, 3)), layer_index=1,
                expected_indices=(1,), final_layer_index=3,
                language_model=LanguageModel(), model=Model(),
                logits_by_layer={}, final_projection_check={},
                torch_module=torch)

    def test_layer_summary_is_finite_bounded_and_uses_requested_indices(self):
        import torch
        logits = {
            4: torch.tensor([0.0, 0.0, 0.0, 0.0]),
            5: torch.tensor([10.0, 0.0, -1.0, -2.0]),
        }
        scores = _layer_entropies(logits, (4, 5))
        self.assertEqual(set(scores), {4, 5})
        self.assertEqual(scores[4], 1.0)
        self.assertGreaterEqual(scores[5], 0.0)
        self.assertLessEqual(scores[5], 1.0)


if __name__ == "__main__":
    unittest.main()
