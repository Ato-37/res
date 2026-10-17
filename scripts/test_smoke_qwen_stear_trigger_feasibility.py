"""CPU-only contract tests for the Qwen STEAR trigger feasibility runner."""
import unittest

from smoke_qwen_stear_trigger_feasibility import (
    _capture_projected_logits,
    _layer_entropies,
)


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
