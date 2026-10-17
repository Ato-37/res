"""CPU tests for the paper's normalized-entropy risk selector."""
import unittest

from stear_uncertainty_trigger import select_uncertain_layer


class StearTriggerTests(unittest.TestCase):
    def test_uniform_distribution_triggers_and_selects_highest_entropy_layer(self):
        import torch
        logits = {4: torch.tensor([0.0, 0.0, 0.0, 0.0]),
                  5: torch.tensor([8.0, 0.0, 0.0, 0.0])}
        triggered, layer, entropy = select_uncertain_layer(
            logits, layer_indices=(4, 5), threshold=0.85)
        self.assertTrue(triggered)
        self.assertEqual(layer, 4)
        self.assertAlmostEqual(entropy, 1.0)

    def test_realistic_vocab_uniform_entropy_does_not_exceed_one(self):
        import torch
        vocab_size = 151_936
        triggered, layer, entropy = select_uncertain_layer(
            {5: torch.zeros(vocab_size)}, layer_indices=(5,), threshold=1.0)
        self.assertFalse(triggered)
        self.assertEqual(layer, 5)
        self.assertEqual(entropy, 1.0)

    def test_trigger_comparison_is_strict_and_entropy_ties_choose_lower_layer(self):
        import torch
        uniform = torch.zeros(3)
        logits = {8: uniform, 6: uniform}
        triggered, layer, entropy = select_uncertain_layer(
            logits, layer_indices=(8, 6), threshold=1.0)
        self.assertFalse(triggered)
        self.assertEqual(layer, 6)
        self.assertAlmostEqual(entropy, 1.0)

    def test_accepts_general_sequences_and_rejects_equal_interior_threshold(self):
        import torch
        logits = {4: torch.tensor([2.0, 0.0, -1.0]),
                  5: torch.tensor([3.0, -1.0, -2.0])}
        _, layer, entropy = select_uncertain_layer(
            logits, layer_indices=range(4, 6), threshold=0.0)
        self.assertEqual(layer, 4)
        triggered, _, observed_entropy = select_uncertain_layer(
            logits, layer_indices=(4, 5), threshold=entropy)
        self.assertAlmostEqual(observed_entropy, entropy)
        self.assertFalse(triggered)

    def test_rejects_duplicate_layers_vocab_mismatch_and_nonfinite_logits(self):
        import torch
        with self.assertRaisesRegex(ValueError, "layer_logits_contract_invalid"):
            select_uncertain_layer({4: torch.zeros(3)}, layer_indices=(4, 4))
        with self.assertRaisesRegex(ValueError, "layer_logits_vocab_mismatch"):
            select_uncertain_layer({4: torch.zeros(3), 5: torch.zeros(4)},
                                   layer_indices=(4, 5))
        with self.assertRaisesRegex(ValueError, "layer_logits_nonfinite"):
            select_uncertain_layer({4: torch.tensor([float("nan"), 0.0])},
                                   layer_indices=(4,))

    def test_rejects_invalid_threshold(self):
        import torch
        for threshold in (float("nan"), True, 10**10000):
            with self.subTest(threshold_type=type(threshold).__name__):
                with self.assertRaisesRegex(ValueError, "entropy_threshold_invalid"):
                    select_uncertain_layer({4: torch.zeros(3)}, layer_indices=(4,),
                                           threshold=threshold)
        with self.assertRaisesRegex(ValueError, "layer_logits_contract_invalid"):
            select_uncertain_layer({True: torch.zeros(3)}, layer_indices=(1,))


if __name__ == "__main__":
    unittest.main()
