"""CPU tests for synthetic temporal-token hook wiring; no model or CUDA needed."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_qwen_temporal_hook_wiring import temporal_group_hook  # noqa: E402


class FakeLayer:
    def register_forward_hook(self, hook):
        self.hook = hook
        return SimpleNamespace(remove=lambda: setattr(self, "hook", None))

    def forward(self, hidden, *extra):
        return self.hook(self, (), (hidden, *extra))


class TemporalHookTests(unittest.TestCase):
    def test_qwen_decoder_tensor_output_is_supported(self):
        original = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        hook = temporal_group_hook([2, 3], 0.25, torch_module=torch)
        modified = hook(None, (), original)
        self.assertEqual(tuple(modified.shape), tuple(original.shape))
        self.assertTrue(torch.equal(modified[:, :2], original[:, :2]))
        self.assertTrue(torch.equal(modified[:, 4:], original[:, 4:]))
        self.assertTrue(torch.equal(modified[:, 2:4], original[:, 2:4] * 1.25))

    def test_zero_strength_is_exact_identity_and_records_positions(self):
        layer = FakeLayer()
        touched = []
        changed = []
        handle = layer.register_forward_hook(temporal_group_hook(
            [2, 3], 0.0, torch_module=torch, touched=touched, changed=changed))
        hidden = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
        extra = object()
        output = layer.forward(hidden, extra)
        handle.remove()

        self.assertIsNone(layer.hook)
        self.assertIs(output[0], hidden)
        self.assertIs(output[1], extra)
        self.assertEqual(touched, [(2, 3)])
        self.assertEqual(changed, [])

    def test_nonzero_strength_changes_only_selected_rows(self):
        layer = FakeLayer()
        touched = []
        changed = []
        handle = layer.register_forward_hook(temporal_group_hook(
            [1, 4], 0.5, torch_module=torch, touched=touched, changed=changed))
        hidden = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
        output = layer.forward(hidden, "metadata")
        handle.remove()

        changed_rows = (output[0] != hidden).any(dim=(0, 2)).nonzero().flatten().tolist()
        self.assertEqual(changed_rows, [1, 4])
        self.assertTrue(torch.equal(hidden, torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)))
        self.assertEqual(output[1], "metadata")
        self.assertEqual(touched, [(1, 4)])
        self.assertEqual(changed, [True])

    def test_invalid_positions_and_strength_fail_before_hook_use(self):
        for positions in ([], [1, 1], [2, 1], [-1]):
            with self.subTest(positions=positions):
                with self.assertRaisesRegex(ValueError, "positions_invalid"):
                    temporal_group_hook(positions, 0.1, torch_module=torch)
        for strength in (True, float("nan"), float("inf"), "0.1"):
            with self.subTest(strength=strength):
                with self.assertRaisesRegex(ValueError, "strength_invalid"):
                    temporal_group_hook([1], strength, torch_module=torch)

    def test_bad_output_tensor_shape_and_out_of_bounds_fail(self):
        hook = temporal_group_hook([1], 0.1, torch_module=torch)
        with self.assertRaisesRegex(ValueError, "output_contract_invalid"):
            hook(None, (), object())
        with self.assertRaisesRegex(ValueError, "hidden_shape_invalid"):
            hook(None, (), (torch.ones(3, 2),))
        with self.assertRaisesRegex(ValueError, "out_of_bounds"):
            hook(None, (), (torch.ones(1, 1, 2),))
        with self.assertRaisesRegex(ValueError, "tensor_invalid"):
            hook(None, (), (SimpleNamespace(shape=(1, 3, 2)),))


if __name__ == "__main__":
    unittest.main()
