import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qta_edge_knockout as ko
from qta_multiquery_video_key_intervention import install_multiquery_video_key_intervention_hooks, remove_hooks


class Attn(torch.nn.Module):
    def forward(self, x, attention_mask=None):
        return attention_mask


def fake_model(n):
    layers = [SimpleNamespace(self_attn=Attn()) for _ in range(n)]
    return SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))


def causal(n, dtype=torch.float32):
    m = torch.zeros(1, 1, n, n, dtype=dtype)
    m[0, 0].masked_fill_(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), torch.finfo(dtype).min)
    return m


class MaskCacheTests(unittest.TestCase):
    def test_knockout_edit_identical_in_every_layer_and_not_stale_across_masks(self):
        model = fake_model(4)
        blocks = [((6, 7), (1, 2, 3))]
        handles, state = ko.install_knockout_hooks(model=model, layers_1based=[1, 2, 3], blocks=blocks, torch_module=torch)
        try:
            mask = causal(8)
            outs = [model.model.language_model.layers[i].self_attn(None, attention_mask=mask) for i in range(3)]
            expected = mask.clone()
            expected[0, 0][torch.tensor([6, 7])[:, None], torch.tensor([1, 2, 3])[None, :]] = torch.finfo(mask.dtype).min
            for o in outs:
                self.assertTrue(torch.equal(o, expected))
            self.assertTrue(torch.equal(mask, causal(8)))  # input untouched
            self.assertEqual(state["calls"], {1: 1, 2: 1, 3: 1})
            self.assertEqual(state["changed"], {1: 6, 2: 6, 3: 6})
            mask2 = causal(8)
            mask2[0, 0, 0, 0] = -1.0  # different content, different object: must not reuse the cached edit
            out2 = model.model.language_model.layers[0].self_attn(None, attention_mask=mask2)
            self.assertEqual(float(out2[0, 0, 0, 0]), -1.0)
        finally:
            ko.remove_hooks(handles)

    def test_multiquery_bias_identical_in_every_layer_and_not_stale(self):
        model = fake_model(3)
        handles = install_multiquery_video_key_intervention_hooks(
            model=model, layer_indices=(0, 1, 2), biased_positions=(1, 2), ablated_positions=(3,), query_positions=(6, 7),
            logit_bias=-2.0, torch_module=torch)
        try:
            mask = causal(8)
            outs = [l.self_attn(None, attention_mask=mask) for l in model.model.language_model.layers]
            expected = mask.clone()
            rows = torch.tensor([6, 7])[:, None]
            expected[0, 0][rows, torch.tensor([1, 2])[None, :]] += -2.0
            expected[0, 0][rows, torch.tensor([3])[None, :]] = float("-inf")
            for o in outs:
                self.assertTrue(torch.equal(o, expected))
            mask2 = causal(8)
            mask2[0, 0, 5, 4] = 0.0
            mask2[0, 0, 0, 0] = -1.0
            out2 = model.model.language_model.layers[1].self_attn(None, attention_mask=mask2)
            self.assertEqual(float(out2[0, 0, 0, 0]), -1.0)
            self.assertEqual(float(out2[0, 0, 6, 1]), -2.0)
        finally:
            remove_hooks(handles)


if __name__ == "__main__":
    unittest.main()
