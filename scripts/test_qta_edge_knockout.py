import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qta_edge_knockout as ko

# prompt layout: 0-1 system, video groups of 3 tokens each at 2..13 (4 groups), text rows 14..17, last = 18
GROUPS = tuple(tuple(range(2 + 3 * g, 5 + 3 * g)) for g in range(4))
SEQ = 19


class WindowTests(unittest.TestCase):
    def test_window_is_centered_and_clipped(self):
        self.assertEqual(ko.window_layers(10, 9, 28), tuple(range(6, 15)))
        self.assertEqual(ko.window_layers(2, 9, 28), tuple(range(1, 7)))
        self.assertEqual(ko.window_layers(27, 9, 28), tuple(range(23, 29)))
        self.assertEqual(ko.window_layers(5, 1, 28), (5,))

    def test_invalid_windows_rejected(self):
        for args in ((0, 9, 28), (29, 9, 28), (5, 4, 28), (5, 0, 28)):
            with self.assertRaises(ValueError):
                ko.window_layers(*args)


class EdgeBlockTests(unittest.TestCase):
    def test_cross_frame_blocks_only_earlier_groups(self):
        blocks = ko.edge_blocks("cross_frame", video_groups=GROUPS, seq_len=SEQ)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0], (GROUPS[1], GROUPS[0]))
        self.assertEqual(blocks[2][1], GROUPS[0] + GROUPS[1] + GROUPS[2])
        self.assertEqual(ko.block_cell_count(blocks), 3 * 3 + 3 * 6 + 3 * 9)

    def test_other_edges(self):
        v2t = ko.edge_blocks("video_to_text", video_groups=GROUPS, seq_len=SEQ)
        self.assertEqual(v2t, [((14, 15, 16, 17), tuple(range(2, 14)))])
        v2l = ko.edge_blocks("video_to_last", video_groups=GROUPS, seq_len=SEQ)
        self.assertEqual(v2l, [((18,), tuple(range(2, 14)))])
        t2l = ko.edge_blocks("text_to_last", video_groups=GROUPS, seq_len=SEQ)
        self.assertEqual(t2l, [((18,), (14, 15, 16, 17))])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ko.edge_blocks("nonsense", video_groups=GROUPS, seq_len=SEQ)
        with self.assertRaises(ValueError):
            ko.edge_blocks("video_to_text", video_groups=GROUPS, seq_len=14)
        with self.assertRaises(ValueError):
            ko.edge_blocks("cross_frame", video_groups=GROUPS[:1], seq_len=SEQ)


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, hidden, attention_mask=None):
        self.seen = attention_mask
        return hidden


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()


class _Model:
    def __init__(self, n):
        class LM:
            pass
        self.model = type("M", (), {})()
        self.model.language_model = LM()
        self.model.language_model.layers = [_Layer() for _ in range(n)]


def _causal_zero_mask(n):
    mask = torch.full((1, 1, n, n), torch.finfo(torch.float32).min)
    for i in range(n):
        mask[0, 0, i, : i + 1] = 0
    return mask


class HookTests(unittest.TestCase):
    def _run(self, edge, layers):
        model = _Model(5)
        blocks = ko.edge_blocks(edge, video_groups=GROUPS, seq_len=SEQ)
        handles, state = ko.install_knockout_hooks(model=model, layers_1based=layers, blocks=blocks, torch_module=torch)
        mask = _causal_zero_mask(SEQ)
        hidden = torch.zeros(1, SEQ, 4)
        for layer in model.model.language_model.layers:
            layer.self_attn(hidden, attention_mask=mask)
        ko.remove_hooks(handles)
        return model, state, mask, blocks

    def test_only_selected_layers_are_edited_and_only_intended_cells(self):
        model, state, mask, blocks = self._run("video_to_last", (2, 3))
        for idx, layer in enumerate(model.model.language_model.layers, start=1):
            seen = layer.self_attn.seen
            if idx in (2, 3):
                changed = (seen != mask).nonzero().tolist()
                self.assertEqual(len(changed), 12)
                self.assertTrue(all(c[2] == 18 and 2 <= c[3] < 14 for c in changed))
            else:
                self.assertTrue(torch.equal(seen, mask))
        self.assertEqual(state["calls"], {2: 1, 3: 1})
        self.assertEqual(state["expected_cells"], 12)

    def test_cross_frame_keeps_diagonal_visible(self):
        model, _, mask, _ = self._run("cross_frame", (1,))
        seen = model.model.language_model.layers[0].self_attn.seen
        low = torch.finfo(seen.dtype).min
        for g, rows in enumerate(GROUPS):
            for r in rows:
                for c in GROUPS[g]:
                    if c <= r:
                        self.assertEqual(seen[0, 0, r, c].item(), 0.0)
        self.assertEqual(seen[0, 0, GROUPS[2][0], GROUPS[0][0]].item(), low)

    def test_hook_refuses_a_mask_where_cells_are_already_blocked(self):
        model = _Model(2)
        blocks = ko.edge_blocks("video_to_last", video_groups=GROUPS, seq_len=SEQ)
        handles, _ = ko.install_knockout_hooks(model=model, layers_1based=(1,), blocks=blocks, torch_module=torch)
        bad = _causal_zero_mask(SEQ)
        bad[0, 0, 18, 5] = torch.finfo(bad.dtype).min
        with self.assertRaises(ValueError):
            model.model.language_model.layers[0].self_attn(torch.zeros(1, SEQ, 4), attention_mask=bad)
        ko.remove_hooks(handles)


if __name__ == "__main__":
    unittest.main()
