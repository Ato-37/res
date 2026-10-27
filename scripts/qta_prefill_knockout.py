"""Knockout hooks that act only on the prefill forward of a generation (WP4).

Same edit as qta_edge_knockout.install_knockout_hooks (video->text blocks on the attention mask of the chosen layers), but the pre-hook is a pass-through whenever the
query length differs from the prompt length (decode steps with a KV cache have q_len = 1). The cached keys/values of the prompt rows were computed under the knockout,
so the generated tokens read the video only through the knocked-out text rows."""
from __future__ import annotations

from typing import Any, Sequence

import qta_edge_knockout as ko


def install_prefill_knockout_hooks(*, model: Any, layers_1based: Sequence[int], blocks, seq_len: int, torch_module: Any):
    torch = torch_module
    layers = model.model.language_model.layers
    expected = ko.block_cell_count(blocks)
    ko._assert_causal_unique(blocks)
    state = {"prefill_calls": {int(i): 0 for i in layers_1based}, "decode_passthrough": 0, "expected_cells": expected}
    handles = []
    cache = {"mask": None, "edited": None, "built": None}
    for layer in layers_1based:
        module = layers[layer - 1].self_attn

        def hook(_module, args, kwargs, *, layer=int(layer)):
            hs = kwargs.get("hidden_states")
            if hs is None and args:
                hs = args[0]
            if not torch.is_tensor(hs) or hs.shape[1] != seq_len:
                state["decode_passthrough"] += 1
                return args, kwargs
            mask = kwargs.get("attention_mask")
            if mask is None:
                if cache["built"] is None:
                    from qta_mask_utils import causal_additive_mask
                    cache["built"] = causal_additive_mask(hs, torch)
                mask = cache["built"]
            if not torch.is_tensor(mask) or mask.ndim != 4 or mask.shape[0] != 1 or mask.shape[-1] != seq_len:
                raise ValueError("prefill_knockout_attention_mask_invalid")
            if cache["mask"] is mask:
                edited = cache["edited"]
            else:
                edited = mask.clone()
                low = torch.finfo(edited.dtype).min
                for rows, cols in blocks:
                    r = torch.tensor(rows, device=edited.device, dtype=torch.long)
                    c = torch.tensor(cols, device=edited.device, dtype=torch.long)
                    if not bool((mask[0, 0][r[:, None], c[None, :]] == 0).all()):
                        raise ValueError("knockout_cells_not_originally_visible")
                    edited[0, 0][r[:, None], c[None, :]] = low
                if int(torch.count_nonzero(edited != mask).item()) != expected:
                    raise ValueError("knockout_changed_cell_count_mismatch")
                cache["mask"], cache["edited"] = mask, edited
            state["prefill_calls"][layer] += 1
            new_kwargs = dict(kwargs)
            new_kwargs["attention_mask"] = edited
            return args, new_kwargs

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    return handles, state
