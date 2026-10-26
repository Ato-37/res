"""Attention-knockout edges for pathway analysis of a video-language model (after Map the Flow, Kim et al., ICLR 2026).

Token groups in a Qwen2.5-VL video prompt: V = video tokens (grouped by temporal group), T = text rows after the last video
token up to (not including) the final position (vision-end marker, question, options, generation prompt), L = the final position.
Edges (query row <- key column that is blocked):
  cross_frame   : V rows of temporal group g cannot attend to V keys of groups < g
  video_to_text : T rows cannot attend to V keys
  video_to_last : L cannot attend to V keys
  text_to_last  : L cannot attend to T keys
Blocking sets the cell of the additive attention mask to the dtype minimum (the value HF uses for masked cells), only inside a
window of decoder layers. Every edit is audited (only the intended cells change, each layer is hit exactly once).
"""
from __future__ import annotations

from typing import Any, Sequence

EDGES = ("cross_frame", "video_to_text", "video_to_last", "text_to_last")


def window_layers(center_1based: int, width: int, layer_count: int) -> tuple[int, ...]:
    """Layers [center - width//2, center + width//2] clipped to 1..layer_count (1-based), as in Geva et al. (k centred on l)."""
    if (type(center_1based) is not int or type(width) is not int or width < 1 or width % 2 == 0
            or not 1 <= center_1based <= layer_count):
        raise ValueError("knockout_window_invalid")
    half = width // 2
    return tuple(range(max(1, center_1based - half), min(layer_count, center_1based + half) + 1))


def edge_blocks(edge: str, *, video_groups: Sequence[Sequence[int]], seq_len: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Blocks (rows, cols) of mask cells to knock out for one edge class. Positions are global token indices."""
    groups = tuple(tuple(g) for g in video_groups)
    if (edge not in EDGES or not groups or any(not g for g in groups)
            or any(tuple(sorted(set(g))) != g for g in groups) or type(seq_len) is not int):
        raise ValueError("edge_blocks_inputs_invalid")
    all_video = tuple(sorted(p for g in groups for p in g))
    last_video = all_video[-1]
    if seq_len < last_video + 3:
        raise ValueError("sequence_too_short_for_text_rows")
    text_rows = tuple(range(last_video + 1, seq_len - 1))
    last = seq_len - 1
    if edge == "cross_frame":
        if len(groups) < 2:
            raise ValueError("cross_frame_needs_two_groups")
        blocks, earlier = [], ()
        for g in groups:
            if earlier:
                blocks.append((g, earlier))
            earlier = earlier + g
        return blocks
    if edge == "video_to_text":
        return [(text_rows, all_video)]
    if edge == "video_to_last":
        return [((last,), all_video)]
    return [((last,), text_rows)]


def block_cell_count(blocks: Sequence[tuple[Sequence[int], Sequence[int]]]) -> int:
    return sum(len(rows) * len(cols) for rows, cols in blocks)


def _assert_causal_unique(blocks) -> None:
    seen = 0
    for rows, cols in blocks:
        if not rows or not cols or max(cols) >= min(rows):
            raise ValueError("knockout_cells_must_be_causally_visible")
        seen += len(rows) * len(cols)
    # blocks of one edge never overlap (cross-frame blocks use disjoint row sets)
    cells = set()
    for rows, cols in blocks:
        for r in rows:
            for c in cols:
                cells.add((r, c))
    if len(cells) != seen:
        raise ValueError("knockout_blocks_overlap")


def install_knockout_hooks(*, model: Any, layers_1based: Sequence[int], blocks, torch_module: Any):
    """Register pre-hooks on the self-attention modules of the given layers; returns (handles, state).

    state["calls"][layer] counts hits, state["changed"][layer] the number of cells changed (audited against the plan)."""
    torch = torch_module
    layers = model.model.language_model.layers
    expected = block_cell_count(blocks)
    _assert_causal_unique(blocks)
    state = {"calls": {int(i): 0 for i in layers_1based}, "changed": {}, "expected_cells": expected}
    handles = []
    # The same causal mask object is passed to every layer of one forward and the edit is identical for every selected layer, so the edited mask and
    # its audits (visibility, changed-cell count) are computed once per mask object and reused; a different mask object recomputes (no stale reuse).
    cache = {"mask": None, "edited": None, "changed": None, "built": None}
    for layer in layers_1based:
        module = layers[layer - 1].self_attn

        def hook(_module, args, kwargs, *, layer=int(layer)):
            mask = kwargs.get("attention_mask")
            if mask is None and torch.is_tensor(kwargs.get("hidden_states")):
                # SDPA hands None for a plain causal forward: build the explicit mask once per installation (hooks are installed per forward)
                if cache.get("built") is None:
                    from qta_mask_utils import causal_additive_mask
                    cache["built"] = causal_additive_mask(kwargs["hidden_states"], torch)
                mask = cache["built"]
            if not torch.is_tensor(mask) or mask.ndim != 4 or mask.shape[0] != 1:
                raise ValueError("knockout_attention_mask_invalid")
            if cache["mask"] is mask:
                edited, changed = cache["edited"], cache["changed"]
            else:
                edited = mask.clone()
                low = torch.finfo(edited.dtype).min
                for rows, cols in blocks:
                    r = torch.tensor(rows, device=edited.device, dtype=torch.long)
                    c = torch.tensor(cols, device=edited.device, dtype=torch.long)
                    if not bool((mask[0, 0][r[:, None], c[None, :]] == 0).all()):
                        raise ValueError("knockout_cells_not_originally_visible")
                    edited[0, 0][r[:, None], c[None, :]] = low
                changed = int(torch.count_nonzero(edited != mask).item())
                if changed != expected:
                    raise ValueError("knockout_changed_cell_count_mismatch")
                cache["mask"], cache["edited"], cache["changed"] = mask, edited, changed
            state["calls"][layer] += 1
            state["changed"][layer] = changed
            new_kwargs = dict(kwargs)
            new_kwargs["attention_mask"] = edited
            return args, new_kwargs

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    return handles, state


def remove_hooks(handles) -> None:
    for handle in reversed(list(handles)):
        handle.remove()
