"""Helpers for the attention-mask hooks."""
from __future__ import annotations


def causal_additive_mask(hidden_states, torch_module):
    """Float additive causal mask [1, 1, q, q] (0 where visible, dtype minimum above the diagonal) for an uncached forward.

    SDPA attention is handed attention_mask=None for a plain causal forward; the mask hooks then need an explicit mask to edit.
    The result is bit-identical to the eager-path causal mask for the same dtype."""
    torch = torch_module
    q = int(hidden_states.shape[-2])
    mask = torch.zeros(1, 1, q, q, dtype=hidden_states.dtype, device=hidden_states.device)
    mask.masked_fill_(torch.triu(torch.ones(q, q, dtype=torch.bool, device=hidden_states.device), 1), torch.finfo(hidden_states.dtype).min)
    return mask
