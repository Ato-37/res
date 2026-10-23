"""Self-guided temporal-group scores from eager attention (SGC, experiments/2026-10-23_self_guided_cropping_plan.md)."""
from __future__ import annotations

import hashlib
from typing import Any, Sequence


def group_mass(attn_weights: Any, rows: Sequence[int], groups: Sequence[Sequence[int]], torch_module: Any):
    """Mean over heads and rows of the attention mass the rows put on the keys of each temporal group.

    attn_weights: tensor [1, heads, q_len, k_len] (eager softmax output). Returns a float32 tensor [len(groups)] on the same device."""
    torch = torch_module
    if attn_weights.ndim != 4 or attn_weights.shape[0] != 1:
        raise ValueError("attention_weights_shape_invalid")
    r = torch.tensor(list(rows), device=attn_weights.device, dtype=torch.long)
    sub = attn_weights[0][:, r, :].float()  # heads x rows x keys
    out = []
    for g in groups:
        k = torch.tensor(list(g), device=attn_weights.device, dtype=torch.long)
        out.append(sub[:, :, k].sum(dim=-1).mean())
    return torch.stack(out)


def top_groups(scores: Sequence[float], k: int) -> tuple[int, ...]:
    """Indices of the k highest scores; ties broken by the lower index (deterministic)."""
    n = len(scores)
    if type(k) is not int or not 1 <= k <= n:
        raise ValueError("top_k_invalid")
    order = sorted(range(n), key=lambda i: (-float(scores[i]), i))
    return tuple(sorted(order[:k]))


def hash_random_groups(k: int, group_count: int, token: str, seed: str = "sgc-random-keep-v1") -> tuple[int, ...]:
    """k groups chosen by SHA-256 ranking (deterministic, outcome-independent)."""
    if type(k) is not int or not 1 <= k <= group_count or not isinstance(token, str) or not token:
        raise ValueError("hash_random_groups_invalid")
    ranked = sorted(range(group_count), key=lambda i: (hashlib.sha256(f"{seed}\0{token}\0{i}".encode()).digest(), i))
    return tuple(sorted(ranked[:k]))


def overlap_with_chance(selected: Sequence[int], annotation: Sequence[int], group_count: int) -> dict:
    """|selected & annotation| and the hypergeometric expectation for random selections of the same size."""
    sel, ann = set(selected), set(annotation)
    k, a = len(sel), len(ann)
    return {"overlap": len(sel & ann), "expected_by_chance": k * a / group_count, "k": k, "annotation_groups": a}
