"""Numerical helper for the normalized-entropy trigger described in STEAR.

This implements only Eq. (5)--(7)'s selector over already-computed per-layer
next-token logits. It does not implement model hooks, evidence selection,
reinjection, counterfactual decoding, or an end-to-end STEAR reproduction.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real


def select_uncertain_layer(layer_logits: Mapping[int, object], *,
                           layer_indices: Sequence[int], threshold: float = 0.85
                           ) -> tuple[bool, int, float]:
    """Return (is_risky, argmax_layer, max_normalized_entropy).

    ``layer_logits`` values must be one-dimensional tensor-like vocabulary
    logits for the same decoding position. Entropy ties choose the lower layer
    index. Intervention is triggered only when the maximum is strictly greater
    than the threshold, matching the paper's ``u_t > tau`` rule.
    """
    import torch

    if (not isinstance(layer_logits, Mapping) or not layer_logits
            or any(type(index) is not int for index in layer_logits)
            or not isinstance(layer_indices, Sequence)
            or isinstance(layer_indices, (str, bytes)) or not layer_indices
            or any(type(index) is not int or index < 0 for index in layer_indices)
            or len(set(layer_indices)) != len(layer_indices)
            or set(layer_logits) != set(layer_indices)):
        raise ValueError("layer_logits_contract_invalid")
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ValueError("entropy_threshold_invalid")
    try:
        threshold = float(threshold)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("entropy_threshold_invalid") from None
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("entropy_threshold_invalid")

    scores: dict[int, float] = {}
    vocab_size: int | None = None
    for layer_index in layer_indices:
        logits = layer_logits[layer_index]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
            raise ValueError("layer_logits_shape_invalid")
        if vocab_size is None:
            vocab_size = int(logits.numel())
        if logits.numel() != vocab_size or vocab_size < 2:
            raise ValueError("layer_logits_vocab_mismatch")
        # Use float64 accumulation: for Qwen's 151,936-token vocabulary,
        # fp32 roundoff can put uniform normalized entropy slightly above 1.
        values = logits.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("layer_logits_nonfinite")
        log_probabilities = torch.log_softmax(values, dim=0)
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum()
        normalized = float((entropy / math.log(vocab_size)).item())
        if not math.isfinite(normalized):
            raise ValueError("normalized_entropy_nonfinite")
        # Normalized Shannon entropy is mathematically in [0, 1]. Clamp only
        # numerical excursions so a uniform distribution never exceeds tau=1.
        normalized = min(1.0, max(0.0, normalized))
        scores[layer_index] = normalized

    selected_layer = min(layer_indices, key=lambda index: (-scores[index], index))
    maximum = scores[selected_layer]
    return maximum > threshold, selected_layer, maximum
