"""Fixed logit-bias treatment for selected Qwen video-key attention.

This is an explicitly named Qwen adaptation, not a STEAR reproduction. The
caller must select and freeze visual key positions from an untreated full
video pass, then reuse those positions in every treatment branch. This module
only edits the additive causal attention mask at the fixed answer query row;
it does not select evidence, load data, score outputs, or authorize a run.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def bias_selected_video_keys(*, attention_mask: Any,
                             selected_positions: Sequence[int],
                             query_position: int,
                             logit_bias: float,
                             torch_module: Any) -> Any:
    """Return a cloned additive 4-D mask with a fixed bias on selected keys.

    The mask must be broadcastable as ``(batch=1, heads=1, query, key)`` and
    use a floating dtype. Only the specified query row changes. Positions
    must be causal predecessors of that row, which also rejects accidentally
    selecting text/future tokens when the caller supplies video-token indices.
    """
    if torch_module is None or not callable(getattr(torch_module, "is_tensor", None)):
        raise ValueError("torch_module_invalid")
    if not torch_module.is_tensor(attention_mask):
        raise ValueError("attention_mask_tensor_required")
    if attention_mask.ndim != 4 or tuple(attention_mask.shape[:2]) != (1, 1):
        raise ValueError("attention_mask_shape_invalid")
    if not attention_mask.dtype.is_floating_point:
        raise ValueError("attention_mask_must_be_additive_float")
    # Causal masks legitimately use -inf for unreachable positions. NaN and
    # +inf anywhere in the mask, including rows/keys outside this treatment,
    # indicate a malformed attention input and must fail closed.
    if (bool(torch_module.isnan(attention_mask).any())
            or bool(torch_module.isposinf(attention_mask).any())):
        raise ValueError("attention_mask_contains_invalid_nonfinite")
    query_length, key_length = map(int, attention_mask.shape[-2:])
    if (type(query_position) is not int or query_position < 0
            or query_position >= query_length or key_length != query_length):
        raise ValueError("query_position_invalid")
    if (isinstance(logit_bias, bool) or not isinstance(logit_bias, (int, float))
            or not math.isfinite(float(logit_bias)) or float(logit_bias) <= 0.0):
        raise ValueError("positive_finite_logit_bias_required")
    if (not isinstance(selected_positions, Sequence)
            or isinstance(selected_positions, (str, bytes))
            or not selected_positions
            or any(type(position) is not int for position in selected_positions)
            or tuple(sorted(set(selected_positions))) != tuple(selected_positions)
            or any(position < 0 or position >= query_position
                   for position in selected_positions)):
        raise ValueError("selected_video_positions_invalid")

    result = attention_mask.clone()
    selected = torch_module.tensor(list(selected_positions),
                                   device=result.device, dtype=torch_module.long)
    result[0, 0, query_position, selected] += float(logit_bias)
    if not bool(torch_module.isfinite(result[0, 0, query_position, selected]).all()):
        raise ValueError("biased_attention_mask_nonfinite")
    return result


def install_video_key_bias_hooks(*, model: Any, layer_indices: Sequence[int],
                                 selected_positions: Sequence[int],
                                 query_position: int, logit_bias: float,
                                 torch_module: Any) -> list[Any]:
    """Install pre-forward hooks that reuse one frozen selector in each layer."""
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise ValueError("qwen_language_layers_unavailable")
    if (not isinstance(layer_indices, Sequence) or isinstance(layer_indices, (str, bytes))
            or not layer_indices
            or any(type(index) is not int or index < 0 or index >= len(layers)
                   for index in layer_indices)
            or len(set(layer_indices)) != len(layer_indices)):
        raise ValueError("intervention_layer_indices_invalid")
    # Validate treatment parameters before registering any hook.
    _validate_selected_positions(selected_positions, query_position, logit_bias)
    # Copy caller-owned sequences before closing over them: a later mutation
    # must not silently change the treatment between branches/forwards.
    frozen_positions = tuple(selected_positions)
    frozen_layers = tuple(layer_indices)
    handles = []
    try:
        for layer_index in frozen_layers:
            attention = getattr(layers[layer_index], "self_attn", None)
            register = getattr(attention, "register_forward_pre_hook", None)
            if not callable(register):
                raise ValueError("self_attention_pre_hook_unavailable")

            def pre_hook(_module, args, kwargs):
                if "attention_mask" not in kwargs:
                    raise ValueError("attention_mask_keyword_missing")
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = bias_selected_video_keys(
                    attention_mask=kwargs["attention_mask"],
                    selected_positions=frozen_positions,
                    query_position=query_position, logit_bias=logit_bias,
                    torch_module=torch_module)
                return args, kwargs

            try:
                handle = register(pre_hook, with_kwargs=True)
            except TypeError:
                raise ValueError("pytorch_keyword_pre_hook_unavailable") from None
            handles.append(handle)
    except Exception:
        remove_hooks(handles)
        raise
    return handles


def bias_video_keys_during_generation(*, attention_mask: Any,
        selected_positions: Sequence[int], first_query_position: int,
        logit_bias: float, torch_module: Any) -> Any:
    """Bias frozen absolute video keys on answer-generation query rows.

    Supports Qwen eager masks for full prefill (Q=K), cached decode (Q=1,
    K=prefix length), and uncached full-prefix decode. This helper alone
    assumes that the key axis begins at absolute position 0; the hook installer
    enforces a DynamicCache-only sequence to rule out static/cropped caches.
    `first_query_position`
    is the prompt's final token row, which predicts the first generated token.
    Keys must be strict causal predecessors; only selected keys and eligible
    query rows change. Do not use alone to authenticate a static or cropped
    cache because mask shape does not reveal key-axis origin.
    """
    if (torch_module is None or not callable(getattr(torch_module, "is_tensor", None))
            or not torch_module.is_tensor(attention_mask)):
        raise ValueError("attention_mask_tensor_required")
    if (attention_mask.ndim != 4 or tuple(attention_mask.shape[:2]) != (1, 1)
            or not attention_mask.dtype.is_floating_point):
        raise ValueError("attention_mask_shape_or_dtype_invalid")
    if (bool(torch_module.isnan(attention_mask).any())
            or bool(torch_module.isposinf(attention_mask).any())):
        raise ValueError("attention_mask_contains_invalid_nonfinite")
    qlen, klen = map(int, attention_mask.shape[-2:])
    if qlen < 1 or klen < qlen:
        raise ValueError("generation_attention_mask_geometry_invalid")
    if type(first_query_position) is not int or first_query_position < 0:
        raise ValueError("first_query_position_invalid")
    if (not isinstance(selected_positions, Sequence)
            or isinstance(selected_positions, (str, bytes)) or not selected_positions
            or any(type(p) is not int for p in selected_positions)
            or tuple(sorted(set(selected_positions))) != tuple(selected_positions)
            or any(p < 0 or p >= klen for p in selected_positions)):
        raise ValueError("selected_video_positions_invalid")
    if (isinstance(logit_bias, bool) or not isinstance(logit_bias, (int, float))
            or not math.isfinite(float(logit_bias)) or logit_bias <= 0):
        raise ValueError("positive_finite_logit_bias_required")
    offset = klen - qlen
    rows = tuple(row for row in range(qlen)
                 if offset + row >= first_query_position)
    if not rows:
        raise ValueError("no_generation_query_rows_in_mask")
    if any(position >= offset + row
           for position in selected_positions for row in rows):
        raise ValueError("selected_video_keys_must_precede_generation_query")
    row_idx = torch_module.tensor(rows, device=attention_mask.device,
                                  dtype=torch_module.long)
    col_idx = torch_module.tensor(tuple(selected_positions),
                                  device=attention_mask.device,
                                  dtype=torch_module.long)
    current = attention_mask[0, 0][row_idx[:, None], col_idx[None, :]]
    if not bool(torch_module.isfinite(current).all()) or not bool((current == 0).all()):
        raise ValueError("selected_video_key_not_visible_zero_in_generation_rows")
    result = attention_mask.clone()
    result[0, 0][row_idx[:, None], col_idx[None, :]] += float(logit_bias)
    if not bool(torch_module.isfinite(
            result[0, 0][row_idx[:, None], col_idx[None, :]]).all()):
        raise ValueError("biased_generation_attention_mask_nonfinite")
    return result


def install_generation_video_key_bias_hooks(*, model: Any,
        layer_indices: Sequence[int], selected_positions: Sequence[int],
        first_query_position: int, logit_bias: float,
        torch_module: Any, observations: list[dict] | None = None) -> list[Any]:
    """Bias frozen temporal keys during a verified DynamicCache generation.

    A stateful contract requires one square Q=K prefill followed only by
    cached Q=1 decode calls whose key length increases by exactly one each
    step, across selected layers in ascending order. This rejects
    static-capacity masks, multi-token cache updates, and cropped/rebased key
    axes rather than treating K-Q as an absolute offset.
    """
    try:
        layers = model.model.language_model.layers
    except AttributeError:
        raise ValueError("qwen_language_layers_unavailable") from None
    if (not isinstance(layer_indices, Sequence) or isinstance(layer_indices, (str, bytes))
            or not layer_indices
            or any(type(i) is not int or i < 0 or i >= len(layers) for i in layer_indices)
            or tuple(sorted(set(layer_indices))) != tuple(layer_indices)):
        raise ValueError("intervention_layer_indices_invalid")
    frozen_layers = tuple(layer_indices)
    frozen_positions = tuple(selected_positions)
    state = {"last_key_length": None, "active_shape": None, "seen_layers": set()}
    handles = []
    try:
        for layer_index in frozen_layers:
            attention = getattr(layers[layer_index], "self_attn", None)
            register = getattr(attention, "register_forward_pre_hook", None)
            if not callable(register):
                raise ValueError("self_attention_pre_hook_unavailable")
            def pre_hook(_module, args, kwargs, *, selected_layer_index=layer_index):
                if "attention_mask" not in kwargs:
                    raise ValueError("attention_mask_keyword_missing")
                kwargs = dict(kwargs)
                mask = kwargs["attention_mask"]
                if (not torch_module.is_tensor(mask) or mask.ndim != 4
                        or tuple(mask.shape[:2]) != (1, 1)):
                    raise ValueError("generation_attention_mask_shape_invalid")
                shape = (int(mask.shape[-2]), int(mask.shape[-1]))
                active = state["active_shape"]
                if active is None:
                    qlen, klen = shape
                    previous = state["last_key_length"]
                    if previous is None:
                        if qlen != klen:
                            raise ValueError("dynamic_cache_prefill_must_be_square")
                    elif qlen != 1 or klen != previous + 1:
                        raise ValueError("dynamic_cache_decode_length_not_incremental")
                    state["active_shape"] = shape
                elif active != shape:
                    raise ValueError("generation_layer_mask_shapes_disagree")
                if selected_layer_index in state["seen_layers"]:
                    raise ValueError("generation_layer_called_twice_before_step_complete")
                changed = bias_video_keys_during_generation(
                    attention_mask=mask,
                    selected_positions=frozen_positions,
                    first_query_position=first_query_position,
                    logit_bias=logit_bias, torch_module=torch_module)
                if observations is not None:
                    qlen, klen = shape
                    effective_bias = float(torch_module.tensor(
                        logit_bias, dtype=mask.dtype).detach().cpu())
                    modified_cells = 0
                    max_error = 0.0
                    for query_index in range(qlen):
                        absolute_query = query_index if qlen > 1 else klen - 1
                        if absolute_query < first_query_position:
                            continue
                        for key_position in frozen_positions:
                            if key_position >= absolute_query or key_position >= klen:
                                continue
                            delta = float((changed[0, 0, query_index, key_position]
                                           - mask[0, 0, query_index, key_position]).detach().cpu())
                            max_error = max(max_error, abs(delta - effective_bias))
                            modified_cells += 1
                    observations.append({
                        "layer_index": selected_layer_index,
                        "query_length": qlen,
                        "key_length": klen,
                        "effective_bias": effective_bias,
                        "modified_cells": modified_cells,
                        "max_abs_delta_error": max_error,
                        "verified": modified_cells > 0 and max_error <= 1e-6,
                    })
                    if not observations[-1]["verified"]:
                        raise ValueError("generation_intervention_mask_delta_unverified")
                kwargs["attention_mask"] = changed
                state["seen_layers"].add(selected_layer_index)
                if len(state["seen_layers"]) == len(frozen_layers):
                    state["last_key_length"] = shape[1]
                    state["active_shape"] = None
                    state["seen_layers"].clear()
                return args, kwargs
            try:
                handles.append(register(pre_hook, with_kwargs=True))
            except TypeError:
                raise ValueError("pytorch_keyword_pre_hook_unavailable") from None
    except Exception:
        remove_hooks(handles)
        raise
    return handles


def remove_hooks(handles: Sequence[Any]) -> None:
    """Remove every registered treatment hook, even if one removal fails."""
    first_error = None
    for handle in reversed(tuple(handles)):
        try:
            handle.remove()
        except Exception as exc:  # pragma: no cover - defensive cleanup path
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise RuntimeError("intervention_hook_cleanup_failed") from first_error


def _validate_selected_positions(selected_positions: Sequence[int],
                                query_position: int, logit_bias: float) -> None:
    if (not isinstance(selected_positions, Sequence)
            or isinstance(selected_positions, (str, bytes))
            or not selected_positions
            or any(type(position) is not int for position in selected_positions)
            or tuple(sorted(set(selected_positions))) != tuple(selected_positions)
            or type(query_position) is not int or query_position <= 0
            or any(position < 0 or position >= query_position
                   for position in selected_positions)
            or isinstance(logit_bias, bool)
            or not isinstance(logit_bias, (int, float))
            or not math.isfinite(float(logit_bias)) or float(logit_bias) <= 0.0):
        raise ValueError("video_key_bias_parameters_invalid")
