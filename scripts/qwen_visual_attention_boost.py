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
