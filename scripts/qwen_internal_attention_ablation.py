"""Fixed-query internal ablation of selected Qwen visual attention keys.

This is a narrowly scoped attention-access intervention: it masks selected
visual key positions only on one answer-query row. It does not prove complete
evidence necessity, map timestamps to tokens, score faithfulness, or authorize
data use. Callers must supply independently frozen, verified positions.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any


def ablate_selected_video_keys(*, attention_mask: Any,
                               selected_positions: Sequence[int],
                               query_position: int,
                               torch_module: Any) -> Any:
    """Set selected, currently visible causal keys to ``-inf`` at one query.

    The operation requires Qwen's eager additive 4-D mask convention and a
    square prompt mask. The caller must set ``use_cache=False``; shape checks
    alone cannot authenticate cache state. It preserves every other mask
    value and fails closed if the selection leaves no zero-valued causal key.
    """
    if torch_module is None or not callable(getattr(torch_module, "is_tensor", None)):
        raise ValueError("torch_module_invalid")
    if not torch_module.is_tensor(attention_mask):
        raise ValueError("attention_mask_tensor_required")
    if (attention_mask.ndim != 4
            or tuple(attention_mask.shape[:2]) != (1, 1)
            or not attention_mask.dtype.is_floating_point):
        raise ValueError("additive_attention_mask_shape_or_dtype_invalid")
    if bool(torch_module.isnan(attention_mask).any()):
        raise ValueError("attention_mask_contains_nan")
    if bool(torch_module.isposinf(attention_mask).any()):
        raise ValueError("attention_mask_contains_positive_infinity")
    query_length, key_length = map(int, attention_mask.shape[-2:])
    if (type(query_position) is not int or query_position < 0
            or query_position >= query_length or key_length != query_length):
        raise ValueError("uncached_query_position_required")
    if (not isinstance(selected_positions, Sequence)
            or isinstance(selected_positions, (str, bytes))
            or not selected_positions
            or any(type(position) is not int for position in selected_positions)
            or tuple(sorted(set(selected_positions))) != tuple(selected_positions)
            or any(position < 0 or position >= query_position
                   for position in selected_positions)):
        raise ValueError("selected_causal_video_keys_invalid")
    frozen_positions = tuple(selected_positions)

    result = attention_mask.clone()
    selected = torch_module.tensor(list(frozen_positions),
                                   device=result.device, dtype=torch_module.long)
    current = result[0, 0, query_position, selected]
    if not bool(torch_module.isfinite(current).all()):
        raise ValueError("selected_key_not_currently_visible")
    if not bool((current == 0).all()):
        raise ValueError("selected_key_mask_value_not_zero")
    result[0, 0, query_position, selected] = float("-inf")
    if bool(torch_module.isnan(result).any()):
        raise ValueError("ablated_attention_mask_contains_nan")
    row = result[0, 0, query_position, :query_position + 1]
    if not bool((row == 0).any()):
        raise ValueError("ablation_removed_all_causal_keys")
    return result


def compose_video_key_bias_and_ablation(*, attention_mask: Any,
                                       biased_positions: Sequence[int],
                                       ablated_positions: Sequence[int],
                                       query_position: int,
                                       logit_bias: float,
                                       torch_module: Any) -> Any:
    """Compose a positive key bias with key ablation; ablation takes precedence.

    Both component operations validate against the same original additive mask.
    The selected bias is then applied, after which ablated positions are set to
    ``-inf``. This avoids order-dependent hook behavior when target and
    evidence/control key sets overlap.
    """
    if not biased_positions and not ablated_positions:
        raise ValueError("at_least_one_key_operation_required")
    result = attention_mask
    if biased_positions:
        from qwen_visual_attention_boost import bias_selected_video_keys
        result = bias_selected_video_keys(
            attention_mask=attention_mask,
            selected_positions=biased_positions,
            query_position=query_position,
            logit_bias=logit_bias,
            torch_module=torch_module)
    if ablated_positions:
        ablated = ablate_selected_video_keys(
            attention_mask=attention_mask,
            selected_positions=ablated_positions,
            query_position=query_position,
            torch_module=torch_module)
        indices = torch_module.tensor(
            list(ablated_positions), device=result.device, dtype=torch_module.long)
        result = result.clone()
        result[0, 0, query_position, indices] = ablated[
            0, 0, query_position, indices]
    if bool(torch_module.isnan(result).any()) or bool(torch_module.isposinf(result).any()):
        raise ValueError("composed_attention_mask_invalid")
    return result


def install_video_key_ablation_hooks(*, model: Any,
                                     layer_indices: Sequence[int],
                                     selected_positions: Sequence[int],
                                     query_position: int,
                                     torch_module: Any) -> list[Any]:
    """Install the same fixed-query key-access mask at selected self-attn layers."""
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise ValueError("qwen_language_layers_unavailable")
    if (not isinstance(layer_indices, Sequence)
            or isinstance(layer_indices, (str, bytes))
            or not layer_indices
            or any(type(index) is not int or index < 0 or index >= len(layers)
                   for index in layer_indices)
            or len(set(layer_indices)) != len(layer_indices)):
        raise ValueError("intervention_layer_indices_invalid")
    if (not isinstance(selected_positions, Sequence)
            or isinstance(selected_positions, (str, bytes))
            or not selected_positions
            or any(type(position) is not int for position in selected_positions)
            or tuple(sorted(set(selected_positions))) != tuple(selected_positions)
            or type(query_position) is not int or query_position <= 0
            or any(position < 0 or position >= query_position
                   for position in selected_positions)):
        raise ValueError("selected_causal_video_keys_invalid")
    frozen_positions = tuple(selected_positions)
    handles = []
    try:
        for layer_index in layer_indices:
            attention = getattr(layers[layer_index], "self_attn", None)
            register = getattr(attention, "register_forward_pre_hook", None)
            if not callable(register):
                raise ValueError("self_attention_pre_hook_unavailable")

            def pre_hook(_module, args, kwargs):
                if "attention_mask" not in kwargs:
                    raise ValueError("attention_mask_keyword_missing")
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = ablate_selected_video_keys(
                    attention_mask=kwargs["attention_mask"],
                    selected_positions=frozen_positions,
                    query_position=query_position,
                    torch_module=torch_module)
                return args, kwargs

            try:
                handles.append(register(pre_hook, with_kwargs=True))
            except TypeError:
                raise ValueError("pytorch_keyword_pre_hook_unavailable") from None
    except Exception:
        remove_hooks(handles)
        raise
    return handles


def install_video_key_intervention_hooks(*, model: Any,
                                         layer_indices: Sequence[int],
                                         biased_positions: Sequence[int],
                                         ablated_positions: Sequence[int],
                                         query_position: int,
                                         logit_bias: float,
                                         torch_module: Any) -> list[Any]:
    """Install a single pre-hook composing bias and ablation per layer."""
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
    if type(query_position) is not int or query_position <= 0:
        raise ValueError("uncached_query_position_required")
    for name, positions in (("biased", biased_positions), ("ablated", ablated_positions)):
        if (not isinstance(positions, Sequence) or isinstance(positions, (str, bytes))
                or any(type(position) is not int for position in positions)
                or tuple(sorted(set(positions))) != tuple(positions)
                or any(position < 0 or position >= query_position for position in positions)):
            raise ValueError(f"{name}_video_positions_invalid")
    if not biased_positions and not ablated_positions:
        raise ValueError("at_least_one_key_operation_required")
    if biased_positions and (isinstance(logit_bias, bool)
                             or not isinstance(logit_bias, (int, float))
                             or not math.isfinite(float(logit_bias)) or logit_bias <= 0):
        raise ValueError("positive_finite_logit_bias_required")

    handles = []
    try:
        for layer_index in layer_indices:
            attention = getattr(layers[layer_index], "self_attn", None)
            register = getattr(attention, "register_forward_pre_hook", None)
            if not callable(register):
                raise ValueError("self_attention_pre_hook_unavailable")

            def pre_hook(_module, args, kwargs):
                if "attention_mask" not in kwargs:
                    raise ValueError("attention_mask_keyword_missing")
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = compose_video_key_bias_and_ablation(
                    attention_mask=kwargs["attention_mask"],
                    biased_positions=tuple(biased_positions),
                    ablated_positions=tuple(ablated_positions),
                    query_position=query_position, logit_bias=logit_bias,
                    torch_module=torch_module)
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
    """Remove all registered hooks and report cleanup failure after attempting each."""
    first_error = None
    for handle in reversed(tuple(handles)):
        try:
            handle.remove()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise RuntimeError("ablation_hook_cleanup_failed") from first_error
