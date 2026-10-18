"""Capture selected question-query rows from one Qwen attention forward.

This records which module objects were hooked and verifies one invocation per
expected layer during one caller-supplied forward. It is an execution
diagnostic, not cryptographic proof of model/source authenticity and never
authorizes scoring.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from qta_attention_capture_contract import LAYERS, MAX_CAPTURE_BYTES
from qta_attention_capture_sidecars import (
    _ATTENTION_SOURCE_DTYPES, _ROW_SUM_TOLERANCE,
)


def capture_qta_attention_forward_once(*, model: Any,
                                        forward: Callable[[], Any],
                                        question_positions: Sequence[int],
                                        input_length: int,
                                        torch_module: Any) -> tuple[Any, dict[int, Any], dict[str, Any]]:
    if (not callable(forward) or torch_module is None
            or type(input_length) is not int or input_length < 1
            or not isinstance(question_positions, Sequence)
            or isinstance(question_positions, (str, bytes)) or not question_positions
            or any(type(position) is not int or position < 0
                   or position >= input_length for position in question_positions)
            or list(question_positions) != list(range(question_positions[0],
                                                       question_positions[-1] + 1))):
        raise ValueError("qta_forward_capture_input_invalid")
    try:
        layers = model.model.language_model.layers
    except AttributeError:
        raise ValueError("qta_forward_capture_model_path_invalid") from None
    if len(layers) < max(LAYERS):
        raise ValueError("qta_forward_capture_layer_count_invalid")

    module_by_layer = {layer: layers[layer - 1].self_attn for layer in LAYERS}
    if len({id(module) for module in module_by_layer.values()}) != len(LAYERS):
        raise ValueError("qta_forward_capture_layer_module_alias")
    heads_by_layer: dict[int, int] = {}
    rows: dict[int, Any] = {}
    calls = {layer: 0 for layer in LAYERS}
    source_dtypes: dict[int, str] = {}
    handles = []
    result = None
    row_sum_deviations: dict[int, float] = {}
    try:
        for layer, expected_module in module_by_layer.items():
            def capture(_module, _inputs, output, *, selected_layer=layer,
                        expected=expected_module):
                if _module is not expected:
                    raise RuntimeError("qta_forward_capture_module_identity_mismatch")
                calls[selected_layer] += 1
                if calls[selected_layer] != 1:
                    raise RuntimeError("qta_forward_capture_layer_called_multiple_times")
                if (not isinstance(output, tuple) or len(output) < 2
                        or not torch_module.is_tensor(output[1])
                        or output[1].ndim != 4 or int(output[1].shape[0]) != 1
                        or tuple(output[1].shape[-2:]) != (input_length, input_length)):
                    raise RuntimeError("qta_forward_capture_attention_output_invalid")
                attention = output[1]
                heads = int(attention.shape[1])
                source_dtype = str(attention.dtype).removeprefix("torch.")
                if source_dtype not in _ATTENTION_SOURCE_DTYPES:
                    raise RuntimeError("qta_forward_capture_attention_source_dtype_invalid")
                if source_dtypes and source_dtype != next(iter(source_dtypes.values())):
                    raise RuntimeError("qta_forward_capture_attention_source_dtype_mismatch")
                source_dtypes[selected_layer] = source_dtype
                projected = len(LAYERS) * heads * len(question_positions) * input_length * 4
                if projected > MAX_CAPTURE_BYTES:
                    raise RuntimeError("qta_forward_capture_payload_too_large")
                selected = attention[0, :, list(question_positions), :].detach().to(
                    device="cpu", dtype=torch_module.float32).contiguous()
                if (not bool(torch_module.isfinite(selected).all())
                        or bool((selected < 0).any())):
                    raise RuntimeError("qta_forward_capture_attention_value_invalid")
                row_sums = selected.sum(dim=-1, dtype=torch_module.float64)
                row_sum_deviations[selected_layer] = float(
                    (row_sums - 1.0).abs().max().item())
                if row_sum_deviations[selected_layer] > _ROW_SUM_TOLERANCE[source_dtype]:
                    raise RuntimeError("qta_forward_capture_attention_row_not_normalized")
                if heads_by_layer and heads != next(iter(heads_by_layer.values())):
                    raise RuntimeError("qta_forward_capture_head_count_mismatch")
                heads_by_layer[selected_layer] = heads
                rows[selected_layer] = selected

            handles.append(module_by_layer[layer].register_forward_hook(capture))
        with torch_module.inference_mode():
            result = forward()
    finally:
        for handle in handles:
            handle.remove()

    if tuple(sorted(rows)) != tuple(LAYERS) or any(calls[layer] != 1 for layer in LAYERS):
        raise RuntimeError("qta_forward_capture_layer_set_incomplete")
    report = {
        "status": "one_forward_attention_rows_captured_non_authorizing",
        "forward_calls": 1,
        "layer_calls": {str(layer): calls[layer] for layer in LAYERS},
        "layer_indices_1_based": list(LAYERS),
        "layer_module_classes": {
            str(layer): (type(module_by_layer[layer]).__module__ + "."
                         + type(module_by_layer[layer]).__qualname__)
            for layer in LAYERS
        },
        "layer_module_paths": {
            str(layer): f"model.model.language_model.layers[{layer - 1}].self_attn"
            for layer in LAYERS
        },
        "question_query_count": len(question_positions),
        "input_length": input_length,
        "attention_row_sum_max_abs_deviation": max(row_sum_deviations.values()),
        "attention_source_dtype": next(iter(source_dtypes.values())),
        "attention_row_sum_tolerance": _ROW_SUM_TOLERANCE[
            next(iter(source_dtypes.values()))],
        "attention_rows_satisfy_source_dtype_normalization_tolerance": True,
        "attention_shape": [len(LAYERS), next(iter(heads_by_layer.values())),
                            len(question_positions), input_length],
        "attention_source_dtypes": {
            str(layer): source_dtypes[layer] for layer in LAYERS
        },
        "forward_source_authenticated": False,
        "attention_capture_authenticated": False,
        "scoring_authorized": False,
    }
    return result, rows, report


def aggregate_captured_qta_rows(*, rows: dict[int, Any],
                                group_positions: Sequence[Sequence[int]],
                                torch_module: Any) -> list[float]:
    """Aggregate already-selected [heads, question_queries, keys] rows."""
    if (not isinstance(rows, dict) or set(rows) != set(LAYERS)
            or not isinstance(group_positions, Sequence) or not group_positions
            or torch_module is None):
        raise ValueError("qta_captured_rows_input_invalid")
    groups = [list(group) for group in group_positions]
    flattened = [position for group in groups for position in group]
    if (any(not group or any(type(position) is not int or position < 0
                             for position in group)
            or group != list(range(group[0], group[-1] + 1)) for group in groups)
            or groups != sorted(groups, key=lambda group: group[0])
            or len(set(flattened)) != len(flattened)
            or len({len(group) for group in groups}) != 1):
        raise ValueError("qta_captured_rows_group_geometry_invalid")
    shape = None
    layer_scores = []
    for layer in LAYERS:
        attention = rows[layer]
        if (not torch_module.is_tensor(attention) or attention.ndim != 3
                or int(attention.shape[0]) < 1 or int(attention.shape[1]) < 1):
            raise ValueError("qta_captured_rows_tensor_shape_invalid")
        current = tuple(attention.shape)
        if shape is None:
            shape = current
        elif current != shape:
            raise ValueError("qta_captured_rows_tensor_shape_mismatch")
        if max(flattened) >= int(attention.shape[2]):
            raise ValueError("qta_captured_rows_group_out_of_bounds")
        values = attention.to(device="cpu", dtype=torch_module.float32)
        if (not bool(torch_module.isfinite(values).all())
                or bool((values < 0).any())):
            raise ValueError("qta_captured_rows_attention_value_invalid")
        averaged = values.mean(dim=(0, 1))
        layer_scores.append([
            float(averaged[group].sum().item()) for group in groups
        ])
    result = [sum(scores[index] for scores in layer_scores) / len(layer_scores)
              for index in range(len(groups))]
    return result
