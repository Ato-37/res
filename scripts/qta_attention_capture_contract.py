"""Synthetic attention-byte contract for QTA selection (non-authorizing).

The contract binds caller-supplied attention bytes to a batch and recomputes
question-to-temporal-group masses. It does not authenticate the producing
forward, model hook, processor, or runtime. The serialized query axis is packed
in the exact order of the bound ``question_positions`` list.
"""
from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = 1
PURPOSE = "qta_attention_capture_bytes_contract_non_authorizing"
LAYERS = list(range(5, 17))
MAX_CAPTURE_BYTES = 128 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BINDING_FIELDS = {
    "record_token", "row_binding_sha256", "typed_batch_sha256",
    "baseline_pq_row_sha256", "prompt_sha256", "input_ids_sha256",
    "model_sha256", "processor_sha256", "tokenizer_sha256",
    "runtime_sha256", "question_positions", "temporal_groups",
}
_CAPTURE_KEYS = _BINDING_FIELDS | {
    "schema_version", "purpose", "input_length", "layers_1_based",
    "attention_shape", "attention_f32_be_hex", "attention_bytes_sha256",
    "layer_group_masses_f32_be_hex", "forward_source_authenticated",
    "attention_capture_authenticated", "model_load_permitted",
    "scoring_authorized", "execution_authorization",
}


def _fail(code: str):
    raise ValueError(code)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _positions(value: Any, name: str, *, upper: int,
               contiguous: bool = False) -> list[int]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or not value or any(type(x) is not int or x < 0 or x >= upper for x in value)
            or len(set(value)) != len(value)):
        _fail(f"capture_{name}_invalid")
    if contiguous and list(value) != list(range(value[0], value[-1] + 1)):
        _fail(f"capture_{name}_noncontiguous")
    return list(value)


def _groups(value: Any, *, upper: int) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        _fail("capture_temporal_groups_invalid")
    groups = []
    seen = set()
    for group in value:
        positions = _positions(group, "temporal_group", upper=upper)
        if positions != list(range(positions[0], positions[-1] + 1)):
            _fail("capture_temporal_group_noncontiguous")
        if seen.intersection(positions):
            _fail("capture_temporal_groups_overlap")
        seen.update(positions)
        groups.append(positions)
    if groups != sorted(groups, key=lambda group: group[0]) or len(
            {len(group) for group in groups}) != 1:
        _fail("capture_temporal_group_order_or_size_invalid")
    return groups


def _pack_attention(attention: Any, *, query_count: int, input_length: int):
    if not isinstance(attention, Sequence) or isinstance(attention, (str, bytes)) \
            or len(attention) != len(LAYERS):
        _fail("capture_layer_count_invalid")
    first_layer = attention[0]
    if not isinstance(first_layer, Sequence) or isinstance(first_layer, (str, bytes)):
        _fail("capture_head_shape_invalid")
    heads = len(first_layer)
    if heads < 1 or len(LAYERS) * heads * query_count * input_length * 4 > MAX_CAPTURE_BYTES:
        _fail("capture_attention_payload_too_large")
    packed = bytearray()
    for layer in attention:
        if not isinstance(layer, Sequence) or isinstance(layer, (str, bytes)) or not layer:
            _fail("capture_head_shape_invalid")
        if len(layer) != heads:
            _fail("capture_head_shape_invalid")
        for head in layer:
            if (not isinstance(head, Sequence) or isinstance(head, (str, bytes))
                    or len(head) != query_count):
                _fail("capture_query_shape_invalid")
            for query in head:
                if (not isinstance(query, Sequence) or isinstance(query, (str, bytes))
                        or len(query) != input_length):
                    _fail("capture_key_shape_invalid")
                for value in query:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        _fail("capture_attention_value_invalid")
                    value = float(value)
                    if not math.isfinite(value) or value < 0:
                        _fail("capture_attention_value_invalid")
                    try:
                        raw = struct.pack(">f", value)
                    except (OverflowError, struct.error):
                        _fail("capture_attention_value_invalid")
                    rounded = struct.unpack(">f", raw)[0]
                    if not math.isfinite(rounded):
                        _fail("capture_attention_value_invalid")
                    packed.extend(raw)
    shape = [len(LAYERS), heads, query_count, input_length]
    return shape, bytes(packed)


def _group_masses(attention, *, shape, query_positions, groups):
    # Mirror the selector's float32 mean and indexed float32 group sum.
    import torch

    layer_count, head_count, query_count, input_length = shape
    masses = []
    for layer_i in range(layer_count):
        layer = torch.tensor(attention[layer_i], dtype=torch.float32)
        averaged = layer.mean(dim=(0, 1))
        row = []
        for group in groups:
            row.append(float(averaged[list(group)].sum().item()))
        masses.append(row)
    return masses


def build_attention_capture(*, binding: Mapping[str, Any], attention: Any,
                            input_length: int) -> dict[str, Any]:
    if not isinstance(binding, Mapping) or set(binding) != _BINDING_FIELDS:
        _fail("capture_binding_schema_invalid")
    for field in ("record_token", "row_binding_sha256", "typed_batch_sha256",
                  "baseline_pq_row_sha256", "prompt_sha256", "input_ids_sha256",
                  "model_sha256", "processor_sha256", "tokenizer_sha256",
                  "runtime_sha256"):
        if not isinstance(binding[field], str) or not _SHA256.fullmatch(binding[field]):
            _fail("capture_binding_digest_invalid")
    if type(input_length) is not int or input_length < 1:
        _fail("capture_input_length_invalid")
    question_positions = _positions(binding["question_positions"], "question_positions",
                                    upper=input_length, contiguous=True)
    groups = _groups(binding["temporal_groups"], upper=input_length)
    if max(group[-1] for group in groups) >= min(question_positions):
        _fail("capture_question_group_order_invalid")
    shape, raw = _pack_attention(attention, query_count=len(question_positions),
                                 input_length=input_length)
    decoded = _unpack_attention(raw, shape)
    if any(abs(sum(query) - 1.0) > 2e-4 for layer in decoded
           for head in layer for query in head):
        _fail("capture_attention_row_not_normalized")
    masses = _group_masses(decoded, shape=shape,
                           query_positions=list(range(len(question_positions))), groups=groups)
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        **dict(binding),
        "input_length": input_length,
        "layers_1_based": LAYERS,
        "attention_shape": shape,
        "attention_f32_be_hex": raw.hex(),
        "attention_bytes_sha256": _digest(raw),
        "layer_group_masses_f32_be_hex": [
            [struct.pack(">f", value).hex() for value in row] for row in masses],
        "forward_source_authenticated": False,
        "attention_capture_authenticated": False,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
    }


def _unpack_attention(raw: bytes, shape: Sequence[int]):
    count = math.prod(shape)
    if len(raw) != count * 4:
        _fail("capture_attention_byte_length_invalid")
    values = (item[0] for item in struct.iter_unpack(">f", raw))
    layer_count, head_count, query_count, input_length = shape
    layers = []
    for _ in range(layer_count):
        heads = []
        for _ in range(head_count):
            queries = []
            for _ in range(query_count):
                queries.append([next(values) for _ in range(input_length)])
            heads.append(queries)
        layers.append(heads)
    return layers


def validate_attention_capture(capture: Any, *, expected_binding: Mapping[str, Any],
                               input_length: int) -> dict[str, Any]:
    if (not isinstance(capture, Mapping) or set(capture) != _CAPTURE_KEYS
            or type(capture.get("schema_version")) is not int
            or capture.get("schema_version") != SCHEMA_VERSION
            or capture.get("purpose") != PURPOSE
            or capture.get("forward_source_authenticated") is not False
            or capture.get("attention_capture_authenticated") is not False
            or capture.get("model_load_permitted") is not False
            or capture.get("scoring_authorized") is not False
            or capture.get("execution_authorization") != "not_granted"):
        _fail("capture_schema_or_authority_invalid")
    if (type(input_length) is not int or input_length < 1
            or not isinstance(expected_binding, Mapping)
            or set(expected_binding) != _BINDING_FIELDS):
        _fail("capture_expected_binding_schema_invalid")
    for field in ("record_token", "row_binding_sha256", "typed_batch_sha256",
                  "baseline_pq_row_sha256", "prompt_sha256", "input_ids_sha256",
                  "model_sha256", "processor_sha256", "tokenizer_sha256",
                  "runtime_sha256"):
        if not isinstance(expected_binding[field], str) or not _SHA256.fullmatch(
                expected_binding[field]):
            _fail("capture_expected_binding_digest_invalid")
    for field in _BINDING_FIELDS:
        if capture.get(field) != expected_binding[field]:
            _fail("capture_binding_mismatch")
    if capture.get("input_length") != input_length:
        _fail("capture_input_length_mismatch")
    qpos = _positions(expected_binding["question_positions"], "question_positions",
                      upper=input_length, contiguous=True)
    groups = _groups(expected_binding["temporal_groups"], upper=input_length)
    if max(group[-1] for group in groups) >= min(qpos):
        _fail("capture_question_group_order_invalid")
    shape = capture.get("attention_shape")
    if (not isinstance(shape, list) or len(shape) != 4
            or any(type(value) is not int or value < 1 for value in shape)
            or shape[0] != len(LAYERS) or shape[2] != len(qpos)
            or shape[3] != input_length or capture.get("layers_1_based") != LAYERS
            or math.prod(shape) * 4 > MAX_CAPTURE_BYTES):
        _fail("capture_attention_shape_invalid")
    encoded = capture.get("attention_f32_be_hex")
    if not isinstance(encoded, str):
        _fail("capture_attention_encoding_invalid")
    if len(encoded) // 2 > MAX_CAPTURE_BYTES:
        _fail("capture_attention_payload_too_large")
    if len(encoded) % 8 or any(c not in "0123456789abcdef" for c in encoded):
        _fail("capture_attention_encoding_invalid")
    raw = bytes.fromhex(encoded)
    if capture.get("attention_bytes_sha256") != _digest(raw):
        _fail("capture_attention_digest_mismatch")
    attention = _unpack_attention(raw, shape)
    if any(not math.isfinite(value) or value < 0 for layer in attention
           for head in layer for query in head for value in query):
        _fail("capture_attention_value_invalid")
    for layer in attention:
        for head in layer:
            for query in head:
                if abs(sum(query) - 1.0) > 2e-4:
                    _fail("capture_attention_row_not_normalized")
    masses = _group_masses(attention, shape=shape,
                           query_positions=list(range(len(qpos))), groups=groups)
    expected_masses = [[struct.pack(">f", value).hex() for value in row]
                       for row in masses]
    if capture.get("layer_group_masses_f32_be_hex") != expected_masses:
        _fail("capture_group_mass_recomputation_mismatch")
    return {
        "status": "capture_bytes_and_group_masses_reproduced_non_authorizing",
        "attention_capture_authenticated": False,
        "forward_source_authenticated": False,
        "layer_count": len(LAYERS),
        "head_count": shape[1],
        "temporal_group_count": len(groups),
        "execution_authorization": "not_granted",
        "scoring_authorized": False,
    }


def validate_capture_selection_bridge(*, capture: Any,
                                      expected_capture_binding: Mapping[str, Any],
                                      input_length: int,
                                      selection_provenance: Any,
                                      expected_selection_bindings: Mapping[str, Any],
                                      plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Join captured attention bytes to the masses used by selection arithmetic.

    This proves byte/mass/selection consistency only. It does not authenticate
    that the bytes came from a Qwen forward or authorize scoring.
    """
    capture_report = validate_attention_capture(
        capture, expected_binding=expected_capture_binding,
        input_length=input_length)
    if not isinstance(selection_provenance, Mapping):
        _fail("capture_selection_provenance_invalid")
    if not isinstance(expected_selection_bindings, Mapping):
        _fail("capture_selection_expected_bindings_invalid")
    from qta_selection_provenance import (
        validate_qta_selection_provenance,
    )

    validate_qta_selection_provenance(
        selection_provenance, expected_bindings=expected_selection_bindings, plan=plan)
    common_fields = (
        "record_token", "row_binding_sha256", "typed_batch_sha256",
        "baseline_pq_row_sha256", "prompt_sha256", "input_ids_sha256",
        "question_positions", "temporal_groups",
    )
    if any(selection_provenance.get(field) != capture.get(field)
           for field in common_fields):
        _fail("capture_selection_binding_mismatch")
    if any(expected_selection_bindings.get(field) != capture.get(field)
           for field in common_fields):
        _fail("capture_selection_expected_binding_mismatch")
    if (selection_provenance.get("selector_layers_1_based")
            != capture.get("layers_1_based")
            or selection_provenance.get("layer_group_mass_f32_be_hex")
            != capture.get("layer_group_masses_f32_be_hex")):
        _fail("capture_selection_group_mass_mismatch")
    return {
        "status": "capture_bytes_joined_to_selection_masses_non_authorizing",
        "attention_bytes_sha256": capture["attention_bytes_sha256"],
        "layer_count": capture_report["layer_count"],
        "temporal_group_count": capture_report["temporal_group_count"],
        "forward_source_authenticated": False,
        "attention_capture_authenticated": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
    }
