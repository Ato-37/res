"""Non-authorizing binary sidecar index for QTA attention captures.

The small canonical JSON index is suitable for a signed receipt role; large
float32 attention payloads stay in content-addressed binary sidecars. Hash and
mass checks establish byte consistency only, not the source model forward.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from qta_attention_capture_contract import (
    LAYERS, MAX_CAPTURE_BYTES, _BINDING_FIELDS, _attention_array,
    _group_masses, _groups, _pack_attention, _positions,
)


SCHEMA_VERSION = 1
PURPOSE = "qta_attention_capture_binary_sidecar_index_non_authorizing"
MAX_INDEX_BYTES = 24_000_000
MAX_TOTAL_SIDECAR_BYTES = 384 * 1024 * 1024
_ATTENTION_SOURCE_DTYPES = {"float32", "bfloat16"}
_ROW_SUM_TOLERANCE = {"float32": 2e-4, "bfloat16": 0.0041}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INDEX_KEYS = {"schema_version", "purpose", "record_count", "records",
               "forward_source_authenticated", "attention_capture_authenticated",
               "model_load_permitted",
               "scoring_authorized", "execution_authorization"}
_ROW_KEYS = _BINDING_FIELDS | {
    "input_length", "layers_1_based", "attention_shape",
    "attention_source_dtype",
    "attention_bytes_length", "attention_bytes_sha256",
    "layer_group_masses_f32_be_hex", "model_load_permitted",
    "scoring_authorized", "execution_authorization",
}


def _fail(code: str):
    raise ValueError(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _strict_json(raw: bytes) -> Any:
    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                _fail("capture_sidecar_duplicate_json_key")
            output[key] = value
        return output

    try:
        return json.loads(raw.decode("ascii"), object_pairs_hook=unique_object,
                          parse_constant=lambda _value: _fail(
                              "capture_sidecar_nonfinite_json_value"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("capture_sidecar_index_json_invalid")


def _validate_binding(binding: Any, input_length: int) -> tuple[list[int], list[list[int]]]:
    if not isinstance(binding, Mapping) or set(binding) != _BINDING_FIELDS:
        _fail("capture_sidecar_binding_schema_invalid")
    for key in _BINDING_FIELDS - {"question_positions", "temporal_groups"}:
        if not isinstance(binding[key], str) or not _SHA256.fullmatch(binding[key]):
            _fail("capture_sidecar_binding_digest_invalid")
    question_positions = _positions(binding["question_positions"], "question_positions",
                                    upper=input_length, contiguous=True)
    groups = _groups(binding["temporal_groups"], upper=input_length)
    if groups[-1][-1] >= question_positions[0]:
        _fail("capture_sidecar_question_group_order_invalid")
    return question_positions, groups


def _mass_hex(attention_array, shape, groups) -> list[list[str]]:
    masses = _group_masses(attention_array, shape=shape,
                           query_positions=range(shape[2]), groups=groups)
    import struct
    return [[struct.pack(">f", value).hex() for value in row] for row in masses]


def pack_qta_attention_sidecar_bytes(*, attention_layers: Mapping[int, Any],
                                    question_positions: Sequence[int],
                                    input_length: int) -> tuple[list[int], bytes, str]:
    """Pack selected query rows from Qwen attention tensors without Python floats."""
    if (not isinstance(attention_layers, Mapping)
            or set(attention_layers) != set(LAYERS)):
        _fail("capture_sidecar_layer_count_invalid")
    if (type(input_length) is not int or input_length < 1
            or not isinstance(question_positions, Sequence)
            or isinstance(question_positions, (str, bytes)) or not question_positions
            or any(type(position) is not int or position < 0
                   or position >= input_length for position in question_positions)
            or list(question_positions) != list(range(question_positions[0],
                                                       question_positions[-1] + 1))):
        _fail("capture_sidecar_query_positions_invalid")
    import numpy as np

    packed = bytearray()
    head_count = None
    source_dtype = None
    for layer_index in LAYERS:
        layer = attention_layers[layer_index]
        tensor = layer.detach() if hasattr(layer, "detach") else layer
        shape = tuple(getattr(tensor, "shape", ()))
        if len(shape) == 4 and shape[0] == 1:
            tensor = tensor[0]
            shape = tuple(tensor.shape)
        if (len(shape) != 3 or shape[1] != input_length
                or shape[2] != input_length or shape[0] < 1):
            _fail("capture_sidecar_attention_tensor_shape_invalid")
        current_dtype = str(getattr(tensor, "dtype", "float32")).removeprefix("torch.")
        if current_dtype not in _ATTENTION_SOURCE_DTYPES:
            _fail("capture_sidecar_attention_source_dtype_invalid")
        if source_dtype is not None and current_dtype != source_dtype:
            _fail("capture_sidecar_attention_source_dtype_mismatch")
        source_dtype = current_dtype
        if head_count is None:
            head_count = shape[0]
            projected_bytes = len(LAYERS) * head_count * len(question_positions) * input_length * 4
            if projected_bytes > MAX_CAPTURE_BYTES:
                _fail("capture_sidecar_payload_too_large")
        elif shape[0] != head_count:
            _fail("capture_sidecar_head_count_mismatch")
        selected = tensor[:, list(question_positions), :]
        if hasattr(selected, "to"):
            selected = selected.to(device="cpu", dtype=__import__(
                "torch").float32).contiguous().numpy()
        else:
            selected = np.asarray(selected, dtype=np.float32)
        selected = np.ascontiguousarray(selected, dtype=np.float32)
        if (not np.isfinite(selected).all() or (selected < 0).any()):
            _fail("capture_sidecar_attention_tensor_value_invalid")
        packed.extend(selected.astype(">f4", copy=True).tobytes(order="C"))
        if len(packed) > MAX_CAPTURE_BYTES:
            _fail("capture_sidecar_payload_too_large")
    return ([len(LAYERS), head_count, len(question_positions), input_length],
            bytes(packed), source_dtype)


def build_qta_capture_sidecar_index(*, entries: Sequence[Mapping[str, Any]]) -> tuple[bytes, dict[str, bytes]]:
    """Build a canonical index and SHA-keyed raw sidecars from synthetic or captured rows."""
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        _fail("capture_sidecar_entries_invalid")
    rows = []
    sidecars: dict[str, bytes] = {}
    total = 0
    for entry in entries:
        if (not isinstance(entry, Mapping)
                or set(entry) not in ({"binding", "attention", "input_length"},
                                     {"binding", "attention_bytes", "attention_shape",
                                      "input_length", "attention_source_dtype"})):
            _fail("capture_sidecar_entry_schema_invalid")
        input_length = entry["input_length"]
        if type(input_length) is not int or input_length < 1:
            _fail("capture_sidecar_input_length_invalid")
        binding = entry["binding"]
        qpos, groups = _validate_binding(binding, input_length)
        if "attention" in entry:
            shape, raw = _pack_attention(entry["attention"], query_count=len(qpos),
                                         input_length=input_length)
            source_dtype = "float32"
        else:
            shape, raw = entry["attention_shape"], entry["attention_bytes"]
            source_dtype = entry["attention_source_dtype"]
            if (not isinstance(source_dtype, str)
                    or source_dtype not in _ATTENTION_SOURCE_DTYPES):
                _fail("capture_sidecar_attention_source_dtype_invalid")
            if not isinstance(raw, bytes):
                _fail("capture_sidecar_payload_bytes_invalid")
            if (not isinstance(shape, list) or len(shape) != 4
                    or any(type(value) is not int or value < 1 for value in shape)
                    or shape[0] != len(LAYERS) or shape[2] != len(qpos)
                    or shape[3] != input_length or shape[1] < 1):
                _fail("capture_sidecar_shape_invalid")
        if len(raw) > MAX_CAPTURE_BYTES:
            _fail("capture_sidecar_payload_too_large")
        total += len(raw)
        if total > MAX_TOTAL_SIDECAR_BYTES:
            _fail("capture_sidecar_bundle_too_large")
        array = _attention_array(raw, shape)
        import numpy as np
        if (not np.isfinite(array).all() or (array < 0).any()
                or (np.abs(array.sum(axis=-1, dtype=np.float64) - 1.0)
                    > _ROW_SUM_TOLERANCE[source_dtype]).any()):
            _fail("capture_sidecar_attention_row_invalid")
        digest = _sha(raw)
        if digest in sidecars and sidecars[digest] != raw:
            _fail("capture_sidecar_digest_collision")
        sidecars[digest] = raw
        rows.append({
            **dict(binding), "input_length": input_length,
            "layers_1_based": list(LAYERS), "attention_shape": shape,
            "attention_source_dtype": source_dtype,
            "attention_bytes_length": len(raw), "attention_bytes_sha256": digest,
            "layer_group_masses_f32_be_hex": _mass_hex(array, shape, groups),
            "model_load_permitted": False, "scoring_authorized": False,
            "execution_authorization": "not_granted",
        })
    index = {
        "schema_version": SCHEMA_VERSION, "purpose": PURPOSE,
        "record_count": len(rows), "records": rows,
        "forward_source_authenticated": False,
        "attention_capture_authenticated": False,
        "model_load_permitted": False, "scoring_authorized": False,
        "execution_authorization": "not_granted",
    }
    raw_index = _canonical(index) + b"\n"
    if len(raw_index) > MAX_INDEX_BYTES:
        _fail("capture_sidecar_index_too_large")
    return raw_index, sidecars


def validate_qta_capture_sidecar_index(*, index_raw: bytes,
                                       sidecars: Mapping[str, bytes],
                                       expected_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify sidecar bytes and exact selection masses for each expected plan row."""
    if (not isinstance(index_raw, bytes) or not index_raw
            or len(index_raw) > MAX_INDEX_BYTES):
        _fail("capture_sidecar_index_bytes_invalid")
    index = _strict_json(index_raw)
    if (not isinstance(index, Mapping) or set(index) != _INDEX_KEYS
            or _canonical(index) + b"\n" != index_raw
            or type(index.get("schema_version")) is not int
            or index["schema_version"] != SCHEMA_VERSION
            or index.get("purpose") != PURPOSE
            or index.get("forward_source_authenticated") is not False
            or index.get("attention_capture_authenticated") is not False
            or index.get("model_load_permitted") is not False
            or index.get("scoring_authorized") is not False
            or index.get("execution_authorization") != "not_granted"):
        _fail("capture_sidecar_index_schema_or_authority_invalid")
    rows = index.get("records")
    if (not isinstance(rows, list) or not isinstance(expected_rows, Sequence)
            or isinstance(expected_rows, (str, bytes)) or not rows
            or len(rows) != len(expected_rows) or type(index.get("record_count")) is not int
            or index["record_count"] != len(rows)):
        _fail("capture_sidecar_record_count_invalid")
    if not isinstance(sidecars, Mapping):
        _fail("capture_sidecar_payload_map_invalid")
    if any(not isinstance(row, Mapping) or set(row) != _ROW_KEYS for row in rows):
        _fail("capture_sidecar_record_schema_invalid")
    expected_sidecar_hashes = {row["attention_bytes_sha256"] for row in rows
                               if isinstance(row["attention_bytes_sha256"], str)}
    if set(sidecars) != expected_sidecar_hashes:
        _fail("capture_sidecar_payload_set_invalid")
    total = 0
    seen_rows = set()
    for row, expected in zip(rows, expected_rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
            _fail("capture_sidecar_record_schema_invalid")
        if (row.get("model_load_permitted") is not False
                or row.get("scoring_authorized") is not False
                or row.get("execution_authorization") != "not_granted"):
            _fail("capture_sidecar_record_authority_invalid")
        if (not isinstance(expected, Mapping) or set(expected) != {
                "expected_capture_binding", "input_length", "selection_provenance",
                "expected_selection_bindings", "expected_attention_source_dtype",
                "plan"}):
            _fail("capture_sidecar_expected_row_schema_invalid")
        expected_source_dtype = expected["expected_attention_source_dtype"]
        if (not isinstance(expected_source_dtype, str)
                or expected_source_dtype not in _ATTENTION_SOURCE_DTYPES):
            _fail("capture_sidecar_expected_source_dtype_invalid")
        binding = expected["expected_capture_binding"]
        input_length = expected["input_length"]
        if type(input_length) is not int or input_length < 1:
            _fail("capture_sidecar_expected_input_length_invalid")
        qpos, groups = _validate_binding(
            {key: row[key] for key in _BINDING_FIELDS}, input_length)
        if (not isinstance(binding, Mapping)
                or set(binding) != _BINDING_FIELDS):
            _fail("capture_sidecar_expected_binding_schema_invalid")
        for key in _BINDING_FIELDS - {"question_positions", "temporal_groups"}:
            if key in binding and (not isinstance(binding[key], str)
                                   or not _SHA256.fullmatch(binding[key])):
                _fail("capture_sidecar_expected_binding_digest_invalid")
        # Every capture binding field, including the runtime digest derived
        # from the signed pre-baseline sampler_model_runtime artifact, must
        # match the indexed row.
        if any(row.get(key) != binding[key] for key in binding):
            _fail("capture_sidecar_binding_mismatch")
        if binding["record_token"] in seen_rows:
            _fail("capture_sidecar_record_token_duplicate")
        seen_rows.add(binding["record_token"])
        shape = row.get("attention_shape")
        source_dtype = row.get("attention_source_dtype")
        if (not isinstance(source_dtype, str)
                or source_dtype not in _ATTENTION_SOURCE_DTYPES):
            _fail("capture_sidecar_attention_source_dtype_invalid")
        if source_dtype != expected_source_dtype:
            _fail("capture_sidecar_attention_source_dtype_mismatch")
        if (type(input_length) is not int or input_length < 1
                or row.get("input_length") != input_length
                or row.get("layers_1_based") != list(LAYERS)
                or not isinstance(shape, list) or len(shape) != 4
                or any(type(value) is not int or value < 1 for value in shape)
                or shape[0] != len(LAYERS) or shape[2] != len(qpos)
                or shape[3] != input_length or math.prod(shape) * 4 > MAX_CAPTURE_BYTES
                or type(row.get("attention_bytes_length")) is not int
                or row["attention_bytes_length"] != math.prod(shape) * 4
                or row["attention_bytes_length"] > MAX_CAPTURE_BYTES):
            _fail("capture_sidecar_shape_or_length_invalid")
        digest = row.get("attention_bytes_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            _fail("capture_sidecar_digest_invalid")
        raw = sidecars[digest]
        if (not isinstance(raw, bytes) or len(raw) != row["attention_bytes_length"]
                or _sha(raw) != digest):
            _fail("capture_sidecar_digest_or_length_mismatch")
        total += len(raw)
        if total > MAX_TOTAL_SIDECAR_BYTES:
            _fail("capture_sidecar_bundle_too_large")
        array = _attention_array(raw, shape)
        import numpy as np
        if (not np.isfinite(array).all() or (array < 0).any()):
            _fail("capture_sidecar_attention_value_invalid")
        if (np.abs(array.sum(axis=-1, dtype=np.float64) - 1.0)
                > _ROW_SUM_TOLERANCE[source_dtype]).any():
            _fail("capture_sidecar_attention_row_not_normalized")
        masses = _mass_hex(array, shape, groups)
        if row.get("layer_group_masses_f32_be_hex") != masses:
            _fail("capture_sidecar_group_mass_recomputation_mismatch")
        from qta_selection_provenance import validate_qta_selection_provenance
        validate_qta_selection_provenance(
            expected["selection_provenance"],
            expected_bindings=expected["expected_selection_bindings"],
            plan=expected["plan"])
        selection = expected["selection_provenance"]
        if (selection.get("selector_layers_1_based") != list(LAYERS)
                or selection.get("layer_group_mass_f32_be_hex") != masses):
            _fail("capture_sidecar_selection_mass_mismatch")
    return {
        "status": "capture_sidecars_joined_to_selection_masses_non_authorizing",
        "record_count": len(rows), "sidecar_count": len(sidecars),
        "total_sidecar_bytes": total,
        "forward_source_authenticated": False,
        "attention_capture_authenticated": False,
        "scoring_authorized": False, "execution_authorization": "not_granted",
    }
