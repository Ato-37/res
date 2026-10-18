"""Strict, non-authorizing receipts for QTA attention hook diagnostics.

Receipts bind a caller-supplied capture report to a record, runtime digest,
query geometry, and attention payload digest. They validate self-reported
instrumentation only; they do not authenticate a model or forward execution.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = 1
PURPOSE = "qta_attention_hook_call_receipt_non_authorizing"
INDEX_PURPOSE = "qta_attention_hook_call_receipt_index_non_authorizing"
MAX_INDEX_BYTES = 24_000_000
MAX_RECORDS = 128
LAYERS_1_BASED = tuple(range(5, 17))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_KEYS = {
    "schema_version", "purpose", "record_token", "runtime_sha256",
    "input_length", "question_positions", "attention_bytes_sha256",
    "capture_report",
    "forward_calls", "layer_indices_1_based", "layer_calls",
    "layer_module_paths", "layer_module_classes", "attention_shape",
    "attention_source_dtypes", "attention_source_dtype",
    "forward_source_authenticated", "attention_capture_authenticated",
    "model_load_permitted", "scoring_authorized", "execution_authorization",
}
_INDEX_KEYS = {
    "schema_version", "purpose", "record_count", "records",
    "forward_source_authenticated", "attention_capture_authenticated",
    "model_load_permitted", "scoring_authorized", "execution_authorization",
}


def _fail(code: str) -> None:
    raise ValueError(code)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _strict_json(raw: bytes) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("call_receipt_duplicate_json_key")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("ascii"), object_pairs_hook=unique_object,
                          parse_constant=lambda _value: _fail(
                              "call_receipt_nonfinite_json_value"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("call_receipt_index_json_invalid")


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _fail(f"call_receipt_{name}_invalid")
    return value


def build_qta_attention_call_receipt(*, report: Mapping[str, Any],
                                     record_token: str,
                                     runtime_sha256: str,
                                     input_length: int,
                                     question_positions: Sequence[int],
                                     attention_bytes_sha256: str,
                                     expected_module_classes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Bind an exact one-forward hook report to row and payload identities.

    The resulting receipt remains self-attested and explicitly non-authorizing.
    """
    if not isinstance(report, Mapping):
        _fail("call_receipt_report_invalid")
    _digest(record_token, "record_token")
    _digest(runtime_sha256, "runtime_sha256")
    _digest(attention_bytes_sha256, "attention_bytes_sha256")
    if type(input_length) is not int or input_length < 1:
        _fail("call_receipt_input_length_invalid")
    if (not isinstance(question_positions, Sequence)
            or isinstance(question_positions, (str, bytes)) or not question_positions
            or any(type(x) is not int or x < 0 or x >= input_length
                   for x in question_positions)
            or list(question_positions) != list(range(question_positions[0],
                                                        question_positions[-1] + 1))):
        _fail("call_receipt_question_positions_invalid")

    expected_keys = {
        "status", "forward_calls", "layer_calls", "layer_indices_1_based",
        "layer_module_classes", "layer_module_paths", "question_query_count",
        "input_length", "attention_row_sum_max_abs_deviation",
        "attention_source_dtype", "attention_row_sum_tolerance",
        "attention_rows_satisfy_source_dtype_normalization_tolerance",
        "attention_shape", "attention_source_dtypes",
        "forward_source_authenticated", "attention_capture_authenticated",
        "scoring_authorized",
    }
    if set(report) != expected_keys:
        _fail("call_receipt_report_schema_invalid")
    if (report["status"] != "one_forward_attention_rows_captured_non_authorizing"
            or type(report["forward_calls"]) is not int
            or report["forward_calls"] != 1
            or report["layer_indices_1_based"] != list(LAYERS_1_BASED)
            or type(report["input_length"]) is not int
            or report["input_length"] != input_length
            or type(report["question_query_count"]) is not int
            or report["question_query_count"] != len(question_positions)
            or report["forward_source_authenticated"] is not False
            or report["attention_capture_authenticated"] is not False
            or report["scoring_authorized"] is not False):
        _fail("call_receipt_report_binding_invalid")
    keys = [str(layer) for layer in LAYERS_1_BASED]
    for field in ("layer_calls", "layer_module_paths", "layer_module_classes",
                  "attention_source_dtypes"):
        value = report[field]
        if not isinstance(value, Mapping) or set(value.keys()) != set(keys):
            _fail(f"call_receipt_{field}_invalid")
    if any(report["layer_calls"][key] != 1
           or type(report["layer_calls"][key]) is not int for key in keys):
        _fail("call_receipt_layer_call_count_invalid")
    expected_paths = {
        str(layer): f"model.model.language_model.layers[{layer - 1}].self_attn"
        for layer in LAYERS_1_BASED
    }
    if dict(report["layer_module_paths"]) != expected_paths:
        _fail("call_receipt_layer_module_paths_invalid")
    classes = report["layer_module_classes"]
    if any(not isinstance(classes[key], str) or not classes[key] for key in keys):
        _fail("call_receipt_layer_module_classes_invalid")
    if expected_module_classes is not None:
        if (not isinstance(expected_module_classes, Mapping)
                or dict(expected_module_classes) != dict(classes)):
            _fail("call_receipt_expected_module_classes_mismatch")
    dtype = report["attention_source_dtype"]
    if not isinstance(dtype, str) or dtype not in {"float32", "bfloat16"} or any(
            report["attention_source_dtypes"][key] != dtype for key in keys):
        _fail("call_receipt_attention_dtype_invalid")
    tolerance = {"float32": 2e-4, "bfloat16": 0.0041}[dtype]
    deviation = report["attention_row_sum_max_abs_deviation"]
    reported_tolerance = report["attention_row_sum_tolerance"]
    if (isinstance(deviation, bool) or not isinstance(deviation, (int, float))
            or not math.isfinite(float(deviation)) or deviation < 0
            or isinstance(reported_tolerance, bool)
            or not isinstance(reported_tolerance, (int, float))
            or float(reported_tolerance) != tolerance
            or deviation > tolerance
            or report["attention_rows_satisfy_source_dtype_normalization_tolerance"] is not True):
        _fail("call_receipt_attention_row_sum_report_invalid")
    shape = report["attention_shape"]
    if (not isinstance(shape, list) or len(shape) != 4
            or any(type(value) is not int for value in shape)
            or shape[0] != len(LAYERS_1_BASED)
            or shape[1] < 1
            or shape[2] != len(question_positions) or shape[3] != input_length):
        _fail("call_receipt_attention_shape_invalid")

    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "record_token": record_token,
        "runtime_sha256": runtime_sha256,
        "input_length": input_length,
        "question_positions": list(question_positions),
        "attention_bytes_sha256": attention_bytes_sha256,
        "capture_report": dict(report),
        "forward_calls": 1,
        "layer_indices_1_based": list(LAYERS_1_BASED),
        "layer_calls": dict(report["layer_calls"]),
        "layer_module_paths": dict(report["layer_module_paths"]),
        "layer_module_classes": dict(report["layer_module_classes"]),
        "attention_shape": list(shape),
        "attention_source_dtypes": dict(report["attention_source_dtypes"]),
        "attention_source_dtype": dtype,
        "forward_source_authenticated": False,
        "attention_capture_authenticated": False,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
    }


def validate_qta_attention_call_receipt(receipt: Any, *,
                                        expected_record_token: str,
                                        expected_runtime_sha256: str,
                                        expected_input_length: int,
                                        expected_question_positions: Sequence[int],
                                        expected_attention_bytes_sha256: str,
                                        expected_module_classes: Mapping[str, str] | None = None
                                        ) -> dict[str, Any]:
    """Validate exact receipt structure/bindings; make no authenticity claim."""
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_KEYS:
        _fail("call_receipt_schema_invalid")
    if (receipt["schema_version"] != SCHEMA_VERSION
            or receipt["purpose"] != PURPOSE
            or receipt["forward_calls"] != 1
            or receipt["forward_source_authenticated"] is not False
            or receipt["attention_capture_authenticated"] is not False
            or receipt["model_load_permitted"] is not False
            or receipt["scoring_authorized"] is not False
            or receipt["execution_authorization"] != "not_granted"):
        _fail("call_receipt_authorization_fields_invalid")
    exact = {
        "record_token": expected_record_token,
        "runtime_sha256": expected_runtime_sha256,
        "input_length": expected_input_length,
        "question_positions": list(expected_question_positions),
        "attention_bytes_sha256": expected_attention_bytes_sha256,
        "layer_indices_1_based": list(LAYERS_1_BASED),
    }
    if any(receipt[key] != value for key, value in exact.items()):
        _fail("call_receipt_expected_binding_mismatch")
    rebuilt = build_qta_attention_call_receipt(
        report=receipt["capture_report"], record_token=expected_record_token,
        runtime_sha256=expected_runtime_sha256,
        input_length=expected_input_length,
        question_positions=expected_question_positions,
        attention_bytes_sha256=expected_attention_bytes_sha256,
        expected_module_classes=expected_module_classes)
    if _canonical(rebuilt) != _canonical(receipt):
        _fail("call_receipt_noncanonical_or_invalid")
    return {"call_receipt_valid": True,
            "call_receipt_joined_to_record_runtime_and_attention_bytes": True,
            "forward_source_authenticated": False,
            "attention_capture_authenticated": False,
            "scoring_authorized": False,
            "execution_authorization": "not_granted"}


def _receipt_expected_from_self(receipt: Mapping[str, Any]) -> dict[str, Any]:
    report = receipt.get("capture_report")
    classes = report.get("layer_module_classes") if isinstance(report, Mapping) else None
    return {
        "expected_record_token": receipt.get("record_token"),
        "expected_runtime_sha256": receipt.get("runtime_sha256"),
        "expected_input_length": receipt.get("input_length"),
        "expected_question_positions": receipt.get("question_positions"),
        "expected_attention_bytes_sha256": receipt.get("attention_bytes_sha256"),
        "expected_module_classes": classes,
    }


def build_qta_attention_call_receipt_index(*,
                                           receipts: Sequence[Mapping[str, Any]]) -> bytes:
    """Build canonical index bytes for an ordered set of self-attested receipts."""
    if (not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes))
            or not 1 <= len(receipts) <= MAX_RECORDS):
        _fail("call_receipt_index_record_count_invalid")
    records = []
    seen = set()
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            _fail("call_receipt_index_record_invalid")
        validate_qta_attention_call_receipt(receipt, **_receipt_expected_from_self(receipt))
        token = receipt["record_token"]
        if token in seen:
            _fail("call_receipt_index_duplicate_record_token")
        seen.add(token)
        records.append({"receipt": dict(receipt)})
    index = {
        "schema_version": SCHEMA_VERSION,
        "purpose": INDEX_PURPOSE,
        "record_count": len(records),
        "records": records,
        "forward_source_authenticated": False,
        "attention_capture_authenticated": False,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
    }
    raw = _canonical(index)
    if len(raw) > MAX_INDEX_BYTES:
        _fail("call_receipt_index_too_large")
    return raw


def validate_qta_attention_call_receipt_index(*, index_raw: bytes,
                                               expected_rows: Sequence[Mapping[str, Any]]
                                               ) -> dict[str, Any]:
    """Validate canonical ordered receipt bytes against frozen row bindings."""
    if not isinstance(index_raw, bytes) or not 1 <= len(index_raw) <= MAX_INDEX_BYTES:
        _fail("call_receipt_index_bytes_invalid")
    index = _strict_json(index_raw)
    if not isinstance(index, Mapping) or set(index) != _INDEX_KEYS:
        _fail("call_receipt_index_schema_invalid")
    if (_canonical(index) != index_raw
            or index["schema_version"] != SCHEMA_VERSION
            or index["purpose"] != INDEX_PURPOSE
            or index["forward_source_authenticated"] is not False
            or index["attention_capture_authenticated"] is not False
            or index["model_load_permitted"] is not False
            or index["scoring_authorized"] is not False
            or index["execution_authorization"] != "not_granted"):
        _fail("call_receipt_index_authorization_or_encoding_invalid")
    if (not isinstance(expected_rows, Sequence)
            or isinstance(expected_rows, (str, bytes))
            or not 1 <= len(expected_rows) <= MAX_RECORDS
            or type(index["record_count"]) is not int
            or index["record_count"] != len(expected_rows)
            or not isinstance(index["records"], list)
            or len(index["records"]) != len(expected_rows)):
        _fail("call_receipt_index_record_count_invalid")
    seen = set()
    for receipt, expected in zip(index["records"], expected_rows, strict=True):
        if not isinstance(expected, Mapping) or set(expected) != {
                "record_token", "runtime_sha256", "input_length",
                "question_positions", "attention_bytes_sha256"}:
            _fail("call_receipt_index_expected_row_schema_invalid")
        token = expected["record_token"]
        if token in seen:
            _fail("call_receipt_index_duplicate_expected_record_token")
        seen.add(token)
        if not isinstance(receipt, Mapping) or set(receipt) != {"receipt"}:
            _fail("call_receipt_index_record_schema_invalid")
        validate_qta_attention_call_receipt(
            receipt["receipt"],
            expected_record_token=expected["record_token"],
            expected_runtime_sha256=expected["runtime_sha256"],
            expected_input_length=expected["input_length"],
            expected_question_positions=expected["question_positions"],
            expected_attention_bytes_sha256=expected["attention_bytes_sha256"])
    return {
        "call_receipt_records_validated": len(expected_rows),
        "call_receipt_index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "forward_source_authenticated": False,
        "attention_capture_authenticated": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
    }
