"""Strict, non-authorizing QTA base processor-batch digest manifest v1.

This module binds three ordered opaque records to typed processor batches and
externally supplied metadata pins. It neither authenticates those pins nor
authorizes model loading or scoring.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from nextgqa_atomic_batch_freeze import batch_sha256

SCHEMA = "qta_processor_batch_digest_manifest_v1"
SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 1_000_000
ROW_COUNT = 3
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROW_FIELDS = {
    "record_token", "base_processor_batch_sha256", "source_video_sha256",
    "prompt_sha256", "input_ids_sha256", "processor_sha256",
    "tokenizer_sha256", "runtime_sha256", "sampler_metadata_sha256",
    "video_grid_thw",
}
_HASH_FIELDS = _ROW_FIELDS - {"record_token", "video_grid_thw"}
_TOP_FIELDS = {"schema_version", "purpose", "status", "execution_authorization",
               "protocol_sha256", "rows"}


class ManifestError(ValueError):
    """Stable validation failure without protected input values."""


def _fail(code: str) -> None:
    raise ManifestError(code)


def _sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        _fail(code)
    return value


def _validate_row(row: Any, position: int) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
        _fail(f"row_{position}_schema_invalid")
    token = _sha(row["record_token"], f"row_{position}_record_token_invalid")
    clean = {"record_token": token}
    for field in sorted(_HASH_FIELDS - {"base_processor_batch_sha256"}):
        clean[field] = _sha(row[field], f"row_{position}_{field}_invalid")
    clean["base_processor_batch_sha256"] = _sha(
        row["base_processor_batch_sha256"], f"row_{position}_batch_digest_invalid")
    grid = row["video_grid_thw"]
    if (not isinstance(grid, (list, tuple)) or len(grid) != 3
            or any(type(value) is not int or value <= 0 for value in grid)):
        _fail(f"row_{position}_video_grid_invalid")
    clean["video_grid_thw"] = list(grid)
    return clean


def _validate_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, (list, tuple)) or len(rows) != ROW_COUNT:
        _fail("exactly_three_rows_required")
    result = [_validate_row(row, i) for i, row in enumerate(rows, 1)]
    tokens = [row["record_token"] for row in result]
    if len(set(tokens)) != ROW_COUNT:
        _fail("record_tokens_not_unique")
    if len({row["source_video_sha256"] for row in result}) != ROW_COUNT:
        _fail("source_video_digests_not_unique")
    return result


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        _fail("manifest_not_canonicalizable")


def build_manifest(*, rows: Any, live_batches: Any,
                   expected_row_pins: Any, protocol_sha256: str,
                   expected_protocol_sha256: str) -> bytes:
    """Build canonical bytes after checking live typed batch digests and pins.

    ``rows`` supplies the ordered row metadata and opaque tokens. The expected
    pins must be an independently supplied, same-order set of complete rows,
    including ``base_processor_batch_sha256``. ``live_batches`` is the matching
    ordered sequence of typed mappings consumed by ``batch_sha256``.
    """
    protocol_sha256 = _sha(protocol_sha256, "protocol_sha256_invalid")
    expected_protocol_sha256 = _sha(
        expected_protocol_sha256, "expected_protocol_sha256_invalid")
    if protocol_sha256 != expected_protocol_sha256:
        _fail("protocol_pin_mismatch")
    prepared = _validate_rows(rows)
    expected = _validate_rows(expected_row_pins)
    if [r["record_token"] for r in expected] != [r["record_token"] for r in prepared]:
        _fail("expected_pin_order_mismatch")
    if not isinstance(live_batches, (list, tuple)) or len(live_batches) != ROW_COUNT:
        _fail("exactly_three_live_batches_required")
    observed = []
    for index, (row, batch) in enumerate(zip(prepared, live_batches, strict=True), 1):
        try:
            observed.append(batch_sha256(batch))
        except Exception as exc:
            raise ManifestError("live_typed_batch_invalid") from exc
        if not isinstance(batch, Mapping):
            _fail(f"row_{index}_live_batch_mapping_required")
        ids = _single_row_integer_list(batch.get("input_ids"),
                                      f"row_{index}_live_input_ids_invalid")
        ids_digest = hashlib.sha256(_canonical(ids)).hexdigest()
        if ids_digest != row["input_ids_sha256"]:
            _fail(f"row_{index}_live_input_ids_digest_mismatch")
        grid = _single_row_integer_list(batch.get("video_grid_thw"),
                                        f"row_{index}_live_video_grid_invalid")
        if len(grid) != 3 or any(value <= 0 for value in grid):
            _fail(f"row_{index}_live_video_grid_invalid")
        if grid != row["video_grid_thw"]:
            _fail(f"row_{index}_live_video_grid_mismatch")
    for index, (row, pins, digest) in enumerate(zip(prepared, expected, observed, strict=True), 1):
        if row != pins or digest != pins["base_processor_batch_sha256"]:
            _fail(f"row_{index}_external_pin_mismatch")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": SCHEMA,
        "status": "candidate_only",
        "execution_authorization": "not_granted",
        "protocol_sha256": protocol_sha256,
        "rows": prepared,
    }
    return _canonical(payload)


def _single_row_integer_list(value: Any, error_code: str) -> list[int]:
    """Normalize one processor row from a CPU tensor/list without coercion."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            if callable(getattr(value, "detach", None)):
                value = value.detach().cpu()
            value = value.tolist()
        except Exception:
            _fail(error_code)
    if isinstance(value, tuple):
        value = list(value)
    if (isinstance(value, list) and len(value) == 1
            and isinstance(value[0], (list, tuple))):
        value = list(value[0])
    if (not isinstance(value, list) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        _fail(error_code)
    return value


def _reject_constant(value: str) -> None:
    raise ManifestError("nonfinite_json_number_rejected")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key_rejected")
        result[key] = value
    return result


def validate_manifest(raw: bytes, record_bindings: Any, *,
                      expected_row_pins: Any,
                      expected_protocol_sha256: str) -> dict[str, Any]:
    """Validate exact canonical bytes and bindings; return aggregate-only data."""
    if not isinstance(raw, bytes):
        _fail("manifest_bytes_required")
    if not raw or len(raw) > MAX_MANIFEST_BYTES:
        _fail("manifest_size_invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
        obj = json.loads(text, object_pairs_hook=_unique_object,
                         parse_constant=_reject_constant)
    except ManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _fail("manifest_json_invalid")
    if not isinstance(obj, dict) or set(obj) != _TOP_FIELDS:
        _fail("manifest_schema_invalid")
    if (type(obj["schema_version"]) is not int
            or obj["schema_version"] != SCHEMA_VERSION or obj["purpose"] != SCHEMA
            or obj["status"] != "candidate_only"
            or obj["execution_authorization"] != "not_granted"):
        _fail("manifest_status_or_version_invalid")
    rows = _validate_rows(obj["rows"])
    if _canonical(obj) != raw:
        _fail("manifest_not_canonical_utf8_json")
    if (not isinstance(record_bindings, (list, tuple))
            or len(record_bindings) != ROW_COUNT):
        _fail("record_bindings_invalid")
    expected_rows = _validate_rows(expected_row_pins)
    if rows != expected_rows:
        _fail("external_row_pin_mismatch")
    expected_protocol_sha256 = _sha(
        expected_protocol_sha256, "expected_protocol_sha256_invalid")
    if obj["protocol_sha256"] != expected_protocol_sha256:
        _fail("protocol_pin_mismatch")
    for index, (row, binding) in enumerate(zip(rows, record_bindings, strict=True), 1):
        if (not isinstance(binding, Mapping)
                or set(binding) != {"record_token", "processor_batch_sha256"}):
            _fail(f"record_binding_{index}_schema_invalid")
        if (binding["record_token"] != row["record_token"]
                or binding["processor_batch_sha256"] != row["base_processor_batch_sha256"]):
            _fail("record_binding_order_or_digest_mismatch")
    return {"status": "candidate_only", "authorization": "not_granted",
            "row_count": ROW_COUNT, "bindings_match": True}
