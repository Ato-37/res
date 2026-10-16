"""Device-independent digest manifest for a fixed 3 x 3 Qwen batch bundle.

This is an integrity primitive only. It does not establish source provenance,
review or clock validity, protocol freeze, authorization, or model readiness.
Tensor bytes are copied to CPU, made contiguous, and hashed together with dtype
and shape. Nested mappings/lists/tuples are supported; arbitrary objects fail
closed rather than being stringified or pickled.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


CONDITIONS = ("full", "claim_evidence_black", "temporal_control_black")
RECORD_COUNT = 3
MANIFEST_SCHEMA = "nextgqa_exact_batch_digest_manifest_v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _tensor_digest(tensor: Any, path: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - expected research dependency
        raise ValueError("torch_unavailable") from exc
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("unsupported_batch_value:" + path)
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError("unsupported_tensor_layout:" + path)
    detached = tensor.detach()
    if detached.is_floating_point() or detached.is_complex():
        try:
            if not bool(torch.isfinite(detached).all()):
                raise ValueError("nonfinite_tensor:" + path)
        except RuntimeError:
            raise ValueError("tensor_finiteness_check_failed:" + path) from None
    try:
        cpu = detached.to(device="cpu").contiguous()
        # Viewing bytes preserves exact bfloat16/float8/integer representations
        # without relying on NumPy support for the source dtype.
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    except Exception:
        raise ValueError("tensor_canonicalization_failed:" + path) from None
    dtype = str(cpu.dtype)
    shape = list(cpu.shape)
    descriptor = {"dtype": dtype, "shape": shape,
                  "byte_count": len(raw), "content_sha256": hashlib.sha256(raw).hexdigest()}
    descriptor["tensor_sha256"] = hashlib.sha256(
        _canonical({"dtype": dtype, "shape": shape, "byte_count": len(raw)})
        + b"\0" + raw).hexdigest()
    return descriptor


def _describe(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError("batch_mapping_key_invalid:" + path)
        return {"type": "mapping", "items": {
            key: _describe(value[key], path + "." + key) for key in sorted(value)}}
    if isinstance(value, (list, tuple)):
        return {"container": "list" if isinstance(value, list) else "tuple",
                "items": [_describe(item, f"{path}[{index}]")
                          for index, item in enumerate(value)]}
    # Qwen's processor returns video_metadata with primitive leaves. Keep
    # explicit tags so bool, int, float and str remain distinguishable.
    if value is None:
        return {"type": "none"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite_primitive:" + path)
        return {"type": "float64_hex", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    # Arbitrary objects are never stringified or pickled.
    return _tensor_digest(value, path)


def _validate_bundle(triplets: Any) -> None:
    if not isinstance(triplets, (list, tuple)) or len(triplets) != RECORD_COUNT:
        raise ValueError("requested_record_count_mismatch")
    common_keys: set[str] | None = None
    for row_index, row in enumerate(triplets, 1):
        if not isinstance(row, Mapping) or set(row) != {"gold_index", "batches"}:
            raise ValueError(f"record_{row_index}_schema_invalid")
        gold = row["gold_index"]
        if type(gold) is not int or not 0 <= gold < 5:
            raise ValueError(f"record_{row_index}_gold_index_invalid")
        batches = row["batches"]
        if not isinstance(batches, Mapping) or set(batches) != set(CONDITIONS):
            raise ValueError(f"record_{row_index}_condition_set_invalid")
        for condition in CONDITIONS:
            batch = batches[condition]
            if not isinstance(batch, Mapping) or not batch:
                raise ValueError(f"record_{row_index}_{condition}_batch_invalid")
            keys = set(batch)
            if common_keys is None:
                common_keys = keys
            elif keys != common_keys:
                raise ValueError(f"record_{row_index}_{condition}_batch_keys_mismatch")


def build_batch_digest_manifest(triplets: Any) -> dict[str, Any]:
    """Digest all nine exact batches; output is explicitly unvalidated."""
    _validate_bundle(triplets)
    rows = []
    for row_index, row in enumerate(triplets, 1):
        conditions = {}
        for condition in CONDITIONS:
            conditions[condition] = _describe(
                row["batches"][condition], f"record_{row_index}.{condition}")
        rows.append({"record_position": row_index,
                     "gold_index": row["gold_index"], "conditions": conditions})
    unsigned = {
        "schema": MANIFEST_SCHEMA,
        "status": "unvalidated_digest_manifest_only",
        "record_count": RECORD_COUNT,
        "condition_order": list(CONDITIONS),
        "row_order": "caller_supplied_order",
        "tensor_encoding": "cpu_contiguous_dtype_shape_bytes_sha256_v1",
        "records": rows,
        "source_authentication": "not_performed",
        "execution_authorization": "not_granted",
    }
    return {**unsigned, "manifest_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest()}


def verify_batch_digest_manifest(triplets: Any, manifest: Any) -> bool:
    """Recompute and compare exact tensor digests; never authenticates origin."""
    if not isinstance(manifest, Mapping):
        raise ValueError("digest_manifest_invalid")
    expected = build_batch_digest_manifest(triplets)
    if dict(manifest) != expected:
        raise ValueError("batch_digest_manifest_mismatch")
    return True
