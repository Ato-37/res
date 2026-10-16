"""Fail-closed CPU input freeze for the fixed NExT-GQA 3 x 3 scorer.

This is an integration primitive, not a production authorization gate. The
caller must first authenticate review/custody/clock/decoder/target artifacts
with their dedicated validators. This module then materializes and hashes all
nine exact processor batches before returning anything to a model loader. It
does not import torch or load model weights.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

CONDITIONS = ("full", "claim_evidence_black", "temporal_control_black")
ROW_COUNT = 3


class PreflightError(ValueError):
    """Stable failure code that does not include paths or protected content."""


class FrozenBatch(dict):
    """A snapshot mapping compatible with the scorer's `.to(device)` call."""

    def to(self, device: str):
        return FrozenBatch({key: _move_tree(value, device)
                            for key, value in self.items()})


def _fail(code: str) -> None:
    raise PreflightError(code)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        _fail("batch_value_not_canonicalizable")


def _tensor_payload(value: Any) -> dict[str, Any] | None:
    """Encode tensor metadata and values without depending on NumPy/Torch APIs."""
    if not (callable(getattr(value, "detach", None))
            and callable(getattr(value, "cpu", None))
            and callable(getattr(value, "tolist", None))):
        return None
    try:
        cpu = value.detach().cpu()
        if str(getattr(cpu, "device", "cpu")) != "cpu":
            _fail("cpu_batch_contains_non_cpu_tensor")
        values = cpu.tolist()
        dtype = str(cpu.dtype)
        shape = list(cpu.shape)
    except PreflightError:
        raise
    except Exception:
        _fail("batch_tensor_materialization_failed")
    _assert_finite_tree(values)
    return {"__tensor__": True, "dtype": dtype, "shape": shape, "values": values}


def _snapshot_tree(value: Any) -> Any:
    """Copy tensors to private CPU storage and recursively detach containers."""
    if (callable(getattr(value, "detach", None))
            and callable(getattr(value, "cpu", None))
            and callable(getattr(value, "clone", None))):
        try:
            snapshot = value.detach().cpu().clone()
        except Exception:
            _fail("batch_tensor_snapshot_failed")
        if str(getattr(snapshot, "device", "cpu")) != "cpu":
            _fail("cpu_batch_contains_non_cpu_tensor")
        return snapshot
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            _fail("batch_mapping_key_invalid")
        return {key: _snapshot_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_tree(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        if isinstance(value, float) and not math.isfinite(value):
            _fail("batch_contains_nonfinite_value")
        return value
    _fail("batch_value_type_unsupported")


def _move_tree(value: Any, device: str) -> Any:
    if callable(getattr(value, "to", None)) and callable(getattr(value, "detach", None)):
        try:
            return value.to(device)
        except Exception:
            _fail("batch_device_transfer_failed")
    if isinstance(value, Mapping):
        return {key: _move_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    return value


def _assert_finite_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("batch_contains_nonfinite_value")
    if isinstance(value, list):
        for item in value:
            _assert_finite_tree(item)
    elif isinstance(value, tuple):
        for item in value:
            _assert_finite_tree(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            _assert_finite_tree(item)


def _encode(value: Any) -> Any:
    tensor = _tensor_payload(value)
    if tensor is not None:
        return tensor
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            _fail("batch_mapping_key_invalid")
        return {key: _encode(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        if isinstance(value, float) and not math.isfinite(value):
            _fail("batch_contains_nonfinite_value")
        return value
    # Processor outputs should be recursively representable as mappings,
    # sequences, scalars, or tensors; silently stringifying objects is unsafe.
    _fail("batch_value_type_unsupported")


def batch_sha256(batch: Any) -> str:
    """Hash a fully materialized batch with keys, tensor dtype/shape, and values."""
    if not isinstance(batch, Mapping) or not batch:
        _fail("condition_batch_schema_invalid")
    return hashlib.sha256(_canonical(_encode(batch))).hexdigest()


def freeze_all_batches(*, triplets: Any, token_ids: Any,
                       expected_record_ids: Any, expected_batch_hashes: Any,
                       authenticated_gate_check) -> dict[str, Any]:
    """Validate the complete finite cohort and return exact CPU batches.

    ``authenticated_gate_check`` is a required zero-argument integration hook.
    It must invoke the concrete repository validators and raise unless review,
    custody, clock/decoder correspondence, target, protocol, and code freeze
    have all passed for this exact cohort. A boolean or caller-supplied status
    is deliberately not accepted as a substitute for the hook.

    All nine `.to('cpu')` conversions and exact batch hashes finish before the
    gate hook runs. The caller may load weights only after this function
    returns successfully. The function itself has no model/load/forward API.
    """
    if not callable(authenticated_gate_check):
        _fail("authenticated_gate_check_required")
    if (not isinstance(token_ids, (list, tuple)) or len(token_ids) != 5
            or any(type(token) is not int or token < 0 for token in token_ids)
            or len(set(token_ids)) != 5):
        _fail("answer_label_token_contract_invalid")
    if not isinstance(triplets, (list, tuple)) or len(triplets) != ROW_COUNT:
        _fail("requested_record_count_mismatch")
    if (not isinstance(expected_record_ids, (list, tuple))
            or len(expected_record_ids) != ROW_COUNT
            or any(not isinstance(value, str) or not value for value in expected_record_ids)
            or len(set(expected_record_ids)) != ROW_COUNT):
        _fail("expected_record_order_invalid")
    if not isinstance(expected_batch_hashes, Mapping):
        _fail("expected_batch_hashes_invalid")

    ids = []
    materialized = []
    observed_hashes = {}
    for row_pos, row in enumerate(triplets, start=1):
        if not isinstance(row, Mapping) or set(row) != {"record_id", "gold_index", "batches"}:
            _fail(f"record_{row_pos}_schema_invalid")
        record_id = row["record_id"]
        if not isinstance(record_id, str) or not record_id or record_id in ids:
            _fail(f"record_{row_pos}_identity_invalid")
        if record_id != expected_record_ids[row_pos - 1]:
            _fail("record_order_or_identity_mismatch")
        ids.append(record_id)
        if type(row["gold_index"]) is not int or not 0 <= row["gold_index"] < 5:
            _fail(f"record_{row_pos}_gold_index_invalid")
        batches = row["batches"]
        if not isinstance(batches, Mapping) or set(batches) != set(CONDITIONS):
            _fail(f"record_{row_pos}_condition_set_invalid")
        cpu_batches = {}
        for condition in CONDITIONS:
            batch = batches[condition]
            if not isinstance(batch, Mapping) or not callable(getattr(batch, "to", None)):
                _fail(f"record_{row_pos}_{condition}_batch_invalid")
            try:
                cpu_batch = batch.to("cpu")
            except Exception:
                _fail(f"record_{row_pos}_{condition}_cpu_materialization_failed")
            if not isinstance(cpu_batch, Mapping):
                _fail(f"record_{row_pos}_{condition}_cpu_batch_invalid")
            cpu_batch = FrozenBatch(_snapshot_tree(cpu_batch))
            digest = batch_sha256(cpu_batch)
            cpu_batches[condition] = cpu_batch
            observed_hashes.setdefault(record_id, {})[condition] = digest
        # Match the exact input contract of nextgqa_claim_mask_scoring.py;
        # identities are carried separately in the frozen envelope.
        materialized.append({"gold_index": row["gold_index"],
                             "batches": cpu_batches})

    if set(expected_batch_hashes) != set(ids):
        _fail("expected_batch_record_set_mismatch")
    for record_id in ids:
        expected_conditions = expected_batch_hashes[record_id]
        if (not isinstance(expected_conditions, Mapping)
                or set(expected_conditions) != set(CONDITIONS)):
            _fail("expected_batch_condition_set_mismatch")
        for condition in CONDITIONS:
            expected = expected_conditions[condition]
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(c not in "0123456789abcdef" for c in expected)
                    or observed_hashes[record_id][condition] != expected):
                _fail("exact_condition_batch_hash_mismatch")

    # This must be wired to concrete, independently audited validators by the
    # production command. No model loader is reachable before it succeeds.
    try:
        gate_result = authenticated_gate_check()
    except Exception:
        _fail("authenticated_upstream_gate_failed")
    if gate_result is not True:
        _fail("authenticated_upstream_gate_not_passed")

    return {
        "status": "all_nine_batches_frozen",
        "scope": "finite_three_record_descriptive_only",
        "record_ids": ids,
        "conditions": list(CONDITIONS),
        "token_ids": list(token_ids),
        "requested_records": ROW_COUNT,
        "requested_forwards": ROW_COUNT * len(CONDITIONS),
        "batch_sha256": observed_hashes,
        "triplets": materialized,
        "execution_authorization": "not_granted",
    }
