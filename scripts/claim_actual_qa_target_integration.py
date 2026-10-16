"""Compose CSV-derived QA bindings with a complete frozen target contract.

CPU-only and unvalidated. The returned QA source contains private question and
choice text; callers must persist it only to an authorized private location and
must not print or publish it. This module performs no scoring or authorization.
"""
from __future__ import annotations

import hashlib
import json

from claim_answer_target_contract import BINDING_KEYS, validate_target_manifest
from nextgqa_actual_qa_source_adapter import build_actual_qa_source


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def bind_actual_csv_to_complete_target(*, csv_raw: bytes, selection: dict,
                                       target_manifest_raw: bytes,
                                       lineage_manifest_raw: bytes,
                                       expected_binding_hashes: dict) -> dict:
    """Bind actual CSV rows and require the complete target structure.

    `expected_binding_hashes` supplies the frozen non-source configuration
    hashes. Its `dataset_source_manifest_sha256` is checked against the digest
    computed from the lineage manifest's dataset input hashes.

    The result includes private QA bytes and linkable target hashes for a
    downstream local preflight; treat the entire result as private. It never
    grants execution authorization.
    """
    if (not isinstance(expected_binding_hashes, dict)
            or set(expected_binding_hashes) != BINDING_KEYS
            or any(not isinstance(value, str) or len(value) != 64
                   or any(c not in "0123456789abcdef" for c in value)
                   for value in expected_binding_hashes.values())):
        raise ValueError("expected_bindings_invalid")

    qa_source, binding = build_actual_qa_source(
        csv_raw=csv_raw, selection=selection,
        target_manifest_raw=target_manifest_raw,
        lineage_manifest_raw=lineage_manifest_raw)
    target = binding.get("target_manifest")
    if not isinstance(target, dict):
        raise ValueError("bound_target_manifest_invalid")
    try:
        source_manifest_hash = target["dataset_source_manifest_sha256"]
        if expected_binding_hashes["dataset_source_manifest_sha256"] != source_manifest_hash:
            raise ValueError("dataset_source_manifest_binding_mismatch")
        # Verify that the actual adapter has not dropped or changed any frozen
        # config binding while constructing the enriched manifest.
        for key in BINDING_KEYS:
            if target.get(key) != expected_binding_hashes[key]:
                raise ValueError("target_config_binding_mismatch")
        rows = target.get("records")
        if not isinstance(rows, list):
            raise ValueError("bound_target_rows_invalid")
        video_ids = [row.get("video_id") for row in rows if isinstance(row, dict)]
        claim_ids = [row.get("claim_id") for row in rows if isinstance(row, dict)]
        if len(video_ids) != 3 or len(claim_ids) != 3:
            raise ValueError("bound_target_rows_invalid")
        target_raw = _canonical(target)
        contract = validate_target_manifest(
            target_raw, video_ids, claim_ids, expected_binding_hashes)
        if contract.get("status") != "target_contract_structurally_valid_unvalidated":
            raise ValueError("target_contract_" + contract["failure_codes"][0])
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("bound_target_manifest_invalid") from exc

    qa_raw = _canonical(qa_source)
    return {
        "status": "actual_csv_target_structurally_bound_unvalidated",
        "scope": "private_cpu_source_binding_and_structural_contract_only",
        "failure_codes": [],
        "requested_records": 3,
        "bound_records": 3,
        "target_contract_status": contract["status"],
        "target_manifest_sha256": hashlib.sha256(target_raw).hexdigest(),
        "qa_source_sha256": hashlib.sha256(qa_raw).hexdigest(),
        "qa_source_raw": qa_raw,
        "target_manifest_raw": target_raw,
        "score_calls": 0,
        "execution_authorization": "not_granted",
    }
