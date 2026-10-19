"""Compose source/review bytes and signed phase-1 pins with the same live batches.

This is a CPU-only, non-authorizing transaction preflight. It reruns the
existing source/review and phase-receipt validators in one call, deriving the
expected live-batch pins from the verified phase-1 receipt rather than
accepting a caller success report. It does not validate the out-of-band trust
root, semantic truth, or clock/frame correspondence, and it cannot load or
call a model.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256
from qta_actual_prebaseline_diagnostic import diagnose_qta_prebaseline
from qta_signed_prebaseline_live_batches import verify_and_bind_signed_phase1_batches
from qta_stage1_source_pin_adapter import derive_stage1_source_pins

ROW_COUNT = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")


def _blocked(code: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "qta_prebaseline_transaction_preflight_only",
        "status": "blocked",
        "requested_rows": ROW_COUNT,
        "structurally_bound_rows": 0,
        "source_review_batch_binding": "blocked",
        "signed_phase1_batch_binding": "blocked",
        "same_live_batch_objects_rechecked": False,
        "external_trust_root_origin_authenticated": False,
        "human_identity_authenticated": False,
        "semantic_review_authenticated": False,
        "clock_frame_correspondence_authenticated": False,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "authorized_forwards": 0,
        "faithfulness_scored": False,
        "execution_authorization": "not_granted",
        "failure_code": code,
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _strict_json(raw: bytes, code: str) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(code)
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=lambda _value: (_ for _ in ()).throw(
                              ValueError(code)))
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise ValueError(code) from None


def _expected_rows_from_receipt(receipt_raw: bytes,
                                source_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    receipt = _strict_json(receipt_raw, "phase1_receipt_invalid")
    rows = receipt.get("record_bindings") if isinstance(receipt, dict) else None
    if not isinstance(rows, list) or len(rows) != ROW_COUNT or len(source_rows) != ROW_COUNT:
        raise ValueError("phase1_record_bindings_invalid")

    expected = []
    seen = set()
    for record, source in zip(rows, source_rows, strict=True):
        if (not isinstance(record, dict)
                or set(record) != {"record_token", "processor_batch_sha256"}
                or not isinstance(source, dict)):
            raise ValueError("phase1_record_binding_schema_invalid")
        token = source.get("review_token")
        source_sha = source.get("source_video_sha256")
        record_token = record.get("record_token")
        batch_sha = record.get("processor_batch_sha256")
        if (not isinstance(token, str) or not _TOKEN.fullmatch(token)
                or not isinstance(source_sha, str) or not _SHA256.fullmatch(source_sha)
                or not isinstance(record_token, str) or not _SHA256.fullmatch(record_token)
                or not isinstance(batch_sha, str) or not _SHA256.fullmatch(batch_sha)
                or record_token != hashlib.sha256(token.encode("ascii")).hexdigest()
                or record_token in seen):
            raise ValueError("phase1_record_source_order_mismatch")
        seen.add(record_token)
        expected.append({"review_token": token,
                         "source_video_sha256": source_sha,
                         "batch_sha256": batch_sha})
    return expected


def preflight_qta_phase1_batch_transaction(
        *, actual_source_inputs: Mapping[str, Any],
        review_bundle_inputs: Mapping[str, Any],
        stage1_audit_inputs: Mapping[str, Any],
        actual_qa_source_inputs: Mapping[str, Any],
        phase1_receipt_raw: bytes,
        phase1_artifact_bytes: Mapping[str, bytes],
        integrity_roster_raw: bytes, release_roster_raw: bytes,
        root_public_key_pem: bytes,
        expected_root_public_der_sha256: str,
        expected_external_cohort_pin_sha256: str,
        live_processor_rows: Any) -> dict[str, Any]:
    """Bind Stage-1 source pins, signed phase-1 receipt, and live batches.

    A shallow snapshot is made once; both independent leaf validators receive
    the same tuple of row mappings and the same ``FrozenBatch`` references.
    Tensor bytes are rehashed after both validators finish. A passing result
    means only that local structural bindings agree: the trust-root origin,
    semantic review, and clock/frame evidence remain unauthenticated.
    """
    try:
        source_snapshot = dict(actual_source_inputs)
        review_snapshot = dict(review_bundle_inputs)
        stage1_snapshot = dict(stage1_audit_inputs)
        qa_snapshot = dict(actual_qa_source_inputs)
        artifact_snapshot = dict(phase1_artifact_bytes)
        if source_snapshot != qa_snapshot:
            return _blocked("actual_qa_source_byte_maps_differ")
        review_roles = ("worksheet_raw", "lineage_raw", "review_raw",
                        "review_index_raw", "expected_lineage_sha256")
        if any(stage1_snapshot.get(role) != review_snapshot.get(role)
               for role in review_roles):
            return _blocked("stage1_review_byte_maps_differ")
        if not isinstance(live_processor_rows, (list, tuple)) or len(live_processor_rows) != ROW_COUNT:
            return _blocked("live_batch_inventory_invalid")
        # Snapshot row metadata but preserve the original in-memory batch refs.
        live_snapshot = []
        initial_refs = []
        initial_digests = []
        for row in live_processor_rows:
            if not isinstance(row, Mapping) or set(row) != {
                    "review_token", "source_video_sha256", "batch_sha256", "batch"}:
                return _blocked("live_batch_row_schema_invalid")
            # Read each value once so even a custom mutable Mapping cannot
            # provide one value during validation and another in the snapshot.
            token = row["review_token"]
            source_sha = row["source_video_sha256"]
            declared_digest = row["batch_sha256"]
            batch = row["batch"]
            if type(batch) is not FrozenBatch:
                return _blocked("live_frozen_batch_required")
            digest = batch_sha256(batch)
            if digest != declared_digest:
                return _blocked("live_batch_digest_invalid")
            live_snapshot.append({
                "review_token": token,
                "source_video_sha256": source_sha,
                "batch_sha256": declared_digest,
                "batch": batch,
            })
            initial_refs.append(id(batch))
            initial_digests.append(digest)
        live_snapshot = tuple(live_snapshot)

        # Rerun Stage-1 and QA from original bytes; do not accept summaries.
        source_pins = derive_stage1_source_pins(
            stage1_audit_inputs=stage1_snapshot,
            actual_qa_source_inputs=qa_snapshot)
        source_rows = source_pins.get("source_rows") if isinstance(source_pins, dict) else None
        tokens = source_pins.get("expected_cohort_tokens") if isinstance(source_pins, dict) else None
        if (not isinstance(source_pins, dict)
                or source_pins.get("status") != "three_row_source_pins_structurally_derived"
                or source_pins.get("effective_records") != ROW_COUNT
                or not isinstance(source_rows, list) or len(source_rows) != ROW_COUNT
                or not isinstance(tokens, list) or len(tokens) != ROW_COUNT
                or [row.get("review_token") for row in source_rows] != tokens
                or source_pins.get("reviewer_identity_verified") is not False
                or source_pins.get("semantic_review_authenticated") is not False):
            return _blocked("stage1_source_pin_derivation_incomplete")

        stage1_raw = artifact_snapshot.get("stage1_review_and_lineage")
        if not isinstance(stage1_raw, bytes) or stage1_raw != _canonical(source_pins):
            return _blocked("phase1_stage1_bytes_do_not_match_derived_pins")

        # Verify the signed receipt against the derived source cohort and the
        # exact same batch references that the source/review diagnostic sees.
        signed = verify_and_bind_signed_phase1_batches(
            receipt_raw=phase1_receipt_raw,
            artifact_bytes=artifact_snapshot,
            integrity_roster_raw=integrity_roster_raw,
            release_roster_raw=release_roster_raw,
            root_public_key_pem=root_public_key_pem,
            expected_root_public_der_sha256=expected_root_public_der_sha256,
            expected_external_cohort_pin_sha256=expected_external_cohort_pin_sha256,
            live_rows=live_snapshot,
        )
        if (signed.get("status") != "passed"
                or signed.get("validated_rows") != ROW_COUNT
                or signed.get("phase_receipt_signature_valid") is not True
                or signed.get("artifact_bytes_match") is not True
                or signed.get("external_cohort_pin_origin_authenticated") is not False
                or signed.get("model_load_permitted") is not False
                or signed.get("scoring_authorized") is not False):
            return _blocked("signed_phase1_batch_binding_incomplete")

        expected_rows = _expected_rows_from_receipt(phase1_receipt_raw, source_rows)
        diagnostic = diagnose_qta_prebaseline(
            actual_source_inputs=source_snapshot,
            review_bundle_inputs=review_snapshot,
            expected_processor_rows=expected_rows,
            live_processor_rows=live_snapshot,
        )
        gates = diagnostic.get("gates") if isinstance(diagnostic, dict) else None
        needed = ("actual_qa_source_binding", "review_bundle_byte_structure",
                  "source_to_review_ordered_join", "processor_batch_binding")
        if (diagnostic.get("status") != "diagnostic_incomplete"
                or not isinstance(gates, dict)
                or any(not isinstance(gates.get(name), dict)
                       or gates[name].get("status") != "passed"
                       or gates[name].get("validated_rows") != ROW_COUNT
                       for name in needed)):
            return _blocked("source_review_live_batch_binding_incomplete")

        # This catches net mutation during validation. A future scorer must
        # still rehash the batch immediately before every model forward.
        if ([id(row["batch"]) for row in live_snapshot] != initial_refs
                or [batch_sha256(row["batch"]) for row in live_snapshot] != initial_digests):
            return _blocked("live_batch_changed_during_preflight")

        return {
            "schema_version": 1,
            "purpose": "qta_prebaseline_transaction_preflight_only",
            "status": "structural_bindings_consistent_unvalidated",
            "requested_rows": ROW_COUNT,
            "structurally_bound_rows": ROW_COUNT,
            "source_review_batch_binding": "passed_unvalidated",
            "signed_phase1_batch_binding": "passed_unvalidated",
            "same_live_batch_objects_rechecked": True,
            "external_trust_root_origin_authenticated": False,
            "human_identity_authenticated": False,
            "semantic_review_authenticated": False,
            "clock_frame_correspondence_authenticated": False,
            "model_load_permitted": False,
            "scoring_authorized": False,
            "authorized_forwards": 0,
            "faithfulness_scored": False,
            "execution_authorization": "not_granted",
            "failure_code": None,
        }
    except Exception:
        # Do not reflect exception text from protected or untrusted inputs.
        return _blocked("transaction_preflight_blocked")
