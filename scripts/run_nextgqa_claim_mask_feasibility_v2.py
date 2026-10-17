#!/usr/bin/env python3
"""Closed, non-authorizing structural candidate for the NExT-GQA v2 run.

The repository does not yet define authenticated production schemas for the
review, custody, browser-clock, calibration, or evidence-audit inputs required
by ``docs/nextgqa_finite_claim_descriptive_protocol_v2.md``. This module
therefore accepts only a deliberately synthetic metadata envelope for tests.
It never reads dataset/media/question/claim content, imports Torch, loads a
model, forwards inputs, computes scores, or grants execution authorization.

The synthetic structural pass is not a production preflight and must not be
used to claim readiness for real inputs.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUESTED_RECORDS = 3
FRAME_COUNT = 32
CONDITIONS = ("full", "claim_evidence_black", "temporal_control_black")

# A hash-only envelope is enough to exercise closed shape/binding behavior.
# These names do not define or authenticate the eventual production artifact
# schemas; all accepted fixtures are explicitly synthetic.
BINDING_KEYS = frozenset({
    "claim_input_manifest", "claim_input_sidecar", "worksheet", "lineage",
    "source_csv", "video_map", "generation_packet", "review_payload",
    "review_index", "custody", "clock_payload", "clock_index",
    "calibration_plan", "target_manifest", "sampler_code",
    "condition_builder_code", "prompt_builder_code", "scorer_code",
    "model_config", "tokenizer", "processor_config",
})
TOP_KEYS = frozenset({
    "schema_version", "purpose", "fixture_kind", "status",
    "execution_authorization", "protocol_sha256", "cohort_sha256",
    "requested_records", "frame_count", "conditions", "binding_hashes",
    "records",
})
RECORD_KEYS = frozenset({
    "record_key_sha256", "source_media_sha256", "requested_frames",
    "declared_gate_status", "evidence_ranks", "control_ranks",
    "condition_input_hashes",
})
CONDITION_HASH_KEYS = frozenset({
    "input_ids_sha256", "pixel_values_sha256", "video_metadata_sha256",
})


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("nonfinite_json")


def _blocked(code: str, *, structural_status: str = "failed") -> dict[str, Any]:
    return {
        "status": "blocked",
        "scope": "synthetic_structural_candidate_only",
        "structural_status": structural_status,
        "failure_codes": [code],
        "requested_records": REQUESTED_RECORDS,
        "validated_records": 0,
        "frame_count": FRAME_COUNT,
        "conditions": list(CONDITIONS),
        "requested_model_forwards": REQUESTED_RECORDS * len(CONDITIONS),
        "model_loads": 0,
        "model_forwards": 0,
        "score_calls": 0,
        "D3": None,
        "production_ready": False,
        "execution_authorization": "not_granted",
    }


def validate_synthetic_candidate(
    candidate_raw: bytes,
    protocol_raw: bytes,
    expected_protocol_sha256: str,
) -> dict[str, Any]:
    """Validate synthetic hash/cardinality/condition geometry only.

    A structural success still has ``execution_authorization=not_granted``.
    Caller-supplied audit declarations are never accepted as authenticated
    review, custody, clock, target, or execution evidence.
    """
    if not isinstance(candidate_raw, bytes) or len(candidate_raw) > 1_000_000:
        return _blocked("candidate_bytes_invalid")
    if not isinstance(protocol_raw, bytes) or not protocol_raw or len(protocol_raw) > 1_000_000:
        return _blocked("protocol_bytes_invalid")
    if not _is_sha(expected_protocol_sha256):
        return _blocked("protocol_pin_invalid")
    protocol_digest = _sha256(protocol_raw)
    if protocol_digest != expected_protocol_sha256:
        return _blocked("protocol_hash_mismatch")

    try:
        candidate = json.loads(
            candidate_raw.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        return _blocked("candidate_json_invalid")

    if not isinstance(candidate, dict) or set(candidate) != TOP_KEYS:
        return _blocked("candidate_schema_invalid")
    if (type(candidate.get("schema_version")) is not int
            or candidate["schema_version"] != 1
            or candidate.get("purpose") != "synthetic_v2_atomic_preflight_fixture"
            or candidate.get("fixture_kind") != "synthetic_only"
            or candidate.get("status") != "candidate_unvalidated"
            or candidate.get("execution_authorization") != "not_granted"):
        return _blocked("candidate_not_explicitly_synthetic_and_non_authorizing")
    if candidate.get("protocol_sha256") != protocol_digest:
        return _blocked("candidate_protocol_binding_mismatch")
    if (not _is_sha(candidate.get("cohort_sha256"))
            or candidate.get("requested_records") != REQUESTED_RECORDS
            or type(candidate.get("requested_records")) is not int
            or candidate.get("frame_count") != FRAME_COUNT
            or type(candidate.get("frame_count")) is not int
            or candidate.get("conditions") != list(CONDITIONS)):
        return _blocked("cohort_frame_or_condition_contract_mismatch")

    bindings = candidate.get("binding_hashes")
    if (not isinstance(bindings, dict) or set(bindings) != BINDING_KEYS
            or any(not _is_sha(value) for value in bindings.values())):
        return _blocked("binding_hash_set_invalid")

    records = candidate.get("records")
    if not isinstance(records, list) or len(records) != REQUESTED_RECORDS:
        return _blocked("record_cardinality_invalid")

    seen_keys: set[str] = set()
    seen_media: set[str] = set()
    for index, row in enumerate(records):
        code_prefix = f"record_{index + 1}"
        if not isinstance(row, dict) or set(row) != RECORD_KEYS:
            return _blocked(f"{code_prefix}_schema_invalid")
        record_key = row.get("record_key_sha256")
        media_hash = row.get("source_media_sha256")
        if (not _is_sha(record_key) or not _is_sha(media_hash)
                or record_key in seen_keys or media_hash in seen_media):
            return _blocked(f"{code_prefix}_identity_binding_invalid")
        seen_keys.add(record_key)
        seen_media.add(media_hash)
        if (type(row.get("requested_frames")) is not int
                or row.get("requested_frames") != FRAME_COUNT):
            return _blocked(f"{code_prefix}_frame_count_invalid")
        if row.get("declared_gate_status") != "synthetic_pass_unverified":
            return _blocked(f"{code_prefix}_gate_declaration_invalid")

        evidence = row.get("evidence_ranks")
        controls = row.get("control_ranks")
        if (not isinstance(evidence, list) or not isinstance(controls, list)
                or not evidence or len(evidence) != len(controls)
                or any(type(rank) is not int or not 0 <= rank < FRAME_COUNT
                       for rank in evidence + controls)
                or len(set(evidence)) != len(evidence)
                or len(set(controls)) != len(controls)
                or set(evidence) & set(controls)):
            return _blocked(f"{code_prefix}_evidence_control_assignment_invalid")

        condition_hashes = row.get("condition_input_hashes")
        if (not isinstance(condition_hashes, dict)
                or set(condition_hashes) != set(CONDITIONS)):
            return _blocked(f"{code_prefix}_condition_set_invalid")
        shared_nonmask_hashes: dict[str, str] | None = None
        pixel_hashes: set[str] = set()
        for condition in CONDITIONS:
            hashes = condition_hashes.get(condition)
            if (not isinstance(hashes, dict) or set(hashes) != CONDITION_HASH_KEYS
                    or any(not _is_sha(value) for value in hashes.values())):
                return _blocked(f"{code_prefix}_{condition}_input_binding_invalid")
            nonmask_hashes = {
                "input_ids_sha256": hashes["input_ids_sha256"],
                "video_metadata_sha256": hashes["video_metadata_sha256"],
            }
            if shared_nonmask_hashes is None:
                shared_nonmask_hashes = nonmask_hashes
            elif nonmask_hashes != shared_nonmask_hashes:
                return _blocked(f"{code_prefix}_nonmask_condition_binding_mismatch")
            pixel_hashes.add(hashes["pixel_values_sha256"])
        if len(pixel_hashes) != len(CONDITIONS):
            return _blocked(f"{code_prefix}_condition_pixel_hashes_not_distinct")

    # This only establishes that a synthetic envelope satisfies its own
    # declared shape. No real evidence schema is defined/authenticated here.
    return {
        "status": "blocked",
        "scope": "synthetic_structural_candidate_only",
        "structural_status": "synthetic_candidate_passed_unvalidated",
        "failure_codes": ["production_authentication_and_authorization_unimplemented"],
        "protocol_sha256": protocol_digest,
        "cohort_sha256": candidate["cohort_sha256"],
        "requested_records": REQUESTED_RECORDS,
        "validated_records": REQUESTED_RECORDS,
        "frame_count": FRAME_COUNT,
        "conditions": list(CONDITIONS),
        "requested_model_forwards": REQUESTED_RECORDS * len(CONDITIONS),
        "model_loads": 0,
        "model_forwards": 0,
        "score_calls": 0,
        "D3": None,
        "production_ready": False,
        "execution_authorization": "not_granted",
    }


def run_candidate(candidate_raw: bytes, protocol_raw: bytes,
                  expected_protocol_sha256: str) -> dict[str, Any]:
    """Fail-closed entry point; no model or score callback exists by design."""
    report = validate_synthetic_candidate(
        candidate_raw, protocol_raw, expected_protocol_sha256)
    # Even the synthetic structural pass is unvalidated and has no execution
    # authority. Keep explicit zero counters and null D3 in every return path.
    report.update({
        "status": "blocked",
        "model_loads": 0,
        "model_forwards": 0,
        "score_calls": 0,
        "D3": None,
        "production_ready": False,
        "execution_authorization": "not_granted",
    })
    return report


if __name__ == "__main__":
    raise SystemExit(
        "This module is a synthetic non-authorizing preflight skeleton; "
        "no production CLI or score path is implemented."
    )
