"""Create and verify signed byte-binding candidates for the two QTA phases.

These receipts bind exact artifacts to the freeze-owner key in the existing
root-signed release roster. They do not authenticate people or judgments,
rerun upstream semantic audits, or authorize model loading/scoring. A future
production runner must independently rerun every required validator.
"""
from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from typing import Any

from audit_nextgqa_review_release_order import _validate_release_roster
from qta_nine_branch_plan import BRANCH_ORDER
from nextgqa_pre_review_integrity_receipt import (
    _MAX_JSON, _canonical, _decode_signature, _parse, _sha, _sign,
    _verify_signature, ReceiptError,
)

ROW_COUNT = 3
MAX_TOTAL_ARTIFACT_BYTES = 96_000_000
MAX_SINGLE_ARTIFACT_BYTES = 24_000_000
PHASE_ARTIFACT_ROLES = {
    "pre_baseline": (
        "stage1_review_and_lineage", "clock_frame_evidence",
        "source_backed_targets", "prompt_answer_token_bindings",
        "sampler_model_runtime", "condition_scorer_source_closure",
        "full_video_processor_batches", "atomic_cohort_preflight",
        "external_cohort_pin",
    ),
    "post_baseline": (
        "phase1_receipt", "baseline_pq_manifest",
        "stage2_prediction_link_review", "nine_branch_plan_manifest",
        "condition_scorer_source_closure", "attention_capture_index",
        "attention_call_receipts",
    ),
}
PHASE_BRANCHES = {
    "pre_baseline": BRANCH_ORDER[:1],
    "post_baseline": BRANCH_ORDER[1:],
}
_FREEZE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TOKEN = re.compile(r"^[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")
_COHORT_SOURCE_ROW_KEYS = {
    "review_token", "question_sha256", "options_sha256", "claim_sha256",
    "source_video_sha256", "video_support_decision",
    "reviewed_intervals_sha256",
}
_EXTERNAL_COHORT_PIN_KEYS = {
    "schema_version", "purpose", "cohort_binding_sha256",
    "execution_authorization",
}
_PHASE_DOMAINS = {
    "pre_baseline": b"qta_two_phase_score_freeze_v1\0",
    "post_baseline": b"qta_two_phase_score_freeze_v3\0",
}


def _cohort_binding_sha256(stage1_review_and_lineage_raw: bytes) -> str:
    """Hash the ordered source/review identity rows, excluding derived scores."""
    if not isinstance(stage1_review_and_lineage_raw, bytes):
        raise ReceiptError("external_cohort_source_bytes_invalid")
    try:
        stage1 = _parse(stage1_review_and_lineage_raw,
                        "external_cohort_source_json_invalid")
    except ReceiptError:
        raise
    tokens = stage1.get("expected_cohort_tokens") if isinstance(stage1, dict) else None
    rows = stage1.get("source_rows") if isinstance(stage1, dict) else None
    if (not isinstance(tokens, list) or len(tokens) != ROW_COUNT
            or any(not isinstance(token, str) or not _REVIEW_TOKEN.fullmatch(token)
                   for token in tokens)
            or len(set(tokens)) != ROW_COUNT
            or not isinstance(rows, list) or len(rows) != ROW_COUNT):
        raise ReceiptError("external_cohort_source_rows_invalid")
    for token, row in zip(tokens, rows, strict=True):
        if (not isinstance(row, dict) or set(row) != _COHORT_SOURCE_ROW_KEYS
                or row.get("review_token") != token
                or row.get("video_support_decision") != "supports_claim"
                or any(not isinstance(row.get(key), str)
                       or not _SHA256.fullmatch(row[key])
                       for key in _COHORT_SOURCE_ROW_KEYS
                       if key not in {"review_token", "video_support_decision"})):
            raise ReceiptError("external_cohort_source_row_invalid")
    return _sha(_canonical({"expected_cohort_tokens": tokens,
                            "source_rows": rows}))


def build_external_cohort_pin_candidate(stage1_review_and_lineage_raw: bytes) -> bytes:
    """Build a non-authorizing pin candidate for independent freeze-owner adoption.

    The returned bytes must be independently checked and their SHA-256 pinned
    through a separate trusted channel before they are supplied to a verifier.
    """
    body = {
        "schema_version": 1,
        "purpose": "qta_external_cohort_pin_candidate_v1",
        "cohort_binding_sha256": _cohort_binding_sha256(
            stage1_review_and_lineage_raw),
        "execution_authorization": "not_granted",
    }
    return _canonical(body) + b"\n"


def _verify_external_cohort_pin(*, stage1_review_and_lineage_raw: bytes,
                                external_cohort_pin_raw: bytes,
                                expected_external_cohort_pin_sha256: str) -> str:
    if (not isinstance(expected_external_cohort_pin_sha256, str)
            or not _SHA256.fullmatch(expected_external_cohort_pin_sha256)):
        raise ReceiptError("external_cohort_pin_expected_hash_invalid")
    if (not isinstance(external_cohort_pin_raw, bytes)
            or _sha(external_cohort_pin_raw) != expected_external_cohort_pin_sha256):
        raise ReceiptError("external_cohort_pin_external_hash_mismatch")
    pin = _parse(external_cohort_pin_raw, "external_cohort_pin_json_invalid")
    if (not isinstance(pin, dict) or set(pin) != _EXTERNAL_COHORT_PIN_KEYS
            or type(pin.get("schema_version")) is not int
            or pin.get("schema_version") != 1
            or pin.get("purpose") != "qta_external_cohort_pin_candidate_v1"
            or pin.get("execution_authorization") != "not_granted"
            or not isinstance(pin.get("cohort_binding_sha256"), str)
            or not _SHA256.fullmatch(pin["cohort_binding_sha256"])
            or _canonical(pin) + b"\n" != external_cohort_pin_raw):
        raise ReceiptError("external_cohort_pin_schema_invalid")
    if pin["cohort_binding_sha256"] != _cohort_binding_sha256(
            stage1_review_and_lineage_raw):
        raise ReceiptError("external_cohort_pin_cohort_binding_mismatch")
    return expected_external_cohort_pin_sha256


def _bindings(phase: str, artifact_bytes: Mapping[str, bytes]) -> list[dict[str, Any]]:
    if phase not in PHASE_ARTIFACT_ROLES:
        raise ReceiptError("phase_invalid")
    if not isinstance(artifact_bytes, Mapping) or set(artifact_bytes) != set(PHASE_ARTIFACT_ROLES[phase]):
        raise ReceiptError("phase_artifact_role_set_invalid")
    total = 0
    rows = []
    for role in PHASE_ARTIFACT_ROLES[phase]:
        raw = artifact_bytes[role]
        if not isinstance(raw, bytes) or not raw:
            raise ReceiptError("phase_artifact_bytes_required")
        if len(raw) > MAX_SINGLE_ARTIFACT_BYTES:
            raise ReceiptError("phase_artifact_too_large")
        total += len(raw)
        rows.append({"role": role, "size_bytes": len(raw), "sha256": _sha(raw)})
    if total > MAX_TOTAL_ARTIFACT_BYTES:
        raise ReceiptError("phase_artifact_bundle_too_large")
    return rows


def _record_bindings(rows: Any) -> list[dict[str, str]]:
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        raise ReceiptError("phase_record_count_invalid")
    out = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"record_token", "processor_batch_sha256"}:
            raise ReceiptError("phase_record_binding_invalid")
        token, digest = row["record_token"], row["processor_batch_sha256"]
        if not isinstance(token, str) or not _TOKEN.fullmatch(token) or token in seen:
            raise ReceiptError("phase_record_token_invalid_or_duplicate")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ReceiptError("phase_processor_batch_digest_invalid")
        seen.add(token)
        out.append({"record_token": token, "processor_batch_sha256": digest})
    return out


def _verify_stage1_record_bindings(stage1_review_and_lineage_raw: bytes,
                                  records: list[dict[str, str]]) -> None:
    """Bind pre-baseline receipt rows to the pinned Stage-1 cohort order."""
    # Reuse the cohort validator so malformed or internally inconsistent rows
    # cannot be converted into record identities here.
    _cohort_binding_sha256(stage1_review_and_lineage_raw)
    stage1 = _parse(stage1_review_and_lineage_raw,
                    "external_cohort_source_json_invalid")
    expected = [_sha(token.encode("ascii"))
                for token in stage1["expected_cohort_tokens"]]
    actual = [row["record_token"] for row in records]
    if actual != expected:
        raise ReceiptError("phase_record_tokens_do_not_match_stage1_cohort")


def _roster_keys(*, integrity_roster_raw: bytes, release_roster_raw: bytes,
                 root_public_key_pem: bytes,
                 expected_root_public_der_sha256: str) -> tuple[dict, dict[str, bytes]]:
    release, role_keys = _validate_release_roster(
        release_roster_raw, root_public_key_pem,
        expected_root_public_der_sha256, integrity_roster_raw)
    key_ids = {entry["role"]: entry["key_id"] for entry in release["keys"]}
    return {"freeze_owner": key_ids["freeze_owner"]}, role_keys


def _body(*, phase: str, freeze_id: str, records: list[dict[str, str]],
          artifact_bindings: list[dict[str, Any]], signer_key_id: str,
          prior_phase1_receipt_sha256: str | None = None) -> dict[str, Any]:
    if phase not in PHASE_ARTIFACT_ROLES:
        raise ReceiptError("phase_invalid")
    version = 1 if phase == "pre_baseline" else 3
    if not isinstance(freeze_id, str) or not _FREEZE_ID.fullmatch(freeze_id):
        raise ReceiptError("freeze_id_invalid")
    result = {
        "schema_version": version,
        "purpose": f"qta_two_phase_score_freeze_candidate_v{version}",
        "phase": phase,
        "freeze_id": freeze_id,
        "record_bindings": records,
        "scheduled_branches_per_row": list(PHASE_BRANCHES[phase]),
        "planned_forward_count": ROW_COUNT * len(PHASE_BRANCHES[phase]),
        "artifact_bindings": artifact_bindings,
        "signer_role": "freeze_owner",
        "signer_key_id": signer_key_id,
        "status": "signed_non_authorizing_candidate",
        "execution_authorization": "not_granted",
        "model_load_permitted": False,
        "scoring_authorized": False,
        "phase_authorized": False,
        "authorized_forwards": 0,
    }
    if phase == "post_baseline":
        if not isinstance(prior_phase1_receipt_sha256, str) or not _SHA256.fullmatch(prior_phase1_receipt_sha256):
            raise ReceiptError("phase1_receipt_hash_invalid")
        result["phase1_receipt_sha256"] = prior_phase1_receipt_sha256
    elif prior_phase1_receipt_sha256 is not None:
        raise ReceiptError("phase1_hash_not_allowed")
    return result


def create_phase_freeze_receipt(*, phase: str, freeze_id: str,
                                record_rows: Any,
                                artifact_bytes: Mapping[str, bytes],
                                integrity_roster_raw: bytes,
                                release_roster_raw: bytes,
                                root_public_key_pem: bytes,
                                expected_root_public_der_sha256: str,
                                freeze_owner_private_key_pem: bytes,
                                prior_phase1_receipt_raw: bytes | None = None) -> bytes:
    """Sign an exact artifact-hash envelope with the rostered freeze-owner key."""
    _, role_keys = _roster_keys(
        integrity_roster_raw=integrity_roster_raw,
        release_roster_raw=release_roster_raw,
        root_public_key_pem=root_public_key_pem,
        expected_root_public_der_sha256=expected_root_public_der_sha256)
    roles = set(PHASE_ARTIFACT_ROLES.get(phase, ()))
    if phase == "post_baseline":
        if not isinstance(prior_phase1_receipt_raw, bytes):
            raise ReceiptError("phase1_receipt_required")
        if artifact_bytes.get("phase1_receipt") != prior_phase1_receipt_raw:
            raise ReceiptError("phase1_receipt_bytes_mismatch")
        prior_hash = _sha(prior_phase1_receipt_raw)
    else:
        if prior_phase1_receipt_raw is not None:
            raise ReceiptError("phase1_receipt_not_allowed")
        prior_hash = None
    if not roles:
        raise ReceiptError("phase_invalid")
    bindings = _bindings(phase, artifact_bytes)
    if phase == "pre_baseline":
        # Validate that the owner is signing a pin for these exact source rows.
        _verify_external_cohort_pin(
            stage1_review_and_lineage_raw=artifact_bytes["stage1_review_and_lineage"],
            external_cohort_pin_raw=artifact_bytes["external_cohort_pin"],
            expected_external_cohort_pin_sha256=_sha(
                artifact_bytes["external_cohort_pin"]))
    records = _record_bindings(record_rows)
    if phase == "pre_baseline":
        _verify_stage1_record_bindings(
            artifact_bytes["stage1_review_and_lineage"], records)
    release, _ = _roster_keys(
        integrity_roster_raw=integrity_roster_raw,
        release_roster_raw=release_roster_raw,
        root_public_key_pem=root_public_key_pem,
        expected_root_public_der_sha256=expected_root_public_der_sha256)
    body = _body(phase=phase, freeze_id=freeze_id, records=records,
                 artifact_bindings=bindings,
                 signer_key_id=release["freeze_owner"],
                 prior_phase1_receipt_sha256=prior_hash)
    signature = _sign(freeze_owner_private_key_pem,
                      role_keys["freeze_owner"],
                      _PHASE_DOMAINS[phase] + _canonical(body))
    return _canonical({**body, "signature_ed25519_base64":
                       base64.b64encode(signature).decode("ascii")}) + b"\n"


def verify_phase_freeze_receipt(*, receipt_raw: bytes, phase: str,
                                artifact_bytes: Mapping[str, bytes],
                                integrity_roster_raw: bytes,
                                release_roster_raw: bytes,
                                root_public_key_pem: bytes,
                                expected_root_public_der_sha256: str,
                                expected_external_cohort_pin_sha256: str,
                                prior_phase1_receipt_raw: bytes | None = None,
                                prior_phase1_artifact_bytes: Mapping[str, bytes] | None = None) -> dict[str, Any]:
    """Verify signatures and exact bytes; return a non-authorizing report only."""
    if not isinstance(receipt_raw, bytes) or not receipt_raw or len(receipt_raw) > _MAX_JSON:
        raise ReceiptError("phase_receipt_bytes_invalid")
    bindings = _bindings(phase, artifact_bytes)
    records_from_call = None
    if phase == "pre_baseline":
        pin_sha256 = _verify_external_cohort_pin(
            stage1_review_and_lineage_raw=artifact_bytes["stage1_review_and_lineage"],
            external_cohort_pin_raw=artifact_bytes["external_cohort_pin"],
            expected_external_cohort_pin_sha256=expected_external_cohort_pin_sha256)
    if phase == "post_baseline":
        if (not isinstance(prior_phase1_receipt_raw, bytes)
                or not isinstance(prior_phase1_artifact_bytes, Mapping)
                or artifact_bytes.get("phase1_receipt") != prior_phase1_receipt_raw):
            raise ReceiptError("phase1_receipt_required_or_mismatched")
        prior_report = verify_phase_freeze_receipt(
            receipt_raw=prior_phase1_receipt_raw, phase="pre_baseline",
            artifact_bytes=prior_phase1_artifact_bytes,
            integrity_roster_raw=integrity_roster_raw,
            release_roster_raw=release_roster_raw,
            root_public_key_pem=root_public_key_pem,
            expected_root_public_der_sha256=expected_root_public_der_sha256,
            expected_external_cohort_pin_sha256=expected_external_cohort_pin_sha256)
        prior_obj = _parse(prior_phase1_receipt_raw, "phase_receipt_json_invalid")
        records_from_call = _record_bindings(prior_obj.get("record_bindings"))
    elif phase != "pre_baseline":
        raise ReceiptError("phase_invalid")
    elif prior_phase1_receipt_raw is not None or prior_phase1_artifact_bytes is not None:
        raise ReceiptError("phase1_receipt_not_allowed")
    release, role_keys = _roster_keys(
        integrity_roster_raw=integrity_roster_raw,
        release_roster_raw=release_roster_raw,
        root_public_key_pem=root_public_key_pem,
        expected_root_public_der_sha256=expected_root_public_der_sha256)
    obj = _parse(receipt_raw, "phase_receipt_json_invalid")
    expected_keys = {"schema_version", "purpose", "phase", "freeze_id",
                     "record_bindings", "scheduled_branches_per_row", "planned_forward_count",
                     "artifact_bindings", "signer_role", "signer_key_id", "status",
                     "execution_authorization", "model_load_permitted",
                     "scoring_authorized", "phase_authorized", "authorized_forwards",
                     "signature_ed25519_base64"}
    if phase == "post_baseline":
        expected_keys.add("phase1_receipt_sha256")
    if not isinstance(obj, dict) or set(obj) != expected_keys:
        raise ReceiptError("phase_receipt_schema_invalid")
    version = 1 if phase == "pre_baseline" else 3
    if (type(obj.get("schema_version")) is not int
            or obj.get("schema_version") != version
            or obj.get("purpose") != f"qta_two_phase_score_freeze_candidate_v{version}"):
        raise ReceiptError("phase_receipt_schema_version_invalid")
    record_rows = _record_bindings(obj["record_bindings"])
    if phase == "pre_baseline":
        _verify_stage1_record_bindings(
            artifact_bytes["stage1_review_and_lineage"], record_rows)
    if records_from_call is not None and record_rows != records_from_call:
        raise ReceiptError("phase2_record_binding_differs_from_phase1")
    prior_hash = _sha(prior_phase1_receipt_raw) if phase == "post_baseline" else None
    body = _body(phase=phase, freeze_id=obj["freeze_id"], records=record_rows,
                 artifact_bindings=bindings,
                 signer_key_id=release["freeze_owner"],
                 prior_phase1_receipt_sha256=prior_hash)
    if any(obj.get(key) != value for key, value in body.items()):
        raise ReceiptError("phase_receipt_binding_mismatch")
    signature = _decode_signature(obj["signature_ed25519_base64"],
                                  "phase_receipt_signature_invalid")
    signed_body = {key: value for key, value in obj.items()
                   if key != "signature_ed25519_base64"}
    if not _verify_signature(role_keys["freeze_owner"], signature,
                             _PHASE_DOMAINS[phase] + _canonical(signed_body)):
        raise ReceiptError("phase_receipt_signature_invalid")
    if phase == "post_baseline":
        prior_role = next(row for row in bindings if row["role"] == "phase1_receipt")
        if prior_role["sha256"] != prior_hash:
            raise ReceiptError("phase1_receipt_hash_binding_mismatch")
    return {
        "status": "verified_signed_byte_bindings_non_authorizing",
        "phase": phase,
        "requested_rows": ROW_COUNT,
        "scheduled_branches": ROW_COUNT * len(PHASE_BRANCHES[phase]),
        "artifact_binding_count": len(bindings),
        "receipt_signature_valid": True,
        "artifact_bytes_match": True,
        "external_cohort_pin_sha256": (pin_sha256 if phase == "pre_baseline"
                                        else expected_external_cohort_pin_sha256),
        "external_cohort_pin_matches_caller_pin": True,
        "external_cohort_pin_origin_authenticated": False,
        "prior_phase1_receipt_valid": phase == "post_baseline",
        "human_identity_verified": False,
        "semantic_review_authenticated": False,
        "execution_authorization": "not_granted",
        "model_load_permitted": False,
        "scoring_authorized": False,
        "authorized_forwards": 0,
    }
