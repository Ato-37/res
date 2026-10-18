"""Derive non-authorizing QTA source pins after rerunning stage-1 and QA audits.

This CPU-only adapter accepts exact source bytes through the existing audit
argument maps. It never accepts a caller-provided success summary, loads a
model, or authorizes scoring. The returned source pins are structural bindings,
not proof of reviewer identity or semantic correctness.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from qta_actual_qa_source_contract import bind_qta_actual_qa_source
from qta_prescore_mapping_binding import interval_list_sha256
from qta_prescore_review_row_adapter import adapt_review_bundle_to_qta_rows
from qta_stage1_bundle_audit import audit_qta_stage1_bundle


ROW_COUNT = 3
SHA256 = re.compile(r"^[0-9a-f]{64}$")
DENIAL = {
    "model_load_permitted": False,
    "execution_authorization": "not_granted",
    "scoring_authorized": False,
}


class Stage1SourcePinError(ValueError):
    """Stable, content-free adapter error."""


def _fail(code: str) -> None:
    raise Stage1SourcePinError(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (UnicodeError, TypeError, ValueError, OverflowError):
        _fail("canonical_json_invalid")


def _strict_json(raw: bytes, code: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("duplicate_json_key")
            result[key] = value
        return result

    if not isinstance(raw, bytes) or not raw:
        _fail(code)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: _fail("nonfinite_json"))
    except Stage1SourcePinError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        _fail(code)


def _require_stage1_pass(report: Any) -> None:
    if (not isinstance(report, dict)
            or report.get("status") != "stage1_structurally_valid_non_authorizing"
            or type(report.get("requested_records")) is not int
            or report.get("requested_records") != ROW_COUNT
            or type(report.get("effective_records")) is not int
            or report.get("effective_records") != ROW_COUNT
            or type(report.get("structurally_valid_records")) is not int
            or report.get("structurally_valid_records") != ROW_COUNT
            or type(report.get("failure_count")) is not int
            or report.get("failure_count") != 0
            or report.get("failure_codes") != []
            or report.get("model_load_permitted") is not False
            or report.get("score_ready") is not False
            or report.get("execution_authorization") != "not_granted"
            or report.get("reviewer_identity_verified") is not False
            or report.get("semantic_review_authenticated") is not False):
        _fail("stage1_audit_not_complete")
    components = report.get("components")
    needed = ("worksheet_bytes", "lineage_bytes", "review_bytes", "review_index_bytes",
              "custody_bytes", "review_handoff", "custody", "release_witness",
              "reviewer_attestation", "artifact_snapshot", "media_snapshot_stability")
    if (not isinstance(components, dict)
            or any(not isinstance(components.get(name), dict)
                   or components[name].get("passed") is not True for name in needed)):
        _fail("stage1_components_incomplete")
    bindings = report.get("artifact_role_bindings")
    required_roles = {
        "claim_mapping_worksheet", "claim_lineage", "review_custody",
        "review_package_html",
        "standalone_review_html",
        *(f"staged_media_{i:02d}" for i in range(1, 4)),
        *(f"review_package_media_{i:02d}" for i in range(1, 4)),
    }
    if (not isinstance(bindings, dict) or not required_roles.issubset(bindings)
            or any(not isinstance(value, dict) or value.get("passed") is not True
                   for value in bindings.values())
            or any(bindings[role].get("passed") is not True for role in required_roles)):
        _fail("stage1_artifact_bindings_incomplete")


def derive_stage1_source_pins(*, stage1_audit_inputs: Mapping[str, Any],
                              actual_qa_source_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Rerun both validators and derive exactly three ordered baseline source pins.

    ``stage1_audit_inputs`` is passed directly to
    ``audit_qta_stage1_bundle``. ``actual_qa_source_inputs`` is passed directly
    to ``bind_qta_actual_qa_source``. Both must contain original exact bytes and
    external pins required by those existing validators.
    """
    if not isinstance(stage1_audit_inputs, Mapping) or not isinstance(actual_qa_source_inputs, Mapping):
        _fail("audit_input_maps_required")
    try:
        stage1 = audit_qta_stage1_bundle(**dict(stage1_audit_inputs))
    except Exception:
        _fail("stage1_audit_call_failed")
    _require_stage1_pass(stage1)

    try:
        review_rows = adapt_review_bundle_to_qta_rows(
            worksheet_raw=stage1_audit_inputs["worksheet_raw"],
            lineage_raw=stage1_audit_inputs["lineage_raw"],
            review_raw=stage1_audit_inputs["review_raw"],
            review_index_raw=stage1_audit_inputs["review_index_raw"],
            expected_lineage_sha256=stage1_audit_inputs["expected_lineage_sha256"],
        )
    except Exception:
        _fail("review_row_adaptation_failed")

    try:
        qa_result = bind_qta_actual_qa_source(**dict(actual_qa_source_inputs))
    except Exception:
        _fail("actual_qa_source_binding_call_failed")
    summary = qa_result.get("summary") if isinstance(qa_result, dict) else None
    private_bundle = qa_result.get("private_bundle") if isinstance(qa_result, dict) else None
    if (not isinstance(summary, dict) or summary.get("status") != "bound_unvalidated"
            or summary.get("requested_rows") != ROW_COUNT
            or summary.get("validated_rows") != ROW_COUNT
            or summary.get("model_load_permitted") is not False
            or summary.get("scoring_authorized") is not False
            or not isinstance(private_bundle, dict)
            or private_bundle.get("classification") != "private_local_only_unvalidated_qa_source"
            or not isinstance(private_bundle.get("raw_json"), bytes)
            or not isinstance(private_bundle.get("sha256"), str)
            or _sha(private_bundle["raw_json"]) != private_bundle["sha256"]):
        _fail("actual_qa_source_binding_not_complete")
    private = _strict_json(private_bundle["raw_json"], "actual_qa_bundle_invalid")
    qa_source = private.get("qa_source")
    crosswalk = private.get("review_source_crosswalk")
    if (not isinstance(qa_source, dict) or not isinstance(qa_source.get("records"), list)
            or not isinstance(crosswalk, list) or len(crosswalk) != ROW_COUNT
            or len(qa_source["records"]) != ROW_COUNT or len(review_rows) != ROW_COUNT):
        _fail("actual_qa_rows_invalid")

    worksheet = _strict_json(stage1_audit_inputs["worksheet_raw"], "worksheet_json_invalid")
    lineage = _strict_json(stage1_audit_inputs["lineage_raw"], "lineage_json_invalid")
    worksheet_rows = worksheet.get("records") if isinstance(worksheet, dict) else None
    lineage_rows = lineage.get("records") if isinstance(lineage, dict) else None
    if (not isinstance(worksheet_rows, list) or len(worksheet_rows) != ROW_COUNT
            or not isinstance(lineage_rows, list) or len(lineage_rows) != ROW_COUNT):
        _fail("stage1_source_rows_invalid")

    pin_rows = []
    seen_tokens: set[str] = set()
    for index, (review, cross, qa, worksheet_row, lineage_row) in enumerate(
            zip(review_rows, crosswalk, qa_source["records"], worksheet_rows,
                lineage_rows, strict=True)):
        if any(not isinstance(row, dict) for row in
               (review, cross, qa, worksheet_row, lineage_row)):
            _fail(f"row_{index + 1}_schema_invalid")
        token = review.get("review_token")
        if (not isinstance(token, str) or token in seen_tokens
                or token != worksheet_row.get("review_token")
                or token != cross.get("review_token")):
            _fail(f"row_{index + 1}_token_order_mismatch")
        seen_tokens.add(token)
        if review.get("support_decision") != "supports_claim":
            _fail(f"row_{index + 1}_video_claim_not_supported")
        intervals_sha = interval_list_sha256(review.get("evidence_intervals_seconds"))
        claim = worksheet_row.get("claim_text")
        question = qa.get("question")
        options = qa.get("choices")
        if (not isinstance(claim, str) or not claim.strip()
                or not isinstance(question, str) or not question.strip()
                or not isinstance(options, list) or len(options) != 5
                or any(not isinstance(option, str) or not option.strip() for option in options)):
            _fail(f"row_{index + 1}_source_text_invalid")
        claim_sha = _sha(claim.encode("utf-8"))
        question_sha = _sha(question.encode("utf-8"))
        options_sha = _sha(_canonical(options))
        if (cross.get("video_id") != worksheet_row.get("video_id")
                or cross.get("claim_id") != worksheet_row.get("claim_id")
                or cross.get("dataset_video_id") != lineage_row.get("dataset_video_id")
                or cross.get("dataset_question_id") != lineage_row.get("dataset_question_id")
                or qa.get("video_id") != worksheet_row.get("video_id")
                or qa.get("dataset_video_id") != cross.get("dataset_video_id")
                or qa.get("dataset_question_id") != cross.get("dataset_question_id")
                or _sha(claim.encode("utf-8")) != worksheet_row.get("claim_text_sha256")
                or question_sha != lineage_row.get("question_sha256")):
            _fail(f"row_{index + 1}_source_identity_mismatch")
        video_sha = worksheet_row.get("video_sha256")
        if (not isinstance(video_sha, str) or not SHA256.fullmatch(video_sha)
                or video_sha != review.get("source_video_sha256")):
            _fail(f"row_{index + 1}_source_video_mismatch")
        pin_rows.append({
            "review_token": token,
            "question_sha256": question_sha,
            "options_sha256": options_sha,
            "claim_sha256": claim_sha,
            "source_video_sha256": video_sha,
            "video_support_decision": "supports_claim",
            "reviewed_intervals_sha256": intervals_sha,
        })
    if len(pin_rows) != ROW_COUNT or len(seen_tokens) != ROW_COUNT:
        _fail("cohort_pin_count_invalid")

    # Internal consistency with hashes already frozen by the QA binder.
    target_rows = private.get("bound_manifest", {}).get("target_manifest", {}).get("records")
    if not isinstance(target_rows, list) or len(target_rows) != ROW_COUNT:
        _fail("actual_qa_target_rows_invalid")
    for index, (pin, qa, target) in enumerate(zip(pin_rows, qa_source["records"],
                                                  target_rows, strict=True)):
        if (target.get("question_sha256") != pin["question_sha256"]
                or target.get("ordered_choice_sha256") != [
                    _sha(option.encode("utf-8")) for option in qa["choices"]]
                or target.get("claim_text_sha256") != pin["claim_sha256"]):
            _fail(f"row_{index + 1}_target_source_pin_mismatch")

    receipt_sha = _sha(stage1_audit_inputs["pre_review_receipt_raw"])
    review_payload_sha = _sha(stage1_audit_inputs["review_raw"])
    expected_baseline_pins = {
        row["review_token"]: {
            "question_sha256": row["question_sha256"],
            "options_sha256": row["options_sha256"],
            "claim_sha256": row["claim_sha256"],
        }
        for row in pin_rows
    }
    cohort_source_pins = {
        row["review_token"]: {
            "source_video_sha256": row["source_video_sha256"],
            "video_support_decision": row["video_support_decision"],
            "reviewed_intervals_sha256": row["reviewed_intervals_sha256"],
        }
        for row in pin_rows
    }
    baseline_candidate_source_bindings = {
        token: {
            **expected_baseline_pins[token],
            **cohort_source_pins[token],
        }
        for token in (row["review_token"] for row in pin_rows)
    }
    result = {
        "schema_version": 1,
        "purpose": "qta_stage1_source_pins_structurally_derived_non_authorizing",
        "classification": "private_local_only_structural_source_pins",
        "status": "three_row_source_pins_structurally_derived",
        "requested_records": ROW_COUNT,
        "effective_records": ROW_COUNT,
        "source_rows": pin_rows,
        "expected_baseline_source_sha256_by_token": expected_baseline_pins,
        "cohort_review_source_pins_by_token": cohort_source_pins,
        "expected_baseline_candidate_source_bindings_by_token":
            baseline_candidate_source_bindings,
        "expected_cohort_tokens": [row["review_token"] for row in pin_rows],
        "expected_video_claim_review_receipt_sha256": receipt_sha,
        "expected_review_payload_sha256": review_payload_sha,
        "stage1_audit_status": stage1["status"],
        "actual_qa_source_status": summary["status"],
        "stage1_source_bytes_sha256": {
            "worksheet": _sha(stage1_audit_inputs["worksheet_raw"]),
            "lineage": _sha(stage1_audit_inputs["lineage_raw"]),
            "review": review_payload_sha,
            "review_index": _sha(stage1_audit_inputs["review_index_raw"]),
            "custody": _sha(stage1_audit_inputs["custody_raw"]),
            "pre_review_receipt": receipt_sha,
        },
        "private_qa_bundle_sha256": private_bundle["sha256"],
        "reviewer_identity_verified": False,
        "semantic_review_authenticated": False,
        "clock_frame_correspondence_verified": False,
        "scoring_authorized": False,
        "limitations": [
            "source pins establish byte and field consistency only",
            "reviewer identity, actual viewing, and semantic correctness are not authenticated",
            "browser player to decoder frame correspondence and clock calibration are separate unverified gates",
            "trust-root, distinct-role ceremony, joint freeze, and prediction-link adjudication remain separate production gates",
            "rights status remains unverified and private QA bytes must remain local",
        ],
        **DENIAL,
    }
    return result
