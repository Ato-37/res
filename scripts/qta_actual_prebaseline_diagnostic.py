"""CPU-only, fixed-cohort diagnostic of QTA source/review input bytes.

This module reruns only byte-oriented leaf validators and reports missing later
gates. It is not a readiness decision, phase gate, model runner, or score tool.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from qta_actual_qa_source_contract import bind_qta_actual_qa_source
from qta_prebaseline_live_batch_binding import validate_live_processor_batches
from qta_prescore_review_row_adapter import adapt_review_bundle_to_qta_rows

ROW_COUNT = 3
DENIAL = {
    "model_load_permitted": False,
    "scoring_authorized": False,
    "phase_authorized": False,
    "forwards": 0,
    "score_calls": 0,
}
SOURCE_KEYS = {
    "csv_raw", "claim_manifest_raw", "selection_raw", "target_manifest_raw",
    "lineage_manifest_raw", "qta_protocol_raw", "expected_qta_protocol_sha256",
    "qta_config_raw", "expected_qta_config_sha256",
}
REVIEW_KEYS = {
    "worksheet_raw", "lineage_raw", "review_raw", "review_index_raw",
    "expected_lineage_sha256",
}
MAX_RAW_INPUT_BYTES = 24_000_000
NOT_SUPPLIED = (
    "stage1_release_custody", "clock_frame_evidence", "clock_calibration",
    "prescore_mapping", "baseline_full_video_pq_freeze",
    "stage2_prediction_link_semantic_review", "human_identity_and_semantic_review_proof",
)


def _component(status: str, *, failure_code: str | None = None,
               rows: int = 0) -> dict[str, Any]:
    return {
        "status": status,
        "requested_rows": ROW_COUNT,
        "validated_rows": rows if status == "passed" else 0,
        "provenance_level": ("not_supplied_unverified" if status == "not_supplied"
                              else "caller_pinned_unverified"),
        "failure_code": failure_code,
    }


def _safe_error(exc: Exception) -> str:
    value = str(exc)
    return value if re.fullmatch(r"[a-z][a-z0-9_]{0,119}", value) else "validator_failed"


def diagnose_qta_prebaseline(*, actual_source_inputs: Mapping[str, Any] | None,
                             review_bundle_inputs: Mapping[str, Any] | None,
                             expected_processor_rows: Any = None,
                             live_processor_rows: Any = None) -> dict[str, Any]:
    """Rerun source and review-byte validators; never grant downstream authority.

    `actual_source_inputs` must be the exact keyword inputs to
    ``bind_qta_actual_qa_source``. `review_bundle_inputs` must be the exact
    keyword inputs to ``adapt_review_bundle_to_qta_rows``. Only bytes plus
    explicitly supplied SHA-256 pin strings are accepted; supplied reports or
    success summaries are not accepted as inputs.
    """
    gates: dict[str, dict[str, Any]] = {}
    private_binding = None
    private_review_rows = None
    private_crosswalk = None

    if actual_source_inputs is None:
        gates["actual_qa_source_binding"] = _component("not_supplied")
    else:
        try:
            if not isinstance(actual_source_inputs, Mapping) or set(actual_source_inputs) != SOURCE_KEYS:
                raise ValueError("source_input_schema_invalid")
            if any(not isinstance(actual_source_inputs[key], bytes)
                   for key in SOURCE_KEYS - {"expected_qta_protocol_sha256",
                                             "expected_qta_config_sha256"}):
                raise ValueError("source_raw_bytes_required")
            if any(isinstance(value, bytes) and len(value) > MAX_RAW_INPUT_BYTES
                   for value in actual_source_inputs.values()):
                raise ValueError("source_raw_input_too_large")
            result = bind_qta_actual_qa_source(**dict(actual_source_inputs))
            summary = result.get("summary") if isinstance(result, dict) else None
            if (not isinstance(summary, dict) or summary.get("status") != "bound_unvalidated"
                    or summary.get("requested_rows") != ROW_COUNT
                    or summary.get("validated_rows") != ROW_COUNT
                    or result.get("private_bundle") is None):
                raise ValueError("source_binding_blocked")
            private_binding = result["private_bundle"]["raw_json"]
            gates["actual_qa_source_binding"] = _component("passed", rows=3)
        except Exception as exc:
            gates["actual_qa_source_binding"] = _component(
                "blocked", failure_code=_safe_error(exc))

    if review_bundle_inputs is None:
        gates["review_bundle_byte_structure"] = _component("not_supplied")
    else:
        try:
            if not isinstance(review_bundle_inputs, Mapping) or set(review_bundle_inputs) != REVIEW_KEYS:
                raise ValueError("review_input_schema_invalid")
            if any(not isinstance(review_bundle_inputs[key], bytes)
                   for key in REVIEW_KEYS - {"expected_lineage_sha256"}):
                raise ValueError("review_raw_bytes_required")
            if any(isinstance(value, bytes) and len(value) > MAX_RAW_INPUT_BYTES
                   for value in review_bundle_inputs.values()):
                raise ValueError("review_raw_input_too_large")
            private_review_rows = adapt_review_bundle_to_qta_rows(
                **dict(review_bundle_inputs))
            if not isinstance(private_review_rows, list) or len(private_review_rows) != ROW_COUNT:
                raise ValueError("review_row_count_invalid")
            gates["review_bundle_byte_structure"] = _component(
                "passed", rows=3)
        except Exception as exc:
            gates["review_bundle_byte_structure"] = _component(
                "blocked", failure_code=_safe_error(exc))

    if private_binding is None or private_review_rows is None:
        gates["source_to_review_ordered_join"] = _component(
            "blocked", failure_code="upstream_byte_validator_blocked")
    else:
        try:
            private = json.loads(private_binding.decode("utf-8"))
            crosswalk = private["review_source_crosswalk"]
            live_source_crosswalk = []
            lineage = json.loads(actual_source_inputs["lineage_manifest_raw"].decode("utf-8"))
            claims = json.loads(actual_source_inputs["claim_manifest_raw"].decode("utf-8"))
            worksheet = json.loads(review_bundle_inputs["worksheet_raw"].decode("utf-8"))
            if (len(crosswalk) != ROW_COUNT or len(lineage["records"]) != ROW_COUNT
                    or len(claims["records"]) != ROW_COUNT
                    or len(worksheet["records"]) != ROW_COUNT
                    or len(private_review_rows) != ROW_COUNT):
                raise ValueError("ordered_join_row_count_invalid")
            for cross, line, claim, worksheet_row, review in zip(
                    crosswalk, lineage["records"], claims["records"],
                    worksheet["records"], private_review_rows, strict=True):
                if (cross.get("review_token") != review.get("review_token")
                        or cross.get("review_token") != worksheet_row.get("review_token")
                        or cross.get("video_id") != line.get("video_id")
                        or cross.get("video_id") != claim.get("video_id")
                        or cross.get("video_id") != worksheet_row.get("video_id")
                        or cross.get("claim_id") != claim.get("claim_id")
                        or cross.get("claim_id") != worksheet_row.get("claim_id")
                        or claim.get("claim_text_sha256")
                        != worksheet_row.get("claim_text_sha256")
                        or line.get("video_sha256") != review.get("source_video_sha256")):
                    raise ValueError("ordered_source_review_claim_video_mismatch")
                live_source_crosswalk.append({
                    "review_token": cross["review_token"],
                    "video_sha256": line["video_sha256"],
                })
            private_crosswalk = live_source_crosswalk
            gates["source_to_review_ordered_join"] = _component(
                "passed", rows=3)
        except Exception as exc:
            gates["source_to_review_ordered_join"] = _component(
                "blocked", failure_code=_safe_error(exc))

    if expected_processor_rows is None and live_processor_rows is None:
        gates["processor_batch_binding"] = _component("not_supplied")
    elif gates["source_to_review_ordered_join"]["status"] != "passed":
        gates["processor_batch_binding"] = _component(
            "blocked", failure_code="upstream_source_review_join_blocked")
    else:
        result = validate_live_processor_batches(
            expected_rows=expected_processor_rows,
            live_rows=live_processor_rows,
            source_crosswalk=private_crosswalk,
        )
        gates["processor_batch_binding"] = _component(
            result["status"], failure_code=result["failure_code"],
            rows=result["validated_rows"],
        )

    for name in NOT_SUPPLIED:
        gates[name] = _component("not_supplied")

    # A structurally complete byte join is still only a diagnostic. In
    # particular, review declarations and signatures never imply human truth.
    return {
        "schema_version": 1,
        "purpose": "qta_prebaseline_input_diagnostic_only",
        "status": "diagnostic_incomplete",
        "requested_rows": ROW_COUNT,
        "gates": gates,
        "human_identity_verified": False,
        "semantic_review_authenticated": False,
        "baseline_pq_frozen": False,
        "faithfulness_scored": False,
        **DENIAL,
    }
