"""Non-authorizing structural pre-load gate for the fixed three-row QTA cohort.

This boundary accepts raw bridge inputs, never caller-produced success summaries.
It re-runs the existing source/review/clock-presence validators through
``qta_actual_candidate_input_bridge`` and discards all row-level material.
Structural composition is not direct-faithfulness readiness: trust-rooted human
review, semantic adjudication, 15 clock/frame correspondences, frozen baseline
p/q and prediction-link review, and production batch provenance remain explicit
denials. This module has no model, CUDA, forward, scoring, or authorization path.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from audit_nextgqa_clock_frame_evidence import audit_evidence as audit_clock_frame_evidence
from qta_actual_candidate_input_bridge import build_qta_actual_candidate_input_material

ROW_COUNT = 3
ANCHOR_COUNT = 15


def evaluate_qta_direct_faithfulness_preload(*, sampler_receipt_raw: bytes,
        candidate_inventory_raw: bytes, media_sha256_by_role: Mapping[str, str],
        actual_source_inputs: Mapping[str, Any], stage1_audit_inputs: Mapping[str, Any],
        actual_qa_source_inputs: Mapping[str, Any],
        clock_inputs_by_token: Mapping[str, Mapping[str, Any]],
        processor_pins_by_token: Mapping[str, Mapping[str, Any]],
        tokenizer_sha256: str, video_duration_seconds_by_token: Mapping[str, Any],
        clock_frame_audit_inputs: Mapping[str, Any] | None = None,
        repo_root=None, code_files=None) -> dict[str, Any]:
    """Revalidate exact inputs and return sanitized aggregate gate statuses.

    No caller-supplied validator summary is accepted. Failure codes and all
    private bridge outputs are intentionally omitted from the public result.
    A structurally valid result still cannot satisfy any execution gate.
    """
    result = {
        "schema_version": 2,
        "purpose": "qta_direct_faithfulness_prebaseline_structure_diagnostic",
        "phase": "pre_baseline_candidate",
        "status": "blocked",
        "requested_rows": ROW_COUNT,
        "validated_rows": 0,
        "structurally_composed_rows": 0,
        "requested_clock_frame_anchors": ANCHOR_COUNT,
        "validated_clock_frame_anchors": 0,
        "signature_bound_frame_pair_decisions": 0,
        "clock_frame_audit_status": "not_supplied",
        "clock_frame_trust_root_origin_authenticated": False,
        "structural_source_composition": "blocked",
        "trusted_human_review": "blocked",
        "semantic_review": "blocked",
        "clock_frame_correspondence": "blocked",
        # These are intentionally not pre-baseline gates. Prediction-link
        # review depends on the exact frozen baseline predictions and p/q.
        "baseline_prediction_manifest": "not_applicable_pre_baseline",
        "prediction_link_review": "not_yet_applicable_baseline_required",
        "processor_batch_provenance": "blocked",
        "direct_faithfulness_ready": False,
        "model_load_permitted": False,
        "cuda_used": False,
        "model_loaded": False,
        "forward_calls": 0,
        "score_calls": 0,
        "faithfulness_scored": False,
        "execution_authorization": "not_granted",
    }
    if clock_frame_audit_inputs is not None:
        # Rerun the evidence audit from raw bytes; never accept a caller's
        # summary. A passing v3 artifact proves only signature and byte
        # consistency against the supplied root. The root's external origin
        # and the signer's real-world identity remain unverified here.
        try:
            if not isinstance(clock_frame_audit_inputs, Mapping):
                raise ValueError("clock_frame_audit_inputs_invalid")
            clock_report = audit_clock_frame_evidence(**dict(clock_frame_audit_inputs))
            if (isinstance(clock_report, dict)
                    and clock_report.get("status") == "candidate_signed_evidence_bytes_consistent"
                    and clock_report.get("requested_records") == ROW_COUNT
                    and clock_report.get("requested_anchors") == ANCHOR_COUNT
                    and clock_report.get("verified_anchors") == ANCHOR_COUNT
                    and clock_report.get("match_decisions") == ANCHOR_COUNT
                    and clock_report.get("ambiguous_decisions") == 0
                    and clock_report.get("mismatch_decisions") == 0
                    and clock_report.get("screenshot_and_decoder_bytes_checked") is True
                    and clock_report.get("reviewer_signature_verified") is True
                    and clock_report.get("frame_reviewer_role_key_verified") is True
                    and clock_report.get("capture_operator_signature_verified") is True
                    and clock_report.get("capture_operator_human_identity_verified") is False
                    and clock_report.get("execution_authorization") == "not_granted"):
                result.update({
                    "signature_bound_frame_pair_decisions": ANCHOR_COUNT,
                    "clock_frame_audit_status": "signed_pairs_consistent_untrusted_root",
                })
            else:
                result["clock_frame_audit_status"] = "blocked_or_incomplete"
        except Exception:
            result["clock_frame_audit_status"] = "blocked_or_incomplete"
    try:
        material = build_qta_actual_candidate_input_material(
            sampler_receipt_raw=sampler_receipt_raw,
            candidate_inventory_raw=candidate_inventory_raw,
            media_sha256_by_role=media_sha256_by_role,
            actual_source_inputs=actual_source_inputs,
            stage1_audit_inputs=stage1_audit_inputs,
            actual_qa_source_inputs=actual_qa_source_inputs,
            clock_inputs_by_token=clock_inputs_by_token,
            processor_pins_by_token=processor_pins_by_token,
            tokenizer_sha256=tokenizer_sha256,
            video_duration_seconds_by_token=video_duration_seconds_by_token,
            **({} if repo_root is None else {"repo_root": repo_root}),
            **({} if code_files is None else {"code_files": code_files}),
        )
        if (material.get("status") != "three_row_candidate_inputs_derived_unvalidated"
                or material.get("requested_records") != ROW_COUNT
                or material.get("effective_records") != ROW_COUNT
                or len(material.get("expected_rows", ())) != ROW_COUNT
                or len(material.get("sampler_rows", ())) != ROW_COUNT
                or material.get("human_identity_verified") is not False
                or material.get("semantic_review_authenticated") is not False
                or material.get("clock_frame_correspondence_verified") is not False
                or material.get("model_load_permitted") is not False
                or material.get("scoring_authorized") is not False
                or material.get("faithfulness_scored") is not False
                or material.get("execution_authorization") != "not_granted"):
            return result
        result.update({
            "status": "structural_inputs_composed_unvalidated",
            # Do not call these rows validated for a faithfulness score. Only
            # source/review candidate structure was composed at this stage.
            "structurally_composed_rows": ROW_COUNT,
            "structural_source_composition": "passed_unvalidated",
        })
        return result
    except Exception:
        # Do not return exception text: it may contain paths or input-derived data.
        return result
