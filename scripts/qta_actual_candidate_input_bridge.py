"""Join authoritative local preflight/review/clock artifacts into candidate pins.

This CPU-only bridge reruns existing structural validators and emits private,
non-authorizing inputs for ``qta_actual_processor_row_candidate``. It does not
authenticate reviewer identity or semantics, nor browser-clock-to-decoder-frame
correspondence. In particular, clock calibration presence is not correspondence
verification and can never authorize model loading or scoring.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from qta_actual_media_source_preflight import preflight_actual_media_source_join
from qta_prescore_clock_row_adapter import adapt_calibration_to_clock_row
from qta_prescore_mapping_binding import interval_list_sha256
from qta_prescore_review_row_adapter import adapt_review_bundle_to_qta_rows
from qta_stage1_source_pin_adapter import derive_stage1_source_pins
from qta_actual_qa_source_contract import bind_qta_actual_qa_source
from run_qta_actual_qa_source_binding_smoke import (
    _build_claim_and_selection, _target_manifest,
)

ROW_COUNT = 3
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(code)
    return value


def _parse_sampler(raw: bytes) -> dict[str, dict[str, Any]]:
    try:
        report = json.loads(raw.decode("utf-8"))
    except (AttributeError, UnicodeError, json.JSONDecodeError):
        raise ValueError("bridge_sampler_receipt_invalid") from None
    if (not isinstance(report, dict) or report.get("purpose")
            != "local_qta_real_video_sampler_eligibility_preflight"
            or report.get("status") != "passed"
            or report.get("execution_authorization") != "not_granted"
            or report.get("gpu_used") is not False
            or report.get("model_loaded") is not False
            or report.get("faithfulness_scored") is not False
            or report.get("requested_records") != ROW_COUNT
            or report.get("effective_records") != ROW_COUNT
            or report.get("frames_per_video") != 32
            or not isinstance(report.get("rows"), list)
            or len(report["rows"]) != ROW_COUNT):
        raise ValueError("bridge_sampler_receipt_invalid")
    by_source: dict[str, dict[str, Any]] = {}
    for row in report["rows"]:
        if not isinstance(row, dict):
            raise ValueError("bridge_sampler_row_invalid")
        source = _digest(row.get("source_sha256"), "bridge_sampler_row_invalid")
        if source in by_source:
            raise ValueError("bridge_sampler_source_duplicate")
        if (type(row.get("decoded_frame_count")) is not int
                or row["decoded_frame_count"] < 32
                or row.get("sampled_frame_count") != 32
                or not isinstance(row.get("sampled_indices"), list)
                or len(row["sampled_indices"]) != 32
                or not isinstance(row.get("sampled_pts"), list)
                or len(row["sampled_pts"]) != 32
                or not isinstance(row.get("effective_sample_fps"), list)
                or len(row["effective_sample_fps"]) != 2
                or not isinstance(row.get("sampled_rgb_sha256"), list)
                or len(row["sampled_rgb_sha256"]) != 32
                or any(not isinstance(x, str) or not SHA256.fullmatch(x)
                       for x in row["sampled_rgb_sha256"])):
            raise ValueError("bridge_sampler_row_incomplete")
        by_source[source] = row
    if len(by_source) != ROW_COUNT:
        raise ValueError("bridge_sampler_row_count_invalid")
    return by_source


def _cross_bind_stage1_to_actual_source(*, source_inputs: Mapping[str, Any],
                                        stage1_pins: Mapping[str, Any]) -> None:
    """Compare private QA/claim records by token without returning their text."""
    try:
        claim_raw, selection_raw, _worksheet, _lineage = _build_claim_and_selection(
            claim_raw=source_inputs["claim_input_manifest_raw"],
            worksheet_raw=source_inputs["worksheet_raw"],
            lineage_raw=source_inputs["lineage_manifest_raw"])
        target_raw = _target_manifest(source_inputs["target_candidate_raw"])
        bound = bind_qta_actual_qa_source(
            csv_raw=source_inputs["csv_raw"], claim_manifest_raw=claim_raw,
            selection_raw=selection_raw, target_manifest_raw=target_raw,
            lineage_manifest_raw=source_inputs["lineage_manifest_raw"],
            qta_protocol_raw=source_inputs["qta_protocol_raw"],
            expected_qta_protocol_sha256=_sha(source_inputs["qta_protocol_raw"]),
            qta_config_raw=source_inputs["qta_config_raw"],
            expected_qta_config_sha256=_sha(source_inputs["qta_config_raw"]))
        private = bound.get("private_bundle", {})
        raw = private.get("raw_json")
        if (bound.get("summary", {}).get("status") != "bound_unvalidated"
                or not isinstance(raw, bytes)
                or _sha(raw) != private.get("sha256")):
            raise ValueError
        private_doc = json.loads(raw.decode("utf-8"))
        actual_rows = private_doc["qa_source"]["records"]
        crosswalk = private_doc["review_source_crosswalk"]
        claims = json.loads(source_inputs["claim_input_manifest_raw"].decode("utf-8"))["claims"]
        lineage = json.loads(source_inputs["lineage_manifest_raw"].decode("utf-8"))["records"]
        actual_by_token = {}
        for link, qa in zip(crosswalk, actual_rows, strict=True):
            token = link["review_token"]
            actual_by_token[token] = (link, qa)
        claim_by_video = {row["video_id"]: row for row in claims}
        lineage_by_video = {row["video_id"]: row for row in lineage}
        if (len(actual_by_token) != ROW_COUNT or len(claim_by_video) != ROW_COUNT
                or len(lineage_by_video) != ROW_COUNT):
            raise ValueError
        for pin in stage1_pins["source_rows"]:
            token = pin["review_token"]
            link, qa = actual_by_token[token]
            video_id = link["video_id"]
            claim = claim_by_video[video_id]
            line = lineage_by_video[video_id]
            if (line.get("video_sha256") != pin["source_video_sha256"]
                    or _sha(qa["question"].encode("utf-8")) != pin["question_sha256"]
                    or _sha(_canonical(qa["choices"])) != pin["options_sha256"]
                    or claim.get("claim_text_sha256") != pin["claim_sha256"]):
                raise ValueError
    except Exception:
        raise ValueError("bridge_stage1_actual_qa_claim_mismatch") from None


def build_qta_actual_candidate_input_material(
        *, sampler_receipt_raw: bytes, candidate_inventory_raw: bytes,
        media_sha256_by_role: Mapping[str, str],
        actual_source_inputs: Mapping[str, Any],
        stage1_audit_inputs: Mapping[str, Any],
        actual_qa_source_inputs: Mapping[str, Any],
        clock_inputs_by_token: Mapping[str, Mapping[str, Any]],
        processor_pins_by_token: Mapping[str, Mapping[str, Any]],
        tokenizer_sha256: str,
        video_duration_seconds_by_token: Mapping[str, Any],
        repo_root=None, code_files=None) -> dict[str, Any]:
    """Build ordered expected pins and sampler rows from audited local inputs.

    ``clock_inputs_by_token`` must contain a calibration and its exact capture
    bytes for each reviewed token. This establishes structural calibration
    presence only. The current upstream contracts do not establish verified
    human semantic review or clock/frame correspondence; those flags remain
    false in every return value.
    """
    if not isinstance(tokenizer_sha256, str) or not SHA256.fullmatch(tokenizer_sha256):
        raise ValueError("bridge_tokenizer_digest_invalid")
    if not isinstance(clock_inputs_by_token, Mapping) or len(clock_inputs_by_token) != ROW_COUNT:
        raise ValueError("bridge_three_clock_inputs_required")
    if not isinstance(processor_pins_by_token, Mapping) or len(processor_pins_by_token) != ROW_COUNT:
        raise ValueError("bridge_three_processor_pins_required")
    if (not isinstance(video_duration_seconds_by_token, Mapping)
            or len(video_duration_seconds_by_token) != ROW_COUNT):
        raise ValueError("bridge_three_video_durations_required")
    if not isinstance(actual_source_inputs, Mapping) or not isinstance(actual_qa_source_inputs, Mapping):
        raise ValueError("bridge_authoritative_source_inputs_required")
    # Cross-bind common source bytes: preflight and Stage-1 must concern the
    # same CSV, lineage, protocol and treatment configuration.
    for source_key, qa_key in (("csv_raw", "csv_raw"),
                               ("lineage_manifest_raw", "lineage_manifest_raw"),
                               ("qta_protocol_raw", "qta_protocol_raw"),
                               ("qta_config_raw", "qta_config_raw")):
        if (not isinstance(actual_source_inputs.get(source_key), bytes)
                or actual_source_inputs[source_key] != actual_qa_source_inputs.get(qa_key)):
            raise ValueError("bridge_source_byte_sets_differ")

    preflight_kwargs = {
        "sampler_receipt_raw": sampler_receipt_raw,
        "candidate_inventory_raw": candidate_inventory_raw,
        "media_sha256_by_role": media_sha256_by_role,
        "actual_source_inputs": actual_source_inputs,
    }
    if repo_root is not None:
        preflight_kwargs["repo_root"] = repo_root
    if code_files is not None:
        preflight_kwargs["code_files"] = code_files
    preflight = preflight_actual_media_source_join(**preflight_kwargs)
    if (preflight.get("status") != "source_video_hash_joined_unvalidated"
            or preflight.get("joined_rows") != ROW_COUNT
            or preflight.get("human_identity_verified") is not False
            or preflight.get("semantic_review_authenticated") is not False):
        raise ValueError("bridge_actual_media_source_join_not_complete")

    pins = derive_stage1_source_pins(
        stage1_audit_inputs=stage1_audit_inputs,
        actual_qa_source_inputs=actual_qa_source_inputs)
    if (pins.get("status") != "three_row_source_pins_structurally_derived"
            or pins.get("effective_records") != ROW_COUNT
            or pins.get("reviewer_identity_verified") is not False
            or pins.get("semantic_review_authenticated") is not False):
        raise ValueError("bridge_stage1_review_inputs_not_complete")
    _cross_bind_stage1_to_actual_source(source_inputs=actual_source_inputs,
                                        stage1_pins=pins)
    source_rows = pins.get("source_rows")
    if not isinstance(source_rows, list) or len(source_rows) != ROW_COUNT:
        raise ValueError("bridge_stage1_rows_invalid")
    tokens = [row.get("review_token") for row in source_rows]
    if (len(set(tokens)) != ROW_COUNT
            or set(tokens) != set(clock_inputs_by_token)
            or set(tokens) != set(processor_pins_by_token)
            or set(tokens) != set(video_duration_seconds_by_token)):
        raise ValueError("bridge_review_clock_processor_cohort_mismatch")

    sampler_by_source = _parse_sampler(sampler_receipt_raw)
    source_set = {_digest(row.get("source_video_sha256"), "bridge_stage1_source_invalid")
                  for row in source_rows}
    if source_set != set(sampler_by_source):
        raise ValueError("bridge_stage1_sampler_source_set_mismatch")
    # Re-run the authoritative review adapter to recover the interval values;
    # only hashes of these values leave this function.
    review_rows = adapt_review_bundle_to_qta_rows(
        worksheet_raw=stage1_audit_inputs["worksheet_raw"],
        lineage_raw=stage1_audit_inputs["lineage_raw"],
        review_raw=stage1_audit_inputs["review_raw"],
        review_index_raw=stage1_audit_inputs["review_index_raw"],
        expected_lineage_sha256=stage1_audit_inputs["expected_lineage_sha256"])
    review_by_token = {row["review_token"]: row for row in review_rows}
    if set(review_by_token) != set(tokens):
        raise ValueError("bridge_review_token_set_mismatch")

    expected_rows = []
    sampler_rows = []
    receipt_sha = pins.get("expected_video_claim_review_receipt_sha256")
    _digest(receipt_sha, "bridge_review_receipt_digest_invalid")
    for source_row in source_rows:
        token = source_row["review_token"]
        source = source_row["source_video_sha256"]
        if source_row.get("video_support_decision") != "supports_claim":
            raise ValueError("bridge_human_support_review_missing")
        review = review_by_token[token]
        if review.get("support_decision") != "supports_claim":
            raise ValueError("bridge_human_support_review_missing")
        intervals = review.get("evidence_intervals_seconds")
        if interval_list_sha256(intervals) != source_row["reviewed_intervals_sha256"]:
            raise ValueError("bridge_review_interval_pin_mismatch")
        clock_input = clock_inputs_by_token[token]
        if (not isinstance(clock_input, Mapping)
                or set(clock_input) != {"calibration", "capture_raw"}
                or not isinstance(clock_input.get("calibration"), dict)
                or not isinstance(clock_input.get("capture_raw"), bytes)
                or not clock_input["capture_raw"]):
            raise ValueError("bridge_clock_calibration_input_missing")
        clock = adapt_calibration_to_clock_row(
            review_token=token, source_video_sha256=source,
            video_claim_review_receipt_sha256=receipt_sha,
            calibration=dict(clock_input["calibration"]),
            capture_raw=clock_input["capture_raw"])
        if clock.get("clock_correspondence_authenticated") is not False:
            raise ValueError("bridge_clock_correspondence_must_remain_unverified")
        processor = processor_pins_by_token[token]
        if (not isinstance(processor, Mapping)
                or set(processor) != {"processor_revision", "processor_fingerprint_sha256"}
                or not isinstance(processor.get("processor_revision"), str)
                or not processor["processor_revision"]):
            raise ValueError("bridge_processor_pin_invalid")
        proc_sha = _digest(processor.get("processor_fingerprint_sha256"),
                           "bridge_processor_pin_invalid")
        try:
            duration = Fraction(str(video_duration_seconds_by_token[token]))
            guard = Fraction(clock["guard_pts_seconds"])
            if duration <= 0 or guard < 0:
                raise ValueError
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError):
            raise ValueError("bridge_duration_or_clock_guard_invalid") from None
        sampler = sampler_by_source[source]
        sampler_sha = _sha(_canonical(sampler))
        expected = {
            "review_token": token,
            "source_video_sha256": source,
            "question_sha256": source_row["question_sha256"],
            "options_sha256": source_row["options_sha256"],
            "claim_sha256": source_row["claim_sha256"],
            "reviewed_intervals_sha256": source_row["reviewed_intervals_sha256"],
            "tokenizer_sha256": tokenizer_sha256,
            "processor_revision": processor["processor_revision"],
            "processor_fingerprint_sha256": proc_sha,
            "sampler_row_sha256": sampler_sha,
            "video_duration_seconds": str(duration),
            "interval_boundary_guard_seconds": str(guard),
        }
        expected_rows.append(expected)
        sampler_rows.append(sampler)
    return {
        "schema_version": 1,
        "purpose": "qta_actual_processor_candidate_inputs_private_non_authorizing",
        "status": "three_row_candidate_inputs_derived_unvalidated",
        "requested_records": ROW_COUNT,
        "effective_records": ROW_COUNT,
        "expected_rows": expected_rows,
        "sampler_rows": sampler_rows,
        "ordered_review_tokens": tokens,
        "preflight_joined_rows": ROW_COUNT,
        "human_review_artifacts_present": True,
        "human_identity_verified": False,
        "semantic_review_authenticated": False,
        "clock_calibration_artifacts_present": True,
        "clock_frame_correspondence_verified": False,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "faithfulness_scored": False,
        "execution_authorization": "not_granted",
        "private_inputs_must_remain_local": True,
    }
