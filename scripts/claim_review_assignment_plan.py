"""CPU-only, fail-closed claim-review interval to PTS assignment planning.

This is an unvalidated structural planning adapter, not a review authenticator,
clock-calibration tool, or scorer. It accepts caller-supplied review/custody
audit reports and calibration artifacts; it checks hashes and structure but
does not independently authenticate those reports or calibration decisions.
It emits no IDs, text, timestamps, or rationales and never authorizes inference.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import re
from datetime import datetime, timezone

from build_nextgqa_claim_mapping_packet import FROZEN_POLICY_SHA256
from presentation_time_utils import (
    assign_evidence_control_frames_from_times,
    presentation_times_from_pts,
    uniform_sample_indices,
)

SHA256 = re.compile(r"^[a-f0-9]{64}$")
TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")
PLAYER_CLOCK = "html5_video_requestVideoFrameCallback_mediaTime_seconds"
MODEL_REVISION = re.compile(r"^.+@[a-f0-9]{40}$")


def _fraction(value, field):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(field + "_invalid")
    try:
        return Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        raise ValueError(field + "_invalid") from None


def _sha(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _text_hash_matches(text, expected_sha256):
    if not isinstance(text, str) or not _sha(expected_sha256):
        return False
    try:
        return hashlib.sha256(text.encode("utf-8")).hexdigest() == expected_sha256
    except UnicodeEncodeError:
        return False


def _clean_text(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        return False
    try:
        value.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def _utc(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)
    except ValueError:
        return False


def _utc_leq(left, right):
    if not _utc(left) or not _utc(right):
        return False
    a = datetime.fromisoformat(left.replace("Z", "+00:00"))
    b = datetime.fromisoformat(right.replace("Z", "+00:00"))
    return a <= b


def _blocked(code, denominator):
    return {
        "status": "blocked", "scope": "unvalidated_planning_only",
        "failure_codes": [code],
        "denominator": denominator, "validated_videos": 0,
        "assignments": [], "score_calls": 0,
        "execution_authorization": "not_granted",
    }


def plan_claim_review_assignments(*, expected_video_ids, worksheet_records,
                                  worksheet_raw, claim_input_manifest_raw,
                                  claim_input_manifest_sha256,
                                  review_records, review_payload_raw,
                                  review_payload_sha256, review_audit_report,
                                  review_index, review_index_raw,
                                  custody_failure_codes,
                                  calibration_records, timing_records,
                                  expected_num_frames):
    """Plan one frozen cohort atomically; no media/model is loaded here.

    ``calibration_records`` are caller-supplied, hash-bound empirical reports.
    Their observed held-out residual guard band is expressed in decoder-relative
    PTS seconds; it is not a future-error bound or uncertainty interval. Each
    sampled frame must remain on the same side of every mapped interval
    boundary under that bound. Any failure blocks all rows and retains the
    requested denominator.
    """
    n = len(expected_video_ids) if isinstance(expected_video_ids, list) else 0
    if (n != 3 or any(not isinstance(x, str) or not x for x in expected_video_ids)
            or len(set(expected_video_ids)) != n):
        return _blocked("frozen_cohort_invalid", n)
    if type(expected_num_frames) is not int or expected_num_frames < 2:
        return _blocked("frame_count_invalid", n)
    if (not isinstance(review_audit_report, dict)
            or set(review_audit_report) != {"status", "failure_codes", "reviewed_records",
                                           "execution_authorization", "semantic_review_authenticated",
                                           "worksheet_sha256", "lineage_manifest_sha256",
                                           "review_payload_sha256", "review_index_sha256"}
            or review_audit_report.get("status") != "unvalidated_handoff_structurally_valid"
            or review_audit_report.get("failure_codes") != []
            or review_audit_report.get("reviewed_records") != n
            or review_audit_report.get("execution_authorization") != "not_granted"
            or review_audit_report.get("semantic_review_authenticated") is not False):
        return _blocked("review_audit_not_passed", n)
    if (not _sha(review_audit_report.get("worksheet_sha256"))
            or not _sha(review_audit_report.get("lineage_manifest_sha256"))
            or not _sha(review_audit_report.get("review_payload_sha256"))
            or not _sha(review_audit_report.get("review_index_sha256"))):
        return _blocked("review_audit_binding_invalid", n)
    if (not isinstance(worksheet_raw, bytes)
            or not isinstance(review_payload_raw, bytes)
            or not isinstance(review_index_raw, bytes)
            or not _sha(claim_input_manifest_sha256)
            or not isinstance(claim_input_manifest_raw, bytes)
            or hashlib.sha256(worksheet_raw).hexdigest() != review_audit_report.get("worksheet_sha256")
            or hashlib.sha256(review_payload_raw).hexdigest() != review_audit_report.get("review_payload_sha256")
            or hashlib.sha256(review_index_raw).hexdigest() != review_audit_report.get("review_index_sha256")):
        return _blocked("audit_report_byte_binding_invalid", n)
    if hashlib.sha256(claim_input_manifest_raw).hexdigest() != claim_input_manifest_sha256:
        return _blocked("claim_input_manifest_hash_mismatch", n)
    try:
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate_json_key")
                result[key] = value
            return result
        worksheet_doc = json.loads(worksheet_raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                                   parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        claim_manifest = json.loads(claim_input_manifest_raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        index_doc = json.loads(review_index_raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        worksheet_canonical = json.dumps(worksheet_doc.get("records") if isinstance(worksheet_doc, dict) else None,
                                        sort_keys=True, separators=(",", ":"), allow_nan=False)
        rows_canonical = json.dumps(worksheet_records, sort_keys=True, separators=(",", ":"), allow_nan=False)
        index_canonical = json.dumps(index_doc, sort_keys=True, separators=(",", ":"), allow_nan=False)
        provided_index_canonical = json.dumps(review_index, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError, OverflowError, RecursionError):
        return _blocked("worksheet_or_claim_manifest_invalid", n)
    if worksheet_canonical != rows_canonical or index_canonical != provided_index_canonical:
        return _blocked("worksheet_records_byte_binding_invalid", n)
    if (not isinstance(worksheet_doc, dict)
            or worksheet_doc.get("occurrence_policy") != "all_relevant_occurrences"
            or worksheet_doc.get("claim_review_policy_id") != "nextgqa_claim_review_v6"
            or worksheet_doc.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
            or worksheet_doc.get("review_procedure_status") != "frozen_pre_review"):
        return _blocked("worksheet_policy_binding_invalid", n)
    manifest_keys = {"schema_version", "eligible_dataset_id", "eligible_manifest_sha256",
                     "claim_policy_sha256", "occurrence_policy", "claims"}
    if (not isinstance(claim_manifest, dict) or set(claim_manifest) != manifest_keys
            or type(claim_manifest.get("schema_version")) is not int
            or claim_manifest.get("schema_version") != 1
            or not _sha(claim_manifest.get("eligible_manifest_sha256"))
            or not _sha(claim_manifest.get("claim_policy_sha256"))
            or claim_manifest.get("occurrence_policy") != "all_relevant_occurrences"
            or not isinstance(claim_manifest.get("eligible_dataset_id"), str)
            or not claim_manifest["eligible_dataset_id"].strip()
            or not isinstance(claim_manifest.get("claims"), list)
            or len(claim_manifest["claims"]) != n):
        return _blocked("claim_input_manifest_schema_invalid", n)
    manifest_by_video = {}
    for row in claim_manifest["claims"]:
        keys = {"video_id", "claim_id", "claim_text", "claim_text_sha256",
                "model_revision", "generation_run_sha256"}
        if (not isinstance(row, dict) or set(row) != keys
                or not isinstance(row.get("video_id"), str)
                or not _clean_text(row.get("claim_id"))
                or not isinstance(row.get("claim_text"), str) or not row["claim_text"].strip()
                or not _sha(row.get("claim_text_sha256"))
                or not _text_hash_matches(row["claim_text"], row["claim_text_sha256"])
                or not _sha(row.get("generation_run_sha256"))
                or not _clean_text(row.get("model_revision"))
                or MODEL_REVISION.fullmatch(row["model_revision"]) is None):
            return _blocked("claim_input_row_invalid", n)
        if row["video_id"] in manifest_by_video:
            return _blocked("claim_input_duplicate_video", n)
        manifest_by_video[row["video_id"]] = row
    if set(manifest_by_video) != set(expected_video_ids):
        return _blocked("claim_input_cohort_mismatch", n)
    worksheet_by_video = {row.get("video_id"): row for row in worksheet_records
                          if isinstance(row, dict)}
    if set(worksheet_by_video) != set(expected_video_ids):
        return _blocked("worksheet_cohort_mismatch", n)
    for video_id in expected_video_ids:
        claim_row = manifest_by_video[video_id]
        worksheet_row = worksheet_by_video[video_id]
        if (worksheet_row.get("claim_id") != claim_row["claim_id"]
                or worksheet_row.get("claim_text") != claim_row["claim_text"]
                or not _text_hash_matches(worksheet_row.get("claim_text"),
                                          worksheet_row.get("claim_text_sha256"))
                or worksheet_row.get("claim_text_sha256") != claim_row["claim_text_sha256"]
                or not _sha(worksheet_row.get("video_sha256"))
                or not _sha(worksheet_row["claim_text_sha256"])):
            return _blocked("worksheet_claim_binding_mismatch", n)
    if (not isinstance(review_payload_raw, bytes)
            or hashlib.sha256(review_payload_raw).hexdigest() != review_payload_sha256
            or not isinstance(review_index, dict)
            or set(review_index) != {"schema_version", "purpose", "review_payload_sha256",
                                    "claim_review_policy_sha256", "created_at_utc",
                                    "review_status", "execution_authorization"}
            or type(review_index.get("schema_version")) is not int
            or review_index.get("schema_version") != 1
            or review_index.get("purpose") != "anonymous_unvalidated_human_review_handoff"
            or review_index.get("execution_authorization") != "not_granted"
            or review_index.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
            or review_index.get("review_status") != "unvalidated_requires_custodian_audit"
            or review_index.get("review_payload_sha256") != review_payload_sha256
            or not _sha(review_payload_sha256)):
        return _blocked("review_export_binding_invalid", n)
    try:
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate_json_key")
                result[key] = value
            return result
        parsed_review = json.loads(review_payload_raw.decode("utf-8"),
                                   object_pairs_hook=unique_pairs,
                                   parse_constant=lambda _: (_ for _ in ()).throw(
                                       ValueError("nonfinite_json")))
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        return _blocked("review_payload_invalid", n)
    review_top_keys = {
        "schema_version", "purpose", "status", "review_completed", "execution_authorization",
        "claim_review_policy_id", "claim_review_policy_sha256", "human_reviewer_role",
        "human_reviewer_private_id", "review_environment", "independence_attestation",
        "export_disclosure_acknowledged", "records",
    }
    try:
        parsed_records_canonical = json.dumps(
            parsed_review.get("records") if isinstance(parsed_review, dict) else None,
            sort_keys=True, separators=(",", ":"), allow_nan=False)
        provided_records_canonical = json.dumps(
            review_records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return _blocked("review_payload_rows_invalid", n)
    if (not isinstance(parsed_review, dict) or set(parsed_review) != review_top_keys
            or type(parsed_review.get("schema_version")) is not int
            or parsed_review.get("schema_version") != 3
            or parsed_review.get("purpose") != "anonymous_local_claim_video_human_review"
            or parsed_review.get("status") != "human_review_recorded_unvalidated"
            or parsed_review.get("review_completed") is not True
            or parsed_review.get("execution_authorization") != "not_granted"
            or parsed_review.get("claim_review_policy_id") != "nextgqa_claim_review_v6"
            or parsed_review.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
            or parsed_review.get("human_reviewer_role") != "independent_human"
            or not isinstance(parsed_review.get("human_reviewer_private_id"), str)
            or not parsed_review["human_reviewer_private_id"].strip()
            or parsed_review.get("independence_attestation") != "reviewer_asserts_independence_from_generation_scoring_and_custody_and_did_not_view_custody_crosswalk"
            or parsed_review.get("export_disclosure_acknowledged") is not True
            or not isinstance(parsed_review.get("review_environment"), dict)
            or set(parsed_review["review_environment"]) != {"browser_user_agent", "offline_static_page", "network_fetch_available"}
            or not isinstance(parsed_review["review_environment"].get("browser_user_agent"), str)
            or not parsed_review["review_environment"]["browser_user_agent"]
            or parsed_review["review_environment"].get("offline_static_page") is not True
            or parsed_review["review_environment"].get("network_fetch_available") is not False
            or parsed_records_canonical != provided_records_canonical):
        return _blocked("review_payload_rows_mismatch", n)
    if custody_failure_codes != []:
        return _blocked("custody_audit_failed", n)
    if (not isinstance(worksheet_records, list) or len(worksheet_records) != n
            or not isinstance(review_records, list) or len(review_records) != n
            or not isinstance(calibration_records, list) or len(calibration_records) != n
            or not isinstance(timing_records, list) or len(timing_records) != n):
        return _blocked("cohort_component_count_mismatch", n)

    def keyed(rows, key):
        result = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get(key), str):
                raise ValueError("cohort_row_invalid")
            if row[key] in result:
                raise ValueError("cohort_duplicate_key")
            result[row[key]] = row
        return result

    try:
        worksheet = keyed(worksheet_records, "video_id")
        reviews = keyed(review_records, "review_token")
        timings = keyed(timing_records, "video_id")
    except ValueError as exc:
        return _blocked(str(exc), n)
    if set(worksheet) != set(expected_video_ids) or set(timings) != set(expected_video_ids):
        return _blocked("cohort_membership_mismatch", n)
    token_to_video = {}
    for video_id, row in worksheet.items():
        token = row.get("review_token")
        if not isinstance(token, str) or TOKEN.fullmatch(token) is None or token in token_to_video:
            return _blocked("worksheet_token_mapping_invalid", n)
        token_to_video[token] = video_id
    if set(reviews) != set(token_to_video):
        return _blocked("review_token_coverage_mismatch", n)

    # Bound clock artifacts are keyed by the opaque review token; synthetic
    # planner fixtures and older callers may instead supply an explicit
    # video_id. Join through the worksheet token map without injecting an ID
    # into the hash-bound calibration artifact.
    if any(not isinstance(row, dict) for row in calibration_records):
        return _blocked("calibration_identity_missing", n)
    identity_shapes = {("video_id" in row, "review_token" in row)
                       for row in calibration_records}
    if len(identity_shapes) > 1:
        return _blocked("calibration_identity_mixed", n)
    if not identity_shapes:
        return _blocked("calibration_identity_missing", n)
    has_video_id, has_review_token = next(iter(identity_shapes), (False, False))
    if ((has_video_id and any(not isinstance(row.get("video_id"), str)
                              for row in calibration_records))
            or (has_review_token and any(not isinstance(row.get("review_token"), str)
                                         for row in calibration_records))):
        return _blocked("calibration_identity_missing", n)
    if has_video_id:
        try:
            calibrations = keyed(calibration_records, "video_id")
        except ValueError as exc:
            return _blocked(str(exc), n)
        if has_review_token and any(
                row["video_id"] != token_to_video.get(row["review_token"])
                for row in calibration_records):
            return _blocked("calibration_token_video_mismatch", n)
    elif has_review_token:
        try:
            by_token = keyed(calibration_records, "review_token")
        except ValueError as exc:
            return _blocked(str(exc), n)
        if set(by_token) != set(token_to_video):
            return _blocked("calibration_token_coverage_mismatch", n)
        calibrations = {token_to_video[token]: row for token, row in by_token.items()}
    else:
        return _blocked("calibration_identity_missing", n)
    if set(calibrations) != set(expected_video_ids):
        return _blocked("calibration_cohort_mismatch", n)
    try:
        for calibration in calibration_records:
            raw = calibration.get("artifact_raw")
            expected_hash = calibration.get("artifact_sha256")
            if not isinstance(raw, bytes) or not _sha(expected_hash):
                raise ValueError("calibration_artifact_bytes_missing")
            if hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ValueError("calibration_artifact_hash_mismatch")
            def unique_pairs(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate_json_key")
                    result[key] = value
                return result
            parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                                parse_constant=lambda _: (_ for _ in ()).throw(
                                    ValueError("nonfinite_json")))
            expected = {key: value for key, value in calibration.items()
                        if key not in {"artifact_raw", "artifact_sha256"}}
            if json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False) != json.dumps(
                    expected, sort_keys=True, separators=(",", ":"), allow_nan=False):
                raise ValueError("calibration_artifact_content_mismatch")
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        code = str(exc) if re.fullmatch(r"[a-z0-9_]+", str(exc)) else "calibration_artifact_invalid"
        return _blocked(code, n)
    planned = []
    try:
        for video_id in expected_video_ids:
            source = worksheet[video_id]
            review_row = reviews[source["review_token"]]
            item = review_row.get("review") if isinstance(review_row.get("review"), dict) else None
            item_keys = {"full_source_video_reviewed", "question_and_claim_only_context_shown",
                         "gold_and_seed_hidden_until_mapping_frozen", "outcome_scores_not_accessed",
                         "media_bytes_verified", "occurrence_policy_applied", "support_decision",
                         "evidence_intervals_seconds", "interval_rationales", "reviewer_role",
                         "first_played_at_utc", "reviewed_at_utc", "review_status", "display_player",
                         "claim_review_policy_sha256", "media_duration_seconds"}
            required_true = ("full_source_video_reviewed", "question_and_claim_only_context_shown",
                             "gold_and_seed_hidden_until_mapping_frozen", "outcome_scores_not_accessed",
                             "media_bytes_verified")
            if (item is None or set(item) != item_keys or item.get("review_status") != "completed"
                    or item.get("reviewer_role") != "independent_human"
                    or item.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
                    or item.get("occurrence_policy_applied") != "all_relevant_occurrences"
                    or any(item.get(key) is not True for key in required_true)
                    or not _utc_leq(item.get("first_played_at_utc"), item.get("reviewed_at_utc"))
                    or not _utc_leq(item.get("reviewed_at_utc"), review_index.get("created_at_utc"))):
                raise ValueError("review_declaration_gate_failed")
            calibration = calibrations[video_id]
            timing = timings[video_id]
            if item.get("support_decision") != "supports_claim":
                raise ValueError("claim_support_gate_failed")
            intervals = item.get("evidence_intervals_seconds")
            rationales = item.get("interval_rationales")
            if (not isinstance(intervals, list) or not intervals
                    or not isinstance(rationales, list) or len(rationales) != len(intervals)
                    or any(not isinstance(text, str) or not text.strip() for text in rationales)
                    or not isinstance(item.get("display_player"), str)
                    or not item["display_player"].strip()):
                raise ValueError("claim_intervals_missing")
            duration = item.get("media_duration_seconds")
            if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise ValueError("review_duration_invalid")
            if (calibration.get("clock_scope") != PLAYER_CLOCK
                    or calibration.get("empirical_residual_within_tolerance") is not True
                    or calibration.get("clock_validation_attestation")
                    != "independent_review_claimed_unverified"
                    or not _sha(calibration.get("capture_evidence_sha256"))
                    or not _sha(calibration.get("frame_correspondence_evidence_sha256"))
                    or calibration.get("validation_scope") != "empirical_heldout_residual_only"
                    or calibration.get("guard_band_source") != "max_observed_heldout_residual_pts_seconds"
                    or calibration.get("uncertainty_status") != "not_evaluated"
                    or type(calibration.get("fit_anchor_count")) is not int
                    or calibration["fit_anchor_count"] < 2
                    or type(calibration.get("heldout_anchor_count")) is not int
                    or calibration["heldout_anchor_count"] < 1
                    or not _sha(calibration.get("source_media_sha256"))
                    or calibration["source_media_sha256"] != source.get("video_sha256")
                    or not _sha(calibration.get("artifact_sha256"))):
                raise ValueError("clock_calibration_gate_failed")
            slope = _fraction(calibration.get("slope"), "slope")
            intercept = _fraction(calibration.get("intercept"), "intercept")
            empirical_guard = _fraction(calibration.get("empirical_guard_band_pts_seconds"), "guard_band")
            tolerance = _fraction(calibration.get("frozen_tolerance_seconds"), "tolerance")
            if slope <= 0 or empirical_guard < 0 or tolerance <= 0 or empirical_guard > tolerance:
                raise ValueError("clock_calibration_values_invalid")
            if (not _sha(timing.get("source_media_sha256"))
                    or timing["source_media_sha256"] != source.get("video_sha256")):
                raise ValueError("timing_media_binding_invalid")
            times = presentation_times_from_pts(timing.get("pts"), tuple(timing.get("time_base")))
            if type(timing.get("num_frames")) is not int or len(times) != timing["num_frames"]:
                raise ValueError("timing_pts_map_invalid")
            sample_indices = uniform_sample_indices(len(times), expected_num_frames)
            if len(sample_indices) != expected_num_frames:
                raise ValueError("sample_frame_count_invalid")
            sample_hashes = timing.get("sampled_pixel_sha256")
            processor_hashes = timing.get("processor_pixel_sha256")
            if (not isinstance(sample_hashes, list) or not isinstance(processor_hashes, list)
                    or len(sample_hashes) != expected_num_frames
                    or sample_hashes != processor_hashes
                    or not all(_sha(x) for x in sample_hashes)):
                raise ValueError("sampled_pixel_identity_invalid")
            mapped = []
            seen_intervals = set()
            for span in intervals:
                if (not isinstance(span, list) or len(span) != 2
                        or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not math.isfinite(x) for x in span)):
                    raise ValueError("review_interval_invalid")
                start, end = map(lambda x: Fraction(str(x)), span)
                if (start < 0 or end <= start or end > Fraction(str(duration))
                        or (start * 1000).denominator != 1 or (end * 1000).denominator != 1):
                    raise ValueError("review_interval_invalid")
                if (start, end) in seen_intervals:
                    raise ValueError("review_interval_invalid")
                seen_intervals.add((start, end))
                pts_start = (start - intercept) / slope
                pts_end = (end - intercept) / slope
                if pts_start < 0 or pts_end <= pts_start:
                    raise ValueError("mapped_interval_invalid")
                # This is only a guard using observed held-out residuals. It is
                # not a statistical or deterministic uncertainty guarantee.
                bound = empirical_guard
                if any(abs(times[i] - boundary) <= bound
                       for i in sample_indices for boundary in (pts_start, pts_end)):
                    raise ValueError("sample_classification_uncertain_at_boundary")
                mapped.append([pts_start, pts_end])
            assignment = assign_evidence_control_frames_from_times(
                len(times), expected_num_frames, times, mapped)
            if assignment["status"] == "no_evidence":
                raise ValueError("zero_sampled_evidence_overlap")
            if assignment["status"] != "ok":
                raise ValueError("evidence_control_assignment_failed")
            planned.append({
                "cohort_ordinal": len(planned) + 1,
                "sample_indices": assignment["sampled_indices"],
                "evidence_indices": assignment["evidence_indices"],
                "control_indices": assignment["control_indices"],
                "matched_rank_distance": assignment["matched_rank_distance"],
                "calibration_sha256": calibration["artifact_sha256"],
            })
    except (ValueError, TypeError, IndexError, OverflowError, ZeroDivisionError) as exc:
        # Keep the frozen denominator and deliberately omit the failing row ID.
        code = str(exc) if re.fullmatch(r"[a-z0-9_]+", str(exc)) else "assignment_invalid"
        return _blocked(code, n)

    return {
        "status": "assignment_plan_structurally_passed_unvalidated",
        "scope": "unvalidated_planning_only",
        "failure_codes": [], "denominator": n, "validated_videos": n,
        "assignments": planned, "review_payload_sha256": review_payload_sha256,
        "worksheet_sha256": hashlib.sha256(worksheet_raw).hexdigest(),
        "claim_input_manifest_sha256": claim_input_manifest_sha256,
        "score_calls": 0, "execution_authorization": "not_granted",
    }
