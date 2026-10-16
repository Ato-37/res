"""Synthetic-only requirement mapper for draft NExT-GQA protocol v2.

This isolated checker binds the exact current draft bytes and validates a
small synthetic three-row structure against the draft's 32-frame/evidence/
control requirements. It is not a production preflight, does not authenticate
human review, custody, browser clock calibration, real media, or loaded code,
and cannot freeze the draft or authorize scoring. A success cannot satisfy any
v2 real-pilot gate. All hash contents and rows are caller supplied; the
synthetic fixture declaration is structurally checked but is not authentication.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Any

from necessity_choice_margin_utils import uniform_sample_indices
from presentation_time_utils import (
    assign_evidence_control_frames_from_times,
    presentation_times_from_pts,
)

FIXTURE_TYPE = "synthetic_nextgqa_v2_preflight_fixture_v1"
PROTOCOL_V2_SHA256 = "db71bc5017d0f85aeb58506480edf97f2cbf8ba5d9d56b5ea22fc2fb8f811973"
PROTOCOL_V2_TITLE = b"# NExT-GQA finite-cohort claim-aligned input-dependence protocol v2"
PROTOCOL_V2_DRAFT = b"Status: **design draft; not frozen, not scoreable, and not an execution"
PROTOCOL_V2_FRAME_RULE = b"Use 32 endpoint-inclusive uniformly sampled decoded-frame indices."
REQUIRED_HASH_NAMES = (
    "claim_input_packet", "target_manifest", "cohort_manifest", "frame_sampler",
    "mask_transform", "prompt_builder", "score_definition", "decoder",
    "clock_mapping_config",
    "model_config", "processor_config", "tokenizer_config",
)
_SHA = re.compile(r"^[a-f0-9]{64}$")
_FRAME_COUNT = 32
_DENOMINATOR = 3


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _blocked(failures: list[str], *, validated: int = 0) -> dict[str, Any]:
    return {
        "status": "blocked_synthetic_v2_preflight",
        "scope": "synthetic_structure_only",
        "synthetic_only": True,
        "real_media_eligible": False,
        "protocol_status": "design_draft_not_frozen",
        "failure_codes": sorted(set(failures)),
        "denominator": _DENOMINATOR,
        "structurally_checked_records": validated,
        "score_calls": 0,
        "execution_authorization": "not_granted",
        "synthetic_fixture_declaration_authentication": "caller_supplied_not_authenticated",
    }


def _protocol_valid(protocol_raw: object) -> bool:
    return (isinstance(protocol_raw, bytes)
            and _sha(protocol_raw) == PROTOCOL_V2_SHA256
            and PROTOCOL_V2_TITLE in protocol_raw
            and PROTOCOL_V2_DRAFT in protocol_raw
            and PROTOCOL_V2_FRAME_RULE in protocol_raw)


def _hash_declarations_valid(contents: object, declarations: object) -> bool:
    if (not isinstance(contents, dict) or not isinstance(declarations, dict)
            or set(contents) != set(REQUIRED_HASH_NAMES)
            or set(declarations) != set(REQUIRED_HASH_NAMES)):
        return False
    for name in REQUIRED_HASH_NAMES:
        raw = contents.get(name)
        digest = declarations.get(name)
        if not isinstance(raw, bytes) or not _valid_sha(digest) or _sha(raw) != digest:
            return False
    return True


def _validate_row(row: object, expected_video_id: str) -> list[str]:
    required = {
        "video_id", "source_media_bytes", "source_media_sha256", "pts", "time_base", "num_frames",
        "decoded_pts_sequence_sha256", "decoded_rgb_pixel_sha256",
        "decoded_rgb_sequence_sha256", "sample_indices", "sampled_pts",
        "sampled_pixel_sha256", "processor_pixel_sha256", "evidence_intervals",
        "clock_slope", "clock_intercept_seconds", "clock_guard_band_seconds",
    }
    if not isinstance(row, dict) or set(row) != required:
        return ["row_schema_invalid"]
    failures = []
    if row.get("video_id") != expected_video_id:
        failures.append("cohort_row_identity_mismatch")
    source_media = row.get("source_media_bytes")
    if (not isinstance(source_media, bytes) or not _valid_sha(row.get("source_media_sha256"))
            or _sha(source_media) != row.get("source_media_sha256")):
        failures.append("source_media_digest_invalid")
    pts = row.get("pts")
    time_base = row.get("time_base")
    if (not isinstance(pts, list) or not pts or any(type(value) is not int for value in pts)
            or not isinstance(time_base, list) or len(time_base) != 2
            or any(type(value) is not int for value in time_base)
            or time_base[0] <= 0 or time_base[1] <= 0):
        return failures + ["source_pts_map_invalid"]
    if type(row.get("num_frames")) is not int or row["num_frames"] != len(pts):
        return failures + ["source_pts_map_invalid"]
    pts_payload = json.dumps(
        [[value, time_base[0], time_base[1]] for value in pts],
        separators=(",", ":"),
    ).encode("ascii")
    if row.get("decoded_pts_sequence_sha256") != _sha(pts_payload):
        failures.append("source_pts_digest_mismatch")
    try:
        times = presentation_times_from_pts(pts, tuple(time_base))
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return failures + ["source_pts_map_invalid"]

    expected_indices = uniform_sample_indices(len(pts), _FRAME_COUNT)
    if (len(set(expected_indices)) != _FRAME_COUNT
            or not isinstance(row.get("sample_indices"), list)
            or any(type(value) is not int for value in row["sample_indices"])
            or row.get("sample_indices") != expected_indices):
        failures.append("sample_indices_not_exact_32_frame_grid")
    indices = row.get("sample_indices")
    if not isinstance(indices, list) or len(indices) != _FRAME_COUNT:
        return failures + ["sample_indices_not_exact_32_frame_grid"]
    sampled_pts = row.get("sampled_pts")
    if (sampled_pts != [pts[index] for index in expected_indices]
            or not isinstance(sampled_pts, list)
            or any(type(value) is not int for value in sampled_pts)):
        failures.append("sampled_pts_binding_mismatch")

    decoded_pixels = row.get("decoded_rgb_pixel_sha256")
    if (not isinstance(decoded_pixels, list) or len(decoded_pixels) != len(pts)
            or not all(_valid_sha(value) for value in decoded_pixels)):
        failures.append("decoded_pixel_map_invalid")
    else:
        pixel_payload = json.dumps(decoded_pixels, separators=(",", ":")).encode("ascii")
        if row.get("decoded_rgb_sequence_sha256") != _sha(pixel_payload):
            failures.append("decoded_pixel_map_digest_mismatch")
        expected_sample_pixels = [decoded_pixels[index] for index in expected_indices]
        sampled_pixels = row.get("sampled_pixel_sha256")
        if sampled_pixels != expected_sample_pixels:
            failures.append("sampled_pixel_binding_mismatch")
    sampled_pixels = row.get("sampled_pixel_sha256")
    processor_pixels = row.get("processor_pixel_sha256")
    if (not isinstance(sampled_pixels, list) or len(sampled_pixels) != _FRAME_COUNT
            or not all(_valid_sha(value) for value in sampled_pixels)
            or not isinstance(processor_pixels, list)
            or processor_pixels != sampled_pixels
            or not all(_valid_sha(value) for value in processor_pixels)):
        failures.append("processor_pixel_identity_mismatch")

    slope = _fraction_pair(row.get("clock_slope"), "clock_slope", positive=True)
    intercept = _fraction_pair(row.get("clock_intercept_seconds"), "clock_intercept")
    guard = _fraction_pair(row.get("clock_guard_band_seconds"), "clock_guard")
    if slope is None or intercept is None or guard is None or guard < 0:
        return failures + ["clock_mapping_values_invalid"]
    mapped_intervals = []
    spans = row.get("evidence_intervals")
    if not isinstance(spans, list) or not spans:
        return failures + ["evidence_control_assignment_invalid"]
    try:
        for span in spans:
            if (not isinstance(span, list) or len(span) != 2
                    or any(isinstance(value, bool)
                           or not isinstance(value, (int, float))
                           or isinstance(value, float) and not math.isfinite(value)
                           for value in span)):
                raise ValueError("invalid_span")
            start, end = (Fraction(str(value)) for value in span)
            if start < 0 or end <= start:
                raise ValueError("invalid_span")
            mapped_intervals.append([(start - intercept) / slope,
                                     (end - intercept) / slope])
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return failures + ["evidence_control_assignment_invalid"]
    if any(start < 0 or end <= start for start, end in mapped_intervals):
        return failures + ["clock_mapping_values_invalid"]
    for index in expected_indices:
        if any(abs(times[index] - boundary) <= guard
               for start, end in mapped_intervals for boundary in (start, end)):
            failures.append("sample_classification_uncertain_at_boundary")
            break
    try:
        assignment = assign_evidence_control_frames_from_times(
            len(pts), _FRAME_COUNT, times, mapped_intervals)
    except (TypeError, ValueError, IndexError, OverflowError, ZeroDivisionError):
        failures.append("evidence_control_assignment_invalid")
    else:
        if assignment.get("status") == "no_evidence":
            failures.append("zero_sampled_evidence_overlap")
        elif assignment.get("status") == "insufficient_control":
            failures.append("insufficient_equal_count_controls")
        elif assignment.get("status") != "ok":
            failures.append("evidence_control_assignment_invalid")
        elif len(assignment.get("evidence_indices", [])) < 1:
            failures.append("zero_sampled_evidence_overlap")
        elif len(assignment.get("control_indices", [])) != len(assignment["evidence_indices"]):
            failures.append("insufficient_equal_count_controls")
    return failures


def _fraction_pair(value, code, *, positive=False):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(part) is not int for part in value)
            or value[1] <= 0):
        return None
    try:
        result = Fraction(value[0], value[1])
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if positive and result <= 0:
        return None
    return result


def _clock_config_expected(rows, expected_ids):
    if (not isinstance(expected_ids, list)
            or any(not isinstance(value, str) or not value for value in expected_ids)
            or len(set(expected_ids)) != len(expected_ids)):
        return None
    material = []
    # Validate identities before using them as dictionary keys: malformed
    # synthetic rows may contain lists/dicts, which must fail closed rather
    # than leak a TypeError from this preflight boundary.
    by_id = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in by_id:
            return None
        by_id[video_id] = row
    if set(by_id) != set(expected_ids):
        return None
    for video_id in expected_ids:
        row = by_id[video_id]
        if (_fraction_pair(row.get("clock_slope"), "clock_slope", positive=True) is None
                or _fraction_pair(row.get("clock_intercept_seconds"), "clock_intercept") is None
                or _fraction_pair(row.get("clock_guard_band_seconds"), "clock_guard") is None):
            return None
        material.append({
            "video_id": video_id,
            "slope": row["clock_slope"],
            "intercept_seconds": row["clock_intercept_seconds"],
            "guard_band_seconds": row["clock_guard_band_seconds"],
        })
    return json.dumps({"rule": "mapped_interval_boundary_guard_inclusive_v1",
                       "rows": material}, sort_keys=True,
                      separators=(",", ":")).encode("ascii")


def preflight_synthetic_v2(*, fixture_type: object, protocol_raw: object,
                           expected_video_ids: object, rows: object,
                           code_config_contents: object,
                           declared_code_config_sha256: object,
                           synthetic_fixture_attestation: object) -> dict[str, Any]:
    """Check one synthetic cohort against draft-v2's structural requirements.

    All failures preserve denominator three and zero calls. A success is only
    a synthetic, structural, unvalidated contract check; it is not readiness.
    """
    failures = []
    if fixture_type != FIXTURE_TYPE:
        failures.append("synthetic_fixture_marker_missing")
    attestation_valid = (
        isinstance(synthetic_fixture_attestation, dict)
        and set(synthetic_fixture_attestation) == {"status", "generator", "no_real_media"}
        and synthetic_fixture_attestation.get("status") == "synthetic_only"
        and synthetic_fixture_attestation.get("generator") == "synthetic_fixture_encoder"
        and type(synthetic_fixture_attestation.get("no_real_media")) is bool
        and synthetic_fixture_attestation.get("no_real_media") is True
    )
    if not attestation_valid:
        failures.append("synthetic_fixture_attestation_invalid")
    if not _protocol_valid(protocol_raw):
        failures.append("protocol_v2_identity_mismatch")
    if (not isinstance(expected_video_ids, list) or len(expected_video_ids) != _DENOMINATOR
            or not all(isinstance(value, str) and value for value in expected_video_ids)
            or len(set(expected_video_ids)) != _DENOMINATOR):
        failures.append("frozen_three_record_cohort_invalid")
    if not isinstance(rows, list) or len(rows) != _DENOMINATOR:
        failures.append("cohort_row_count_mismatch")
    if not _hash_declarations_valid(code_config_contents, declared_code_config_sha256):
        failures.append("code_config_hash_set_invalid")
    elif (isinstance(expected_video_ids, list) and isinstance(rows, list)
          and code_config_contents["clock_mapping_config"]
          != _clock_config_expected(rows, expected_video_ids)):
        failures.append("clock_mapping_config_content_mismatch")
    if failures:
        return _blocked(failures)
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("video_id"), str):
            failures.append("row_schema_invalid")
            continue
        if row["video_id"] in by_id:
            failures.append("duplicate_cohort_video_id")
        by_id[row["video_id"]] = row
    if set(by_id) != set(expected_video_ids):
        failures.append("cohort_membership_mismatch")
    if failures:
        return _blocked(failures)
    validated = 0
    for video_id in expected_video_ids:
        row_failures = _validate_row(by_id[video_id], video_id)
        if row_failures:
            failures.extend(row_failures)
        else:
            validated += 1
    if failures:
        return _blocked(failures, validated=validated)
    return {
        "status": "synthetic_structural_preflight_passed_unvalidated",
        "scope": "synthetic_structure_only",
        "synthetic_only": True,
        "real_media_eligible": False,
        "protocol_status": "design_draft_not_frozen",
        "synthetic_fixture_declaration_matched": True,
        "synthetic_fixture_declaration_authentication": "caller_supplied_not_authenticated",
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "failure_codes": [],
        "denominator": _DENOMINATOR,
        "structurally_checked_records": validated,
        "score_calls": 0,
        "execution_authorization": "not_granted",
    }
