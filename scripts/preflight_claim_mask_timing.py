"""Synthetic-only contract check for an atomic claim-mask timing preflight.

This module is not wired into a scorer and cannot authorize real-media work.
It checks that an entire supplied cohort has coherent PTS, sampled-frame
identity, processor FPS/index metadata, and identical full/selected/control
timing declarations before a hypothetical scorer is allowed to start.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math

from presentation_time_utils import (
    assign_evidence_control_frames_from_times,
    presentation_times_from_pts,
    uniform_sampling_rate_from_times,
)

HEX = frozenset("0123456789abcdef")
MAX_PREFLIGHT_FRAME_COUNT = 4096


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in HEX for char in value))


def _fps_matches(actual, expected):
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    try:
        actual = float(actual)
        expected = float(expected)
    except (OverflowError, TypeError, ValueError):
        return False
    if (not math.isfinite(actual) or not math.isfinite(expected)
            or actual <= 0 or expected <= 0):
        return False
    # Metadata is a float at the Qwen boundary. Bound conversion error only;
    # the underlying rate is derived exactly from rational PTS.
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=0.0)


def validate_video_timing(record, tolerance_seconds, expected_num_frames):
    """Return stable failure codes for one synthetic/video timing record."""
    failures = []
    required = {
        "video_id", "pts", "time_base", "num_frames", "evidence_intervals",
        "sample_indices", "sampled_pixel_sha256",
        "processor_pixel_sha256", "processor_fps", "processor_frame_indices",
        "condition_timing_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        return ["record_schema_invalid"]
    if not isinstance(record["video_id"], str) or not record["video_id"]:
        failures.append("video_id_invalid")
    if (type(record["num_frames"]) is not int
            or record["num_frames"] != expected_num_frames):
        return failures + ["cohort_sampling_count_mismatch"]
    try:
        tb = record["time_base"]
        times = presentation_times_from_pts(record["pts"], tuple(tb))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return failures + ["pts_map_invalid"]

    indices = record["sample_indices"]
    if (not isinstance(indices, list) or len(indices) < 2
            or any(type(i) is not int or i < 0 or i >= len(times) for i in indices)
            or len(set(indices)) != len(indices)
            or any(right <= left for left, right in zip(indices, indices[1:]))):
        return failures + ["sample_indices_invalid"]
    try:
        assignment = assign_evidence_control_frames_from_times(
            len(times), record["num_frames"], times, record["evidence_intervals"])
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        failures.append("assignment_invalid")
    else:
        if assignment["status"] != "ok":
            failures.append("evidence_or_control_assignment_failed")
        if assignment["sampled_indices"] != indices:
            failures.append("sample_indices_do_not_match_frozen_sampler")
    if (not isinstance(record["sampled_pixel_sha256"], list)
            or len(record["sampled_pixel_sha256"]) != len(indices)):
        failures.append("sampled_pixel_count_mismatch")
    sampled_hashes = record["sampled_pixel_sha256"]
    processor_hashes = record["processor_pixel_sha256"]
    if (not isinstance(sampled_hashes, list) or not all(_sha(item) for item in sampled_hashes)
            or not isinstance(processor_hashes, list) or not all(_sha(item) for item in processor_hashes)):
        failures.append("pixel_identity_invalid")
    elif sampled_hashes != processor_hashes:
        failures.append("pixel_identity_mismatch")

    if not isinstance(tolerance_seconds, Fraction) or tolerance_seconds < 0:
        failures.append("tolerance_invalid")
        return failures
    try:
        sampled_times = [times[index] for index in indices]
        expected_fps, _deviation = uniform_sampling_rate_from_times(
            sampled_times, tolerance_seconds)
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        failures.append("sampled_pts_not_uniform")
        return failures
    if not _fps_matches(record["processor_fps"], expected_fps):
        failures.append("processor_fps_mismatch")
    frame_indices = record["processor_frame_indices"]
    if (not isinstance(frame_indices, list)
            or any(type(index) is not int for index in frame_indices)
            or frame_indices != list(range(len(indices)))):
        failures.append("processor_frame_order_mismatch")
    hashes = record["condition_timing_sha256"]
    if (not isinstance(hashes, list) or len(hashes) != 3
            or not all(_sha(item) for item in hashes)):
        failures.append("condition_timing_binding_invalid")
    elif len(set(hashes)) != 1:
        failures.append("condition_timing_binding_mismatch")
    return failures


def preflight_cohort(records, *, expected_video_ids, expected_num_frames,
                     tolerance_seconds):
    """Validate the full frozen denominator atomically; never invokes a scorer."""
    blockers = []
    if (not isinstance(expected_video_ids, list)
            or len(expected_video_ids) < 2
            or any(not isinstance(item, str) or not item for item in expected_video_ids)
            or len(set(expected_video_ids)) != len(expected_video_ids)):
        return {"status": "blocked", "failure_codes": ["expected_cohort_invalid"],
                "denominator": 0, "validated_videos": 0, "score_calls": 0,
                "execution_authorization": "not_granted"}
    if (type(expected_num_frames) is not int or expected_num_frames < 2
            or expected_num_frames > MAX_PREFLIGHT_FRAME_COUNT):
        return {"status": "blocked", "failure_codes": ["expected_frame_count_invalid"],
                "denominator": len(expected_video_ids), "validated_videos": 0,
                "score_calls": 0, "execution_authorization": "not_granted"}
    if not isinstance(records, list):
        records = []
        blockers.append("records_invalid")
    ids = [r.get("video_id") for r in records if isinstance(r, dict)]
    ids_are_strings = all(isinstance(item, str) for item in ids)
    if (len(records) != len(expected_video_ids) or not ids_are_strings
            or set(ids) != set(expected_video_ids) or len(ids) != len(set(ids))):
        blockers.append("cohort_membership_mismatch")
    per_video = {}
    validated = 0
    for record in records:
        key = record.get("video_id") if isinstance(record, dict) else "<invalid>"
        codes = validate_video_timing(record, tolerance_seconds, expected_num_frames)
        if codes:
            per_video[str(key)] = codes
        else:
            validated += 1
    if per_video:
        blockers.append("video_timing_invalid")
    return {
        "status": "blocked" if blockers else "synthetic_timing_contract_passed",
        "failure_codes": sorted(set(blockers)),
        "video_failures": per_video,
        "denominator": len(expected_video_ids),
        "validated_videos": validated,
        "score_calls": 0,
        "execution_authorization": "not_granted",
    }


def sha256_frame_buffer(buffer):
    """Hash a synthetic RGB frame buffer; production must hash actual input bytes."""
    if not isinstance(buffer, bytes):
        raise TypeError("frame buffer must be bytes")
    return hashlib.sha256(buffer).hexdigest()
