"""Pure adapter from opt-in decoder timing metadata to planner timing rows.

This binds exact decoded PTS and sampled RGB pixel hashes. It does not
authenticate annotation-clock calibration, human review, or execution rights.
"""
from __future__ import annotations

import hashlib
import json
import re

from necessity_choice_margin_utils import uniform_sample_indices
from presentation_time_utils import presentation_times_from_pts

_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _sha(value) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _fail(code: str):
    raise ValueError(code)


def planner_timing_record(metadata: dict, *, video_id: str,
                          processor_pixel_sha256: list[str]) -> dict:
    """Validate decoder evidence and return one planner-compatible timing row.

    `processor_pixel_sha256` must be independently computed from the exact
    processor-bound frame buffers. The function only compares those supplied
    hashes with the decoder's RGB hashes; it cannot verify their provenance.
    """
    if not isinstance(metadata, dict) or not isinstance(video_id, str) or not video_id:
        _fail("decoder_timing_metadata_invalid")
    if metadata.get("purpose") != "claim_mask_pyav_decode_sample_provenance":
        _fail("decoder_timing_purpose_invalid")
    if metadata.get("annotation_clock_calibrated") is not False:
        _fail("decoder_annotation_clock_status_invalid")
    if metadata.get("execution_authorization") != "not_granted":
        _fail("decoder_execution_authorization_invalid")
    if (metadata.get("timestamp_origin") != "first_decoded_presentation_timestamp"
            or metadata.get("sampler") != "endpoint_inclusive_uniform_decoded_frame_index_v1"):
        _fail("decoder_sampling_contract_invalid")
    source_sha = metadata.get("source_sha256")
    if not _sha(source_sha):
        _fail("decoder_source_digest_invalid")
    timing = metadata.get("full_timing_map")
    if (not isinstance(timing, dict)
            or type(timing.get("schema_version")) is not int
            or timing.get("schema_version") != 1):
        _fail("full_timing_map_missing_or_invalid")
    pts = timing.get("pts")
    time_base = timing.get("time_base")
    if (not isinstance(pts, list) or not pts or any(type(value) is not int for value in pts)
            or not isinstance(time_base, list) or len(time_base) != 2
            or any(type(value) is not int for value in time_base)
            or time_base[0] <= 0 or time_base[1] <= 0):
        _fail("full_timing_map_values_invalid")
    decoded_time_base = metadata.get("decoded_time_base")
    if (type(metadata.get("decoded_frame_count")) is not int
            or metadata.get("decoded_frame_count") != len(pts)
            or not isinstance(decoded_time_base, list) or len(decoded_time_base) != 2
            or any(type(value) is not int for value in decoded_time_base)
            or decoded_time_base != time_base):
        _fail("full_timing_map_metadata_mismatch")
    pts_raw = json.dumps([[value, time_base[0], time_base[1]] for value in pts],
                         separators=(",", ":")).encode("ascii")
    pts_sha = hashlib.sha256(pts_raw).hexdigest()
    if (timing.get("decoded_pts_sequence_sha256") != pts_sha
            or metadata.get("decoded_pts_sequence_sha256") != pts_sha
            or timing.get("timestamp_origin") != "first_decoded_presentation_timestamp"):
        _fail("full_timing_map_digest_mismatch")
    try:
        relative_times = presentation_times_from_pts(pts, tuple(time_base))
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("full_timing_map_pts_invalid") from exc
    if (type(metadata.get("requested_frame_count")) is not int
            or metadata["requested_frame_count"] < 2):
        _fail("sampled_frame_count_invalid")
    count = metadata["requested_frame_count"]
    indices = uniform_sample_indices(len(pts), count)
    if len(indices) != count:
        _fail("sampled_frame_count_invalid")
    sampled_rows = metadata.get("sampled_frames")
    if not isinstance(sampled_rows, list) or len(sampled_rows) != count:
        _fail("sampled_frame_records_invalid")
    sampled_hashes = []
    for index, row in zip(indices, sampled_rows, strict=True):
        relative = row.get("relative_presentation_time") if isinstance(row, dict) else None
        if (not isinstance(row, dict)
                or type(row.get("decoded_frame_index")) is not int
                or row.get("decoded_frame_index") != index
                or type(row.get("pts")) is not int
                or row.get("pts") != pts[index]
                or not isinstance(relative, list) or len(relative) != 2
                or any(type(value) is not int for value in relative)
                or relative != [relative_times[index].numerator,
                                relative_times[index].denominator]):
            _fail("sampled_timing_binding_mismatch")
        digest = row.get("rgb_pixel_sha256")
        if not _sha(digest):
            _fail("sampled_pixel_digest_invalid")
        sampled_hashes.append(digest)
    if (not isinstance(processor_pixel_sha256, list)
            or processor_pixel_sha256 != sampled_hashes
            or not all(_sha(value) for value in processor_pixel_sha256)):
        _fail("processor_pixel_hash_binding_mismatch")
    return {
        "video_id": video_id,
        "source_media_sha256": source_sha,
        "pts": pts.copy(),
        "time_base": time_base.copy(),
        "num_frames": len(pts),
        "sampled_pixel_sha256": sampled_hashes,
        "processor_pixel_sha256": processor_pixel_sha256.copy(),
        "timing_evidence_scope": "decoder_pts_and_supplied_processor_hashes_match_unverified",
        "annotation_clock_provenance": "unauthenticated_not_calibrated_by_decoder",
        "review_provenance": "unauthenticated_not_provided_by_decoder",
        "execution_authorization": "not_granted",
    }
