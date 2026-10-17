"""Adapt one HTML5 calibration result to the non-authorizing QTA clock row.

This CPU-only adapter checks byte/self-consistency and caller-supplied token and
video bindings. It does not authenticate the upstream bundle, review receipt,
browser telemetry, reviewer, or human frame correspondence and never grants
authorization.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")
_CAPTURE_KEYS = {
    "schema_version", "purpose", "player_clock", "capture_policy_id",
    "capture_thresholds", "browser_user_agent", "current_src_matches_staged_media",
    "source_media_sha256", "playback_rate", "muted", "text_tracks_disabled",
    "fit_anchors", "heldout_anchors",
}
_CAPTURE_THRESHOLDS = {
    "max_callback_age_ms": 250.0,
    "max_request_delay_ms": 250.0,
    "max_click_player_time_delta_seconds": 0.1,
    "max_callback_player_time_delta_seconds": 0.1,
    "callback_wait_timeout_ms": 1000.0,
}
_FIT_ANCHOR_IDS = {"fit_begin", "fit_midpoint", "fit_end"}
_HELDOUT_ANCHOR_IDS = {"heldout_quarter", "heldout_three_quarter"}
_BOUND_ARTIFACT_KEYS = {
    "clock_scope", "capture_policy_id", "capture_thresholds",
    "empirical_residual_within_tolerance", "validation_scope",
    "guard_band_source", "uncertainty_status", "fit_anchor_count",
    "heldout_anchor_count", "source_media_sha256", "slope", "intercept",
    "empirical_guard_band_pts_seconds", "frozen_tolerance_seconds",
    "capture_sha256", "execution_authorization", "purpose",
    "provenance_status", "upstream_hash_verification",
    "observations_sha256", "observation_index_sha256", "anchor_plan_sha256",
    "review_token", "anchor_plan_provenance_sha256",
    "limitations",
}


def _fraction(value: Any, field: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Fraction)):
        raise ValueError(f"{field}_invalid")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field}_invalid")
    try:
        return value if isinstance(value, Fraction) else Fraction(str(value))
    except (OverflowError, TypeError, ValueError, ZeroDivisionError):
        raise ValueError(f"{field}_invalid") from None


def _canonical_rational(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field}_invalid")
    parsed = _fraction(value, field)
    if str(parsed) != value:
        raise ValueError(f"{field}_noncanonical")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("capture_duplicate_json_key")
        result[key] = value
    return result


def adapt_calibration_to_clock_row(*, review_token: str,
                                   source_video_sha256: str,
                                   video_claim_review_receipt_sha256: str,
                                   calibration: dict,
                                   capture_raw: bytes) -> dict:
    """Validate one builder result and create the exact prescore clock-row schema.

    A result whose empirical held-out residual exceeds its recorded frozen
    tolerance is rejected. The returned calibration hash is over the compact
    six-field payload expected by ``qta_prescore_mapping_binding``, not the
    builder artifact hash.
    """
    if not isinstance(review_token, str) or not _TOKEN.fullmatch(review_token):
        raise ValueError("review_token_invalid")
    for field, value in (("source_video_sha256", source_video_sha256),
                         ("video_claim_review_receipt_sha256",
                          video_claim_review_receipt_sha256)):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{field}_invalid")
    if not isinstance(calibration, dict):
        raise ValueError("calibration_invalid")
    if set(calibration) != _BOUND_ARTIFACT_KEYS | {"artifact_sha256", "artifact_raw"}:
        raise ValueError("calibration_bound_artifact_keys_invalid")
    if not isinstance(capture_raw, bytes) or not capture_raw:
        raise ValueError("capture_raw_invalid")
    if calibration.get("review_token") != review_token:
        raise ValueError("calibration_review_token_mismatch")
    if (calibration.get("purpose") != "unvalidated_html5_clock_calibration_with_lineage"
            or calibration.get("provenance_status")
            != "capture_and_media_byte_bound_upstream_digests_unverified"
            or calibration.get("upstream_hash_verification")
            != "digest_strings_carried_from_prepared_bundle_not_reauthenticated"):
        raise ValueError("calibration_bound_lineage_schema_invalid")
    if (calibration.get("source_media_sha256") != source_video_sha256
            or ("source_video_sha256" in calibration
                and calibration.get("source_media_sha256")
                != calibration.get("source_video_sha256"))):
        raise ValueError("calibration_source_video_mismatch")
    if calibration.get("execution_authorization") != "not_granted":
        raise ValueError("calibration_authorization_field_invalid")

    # Verify the builder's exact artifact envelope, preventing accidental use
    # of a mutated in-memory result. This is integrity only, not authentication.
    artifact_raw = calibration.get("artifact_raw")
    artifact_sha = calibration.get("artifact_sha256")
    if (not isinstance(artifact_raw, bytes) or not isinstance(artifact_sha, str)
            or not _SHA256.fullmatch(artifact_sha)
            or hashlib.sha256(artifact_raw).hexdigest() != artifact_sha):
        raise ValueError("calibration_artifact_digest_invalid")
    try:
        artifact = json.loads(artifact_raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("calibration_artifact_json_invalid") from None
    expected_artifact = {key: value for key, value in calibration.items()
                         if key not in {"artifact_raw", "artifact_sha256"}}
    if (not isinstance(artifact, dict) or set(artifact) != _BOUND_ARTIFACT_KEYS
            or artifact != expected_artifact):
        raise ValueError("calibration_artifact_content_mismatch")
    if (calibration.get("clock_scope")
            != "html5_video_requestVideoFrameCallback_mediaTime_seconds"
            or calibration.get("capture_policy_id") != "rvfc_click_bound_v1"
            or calibration.get("capture_thresholds") != _CAPTURE_THRESHOLDS
            or calibration.get("validation_scope") != "empirical_heldout_residual_only"
            or calibration.get("guard_band_source")
            != "max_observed_heldout_residual_pts_seconds"
            or calibration.get("uncertainty_status") != "not_evaluated"
            or calibration.get("limitations") != [
                "prepared bundle digests are caller-supplied values and are not reauthenticated from original observation/index/plan bytes",
                "a fabricated bundle with matching capture and media bytes can produce this record",
                "browser telemetry, reviewer identity, and decoder-frame correspondence are not authenticated",
                "this record does not authorize scoring or inference",
            ]):
        raise ValueError("calibration_builder_fields_invalid")

    if hashlib.sha256(capture_raw).hexdigest() != calibration.get("capture_sha256"):
        raise ValueError("calibration_capture_hash_mismatch")
    try:
        capture = json.loads(capture_raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("calibration_capture_json_invalid") from None
    if (not isinstance(capture, dict) or set(capture) != _CAPTURE_KEYS
            or type(capture.get("schema_version")) is not int
            or capture.get("schema_version") != 3
            or capture.get("purpose") != "local_html5_player_clock_anchor_capture"
            or capture.get("player_clock")
            != "html5_video_requestVideoFrameCallback_mediaTime_seconds"
            or capture.get("capture_policy_id") != "rvfc_click_bound_v1"
            or capture.get("capture_thresholds") != _CAPTURE_THRESHOLDS
            or not isinstance(capture.get("browser_user_agent"), str)
            or not capture["browser_user_agent"].strip()
            or capture.get("current_src_matches_staged_media") is not True
            or capture.get("playback_rate") != "1"
            or capture.get("muted") is not True
            or capture.get("text_tracks_disabled") is not True
            or capture.get("source_media_sha256") != source_video_sha256):
        raise ValueError("calibration_capture_schema_invalid")

    tolerance = _fraction(calibration.get("frozen_tolerance_seconds"),
                          "frozen_tolerance_seconds")
    guard = _fraction(calibration.get("empirical_guard_band_pts_seconds"),
                      "empirical_guard_band_pts_seconds")
    if tolerance <= 0 or guard < 0:
        raise ValueError("calibration_residual_values_invalid")
    if calibration.get("empirical_residual_within_tolerance") is not True or guard > tolerance:
        raise ValueError("calibration_residual_exceeds_frozen_tolerance")

    slope = _canonical_rational(calibration.get("slope"), "slope")
    intercept = _canonical_rational(calibration.get("intercept"), "intercept")
    if _fraction(slope, "slope") <= 0:
        raise ValueError("calibration_slope_not_positive")
    anchor_pts: list[Fraction] = []
    seen_anchor_ids: set[str] = set()
    split_anchor_ids: dict[str, set[str]] = {}
    anchor_counts = {}
    for name in ("fit_anchors", "heldout_anchors"):
        anchors = capture.get(name)
        if not isinstance(anchors, list) or not anchors:
            raise ValueError("calibration_anchors_invalid")
        anchor_counts[name] = len(anchors)
        split_anchor_ids[name] = set()
        for anchor in anchors:
            if (not isinstance(anchor, dict)
                    or set(anchor) != {"anchor_id", "decoder_pts_seconds",
                                       "html5_media_time_seconds"}):
                raise ValueError("calibration_anchors_invalid")
            anchor_id = anchor.get("anchor_id")
            if (not isinstance(anchor_id, str) or not anchor_id.strip()
                    or anchor_id != anchor_id.strip() or anchor_id in seen_anchor_ids):
                raise ValueError("calibration_anchor_identity_invalid")
            seen_anchor_ids.add(anchor_id)
            split_anchor_ids[name].add(anchor_id)
            pts = _canonical_rational(anchor.get("decoder_pts_seconds"),
                                      "decoder_pts_seconds")
            player_time = _canonical_rational(anchor.get("html5_media_time_seconds"),
                                               "html5_media_time_seconds")
            parsed_pts = _fraction(pts, "decoder_pts_seconds")
            if parsed_pts < 0 or _fraction(player_time, "html5_media_time_seconds") < 0:
                raise ValueError("calibration_anchor_time_negative")
            anchor_pts.append(parsed_pts)
    if (split_anchor_ids["fit_anchors"] != _FIT_ANCHOR_IDS
            or split_anchor_ids["heldout_anchors"] != _HELDOUT_ANCHOR_IDS
            or seen_anchor_ids != _FIT_ANCHOR_IDS | _HELDOUT_ANCHOR_IDS):
        raise ValueError("calibration_anchor_id_sets_invalid")
    if (calibration.get("fit_anchor_count") != anchor_counts["fit_anchors"]
            or calibration.get("heldout_anchor_count") != anchor_counts["heldout_anchors"]):
        raise ValueError("calibration_anchor_count_mismatch")
    anchor_min, anchor_max = min(anchor_pts), max(anchor_pts)
    if anchor_max <= anchor_min:
        raise ValueError("calibration_anchor_span_invalid")

    payload = {
        "source_video_sha256": source_video_sha256,
        "slope": slope,
        "intercept": intercept,
        "anchor_pts_min_seconds": str(anchor_min),
        "anchor_pts_max_seconds": str(anchor_max),
        "guard_pts_seconds": str(guard),
    }
    payload_raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
    return {
        "review_token": review_token,
        "source_video_sha256": source_video_sha256,
        "video_claim_review_receipt_sha256": video_claim_review_receipt_sha256,
        "calibration_sha256": hashlib.sha256(payload_raw).hexdigest(),
        "calibration_binding_payload": payload,
        "slope": slope,
        "intercept": intercept,
        "anchor_pts_min_seconds": payload["anchor_pts_min_seconds"],
        "anchor_pts_max_seconds": payload["anchor_pts_max_seconds"],
        "guard_pts_seconds": payload["guard_pts_seconds"],
        "calibration_artifact_sha256": artifact_sha,
        "clock_correspondence_authenticated": False,
        "clock_correspondence_review_status": "not_evaluated",
        "source_authentication_status": "caller_supplied_video_hash_only",
        "execution_authorization": "not_granted",
    }
