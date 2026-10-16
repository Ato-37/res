"""Build an unvalidated clock-calibration record from local HTML5 observations.

This CPU-only helper consumes a capture made through the actual HTML5 review
player and paired decoder-PTS anchors. It checks structure and media-byte
binding, fits only the declared fit anchors, and reports the maximum observed
held-out residual in PTS seconds. It cannot authenticate browser telemetry,
anchor correspondence, temporal coverage, reviewer independence, or rights;
the empirical residual is not a statistical uncertainty interval or future
error bound, and the result never authorizes scoring.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import re
from typing import Any

from annotation_clock_calibration import evaluate_affine_calibration

SHA256 = re.compile(r"^[a-f0-9]{64}$")
PLAYER_CLOCK = "html5_video_requestVideoFrameCallback_mediaTime_seconds"
CAPTURE_POLICY_ID = "rvfc_click_bound_v1"
CAPTURE_THRESHOLDS = {
    "max_callback_age_ms": 250.0,
    "max_request_delay_ms": 250.0,
    "max_click_player_time_delta_seconds": 0.1,
    "max_callback_player_time_delta_seconds": 0.1,
    "callback_wait_timeout_ms": 1000.0,
}


def _fraction(value: Any, field: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(field + "_invalid")
    try:
        return Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        raise ValueError(field + "_invalid") from None


def _unique_object(pairs):
    """Reject ambiguous duplicate JSON keys even when called outside converter."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("capture_duplicate_json_key")
        result[key] = value
    return result


def build_html5_clock_calibration(capture_raw: bytes, media_bytes: bytes,
                                 frozen_tolerance_pts_seconds: str) -> dict:
    """Return a planner-compatible empirical calibration record.

    Capture schema v3 requires explicit actual-player metadata and disjoint
    anchor lists. Anchor rows pair decoder PTS with rVFC ``mediaTime``. The
    caller must separately verify that metadata is fresh and the displayed
    frame corresponds to the decoder frame; this helper cannot establish that.
    The caller must freeze the tolerance and anchor selection before capture.
    """
    try:
        capture = json.loads(capture_raw, object_pairs_hook=_unique_object)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("capture_json_invalid") from None
    required = {"schema_version", "purpose", "player_clock", "capture_policy_id",
                "capture_thresholds", "browser_user_agent",
                "current_src_matches_staged_media", "source_media_sha256",
                "playback_rate", "muted", "text_tracks_disabled",
                "fit_anchors", "heldout_anchors"}
    if not isinstance(capture, dict) or set(capture) != required:
        raise ValueError("capture_schema_invalid")
    if (type(capture["schema_version"]) is not int or capture["schema_version"] != 3
            or capture["purpose"] != "local_html5_player_clock_anchor_capture"
            or capture["capture_policy_id"] != CAPTURE_POLICY_ID
            or capture["capture_thresholds"] != CAPTURE_THRESHOLDS
            or capture["player_clock"] != PLAYER_CLOCK):
        raise ValueError("capture_identity_invalid")
    if (not isinstance(capture["browser_user_agent"], str)
            or not capture["browser_user_agent"].strip()
            or capture["current_src_matches_staged_media"] is not True
            or capture["playback_rate"] != "1"
            or capture["muted"] is not True
            or capture["text_tracks_disabled"] is not True):
        raise ValueError("html5_player_state_invalid")
    if not isinstance(media_bytes, bytes) or not media_bytes:
        raise ValueError("media_bytes_invalid")
    media_sha = hashlib.sha256(media_bytes).hexdigest()
    if capture["source_media_sha256"] != media_sha:
        raise ValueError("source_media_binding_mismatch")

    def anchors(field: str) -> list[dict]:
        rows = capture[field]
        if not isinstance(rows, list) or not rows:
            raise ValueError(field + "_invalid")
        converted = []
        seen = set()
        for row in rows:
            if (not isinstance(row, dict)
            or set(row) != {"anchor_id", "decoder_pts_seconds", "html5_media_time_seconds"}):
                raise ValueError(field + "_invalid")
            anchor_id = row["anchor_id"]
            if (not isinstance(anchor_id, str) or not anchor_id.strip()
                    or anchor_id != anchor_id.strip() or anchor_id in seen):
                raise ValueError("anchor_identity_invalid")
            seen.add(anchor_id)
            pts = _fraction(row["decoder_pts_seconds"], "decoder_pts")
            player = _fraction(row["html5_media_time_seconds"], "html5_media_time")
            if pts < 0 or player < 0:
                raise ValueError("anchor_time_negative")
            converted.append({"anchor_id": anchor_id, "pts_time": str(pts),
                              "annotation_time": str(player)})
        return converted

    fit = anchors("fit_anchors")
    heldout = anchors("heldout_anchors")
    if {row["anchor_id"] for row in fit} & {row["anchor_id"] for row in heldout}:
        raise ValueError("fit_heldout_identity_overlap")
    tolerance = _fraction(frozen_tolerance_pts_seconds, "tolerance")
    if tolerance <= 0:
        raise ValueError("tolerance_not_positive")
    evaluation = evaluate_affine_calibration(fit, heldout, frozen_tolerance_pts_seconds)
    slope = Fraction(evaluation["slope"])
    # Convert held-out player-clock residuals to decoder PTS seconds. This is
    # the observed maximum only, with no iid or future-error interpretation.
    guard = Fraction(evaluation["heldout_max_abs_residual"]) / slope
    artifact = {
        "clock_scope": PLAYER_CLOCK,
        "capture_policy_id": CAPTURE_POLICY_ID,
        "capture_thresholds": CAPTURE_THRESHOLDS,
        # This is only an arithmetic result. It is deliberately not named
        # ``heldout_pass`` so downstream code cannot mistake it for validation.
        "empirical_residual_within_tolerance": guard <= tolerance,
        "validation_scope": "empirical_heldout_residual_only",
        "guard_band_source": "max_observed_heldout_residual_pts_seconds",
        "uncertainty_status": "not_evaluated",
        "fit_anchor_count": len(fit),
        "heldout_anchor_count": len(heldout),
        "source_media_sha256": media_sha,
        "slope": evaluation["slope"],
        "intercept": evaluation["intercept"],
        "empirical_guard_band_pts_seconds": str(guard),
        "frozen_tolerance_seconds": str(tolerance),
        "capture_sha256": hashlib.sha256(capture_raw).hexdigest(),
        "execution_authorization": "not_granted",
    }
    artifact_raw = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    artifact["artifact_sha256"] = hashlib.sha256(artifact_raw).hexdigest()
    artifact["artifact_raw"] = artifact_raw
    return artifact
