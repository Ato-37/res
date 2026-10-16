#!/usr/bin/env python3
"""Bind a prepared HTML5 calibration capture to an unvalidated record.

This helper preserves lineage digests and the opaque review token while
building calibration coefficients. It does not authenticate the converter
bundle, browser telemetry, media identity beyond supplied bytes, or decoded
frame correspondence, and it never authorizes scoring.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from html5_clock_capture_calibration import build_html5_clock_calibration

SHA256 = re.compile(r"^[a-f0-9]{64}$")
TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")
BUNDLE_PURPOSE = "unvalidated_html5_clock_calibration_input_bundle"
BUNDLE_STATUS = "raw_observations_paired_with_declared_plan_unvalidated"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def build_bound_html5_clock_calibration(*, prepared_bundle: dict,
                                       review_token: str,
                                       capture_raw: bytes,
                                       media_bytes: bytes,
                                       frozen_tolerance_pts_seconds: str) -> dict:
    """Return a calibration record bound to one prepared bundle capture.

    ``capture_raw`` must encode the exact capture object in the prepared bundle
    (JSON whitespace/key order may differ). Bundle digests are carried forward,
    but cannot be reauthenticated here because the original observation/index/
    plan bytes are not part of the prepared bundle.
    """
    if (not isinstance(prepared_bundle, dict)
            or set(prepared_bundle) != {"schema_version", "purpose", "status",
                                        "observations_sha256", "observation_index_sha256",
                                        "anchor_plan_sha256", "captures",
                                        "execution_authorization", "limitations"}
            or type(prepared_bundle.get("schema_version")) is not int
            or prepared_bundle.get("schema_version") != 1
            or prepared_bundle.get("purpose") != BUNDLE_PURPOSE
            or prepared_bundle.get("status") != BUNDLE_STATUS
            or prepared_bundle.get("execution_authorization") != "not_granted"):
        raise ValueError("prepared_bundle_invalid")
    for field in ("observations_sha256", "observation_index_sha256", "anchor_plan_sha256"):
        if not isinstance(prepared_bundle.get(field), str) or not SHA256.fullmatch(prepared_bundle[field]):
            raise ValueError("prepared_bundle_hash_invalid")
    if not isinstance(review_token, str) or not TOKEN.fullmatch(review_token):
        raise ValueError("review_token_invalid")
    if not isinstance(capture_raw, bytes) or not isinstance(media_bytes, bytes):
        raise ValueError("capture_or_media_bytes_invalid")

    captures = prepared_bundle.get("captures")
    if not isinstance(captures, list) or len(captures) != 3:
        raise ValueError("prepared_bundle_captures_invalid")
    matches = [row for row in captures
               if isinstance(row, dict) and row.get("review_token") == review_token]
    if len(matches) != 1 or set(matches[0]) != {"review_token", "capture"}:
        raise ValueError("prepared_bundle_token_mismatch")

    try:
        supplied_capture = json.loads(capture_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("capture_json_invalid") from None
    expected_capture = matches[0].get("capture")
    if not isinstance(expected_capture, dict) or _canonical(supplied_capture) != _canonical(expected_capture):
        raise ValueError("capture_bundle_content_mismatch")
    actual_media_sha256 = hashlib.sha256(media_bytes).hexdigest()
    if expected_capture.get("source_media_sha256") != actual_media_sha256:
        raise ValueError("source_media_binding_mismatch")

    calibration = build_html5_clock_calibration(
        capture_raw, media_bytes, frozen_tolerance_pts_seconds)
    record = {key: value for key, value in calibration.items()
              if key not in {"artifact_raw", "artifact_sha256"}}
    record.update({
        "purpose": "unvalidated_html5_clock_calibration_with_lineage",
        "provenance_status": "capture_and_media_byte_bound_upstream_digests_unverified",
        "upstream_hash_verification": "digest_strings_carried_from_prepared_bundle_not_reauthenticated",
        "observations_sha256": prepared_bundle["observations_sha256"],
        "observation_index_sha256": prepared_bundle["observation_index_sha256"],
        "anchor_plan_sha256": prepared_bundle["anchor_plan_sha256"],
        "review_token": review_token,
        "capture_sha256": hashlib.sha256(capture_raw).hexdigest(),
        "source_media_sha256": actual_media_sha256,
        "execution_authorization": "not_granted",
        "limitations": [
            "prepared bundle digests are caller-supplied values and are not reauthenticated from original observation/index/plan bytes",
            "a fabricated bundle with matching capture and media bytes can produce this record",
            "browser telemetry, reviewer identity, and decoder-frame correspondence are not authenticated",
            "this record does not authorize scoring or inference",
        ],
    })
    artifact_raw = _canonical(record)
    record["artifact_sha256"] = hashlib.sha256(artifact_raw).hexdigest()
    record["artifact_raw"] = artifact_raw
    return record
