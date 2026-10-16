#!/usr/bin/env python3
"""Audit private screenshot-to-decoder-frame correspondence evidence.

This tool verifies byte-level joins and a separately signed reader decision.
It does not create screenshots, decide visual correspondence, calibrate the
browser clock, freeze a protocol, or authorize scoring.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any

from PIL import Image

from claim_mask_video_decoder import _stream_and_decode
from prepare_html5_clock_calibration_inputs import prepare_capture_bundle

SHA256 = re.compile(r"^[0-9a-f]{64}$")
TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")
MAX_SCREENSHOT_EVENT_DELTA_MS = 250.0  # Candidate value; joint protocol must freeze it.
FIT_ANCHOR_IDS = {"fit_begin", "fit_midpoint", "fit_end"}
HELDOUT_ANCHOR_IDS = {"heldout_quarter", "heldout_three_quarter"}
ANCHOR_KEYS = {"anchor_id", "decoder_pts_seconds"}
EVIDENCE_KEYS = {
    "review_token", "anchor_id", "capture_event_sha256",
    "screenshot_file", "screenshot_sha256", "screenshot_captured_at_utc",
    "viewport_bbox_xywh", "screenshot_crop_rgb_sha256",
    "decoder_frame_index", "decoder_pts_seconds", "decoder_rgb_sha256",
    "correspondence_decision",
}
TOP_KEYS = {
    "schema_version", "purpose", "plan_sha256", "plan_provenance_sha256",
    "observations_sha256", "observation_index_sha256", "capture_receipt_sha256",
    "reviewer_public_key_sha256", "reviewer_pseudonym", "reviewer_role",
    "independence_attestation", "visual_inspection_attestation", "created_at_utc",
    "records", "signature_ed25519_base64",
}
CAPTURE_RECEIPT_KEYS = {
    "schema_version", "purpose", "status", "execution_authorization", "plan_sha256",
    "plan_provenance_sha256", "capture_method", "capture_helper_file_sha256",
    "created_at_utc", "records",
}
CAPTURE_RECORD_KEYS = {
    "review_token", "anchor_id", "screenshot_file", "screenshot_sha256",
    "screenshot_captured_at_utc", "capture_method", "foreground_process",
    "foreground_pid", "foreground_window_title", "foreground_window_rect_xywh",
    "virtual_screen_rect_xywh", "hotkey_message_tick_ms",
    "capture_complete_tick_ms", "hotkey_to_capture_ms",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _strict_json(raw: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def _utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("evidence_timestamp_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("evidence_timestamp_invalid") from None
    if parsed.tzinfo is None:
        raise ValueError("evidence_timestamp_invalid")
    return parsed


def _rgb_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    material = f"{rgb.width}x{rgb.height}:".encode("ascii") + rgb.tobytes()
    return _sha(material)


def _require_private_dir(path: Path) -> None:
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError("private_evidence_directory_invalid")


def _verify_signature(public_key_raw: bytes, signed_raw: bytes, signature_b64: str) -> None:
    try:
        signature = base64.b64decode(signature_b64, validate=True)
    except (ValueError, TypeError):
        raise ValueError("reviewer_signature_encoding_invalid") from None
    if len(signature) != 64:
        raise ValueError("reviewer_signature_encoding_invalid")
    with tempfile.TemporaryDirectory(prefix="clock-evidence-signature-") as temp:
        root = Path(temp)
        key_path, message_path, signature_path = root / "key.pem", root / "payload", root / "signature"
        der_path = root / "key.der"
        key_path.write_bytes(public_key_raw)
        message_path.write_bytes(signed_raw)
        signature_path.write_bytes(signature)
        try:
            key_result = subprocess.run(
                ["openssl", "pkey", "-pubin", "-in", str(key_path), "-outform", "DER",
                 "-out", str(der_path)],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10,
            )
            if key_result.returncode != 0:
                raise ValueError("reviewer_public_key_invalid")
            der = der_path.read_bytes()
            # Ed25519 SubjectPublicKeyInfo: fixed OID prefix and 32-byte key.
            if len(der) != 44 or not der.startswith(bytes.fromhex("302a300506032b6570032100")):
                raise ValueError("reviewer_public_key_not_ed25519")
            result = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", str(key_path),
                 "-rawin", "-in", str(message_path), "-sigfile", str(signature_path)],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise ValueError("reviewer_signature_verifier_unavailable") from None
    if result.returncode != 0:
        raise ValueError("reviewer_signature_invalid")


def audit_evidence(*, plan_raw: bytes, provenance_raw: bytes,
                   observations_raw: bytes, observation_index_raw: bytes,
                   capture_receipt_raw: bytes, evidence_raw: bytes, public_key_raw: bytes,
                   media_dir: Path, screenshot_dir: Path,
                   expected_plan_sha256: str,
                   expected_plan_provenance_sha256: str,
                   expected_capture_helper_sha256: str,
                   expected_reviewer_public_key_sha256: str) -> dict:
    """Return a sanitized private audit summary; never authorizes scoring."""
    inputs = (plan_raw, provenance_raw, observations_raw,
              observation_index_raw, capture_receipt_raw, evidence_raw, public_key_raw)
    if any(not isinstance(raw, bytes) or not raw or len(raw) > 20_000_000 for raw in inputs):
        raise ValueError("evidence_input_bytes_invalid")
    for label, expected in (("plan", expected_plan_sha256),
                            ("plan_provenance", expected_plan_provenance_sha256),
                            ("reviewer_public_key", expected_reviewer_public_key_sha256)):
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError(f"{label}_expected_hash_invalid")
    if _sha(plan_raw) != expected_plan_sha256:
        raise ValueError("plan_hash_mismatch")
    if _sha(provenance_raw) != expected_plan_provenance_sha256:
        raise ValueError("plan_provenance_hash_mismatch")
    if (not isinstance(expected_capture_helper_sha256, str)
            or not SHA256.fullmatch(expected_capture_helper_sha256)):
        raise ValueError("capture_helper_expected_hash_invalid")
    if _sha(public_key_raw) != expected_reviewer_public_key_sha256:
        raise ValueError("reviewer_public_key_hash_mismatch")

    plan, provenance, observations, index, capture_receipt, evidence = map(
        _strict_json, (plan_raw, provenance_raw, observations_raw,
                       observation_index_raw, capture_receipt_raw, evidence_raw))
    if (not isinstance(plan, dict) or set(plan) != {
            "schema_version", "purpose", "status", "execution_authorization", "records"}
            or plan.get("schema_version") != 1
            or plan.get("purpose") != "predeclared_html5_clock_pts_anchor_plan"
            or plan.get("status") != "predeclared_frozen"
            or plan.get("execution_authorization") != "not_granted"):
        raise ValueError("anchor_plan_schema_invalid")
    if (not isinstance(provenance, dict)
            or provenance.get("purpose") != "private_clock_anchor_plan_candidate_provenance"
            or provenance.get("status") != "candidate_not_jointly_frozen"
            or provenance.get("execution_authorization") != "not_granted"
            or provenance.get("plan_sha256") != _sha(plan_raw)
            or provenance.get("joint_freeze_receipt_sha256") is not None):
        raise ValueError("plan_provenance_not_candidate_or_plan_mismatch")
    if not isinstance(plan["records"], list) or len(plan["records"]) != 3:
        raise ValueError("anchor_plan_record_count_invalid")
    if (not isinstance(observations, dict) or observations.get("schema_version") != 3
            or observations.get("status") != "raw_player_clock_observations_only"
            or observations.get("execution_authorization") != "not_granted"
            or observations.get("decoder_frame_identity") != "not_verified"
            or not isinstance(observations.get("records"), list)
            or len(observations["records"]) != 3):
        raise ValueError("clock_observations_schema_invalid")
    if (not isinstance(index, dict) or index.get("purpose") !=
            "unvalidated_html5_clock_observation_integrity_index"
            or index.get("payload_sha256") != _sha(observations_raw)
            or index.get("status") != "raw_observations_only"
            or index.get("execution_authorization") != "not_granted"):
        raise ValueError("clock_observation_index_invalid")
    prepare_capture_bundle(
        observations_raw, observation_index_raw, plan_raw, _sha(plan_raw),
        provenance_raw, _sha(provenance_raw),
    )
    if (not isinstance(evidence, dict) or set(evidence) != TOP_KEYS
            or type(evidence.get("schema_version")) is not int
            or evidence.get("schema_version") != 1
            or evidence.get("purpose") != "private_clock_frame_correspondence_reader_attestation"
            or evidence.get("plan_sha256") != _sha(plan_raw)
            or evidence.get("plan_provenance_sha256") != _sha(provenance_raw)
            or evidence.get("observations_sha256") != _sha(observations_raw)
            or evidence.get("observation_index_sha256") != _sha(observation_index_raw)
            or evidence.get("capture_receipt_sha256") != _sha(capture_receipt_raw)
            or evidence.get("reviewer_public_key_sha256") != _sha(public_key_raw)
            or not isinstance(evidence.get("reviewer_pseudonym"), str)
            or not evidence["reviewer_pseudonym"] or len(evidence["reviewer_pseudonym"]) > 80
            or evidence.get("reviewer_role") != "independent_evidence_reader"
            or evidence.get("independence_attestation") !=
                "independent_of_claim_generation_scoring_and_custody"
            or evidence.get("visual_inspection_attestation") !=
                "personally_compared_each_screenshot_crop_to_decoder_frame"
            ):
        raise ValueError("evidence_manifest_schema_or_binding_invalid")
    _utc(evidence["created_at_utc"])

    if (not isinstance(capture_receipt, dict) or set(capture_receipt) != CAPTURE_RECEIPT_KEYS
            or capture_receipt.get("schema_version") != 1
            or capture_receipt.get("purpose") != "private_windows_compositor_screenshot_capture_receipt_candidate"
            or capture_receipt.get("status") != "self_reported_capture_receipt_not_independently_authenticated"
            or capture_receipt.get("execution_authorization") != "not_granted"
            or capture_receipt.get("plan_sha256") != _sha(plan_raw)
            or capture_receipt.get("plan_provenance_sha256") != _sha(provenance_raw)
            or capture_receipt.get("capture_method") != "windows_gdi_copyfromscreen_foreground_window_v1"
            or capture_receipt.get("capture_helper_file_sha256") != expected_capture_helper_sha256
            or not isinstance(capture_receipt.get("records"), list)
            or len(capture_receipt["records"]) != 15):
        raise ValueError("os_capture_receipt_invalid_or_unpinned")
    _utc(capture_receipt["created_at_utc"])

    signature = evidence["signature_ed25519_base64"]
    signed_object = {key: value for key, value in evidence.items()
                     if key != "signature_ed25519_base64"}
    _verify_signature(public_key_raw, _canonical(signed_object), signature)

    obs_by_token = {}
    for row in observations["records"]:
        if (not isinstance(row, dict) or set(row) != {"review_token", "events"}
                or not isinstance(row["review_token"], str)
                or TOKEN.fullmatch(row["review_token"]) is None
                or row["review_token"] in obs_by_token
                or not isinstance(row["events"], list)):
            raise ValueError("clock_observation_record_invalid")
        obs_by_token[row["review_token"]] = row
    plan_by_token = {}
    for row in plan["records"]:
        if (not isinstance(row, dict) or not isinstance(row.get("review_token"), str)
                or TOKEN.fullmatch(row["review_token"]) is None
                or row["review_token"] in plan_by_token
                or not isinstance(row.get("source_media_sha256"), str)
                or not SHA256.fullmatch(row["source_media_sha256"])):
            raise ValueError("anchor_plan_record_invalid")
        plan_by_token[row["review_token"]] = row
    if set(obs_by_token) != set(plan_by_token):
        raise ValueError("clock_observation_plan_token_set_mismatch")
    _require_private_dir(media_dir)
    _require_private_dir(screenshot_dir)

    expected_anchors: dict[tuple[str, str], Fraction] = {}
    event_by_key: dict[tuple[str, str], dict] = {}
    video_cache = {}
    for token, plan_row in plan_by_token.items():
        anchor_list = []
        fit_anchors, heldout_anchors = plan_row.get("fit_anchors"), plan_row.get("heldout_anchors")
        if (not isinstance(fit_anchors, list) or len(fit_anchors) != 3
                or not isinstance(heldout_anchors, list) or len(heldout_anchors) != 2
                or {a.get("anchor_id") for a in fit_anchors if isinstance(a, dict)} != FIT_ANCHOR_IDS
                or {a.get("anchor_id") for a in heldout_anchors if isinstance(a, dict)} != HELDOUT_ANCHOR_IDS):
            raise ValueError("anchor_plan_anchors_invalid")
        anchor_list.extend(fit_anchors)
        anchor_list.extend(heldout_anchors)
        for anchor in anchor_list:
            if (not isinstance(anchor, dict) or set(anchor) != ANCHOR_KEYS
                    or not isinstance(anchor["anchor_id"], str)
                    or not anchor["anchor_id"] or len(anchor["anchor_id"]) > 40):
                raise ValueError("anchor_plan_anchor_invalid")
            try:
                pts = Fraction(anchor["decoder_pts_seconds"])
            except (TypeError, ValueError, ZeroDivisionError):
                raise ValueError("anchor_plan_pts_invalid") from None
            key = (token, anchor["anchor_id"])
            if key in expected_anchors:
                raise ValueError("anchor_plan_duplicate_anchor")
            expected_anchors[key] = pts
        observation_row = obs_by_token[token]
        sequences = []
        for event in observation_row["events"]:
            if (not isinstance(event, dict) or not isinstance(event.get("anchor_id"), str)
                    or type(event.get("sequence")) is not int):
                raise ValueError("clock_event_invalid")
            key = (token, event["anchor_id"])
            if key in event_by_key:
                raise ValueError("clock_event_duplicate_anchor")
            if event.get("frame_match_status") != "fresh_click_bound_callback_unverified_decoder":
                raise ValueError("clock_event_status_invalid")
            event_by_key[key] = event
            sequences.append(event["sequence"])
        if sequences != list(range(1, 6)):
            raise ValueError("clock_event_sequence_invalid")
    if set(event_by_key) != set(expected_anchors):
        raise ValueError("clock_event_anchor_set_mismatch")

    records = evidence.get("records")
    if not isinstance(records, list) or len(records) != len(expected_anchors):
        raise ValueError("evidence_anchor_count_invalid")
    seen: set[tuple[str, str]] = set()
    capture_by_key = {}
    for capture_row in capture_receipt["records"]:
        if not isinstance(capture_row, dict) or set(capture_row) != CAPTURE_RECORD_KEYS:
            raise ValueError("os_capture_record_schema_invalid")
        capture_token, capture_anchor = capture_row.get("review_token"), capture_row.get("anchor_id")
        if (not isinstance(capture_token, str) or TOKEN.fullmatch(capture_token) is None
                or not isinstance(capture_anchor, str) or not capture_anchor):
            raise ValueError("os_capture_record_identity_invalid")
        capture_key = (capture_token, capture_anchor)
        if capture_key in capture_by_key:
            raise ValueError("os_capture_record_duplicate")
        if (capture_row.get("capture_method") != capture_receipt["capture_method"]
                or capture_row.get("foreground_process") != "msedge"
                or type(capture_row.get("foreground_pid")) is not int
                or capture_row["foreground_pid"] <= 0
                or not isinstance(capture_row.get("foreground_window_title"), str)
                or "Private offline video review" not in capture_row["foreground_window_title"]):
            raise ValueError("os_capture_window_declaration_invalid")
        capture_time = _utc(capture_row.get("screenshot_captured_at_utc"))
        hotkey_tick = capture_row.get("hotkey_message_tick_ms")
        complete_tick = capture_row.get("capture_complete_tick_ms")
        elapsed_tick = capture_row.get("hotkey_to_capture_ms")
        if (any(type(value) is not int or value < 0 or value > 0xFFFFFFFF
                for value in (hotkey_tick, complete_tick, elapsed_tick))
                or elapsed_tick != ((complete_tick - hotkey_tick) & 0xFFFFFFFF)
                or elapsed_tick > 250):
            raise ValueError("os_capture_timing_declaration_invalid")
        window_rect, virtual_rect = (capture_row.get("foreground_window_rect_xywh"),
                                     capture_row.get("virtual_screen_rect_xywh"))
        if (not isinstance(window_rect, list) or len(window_rect) != 4
                or any(type(value) is not int for value in window_rect)
                or not isinstance(virtual_rect, list) or len(virtual_rect) != 4
                or any(type(value) is not int for value in virtual_rect)):
            raise ValueError("os_capture_geometry_invalid")
        wx, wy, ww, wh = window_rect
        vx, vy, vw, vh = virtual_rect
        if (ww < 100 or wh < 100 or vw < 1 or vh < 1
                or wx < vx or wy < vy or wx + ww > vx + vw or wy + wh > vy + vh):
            raise ValueError("os_capture_geometry_invalid")
        capture_by_key[capture_key] = (capture_row, capture_time)
    if set(capture_by_key) != set(expected_anchors):
        raise ValueError("os_capture_anchor_set_mismatch")
    decisions = []
    for row in records:
        if not isinstance(row, dict) or set(row) != EVIDENCE_KEYS:
            raise ValueError("evidence_record_schema_invalid")
        token, anchor_id = row.get("review_token"), row.get("anchor_id")
        if (not isinstance(token, str) or TOKEN.fullmatch(token) is None
                or not isinstance(anchor_id, str) or not anchor_id):
            raise ValueError("evidence_anchor_identity_invalid")
        key = (token, anchor_id)
        if key not in expected_anchors or key in seen:
            raise ValueError("evidence_anchor_identity_invalid")
        seen.add(key)
        event = event_by_key[key]
        if key not in capture_by_key:
            raise ValueError("os_capture_anchor_missing")
        capture_row, capture_time = capture_by_key[key]
        if (capture_row["screenshot_file"] != row["screenshot_file"]
                or capture_row["screenshot_sha256"] != row["screenshot_sha256"]
                or capture_row["screenshot_captured_at_utc"] != row["screenshot_captured_at_utc"]):
            raise ValueError("os_capture_screenshot_binding_mismatch")
        if row["capture_event_sha256"] != _sha(_canonical(event)):
            raise ValueError("evidence_capture_event_mismatch")
        if (not isinstance(row["correspondence_decision"], str)
                or row["correspondence_decision"] not in {"match", "ambiguous", "mismatch"}):
            raise ValueError("evidence_correspondence_decision_invalid")
        decisions.append(row["correspondence_decision"])

        filename = row["screenshot_file"]
        if (not isinstance(filename, str) or not filename or Path(filename).name != filename
                or filename in {".", ".."}):
            raise ValueError("screenshot_filename_invalid")
        screenshot_path = screenshot_dir / filename
        info = screenshot_path.lstat()
        if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077):
            raise ValueError("screenshot_file_invalid")
        screenshot_raw = screenshot_path.read_bytes()
        if _sha(screenshot_raw) != row["screenshot_sha256"]:
            raise ValueError("screenshot_hash_mismatch")
        _utc(row["screenshot_captured_at_utc"])
        skew_ms = abs((capture_time - _utc(event["captured_at_utc"])).total_seconds() * 1000)
        if skew_ms > MAX_SCREENSHOT_EVENT_DELTA_MS:
            raise ValueError("screenshot_capture_event_time_skew")
        bbox = row["viewport_bbox_xywh"]
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(type(value) is not int for value in bbox)):
            raise ValueError("screenshot_viewport_geometry_invalid")
        x, y, width, height = bbox
        if min(x, y) < 0 or min(width, height) < 1:
            raise ValueError("screenshot_viewport_geometry_invalid")
        try:
            with Image.open(io.BytesIO(screenshot_raw)) as image:
                image.load()
                if x + width > image.width or y + height > image.height:
                    raise ValueError("screenshot_viewport_out_of_bounds")
                if image.size != (ww, wh):
                    raise ValueError("screenshot_window_capture_dimensions_mismatch")
                crop_digest = _rgb_digest(image.crop((x, y, x + width, y + height)))
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc) == "screenshot_viewport_out_of_bounds":
                raise
            raise ValueError("screenshot_image_invalid") from None
        if crop_digest != row["screenshot_crop_rgb_sha256"]:
            raise ValueError("screenshot_crop_hash_mismatch")

        media_name = token + ".mp4"
        if media_name not in video_cache:
            media_path = media_dir / media_name
            media_info = media_path.lstat()
            if (stat.S_ISLNK(media_info.st_mode) or not stat.S_ISREG(media_info.st_mode)
                    or media_info.st_uid != os.geteuid()
                    or stat.S_IMODE(media_info.st_mode) & 0o077):
                raise ValueError("staged_media_file_invalid")
            media_raw = media_path.read_bytes()
            media_digest = _sha(media_raw)
            if media_digest != plan_by_token[token]["source_media_sha256"]:
                raise ValueError("staged_media_hash_mismatch")
            pts, time_base, relative_times, _pixel_spec, pixel_hashes, _stream_info = \
                _stream_and_decode(io.BytesIO(media_raw))
            video_cache[media_name] = (relative_times, pixel_hashes)
        relative_times, pixel_hashes = video_cache[media_name]
        frame_index = row["decoder_frame_index"]
        if type(frame_index) is not int or not 0 <= frame_index < len(relative_times):
            raise ValueError("decoder_frame_index_invalid")
        if (relative_times[frame_index] != expected_anchors[key]
                or row["decoder_pts_seconds"] != str(relative_times[frame_index])):
            raise ValueError("decoder_frame_pts_plan_mismatch")
        if row["decoder_rgb_sha256"] != pixel_hashes[frame_index]:
            raise ValueError("decoder_frame_rgb_hash_mismatch")

    if seen != set(expected_anchors):
        raise ValueError("evidence_anchor_set_incomplete")
    pass_status = all(decision == "match" for decision in decisions)
    return {
        "status": "candidate_signed_evidence_bytes_consistent" if pass_status
                  else "blocked_correspondence_not_all_match",
        "scope": "private_three_video_clock_frame_correspondence_audit",
        "requested_records": 3,
        "requested_anchors": len(expected_anchors),
        "verified_anchors": len(seen),
        "match_decisions": decisions.count("match"),
        "ambiguous_decisions": decisions.count("ambiguous"),
        "mismatch_decisions": decisions.count("mismatch"),
        "screenshot_and_decoder_bytes_checked": True,
        "reviewer_signature_verified": True,
        "reviewer_attestations_present": True,
        "joint_freeze_status": "not_jointly_frozen",
        "reviewer_key_trust_status": "expected_hash_supplied_by_caller_not_joint_freeze_authenticated",
        "execution_authorization": "not_granted",
        "clock_calibration_status": "not_evaluated",
        "limitations": [
            "signature verification authenticates the selected key only when its hash is frozen independently",
            "the tool verifies recorded correspondence decisions but does not itself inspect or compare displayed imagery",
            "the 250 ms screenshot-to-event proximity threshold is a candidate and must be jointly frozen before use",
            "all browser-clock, protocol-freeze, target, and production-scoring gates remain separate",
        ],
    }


def _read_private(path: Path) -> bytes:
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError("private_input_file_invalid")
    return path.read_bytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-provenance", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--observation-index", required=True, type=Path)
    parser.add_argument("--capture-receipt", required=True, type=Path)
    parser.add_argument("--evidence-manifest", required=True, type=Path)
    parser.add_argument("--reviewer-public-key", required=True, type=Path)
    parser.add_argument("--media-dir", required=True, type=Path)
    parser.add_argument("--screenshot-dir", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-plan-provenance-sha256", required=True)
    parser.add_argument("--expected-capture-helper-sha256", required=True)
    parser.add_argument("--expected-reviewer-public-key-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        summary = audit_evidence(
            plan_raw=_read_private(args.plan),
            provenance_raw=_read_private(args.plan_provenance),
            observations_raw=_read_private(args.observations),
            observation_index_raw=_read_private(args.observation_index),
            capture_receipt_raw=_read_private(args.capture_receipt),
            evidence_raw=_read_private(args.evidence_manifest),
            public_key_raw=_read_private(args.reviewer_public_key),
            media_dir=args.media_dir, screenshot_dir=args.screenshot_dir,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_plan_provenance_sha256=args.expected_plan_provenance_sha256,
            expected_capture_helper_sha256=args.expected_capture_helper_sha256,
            expected_reviewer_public_key_sha256=args.expected_reviewer_public_key_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failure_code = str(exc) if isinstance(exc, ValueError) else "private_input_io_error"
        print(json.dumps({"status": "blocked", "failure_code": failure_code,
                          "execution_authorization": "not_granted"}, sort_keys=True))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "candidate_signed_evidence_bytes_consistent" else 2


if __name__ == "__main__":
    raise SystemExit(main())
