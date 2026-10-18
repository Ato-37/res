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

from build_blinded_frame_pair_review import audit_export_bytes
from claim_mask_video_decoder import _stream_and_decode
from prepare_html5_clock_calibration_inputs import prepare_capture_bundle
from nextgqa_capture_operator_attestation import verify_attestation
from audit_nextgqa_review_release_order import _validate_release_roster
from nextgqa_pre_review_integrity_receipt import ReceiptError, _public_der

MAX_STAGED_VIDEO_BYTES = 128 * 1024 * 1024
MAX_SCREENSHOT_BYTES = 40_000_000
MAX_SNAPSHOT_BYTES = 768 * 1024 * 1024
MAX_SCREENSHOT_PIXELS = 50_000_000
MAX_IMAGE_SIDE = 12_000
MAX_DECODED_VIDEO_FRAMES = 10_000

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
EVIDENCE_V3_RECORD_KEYS = EVIDENCE_KEYS | {"pair_id"}
TOP_KEYS = {
    "schema_version", "purpose", "plan_sha256", "plan_provenance_sha256",
    "observations_sha256", "observation_index_sha256", "capture_receipt_sha256",
    "reviewer_public_key_sha256", "reviewer_pseudonym", "reviewer_role",
    "independence_attestation", "visual_inspection_attestation", "created_at_utc",
    "records", "signature_ed25519_base64",
}
TOP_KEYS_V2 = {
    "schema_version", "purpose", "plan_sha256", "plan_provenance_sha256",
    "observations_sha256", "observation_index_sha256", "capture_receipt_sha256",
    "capture_operator_attestation_sha256", "capture_operator_integrity_roster_sha256",
    "capture_operator_release_roster_sha256", "capture_operator_trust_root_der_sha256",
    "capture_operator_key_id", "capture_operator_key_der_sha256",
    "capture_operator_signature_verified", "capture_operator_human_identity_verified",
    "capture_operator_process_authenticated", "capture_operator_frame_correspondence_reviewed",
    "capture_operator_score_ready", "model_load_permitted", "scoring_authorized",
    "execution_authorization", "reviewer_identity_verified",
    "frame_reviewer_key_id", "frame_reviewer_key_der_sha256",
    "frame_reviewer_role_signature_verified",
    "reviewer_public_key_sha256", "reviewer_pseudonym",
    "reviewer_role", "independence_attestation", "visual_inspection_attestation",
    "created_at_utc", "records", "signature_ed25519_base64",
}
TOP_KEYS_V3 = (TOP_KEYS_V2 - {"signature_ed25519_base64"}) | {
    "pair_package_sha256", "pair_identity_map_sha256",
    "pair_review_export_sha256", "pair_reviewer_html_sha256",
    "pair_package_nonce", "signature_ed25519_base64",
}
EVIDENCE_V2_DOMAIN = b"nextgqa_clock_frame_evidence_manifest_v2\0"
EVIDENCE_V3_DOMAIN = b"nextgqa_clock_frame_evidence_manifest_v3\0"
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


def _snapshot_private_directory(path: Path, expected_names: set[str],
                                *, label: str,
                                additional_names: set[str] | None = None
                                ) -> tuple[dict[str, bytes], dict[str, tuple], tuple]:
    """Read exact leaves once through an owner-only directory descriptor."""
    if (not isinstance(expected_names, set) or not expected_names
            or any(not isinstance(name, str) or Path(name).name != name
                   or name in {".", ".."} for name in expected_names)):
        raise ValueError(label + "_names_invalid")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    dir_fd = os.open(path, flags)
    try:
        before = os.fstat(dir_fd)
        if (not stat.S_ISDIR(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) & 0o077):
            raise ValueError(label + "_directory_invalid")
        additional_names = additional_names or set()
        if (expected_names & additional_names
                or any(not isinstance(name, str) or Path(name).name != name
                       or name in {".", ".."} for name in additional_names)
                or set(os.listdir(dir_fd)) != expected_names | additional_names):
            raise ValueError(label + "_directory_contents_invalid")
        directory_identity = (before.st_dev, before.st_ino, before.st_mode, before.st_uid)
        values: dict[str, bytes] = {}
        identities: dict[str, tuple] = {}
        aggregate_bytes = 0
        for name in sorted(expected_names | additional_names):
            fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=dir_fd)
            try:
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                        or stat.S_IMODE(info.st_mode) & 0o077):
                    raise ValueError(label + "_file_invalid")
                file_limit = (MAX_STAGED_VIDEO_BYTES if label == "staged_media"
                              and name.endswith(".mp4") else MAX_SCREENSHOT_BYTES
                              if label == "screenshot" else 20_000_000)
                if info.st_size < 1 or info.st_size > file_limit:
                    raise ValueError(label + "_file_size_exceeds_limit")
                aggregate_bytes += info.st_size
                if aggregate_bytes > MAX_SNAPSHOT_BYTES:
                    raise ValueError(label + "_snapshot_size_exceeds_limit")
                chunks = []
                remaining = info.st_size
                while remaining:
                    chunk = os.read(fd, min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                if remaining or os.read(fd, 1):
                    raise ValueError(label + "_file_changed_during_read")
                after = os.fstat(fd)
                if (after.st_size != info.st_size or after.st_ino != info.st_ino
                        or after.st_mtime_ns != info.st_mtime_ns
                        or after.st_ctime_ns != info.st_ctime_ns):
                    raise ValueError(label + "_file_changed_during_read")
                values[name] = b"".join(chunks)
                identities[name] = (info.st_dev, info.st_ino, info.st_mode, info.st_uid,
                                    info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            finally:
                os.close(fd)
        current = os.fstat(dir_fd)
        if (current.st_dev, current.st_ino, current.st_mode, current.st_uid) != directory_identity:
            raise ValueError(label + "_directory_changed_during_read")
        return values, identities, directory_identity
    finally:
        os.close(dir_fd)


def _verify_private_snapshot_still_current(path: Path, names: set[str],
                                           file_identities: dict[str, tuple],
                                           directory_identity: tuple,
                                           *, label: str) -> None:
    try:
        directory = os.stat(path, follow_symlinks=False)
        if (directory.st_dev, directory.st_ino, directory.st_mode, directory.st_uid) != directory_identity:
            raise ValueError(label + "_directory_path_changed_after_snapshot")
        dir_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened_directory = os.fstat(dir_fd)
            if (opened_directory.st_dev, opened_directory.st_ino,
                    opened_directory.st_mode, opened_directory.st_uid) != directory_identity:
                raise ValueError(label + "_directory_path_changed_after_snapshot")
            if set(os.listdir(dir_fd)) != names:
                raise ValueError(label + "_directory_contents_changed_after_snapshot")
            for name in names:
                info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                observed = (info.st_dev, info.st_ino, info.st_mode, info.st_uid,
                            info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                if observed != file_identities.get(name):
                    raise ValueError(label + "_path_changed_after_snapshot")
        finally:
            os.close(dir_fd)
    except OSError:
        raise ValueError(label + "_path_changed_after_snapshot") from None


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
                   expected_reviewer_public_key_sha256: str,
                   capture_operator_attestation_raw: bytes | None = None,
                   crop_map_raw: bytes | None = None,
                   crop_crosswalk_raw: bytes | None = None,
                   integrity_roster_raw: bytes | None = None,
                   release_roster_raw: bytes | None = None,
                   root_public_key_raw: bytes | None = None,
                   expected_root_public_der_sha256: str | None = None,
                   crop_dir: Path | None = None,
                   pair_package_raw: bytes | None = None,
                   pair_identity_map_raw: bytes | None = None,
                   pair_review_export_raw: bytes | None = None,
                   pair_reviewer_html_raw: bytes | None = None) -> dict:
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
    evidence_version = evidence.get("schema_version") if isinstance(evidence, dict) else None
    evidence_keys = (TOP_KEYS if evidence_version == 1 else TOP_KEYS_V2
                     if evidence_version == 2 else TOP_KEYS_V3)
    expected_purpose = ("private_clock_frame_correspondence_reader_attestation"
                        if evidence_version == 1 else
                        "private_clock_frame_correspondence_reader_attestation_v2"
                        if evidence_version == 2 else
                        "private_clock_frame_correspondence_reader_attestation_v3")
    if (not isinstance(evidence, dict) or set(evidence) != evidence_keys
            or type(evidence_version) is not int
            or evidence_version not in {1, 2, 3}
            or evidence.get("purpose") != expected_purpose
            or evidence.get("plan_sha256") != _sha(plan_raw)
            or evidence.get("plan_provenance_sha256") != _sha(provenance_raw)
            or evidence.get("observations_sha256") != _sha(observations_raw)
            or evidence.get("observation_index_sha256") != _sha(observation_index_raw)
            or evidence.get("capture_receipt_sha256") != _sha(capture_receipt_raw)
            or evidence.get("reviewer_public_key_sha256") != _sha(public_key_raw)
            or not isinstance(evidence.get("reviewer_pseudonym"), str)
            or not evidence["reviewer_pseudonym"] or len(evidence["reviewer_pseudonym"]) > 80
            or evidence.get("reviewer_role") != (
                "independent_evidence_reader" if evidence_version == 1 else "frame_reviewer")
            or (evidence_version in {2, 3} and evidence.get("reviewer_identity_verified") is not False)
            or evidence.get("independence_attestation") !=
                "independent_of_claim_generation_scoring_and_custody"
            or evidence.get("visual_inspection_attestation") !=
                "personally_compared_each_screenshot_crop_to_decoder_frame"
            ):
        raise ValueError("evidence_manifest_schema_or_binding_invalid")
    _utc(evidence["created_at_utc"])

    if evidence_version in {2, 3} and any(value is None for value in (
            capture_operator_attestation_raw, crop_map_raw, crop_crosswalk_raw,
            integrity_roster_raw, release_roster_raw, root_public_key_raw,
            expected_root_public_der_sha256, crop_dir)):
        raise ValueError("capture_operator_audit_inputs_missing")
    pair_audit = None
    if evidence_version == 3:
        if any(raw is None for raw in (pair_package_raw, pair_identity_map_raw,
                pair_review_export_raw, pair_reviewer_html_raw)):
            raise ValueError("pair_review_audit_inputs_missing")
        if (evidence.get("pair_package_sha256") != _sha(pair_package_raw)
                or evidence.get("pair_identity_map_sha256") != _sha(pair_identity_map_raw)
                or evidence.get("pair_review_export_sha256") != _sha(pair_review_export_raw)
                or evidence.get("pair_reviewer_html_sha256") != _sha(pair_reviewer_html_raw)):
            raise ValueError("pair_review_artifact_hash_mismatch")
        if (not isinstance(pair_identity_map_raw, bytes)
                or len(pair_identity_map_raw) > 20_000_000):
            raise ValueError("pair_identity_map_bytes_invalid")
        pair_audit = audit_export_bytes(
            export_raw=pair_review_export_raw, package_raw=pair_package_raw,
            reviewer_html_raw=pair_reviewer_html_raw)
        if (pair_audit["status"] != "candidate_review_complete_not_authorizing"
                or pair_audit["package_manifest"].get("plan_sha256") != _sha(plan_raw)
                or pair_audit["package_manifest"].get("package_nonce") !=
                    evidence.get("pair_package_nonce")):
            raise ValueError("pair_review_manifest_binding_invalid")
    elif any(raw is not None for raw in (pair_package_raw, pair_identity_map_raw,
            pair_review_export_raw, pair_reviewer_html_raw)):
        raise ValueError("pair_review_inputs_unexpected_for_legacy_manifest")

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
    _verify_signature(public_key_raw,
                      (EVIDENCE_V2_DOMAIN if evidence_version == 2 else
                       EVIDENCE_V3_DOMAIN if evidence_version == 3 else b"")
                      + _canonical(signed_object), signature)
    if evidence_version in {2, 3}:
        if any(value is None for value in (capture_operator_attestation_raw,
                crop_map_raw, crop_crosswalk_raw, integrity_roster_raw,
                release_roster_raw, root_public_key_raw,
                expected_root_public_der_sha256, crop_dir)):
            raise ValueError("capture_operator_audit_inputs_missing")
        if (not isinstance(expected_root_public_der_sha256, str)
                or not SHA256.fullmatch(expected_root_public_der_sha256)):
            raise ValueError("capture_operator_root_pin_invalid")
    elif any(value is not None for value in (capture_operator_attestation_raw,
            crop_map_raw, crop_crosswalk_raw, integrity_roster_raw,
            release_roster_raw, root_public_key_raw,
            expected_root_public_der_sha256, crop_dir)):
        raise ValueError("unexpected_capture_operator_inputs_for_v1")

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
    if evidence_version == 3:
        identity = _strict_json(pair_identity_map_raw)
        identity_rows = identity.get("records") if isinstance(identity, dict) else None
        if (not isinstance(identity, dict)
                or set(identity) not in ({"schema_version", "purpose", "status",
                    "package_manifest_sha256", "records"},
                    {"schema_version", "purpose", "status", "package_manifest_sha256",
                     "input_manifest_sha256", "records"})
                or identity.get("schema_version") != 1
                or identity.get("purpose") != "private_blinded_frame_pair_identity_map"
                or identity.get("status") != "custodian_only"
                or identity.get("package_manifest_sha256") != _sha(pair_package_raw)
                or ("input_manifest_sha256" in identity
                    and (not isinstance(identity["input_manifest_sha256"], str)
                         or not SHA256.fullmatch(identity["input_manifest_sha256"])))
                or not isinstance(identity_rows, list) or len(identity_rows) != 15
                or any(not isinstance(item, dict) or set(item) != {
                    "pair_id", "review_token", "anchor_id", "A_source", "B_source"}
                    for item in identity_rows)):
            raise ValueError("pair_identity_map_schema_invalid")
        mapped_keys = [(item["review_token"], item["anchor_id"]) for item in identity_rows]
        mapped_pairs = [item["pair_id"] for item in identity_rows]
        signed_pairs = [item.get("pair_id") for item in records]
        if (any(not isinstance(token, str) or not TOKEN.fullmatch(token)
                or not isinstance(anchor, str) or not anchor for token, anchor in mapped_keys)
                or any(not isinstance(pair, str) or not TOKEN.fullmatch(pair)
                       for pair in mapped_pairs + signed_pairs)
                or set(mapped_keys) != set(expected_anchors) or len(set(mapped_keys)) != 15
                or len(set(mapped_pairs)) != 15 or len(set(signed_pairs)) != 15
                or set(mapped_pairs) != set(signed_pairs)):
            raise ValueError("pair_identity_map_coverage_invalid")
    expected_screenshot_names = {
        f"{token}_{anchor_id}.png" for token, anchor_id in expected_anchors
    }
    observed_screenshot_names = []
    observed_evidence_keys = []
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("evidence_record_schema_invalid")
        token, anchor_id, filename = (row.get("review_token"), row.get("anchor_id"),
                                      row.get("screenshot_file"))
        if (not isinstance(token, str) or not isinstance(anchor_id, str)
                or (token, anchor_id) not in expected_anchors
                or filename != f"{token}_{anchor_id}.png"):
            raise ValueError("screenshot_anchor_filename_binding_invalid")
        observed_evidence_keys.append((token, anchor_id))
        observed_screenshot_names.append(filename)
    if (set(observed_evidence_keys) != set(expected_anchors)
            or len(set(observed_evidence_keys)) != len(expected_anchors)):
        raise ValueError("evidence_anchor_coverage_invalid")
    if (set(observed_screenshot_names) != expected_screenshot_names
            or len(set(observed_screenshot_names)) != len(expected_screenshot_names)):
        raise ValueError("screenshot_anchor_reuse_or_coverage_invalid")
    screenshot_snapshot_names = expected_screenshot_names | {"capture_receipt.json"}
    screenshot_snapshot, screenshot_identities, screenshot_directory_identity = \
        _snapshot_private_directory(screenshot_dir, expected_screenshot_names,
                                    label="screenshot",
                                    additional_names={"capture_receipt.json"})
    if screenshot_snapshot["capture_receipt.json"] != capture_receipt_raw:
        raise ValueError("capture_receipt_file_byte_mismatch")
    crop_snapshot = None
    crop_snapshot_names: set[str] = set()
    crop_identities = None
    crop_directory_identity = None
    if evidence_version in {2, 3}:
        crop_crosswalk = _strict_json(crop_crosswalk_raw)
        if (not isinstance(crop_crosswalk, dict)
                or not isinstance(crop_crosswalk.get("records"), list)):
            raise ValueError("capture_operator_crop_crosswalk_invalid")
        crop_snapshot_names = {row.get("crop_file") for row in crop_crosswalk["records"]
                               if isinstance(row, dict)}
        if (len(crop_snapshot_names) != 15 or None in crop_snapshot_names
                or any(not isinstance(name, str) or Path(name).name != name
                       or name in {".", ".."} for name in crop_snapshot_names)):
            raise ValueError("capture_operator_crop_artifact_set_invalid")
        crop_snapshot, crop_identities, crop_directory_identity = \
            _snapshot_private_directory(crop_dir, crop_snapshot_names, label="screenshot")
        operator_report = verify_attestation(
            attestation_raw=capture_operator_attestation_raw,
            plan_raw=plan_raw, expected_plan_sha256=expected_plan_sha256,
            plan_provenance_raw=provenance_raw,
            expected_plan_provenance_sha256=expected_plan_provenance_sha256,
            observations_raw=observations_raw, observation_index_raw=observation_index_raw,
            capture_receipt_raw=capture_receipt_raw, crop_map_raw=crop_map_raw,
            crop_crosswalk_raw=crop_crosswalk_raw,
            screenshot_files={name: screenshot_snapshot[name]
                              for name in expected_screenshot_names},
            crop_files=crop_snapshot,
            integrity_roster_raw=integrity_roster_raw,
            release_roster_raw=release_roster_raw,
            root_public_key_pem=root_public_key_raw,
            expected_root_public_der_sha256=expected_root_public_der_sha256)
        try:
            release_roster, role_public_keys = _validate_release_roster(
                release_roster_raw, root_public_key_raw,
                expected_root_public_der_sha256, integrity_roster_raw)
        except ReceiptError as exc:
            raise ValueError("release_roster_invalid:" + str(exc)) from None
        frame_public = role_public_keys.get("frame_reviewer")
        frame_entry = next((entry for entry in release_roster["keys"]
                            if entry.get("role") == "frame_reviewer"), None)
        if (frame_public is None or not isinstance(frame_entry, dict)
                or _public_der(public_key_raw) != _public_der(frame_public)
                or evidence.get("reviewer_role") != "frame_reviewer"
                or evidence.get("frame_reviewer_key_id") != frame_entry.get("key_id")
                or evidence.get("frame_reviewer_key_der_sha256") !=
                    _sha(_public_der(frame_public))
                or evidence.get("frame_reviewer_role_signature_verified") is not True):
            raise ValueError("frame_reviewer_rostered_key_binding_invalid")
        operator_envelope = _strict_json(capture_operator_attestation_raw)
        if (operator_report.get("status") !=
                "capture_operator_key_signature_valid_candidate_artifact_bytes"
                or evidence["capture_operator_attestation_sha256"] != _sha(capture_operator_attestation_raw)
                or evidence["capture_operator_integrity_roster_sha256"] != _sha(integrity_roster_raw)
                or evidence["capture_operator_release_roster_sha256"] != _sha(release_roster_raw)
                or evidence["capture_operator_trust_root_der_sha256"] != expected_root_public_der_sha256
                or evidence["capture_operator_key_id"] != operator_report.get("capture_operator_key_id")
                or evidence["capture_operator_key_der_sha256"] != operator_envelope.get("capture_operator_key_der_sha256")
                or evidence["capture_operator_signature_verified"] is not True
                or evidence["capture_operator_human_identity_verified"] is not False
                or evidence["capture_operator_process_authenticated"] is not False
                or evidence["capture_operator_frame_correspondence_reviewed"] is not False
                or evidence["capture_operator_score_ready"] is not False
                or evidence["model_load_permitted"] is not False
                or evidence["scoring_authorized"] is not False
                or evidence["execution_authorization"] != "not_granted"):
            raise ValueError("capture_operator_evidence_binding_or_authority_invalid")
    expected_media_names = {token + ".mp4" for token in plan_by_token}
    media_snapshot, media_identities, media_directory_identity = \
        _snapshot_private_directory(media_dir, expected_media_names, label="staged_media")
    seen: set[tuple[str, str]] = set()
    capture_by_key = {}
    capture_filenames = []
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
        expected_filename = f"{capture_token}_{capture_anchor}.png"
        if capture_row.get("screenshot_file") != expected_filename:
            raise ValueError("os_capture_screenshot_anchor_filename_mismatch")
        capture_filenames.append(capture_row["screenshot_file"])
    if (len(set(capture_filenames)) != len(expected_anchors)
            or set(capture_filenames) != expected_screenshot_names):
        raise ValueError("os_capture_screenshot_reuse_or_coverage_invalid")
    if set(capture_by_key) != set(expected_anchors):
        raise ValueError("os_capture_anchor_set_mismatch")
    decisions = []
    for row in records:
        expected_record_keys = (EVIDENCE_KEYS if evidence_version in {1, 2}
                                else EVIDENCE_V3_RECORD_KEYS)
        if not isinstance(row, dict) or set(row) != expected_record_keys:
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
        screenshot_raw = screenshot_snapshot[filename]
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
                if image.format != "PNG":
                    raise ValueError("screenshot_image_invalid")
                if (image.width < 1 or image.height < 1
                        or image.width > MAX_IMAGE_SIDE or image.height > MAX_IMAGE_SIDE
                        or image.width * image.height > MAX_SCREENSHOT_PIXELS):
                    raise ValueError("screenshot_dimensions_exceed_limit")
                image.load()
                if x + width > image.width or y + height > image.height:
                    raise ValueError("screenshot_viewport_out_of_bounds")
                if image.size != (ww, wh):
                    raise ValueError("screenshot_window_capture_dimensions_mismatch")
                crop_digest = _rgb_digest(image.crop((x, y, x + width, y + height)))
        except (OSError, ValueError) as exc:
            if (isinstance(exc, ValueError) and str(exc) in {
                    "screenshot_viewport_out_of_bounds",
                    "screenshot_dimensions_exceed_limit"}):
                raise
            raise ValueError("screenshot_image_invalid") from None
        if crop_digest != row["screenshot_crop_rgb_sha256"]:
            raise ValueError("screenshot_crop_hash_mismatch")

        media_name = token + ".mp4"
        if media_name not in video_cache:
            media_raw = media_snapshot[media_name]
            media_digest = _sha(media_raw)
            if media_digest != plan_by_token[token]["source_media_sha256"]:
                raise ValueError("staged_media_hash_mismatch")
            pts, time_base, relative_times, _pixel_spec, pixel_hashes, _stream_info = \
                _stream_and_decode(io.BytesIO(media_raw),
                                   max_frames=MAX_DECODED_VIDEO_FRAMES,
                                   max_pixels=MAX_SCREENSHOT_PIXELS,
                                   max_side=MAX_IMAGE_SIDE)
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

        if evidence_version == 3:
            package = pair_audit["package_manifest"]
            identity = _strict_json(pair_identity_map_raw)
            identity_rows = identity.get("records") if isinstance(identity, dict) else None
            mapping = [item for item in identity_rows or []
                       if isinstance(item, dict)
                       and item.get("review_token") == token
                       and item.get("anchor_id") == anchor_id]
            pair_id = row.get("pair_id")
            if (len(mapping) != 1 or mapping[0].get("pair_id") != pair_id
                    or mapping[0].get("A_source") not in {"screenshot_crop", "decoder_frame"}
                    or mapping[0].get("B_source") not in {"screenshot_crop", "decoder_frame"}
                    or mapping[0].get("A_source") == mapping[0].get("B_source")):
                raise ValueError("pair_identity_map_evidence_join_invalid")
            package_rows = {item["pair_id"]: item for item in package["pairs"]}
            review_rows = {item["pair_id"]: item
                           for item in pair_audit["review_export"]["records"]}
            html_rows = {item["id"]: item for item in pair_audit["reviewer_html_pairs"]}
            if (pair_id not in package_rows or pair_id not in review_rows
                    or pair_id not in html_rows
                    or review_rows[pair_id].get("crop_purity_attestation") is not True
                    or review_rows[pair_id].get("decision") != "match"):
                raise ValueError("pair_review_evidence_join_invalid")
            for side in ("A", "B"):
                encoded = html_rows[pair_id][side].split(",", 1)[1]
                png = base64.b64decode(encoded, validate=True)
                if _sha(png) != package_rows[pair_id][side + "_sha256"]:
                    raise ValueError("pair_package_html_image_hash_mismatch")
                try:
                    with Image.open(io.BytesIO(png)) as image:
                        image.load()
                        source = mapping[0][side + "_source"]
                        rgb = image.convert("RGB")
                        digest = (_rgb_digest(rgb) if source == "screenshot_crop"
                                  else _sha(rgb.tobytes()))
                except (OSError, ValueError):
                    raise ValueError("pair_review_image_invalid") from None
                expected_rgb = (row["screenshot_crop_rgb_sha256"]
                                if source == "screenshot_crop"
                                else row["decoder_rgb_sha256"])
                if digest != expected_rgb:
                    raise ValueError("pair_review_image_evidence_mismatch_" + side + "_" +
                                     mapping[0][side + "_source"])

    if seen != set(expected_anchors):
        raise ValueError("evidence_anchor_set_incomplete")
    _verify_private_snapshot_still_current(
        screenshot_dir, screenshot_snapshot_names, screenshot_identities,
        screenshot_directory_identity, label="screenshot")
    _verify_private_snapshot_still_current(
        media_dir, expected_media_names, media_identities,
        media_directory_identity, label="staged_media")
    if evidence_version in {2, 3}:
        _verify_private_snapshot_still_current(
            crop_dir, crop_snapshot_names, crop_identities,
            crop_directory_identity, label="screenshot")
    pass_status = all(decision == "match" for decision in decisions)
    status = (
        "candidate_signed_evidence_bytes_consistent"
        if pass_status and evidence_version == 3 else
        "legacy_v2_unrejoined_pair_artifacts_candidate"
        if pass_status and evidence_version == 2 else
        "legacy_v1_unrostered_key_candidate"
        if pass_status else
        "blocked_correspondence_not_all_match"
    )
    return {
        "status": status,
        "scope": "private_three_video_clock_frame_correspondence_audit",
        "requested_records": 3,
        "requested_anchors": len(expected_anchors),
        "verified_anchors": len(seen),
        "match_decisions": decisions.count("match"),
        "ambiguous_decisions": decisions.count("ambiguous"),
        "mismatch_decisions": decisions.count("mismatch"),
        "screenshot_and_decoder_bytes_checked": True,
        "reviewer_signature_verified": True,
        "frame_reviewer_role_key_verified": evidence_version in {2, 3},
        "capture_operator_signature_verified": evidence_version in {2, 3},
        "capture_operator_human_identity_verified": False,
        "capture_operator_frame_correspondence_reviewed": False,
        "reviewer_attestations_present": True,
        "joint_freeze_status": "not_jointly_frozen",
        "reviewer_key_trust_status": (
            "root_signed_frame_reviewer_role_key_match_root_pin_externality_unverified"
            if evidence_version in {2, 3} else
            "expected_hash_supplied_by_caller_not_joint_freeze_authenticated"),
        "execution_authorization": "not_granted",
        "clock_calibration_status": "not_evaluated",
        "limitations": [
            "signature verification authenticates the selected key only when its hash is frozen independently",
            "the tool verifies recorded correspondence decisions but does not itself inspect or compare displayed imagery",
            "the 250 ms screenshot-to-event proximity threshold is a candidate and must be jointly frozen before use",
            "all browser-clock, protocol-freeze, target, and production-scoring gates remain separate",
        ],
    }


def _read_private(path: Path, *, max_bytes: int = 20_000_000) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size < 1 or before.st_size > max_bytes):
            raise ValueError("private_input_file_invalid")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining or os.read(fd, 1):
            raise ValueError("private_input_file_changed_during_read")
        after = os.fstat(fd)
        if (after.st_size != before.st_size or after.st_ino != before.st_ino
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns):
            raise ValueError("private_input_file_changed_during_read")
        return b"".join(chunks)
    finally:
        os.close(fd)


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
    parser.add_argument("--capture-operator-attestation", type=Path)
    parser.add_argument("--crop-map", type=Path)
    parser.add_argument("--crop-crosswalk", type=Path)
    parser.add_argument("--crop-dir", type=Path)
    parser.add_argument("--integrity-roster", type=Path)
    parser.add_argument("--release-roster", type=Path)
    parser.add_argument("--trust-root-public-key", type=Path)
    parser.add_argument("--pair-package", type=Path)
    parser.add_argument("--pair-identity-map", type=Path)
    parser.add_argument("--pair-review-export", type=Path)
    parser.add_argument("--pair-reviewer-html", type=Path)
    parser.add_argument("--expected-trust-root-der-sha256")
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
            capture_operator_attestation_raw=(
                _read_private(args.capture_operator_attestation)
                if args.capture_operator_attestation else None),
            crop_map_raw=_read_private(args.crop_map) if args.crop_map else None,
            crop_crosswalk_raw=_read_private(args.crop_crosswalk) if args.crop_crosswalk else None,
            integrity_roster_raw=_read_private(args.integrity_roster) if args.integrity_roster else None,
            release_roster_raw=_read_private(args.release_roster) if args.release_roster else None,
            root_public_key_raw=_read_private(args.trust_root_public_key)
                if args.trust_root_public_key else None,
            expected_root_public_der_sha256=args.expected_trust_root_der_sha256,
            crop_dir=args.crop_dir,
            pair_package_raw=_read_private(args.pair_package)
                if args.pair_package else None,
            pair_identity_map_raw=_read_private(args.pair_identity_map)
                if args.pair_identity_map else None,
            pair_review_export_raw=_read_private(args.pair_review_export)
                if args.pair_review_export else None,
            pair_reviewer_html_raw=_read_private(args.pair_reviewer_html, max_bytes=50_000_000)
                if args.pair_reviewer_html else None,
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
