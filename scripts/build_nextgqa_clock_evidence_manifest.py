#!/usr/bin/env python3
"""Join private clock evidence artifacts and sign a non-authorizing candidate.

This offline builder does not inspect videos or decide image correspondence.
It joins the candidate plan, raw observations, OS capture receipt, crop and
decoder crosswalks, blinded review export, and the custodian identity map. A
human evidence reader must supply an existing Ed25519 private key and explicit
attestations. The result remains a candidate and never authorizes calibration
or scoring.
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
import shutil
import stat
import subprocess
from typing import Any

from PIL import Image
from build_blinded_frame_pair_review import audit_export_bytes
from prepare_html5_clock_calibration_inputs import prepare_capture_bundle
from nextgqa_capture_operator_attestation import verify_attestation
from audit_nextgqa_review_release_order import _validate_release_roster
from nextgqa_pre_review_integrity_receipt import ReceiptError, _public_der, _verify_signature

TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{24}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ANCHORS = {"fit_begin", "heldout_quarter", "fit_midpoint",
           "heldout_three_quarter", "fit_end"}
EVIDENCE_KEYS = {
    "review_token", "anchor_id", "capture_event_sha256", "screenshot_file",
    "screenshot_sha256", "screenshot_captured_at_utc", "viewport_bbox_xywh",
    "screenshot_crop_rgb_sha256", "decoder_frame_index", "decoder_pts_seconds",
    "decoder_rgb_sha256", "correspondence_decision",
}
EVIDENCE_V2_KEYS = {
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
    "created_at_utc", "records",
}
EVIDENCE_V3_KEYS = EVIDENCE_V2_KEYS | {
    "pair_package_sha256", "pair_identity_map_sha256",
    "pair_review_export_sha256", "pair_reviewer_html_sha256",
    "pair_package_nonce",
}
EVIDENCE_V3_RECORD_KEYS = EVIDENCE_KEYS | {"pair_id"}
MAX_FILE = 20_000_000
MAX_CAPTURE_ARTIFACT = 40_000_000
EVIDENCE_DOMAIN_V2 = b"nextgqa_clock_frame_evidence_manifest_v2\0"
EVIDENCE_DOMAIN_V3 = b"nextgqa_clock_frame_evidence_manifest_v3\0"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _json(raw: bytes, label: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(label + "_duplicate_key")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError(label + "_nonfinite")))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(label + "_json_invalid") from None


def _read_private(path: Path, label: str, *, exact_mode: int = 0o600,
                  max_bytes: int = MAX_FILE) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != exact_mode
                or info.st_size < 1 or info.st_size > max_bytes):
            raise ValueError(label + "_file_invalid")
        chunks = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) != info.st_size:
            raise ValueError(label + "_file_changed_or_size_invalid")
        return raw
    finally:
        os.close(fd)


def _private_dir(path: Path, label: str, *, exact: bool = True) -> None:
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or (mode != 0o700 if exact else bool(mode & 0o077))):
        raise ValueError(label + "_directory_invalid")


def _utc(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(label + "_timestamp_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(label + "_timestamp_invalid") from None
    if parsed.tzinfo is None:
        raise ValueError(label + "_timestamp_invalid")


def _rgb_hash(raw_png: bytes) -> str:
    try:
        with Image.open(io.BytesIO(raw_png)) as image:
            image.load()
            rgb = image.convert("RGB")
            return _sha(f"{rgb.width}x{rgb.height}:".encode("ascii") + rgb.tobytes())
    except (OSError, ValueError):
        raise ValueError("screenshot_crop_png_invalid") from None


def _map_rows(rows: Any, label: str, key_fields=("review_token", "anchor_id")) -> dict:
    if not isinstance(rows, list) or len(rows) != 15:
        raise ValueError(label + "_row_count_invalid")
    mapped = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(label + "_row_invalid")
        token, anchor = (row.get(field) for field in key_fields)
        key = (token, anchor)
        if (not isinstance(token, str) or not TOKEN_RE.fullmatch(token)
                or not isinstance(anchor, str) or anchor not in ANCHORS or key in mapped):
            raise ValueError(label + "_identity_duplicate_or_invalid")
        mapped[key] = row
    if {anchor for _, anchor in mapped} != ANCHORS or len({t for t, _ in mapped}) != 3:
        raise ValueError(label + "_coverage_invalid")
    return mapped


def _leaf(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value or Path(value).name != value
            or value in {".", ".."}):
        raise ValueError(label + "_filename_invalid")
    return value


def _open_absolute_directory(directory: Path) -> int:
    if not directory.is_absolute() or any(part in {".", ".."} for part in directory.parts):
        raise ValueError("artifact_directory_path_invalid")
    flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    current = os.open("/", flags)
    try:
        for part in directory.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        return current
    except Exception:
        os.close(current)
        raise


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_dir_leaf(directory_fd: int, name: str, label: str) -> tuple[bytes, tuple]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or not 0 < before.st_size <= MAX_CAPTURE_ARTIFACT):
            raise ValueError(label + "_artifact_file_invalid")
        chunks, remaining = [], before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(label + "_artifact_changed_during_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ValueError(label + "_artifact_changed_during_read")
        after = os.fstat(fd)
        if _stat_identity(before) != _stat_identity(after):
            raise ValueError(label + "_artifact_changed_during_read")
        return b"".join(chunks), _stat_identity(before)
    finally:
        os.close(fd)


def _artifact_directory(directory: Path, expected_names: set[str], label: str) -> dict[str, bytes]:
    if (not isinstance(expected_names, set) or not expected_names
            or any(not isinstance(name, str) or Path(name).name != name
                   or name in {".", ".."} for name in expected_names)):
        raise ValueError(label + "_artifact_set_invalid")
    _private_dir(directory, label)
    try:
        directory_fd = _open_absolute_directory(directory)
    except (OSError, ValueError):
        raise ValueError(label + "_directory_unavailable_or_symlink") from None
    try:
        before_dir = os.fstat(directory_fd)
        if (not stat.S_ISDIR(before_dir.st_mode) or before_dir.st_uid != os.geteuid()
                or stat.S_IMODE(before_dir.st_mode) != 0o700):
            raise ValueError(label + "_directory_invalid")
        if set(os.listdir(directory_fd)) != expected_names:
            raise ValueError(label + "_artifact_set_invalid")
        files, identities = {}, {}
        for name in sorted(expected_names):
            files[name], identities[name] = _read_dir_leaf(directory_fd, name, label)
        if set(os.listdir(directory_fd)) != expected_names:
            raise ValueError(label + "_directory_changed_during_read")
        after_dir = os.fstat(directory_fd)
        dir_identity = (before_dir.st_dev, before_dir.st_ino, before_dir.st_mode,
                        before_dir.st_uid)
        if dir_identity != (after_dir.st_dev, after_dir.st_ino, after_dir.st_mode,
                            after_dir.st_uid):
            raise ValueError(label + "_directory_changed_during_read")
        try:
            path_fd = _open_absolute_directory(directory)
        except (OSError, ValueError):
            raise ValueError(label + "_directory_path_changed_during_read") from None
        try:
            path_info = os.fstat(path_fd)
            if dir_identity != (path_info.st_dev, path_info.st_ino,
                                path_info.st_mode, path_info.st_uid):
                raise ValueError(label + "_directory_path_changed_during_read")
            if set(os.listdir(path_fd)) != expected_names:
                raise ValueError(label + "_directory_changed_during_read")
            for name, identity in identities.items():
                try:
                    check_fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                                       | getattr(os, "O_CLOEXEC", 0), dir_fd=path_fd)
                except OSError:
                    raise ValueError(label + "_artifact_path_changed_during_read") from None
                try:
                    if _stat_identity(os.fstat(check_fd)) != identity:
                        raise ValueError(label + "_artifact_path_changed_during_read")
                finally:
                    os.close(check_fd)
        finally:
            os.close(path_fd)
        return files
    finally:
        os.close(directory_fd)


def _validate_and_build(*, files: dict[str, bytes], crop_dir: Path,
                        screenshot_dir: Path,
                        public_key_raw: bytes, expected_public_key_sha256: str,
                        expected_capture_helper_sha256: str,
                        root_public_key_raw: bytes,
                        expected_root_public_der_sha256: str,
                        reviewer_pseudonym: str, created_at_utc: str,
                        attest_independence: bool,
                        attest_visual_comparison: bool) -> dict:
    parsed = {name: _json(files[name], name) for name in (
        "plan", "provenance", "observations", "observation_index",
        "capture_receipt", "crop_crosswalk", "decoder_crosswalk",
        "pair_package", "identity_map", "review_export",
        "capture_operator_attestation")}
    plan = parsed["plan"]
    provenance = parsed["provenance"]
    observations = parsed["observations"]
    obs_index = parsed["observation_index"]
    receipt = parsed["capture_receipt"]
    crop = parsed["crop_crosswalk"]
    decoder = parsed["decoder_crosswalk"]
    package = parsed["pair_package"]
    identity = parsed["identity_map"]
    review = parsed["review_export"]
    hashes = {name: _sha(raw) for name, raw in files.items()}
    package = parsed["pair_package"]
    identity = parsed["identity_map"]
    review = parsed["review_export"]

    if not _is_sha(expected_public_key_sha256) or _sha(public_key_raw) != expected_public_key_sha256:
        raise ValueError("caller_pinned_public_key_hash_mismatch")
    if (not isinstance(plan, dict) or plan.get("purpose") != "predeclared_html5_clock_pts_anchor_plan"
            or plan.get("status") != "predeclared_frozen" or plan.get("execution_authorization") != "not_granted"
            or not isinstance(plan.get("records"), list) or len(plan["records"]) != 3):
        raise ValueError("plan_candidate_schema_invalid")
    plan_sha, prov_sha = hashes["plan"], hashes["provenance"]
    if (not isinstance(provenance, dict)
            or provenance.get("purpose") != "private_clock_anchor_plan_candidate_provenance"
            or provenance.get("status") != "candidate_not_jointly_frozen"
            or provenance.get("execution_authorization") != "not_granted"
            or provenance.get("plan_sha256") != plan_sha
            or provenance.get("joint_freeze_receipt_sha256") is not None):
        raise ValueError("plan_provenance_candidate_binding_invalid")
    provenance_rows = provenance.get("records")
    if not isinstance(provenance_rows, list) or len(provenance_rows) != 3:
        raise ValueError("plan_provenance_records_invalid")
    decoded_counts = {}
    for row in provenance_rows:
        if (not isinstance(row, dict) or not isinstance(row.get("review_token"), str)
                or row["review_token"] in decoded_counts
                or type(row.get("decoded_frame_count")) is not int
                or row["decoded_frame_count"] < 1):
            raise ValueError("plan_provenance_records_invalid")
        decoded_counts[row["review_token"]] = row["decoded_frame_count"]

    expected = set()
    plan_media = {}
    plan_pts = {}
    for row in plan["records"]:
        if not isinstance(row, dict) or not isinstance(row.get("review_token"), str) or not TOKEN_RE.fullmatch(row["review_token"]):
            raise ValueError("plan_token_invalid")
        token = row["review_token"]
        if token in plan_media or not _is_sha(row.get("source_media_sha256")):
            raise ValueError("plan_media_identity_invalid")
        plan_media[token] = row["source_media_sha256"]
        if any(not isinstance(row.get(k), list) for k in ("fit_anchors", "heldout_anchors")):
            raise ValueError("plan_anchors_invalid")
        fit_anchors, heldout_anchors = row["fit_anchors"], row["heldout_anchors"]
        if not all(isinstance(anchor, dict) and isinstance(anchor.get("anchor_id"), str)
                   for anchor in fit_anchors + heldout_anchors):
            raise ValueError("plan_anchor_invalid")
        fit_ids = {a.get("anchor_id") for a in fit_anchors if isinstance(a, dict)}
        heldout_ids = {a.get("anchor_id") for a in heldout_anchors if isinstance(a, dict)}
        if (len(fit_anchors) != 3 or len(heldout_anchors) != 2
                or fit_ids != {"fit_begin", "fit_midpoint", "fit_end"}
                or heldout_ids != {"heldout_quarter", "heldout_three_quarter"}):
            raise ValueError("plan_anchor_count_invalid")
        for anchor in fit_anchors + heldout_anchors:
            if (not isinstance(anchor, dict) or not isinstance(anchor.get("anchor_id"), str)
                    or anchor.get("anchor_id") not in ANCHORS
                    or not isinstance(anchor.get("decoder_pts_seconds"), str)):
                raise ValueError("plan_anchor_invalid")
            key = (token, anchor["anchor_id"])
            if key in expected:
                raise ValueError("plan_anchor_duplicate")
            try:
                pts_value = Fraction(anchor["decoder_pts_seconds"])
            except (ValueError, ZeroDivisionError, TypeError, OverflowError):
                raise ValueError("plan_anchor_pts_invalid") from None
            if pts_value < 0:
                raise ValueError("plan_anchor_pts_invalid")
            expected.add(key)
            plan_pts[key] = pts_value
    if len(expected) != 15 or len({t for t, _ in expected}) != 3:
        raise ValueError("plan_coverage_invalid")

    if (not isinstance(observations, dict) or observations.get("schema_version") != 3
            or observations.get("purpose") != "local_unvalidated_html5_player_clock_observations"
            or observations.get("status") != "raw_player_clock_observations_only"
            or observations.get("execution_authorization") != "not_granted"):
        raise ValueError("observations_schema_invalid")
    obs_records = {}
    if not isinstance(observations.get("records"), list) or len(observations["records"]) != 3:
        raise ValueError("observations_record_count_invalid")
    for row in observations["records"]:
        if not isinstance(row, dict) or not isinstance(row.get("review_token"), str) or row["review_token"] in obs_records:
            raise ValueError("observations_record_duplicate_or_invalid")
        events = _map_events(row.get("events"), row["review_token"])
        obs_records[row["review_token"]] = events
    if set(obs_records) != {t for t, _ in expected}:
        raise ValueError("observations_token_set_mismatch")
    if (not isinstance(obs_index, dict)
            or obs_index.get("purpose") != "unvalidated_html5_clock_observation_integrity_index"
            or obs_index.get("payload_sha256") != hashes["observations"]
            or obs_index.get("status") != "raw_observations_only"
            or obs_index.get("execution_authorization") != "not_granted"):
        raise ValueError("observation_index_binding_invalid")
    prepare_capture_bundle(files["observations"], files["observation_index"],
                           files["plan"], plan_sha, files["provenance"], prov_sha)

    if (not isinstance(receipt, dict) or set(receipt) != {
                "schema_version", "purpose", "status", "execution_authorization",
                "plan_sha256", "plan_provenance_sha256", "capture_method",
                "capture_helper_file_sha256", "created_at_utc", "records"}
            or receipt.get("schema_version") != 1
            or receipt.get("purpose") != "private_windows_compositor_screenshot_capture_receipt_candidate"
            or receipt.get("status") != "self_reported_capture_receipt_not_independently_authenticated"
            or receipt.get("execution_authorization") != "not_granted"
            or receipt.get("plan_sha256") != plan_sha
            or receipt.get("plan_provenance_sha256") != prov_sha
            or receipt.get("capture_method") != "windows_gdi_copyfromscreen_foreground_window_v1"):
        raise ValueError("capture_receipt_binding_invalid")
    _utc(receipt.get("created_at_utc"), "capture_receipt")
    receipt_rows = _map_rows(receipt.get("records"), "capture_receipt")
    if (not isinstance(expected_capture_helper_sha256, str)
            or not _is_sha(expected_capture_helper_sha256)
            or receipt.get("capture_helper_file_sha256") != expected_capture_helper_sha256):
        raise ValueError("capture_helper_hash_pin_mismatch")
    receipt_record_keys = {
        "review_token", "anchor_id", "screenshot_file", "screenshot_sha256",
        "screenshot_captured_at_utc", "capture_method", "foreground_process",
        "foreground_pid", "foreground_window_title", "foreground_window_rect_xywh",
        "virtual_screen_rect_xywh", "hotkey_message_tick_ms", "capture_complete_tick_ms",
        "hotkey_to_capture_ms",
    }
    for row in receipt_rows.values():
        if set(row) != receipt_record_keys:
            raise ValueError("capture_receipt_record_schema_invalid")
        _leaf(row.get("screenshot_file"), "screenshot")
        if not _is_sha(row.get("screenshot_sha256")):
            raise ValueError("capture_receipt_screenshot_hash_invalid")
        _utc(row.get("screenshot_captured_at_utc"), "screenshot")
        if (row.get("capture_method") != receipt["capture_method"]
                or row.get("foreground_process") != "msedge"
                or type(row.get("foreground_pid")) is not int or row["foreground_pid"] <= 0
                or not isinstance(row.get("foreground_window_title"), str)
                or "Private offline video review" not in row["foreground_window_title"]):
            raise ValueError("capture_receipt_window_invalid")
        ticks = (row.get("hotkey_message_tick_ms"), row.get("capture_complete_tick_ms"),
                 row.get("hotkey_to_capture_ms"))
        if (any(type(value) is not int or value < 0 or value > 0xFFFFFFFF for value in ticks)
                or ticks[2] != ((ticks[1] - ticks[0]) & 0xFFFFFFFF) or ticks[2] > 250):
            raise ValueError("capture_receipt_timing_invalid")
        window, virtual = row.get("foreground_window_rect_xywh"), row.get("virtual_screen_rect_xywh")
        if (not isinstance(window, list) or len(window) != 4
                or any(type(value) is not int for value in window)
                or not isinstance(virtual, list) or len(virtual) != 4
                or any(type(value) is not int for value in virtual)):
            raise ValueError("capture_receipt_geometry_invalid")
        wx, wy, ww, wh = window
        vx, vy, vw, vh = virtual
        if (ww < 100 or wh < 100 or vw < 1 or vh < 1
                or wx < vx or wy < vy or wx + ww > vx + vw or wy + wh > vy + vh):
            raise ValueError("capture_receipt_geometry_invalid")

    if (not isinstance(crop, dict) or crop.get("purpose") != "private_candidate_screenshot_crop_crosswalk"
            or crop.get("status") != "candidate_only" or crop.get("execution_authorization") != "not_granted"
            or crop.get("plan_sha256") != plan_sha or crop.get("plan_provenance_sha256") != prov_sha
            or crop.get("capture_receipt_sha256") != hashes["capture_receipt"]):
        raise ValueError("crop_crosswalk_binding_invalid")
    crop_rows = _map_rows(crop.get("records"), "crop_crosswalk")
    for row in crop_rows.values():
        bbox = row.get("bbox_xywh")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(type(n) is not int for n in bbox)
                or bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 1 or bbox[3] < 1
                or not _is_sha(row.get("crop_sha256"))):
            raise ValueError("crop_crosswalk_record_invalid")
        _leaf(row.get("crop_file"), "crop")
        _leaf(row.get("screenshot_file"), "screenshot")
    if (not isinstance(decoder, dict) or decoder.get("purpose") != "private_candidate_decoder_anchor_png_crosswalk"
            or decoder.get("status") != "candidate_artifact_export_only"
            or decoder.get("execution_authorization") != "not_granted"
            or decoder.get("plan_sha256") != plan_sha or decoder.get("provenance_sha256") != prov_sha
            or decoder.get("effective_anchors") != 15):
        raise ValueError("decoder_crosswalk_binding_invalid")
    decoder_rows = _map_rows(decoder.get("records"), "decoder_crosswalk")
    for row in decoder_rows.values():
        if (type(row.get("decoder_frame_index")) is not int or row["decoder_frame_index"] < 0
                or not isinstance(row.get("decoder_relative_pts_seconds"), str)
                or not _is_sha(row.get("rgb_sha256"))
                or not _is_sha(row.get("png_sha256"))):
            raise ValueError("decoder_crosswalk_record_invalid")
        _leaf(row.get("png_file"), "decoder_frame")
    if set(receipt_rows) != expected or set(crop_rows) != expected or set(decoder_rows) != expected:
        raise ValueError("crosswalk_anchor_set_mismatch")

    screenshot_names = {row["screenshot_file"] for row in receipt_rows.values()}
    crop_names = {row["crop_file"] for row in crop_rows.values()}
    screenshot_files = _artifact_directory(screenshot_dir, screenshot_names, "screenshots")
    crop_files = _artifact_directory(crop_dir, crop_names, "crops")
    operator_inputs = {
        "plan_raw": files["plan"], "expected_plan_sha256": plan_sha,
        "plan_provenance_raw": files["provenance"],
        "expected_plan_provenance_sha256": prov_sha,
        "observations_raw": files["observations"],
        "observation_index_raw": files["observation_index"],
        "capture_receipt_raw": files["capture_receipt"],
        "crop_map_raw": files["crop_map"],
        "crop_crosswalk_raw": files["crop_crosswalk"],
        "screenshot_files": screenshot_files, "crop_files": crop_files,
        "integrity_roster_raw": files["integrity_roster"],
        "release_roster_raw": files["release_roster"],
        "root_public_key_pem": root_public_key_raw,
        "expected_root_public_der_sha256": expected_root_public_der_sha256,
    }
    try:
        operator_report = verify_attestation(
            attestation_raw=files["capture_operator_attestation"], **operator_inputs)
    except Exception as exc:
        code = str(exc)
        if not re.fullmatch(r"[a-zA-Z0-9_:.-]{1,160}", code):
            code = "capture_operator_attestation_invalid"
        raise ValueError("capture_operator_" + code) from None
    if (operator_report.get("status") !=
            "capture_operator_key_signature_valid_candidate_artifact_bytes"
            or operator_report.get("anchor_count") != 15
            or operator_report.get("all_exact_artifact_hashes_recomputed") is not True
            or operator_report.get("human_identity_verified") is not False
            or operator_report.get("capture_process_authenticated") is not False
            or operator_report.get("frame_correspondence_reviewed") is not False
            or operator_report.get("score_ready") is not False
            or operator_report.get("model_load_permitted") is not False
            or operator_report.get("execution_authorization") != "not_granted"):
        raise ValueError("capture_operator_report_authority_invalid")

    try:
        release_roster, role_public_keys = _validate_release_roster(
            files["release_roster"], root_public_key_raw,
            expected_root_public_der_sha256, files["integrity_roster"])
    except ReceiptError as exc:
        raise ValueError("release_roster_invalid:" + str(exc)) from None
    frame_key_raw = role_public_keys.get("frame_reviewer")
    if (frame_key_raw is None
            or _public_der(public_key_raw) != _public_der(frame_key_raw)):
        raise ValueError("reviewer_key_not_rostered_frame_reviewer")
    frame_entry = next((entry for entry in release_roster["keys"]
                        if entry.get("role") == "frame_reviewer"), None)
    if not isinstance(frame_entry, dict) or not isinstance(frame_entry.get("key_id"), str):
        raise ValueError("release_roster_frame_reviewer_missing")

    if (not isinstance(package, dict) or package.get("purpose") != "private_blinded_frame_pair_review_package"
            or package.get("status") != "not_authorizing" or package.get("execution_authorization") != "not_granted"
            or package.get("plan_sha256") != plan_sha or package.get("pair_count") != 15):
        raise ValueError("pair_package_binding_invalid")
    package_sha = hashes["pair_package"]
    if (not isinstance(identity, dict) or identity.get("purpose") != "private_blinded_frame_pair_identity_map"
            or identity.get("status") != "custodian_only"
            or identity.get("package_manifest_sha256") != package_sha):
        raise ValueError("identity_map_binding_invalid")
    identity_top_keys = {
        frozenset({"schema_version", "purpose", "status",
                   "package_manifest_sha256", "records"}),
        frozenset({"schema_version", "purpose", "status",
                   "package_manifest_sha256", "input_manifest_sha256", "records"}),
    }
    if (frozenset(identity) not in identity_top_keys
            or type(identity.get("schema_version")) is not int
            or identity.get("schema_version") != 1
            or ("input_manifest_sha256" in identity
                and not _is_sha(identity["input_manifest_sha256"]))):
        raise ValueError("identity_map_schema_invalid")
    if (not isinstance(review, dict) or review.get("purpose") != "private_blinded_frame_pair_review_export"
            or review.get("status") != "not_authorizing" or review.get("execution_authorization") != "not_granted"
            or review.get("package_manifest_sha256") != package_sha
            or review.get("package_nonce") != package.get("package_nonce")
            or review.get("reviewer_provenance_status") != "self_reported_unauthenticated"
            or review.get("signature_status") != "none"):
        raise ValueError("review_export_binding_invalid")

    package_pairs = {}
    if not isinstance(package.get("pairs"), list) or len(package["pairs"]) != 15:
        raise ValueError("package_pair_count_invalid")
    for row in package["pairs"]:
        if not isinstance(row, dict):
            raise ValueError("package_pair_invalid")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not TOKEN_RE.fullmatch(pair_id) or pair_id in package_pairs:
            raise ValueError("package_pair_duplicate")
        package_pairs[pair_id] = row
    identity_by_key = {}
    identity_by_pair = {}
    if not isinstance(identity.get("records"), list) or len(identity["records"]) != 15:
        raise ValueError("identity_record_count_invalid")
    for row in identity.get("records", []):
        if (not isinstance(row, dict) or set(row) != {
                "pair_id", "review_token", "anchor_id", "A_source", "B_source"}):
            raise ValueError("identity_row_schema_invalid")
        pair_id, token, anchor = row.get("pair_id"), row.get("review_token"), row.get("anchor_id")
        key = (token, anchor)
        if (not isinstance(pair_id, str) or not TOKEN_RE.fullmatch(pair_id)
                or not isinstance(token, str) or not isinstance(anchor, str)
                or pair_id in identity_by_pair or key in identity_by_key or key not in expected):
            raise ValueError("identity_pair_ambiguous_or_invalid")
        if row.get("A_source") not in {"screenshot_crop", "decoder_frame"} or row.get("B_source") != ("decoder_frame" if row["A_source"] == "screenshot_crop" else "screenshot_crop"):
            raise ValueError("identity_source_map_invalid")
        identity_by_pair[pair_id] = row
        identity_by_key[key] = row
    if set(identity_by_key) != expected or set(identity_by_pair) != set(package_pairs):
        raise ValueError("identity_pair_coverage_invalid")

    review_by_pair = {}
    if not isinstance(review.get("records"), list) or len(review["records"]) != 15:
        raise ValueError("review_record_count_invalid")
    for row in review.get("records", []):
        if (not isinstance(row, dict) or not isinstance(row.get("pair_id"), str)
                or not TOKEN_RE.fullmatch(row["pair_id"]) or row.get("pair_id") in review_by_pair):
            raise ValueError("review_pair_duplicate_or_invalid")
        if row.get("crop_purity_attestation") is not True or row.get("decision") != "match":
            raise ValueError("review_not_all_pure_and_matching")
        review_by_pair[row["pair_id"]] = row
    if set(review_by_pair) != set(package_pairs):
        raise ValueError("review_pair_coverage_invalid")

    pair_audit = audit_export_bytes(
        export_raw=files["review_export"],
        package_raw=files["pair_package"],
        reviewer_html_raw=files["pair_reviewer_html"],
    )
    if pair_audit["status"] != "candidate_review_complete_not_authorizing":
        raise ValueError("review_export_not_complete_pure_and_matching")
    package = pair_audit["package_manifest"]
    review = pair_audit["review_export"]

    if (not isinstance(reviewer_pseudonym, str) or not reviewer_pseudonym.strip()
            or len(reviewer_pseudonym) > 80
            or reviewer_pseudonym != review.get("reviewer_pseudonym")):
        raise ValueError("reviewer_pseudonym_invalid")
    if attest_independence is not True or attest_visual_comparison is not True:
        raise ValueError("reader_attestations_required")
    _utc(created_at_utc, "created_at")

    evidence_rows = []
    for key in sorted(expected):
        token, anchor = key
        cap = receipt_rows[key]
        crop_row = crop_rows[key]
        dec = decoder_rows[key]
        ident = identity_by_key[key]
        event = obs_records[token][anchor]
        pair = package_pairs[ident["pair_id"]]
        try:
            decoder_pts = Fraction(dec["decoder_relative_pts_seconds"])
        except (ValueError, ZeroDivisionError, TypeError, OverflowError):
            raise ValueError("decoder_crosswalk_pts_invalid") from None
        if (decoder_pts != plan_pts[key]
                or dec["decoder_relative_pts_seconds"] != str(plan_pts[key])):
            raise ValueError("decoder_crosswalk_plan_pts_mismatch")
        if (crop_row.get("screenshot_file") != cap.get("screenshot_file")
                or crop_row.get("screenshot_sha256") != cap.get("screenshot_sha256")
                or dec.get("decoder_relative_pts_seconds") is None
                or dec.get("decoder_frame_index") is None
                or dec["decoder_frame_index"] >= decoded_counts[token]
                or dec.get("source_media_sha256") != plan_media[token]
                or dec.get("rgb_sha256") is None):
            raise ValueError("crosswalk_capture_or_decoder_join_invalid")
        crop_png = crop_files[crop_row["crop_file"]]
        if _sha(crop_png) != crop_row.get("crop_sha256"):
            raise ValueError("screenshot_crop_bytes_mismatch")
        source_hashes = {"A": pair.get("A_sha256"), "B": pair.get("B_sha256")}
        crop_side = "A" if ident["A_source"] == "screenshot_crop" else "B"
        decoder_side = "B" if crop_side == "A" else "A"
        if (source_hashes[crop_side] != crop_row.get("crop_sha256")
                or source_hashes[decoder_side] != dec.get("png_sha256")):
            raise ValueError("blinded_pair_image_hash_join_invalid")
        if review_by_pair[ident["pair_id"]]["decision"] != "match":
            raise ValueError("review_correspondence_not_match")
        bbox = crop_row.get("bbox_xywh")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("crop_bbox_invalid")
        record = {
            "review_token": token, "anchor_id": anchor,
            "pair_id": ident["pair_id"],
            "capture_event_sha256": _sha(_canonical(event)),
            "screenshot_file": cap["screenshot_file"],
            "screenshot_sha256": cap["screenshot_sha256"],
            "screenshot_captured_at_utc": cap["screenshot_captured_at_utc"],
            "viewport_bbox_xywh": bbox,
            "screenshot_crop_rgb_sha256": _rgb_hash(crop_png),
            "decoder_frame_index": dec["decoder_frame_index"],
            "decoder_pts_seconds": dec["decoder_relative_pts_seconds"],
            "decoder_rgb_sha256": dec["rgb_sha256"],
            "correspondence_decision": "match",
        }
        if set(record) != EVIDENCE_KEYS | {"pair_id"}:
            raise AssertionError("internal_evidence_schema_invalid")
        evidence_rows.append(record)

    return {
        "schema_version": 3,
        "purpose": "private_clock_frame_correspondence_reader_attestation_v3",
        "plan_sha256": plan_sha,
        "plan_provenance_sha256": prov_sha,
        "observations_sha256": hashes["observations"],
        "observation_index_sha256": hashes["observation_index"],
        "capture_receipt_sha256": hashes["capture_receipt"],
        "capture_operator_attestation_sha256": hashes["capture_operator_attestation"],
        "capture_operator_integrity_roster_sha256": _sha(files["integrity_roster"]),
        "capture_operator_release_roster_sha256": _sha(files["release_roster"]),
        "capture_operator_trust_root_der_sha256": expected_root_public_der_sha256,
        "capture_operator_key_id": operator_report["capture_operator_key_id"],
        "capture_operator_key_der_sha256": parsed["capture_operator_attestation"][
            "capture_operator_key_der_sha256"],
        "capture_operator_signature_verified": True,
        "capture_operator_human_identity_verified": False,
        "capture_operator_process_authenticated": False,
        "capture_operator_frame_correspondence_reviewed": False,
        "capture_operator_score_ready": False,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
        "reviewer_identity_verified": False,
        "frame_reviewer_key_id": frame_entry["key_id"],
        "frame_reviewer_key_der_sha256": _sha(_public_der(frame_key_raw)),
        "frame_reviewer_role_signature_verified": True,
        "pair_package_sha256": hashes["pair_package"],
        "pair_identity_map_sha256": hashes["identity_map"],
        "pair_review_export_sha256": hashes["review_export"],
        "pair_reviewer_html_sha256": hashes["pair_reviewer_html"],
        "pair_package_nonce": package["package_nonce"],
        "reviewer_public_key_sha256": _sha(public_key_raw),
        "reviewer_pseudonym": reviewer_pseudonym.strip(),
        "reviewer_role": "frame_reviewer",
        "independence_attestation": "independent_of_claim_generation_scoring_and_custody",
        "visual_inspection_attestation": "personally_compared_each_screenshot_crop_to_decoder_frame",
        "created_at_utc": created_at_utc,
        "records": evidence_rows,
    }


def verify_manifest_signature(*, manifest_raw: bytes,
                              reviewer_public_key_raw: bytes,
                              expected_reviewer_public_key_sha256: str) -> dict:
    """Verify a roster-bound reader signature; v1 remains explicitly unsupported."""
    manifest = _json(manifest_raw, "evidence_manifest")
    version = manifest.get("schema_version") if isinstance(manifest, dict) else None
    purpose = ("private_clock_frame_correspondence_reader_attestation_v2"
               if version == 2 else
               "private_clock_frame_correspondence_reader_attestation_v3")
    if (not isinstance(manifest, dict)
            or type(version) is not int or version not in {2, 3}
            or manifest.get("purpose") != purpose):
        raise ValueError("evidence_manifest_schema_invalid")
    schema_keys = EVIDENCE_V2_KEYS if version == 2 else EVIDENCE_V3_KEYS
    expected_keys = schema_keys | {"signature_ed25519_base64"}
    if set(manifest) != expected_keys:
        raise ValueError("evidence_manifest_schema_invalid")
    if (not _is_sha(expected_reviewer_public_key_sha256)
            or _sha(reviewer_public_key_raw) != expected_reviewer_public_key_sha256
            or manifest.get("reviewer_public_key_sha256") != expected_reviewer_public_key_sha256):
        raise ValueError("evidence_manifest_reviewer_key_pin_mismatch")
    if (manifest.get("capture_operator_signature_verified") is not True
            or manifest.get("capture_operator_human_identity_verified") is not False
            or manifest.get("capture_operator_process_authenticated") is not False
            or manifest.get("capture_operator_frame_correspondence_reviewed") is not False
            or manifest.get("capture_operator_score_ready") is not False
            or manifest.get("model_load_permitted") is not False
            or manifest.get("scoring_authorized") is not False
            or manifest.get("reviewer_identity_verified") is not False
            or manifest.get("reviewer_role") != "frame_reviewer"
            or not isinstance(manifest.get("frame_reviewer_key_id"), str)
            or not manifest["frame_reviewer_key_id"]
            or manifest.get("frame_reviewer_key_der_sha256") !=
                _sha(_public_der(reviewer_public_key_raw))
            or manifest.get("frame_reviewer_role_signature_verified") is not True
            or manifest.get("execution_authorization") != "not_granted"):
        raise ValueError("evidence_manifest_authority_invalid")
    for field in (
            "plan_sha256", "plan_provenance_sha256", "observations_sha256",
            "observation_index_sha256", "capture_receipt_sha256",
            "capture_operator_attestation_sha256",
            "capture_operator_integrity_roster_sha256",
            "capture_operator_release_roster_sha256",
            "capture_operator_trust_root_der_sha256",
            "capture_operator_key_der_sha256"):
        if not _is_sha(manifest.get(field)):
            raise ValueError("evidence_manifest_digest_invalid")
    rows = manifest.get("records")
    record_keys = EVIDENCE_KEYS if version == 2 else EVIDENCE_V3_RECORD_KEYS
    if (not isinstance(rows, list) or len(rows) != 15
            or any(not isinstance(row, dict) or set(row) != record_keys for row in rows)):
        raise ValueError("evidence_manifest_records_invalid")
    if version == 3 and (
            any(not _is_sha(manifest.get(field)) for field in (
                "pair_package_sha256", "pair_identity_map_sha256",
                "pair_review_export_sha256", "pair_reviewer_html_sha256"))
            or not _is_sha(manifest.get("pair_package_nonce"))
            or len({row["pair_id"] for row in rows
                    if isinstance(row.get("pair_id"), str)}) != 15
            or any(not isinstance(row.get("pair_id"), str)
                   or TOKEN_RE.fullmatch(row["pair_id"]) is None for row in rows)):
        raise ValueError("evidence_manifest_pair_review_binding_invalid")
    identities = [(row.get("review_token"), row.get("anchor_id")) for row in rows]
    if any(not isinstance(token, str) or not TOKEN_RE.fullmatch(token)
           or not isinstance(anchor, str) or anchor not in ANCHORS
           for token, anchor in identities):
        raise ValueError("evidence_manifest_records_invalid")
    if (len(set(identities)) != 15
            or {anchor for _, anchor in identities} != ANCHORS
            or len({token for token, _ in identities}) != 3
            or any(row.get("correspondence_decision") != "match"
                   for row, (token, anchor) in zip(rows, identities, strict=True))):
        raise ValueError("evidence_manifest_records_invalid")
    try:
        signature = base64.b64decode(manifest["signature_ed25519_base64"], validate=True)
    except (TypeError, ValueError):
        raise ValueError("evidence_manifest_signature_invalid") from None
    if (len(signature) != 64 or not _verify_signature(
            reviewer_public_key_raw, signature,
            (EVIDENCE_DOMAIN_V2 if version == 2 else EVIDENCE_DOMAIN_V3)
            + _canonical({k: v for k, v in manifest.items()
                                             if k != "signature_ed25519_base64"}))):
        raise ValueError("evidence_manifest_signature_invalid")
    return {
        "status": f"evidence_manifest_v{version}_signature_valid_non_authorizing",
        "verified_anchors": 15,
        "capture_operator_signature_verified": True,
        "human_identity_verified": False,
        "capture_process_authenticated": False,
        "frame_correspondence_reviewed": False,
        "score_ready": False,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
    }


def _map_events(events: Any, token: str) -> dict:
    if not isinstance(events, list) or len(events) != 5:
        raise ValueError("observation_event_count_invalid")
    mapped = {}
    sequences = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("observation_event_invalid")
        anchor = event.get("anchor_id")
        if (not isinstance(anchor, str) or anchor not in ANCHORS or anchor in mapped
                or event.get("frame_match_status") != "fresh_click_bound_callback_unverified_decoder"
                or type(event.get("sequence")) is not int):
            raise ValueError("observation_event_duplicate_or_invalid")
        _utc(event.get("captured_at_utc"), "observation_event")
        sequences.append(event["sequence"])
        mapped[anchor] = event
    if set(mapped) != ANCHORS or sequences != [1, 2, 3, 4, 5]:
        raise ValueError("observation_event_coverage_invalid")
    return mapped


def _load_public_der(public_path: Path) -> bytes:
    fd = os.open(public_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("reviewer_public_key_file_invalid")
        proc_path = f"/proc/self/fd/{fd}"
        result = subprocess.run(["openssl", "pkey", "-pubin", "-in", proc_path,
                                 "-outform", "DER"], pass_fds=(fd,), capture_output=True,
                                timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("openssl_public_key_parse_failed") from None
    finally:
        os.close(fd)
    if result.returncode != 0 or len(result.stdout) != 44 or not result.stdout.startswith(bytes.fromhex("302a300506032b6570032100")):
        raise ValueError("reviewer_public_key_not_ed25519")
    return result.stdout


def _sign(payload: bytes, private_key_path: Path, public_key_der: bytes) -> bytes:
    if not hasattr(os, "memfd_create"):
        raise ValueError("anonymous_payload_fd_unavailable")
    fd = os.open(private_key_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    payload_fd = os.memfd_create("clock-evidence-signing-payload", 0)
    try:
        os.write(payload_fd, payload)
        os.lseek(payload_fd, 0, os.SEEK_SET)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("reviewer_private_key_file_invalid")
        if info.st_size < 1 or info.st_size > MAX_FILE:
            raise ValueError("reviewer_private_key_file_invalid")
        proc_path = f"/proc/self/fd/{fd}"
        os.lseek(fd, 0, os.SEEK_SET)
        derived = subprocess.run(["openssl", "pkey", "-in", proc_path, "-passin", "pass:",
                                  "-pubout", "-outform", "DER"],
                                 pass_fds=(fd,), capture_output=True, timeout=10, check=False)
        if derived.returncode != 0 or derived.stdout != public_key_der:
            raise ValueError("reviewer_private_public_key_mismatch")
        os.lseek(fd, 0, os.SEEK_SET)
        signed = subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", proc_path,
                                 "-passin", "pass:", "-rawin", "-in",
                                 f"/proc/self/fd/{payload_fd}"],
                                pass_fds=(fd, payload_fd), capture_output=True,
                                timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("openssl_ed25519_sign_failed") from None
    finally:
        os.close(fd)
        os.close(payload_fd)
    if signed.returncode != 0 or len(signed.stdout) != 64:
        raise ValueError("openssl_ed25519_sign_failed")
    return signed.stdout


def build_manifest(*, input_files: dict[str, bytes], crop_dir: Path,
                   screenshot_dir: Path,
                   private_key_path: Path, public_key_path: Path,
                   expected_public_key_sha256: str, expected_capture_helper_sha256: str,
                   root_public_key_raw: bytes,
                   expected_root_public_der_sha256: str,
                   reviewer_pseudonym: str,
                   created_at_utc: str, attest_independence: bool,
                   attest_visual_comparison: bool) -> dict:
    _private_dir(private_key_path.parent, "private_key_parent", exact=False)
    _private_dir(public_key_path.parent, "public_key_parent", exact=False)
    _private_dir(crop_dir, "crop_directory")
    public_raw = _read_private(public_key_path, "reviewer_public_key")
    if _sha(public_raw) != expected_public_key_sha256:
        raise ValueError("caller_pinned_public_key_hash_mismatch")
    public_der = _load_public_der(public_key_path)
    evidence = _validate_and_build(files=input_files, crop_dir=crop_dir,
                                   screenshot_dir=screenshot_dir,
                                   public_key_raw=public_raw,
                                   expected_public_key_sha256=expected_public_key_sha256,
                                   expected_capture_helper_sha256=expected_capture_helper_sha256,
                                   root_public_key_raw=root_public_key_raw,
                                   expected_root_public_der_sha256=expected_root_public_der_sha256,
                                   reviewer_pseudonym=reviewer_pseudonym,
                                   created_at_utc=created_at_utc,
                                   attest_independence=attest_independence,
                                   attest_visual_comparison=attest_visual_comparison)
    signature = _sign(EVIDENCE_DOMAIN_V3 + _canonical(evidence), private_key_path, public_der)
    evidence["signature_ed25519_base64"] = base64.b64encode(signature).decode("ascii")
    return evidence


INPUTS = (
    "plan", "provenance", "observations", "observation_index", "capture_receipt",
    "crop_crosswalk", "decoder_crosswalk", "pair_package", "identity_map", "review_export",
    "crop_map", "capture_operator_attestation", "integrity_roster", "release_roster",
    "pair_reviewer_html",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--crop-dir", required=True, type=Path)
    parser.add_argument("--screenshot-dir", required=True, type=Path)
    parser.add_argument("--trust-root-public-key", required=True, type=Path)
    parser.add_argument("--expected-trust-root-der-sha256", required=True)
    parser.add_argument("--reviewer-private-key", required=True, type=Path)
    parser.add_argument("--reviewer-public-key", required=True, type=Path)
    parser.add_argument("--expected-reviewer-public-key-sha256", required=True)
    parser.add_argument("--expected-capture-helper-sha256", required=True)
    parser.add_argument("--reviewer-pseudonym", required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--attest-independent-of-claim-generation-scoring-custody", action="store_true")
    parser.add_argument("--attest-personally-compared-every-pair", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    created = False
    try:
        _private_dir(args.crop_dir, "crop_directory")
        _private_dir(args.screenshot_dir, "screenshots")
        _private_dir(args.trust_root_public_key.parent, "trust_root_public_key_parent", exact=False)
        _private_dir(args.reviewer_private_key.parent, "private_key_parent", exact=False)
        _private_dir(args.reviewer_public_key.parent, "public_key_parent", exact=False)
        parent = output_dir.parent
        _private_dir(parent, "output_parent")
        if os.path.lexists(output_dir) or output_dir.name in {"", ".", ".."}:
            raise ValueError("output_directory_exists_or_invalid")
        for name in INPUTS:
            _private_dir(getattr(args, name).parent, name + "_parent", exact=False)
        files = {name: _read_private(
            getattr(args, name), name,
            max_bytes=50_000_000 if name == "pair_reviewer_html" else MAX_FILE)
            for name in INPUTS}
        root_public_key_raw = _read_private(args.trust_root_public_key, "trust_root_public_key")
        evidence = build_manifest(
            input_files=files, crop_dir=args.crop_dir, screenshot_dir=args.screenshot_dir,
            private_key_path=args.reviewer_private_key,
            public_key_path=args.reviewer_public_key,
            expected_public_key_sha256=args.expected_reviewer_public_key_sha256,
            expected_capture_helper_sha256=args.expected_capture_helper_sha256,
            root_public_key_raw=root_public_key_raw,
            expected_root_public_der_sha256=args.expected_trust_root_der_sha256,
            reviewer_pseudonym=args.reviewer_pseudonym,
            created_at_utc=args.created_at_utc,
            attest_independence=args.attest_independent_of_claim_generation_scoring_custody,
            attest_visual_comparison=args.attest_personally_compared_every_pair)
        output_dir.mkdir(mode=0o700)
        created = True
        os.chmod(output_dir, 0o700)
        evidence_path = output_dir / "clock_frame_evidence_manifest.json"
        fd = os.open(evidence_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(evidence) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        report = {
            "status": "candidate_signed_evidence_manifest_created",
            "scope": "offline_metadata_join_and_human_attestation_signature_only",
            "requested_anchors": 15,
            "effective_anchors": len(evidence["records"]),
            "reviewer_key_trust_status": "root_signed_frame_reviewer_role_key_match_root_pin_externality_unverified",
            "capture_helper_pin_status": "caller_pinned_hash_match_not_joint_freeze_authenticated",
            "capture_operator_signature_status": "rostered_key_signature_valid_exact_artifact_hashes_recomputed",
            "capture_operator_identity_status": "unverified",
            "reviewer_public_key_sha256": args.expected_reviewer_public_key_sha256,
            "joint_freeze_status": "not_jointly_frozen",
            "execution_authorization": "not_granted",
            "output_mode": "0600",
            "output_dir_mode": "0700",
        }
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        if created:
            shutil.rmtree(output_dir, ignore_errors=True)
        code = str(exc) if isinstance(exc, ValueError) else "private_input_or_openssl_error"
        print(json.dumps({"status": "blocked", "failure_code": code,
                          "execution_authorization": "not_granted"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
