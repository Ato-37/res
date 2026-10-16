#!/usr/bin/env python3
"""Create private candidate video-content crops from receipt-bound screenshots.

The bbox is custodian supplied and only asserts an intended video region. This
tool cannot establish crop purity, screenshot authenticity, or authorization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
from typing import Any

import av
from PIL import Image, UnidentifiedImageError

ANCHORS = ("fit_begin", "heldout_quarter", "fit_midpoint",
           "heldout_three_quarter", "fit_end")
FIT = {"fit_begin", "fit_midpoint", "fit_end"}
HELDOUT = {"heldout_quarter", "heldout_three_quarter"}
TOK = re.compile(r"^[A-Za-z0-9_-]{24}$")
SHA = re.compile(r"^[a-f0-9]{64}$")
PLAN_KEYS = {"schema_version", "purpose", "status", "execution_authorization", "records"}
PROV_KEYS = {"schema_version", "purpose", "status", "execution_authorization",
             "input_sha256", "source_sha256", "plan_sha256", "created_at_utc",
             "python_version", "pyav_version", "ffmpeg_library_versions",
             "joint_freeze_receipt_sha256", "records"}
SOURCE_FILES = {
    "builder": "build_nextgqa_clock_anchor_plan.py",
    "decoder": "claim_mask_video_decoder.py",
    "custody_auditor": "audit_nextgqa_review_handoff.py",
    "private_output_policy": "private_output_paths.py",
}
RECEIPT_KEYS = {"schema_version", "purpose", "status", "execution_authorization", "plan_sha256",
                "plan_provenance_sha256", "capture_method", "capture_helper_file_sha256",
                "created_at_utc", "records"}
CAPTURE_ROW_KEYS = {"review_token", "anchor_id", "screenshot_file", "screenshot_sha256",
                    "screenshot_captured_at_utc", "capture_method", "foreground_process",
                    "foreground_pid", "foreground_window_title", "foreground_window_rect_xywh",
                    "virtual_screen_rect_xywh", "hotkey_message_tick_ms",
                    "capture_complete_tick_ms", "hotkey_to_capture_ms"}
CROP_MAP_KEYS = {"schema_version", "purpose", "plan_sha256", "plan_provenance_sha256",
                 "capture_receipt_sha256", "records"}
MAX_JSON = 2_000_000
MAX_SCREENSHOT = 40_000_000
MAX_PIXELS = 50_000_000
MAX_SIDE = 12_000
INPUT_HASH_KEYS = {"worksheet", "lineage_manifest", "custody_manifest"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _private_dir(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError:
        raise ValueError(label) from None
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError(label)


def _private_read(path: Path, label: str, limit: int) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        raise ValueError(label) from None
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size < 1 or info.st_size > limit):
            raise ValueError(label)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) != info.st_size:
            raise ValueError(label)
        return raw
    finally:
        os.close(fd)


def _plan_assignments(plan_raw: bytes, plan_sha: str) -> tuple[set[tuple[str, str]], dict[str, str]]:
    if _sha(plan_raw) != plan_sha:
        raise ValueError("plan_sha256_mismatch")
    plan = _json(plan_raw, "plan")
    if (not isinstance(plan, dict) or set(plan) != PLAN_KEYS
            or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("purpose") != "predeclared_html5_clock_pts_anchor_plan"
            or plan.get("status") != "predeclared_frozen"
            or plan.get("execution_authorization") != "not_granted"
            or not isinstance(plan.get("records"), list) or len(plan["records"]) != 3):
        raise ValueError("plan_candidate_invalid")
    tokens, assignments, media_hashes = set(), set(), {}
    for row in plan["records"]:
        if (not isinstance(row, dict) or set(row) != {
                "review_token", "source_media_sha256", "fit_anchors", "heldout_anchors"}
                or not isinstance(row.get("review_token"), str)
                or not isinstance(row.get("source_media_sha256"), str)
                or not SHA.fullmatch(row["source_media_sha256"])):
            raise ValueError("plan_record_invalid")
        token = row["review_token"]
        if not TOK.fullmatch(token) or token in tokens:
            raise ValueError("plan_token_invalid_or_duplicate")
        tokens.add(token)
        media_hashes[token] = row["source_media_sha256"]
        for field, expected in (("fit_anchors", FIT), ("heldout_anchors", HELDOUT)):
            anchors = row.get(field)
            if not isinstance(anchors, list):
                raise ValueError("plan_anchor_ids_invalid")
            if any(not isinstance(a, dict) or set(a) != {"anchor_id", "decoder_pts_seconds"}
                   or not isinstance(a.get("anchor_id"), str)
                   or not isinstance(a.get("decoder_pts_seconds"), str)
                   or not a["decoder_pts_seconds"] for a in anchors):
                raise ValueError("plan_anchor_ids_invalid")
            ids = [a["anchor_id"] for a in anchors]
            if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != expected:
                raise ValueError("plan_anchor_ids_invalid")
            assignments.update((token, anchor) for anchor in ids)
    return assignments, media_hashes


def _validate_provenance(raw: bytes, expected_sha: str, plan_sha: str,
                         media_hashes: dict[str, str]) -> None:
    if not SHA.fullmatch(expected_sha) or _sha(raw) != expected_sha:
        raise ValueError("provenance_sha256_mismatch")
    value = _json(raw, "provenance")
    if (not isinstance(value, dict) or set(value) != PROV_KEYS
            or type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("purpose") != "private_clock_anchor_plan_candidate_provenance"
            or value.get("status") != "candidate_not_jointly_frozen"
            or value.get("execution_authorization") != "not_granted"
            or value.get("plan_sha256") != plan_sha
            or value.get("joint_freeze_receipt_sha256") is not None):
        raise ValueError("provenance_candidate_invalid")
    input_hashes = value.get("input_sha256")
    if (not isinstance(input_hashes, dict) or set(input_hashes) != INPUT_HASH_KEYS
            or any(not isinstance(digest, str) or not SHA.fullmatch(digest)
                   for digest in input_hashes.values())):
        raise ValueError("provenance_input_hashes_invalid")
    created = value.get("created_at_utc")
    try:
        parsed_created = datetime.fromisoformat(created[:-1] + "+00:00") if (
            isinstance(created, str) and created.endswith("Z")) else None
    except ValueError:
        parsed_created = None
    if (parsed_created is None or parsed_created.tzinfo is None
            or parsed_created.utcoffset() != timezone.utc.utcoffset(parsed_created)):
        raise ValueError("provenance_created_at_invalid")
    python_version = value.get("python_version")
    if not isinstance(python_version, str) or not re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?", python_version):
        raise ValueError("provenance_python_version_invalid")
    if python_version != platform.python_version():
        raise ValueError("provenance_python_version_mismatch")
    if value.get("pyav_version") != str(av.__version__):
        raise ValueError("provenance_pyav_version_mismatch")
    expected_ffmpeg = {key: list(version) for key, version in av.library_versions.items()}
    if value.get("ffmpeg_library_versions") != expected_ffmpeg:
        raise ValueError("provenance_ffmpeg_versions_mismatch")
    source_hashes = value.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_FILES):
        raise ValueError("provenance_source_hash_mismatch")
    for key, filename in SOURCE_FILES.items():
        try:
            actual = _sha((Path(__file__).resolve().parent / filename).read_bytes())
        except OSError:
            raise ValueError("provenance_source_file_unavailable") from None
        if source_hashes.get(key) != actual:
            raise ValueError("provenance_source_hash_mismatch")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("provenance_records_invalid")
    seen = set()
    for row in records:
        if (not isinstance(row, dict) or set(row) != {
                "review_token", "staged_media_sha256", "decoded_frame_count",
                "decoded_pts_sequence_sha256", "time_base", "pixel_spec", "decoder_stream_info"}
                or not isinstance(row.get("review_token"), str)
                or not TOK.fullmatch(row["review_token"])
                or row["review_token"] in seen
                or not isinstance(row.get("staged_media_sha256"), str)
                or not SHA.fullmatch(row["staged_media_sha256"])
                or row.get("staged_media_sha256") != media_hashes.get(row["review_token"])
                or type(row.get("decoded_frame_count")) is not int or row["decoded_frame_count"] < 5
                or not isinstance(row.get("decoded_pts_sequence_sha256"), str)
                or not SHA.fullmatch(row["decoded_pts_sequence_sha256"])
                or not isinstance(row.get("time_base"), list) or len(row["time_base"]) != 2
                or any(type(part) is not int for part in row["time_base"])
                or not isinstance(row.get("pixel_spec"), list) or len(row["pixel_spec"]) != 3
                or any(type(part) is not int or part < 1 for part in row["pixel_spec"][:2])
                or not isinstance(row["pixel_spec"][2], str)
                or not isinstance(row.get("decoder_stream_info"), dict)):
            raise ValueError("provenance_records_invalid")
        try:
            if Fraction(*row["time_base"]) <= 0:
                raise ValueError
        except (ValueError, ZeroDivisionError):
            raise ValueError("provenance_records_invalid") from None
        seen.add(row["review_token"])
    if seen != set(media_hashes):
        raise ValueError("provenance_records_invalid")


def _validate_crop_map(raw: bytes, plan_sha: str, provenance_sha: str,
                       receipt_sha: str, expected: set[tuple[str, str]]) -> dict:
    value = _json(raw, "crop_map")
    if (not isinstance(value, dict) or set(value) != CROP_MAP_KEYS
            or type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("purpose") != "private_candidate_screenshot_video_crop_map"
            or value.get("plan_sha256") != plan_sha
            or value.get("plan_provenance_sha256") != provenance_sha
            or value.get("capture_receipt_sha256") != receipt_sha
            or not isinstance(value.get("records"), list) or len(value["records"]) != 15):
        raise ValueError("crop_map_binding_or_count_invalid")
    seen = set()
    for row in value["records"]:
        if not isinstance(row, dict) or set(row) != {"review_token", "anchor_id", "bbox_xywh"}:
            raise ValueError("crop_map_record_invalid")
        token, anchor = row.get("review_token"), row.get("anchor_id")
        if (not isinstance(token, str) or not TOK.fullmatch(token)
                or not isinstance(anchor, str) or anchor not in ANCHORS):
            raise ValueError("crop_map_record_invalid")
        key = (token, anchor)
        bbox = row["bbox_xywh"]
        if (key not in expected or key in seen or not isinstance(bbox, list) or len(bbox) != 4
                or any(type(n) is not int for n in bbox)
                or bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 1 or bbox[3] < 1):
            raise ValueError("crop_map_record_invalid")
        seen.add(key)
    if seen != expected:
        raise ValueError("crop_map_assignment_mismatch")
    return value


def _validate_receipt(raw: bytes, expected_sha: str, plan_sha: str, provenance_sha: str,
                      expected: set[tuple[str, str]]) -> dict:
    if not SHA.fullmatch(expected_sha) or _sha(raw) != expected_sha:
        raise ValueError("capture_receipt_sha256_mismatch")
    receipt = _json(raw, "capture_receipt")
    if (not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS
            or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
            or receipt.get("purpose") != "private_windows_compositor_screenshot_capture_receipt_candidate"
            or receipt.get("status") != "self_reported_capture_receipt_not_independently_authenticated"
            or receipt.get("execution_authorization") != "not_granted"
            or receipt.get("plan_sha256") != plan_sha
            or receipt.get("plan_provenance_sha256") != provenance_sha
            or receipt.get("capture_method") != "windows_gdi_copyfromscreen_foreground_window_v1"
            or not isinstance(receipt.get("records"), list) or len(receipt["records"]) != 15):
        raise ValueError("capture_receipt_candidate_invalid")
    mapped, filenames = {}, set()
    for row in receipt["records"]:
        if not isinstance(row, dict) or set(row) != CAPTURE_ROW_KEYS:
            raise ValueError("capture_receipt_record_invalid")
        token, anchor = row.get("review_token"), row.get("anchor_id")
        if (not isinstance(token, str) or not TOK.fullmatch(token)
                or not isinstance(anchor, str) or anchor not in ANCHORS):
            raise ValueError("capture_receipt_record_invalid")
        key = (token, anchor)
        filename = row.get("screenshot_file")
        if (key not in expected or key in mapped or not isinstance(filename, str)
                or not filename or Path(filename).name != filename or filename in {".", ".."}
                or filename in filenames
                or not isinstance(row.get("screenshot_sha256"), str)
                or not SHA.fullmatch(row["screenshot_sha256"])
                or row.get("capture_method") != receipt["capture_method"]):
            raise ValueError("capture_receipt_record_invalid")
        mapped[key] = row
        filenames.add(filename)
    if set(mapped) != expected:
        raise ValueError("capture_receipt_assignment_mismatch")
    return mapped


def _image_crop(raw: bytes, bbox: list[int]) -> tuple[bytes, tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.width < 1 or image.height < 1:
                raise ValueError("screenshot_image_invalid")
            if image.width > MAX_SIDE or image.height > MAX_SIDE or image.width * image.height > MAX_PIXELS:
                raise ValueError("screenshot_dimensions_exceeded")
            image.load()
            x, y, width, height = bbox
            if x + width > image.width or y + height > image.height:
                raise ValueError("crop_bbox_out_of_bounds")
            crop = image.crop((x, y, x + width, y + height)).convert("RGB")
            out = io.BytesIO()
            crop.save(out, format="PNG", optimize=False)
            return out.getvalue(), crop.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("screenshot_image_invalid") from None


def crop_screenshots(plan_path: Path, expected_plan_sha: str,
                     provenance_path: Path, expected_provenance_sha: str,
                     receipt_path: Path, expected_receipt_sha: str,
                     screenshot_dir: Path, crop_map_path: Path,
                     output_dir: Path) -> dict:
    _private_dir(screenshot_dir, "screenshot_directory_not_private")
    _private_dir(plan_path.parent, "plan_directory_not_private")
    _private_dir(provenance_path.parent, "provenance_directory_not_private")
    _private_dir(receipt_path.parent, "receipt_directory_not_private")
    _private_dir(crop_map_path.parent, "crop_map_directory_not_private")
    parent = output_dir.parent
    _private_dir(parent, "output_parent_not_private")
    if os.path.lexists(output_dir):
        raise ValueError("output_directory_exists")
    plan_raw = _private_read(plan_path, "plan_file_invalid", MAX_JSON)
    if not SHA.fullmatch(expected_plan_sha) or _sha(plan_raw) != expected_plan_sha:
        raise ValueError("plan_sha256_mismatch")
    expected, media_hashes = _plan_assignments(plan_raw, expected_plan_sha)
    provenance_raw = _private_read(provenance_path, "provenance_file_invalid", MAX_JSON)
    _validate_provenance(provenance_raw, expected_provenance_sha, expected_plan_sha, media_hashes)
    receipt_raw = _private_read(receipt_path, "capture_receipt_file_invalid", MAX_JSON)
    receipt_sha = _sha(receipt_raw)
    receipt_rows = _validate_receipt(receipt_raw, expected_receipt_sha, expected_plan_sha,
                                     expected_provenance_sha, expected)
    crop_map_raw = _private_read(crop_map_path, "crop_map_file_invalid", MAX_JSON)
    crop_map = _validate_crop_map(crop_map_raw, expected_plan_sha, expected_provenance_sha,
                                  receipt_sha, expected)
    crop_by_key = {(r["review_token"], r["anchor_id"]): r["bbox_xywh"]
                   for r in crop_map["records"]}

    output_created = False
    try:
        output_dir.mkdir(mode=0o700)
        output_created = True
        os.chmod(output_dir, 0o700)
        records = []
        for key in sorted(expected):
            row = receipt_rows[key]
            filename = row["screenshot_file"]
            raw = _private_read(screenshot_dir / filename, "screenshot_file_invalid", MAX_SCREENSHOT)
            if _sha(raw) != row["screenshot_sha256"]:
                raise ValueError("screenshot_sha256_mismatch")
            bbox = crop_by_key[key]
            png, (width, height) = _image_crop(raw, bbox)
            out_name = key[0] + "_" + key[1] + ".png"
            fd = os.open(output_dir / out_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(png)
                stream.flush()
                os.fsync(stream.fileno())
            records.append({"review_token": key[0], "anchor_id": key[1],
                            "screenshot_file": filename,
                            "screenshot_sha256": row["screenshot_sha256"],
                            "bbox_xywh": bbox, "crop_file": out_name,
                            "crop_sha256": _sha(png), "crop_rgb_dimensions": [width, height],
                            "crop_purity": "human_review_required"})
        crosswalk = {"schema_version": 1, "purpose": "private_candidate_screenshot_crop_crosswalk",
                     "status": "candidate_only", "execution_authorization": "not_granted",
                     "plan_sha256": expected_plan_sha,
                     "plan_provenance_sha256": expected_provenance_sha,
                     "capture_receipt_sha256": receipt_sha,
                     "crop_map_sha256": _sha(crop_map_raw),
                     "bbox_video_only_purity": "not_established_human_review_required",
                     "records": records}
        fd = os.open(output_dir / "crop_crosswalk.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(crosswalk))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if output_created:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return {"status": "candidate_crops_created", "records": 15,
            "execution_authorization": "not_granted"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--expected-provenance-sha256", required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--expected-capture-receipt-sha256", required=True)
    parser.add_argument("--screenshot-dir", type=Path, required=True)
    parser.add_argument("--crop-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = crop_screenshots(args.plan, args.expected_plan_sha256,
                                  args.provenance, args.expected_provenance_sha256,
                                  args.capture_receipt, args.expected_capture_receipt_sha256,
                                  args.screenshot_dir, args.crop_map, args.output_dir)
    except (OSError, ValueError) as exc:
        code = str(exc) if isinstance(exc, ValueError) else "private_crop_failed"
        parser.exit(2, "error: " + code + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
