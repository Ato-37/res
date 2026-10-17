#!/usr/bin/env python3
"""Run the frozen 32-frame sampler on exactly three local QTA candidate videos.

This CPU-only preflight verifies media decode/sampling feasibility. It never
opens image pixels for display, loads a model, scores faithfulness, or grants
execution authorization. The detailed manifest is local-only and mode 0600.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any

from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
from necessity_choice_margin_utils import uniform_sample_indices


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = REPO_ROOT / "data" / "private_nextgqa"
DEFAULT_MEDIA_DIR = PRIVATE_ROOT / "review_media_v6_20261017_0635"
FRAMES = 32
GRID_TOLERANCE = Fraction(1, 20)
MEDIA_NAME = re.compile(r"[A-Za-z0-9_-]{24}\.mp4\Z")
INVENTORY_MAX_BYTES = 4 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fraction_pair(value: Any, code: str) -> Fraction:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(part) is not int for part in value)):
        raise ValueError(code)
    try:
        return Fraction(value[0], value[1])
    except (ValueError, ZeroDivisionError, OverflowError):
        raise ValueError(code) from None


def _validate_private_root() -> Path:
    if PRIVATE_ROOT.is_symlink():
        raise ValueError("private_root_symlink_rejected")
    root = PRIVATE_ROOT.resolve(strict=True)
    info = root.stat(follow_symlinks=False)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private_root_not_private")
    return root


def _validate_private_directory(path: Path) -> Path:
    _validate_private_root()
    if path.is_symlink():
        raise ValueError("media_directory_symlink_rejected")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(PRIVATE_ROOT.resolve(strict=True)):
        raise ValueError("media_directory_outside_private_root")
    info = resolved.stat(follow_symlinks=False)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("media_directory_not_private")
    return resolved


def _validate_output_path(path: Path) -> Path:
    _validate_private_root()
    if path.is_symlink():
        raise ValueError("output_symlink_rejected")
    parent = path.parent.resolve(strict=True)
    if parent != PRIVATE_ROOT.resolve(strict=True):
        raise ValueError("output_must_be_direct_child_of_private_root")
    if path.exists() or path.is_symlink():
        raise ValueError("output_already_exists")
    return parent / path.name


def load_candidate_inventory(path: Path) -> tuple[dict[str, str], str]:
    """Validate a local candidate inventory and return its frozen media hashes."""
    _validate_private_root()
    if path.is_symlink():
        raise ValueError("candidate_inventory_symlink_rejected")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(PRIVATE_ROOT.resolve(strict=True)):
        raise ValueError("inventory_outside_private_root")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size < 1 or before.st_size > INVENTORY_MAX_BYTES):
            raise ValueError("candidate_inventory_file_invalid")
        chunks = []
        remaining = before.st_size
        while remaining:
            block = os.read(fd, min(65536, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(fd)
        if (remaining or os.read(fd, 1) or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ctime_ns != after.st_ctime_ns):
            raise ValueError("candidate_inventory_changed_during_read")
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    try:
        inventory = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                               parse_constant=lambda _: (_ for _ in ()).throw(
                                   ValueError("candidate_inventory_nonfinite")))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        raise ValueError("candidate_inventory_json_invalid") from None
    if (not isinstance(inventory, dict)
            or inventory.get("purpose") != "pre_review_candidate_byte_inventory"
            or inventory.get("status") != "candidate_only"
            or inventory.get("execution_authorization") != "not_granted"
            or inventory.get("joint_freeze") is not False
            or inventory.get("score_ready") is not False
            or inventory.get("rights_status") != "unverified_local_only"):
        raise ValueError("candidate_inventory_status_invalid")
    if inventory.get("missing_required_slots") != []:
        raise ValueError("candidate_inventory_incomplete")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("candidate_inventory_artifacts_invalid")
    by_role = {}
    for item in artifacts:
        if (not isinstance(item, dict) or not isinstance(item.get("role"), str)
                or item["role"] in by_role):
            raise ValueError("candidate_inventory_artifacts_invalid")
        by_role[item["role"]] = item
    media_roles = [f"staged_media_{index:02d}" for index in range(1, 4)]
    media_hashes = {}
    for role in media_roles:
        item = by_role.get(role)
        digest = item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("candidate_inventory_media_binding_missing")
        media_hashes[role] = digest
    current_files = {
        "qta_protocol_draft": REPO_ROOT / "docs/qwen_question_attention_faithfulness_protocol_v1_draft.md",
        "review_runbook": REPO_ROOT / "docs/nextgqa_v6_human_review_runbook.md",
        "video_decoder": REPO_ROOT / "scripts/claim_mask_video_decoder.py",
        "necessity_choice_margin_utils": REPO_ROOT / "scripts/necessity_choice_margin_utils.py",
        "presentation_time_utils": REPO_ROOT / "scripts/presentation_time_utils.py",
        "qta_real_video_sampler_preflight": Path(__file__).resolve(),
    }
    for role, source in current_files.items():
        item = by_role.get(role)
        if (not isinstance(item, dict)
                or item.get("sha256") != _sha256(source)
                or item.get("size_bytes") != source.stat().st_size):
            raise ValueError("candidate_inventory_source_binding_stale")
    return media_hashes, hashlib.sha256(raw).hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _decode_row(index: int, path: Path, expected_sha256: str) -> dict[str, Any]:
    info = path.stat(follow_symlinks=False)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or not MEDIA_NAME.fullmatch(path.name)):
        raise ValueError("media_file_not_private_or_name_invalid")
    source_digest_before = _sha256(path)
    frames, metadata = decode_sampled_video(
        path, num_frames=FRAMES, uniform_tolerance_seconds=GRID_TOLERANCE,
        include_timing_map=True)
    verify_sampled_frame_pixels(frames, metadata)
    source_digest_after = _sha256(path)
    if (source_digest_before != expected_sha256
            or source_digest_before != source_digest_after
            or metadata.get("source_sha256") != source_digest_before):
        raise ValueError("source_inventory_hash_mismatch")
    if metadata.get("requested_frame_count") != FRAMES:
        raise ValueError("requested_frame_count_mismatch")
    decoded_count = metadata.get("decoded_frame_count")
    if type(decoded_count) is not int or decoded_count < FRAMES:
        raise ValueError("insufficient_decoded_frames")
    sampled = metadata.get("sampled_frames")
    if not isinstance(sampled, list) or len(sampled) != FRAMES:
        raise ValueError("sampled_frame_count_mismatch")
    expected_indices = uniform_sample_indices(decoded_count, FRAMES)
    observed_indices = [row.get("decoded_frame_index") for row in sampled]
    if observed_indices != expected_indices or len(set(observed_indices)) != FRAMES:
        raise ValueError("endpoint_sampler_indices_mismatch")
    tolerance = _fraction_pair(metadata.get("uniform_grid_tolerance_seconds"),
                               "grid_tolerance_invalid")
    deviation = _fraction_pair(metadata.get("max_sampled_grid_deviation_seconds"),
                               "grid_deviation_invalid")
    fps = _fraction_pair(metadata.get("effective_sample_fps"), "effective_fps_invalid")
    if tolerance != GRID_TOLERANCE or deviation > tolerance or fps <= 0:
        raise ValueError("sampled_grid_tolerance_failed")
    timing = metadata.get("full_timing_map")
    if (not isinstance(timing, dict) or timing.get("timestamp_origin")
            != "first_decoded_presentation_timestamp"
            or len(timing.get("pts", [])) != decoded_count
            or timing.get("decoded_pts_sequence_sha256")
            != metadata.get("decoded_pts_sequence_sha256")):
        raise ValueError("full_timing_map_invalid")
    if any(not isinstance(row.get("rgb_pixel_sha256"), str)
           or not re.fullmatch(r"[0-9a-f]{64}", row["rgb_pixel_sha256"])
           for row in sampled):
        raise ValueError("sampled_rgb_hash_invalid")
    # Keep exact PTS/hashes in the local manifest, but do not include file names,
    # reviewer tokens, claims, questions, or answer text.
    return {
        "record_index": index,
        "source_sha256": source_digest_before,
        "decoded_frame_count": decoded_count,
        "reported_count_matches_decoded": metadata.get(
            "reported_stream_frame_count_matches_decoded"),
        "sampled_frame_count": FRAMES,
        "sampled_indices": observed_indices,
        "sampled_pts": [row["relative_presentation_time"] for row in sampled],
        "sampled_rgb_sha256": [row["rgb_pixel_sha256"] for row in sampled],
        "pts_sequence_sha256": metadata["decoded_pts_sequence_sha256"],
        "pass1_rgb_sequence_sha256": metadata["pass1_decoded_rgb_sequence_sha256"],
        "effective_sample_fps": [fps.numerator, fps.denominator],
        "max_grid_deviation_seconds": [deviation.numerator, deviation.denominator],
        "grid_tolerance_seconds": [tolerance.numerator, tolerance.denominator],
        "stream": metadata["stream"],
        "decoder": metadata["decoder"],
        "full_timing_map": timing,
    }


def build_report(media_dir: Path, candidate_inventory: Path) -> dict[str, Any]:
    directory = _validate_private_directory(media_dir)
    if directory != DEFAULT_MEDIA_DIR.resolve(strict=True):
        raise ValueError("media_directory_not_frozen_candidate")
    media_hashes, inventory_sha256 = load_candidate_inventory(candidate_inventory)
    paths = sorted(directory.iterdir(), key=lambda item: item.name)
    if (len(paths) != 3 or any(not MEDIA_NAME.fullmatch(path.name) for path in paths)
            or len({path.name for path in paths}) != 3):
        raise ValueError("expected_exactly_three_private_mp4_files")
    rows = []
    for index, (path, role) in enumerate(zip(paths, sorted(media_hashes)), 1):
        rows.append(_decode_row(index, path, media_hashes[role]))
    return {
        "schema_version": 1,
        "purpose": "local_qta_real_video_sampler_eligibility_preflight",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "passed",
        "requested_records": 3,
        "effective_records": len(rows),
        "sampler": "endpoint_inclusive_uniform_decoded_frame_index_v1",
        "python_version": sys.version.split()[0],
        "frames_per_video": FRAMES,
        "grid_tolerance_seconds": [GRID_TOLERANCE.numerator,
                                    GRID_TOLERANCE.denominator],
        "candidate_inventory_sha256": inventory_sha256,
        "rows": rows,
        "gpu_used": False,
        "model_loaded": False,
        "faithfulness_scored": False,
        "annotation_clock_calibrated": False,
        "rights_status": "unverified_local_only",
        "execution_authorization": "not_granted",
    }


def write_private_report(report: dict[str, Any], output_path: Path) -> None:
    target = _validate_output_path(output_path)
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False, default=lambda value:
                      [value.numerator, value.denominator]
                      if isinstance(value, Fraction) else _raise_json_type(value))
           + "\n").encode("ascii")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
    except BaseException:
        try:
            os.unlink(target)
        except OSError:
            pass
        raise


def _raise_json_type(value: Any):
    raise TypeError(type(value).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-inventory", type=Path, required=True,
                        help="current mode-0600 candidate inventory generated by the local builder")
    parser.add_argument("--output-json", type=Path, required=True,
                        help="new JSON path directly under data/private_nextgqa")
    args = parser.parse_args()
    try:
        report = build_report(DEFAULT_MEDIA_DIR, args.candidate_inventory)
        write_private_report(report, args.output_json)
    except (OSError, ValueError, TypeError) as exc:
        code = str(exc)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", code):
            code = "preflight_failed"
        print(json.dumps({"status": "blocked", "failure_code": code,
                          "requested_records": 3, "effective_records": 0,
                          "gpu_used": False, "model_loaded": False,
                          "faithfulness_scored": False,
                          "execution_authorization": "not_granted"},
                         sort_keys=True), file=sys.stderr)
        return 2
    deviations = [float(Fraction(*row["max_grid_deviation_seconds"]))
                  for row in report["rows"]]
    fps = [float(Fraction(*row["effective_sample_fps"])) for row in report["rows"]]
    print(json.dumps({
        "status": report["status"], "requested_records": 3,
        "effective_records": 3, "sampled_frames_per_video": [FRAMES] * 3,
        "decoded_frame_counts": [row["decoded_frame_count"] for row in report["rows"]],
        "max_grid_deviation_ms": [round(value * 1000, 3) for value in deviations],
        "effective_fps": [round(value, 6) for value in fps],
        "gpu_used": False, "model_loaded": False,
        "faithfulness_scored": False,
        "execution_authorization": "not_granted",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
