#!/usr/bin/env python3
"""Export private RGB PNGs for the exact 3×5 anchors in a candidate plan.

This CPU-only helper checks actual hash-pinned plan/provenance bytes and staged
media, decodes each video once, retains only selected anchor frames, and never
authorizes browser calibration or scoring.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any

import av
from PIL import Image

ANCHOR_IDS = ("fit_begin", "heldout_quarter", "fit_midpoint",
              "heldout_three_quarter", "fit_end")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{24}$")
SHA_RE = re.compile(r"^[a-f0-9]{64}$")
PLAN_KEYS = {"schema_version", "purpose", "status", "execution_authorization", "records"}
PROVENANCE_KEYS = {"schema_version", "purpose", "status", "execution_authorization",
                   "input_sha256", "source_sha256", "plan_sha256", "created_at_utc",
                   "python_version", "pyav_version", "ffmpeg_library_versions",
                   "joint_freeze_receipt_sha256", "records"}
SOURCE_FILES = {
    "builder": "build_nextgqa_clock_anchor_plan.py",
    "decoder": "claim_mask_video_decoder.py",
    "custody_auditor": "audit_nextgqa_review_handoff.py",
    "private_output_policy": "private_output_paths.py",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _unique_json(raw: bytes, label: str) -> Any:
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


def _private_dir(path: Path, label: str) -> None:
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError(label + "_directory_not_private")


def _read_private(path: Path, label: str, limit: int) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        raise ValueError(label + "_file_invalid") from None
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size < 1 or info.st_size > limit):
            raise ValueError(label + "_file_invalid")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) != info.st_size:
            raise ValueError(label + "_file_invalid")
        return raw
    finally:
        os.close(fd)


def _parse_inputs(plan_raw: bytes, provenance_raw: bytes,
                  expected_plan_sha256: str,
                  expected_provenance_sha256: str,
                  scripts_dir: Path | None = None) -> tuple[dict, dict, dict[str, dict]]:
    if (not isinstance(expected_plan_sha256, str) or not SHA_RE.fullmatch(expected_plan_sha256)
            or _sha(plan_raw) != expected_plan_sha256):
        raise ValueError("plan_sha256_mismatch")
    if (not isinstance(expected_provenance_sha256, str) or not SHA_RE.fullmatch(expected_provenance_sha256)
            or _sha(provenance_raw) != expected_provenance_sha256):
        raise ValueError("provenance_sha256_mismatch")
    plan = _unique_json(plan_raw, "plan")
    provenance = _unique_json(provenance_raw, "provenance")
    if (not isinstance(plan, dict) or set(plan) != PLAN_KEYS
            or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("purpose") != "predeclared_html5_clock_pts_anchor_plan"
            or plan.get("status") != "predeclared_frozen"
            or plan.get("execution_authorization") != "not_granted"
            or not isinstance(plan.get("records"), list) or len(plan["records"]) != 3):
        raise ValueError("plan_schema_or_authorization_invalid")
    if (not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS
            or type(provenance.get("schema_version")) is not int or provenance["schema_version"] != 1
            or provenance.get("purpose") != "private_clock_anchor_plan_candidate_provenance"
            or provenance.get("status") != "candidate_not_jointly_frozen"
            or provenance.get("execution_authorization") != "not_granted"
            or provenance.get("plan_sha256") != expected_plan_sha256
            or provenance.get("joint_freeze_receipt_sha256") is not None
            or not isinstance(provenance.get("records"), list)
            or len(provenance["records"]) != 3):
        raise ValueError("provenance_schema_or_plan_binding_invalid")
    if provenance.get("pyav_version") != str(av.__version__):
        raise ValueError("provenance_pyav_version_mismatch")
    expected_ffmpeg_versions = {key: list(value) for key, value in av.library_versions.items()}
    if provenance.get("ffmpeg_library_versions") != expected_ffmpeg_versions:
        raise ValueError("provenance_ffmpeg_versions_mismatch")
    source_root = scripts_dir or Path(__file__).resolve().parent
    actual_source_hashes = {}
    for key, filename in SOURCE_FILES.items():
        try:
            actual_source_hashes[key] = _sha((source_root / filename).read_bytes())
        except OSError:
            raise ValueError("provenance_source_file_unavailable") from None
    source_hashes = provenance.get("source_sha256")
    if (not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_FILES)
            or source_hashes != actual_source_hashes):
        raise ValueError("provenance_source_hash_mismatch")
    plan_by_token: dict[str, dict] = {}
    provenance_by_token: dict[str, dict] = {}
    assignments = set()
    for row in plan["records"]:
        if (not isinstance(row, dict) or set(row) != {
                "review_token", "source_media_sha256", "fit_anchors", "heldout_anchors"}
                or not isinstance(row.get("review_token"), str)
                or not TOKEN_RE.fullmatch(row["review_token"])
                or row["review_token"] in plan_by_token
                or not isinstance(row.get("source_media_sha256"), str)
                or not SHA_RE.fullmatch(row["source_media_sha256"])):
            raise ValueError("plan_record_invalid")
        token = row["review_token"]
        anchors_by_split = ((row["fit_anchors"], {"fit_begin", "fit_midpoint", "fit_end"}),
                            (row["heldout_anchors"], {"heldout_quarter", "heldout_three_quarter"}))
        for anchors, expected_ids in anchors_by_split:
            if not isinstance(anchors, list) or len(anchors) != len(expected_ids):
                raise ValueError("plan_anchor_schema_invalid")
            seen_ids = set()
            for anchor in anchors:
                if (not isinstance(anchor, dict) or set(anchor) != {"anchor_id", "decoder_pts_seconds"}
                        or not isinstance(anchor.get("anchor_id"), str)
                        or not isinstance(anchor.get("decoder_pts_seconds"), str)):
                    raise ValueError("plan_anchor_schema_invalid")
                try:
                    target = Fraction(anchor["decoder_pts_seconds"])
                except (ValueError, ZeroDivisionError, TypeError, OverflowError):
                    raise ValueError("plan_anchor_time_invalid") from None
                if target < 0 or anchor["anchor_id"] in seen_ids:
                    raise ValueError("plan_anchor_invalid")
                seen_ids.add(anchor["anchor_id"])
                assignments.add((token, anchor["anchor_id"]))
            if seen_ids != expected_ids:
                raise ValueError("plan_anchor_ids_invalid")
        plan_by_token[token] = row
    for row in provenance["records"]:
        if not isinstance(row, dict) or row.get("review_token") in provenance_by_token:
            raise ValueError("provenance_record_invalid")
        token = row.get("review_token")
        if token not in plan_by_token or set(row) != {
                "review_token", "staged_media_sha256", "decoded_frame_count",
                "decoded_pts_sequence_sha256", "time_base", "pixel_spec", "decoder_stream_info"}:
            raise ValueError("provenance_record_invalid")
        plan_row = plan_by_token[token]
        if (row.get("staged_media_sha256") != plan_row["source_media_sha256"]
                or type(row.get("decoded_frame_count")) is not int or row["decoded_frame_count"] < 5
                or not isinstance(row.get("decoded_pts_sequence_sha256"), str)
                or not SHA_RE.fullmatch(row["decoded_pts_sequence_sha256"])
                or not isinstance(row.get("time_base"), list) or len(row["time_base"]) != 2
                or any(type(x) is not int for x in row["time_base"])
                or not isinstance(row.get("pixel_spec"), list) or len(row["pixel_spec"]) != 3
                or any(type(x) is not int or x < 1 for x in row["pixel_spec"][:2])
                or not isinstance(row["pixel_spec"][2], str)
                or not isinstance(row.get("decoder_stream_info"), dict)):
            raise ValueError("provenance_record_invalid")
        try:
            if Fraction(*row["time_base"]) <= 0:
                raise ValueError
        except (ValueError, ZeroDivisionError):
            raise ValueError("provenance_time_base_invalid") from None
        provenance_by_token[token] = row
    if set(provenance_by_token) != set(plan_by_token) or len(assignments) != 15:
        raise ValueError("provenance_token_set_mismatch")
    return plan, provenance, provenance_by_token


def _hash_open_file(stream) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def _rgb_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def _write_private(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _decode_video(media_path: Path, media_sha256: str, plan_record: dict,
                  provenance_record: dict, out_dir: Path,
                  crosswalk_rows: list[dict]) -> None:
    try:
        fd = os.open(media_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        raise ValueError("staged_media_file_invalid") from None
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("staged_media_file_invalid")
        with os.fdopen(fd, "rb", closefd=False) as source:
            if _hash_open_file(source) != media_sha256:
                raise ValueError("staged_media_sha256_mismatch")
            expected_tb = Fraction(*provenance_record["time_base"])
            targets = {}
            for row in plan_record["fit_anchors"] + plan_record["heldout_anchors"]:
                targets[row["anchor_id"]] = Fraction(row["decoder_pts_seconds"])
            found = set()
            pts_digest = hashlib.sha256()
            pts_digest.update(b"[")
            pts_count = 0
            first_pts = None
            time_bases = set()
            pixel_specs = set()
            last_pts = None
            with av.open(source, mode="r") as container:
                streams = list(container.streams.video)
                if len(streams) != 1:
                    raise ValueError("video_stream_count_invalid")
                stream = streams[0]
                stream.codec_context.thread_count = 1
                for frame in container.decode(stream):
                    if getattr(frame, "is_corrupt", False) or frame.pts is None:
                        raise ValueError("decoded_frame_invalid")
                    pts = int(frame.pts)
                    if last_pts is not None and pts <= last_pts:
                        raise ValueError("decoded_pts_not_strictly_increasing")
                    last_pts = pts
                    if pts_count:
                        pts_digest.update(b",")
                    pts_digest.update(str(pts).encode("ascii"))
                    pts_count += 1
                    if first_pts is None:
                        first_pts = pts
                    tb = Fraction(frame.time_base if frame.time_base is not None else stream.time_base)
                    time_bases.add(tb)
                    pixel_spec = (int(frame.width), int(frame.height), frame.format.name)
                    pixel_specs.add(pixel_spec)
                    relative = (pts - first_pts) * tb
                    for anchor_id, target in targets.items():
                        if relative == target:
                            if anchor_id in found:
                                raise ValueError("anchor_pts_not_unique")
                            image = frame.to_image().convert("RGB")
                            png_io = io.BytesIO()
                            image.save(png_io, format="PNG", optimize=False)
                            png_raw = png_io.getvalue()
                            filename = plan_record["review_token"] + "_" + anchor_id + ".png"
                            _write_private(out_dir / filename, png_raw)
                            crosswalk_rows.append({
                                "review_token": plan_record["review_token"],
                                "anchor_id": anchor_id,
                                "source_media_sha256": media_sha256,
                                "decoder_frame_index": pts_count - 1,
                                "decoder_pts": pts,
                                "decoder_relative_pts_seconds": str(relative),
                                "rgb_sha256": _rgb_hash(image),
                                "png_sha256": _sha(png_raw),
                                "png_file": filename,
                            })
                            found.add(anchor_id)
            pts_digest.update(b"]")
            if (pts_count != provenance_record["decoded_frame_count"]
                    or pts_digest.hexdigest() != provenance_record["decoded_pts_sequence_sha256"]
                    or time_bases != {expected_tb}
                    or pixel_specs != {tuple(provenance_record["pixel_spec"])}
                    or found != set(targets)):
                raise ValueError("decoded_sequence_or_anchor_mismatch")
            if _hash_open_file(source) != media_sha256:
                raise ValueError("staged_media_changed_during_decode")
    finally:
        os.close(fd)


def export_anchor_pngs(plan_path: Path, provenance_path: Path, media_dir: Path,
                       output_dir: Path, expected_plan_sha256: str,
                       expected_provenance_sha256: str,
                       scripts_dir: Path | None = None) -> dict:
    _private_dir(plan_path.parent, "plan_parent")
    _private_dir(provenance_path.parent, "provenance_parent")
    if plan_path.parent != provenance_path.parent:
        raise ValueError("plan_provenance_directory_mismatch")
    _private_dir(media_dir, "media")
    parent = output_dir.parent
    _private_dir(parent, "output_parent")
    if os.path.lexists(output_dir) or output_dir.name in {"", ".", ".."}:
        raise ValueError("output_directory_exists_or_invalid")
    plan_raw = _read_private(plan_path, "plan", 2_000_000)
    provenance_raw = _read_private(provenance_path, "provenance", 5_000_000)
    plan, provenance, provenance_by_token = _parse_inputs(
        plan_raw, provenance_raw, expected_plan_sha256, expected_provenance_sha256,
        scripts_dir=scripts_dir)
    output_created = False
    try:
        output_dir.mkdir(mode=0o700)
        output_created = True
        os.chmod(output_dir, 0o700)
        crosswalk_rows: list[dict] = []
        for plan_record in plan["records"]:
            token = plan_record["review_token"]
            media_sha = plan_record["source_media_sha256"]
            _decode_video(media_dir / (token + ".mp4"), media_sha, plan_record,
                          provenance_by_token[token], output_dir, crosswalk_rows)
        crosswalk = {"schema_version": 1,
                     "purpose": "private_candidate_decoder_anchor_png_crosswalk",
                     "status": "candidate_artifact_export_only",
                     "execution_authorization": "not_granted",
                     "plan_sha256": _sha(plan_raw),
                     "provenance_sha256": _sha(provenance_raw),
                     "requested_anchors": 15,
                     "effective_anchors": len(crosswalk_rows),
                     "records": crosswalk_rows}
        if len(crosswalk_rows) != 15:
            raise ValueError("decoder_anchor_export_incomplete")
        _write_private(output_dir / "crosswalk.json", _canonical(crosswalk))
        return {"status": "private_candidate_decoder_anchor_pngs_written",
                "records": len(crosswalk_rows), "execution_authorization": "not_granted"}
    except BaseException:
        if output_created:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--expected-provenance-sha256", required=True)
    parser.add_argument("--media-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = export_anchor_pngs(args.plan, args.provenance, args.media_dir,
                                    args.output_dir, args.expected_plan_sha256,
                                    args.expected_provenance_sha256)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    except (OSError, av.error.FFmpegError):
        parser.exit(2, "private_anchor_export_failed\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
