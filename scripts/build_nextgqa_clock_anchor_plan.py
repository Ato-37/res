#!/usr/bin/env python3
"""Build a private, outcome-blind HTML5 clock anchor plan for the 3-item pilot.

The plan selects fixed fractions of each exact staged video timeline and
freezes disjoint fit/held-out anchors. It does not collect browser evidence,
verify compositor-to-decoder correspondence, calibrate a clock, or authorize
scoring. The candidate plan must be included in the later joint protocol/input
freeze before it is used for a capture.
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
import secrets
import stat
import sys
from datetime import datetime, timezone
from typing import Any

import av

from audit_nextgqa_review_handoff import audit_custody
from claim_mask_video_decoder import _stream_and_decode
from private_output_paths import validate_private_output

PURPOSE = "predeclared_html5_clock_pts_anchor_plan"
TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")
FRACTIONS = (
    ("fit_begin", Fraction(1, 20), "fit"),
    ("heldout_quarter", Fraction(1, 4), "heldout"),
    ("fit_midpoint", Fraction(1, 2), "fit"),
    ("heldout_three_quarter", Fraction(3, 4), "heldout"),
    ("fit_end", Fraction(19, 20), "fit"),
)


def _nearest_index(times: tuple[Fraction, ...], target: Fraction) -> int:
    return min(range(len(times)), key=lambda index: (abs(times[index] - target), index))


def select_anchor_points(pts_values: list[int], time_base: Fraction) -> tuple[dict, dict]:
    """Select 5 distinct fixed-fraction anchors from display-order decoder PTS."""
    if not isinstance(time_base, Fraction) or time_base <= 0:
        raise ValueError("anchor_time_base_invalid")
    if (not isinstance(pts_values, list) or len(pts_values) < len(FRACTIONS)
            or any(type(value) is not int or value < -(1 << 63) or value > (1 << 63) - 1
                   for value in pts_values)):
        raise ValueError("anchor_frame_count_invalid")
    if any(right <= left for left, right in zip(pts_values, pts_values[1:])):
        raise ValueError("anchor_pts_not_strictly_increasing")
    relative_times = tuple((value - pts_values[0]) * time_base for value in pts_values)
    duration = relative_times[-1]
    if duration <= 0:
        raise ValueError("anchor_duration_invalid")

    selected: dict[str, list[dict[str, str]]] = {"fit": [], "heldout": []}
    selected_indices = set()
    for anchor_id, fraction, split in FRACTIONS:
        index = _nearest_index(relative_times, duration * fraction)
        if index in selected_indices:
            raise ValueError("anchor_targets_not_distinct")
        selected_indices.add(index)
        selected[split].append({
            "anchor_id": anchor_id,
            "decoder_pts_seconds": str(relative_times[index]),
        })
    return selected["fit"], selected["heldout"]


def build_anchor_plan(records: list[dict[str, Any]]) -> dict:
    """Build the exact schema consumed by the raw clock-capture converter."""
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("anchor_cohort_must_have_three_records")
    planned = []
    tokens = set()
    for row in records:
        if not isinstance(row, dict) or set(row) != {
                "review_token", "source_media_sha256", "pts", "time_base"}:
            raise ValueError("anchor_source_record_invalid")
        token = row["review_token"]
        if not isinstance(token, str) or TOKEN.fullmatch(token) is None or token in tokens:
            raise ValueError("anchor_review_token_invalid")
        tokens.add(token)
        digest = row["source_media_sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("anchor_media_hash_invalid")
        time_base_value = row["time_base"]
        if (not isinstance(time_base_value, (list, tuple)) or len(time_base_value) != 2
                or any(type(value) is not int for value in time_base_value)):
            raise ValueError("anchor_time_base_invalid")
        try:
            time_base = Fraction(*time_base_value)
        except (ZeroDivisionError, ValueError) as exc:
            raise ValueError("anchor_time_base_invalid") from exc
        fit, heldout = select_anchor_points(row["pts"], time_base)
        planned.append({"review_token": token, "source_media_sha256": digest,
                        "fit_anchors": fit, "heldout_anchors": heldout})
    return {"schema_version": 1, "purpose": PURPOSE,
            # The downstream schema uses this value to mean only that anchor
            # selection is fixed before observations; it does not freeze the
            # joint cohort/claim/target/scoring protocol or grant execution.
            "status": "predeclared_frozen", "execution_authorization": "not_granted",
            "records": planned}


def _strict_json(raw: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def build_from_private_sources(worksheet_raw: bytes, lineage_raw: bytes,
                               custody_raw: bytes, media_dir: Path,
                               expected_lineage_sha256: str,
                               expected_custody_sha256: str) -> tuple[dict, list[dict]]:
    worksheet, lineage, custody = map(_strict_json, (worksheet_raw, lineage_raw, custody_raw))
    if hashlib.sha256(lineage_raw).hexdigest() != expected_lineage_sha256:
        raise ValueError("lineage_manifest_hash_mismatch")
    if not isinstance(worksheet, dict) or not isinstance(worksheet.get("records"), list):
        raise ValueError("worksheet_invalid")
    failures = audit_custody(worksheet, worksheet_raw, lineage, lineage_raw,
                             custody, custody_raw, media_dir, expected_custody_sha256)
    if failures:
        raise ValueError("custody_audit_failed")
    records = []
    provenance_records = []
    for row in worksheet["records"]:
        token = row["review_token"]
        media_path = media_dir / (token + ".mp4")
        media_bytes = media_path.read_bytes()
        media_sha256 = hashlib.sha256(media_bytes).hexdigest()
        pts, time_base, _times, pixel_spec, _pixel_hashes, stream_info = _stream_and_decode(io.BytesIO(media_bytes))
        if media_sha256 != row.get("video_sha256"):
            raise ValueError("anchor_source_media_mismatch")
        records.append({"review_token": token, "source_media_sha256": media_sha256,
                        "pts": pts, "time_base": (time_base.numerator, time_base.denominator)})
        pts_bytes = json.dumps(pts, separators=(",", ":")).encode("ascii")
        provenance_records.append({
            "review_token": token,
            "staged_media_sha256": media_sha256,
            "decoded_frame_count": len(pts),
            "decoded_pts_sequence_sha256": hashlib.sha256(pts_bytes).hexdigest(),
            "time_base": [time_base.numerator, time_base.denominator],
            "pixel_spec": list(pixel_spec),
            "decoder_stream_info": {
                key: str(value) if isinstance(value, Fraction) else value
                for key, value in stream_info.items()
            },
        })
    return build_anchor_plan(records), provenance_records


def _write_private_exclusive(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _write_private_pair_exclusive(plan_path: Path, plan_raw: bytes,
                                  manifest_path: Path, manifest_raw: bytes) -> None:
    """Install a same-directory private pair; roll back on ordinary failures."""
    if plan_path.parent != manifest_path.parent or plan_path == manifest_path:
        raise ValueError("plan_manifest_output_directory_invalid")
    suffix = secrets.token_hex(12)
    temps = [plan_path.parent / f".{plan_path.name}.{suffix}.tmp",
             plan_path.parent / f".{manifest_path.name}.{suffix}.tmp"]
    installed: list[Path] = []
    try:
        for path, raw in zip(temps, (plan_raw, manifest_raw), strict=True):
            _write_private_exclusive(path, raw)
        for temp, target in zip(temps, (plan_path, manifest_path), strict=True):
            os.link(temp, target, follow_symlinks=False)
            installed.append(target)
        directory_fd = os.open(plan_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        for path in installed:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for path in temps:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _build_provenance_manifest(*, worksheet_raw: bytes, lineage_raw: bytes,
                               custody_raw: bytes, plan_raw: bytes,
                               source_bytes: dict[str, bytes],
                               provenance_records: list[dict], created_at_utc: str) -> dict:
    return {
        "schema_version": 1,
        "purpose": "private_clock_anchor_plan_candidate_provenance",
        "status": "candidate_not_jointly_frozen",
        "execution_authorization": "not_granted",
        "input_sha256": {
            "worksheet": hashlib.sha256(worksheet_raw).hexdigest(),
            "lineage_manifest": hashlib.sha256(lineage_raw).hexdigest(),
            "custody_manifest": hashlib.sha256(custody_raw).hexdigest(),
        },
        "source_sha256": {name: hashlib.sha256(raw).hexdigest()
                          for name, raw in source_bytes.items()},
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "created_at_utc": created_at_utc,
        "python_version": sys.version.split()[0],
        "pyav_version": av.__version__,
        "ffmpeg_library_versions": {key: list(value) for key, value in av.library_versions.items()},
        "joint_freeze_receipt_sha256": None,
        "records": provenance_records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", required=True)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--custody-manifest", required=True)
    parser.add_argument("--expected-custody-sha256", required=True)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    paths = [Path(args.worksheet), Path(args.lineage_manifest), Path(args.custody_manifest)]
    try:
        for path in paths:
            info = path.lstat()
            if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
                raise ValueError("private_input_mode_invalid")
        repo_root = Path(__file__).resolve().parents[1]
        private_dir = repo_root / "data" / "private_nextgqa"
        output = validate_private_output(args.output, repo_root)
        manifest_output = validate_private_output(args.manifest_output, repo_root)
        if output.parent != private_dir or manifest_output.parent != private_dir:
            raise ValueError("plan_manifest_outputs_must_be_private")
        if output == manifest_output:
            raise ValueError("plan_manifest_outputs_must_differ")
        worksheet_raw, lineage_raw, custody_raw = (path.read_bytes() for path in paths)
        plan, provenance_records = build_from_private_sources(
            worksheet_raw, lineage_raw, custody_raw, Path(args.media_dir),
            args.expected_lineage_sha256, args.expected_custody_sha256)
        raw = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8") + b"\n"
        source_paths = {
            "builder": Path(__file__),
            "decoder": Path(__file__).with_name("claim_mask_video_decoder.py"),
            "custody_auditor": Path(__file__).with_name("audit_nextgqa_review_handoff.py"),
            "private_output_policy": Path(__file__).with_name("private_output_paths.py"),
        }
        manifest = _build_provenance_manifest(
            worksheet_raw=worksheet_raw, lineage_raw=lineage_raw,
            custody_raw=custody_raw, plan_raw=raw,
            source_bytes={name: path.read_bytes() for name, path in source_paths.items()},
            provenance_records=provenance_records,
            created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        manifest_raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        _write_private_pair_exclusive(output, raw, manifest_output, manifest_raw)
    except FileExistsError:
        parser.error("refusing_to_overwrite_output")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "anchor_plan_invalid")
    print("private candidate anchor plan created; records=3; execution_authorization=not_granted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
