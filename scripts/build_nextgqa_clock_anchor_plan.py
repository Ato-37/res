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
import stat
from typing import Any

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
                               expected_custody_sha256: str) -> dict:
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
    for row in worksheet["records"]:
        token = row["review_token"]
        media_path = media_dir / (token + ".mp4")
        media_bytes = media_path.read_bytes()
        media_sha256 = hashlib.sha256(media_bytes).hexdigest()
        pts, time_base, _times, _spec, _pixel_hashes, _info = _stream_and_decode(io.BytesIO(media_bytes))
        if media_sha256 != row.get("video_sha256"):
            raise ValueError("anchor_source_media_mismatch")
        records.append({"review_token": token, "source_media_sha256": media_sha256,
                        "pts": pts, "time_base": (time_base.numerator, time_base.denominator)})
    return build_anchor_plan(records)


def _write_private_exclusive(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", required=True)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--custody-manifest", required=True)
    parser.add_argument("--expected-custody-sha256", required=True)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--output", required=True)
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
        output = validate_private_output(args.output, Path(__file__).resolve().parents[1])
        plan = build_from_private_sources(paths[0].read_bytes(), paths[1].read_bytes(),
                                          paths[2].read_bytes(), Path(args.media_dir),
                                          args.expected_lineage_sha256,
                                          args.expected_custody_sha256)
        raw = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8") + b"\n"
        _write_private_exclusive(output, raw)
    except FileExistsError:
        parser.error("refusing_to_overwrite_output")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "anchor_plan_invalid")
    print("private candidate anchor plan created; records=3; execution_authorization=not_granted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
