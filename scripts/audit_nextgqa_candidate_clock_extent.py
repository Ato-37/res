#!/usr/bin/env python3
"""Private, aggregate-only duration/PTS diagnostic for a frozen 3-video claim set.

This reads local media and only the dataset duration and video-path mapping
fields. It never emits identifiers, claims, questions, annotation spans, or
frames. Duration agreement is not annotation-clock calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import tempfile
from typing import Any

import av

from audit_nextgqa_claim_packet_lineage import DATASET_KEYS, LINEAGE_KEYS, LINEAGE_ROW_KEYS
from claim_mask_video_decoder import _stream_and_decode, _file_sha256
from run_nextgqa_gold_interval_margin_pilot import video_key


SHA256 = re.compile(r"^[a-f0-9]{64}$")
MAX_MANIFEST_BYTES = 1_000_000


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def _read_json(path: Path):
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    return value, raw


def _skip_json_value(text: str, pos: int) -> int:
    """Advance over one JSON value without constructing nested annotation data."""
    decoder = json.JSONDecoder(parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    if pos >= n:
        raise ValueError("gsub_json_truncated")
    if text[pos] == '"':
        _, end = decoder.raw_decode(text, pos)
        return end
    if text[pos] in "[{":
        opening = text[pos]; closing = "]" if opening == "[" else "}"
        pos += 1
        while pos < n and text[pos].isspace(): pos += 1
        if pos < n and text[pos] == closing: return pos + 1
        while True:
            if opening == "{":
                _, pos = decoder.raw_decode(text, pos)
                while pos < n and text[pos].isspace(): pos += 1
                if pos >= n or text[pos] != ":": raise ValueError("gsub_json_invalid")
                pos += 1
            pos = _skip_json_value(text, pos)
            while pos < n and text[pos].isspace(): pos += 1
            if pos >= n: raise ValueError("gsub_json_truncated")
            if text[pos] == closing: return pos + 1
            if text[pos] != ",": raise ValueError("gsub_json_invalid")
            pos += 1
            while pos < n and text[pos].isspace(): pos += 1
    value, end = decoder.raw_decode(text, pos)
    if isinstance(value, (dict, list)): raise ValueError("gsub_json_invalid")
    return end


def _duration_from_gsub_value(text: str, pos: int) -> tuple[Any, int]:
    """Read only a candidate record's duration field; skip all span payloads."""
    decoder = json.JSONDecoder(object_pairs_hook=_unique,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    while pos < len(text) and text[pos].isspace(): pos += 1
    if pos >= len(text) or text[pos] != "{":
        raise ValueError("gsub_record_invalid")
    pos += 1
    duration = None
    found = False
    seen = set()
    has_items = False
    expect_item = True
    while True:
        while pos < len(text) and text[pos].isspace(): pos += 1
        if pos >= len(text): raise ValueError("gsub_json_truncated")
        if text[pos] == "}":
            if not expect_item or not has_items:
                return (duration, pos + 1) if found else (None, pos + 1)
            raise ValueError("gsub_json_invalid")
        if not expect_item:
            if text[pos] != ",": raise ValueError("gsub_json_invalid")
            pos += 1
            while pos < len(text) and text[pos].isspace(): pos += 1
            expect_item = True
        key, end = decoder.raw_decode(text, pos)
        if not isinstance(key, str): raise ValueError("gsub_record_invalid")
        if key in seen: raise ValueError("gsub_duplicate_key")
        seen.add(key)
        pos = end
        while pos < len(text) and text[pos].isspace(): pos += 1
        if pos >= len(text) or text[pos] != ":": raise ValueError("gsub_json_invalid")
        pos += 1
        while pos < len(text) and text[pos].isspace(): pos += 1
        if key == "duration":
            duration, end = decoder.raw_decode(text, pos)
            found = True
            pos = end
        else:
            pos = _skip_json_value(text, pos)
        expect_item = False
        has_items = True


def _read_candidate_durations(path: Path, candidate_ids: set[str]) -> dict[str, Any]:
    # Read text but never decode annotation values into Python containers.
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder(object_pairs_hook=_unique)
    pos = 0
    while pos < len(text) and text[pos].isspace(): pos += 1
    if pos >= len(text) or text[pos] != "{": raise ValueError("gsub_root_invalid")
    pos += 1
    result = {}
    seen_root = set()
    expect_item = True
    has_items = False
    while True:
        while pos < len(text) and text[pos].isspace(): pos += 1
        if pos >= len(text): raise ValueError("gsub_json_truncated")
        if text[pos] == "}":
            if expect_item and has_items: raise ValueError("gsub_json_invalid")
            pos += 1; break
        if not expect_item:
            if text[pos] != ",": raise ValueError("gsub_json_invalid")
            pos += 1
            while pos < len(text) and text[pos].isspace(): pos += 1
            expect_item = True
        key, end = decoder.raw_decode(text, pos)
        if not isinstance(key, str): raise ValueError("gsub_root_invalid")
        if key in seen_root: raise ValueError("gsub_duplicate_key")
        seen_root.add(key)
        pos = end
        while pos < len(text) and text[pos].isspace(): pos += 1
        if pos >= len(text) or text[pos] != ":": raise ValueError("gsub_json_invalid")
        pos += 1
        if key in candidate_ids:
            result[key], pos = _duration_from_gsub_value(text, pos)
        else:
            pos = _skip_json_value(text, pos)
        expect_item = False
        has_items = True
    if text[pos:].strip(): raise ValueError("gsub_json_trailing_data")
    return result


def _frame2time_diagnostics(points: Any, decoded_relative_pts: list[Fraction]) -> dict:
    if (not decoded_relative_pts or not isinstance(points, list) or not points
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) or float(value) < 0
                   for value in points)):
        raise ValueError("dataset_frame2time_invalid")
    frame_times = [Fraction(str(value)) for value in points]
    if any(right < left for left, right in zip(frame_times, frame_times[1:])):
        raise ValueError("dataset_frame2time_nonmonotonic")
    nearest_indices = [min(range(len(decoded_relative_pts)),
                           key=lambda index: abs(point - decoded_relative_pts[index]))
                       for point in frame_times]
    errors = [abs(point - decoded_relative_pts[index])
              for point, index in zip(frame_times, nearest_indices)]
    return {
        "point_count": len(frame_times),
        "strictly_increasing": all(b > a for a, b in zip(frame_times, frame_times[1:])),
        "duplicate_point_count": len(frame_times) - len(set(frame_times)),
        "outside_decoded_pts_range_count": sum(
            point < decoded_relative_pts[0] or point > decoded_relative_pts[-1]
            for point in frame_times),
        "nearest_decoded_pts_collision_count": len(nearest_indices) - len(set(nearest_indices)),
        "nearest_residual_median": statistics.median(errors),
        "nearest_residual_max": max(errors),
    }


def _ratio(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize_media(path: Path, dataset_duration: Any, expected_source_sha256: str,
                   frame2time: Any) -> dict:
    if (isinstance(dataset_duration, bool)
            or not isinstance(dataset_duration, (int, float))
            or not math.isfinite(float(dataset_duration))
            or float(dataset_duration) <= 0):
        raise ValueError("dataset_duration_invalid")
    annotated_duration = Fraction(str(dataset_duration))
    with path.open("rb") as source:
        source_sha = _file_sha256(source)
        if source_sha != expected_source_sha256:
            raise ValueError("source_media_hash_mismatch_or_duplicate")
        source_size = os.fstat(source.fileno()).st_size
        try:
            pts, time_base, times, pixel_spec, pixel_hashes, stream = _stream_and_decode(source)
        except (av.error.FFmpegError, ValueError, OSError):
            raise ValueError("source_media_decode_failed") from None
        if _file_sha256(source) != source_sha:
            raise ValueError("source_media_changed_during_decode")
    if len(pts) < 2:
        raise ValueError("decoded_frame_count_invalid")
    deltas = [right - left for left, right in zip(times, times[1:])]
    last_frame_duration = deltas[-1]
    decoded_extent = times[-1] + last_frame_duration
    frame2time_stats = _frame2time_diagnostics(frame2time, times)
    return {
        "source_sha256": source_sha,
        "source_size_bytes": source_size,
        "codec": stream["codec_name"],
        "decoded_frame_count": len(pts),
        "pts_strictly_increasing": all(right > left for left, right in zip(pts, pts[1:])),
        "decoded_time_base": _ratio(time_base),
        "first_absolute_pts_seconds": _ratio(Fraction(pts[0]) * time_base),
        "last_relative_pts_seconds": _ratio(times[-1]),
        "frame_interval_min_seconds": _ratio(min(deltas)),
        "frame_interval_median_seconds": _ratio(statistics.median(deltas)),
        "frame_interval_max_seconds": _ratio(max(deltas)),
        "distinct_decoded_rgb_hashes": len(set(pixel_hashes)),
        "rgb_size": [pixel_spec[0], pixel_spec[1]],
        "pixel_format": pixel_spec[2],
        "dataset_duration_seconds": _ratio(annotated_duration),
        "last_pts_plus_previous_pts_interval_seconds": _ratio(decoded_extent),
        "dataset_minus_decoded_extent_seconds": _ratio(annotated_duration - decoded_extent),
        "frame2time_point_count": frame2time_stats["point_count"],
        "frame2time_strictly_increasing": frame2time_stats["strictly_increasing"],
        "frame2time_duplicate_point_count": frame2time_stats["duplicate_point_count"],
        "frame2time_outside_decoded_pts_range_count": frame2time_stats["outside_decoded_pts_range_count"],
        "frame2time_nearest_pts_collision_count": frame2time_stats["nearest_decoded_pts_collision_count"],
        "frame2time_nearest_decoded_pts_median_error_seconds": _ratio(
            frame2time_stats["nearest_residual_median"]),
        "frame2time_nearest_decoded_pts_max_error_seconds": _ratio(
            frame2time_stats["nearest_residual_max"]),
    }


def audit(manifest: dict, manifest_raw: bytes, expected_manifest_sha256: str,
          repo: Path, video_root: Path) -> dict:
    if len(manifest_raw) > MAX_MANIFEST_BYTES:
        raise ValueError("lineage_manifest_too_large")
    parsed = json.loads(manifest_raw.decode("utf-8"), object_pairs_hook=_unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if (parsed != manifest or not isinstance(manifest, dict)
            or not SHA256.fullmatch(expected_manifest_sha256)
            or manifest_sha != expected_manifest_sha256
            or len(manifest_raw) > MAX_MANIFEST_BYTES
            or manifest.get("schema_version") != 1
            or set(manifest) != LINEAGE_KEYS
            or manifest.get("purpose") != "private_source_lineage_crosscheck_for_claim_review"
            or manifest.get("status") != "metadata_lineage_crosschecked"
            or manifest.get("requested_records") != 3
            or manifest.get("effective_records") != 3
            or not isinstance(manifest.get("records"), list)
            or len(manifest["records"]) != 3):
        raise ValueError("lineage_manifest_invalid_or_hash_mismatch")
    dataset = repo / "NExT-GQA" / "datasets" / "nextgqa"
    hashes = {key: hashlib.sha256((dataset / key).read_bytes()).hexdigest()
              for key in DATASET_KEYS}
    if hashes != manifest.get("dataset_inputs_sha256"):
        raise ValueError("dataset_input_hash_mismatch")
    mapping, _ = _read_json(dataset / "map_vid_vidorID.json")
    frame2time, _ = _read_json(dataset / "frame2time_test.json")
    records = manifest["records"]
    candidate_ids = {row.get("dataset_video_id") for row in records if isinstance(row, dict)}
    durations = _read_candidate_durations(dataset / "gsub_test.json", candidate_ids)
    root = video_root.resolve()
    summaries = []
    seen_hashes = set()
    for row in manifest["records"]:
        if not isinstance(row, dict) or set(row) != LINEAGE_ROW_KEYS:
            raise ValueError("lineage_row_invalid")
        relative = row["relative_video_path"]
        if (not isinstance(relative, str) or "\\" in relative
                or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts
                or row["video_key"] != video_key(relative)):
            raise ValueError("relative_media_path_invalid")
        vid = row["dataset_video_id"]
        if (not isinstance(vid, str) or vid not in mapping
                or video_key(mapping[vid]) != row["video_key"]):
            raise ValueError("dataset_video_mapping_mismatch")
        media = (root / relative).resolve()
        if not media.is_relative_to(root) or not media.is_file():
            raise ValueError("source_media_missing_or_outside_root")
        if vid not in durations or durations[vid] is None:
            raise ValueError("dataset_duration_missing")
        if not SHA256.fullmatch(str(row["video_sha256"])):
            raise ValueError("source_media_hash_mismatch_or_duplicate")
        points = frame2time.get(vid)
        summary = summarize_media(media, durations[vid], row["video_sha256"], points)
        media_sha = summary["source_sha256"]
        if media_sha in seen_hashes:
            raise ValueError("source_media_hash_mismatch_or_duplicate")
        seen_hashes.add(media_sha)
        summaries.append(summary)
    source_hashes = sorted(row["source_sha256"] for row in summaries)
    frame_delta_summaries = sorted({
        (tuple(row["frame_interval_min_seconds"]),
         tuple(row["frame_interval_median_seconds"]),
         tuple(row["frame_interval_max_seconds"])) for row in summaries
    })
    return {
        "schema_version": 1,
        "purpose": "private_aggregate_nextgqa_source_clock_extent_diagnostic",
        "status": "metadata_decoder_extent_compared_not_calibrated",
        "annotation_clock_calibrated": False,
        "requested_records": 3,
        "effective_records": len(summaries),
        "decode_failures": 0,
        "source_lineage_manifest_sha256": manifest_sha,
        "dataset_input_sha256": hashes,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "project_commit": _git_head(repo),
        "pyav_version": av.__version__,
        "media_set_fingerprint_sha256": hashlib.sha256(
            "\n".join(source_hashes).encode("ascii")).hexdigest(),
        "pts_strictly_increasing_count": sum(
            row["pts_strictly_increasing"] for row in summaries),
        "codec_counts": dict(sorted(Counter(row["codec"] for row in summaries).items())),
        "decoded_frame_counts": sorted(row["decoded_frame_count"] for row in summaries),
        "time_base_counts": dict(sorted(Counter(
            ":".join(map(str, row["decoded_time_base"])) for row in summaries).items())),
        "relative_duration_seconds": sorted(
            row["last_relative_pts_seconds"] for row in summaries),
        "first_absolute_pts_seconds": sorted(
            row["first_absolute_pts_seconds"] for row in summaries),
        "frame_interval_min_median_max_seconds": [
            [list(a), list(b), list(c)] for a, b, c in frame_delta_summaries],
        "dataset_duration_seconds": sorted(
            row["dataset_duration_seconds"] for row in summaries),
        "duration_difference_seconds": sorted(
            row["dataset_minus_decoded_extent_seconds"] for row in summaries),
        "frame2time_points_per_video": sorted(row["frame2time_point_count"] for row in summaries),
        "frame2time_nearest_pts_median_errors_seconds": sorted(
            row["frame2time_nearest_decoded_pts_median_error_seconds"] for row in summaries),
        "frame2time_nearest_pts_max_errors_seconds": sorted(
            row["frame2time_nearest_decoded_pts_max_error_seconds"] for row in summaries),
        "frame2time_outside_decoded_pts_range_counts": sorted(
            row["frame2time_outside_decoded_pts_range_count"] for row in summaries),
        "frame2time_nearest_pts_collision_counts": sorted(
            row["frame2time_nearest_pts_collision_count"] for row in summaries),
        "rgb_size_counts": dict(sorted(Counter(
            f"{row['rgb_size'][0]}x{row['rgb_size'][1]}" for row in summaries).items())),
        "interpretation_limit": (
            "The terminal extent extrapolates the preceding PTS interval; it is not a decoded "
            "terminal-frame duration. Whole-second duration agreement cannot identify source-player "
            "origin, seek behavior, edit alignment, drift, or annotation-clock calibration. "
            "Frame2time nearest-PTS residuals are geometric clock-consistency checks only; they "
            "do not prove frame identity or source-player/annotation calibration."
        ),
        "execution_authorization": "not_granted",
    }


def write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    manifest_path, output, repo = (Path(args.lineage_manifest).resolve(), args.output.resolve(),
                                   args.repo.resolve())
    if output.is_relative_to(repo) or output.exists():
        parser.error("output_must_be_exclusive_and_outside_repository")
    try:
        if (manifest_path.stat().st_mode & 0o777) != 0o600:
            raise ValueError("private_input_mode_invalid")
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("lineage_manifest_too_large")
        with manifest_path.open("rb") as stream:
            raw = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError("lineage_manifest_too_large")
        manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        report = audit(manifest, raw, args.expected_lineage_sha256, repo,
                       args.video_root.resolve())
        encoded = (json.dumps(report, ensure_ascii=False, sort_keys=True,
                              indent=2, allow_nan=False) + "\n").encode()
        write_exclusive(output, encoded)
    except FileExistsError:
        parser.error("output_must_be_exclusive_and_outside_repository")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError,
            KeyError, OverflowError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "clock_extent_audit_failed")
    print("Private aggregate source-clock extent diagnostic completed; calibration=false; scoring=not_granted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
