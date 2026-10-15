#!/usr/bin/env python3
"""Counts-only local audit of NExT-GQA annotation times against decoded PTS.

Reads local benchmark annotations and media but writes only aggregate timing
diagnostics: no IDs, QA text, paths, frames, or annotation payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

import av


def _dist(values):
    if not values:
        return None
    values = sorted(values)
    return {"n": len(values), "min": values[0], "median": statistics.median(values),
            "p95": values[min(len(values) - 1, math.ceil(.95 * len(values)) - 1)],
            "max": values[-1], "mean": statistics.fmean(values)}


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _key(value):
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    if parts:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return "/".join(parts[-2:])


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_clock(path):
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError("video_stream_count")
        stream = streams[0]
        stream.codec_context.thread_count = 1
        pts_times = []
        frame_rate = []
        for frame in container.decode(stream):
            if frame.is_corrupt or frame.pts is None:
                raise ValueError("frame_pts_or_corruption")
            tb = frame.time_base if frame.time_base is not None else stream.time_base
            if tb is None or tb <= 0:
                raise ValueError("invalid_time_base")
            pts_times.append(int(frame.pts) * float(tb))
            if frame.duration and frame.duration > 0:
                frame_rate.append(1.0 / (int(frame.duration) * float(tb)))
    if len(pts_times) < 2:
        raise ValueError("insufficient_frames")
    deltas = [b - a for a, b in zip(pts_times, pts_times[1:])]
    if any(not math.isfinite(delta) or delta <= 0 for delta in deltas):
        raise ValueError("nonmonotonic_pts")
    return pts_times, statistics.median(frame_rate) if frame_rate else 1 / statistics.median(deltas)


def audit(repo: Path, video_root: Path, limit: int):
    dataset = repo / "NExT-GQA" / "datasets" / "nextgqa"
    split = "test"
    video_map = json.loads((dataset / "map_vid_vidorID.json").read_text(encoding="utf-8"))
    gsub = json.loads((dataset / "gsub_test.json").read_text(encoding="utf-8"))
    frame2time = json.loads((dataset / "frame2time_test.json").read_text(encoding="utf-8"))
    paths_by_key = {}
    for path in video_root.rglob("*.mp4"):
        paths_by_key.setdefault(_key(path.relative_to(video_root)), []).append(path)
    by_key = {key: paths[0] for key, paths in paths_by_key.items() if len(paths) == 1}
    ambiguous_media_keys = sum(len(paths) > 1 for paths in paths_by_key.values())
    used = set()
    manifest_hashes = []
    for manifest in (repo / "data").glob("nextgqa_samples_*.json"):
        try:
            manifest_bytes = manifest.read_bytes()
            manifest_hashes.append(_sha(manifest))
            rows = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rows, list):
            used.update(_key(row["video"]) for row in rows
                        if isinstance(row, dict) and isinstance(row.get("video"), str))
    questions = {}
    with (dataset / f"{split}.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            vid, qid = str(row["video_id"]), str(row["qid"])
            if vid not in video_map:
                continue
            info = gsub.get(vid)
            loc = info.get("location") if isinstance(info, dict) else None
            spans = loc.get(qid) if isinstance(loc, dict) else None
            if not isinstance(info, dict) or not isinstance(spans, list) or not spans:
                continue
            try:
                duration = float(info["duration"])
                spans = [(float(s[0]), float(s[1])) for s in spans]
                fps = _finite(info.get("fps"))
            except (KeyError, TypeError, ValueError, IndexError, OverflowError):
                continue
            key = _key(video_map[vid])
            if key not in by_key or key in used or duration <= 0:
                continue
            if not all(math.isfinite(a) and math.isfinite(b) and 0 <= a <= b <= duration
                       for a, b in spans):
                continue
            record = questions.setdefault(vid, {"key": key, "duration": duration,
                                                  "fps": fps, "spans": [], "qids": set()})
            if qid not in record["qids"]:
                record["spans"].extend(spans)
                record["qids"].add(qid)

    ordered = sorted(questions.items(), key=lambda item: hashlib.sha256(
        item[0].encode("utf-8")).digest())[:limit]
    origin, duration_errors, fps_errors, grid_nearest = [], [], [], []
    frame2time_point_count = 0
    frame2time_invalid_point_count = 0
    frame2time_before_pts_count = 0
    frame2time_after_last_pts_count = 0
    frame2time_max_out_of_range_seconds = 0.0
    interval_frame_coverage = []
    sampled_span_coverage = {count: {"all": [], "within_1ms": [], "within_50ms": []}
                             for count in (4, 8, 16, 32, 64)}
    questions_per_video = []
    regularity = {count: {"tested": 0, "within_1ms": 0, "within_50ms": 0,
                          "max_deviation_seconds": []}
                  for count in (4, 8, 16, 32, 64)}
    selection_hash_rows = []
    failures = 0
    for vid, row in ordered:
        try:
            path = by_key[row["key"]]
            times, decoded_fps = _decode_clock(path)
        except (OSError, ValueError, av.error.FFmpegError):
            failures += 1
            continue
        relative = [t - times[0] for t in times]
        # Compare duration on identical end-PTS basis to avoid inventing an
        # unknown last-frame display duration.
        observed = relative[-1]
        origin.append(times[0])
        duration_errors.append(abs(observed - row["duration"]))
        if row["fps"] and row["fps"] > 0:
            fps_errors.append(abs(decoded_fps - row["fps"]))
        questions_per_video.append(len(row["qids"]))
        selected_samples = {}
        for count in regularity:
            indices = [round(i * (len(relative) - 1) / (count - 1)) for i in range(count)]
            sample_times = [relative[index] for index in indices]
            step = (sample_times[-1] - sample_times[0]) / (count - 1)
            deviation = max(abs(value - (sample_times[0] + i * step))
                            for i, value in enumerate(sample_times))
            regularity[count]["tested"] += 1
            regularity[count]["within_1ms"] += int(deviation <= .001)
            regularity[count]["within_50ms"] += int(deviation <= .05)
            regularity[count]["max_deviation_seconds"].append(deviation)
            selected_samples[count] = (sample_times, deviation <= .001,
                                       deviation <= .05)
        grid = frame2time.get(vid)
        if isinstance(grid, list):
            for raw_point in grid:
                point = _finite(raw_point)
                if point is None:
                    frame2time_invalid_point_count += 1
                    continue
                frame2time_point_count += 1
                if point < 0:
                    frame2time_before_pts_count += 1
                    frame2time_max_out_of_range_seconds = max(
                        frame2time_max_out_of_range_seconds, -point)
                elif point > observed:
                    frame2time_after_last_pts_count += 1
                    frame2time_max_out_of_range_seconds = max(
                        frame2time_max_out_of_range_seconds, point - observed)
                else:
                    grid_nearest.append(min(abs(t - point) for t in relative))
        for start, end in row["spans"]:
            interval_frame_coverage.append(any(start <= t <= end for t in relative))
            for count, (sample_times, pass_1ms, pass_50ms) in selected_samples.items():
                covered = any(start <= t <= end for t in sample_times)
                sampled_span_coverage[count]["all"].append(covered)
                if pass_1ms:
                    sampled_span_coverage[count]["within_1ms"].append(covered)
                if pass_50ms:
                    sampled_span_coverage[count]["within_50ms"].append(covered)
        selection_hash_rows.append(f"{vid}:{_sha(path)}")

    effective = len(origin)
    return {
        "schema_version": 2,
        "audit_type": "local_nextgqa_annotation_seconds_vs_decoded_presentation_timestamps",
        "split": split,
        "selection": "distinct valid annotated local test videos absent from existing sample manifests, deterministic SHA-256 order",
        "requested_videos": limit,
        "available_candidates": len(questions),
        "ambiguous_local_media_keys_excluded": ambiguous_media_keys,
        "private_exclusion_manifest_count": len(manifest_hashes),
        "private_exclusion_manifest_digest_sha256": hashlib.sha256(
            "\n".join(sorted(manifest_hashes)).encode("ascii")).hexdigest(),
        "attempted_videos": len(ordered),
        "effective_videos": effective,
        "decode_failures": failures,
        "first_decoded_pts_seconds_from_zero": _dist(origin),
        "absolute_gsub_vs_last_pts_duration_error_seconds": _dist(duration_errors),
        "absolute_annotation_vs_decoded_fps_error": _dist(fps_errors),
        "frame2time_nearest_decoded_pts_error_seconds": _dist(grid_nearest),
        "frame2time_points_against_decoded_pts_range": {
            "finite_points": frame2time_point_count,
            "invalid_values": frame2time_invalid_point_count,
            "before_first_pts": frame2time_before_pts_count,
            "after_last_pts": frame2time_after_last_pts_count,
            "max_out_of_range_seconds": frame2time_max_out_of_range_seconds,
        },
        "uniform_index_sample_max_deviation_by_frame_count_seconds": {
            str(count): _dist(values["max_deviation_seconds"])
            for count, values in regularity.items()},
        "uniform_index_sample_within_1ms_by_frame_count": {
            str(count): {"tested": values["tested"], "passed": values["within_1ms"],
                         "fraction": values["within_1ms"] / values["tested"]
                         if values["tested"] else None}
            for count, values in regularity.items()},
        "uniform_index_sample_within_50ms_by_frame_count": {
            str(count): {"tested": values["tested"], "passed": values["within_50ms"],
                         "fraction": values["within_50ms"] / values["tested"]
                         if values["tested"] else None}
            for count, values in regularity.items()},
        "decoded_frame_any_overlap_with_span_fraction": (
            sum(interval_frame_coverage) / len(interval_frame_coverage)
            if interval_frame_coverage else None),
        "sampled_frame_any_overlap_by_frame_count": {
            str(count): {
                "all_videos": {"covered": sum(values["all"]),
                               "spans": len(values["all"]),
                               "fraction": sum(values["all"]) / len(values["all"])
                               if values["all"] else None},
                "videos_with_uniform_grid_within_1ms": {
                    "covered": sum(values["within_1ms"]),
                    "spans": len(values["within_1ms"]),
                    "fraction": sum(values["within_1ms"]) / len(values["within_1ms"])
                    if values["within_1ms"] else None},
                "videos_with_uniform_grid_within_50ms": {
                    "covered": sum(values["within_50ms"]),
                    "spans": len(values["within_50ms"]),
                    "fraction": sum(values["within_50ms"]) / len(values["within_50ms"])
                    if values["within_50ms"] else None},
            }
            for count, values in sampled_span_coverage.items()},
        "decoded_frame_span_overlap_count": sum(interval_frame_coverage),
        "sampled_16frame_span_overlap_count": sum(sampled_span_coverage),
        "valid_gold_span_count": len(interval_frame_coverage),
        "videos_with_valid_annotation_question_count": _dist(questions_per_video),
        "private_selected_video_media_fingerprint_sha256": hashlib.sha256(
            "\n".join(selection_hash_rows).encode("utf-8")).hexdigest(),
        "input_sha256": {name: _sha(dataset / name) for name in (
            "test.csv", "gsub_test.json", "map_vid_vidorID.json", "frame2time_test.json")},
        "project_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "dataset_checkout_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo / "NExT-GQA", text=True).strip(),
        "runner_sha256": _sha(Path(__file__)),
        "pyav_version": av.__version__,
        "limitations": [
            "Decoded frame PTS and interval seconds are compared using a first-frame-relative clock; this cannot establish that the dataset annotation player used the same source edit or timestamp origin.",
            "Comparing span overlap with any decoded frame is a geometric audit, not evidence that the frame supports the answer or an explanation.",
            "Video-level cohort is a deterministic feasibility subset, not a probability sample.",
        ],
        "privacy": "No identifiers, questions, answers, raw intervals, paths, or media are emitted.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", default=str(Path.home() / "nextgqa_sample/NExTVideo"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 100:
        parser.error("--limit must be between 1 and 100")
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite existing output")
    report = audit(Path(__file__).resolve().parents[1], Path(args.video_root), args.limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, output)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
    print(f"NExT-GQA PTS audit: {report['effective_videos']}/{report['requested_videos']} "
          f"videos decoded; failures={report['decode_failures']}; "
          f"decoded-frame span coverage={report['decoded_frame_any_overlap_with_span_fraction']}; "
          "sampled counts=" + ",".join(
              f"{count}f:{values['all_videos']['fraction']}"
              for count, values in report["sampled_frame_any_overlap_by_frame_count"].items()))
    print(f"Wrote counts-only local summary: {output.name}")


if __name__ == "__main__":
    main()
