#!/usr/bin/env python3
"""Counts-only local frame2time cross-check for a frozen NExT-GQA pilot cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from run_nextgqa_gold_interval_margin_pilot import (
    freeze_cohort, sha256_file, video_key,
)
from audit_nextgqa_pts_alignment_local import _decode_clock
from necessity_choice_margin_utils import uniform_sample_indices


def distribution(values):
    values = list(values)
    if not values:
        return {"n": 0, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {"n": len(values), "median": statistics.median(values),
            "p95": ordered[min(len(ordered) - 1, int(.95 * (len(ordered) - 1)))],
            "max": max(values)}


def uniform_grid_coverage(decoded_relative, spans, frame_count, tolerance_seconds=0.05):
    """Describe evidence-span coverage for the repository's rounded frame-index sampler."""
    if len(decoded_relative) < frame_count:
        return {"compliant": False, "max_deviation_seconds": None,
                "covered_spans": 0, "total_spans": len(spans)}
    indices = uniform_sample_indices(len(decoded_relative), frame_count)
    sampled_times = [decoded_relative[index] for index in indices]
    interval = (sampled_times[-1] - sampled_times[0]) / (frame_count - 1)
    max_deviation = max(abs(value - (sampled_times[0] + i * interval))
                        for i, value in enumerate(sampled_times))
    compliant = len(set(indices)) == frame_count and max_deviation <= tolerance_seconds
    covered = (sum(any(start <= point <= end for point in sampled_times)
                   for start, end in spans) if compliant else 0)
    return {"compliant": compliant, "max_deviation_seconds": max_deviation,
            "covered_spans": covered, "total_spans": len(spans)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-summary", required=True)
    parser.add_argument("--exclude-media-sha256", required=True,
                        help="private media-hash list used by the frozen pilot cohort")
    parser.add_argument("--pilot-runner-source",
                        help="exact local runner source used by the bound preflight")
    parser.add_argument("--frames", type=int, choices=(16, 32), default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-root", default=str(Path.home() / "nextgqa_sample/NExTVideo"))
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args()
    if not args.user_authorized_local_only:
        parser.error("explicit local-only authorization flag required")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("refusing to overwrite output")
    repo = Path(__file__).resolve().parents[1]
    dataset = repo / "NExT-GQA/datasets/nextgqa"
    gate_path = Path(args.preflight_summary).resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    exclusion_path = Path(args.exclude_media_sha256).resolve()
    exclusion_payload = json.loads(exclusion_path.read_text(encoding="utf-8"))
    excluded_media_sha256 = exclusion_payload.get("media_sha256")
    if (exclusion_payload.get("schema_version") != 1
            or not isinstance(excluded_media_sha256, list)
            or not excluded_media_sha256
            or any(not isinstance(value, str) or len(value) != 64
                   or any(c not in "0123456789abcdef" for c in value)
                   for value in excluded_media_sha256)):
        parser.error("private_media_exclusion_manifest_invalid")
    excluded_media_digest = hashlib.sha256(json.dumps(
        sorted(excluded_media_sha256), ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    runner = (Path(args.pilot_runner_source).resolve() if args.pilot_runner_source
              else repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py")
    runner_sha = sha256_file(runner)
    if (gate.get("status") != "passed" or gate.get("mode") != "outcome_blind_preflight"
            or gate.get("runner_sha256") != runner_sha):
        parser.error("preflight_runner_binding_mismatch")
    selected, counts, meta, cohort_hash = freeze_cohort(
        repo, Path(args.video_root).resolve(), excluded_media_sha256, args.frames)
    if gate.get("cohort_sha256") != cohort_hash:
        parser.error("frozen_cohort_mismatch")
    if (gate.get("prior_cohort_exclusion_file_sha256") != sha256_file(exclusion_path)
            or gate.get("prior_cohort_exclusion_media_digest_sha256")
                != excluded_media_digest):
        parser.error("frozen_exclusion_binding_mismatch")
    frame2time_path = dataset / "frame2time_test.json"
    frame2time = json.loads(frame2time_path.read_text(encoding="utf-8"))
    video_map = json.loads((dataset / "map_vid_vidorID.json").read_text(encoding="utf-8"))
    key_to_ids = {}
    for video_id, value in video_map.items():
        key_to_ids.setdefault(video_key(str(value)), []).append(video_id)
    nearest_errors, full_pts_errors, grid_sizes, decoded_frame_counts = [], [], [], []
    per_video_clock_residuals = []
    frame_grid_coverage = {count: {"grid_compliant_videos": 0, "candidate_videos": 0,
                                   "covered_spans": 0, "eligible_spans": 0}
                           for count in (16, 32, 64)}
    per_video_uniform_span_coverage = []
    span_total = span_frame2time = span_sampled_pts = 0
    ambiguous_keys = missing_vectors = 0
    for video_index, sample in enumerate(selected):
        video_ids = key_to_ids.get(sample["video_key"], [])
        if len(video_ids) != 1:
            ambiguous_keys += 1
            continue
        points = frame2time.get(video_ids[0])
        if not isinstance(points, list):
            missing_vectors += 1
            continue
        points = [float(value) for value in points
                  if isinstance(value, (int, float)) and math.isfinite(float(value))]
        if not points:
            missing_vectors += 1
            continue
        grid_sizes.append(len(points))
        decoded_pts, _ = _decode_clock(Path(sample["video_path"]))
        decoded_relative = [value - decoded_pts[0] for value in decoded_pts]
        decoded_frame_counts.append(len(decoded_relative))
        spans = sample["spans"]
        for frame_count, summary in frame_grid_coverage.items():
            coverage = uniform_grid_coverage(decoded_relative, spans, frame_count)
            if coverage["max_deviation_seconds"] is not None:
                summary["candidate_videos"] += 1
            if coverage["compliant"]:
                summary["grid_compliant_videos"] += 1
                summary["eligible_spans"] += coverage["total_spans"]
                summary["covered_spans"] += coverage["covered_spans"]
            if frame_count == 16:
                per_video_uniform_span_coverage.append({
                    "anonymous_video_index": video_index,
                    "covered_spans": coverage["covered_spans"],
                    "total_spans": coverage["total_spans"],
                    "all_spans_covered": (
                        coverage["compliant"]
                        and coverage["covered_spans"] == coverage["total_spans"]),
                })
        signed_residuals = []
        absolute_residuals = []
        for point in points:
            nearest_pts = min(decoded_relative, key=lambda pts: abs(point - pts))
            residual = point - nearest_pts
            full_pts_errors.append(abs(residual))
            signed_residuals.append(residual)
            absolute_residuals.append(abs(residual))
        worst_index = max(range(len(absolute_residuals)), key=absolute_residuals.__getitem__)
        per_video_clock_residuals.append({
            "anonymous_video_index": video_index,
            "frame2time_points": len(points),
            "signed_residual_median_seconds": statistics.median(signed_residuals),
            "absolute_residual_median_seconds": statistics.median(
                abs(value) for value in signed_residuals),
            "absolute_residual_max_seconds": max(abs(value) for value in signed_residuals),
            "max_residual_frame2time_index": worst_index,
            "max_residual_signed_seconds": signed_residuals[worst_index],
        })
        sampled_pts = [float(value) for value in sample["sampled_times"]]
        nearest_errors.extend(min(abs(pts - point) for point in points)
                              for pts in sampled_pts)
        for start, end in spans:
            span_total += 1
            span_frame2time += int(any(start <= point <= end for point in points))
            span_sampled_pts += int(any(start <= pts <= end for pts in sampled_pts))
    inputs = {name: sha256_file(dataset / name) for name in
              ("test.csv", "gsub_test.json", "map_vid_vidorID.json", "frame2time_test.json")}
    payload = {
        "schema_version": 1,
        "purpose": "local_counts_only_frozen_cohort_frame2time_alignment_audit",
        "authorization_scope": "user-directed local-only; rights not independently verified",
        "status": "passed" if (not ambiguous_keys and not missing_vectors
                               and len(grid_sizes) == len(selected)) else "partial",
        "source_commit": gate.get("source_commit"),
        "audit_runner_sha256": sha256_file(Path(__file__)),
        "pilot_runner_sha256": runner_sha,
        "preflight_sha256": sha256_file(gate_path),
        "private_exclusion_manifest_sha256": sha256_file(exclusion_path),
        "private_exclusion_media_digest_sha256": excluded_media_digest,
        "cohort_sha256": cohort_hash,
        "candidate_video_count": meta["candidate_video_count"],
        "scanned_videos": (counts["timing_rejected_videos"] + counts["timing_pass_videos"]
                           + counts["decode_failures"]),
        "unscanned_videos": meta["candidate_video_count"]
            - counts["timing_rejected_videos"] - counts["timing_pass_videos"]
            - counts["decode_failures"],
        "effective_videos": len(selected),
        "ambiguous_video_keys": ambiguous_keys,
        "missing_frame2time_vectors": missing_vectors,
        "frame2time_points_per_video": distribution(grid_sizes),
        "decoded_frames_per_video": distribution(decoded_frame_counts),
        "frame2time_to_all_decoded_pts_error_seconds": distribution(full_pts_errors),
        "per_video_frame2time_clock_residuals": per_video_clock_residuals,
        "nearest_frame2time_to_sampled_pts_error_seconds": distribution(nearest_errors),
        "gsub_span_overlap": {
            "frame2time_points": {"covered": span_frame2time, "total": span_total,
                                  "fraction": span_frame2time / span_total if span_total else None},
            "pilot_sampled_pts": {"covered": span_sampled_pts, "total": span_total,
                                  "fraction": span_sampled_pts / span_total if span_total else None},
        },
        "uniform_index_grid_span_coverage_50ms_tolerance": {
            str(count): summary for count, summary in frame_grid_coverage.items()
        },
        "per_video_uniform_16_frame_span_coverage": per_video_uniform_span_coverage,
        "input_sha256": inputs,
        "limitations": [
            "This geometric audit does not calibrate the annotation-player/source-edit clock to decoded PTS.",
            "Frame2time grid points and decoded sampled frames are compared by nearest timestamp; this does not establish pixel identity or annotation semantics.",
            "No model scores, QA text, IDs, paths, intervals, or media are emitted.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_name, output)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    print(json.dumps({key: payload[key] for key in (
        "status", "effective_videos", "scanned_videos", "unscanned_videos",
        "ambiguous_video_keys", "missing_frame2time_vectors", "frame2time_points_per_video",
        "decoded_frames_per_video", "frame2time_to_all_decoded_pts_error_seconds",
        "per_video_frame2time_clock_residuals",
        "nearest_frame2time_to_sampled_pts_error_seconds", "gsub_span_overlap",
        "uniform_index_grid_span_coverage_50ms_tolerance")}, sort_keys=True))


if __name__ == "__main__":
    main()
