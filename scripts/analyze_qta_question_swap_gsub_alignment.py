#!/usr/bin/env python3
"""Compare frozen QTA question-selector groups with NExT-GQA time labels.

This uses the already-run selector-only 80-video artifact and official QA
temporal annotations. It is an exploratory localization analysis, not a
prediction-dependence or explanation-faithfulness result. All output is local
and aggregate-only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import subprocess
import sys
from typing import Any

import av

from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
from qta_question_swap_gsub_alignment import (
    exact_two_group_mean_iou, interval_union_duration,
    intervals_within_media, map_intervals_with_processor_grid_check, set_iou,
    validate_selected_groups, validate_source_clock,
    validate_video_question_pair, video_support_end, conservative_video_support_end,
)

FRAME_COUNT = 32
TEMPORAL_GROUPS = 16
TEMPORAL_PATCH_SIZE = 2
TOLERANCE = Fraction(1, 20)
PRIVATE_ROOT = Path(__file__).resolve().parents[1] / "data" / "private_nextgqa"
DEFAULT_PLAN = PRIVATE_ROOT / "qta_question_swap_cohort_20261019_0020.json"
DEFAULT_SELECTOR = PRIVATE_ROOT / "qta_question_swap_full_20261019_0030.json"
DATASET_ROOT = Path(__file__).resolve().parents[1] / "NExT-GQA" / "datasets" / "nextgqa"
SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "qta_question_swap_gsub_alignment.py",
    Path(__file__).resolve().parent / "claim_mask_video_decoder.py",
    Path(__file__).resolve().parent / "presentation_time_utils.py",
    Path(__file__).resolve().parent / "necessity_choice_margin_utils.py",
    DATASET_ROOT.parents[1] / "code" / "TempGQA" / "tools" / "getspan.py",
    DATASET_ROOT / "README.md",
)
DATA_PATHS = tuple(DATASET_ROOT / name for name in (
    "test.csv", "gsub_test.json", "frame2time_test.json", "upbd_test.json",
    "map_vid_vidorID.json"))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_private(path: Path) -> tuple[bytes, Any]:
    root = PRIVATE_ROOT.resolve(strict=True)
    info_root = root.stat(follow_symlinks=False)
    if (stat.S_IMODE(info_root.st_mode) != 0o700 or info_root.st_uid != os.geteuid()
            or path.is_symlink() or path.parent.resolve(strict=True) != root):
        raise ValueError("private_input_path_invalid")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600 or before.st_size < 1
                or before.st_size > 32_000_000):
            raise ValueError("private_input_file_invalid")
        chunks = []
        remaining = before.st_size
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(fd)
        if (remaining or os.read(fd, 1) or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ctime_ns != after.st_ctime_ns):
            raise ValueError("private_input_changed_during_read")
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    return raw, json.loads(raw.decode("utf-8"))


def _write_private_exclusive(path: Path, payload: dict[str, Any]) -> None:
    root = PRIVATE_ROOT.resolve(strict=True)
    root_info = root.stat(follow_symlinks=False)
    if (stat.S_IMODE(root_info.st_mode) != 0o700 or root_info.st_uid != os.geteuid()
            or path.is_symlink() or path.parent.resolve(strict=True) != root
            or os.path.lexists(path)):
        raise ValueError("private_output_path_invalid_or_exists")
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def leave_one_video_out_summary(video_pair_values: list[float]) -> dict[str, Any]:
    """Summarize finite-cohort influence without retaining video identifiers."""
    if (len(video_pair_values) < 2
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(float(value)) for value in video_pair_values)):
        raise ValueError("video_pair_values_invalid")
    values = [float(value) for value in video_pair_values]
    full_mean = math.fsum(values) / len(values)
    leave_one_out = [(math.fsum(values) - value) / (len(values) - 1)
                     for value in values]
    return {
        "n_video_pairs": len(values),
        "full_mean": full_mean,
        "leave_one_out_mean_min": min(leave_one_out),
        "leave_one_out_mean_max": max(leave_one_out),
        "max_absolute_leave_one_out_shift": max(
            abs(value - full_mean) for value in leave_one_out),
        "inference_performed": False,
    }


def _question_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    output = {}
    for row in rows:
        key = (str(row.get("video_id", "")), str(row.get("qid", "")))
        if not all(key) or key in output:
            raise ValueError("dataset_question_keys_invalid_or_duplicate")
        output[key] = {"question": str(row.get("question", "")).strip()}
    return output


def _git_bytes(repo: Path, relative: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show",
                                    f"HEAD:{relative}"], stderr=subprocess.DEVNULL)


def _intervals_for(annotations: dict, video_id: str, qid: str) -> list[list[float]]:
    video = annotations.get(video_id)
    locations = video.get("location") if isinstance(video, dict) else None
    intervals = locations.get(qid) if isinstance(locations, dict) else None
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("official_temporal_annotation_missing")
    duration = video.get("duration")
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration)) or duration <= 0):
        raise ValueError("official_annotation_duration_invalid")
    checked = []
    for pair in intervals:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(float(x)) for x in pair)
                or pair[0] < 0 or pair[1] <= pair[0] or pair[1] > duration):
            raise ValueError("official_temporal_interval_invalid")
        checked.append([float(pair[0]), float(pair[1])])
    return checked


def _score_one_selection(*, selected: tuple[int, ...], evidence: tuple[int, ...],
                         sampled_pts: list[Fraction], intervals: list[list[float]],
                         duration: float) -> dict[str, float | int]:
    inter = len(set(selected) & set(evidence))
    selected_support = []
    for group in selected:
        first = group * TEMPORAL_PATCH_SIZE
        last = first + TEMPORAL_PATCH_SIZE - 1
        start = 0.0 if first == 0 else float(
            (sampled_pts[first - 1] + sampled_pts[first]) / 2)
        end = duration if last == FRAME_COUNT - 1 else float(
            (sampled_pts[last] + sampled_pts[last + 1]) / 2)
        selected_support.append((start, end))
    union_total = float(interval_union_duration(intervals))
    # Merge intervals before the ratio so overlapping labels are counted once.
    merged = []
    for a, b in sorted((float(x), float(y)) for x, y in intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    covered = sum(max(0.0, min(b, right) - max(a, left))
                  for a, b in merged for left, right in selected_support)
    return {
        "group_iou": set_iou(selected, evidence),
        "selected_group_precision": inter / len(selected),
        "evidence_group_recall": inter / len(evidence),
        "interval_time_coverage": min(1.0, covered / union_total),
        "evidence_group_count": len(evidence),
        "selected_evidence_group_count": inter,
    }


def analyze(*, plan_path: Path, selector_path: Path) -> dict[str, Any]:
    analysis_started_at = datetime.now(timezone.utc).isoformat()
    plan_raw, plan = _read_private(plan_path)
    selector_raw, result = _read_private(selector_path)
    if (plan.get("status") != "frozen_local_question_swap_selector_diagnostic_not_faithfulness"
            or plan.get("effective_videos") != 80
            or result.get("status") != "passed" or result.get("faithfulness_result") is not False
            or result.get("local_only") is not True
            or result.get("cohort_plan_sha256") != _sha(plan_raw)
            or result.get("requested_videos") != 80 or result.get("effective_videos") != 80
            or result.get("completed_forwards") != 320
            or result.get("failed_forwards") != 0 or result.get("failed_videos") != 0
            or result.get("frame_count") != FRAME_COUNT
            or result.get("group_count") != TEMPORAL_GROUPS):
        raise ValueError("selector_result_plan_binding_invalid")
    plan_rows, result_rows = plan.get("fixed_records"), result.get("fixed_ordinal_video_metrics_local_only")
    if (not isinstance(plan_rows, list) or len(plan_rows) != 80
            or not isinstance(result_rows, list) or len(result_rows) != 80):
        raise ValueError("fixed_cohort_row_count_invalid")
    qa = _question_map(DATASET_ROOT / "test.csv")
    annotations = json.loads((DATASET_ROOT / "gsub_test.json").read_text(encoding="utf-8"))
    dataset_repo = DATASET_ROOT.parents[1]
    dataset_commit = subprocess.check_output(
        ["git", "-C", str(dataset_repo), "rev-parse", "HEAD"],
        text=True, stderr=subprocess.DEVNULL).strip()
    test_csv_raw = (DATASET_ROOT / "test.csv").read_bytes()
    gsub_raw = (DATASET_ROOT / "gsub_test.json").read_bytes()
    frame2time_raw = (DATASET_ROOT / "frame2time_test.json").read_bytes()
    upbd_raw = (DATASET_ROOT / "upbd_test.json").read_bytes()
    video_map_raw = (DATASET_ROOT / "map_vid_vidorID.json").read_bytes()
    input_dataset_bytes = {path.name: path.read_bytes() for path in DATA_PATHS}
    input_source_hashes = {path.name: _sha(path.read_bytes()) for path in SOURCE_PATHS}
    if (test_csv_raw != _git_bytes(dataset_repo, "datasets/nextgqa/test.csv")
            or gsub_raw != _git_bytes(dataset_repo, "datasets/nextgqa/gsub_test.json")
            or frame2time_raw != _git_bytes(dataset_repo, "datasets/nextgqa/frame2time_test.json")
            or upbd_raw != _git_bytes(dataset_repo, "datasets/nextgqa/upbd_test.json")
            or video_map_raw != _git_bytes(dataset_repo, "datasets/nextgqa/map_vid_vidorID.json")):
        raise ValueError("dataset_bytes_differ_from_nested_git_head")
    frame2time = json.loads(frame2time_raw.decode("utf-8"))
    upbd = json.loads(upbd_raw.decode("utf-8"))
    video_map = json.loads(video_map_raw.decode("utf-8"))
    repo_root = Path(__file__).resolve().parents[1]
    repo_commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True, stderr=subprocess.DEVNULL).strip()
    media_root = Path(plan.get("video_root", "")).resolve(strict=True)
    if not media_root.is_dir():
        raise ValueError("video_root_invalid")

    matched: dict[str, list[float]] = {name: [] for name in (
        "A_matched_iou", "B_matched_iou", "A_cross_iou", "B_cross_iou",
        "A_random_two_group_iou", "B_random_two_group_iou",
        "A_matched_interval_coverage", "B_matched_interval_coverage",
    )}
    failures: dict[str, int] = {}
    failure_exception_types: dict[str, int] = {}
    interval_count = 0
    qid_count = 0
    media_count = 0
    evaluated_media_count = 0
    unlocalized_qid_count = 0
    out_of_media_qid_count = 0
    source_clock_excluded_video_count = 0
    pair_excluded_source_clock_count = 0
    pair_excluded_unlocalized_count = 0
    pair_excluded_out_of_media_count = 0
    pair_excluded_processor_grid_mismatch_count = 0
    group_count_values = []
    duration_gaps = []
    frame_clock_residuals = []
    frame2time_upbd_equal = 0
    upbd_outside_video_count = 0
    upbd_duplicate_pts_match_video_count = 0
    upbd_pts_mismatch_video_count = 0
    checked_qid_count = 0
    mapped_qid_count = 0
    interval_duration_methods: dict[str, int] = {}
    paired_delta = []
    video_pair_delta_means = []
    for ordinal, (planned, observed) in enumerate(zip(plan_rows, result_rows, strict=True), start=1):
        try:
            if (observed.get("fixed_video_ordinal") != ordinal
                    or planned.get("fixed_video_ordinal") != ordinal):
                raise ValueError("cohort_order_mismatch")
            vid = str(planned["video_id"])
            relative_path = Path(planned["video_relative_path"])
            unresolved_media_path = media_root / relative_path
            if relative_path.is_absolute() or unresolved_media_path.is_symlink():
                raise ValueError("media_path_invalid")
            media_path = unresolved_media_path.resolve(strict=True)
            if not media_path.is_relative_to(media_root):
                raise ValueError("media_path_invalid")
            mapped_video_path = video_map.get(vid)
            if not isinstance(mapped_video_path, str):
                raise ValueError("official_video_path_mapping_missing")
            validate_video_question_pair(
                video_id=vid, relative_video_path=relative_path.as_posix(),
                mapped_video_path=mapped_video_path,
                question_a=planned["question_a"], question_b=planned["question_b"],
                question_map=qa)
            qid_count += 2
            media_sha = hashlib.sha256(media_path.read_bytes()).hexdigest()
            if media_sha != planned.get("media_sha256"):
                raise ValueError("media_hash_mismatch")
            frames, metadata = decode_sampled_video(
                media_path, num_frames=FRAME_COUNT,
                uniform_tolerance_seconds=TOLERANCE, include_timing_map=True)
            verify_sampled_frame_pixels(frames, metadata)
            if metadata.get("source_sha256") != media_sha:
                raise ValueError("decoder_media_hash_mismatch")
            sampled = metadata["sampled_frames"]
            timing = metadata["full_timing_map"]
            time_base = Fraction(*timing["time_base"])
            pts = [Fraction(row["pts"]) * time_base for row in sampled]
            raw_pts = [Fraction(value) * time_base for value in timing["pts"]]
            with av.open(str(media_path), mode="r") as container:
                streams = list(container.streams.video)
                if (len(streams) != 1 or container.duration is None
                        or container.start_time is None or container.start_time != 0):
                    raise ValueError("container_clock_metadata_invalid")
                stream = streams[0]
                stream_tb = Fraction(stream.time_base)
                container_duration = Fraction(container.duration, 1_000_000)
            frame_period = statistics.median([b - a for a, b in zip(raw_pts, raw_pts[1:])])
            stream_start = (Fraction(stream.start_time) * stream_tb
                            if stream.start_time is not None else None)
            stream_length = (Fraction(stream.duration) * stream_tb
                             if stream.duration is not None else None)
            stream_end, stream_end_method = video_support_end(
                container_start=Fraction(0), container_duration=container_duration,
                stream_start=stream_start, stream_duration=stream_length,
                last_pts=raw_pts[-1], frame_period=frame_period)
            duration, duration_method = conservative_video_support_end(last_pts=raw_pts[-1])
            interval_duration_methods[duration_method] = interval_duration_methods.get(duration_method, 0) + 1
            if (stream_start is not None and
                    abs(stream_start - raw_pts[0]) > frame_period / 2):
                raise ValueError("video_stream_first_pts_origin_mismatch")
            if (duration <= 0 or raw_pts[0] < 0 or raw_pts[-1] != duration
                    or stream_end > container_duration + frame_period
                    or pts[0] < 0 or pts[-1] > duration
                    or any(b <= a for a, b in zip(pts, pts[1:]))):
                raise ValueError("video_stream_clock_bounds_invalid")
            duration_gaps.append(float(duration - pts[-1]))
            source_times = frame2time.get(vid)
            sampled_times = upbd.get(vid)
            if (not isinstance(source_times, list) or len(source_times) != 32
                    or not isinstance(sampled_times, list) or len(sampled_times) != 32
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(float(x)) for x in source_times)):
                raise ValueError("official_frame_clock_mapping_invalid")
            if (any(isinstance(x, bool) or not isinstance(x, (int, float))
                    or not math.isfinite(float(x)) for x in sampled_times)):
                raise ValueError("official_sampled_clock_mapping_invalid")
            if source_times == sampled_times:
                frame2time_upbd_equal += 1
            try:
                residuals = list(validate_source_clock(
                    source_times=source_times, decoded_pts=raw_pts,
                    media_start=0, media_end=stream_end, tolerance=frame_period / 2))
            except ValueError as exc:
                if str(exc) in {"official_source_clock_outside_video",
                                "official_source_clock_pts_mismatch"}:
                    source_clock_excluded_video_count += 1
                    pair_excluded_source_clock_count += 1
                    media_count += 1
                    continue
                raise
            upbd_values = [Fraction(str(value)) for value in sampled_times]
            upbd_matches = [min(range(len(raw_pts)), key=lambda index: abs(value - raw_pts[index]))
                            for value in upbd_values]
            upbd_residuals = [abs(value - raw_pts[index])
                              for value, index in zip(upbd_values, upbd_matches, strict=True)]
            upbd_outside_video_count += int(any(value < 0 or value > stream_end
                                                for value in upbd_values))
            upbd_duplicate_pts_match_video_count += int(len(set(upbd_matches)) < len(upbd_matches))
            upbd_pts_mismatch_video_count += int(
                any(value > frame_period / 2 for value in upbd_residuals)
                or any(b <= a for a, b in zip(upbd_matches, upbd_matches[1:])))
            frame_clock_residuals.extend(residuals)
            qa_data = []
            local_interval_count = 0
            local_group_counts = []
            local_unlocalized = 0
            local_outside = 0
            local_mapped_qids = 0
            local_interval_checked_qids = 0
            local_processor_grid_mismatch_qids = 0
            for tag, question in zip(("A", "B"),
                                     (planned["question_a"], planned["question_b"]), strict=True):
                key = (str(question["video_id"]), str(question["qid"]))
                if key not in qa:
                    raise ValueError("dataset_qa_pair_missing")
                intervals = _intervals_for(annotations, *key)
                local_interval_count += len(intervals)
                local_interval_checked_qids += 1
                if not intervals_within_media(intervals, duration):
                    local_outside += 1
                    qa_data.append(None)
                    continue
                try:
                    group_tuple = map_intervals_with_processor_grid_check(
                        sampled_pts=pts,
                        temporal_patch_size=TEMPORAL_PATCH_SIZE,
                        temporal_groups=TEMPORAL_GROUPS,
                        support_duration=duration, intervals=intervals)
                except ValueError as exc:
                    if str(exc) != "actual_vs_processor_grid_group_mismatch":
                        raise
                    local_processor_grid_mismatch_qids += 1
                    qa_data.append(None)
                    continue
                local_group_counts.append(len(group_tuple))
                local_mapped_qids += 1
                if len(group_tuple) == TEMPORAL_GROUPS:
                    local_unlocalized += 1
                    qa_data.append(None)
                    continue
                selected = validate_selected_groups(
                    observed["selected_group_sets"][f"{tag}1"],
                    temporal_groups=TEMPORAL_GROUPS)
                repeated = validate_selected_groups(
                    observed["selected_group_sets"][f"{tag}2"],
                    temporal_groups=TEMPORAL_GROUPS)
                if selected != repeated:
                    raise ValueError("same_question_selector_repeat_mismatch")
                qa_data.append((intervals, group_tuple, selected))
            interval_count += local_interval_count
            checked_qid_count += local_interval_checked_qids
            mapped_qid_count += local_mapped_qids
            group_count_values.extend(local_group_counts)
            unlocalized_qid_count += local_unlocalized
            out_of_media_qid_count += local_outside
            if any(value is None for value in qa_data):
                if local_outside:
                    pair_excluded_out_of_media_count += 1
                elif local_unlocalized:
                    pair_excluded_unlocalized_count += 1
                elif local_processor_grid_mismatch_qids:
                    pair_excluded_processor_grid_mismatch_count += 1
                media_count += 1
                continue
            a, b = qa_data
            a_score = _score_one_selection(selected=a[2], evidence=a[1], sampled_pts=pts,
                                           intervals=a[0], duration=duration)
            b_score = _score_one_selection(selected=b[2], evidence=b[1], sampled_pts=pts,
                                           intervals=b[0], duration=duration)
            a_cross = _score_one_selection(selected=a[2], evidence=b[1], sampled_pts=pts,
                                           intervals=b[0], duration=duration)
            b_cross = _score_one_selection(selected=b[2], evidence=a[1], sampled_pts=pts,
                                           intervals=a[0], duration=duration)
            matched["A_matched_iou"].append(float(a_score["group_iou"]))
            matched["B_matched_iou"].append(float(b_score["group_iou"]))
            matched["A_cross_iou"].append(float(a_cross["group_iou"]))
            matched["B_cross_iou"].append(float(b_cross["group_iou"]))
            matched["A_random_two_group_iou"].append(
                exact_two_group_mean_iou(temporal_groups=TEMPORAL_GROUPS,
                                         evidence_groups=a[1]))
            matched["B_random_two_group_iou"].append(
                exact_two_group_mean_iou(temporal_groups=TEMPORAL_GROUPS,
                                         evidence_groups=b[1]))
            matched["A_matched_interval_coverage"].append(
                float(a_score["interval_time_coverage"]))
            matched["B_matched_interval_coverage"].append(
                float(b_score["interval_time_coverage"]))
            local_pair_deltas = [
                float(a_score["group_iou"])-float(a_cross["group_iou"]),
                float(b_score["group_iou"])-float(b_cross["group_iou"]),
            ]
            paired_delta.extend(local_pair_deltas)
            video_pair_delta_means.append(math.fsum(local_pair_deltas) / 2)
            media_count += 1
            evaluated_media_count += 1
        except Exception as exc:
            exception_type = type(exc).__name__
            if exception_type.isascii() and exception_type.isalnum():
                failure_exception_types[exception_type] = (
                    failure_exception_types.get(exception_type, 0) + 1)
            code = str(exc)
            if (not code.isascii() or not code.replace("_", "").isalnum()
                    or not code or not code[0].isalpha()):
                code = "row_validation_failed"
            failures[code] = failures.get(code, 0) + 1

    complete = media_count == 80 and qid_count == 160 and evaluated_media_count > 0 and not failures
    if ({path.name: path.read_bytes() for path in DATA_PATHS} != input_dataset_bytes
            or {path.name: _sha(path.read_bytes()) for path in SOURCE_PATHS} != input_source_hashes):
        raise ValueError("source_or_dataset_changed_during_analysis")
    summaries = {}
    if complete:
        for name, values in matched.items():
            summaries[name] = {
                "n_question_rows": len(values),
                "mean": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2]
                          if len(values) % 2 else
                          (sorted(values)[len(values)//2-1] + sorted(values)[len(values)//2]) / 2,
            }
        summaries["matched_minus_cross_iou"] = {
            "n_question_rows": len(paired_delta),
            "mean": sum(paired_delta) / len(paired_delta),
            "median": sorted(paired_delta)[len(paired_delta) // 2]
                      if len(paired_delta) % 2 else
                      (sorted(paired_delta)[len(paired_delta)//2-1]
                       + sorted(paired_delta)[len(paired_delta)//2]) / 2,
        }
    leave_one_out = (leave_one_video_out_summary(video_pair_delta_means)
                     if complete and len(video_pair_delta_means) == evaluated_media_count
                     else None)
    return {
        "schema_version": 1,
        "purpose": "exploratory_qta_question_selector_alignment_to_official_qid_spans",
        "status": "completed_aggregate_only" if complete else "blocked_all_or_none",
        "classification": "auxiliary_selector_localization_not_faithfulness",
        "dataset_split": "test-associated NExT-GQA",
        "requested_videos": 80,
        "processed_videos": media_count,
        "failed_video_rows": sum(failures.values()),
        "source_clock_excluded_video_count": source_clock_excluded_video_count,
        "requested_questions": 160,
        "joined_questions": qid_count,
        "interval_checked_questions": checked_qid_count,
        "mapped_questions": mapped_qid_count,
        "evaluated_video_pairs": evaluated_media_count,
        "unlocalized_question_count": unlocalized_qid_count,
        "out_of_media_question_count": out_of_media_qid_count,
        "excluded_video_pairs": {
            "source_clock_invalid": pair_excluded_source_clock_count,
            "annotation_outside_media": pair_excluded_out_of_media_count,
            "all_temporal_groups_no_localization": pair_excluded_unlocalized_count,
            "processor_grid_group_mismatch": pair_excluded_processor_grid_mismatch_count},
        "validated_annotation_interval_count": interval_count if complete else 0,
        "video_duration_source_counts": interval_duration_methods,
        "group_count_distribution": ({str(k): group_count_values.count(k)
                                       for k in sorted(set(group_count_values))}
                                      if complete else {}),
        "video_duration_minus_last_sampled_pts_seconds": (
            {"min": min(duration_gaps), "median": statistics.median(duration_gaps),
             "max": max(duration_gaps)} if complete else None),
        "official_timestamp_to_raw_pts_nearest_residual_seconds": (
            {"median": statistics.median(frame_clock_residuals),
             "p90": sorted(frame_clock_residuals)[int(0.9 * (len(frame_clock_residuals) - 1))],
             "max": max(frame_clock_residuals),
             "timestamps": len(frame_clock_residuals),
             "frame2time_equals_upbd_videos": frame2time_upbd_equal}
            if complete and frame_clock_residuals else None),
        "upbd_auxiliary_clock_diagnostic": ({
            "purpose": "descriptive_only_not_used_as_source_clock_gate",
            "videos_with_timestamps_outside_video_stream": upbd_outside_video_count,
            "videos_with_nonunique_nearest_pts_matches": upbd_duplicate_pts_match_video_count,
            "videos_with_pts_residual_or_order_mismatch": upbd_pts_mismatch_video_count,
            "frame2time_upbd_identical_videos": frame2time_upbd_equal,
        } if complete else None),
        "summaries": summaries,
        "leave_one_video_out_influence": leave_one_out,
        "failures_by_code": failures,
        "failure_exception_types": failure_exception_types,
        "provenance_sha256": {
            "cohort_plan": _sha(plan_raw), "selector_result": _sha(selector_raw),
            "test_csv": _sha(test_csv_raw), "gsub_test": _sha(gsub_raw),
            "frame2time_test": _sha(frame2time_raw), "upbd_test": _sha(upbd_raw),
            "map_vid_vidorID": _sha(video_map_raw),
            "analysis_source": input_source_hashes[Path(__file__).name],
            "alignment_helper": input_source_hashes["qta_question_swap_gsub_alignment.py"],
            "video_decoder": input_source_hashes["claim_mask_video_decoder.py"],
            "presentation_time_utils": input_source_hashes["presentation_time_utils.py"],
            "sample_index_utils": input_source_hashes["necessity_choice_margin_utils.py"],
            "official_interval_mapper": input_source_hashes["getspan.py"],
            "official_dataset_readme": input_source_hashes["README.md"],
        },
        "repository_commits": {"res": repo_commit, "NExT-GQA": dataset_commit},
        "sampled_frame_count": FRAME_COUNT,
        "temporal_groups": TEMPORAL_GROUPS,
        "temporal_patch_size": TEMPORAL_PATCH_SIZE,
        "max_grid_deviation_seconds": [TOLERANCE.numerator, TOLERANCE.denominator],
        "timestamp_origin": "raw absolute video PTS seconds; frame2time frame-id timestamps are the primary span clock and must map one-to-one in order to decoded PTS within half a native-frame period. Official getspan.py uses frame2time; upbd is recorded only as descriptive sampled-timestamp metadata because its mapping to these exact media bytes is not specified. Annotation support is right-censored at last decoded PTS because exact presentation-end semantics are uncertain; intervals are never clipped",
        "inference": {"test_performed": False, "population_interval": False},
        "faithfulness_result": False,
        "prediction_dependence_result": False,
        "human_claim_review_completed": False,
        "local_only": True,
        "rights_status": "unverified_local_only",
        "started_at_utc": analysis_started_at,
        "limitations": [
            "The selected videos were previously used for selector-mechanism analysis; this alignment analysis is exploratory and not an independent confirmation.",
            "NExT-GQA labels question-relevant spans, not evidence explicitly indicated by generated explanations.",
            "No model answer scores or interventions are analyzed; attention overlap is not prediction dependence or causal evidence use.",
            "frame2time empirically corroborates the dataset source clock but does not replace independent human browser/player-to-decoder visual correspondence review. upbd metadata is not used to exclude a video because the published files do not define its timestamp-to-source-frame mapping.",
            "Pairs with source timestamps outside decoded PTS support or intervals beyond the last decoded PTS are excluded before overlap scoring; no interval is clipped and no replacements are added. Stream/container end metadata is retained only as a bounds check because its relation to the annotation endpoint is uncertain.",
            "The 16 temporal groups are those from the prior frozen 32-frame selector run; interval coverage is sensitive to coarse group resolution.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--selector-result", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(plan_path=args.cohort_plan, selector_path=args.selector_result)
        _write_private_exclusive(args.output, report)
    except Exception as exc:
        message = str(exc)
        code = message if message.isascii() and message.replace("_", "").isalnum() else "unexpected_error"
        print(f"analysis_failed:{type(exc).__name__}:{code}", file=sys.stderr)
        return 2
    print(f"status={report['status']} videos={report['processed_videos']}/80 "
          f"questions={report['joined_questions']}/160 output={args.output.name}")
    return 0 if report["status"] == "completed_aggregate_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
