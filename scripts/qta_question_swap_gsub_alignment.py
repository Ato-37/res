"""Outcome-free alignment helpers for QTA selectors and NExT-GQA labels.

This is an auxiliary selector-localization diagnostic. It does not score
predictions or establish explanation faithfulness.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Sequence
from pathlib import PurePosixPath


def _fraction(value, label: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{label}_invalid")
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        raise ValueError(f"{label}_invalid") from None
    return result


def validate_video_question_pair(*, video_id: str, relative_video_path: str,
                                 mapped_video_path: str, question_a: dict,
                                 question_b: dict, question_map: dict) -> tuple[dict, dict]:
    """Require the frozen video/path and both exact QA rows to agree."""
    relative = PurePosixPath(relative_video_path)
    mapped = PurePosixPath(mapped_video_path)
    if (not video_id or relative.is_absolute() or mapped.is_absolute()
            or ".." in relative.parts or ".." in mapped.parts
            or relative.suffix.lower() != ".mp4"
            or str(relative.with_suffix("")) != str(mapped)
            or any(not isinstance(q, dict) for q in (question_a, question_b))):
        raise ValueError("frozen_video_path_binding_invalid")
    qids = []
    rows = []
    for question in (question_a, question_b):
        qid = str(question.get("qid", ""))
        key = (video_id, qid)
        row = question_map.get(key)
        if (question.get("video_id") != video_id or not qid or row is None
                or str(question.get("question", "")).strip() != row.get("question")):
            raise ValueError("frozen_question_row_binding_invalid")
        qids.append(qid)
        rows.append(row)
    if qids[0] == qids[1]:
        raise ValueError("frozen_question_pair_not_distinct")
    return rows[0], rows[1]


def validate_source_clock(*, source_times: Sequence, decoded_pts: Sequence,
                          media_start, media_end, tolerance) -> tuple[float, ...]:
    """Bind the ordered official source timestamps to the decoded video clock."""
    start = _fraction(media_start, "media_start")
    end = _fraction(media_end, "media_end")
    tol = _fraction(tolerance, "clock_tolerance")
    pts = tuple(_fraction(value, "decoded_pts") for value in decoded_pts)
    times = tuple(_fraction(value, "source_timestamp") for value in source_times)
    if (not times or not pts or end <= start or tol < 0
            or any(b <= a for a, b in zip(times, times[1:]))
            or any(b <= a for a, b in zip(pts, pts[1:]))
            or any(value < start or value > end for value in times)):
        raise ValueError("official_source_clock_outside_video")
    matched = tuple(min(range(len(pts)), key=lambda index: abs(value - pts[index]))
                    for value in times)
    residuals = tuple(abs(value - pts[index]) for value, index in zip(times, matched))
    if (any(value > tol for value in residuals)
            or any(b <= a for a, b in zip(matched, matched[1:]))):
        raise ValueError("official_source_clock_pts_mismatch")
    return tuple(float(value) for value in residuals)


def intervals_within_media(intervals: Sequence[Sequence], media_end) -> bool:
    end = _fraction(media_end, "media_end")
    return bool(intervals) and all(
        isinstance(pair, (tuple, list)) and len(pair) == 2
        and _fraction(pair[0], "interval_start") >= 0
        and _fraction(pair[1], "interval_end") <= end
        and _fraction(pair[1], "interval_end") > _fraction(pair[0], "interval_start")
        for pair in intervals)


def video_support_end(*, container_start, container_duration, stream_start,
                      stream_duration, last_pts, frame_period) -> tuple[Fraction, str]:
    """Choose the measured video end, never the longer container/audio end."""
    start = _fraction(container_start, "container_start")
    container_end = _fraction(container_duration, "container_duration")
    last = _fraction(last_pts, "last_pts")
    period = _fraction(frame_period, "frame_period")
    if start != 0 or container_end <= 0 or period <= 0:
        raise ValueError("container_video_clock_invalid")
    if stream_duration is None:
        end = last + period
        method = "last_decoded_pts_plus_median_frame_period_fallback"
    else:
        stream_start_value = _fraction(0 if stream_start is None else stream_start,
                                       "stream_start")
        stream_length = _fraction(stream_duration, "stream_duration")
        if stream_length <= 0:
            raise ValueError("video_stream_duration_invalid")
        end = stream_start_value + stream_length
        method = "video_stream_start_plus_duration"
    if end < last or end > container_end + period:
        raise ValueError("video_stream_container_bounds_disagree")
    return end, method


def conservative_video_support_end(*, last_pts):
    """Right-censor annotation support at the final decoded presentation time."""
    end = _fraction(last_pts, "last_pts")
    if end <= 0:
        raise ValueError("video_support_end_invalid")
    return end, "right_censored_at_last_decoded_pts"


def map_intervals_to_temporal_groups(*, sampled_pts: Sequence,
                                     temporal_patch_size: int,
                                     temporal_groups: int,
                                     support_duration,
                                     intervals: Sequence[Sequence]) -> tuple[int, ...]:
    """Map intervals by positive overlap with midpoint support cells.

    `sampled_pts`, annotations, and `support_duration` must use the same source
    video clock. The caller is responsible for establishing that mapping. QTA's
    32-frame input uses 16 temporal groups of two frames each.
    """
    try:
        pts = tuple(_fraction(value, "sampled_pts") for value in sampled_pts)
        duration = _fraction(support_duration, "support_duration")
    except TypeError:
        raise ValueError("time_input_invalid") from None
    if (type(temporal_patch_size) is not int or temporal_patch_size < 1
            or type(temporal_groups) is not int or temporal_groups < 1
            or len(pts) != temporal_patch_size * temporal_groups
            or duration <= 0 or pts[0] < 0 or pts[-1] > duration
            or any(b <= a for a, b in zip(pts, pts[1:]))):
        raise ValueError("temporal_geometry_invalid")
    if not isinstance(intervals, (tuple, list)) or not intervals:
        raise ValueError("annotation_intervals_invalid")
    normalized = []
    for interval in intervals:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            raise ValueError("annotation_interval_invalid")
        start, end = (_fraction(v, "annotation_interval") for v in interval)
        if start < 0 or end <= start or end > duration:
            raise ValueError("annotation_interval_invalid")
        normalized.append((start, end))
    boundaries = [Fraction(0)]
    boundaries.extend((a + b) / 2 for a, b in zip(pts, pts[1:]))
    boundaries.append(duration)
    covered = []
    for group in range(temporal_groups):
        left = boundaries[group * temporal_patch_size]
        right = boundaries[(group + 1) * temporal_patch_size]
        if any(left < end and right > start for start, end in normalized):
            covered.append(group)
    if not covered:
        raise ValueError("annotation_maps_to_no_temporal_group")
    return tuple(covered)


def validate_selected_groups(values, *, temporal_groups: int,
                             selected_count: int = 2) -> tuple[int, ...]:
    if (not isinstance(values, (tuple, list)) or len(values) != selected_count
            or any(type(value) is not int or not 0 <= value < temporal_groups
                   for value in values)
            or tuple(sorted(set(values))) != tuple(values)):
        raise ValueError("selected_temporal_groups_invalid")
    return tuple(values)


def map_intervals_with_processor_grid_check(*, sampled_pts: Sequence,
                                            temporal_patch_size: int,
                                            temporal_groups: int,
                                            support_duration,
                                            intervals: Sequence[Sequence]) -> tuple[int, ...]:
    actual = map_intervals_to_temporal_groups(
        sampled_pts=sampled_pts, temporal_patch_size=temporal_patch_size,
        temporal_groups=temporal_groups, support_duration=support_duration,
        intervals=intervals)
    pts = tuple(_fraction(value, "sampled_pts") for value in sampled_pts)
    implied_step = (pts[-1] - pts[0]) / (len(pts) - 1)
    implied = tuple(implied_step * index for index in range(len(pts)))
    processor = map_intervals_to_temporal_groups(
        sampled_pts=implied, temporal_patch_size=temporal_patch_size,
        temporal_groups=temporal_groups, support_duration=support_duration,
        intervals=intervals)
    if actual != processor:
        raise ValueError("actual_vs_processor_grid_group_mismatch")
    return actual


def set_iou(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        raise ValueError("group_set_empty")
    return len(a & b) / len(a | b)


def exact_two_group_mean_iou(*, temporal_groups: int,
                             evidence_groups: Sequence[int]) -> float:
    """Finite exact mean IoU over all equal-budget two-group choices."""
    evidence = set(evidence_groups)
    if (type(temporal_groups) is not int or temporal_groups < 2 or not evidence
            or any(type(i) is not int or not 0 <= i < temporal_groups
                   for i in evidence)):
        raise ValueError("chance_reference_input_invalid")
    choices = tuple(combinations(range(temporal_groups), 2))
    return sum(set_iou(choice, tuple(evidence)) for choice in choices) / len(choices)


def interval_union_duration(intervals: Sequence[Sequence]) -> Fraction:
    if not isinstance(intervals, (tuple, list)) or not intervals:
        raise ValueError("annotation_intervals_invalid")
    normalized = sorted((_fraction(pair[0], "interval_start"),
                         _fraction(pair[1], "interval_end")) for pair in intervals
                        if isinstance(pair, (tuple, list)) and len(pair) == 2)
    if len(normalized) != len(intervals):
        raise ValueError("annotation_interval_invalid")
    if any(start < 0 or end <= start for start, end in normalized):
        raise ValueError("annotation_interval_invalid")
    total = Fraction(0)
    left, right = normalized[0]
    for start, end in normalized[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + right - left
