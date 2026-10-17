"""Map a reviewed time interval to complete Qwen temporal-grid key groups.

The helper establishes processor-layout geometry only. It cannot authenticate
reviewers, source clocks, decoded PTS, semantic support, or faithfulness.
Control selection is a deterministic design candidate that must be frozen in
the study protocol before scoring.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real


def _finite_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}_invalid")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{label}_invalid") from None
    if not math.isfinite(result):
        raise ValueError(f"{label}_invalid")
    return result


def map_review_interval_to_qwen_groups(
    *, input_ids: Sequence[int], video_token_id: int,
    video_grid_thw: Sequence[int], spatial_merge_size: int,
    temporal_patch_size: int, sampled_pts_seconds: Sequence[Real],
    video_duration_seconds: Real, reviewed_interval_seconds: Sequence[Real],
) -> dict:
    """Return complete temporal-grid groups overlapping the reviewed time cells.

    Sampled-frame support cells use midpoint boundaries clipped to
    ``video_duration_seconds``; contiguous frames in one temporal patch form
    one cell span. The matched control is a contiguous, disjoint window with
    the same number of temporal groups, minimizing the difference in temporal
    duration, then center-time distance, then choosing the earlier start. If
    no such window exists, the helper fails closed. Positions are absolute
    token positions in the caller-validated unpadded, single-video sequence.
    """
    try:
        ids = tuple(input_ids)
    except (TypeError, ValueError):
        raise ValueError("input_ids_invalid") from None
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("input_ids_invalid")
    if type(video_token_id) is not int or video_token_id < 0:
        raise ValueError("video_token_id_invalid")
    try:
        grid = tuple(video_grid_thw)
    except (TypeError, ValueError):
        raise ValueError("video_grid_thw_invalid") from None
    if (len(grid) != 3 or any(type(value) is not int or value <= 0 for value in grid)
            or type(spatial_merge_size) is not int or spatial_merge_size <= 0
            or type(temporal_patch_size) is not int or temporal_patch_size <= 0):
        raise ValueError("video_grid_geometry_invalid")
    temporal_steps, height, width = grid
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("video_spatial_grid_not_divisible")

    try:
        pts = tuple(_finite_real(value, "sampled_pts") for value in sampled_pts_seconds)
        interval = tuple(_finite_real(value, "reviewed_interval")
                         for value in reviewed_interval_seconds)
        duration = _finite_real(video_duration_seconds, "video_duration")
    except TypeError:
        raise ValueError("time_mapping_invalid") from None
    if (not pts or len(pts) != temporal_steps * temporal_patch_size
            or pts[0] < 0
            or any(right <= left for left, right in zip(pts, pts[1:]))):
        raise ValueError("sampled_pts_grid_mismatch")
    if duration <= 0 or pts[-1] > duration:
        raise ValueError("video_duration_or_sampled_pts_invalid")
    if (len(interval) != 2 or interval[0] < 0 or interval[0] >= interval[1]
            or interval[1] > duration):
        raise ValueError("reviewed_interval_invalid")

    positions = tuple(index for index, token in enumerate(ids)
                      if token == video_token_id)
    tokens_per_group = (height // spatial_merge_size) * (width // spatial_merge_size)
    if (len(positions) != temporal_steps * tokens_per_group or not positions
            or positions != tuple(range(positions[0], positions[0] + len(positions)))):
        raise ValueError("video_token_block_geometry_mismatch")

    boundaries = [0.0]
    boundaries.extend((left + right) / 2.0 for left, right in zip(pts, pts[1:]))
    boundaries.append(duration)
    sampled_gaps = [right - left for left, right in zip(pts, pts[1:])]
    groups = []
    for grid_index in range(temporal_steps):
        frame_start = grid_index * temporal_patch_size
        frame_stop = frame_start + temporal_patch_size
        token_start = grid_index * tokens_per_group
        token_stop = token_start + tokens_per_group
        group_start, group_end = boundaries[frame_start], boundaries[frame_stop]
        groups.append({
            "temporal_grid_index": grid_index,
            "frame_start": frame_start,
            "frame_stop_exclusive": frame_stop,
            "sampled_pts_start_seconds": pts[frame_start],
            "sampled_pts_end_seconds": pts[frame_stop - 1],
            "coverage_start_seconds": group_start,
            "coverage_end_seconds": group_end,
            "token_positions": list(positions[token_start:token_stop]),
        })

    evidence = [group["temporal_grid_index"] for group in groups
                if group["coverage_start_seconds"] < interval[1]
                and group["coverage_end_seconds"] > interval[0]]
    if not evidence:
        raise ValueError("reviewed_interval_has_no_sampled_grid")
    evidence_count = len(evidence)
    candidates = [list(range(start, start + evidence_count))
                  for start in range(temporal_steps - evidence_count + 1)
                  if not set(range(start, start + evidence_count)).intersection(evidence)]
    if not candidates:
        raise ValueError("equal_count_temporal_control_unavailable")
    evidence_start = groups[evidence[0]]["coverage_start_seconds"]
    evidence_end = groups[evidence[-1]]["coverage_end_seconds"]
    evidence_center = (evidence_start + evidence_end) / 2
    evidence_duration = evidence_end - evidence_start
    control = min(candidates, key=lambda row: (
        abs((groups[row[-1]]["coverage_end_seconds"]
             - groups[row[0]]["coverage_start_seconds"]) - evidence_duration),
        abs((groups[row[0]]["coverage_start_seconds"]
             + groups[row[-1]]["coverage_end_seconds"]) / 2 - evidence_center),
        row[0]))
    evidence_positions = tuple(position for index in evidence
                               for position in groups[index]["token_positions"])
    control_positions = tuple(position for index in control
                               for position in groups[index]["token_positions"])
    if (len(evidence_positions) != len(control_positions)
            or set(evidence_positions).intersection(control_positions)):
        raise ValueError("internal_group_budget_or_overlap_invalid")

    return {
        "status": "processor_layout_geometry_only",
        "evidence_group_indices": evidence,
        "control_group_indices": control,
        "evidence_token_positions": list(evidence_positions),
        "control_token_positions": list(control_positions),
        "temporal_group_count": evidence_count,
        "token_budget_per_arm": len(evidence_positions),
        "max_sampled_pts_gap_seconds": max(sampled_gaps, default=0.0),
        "first_edge_cell_duration_seconds": boundaries[1] - boundaries[0],
        "last_edge_cell_duration_seconds": boundaries[-1] - boundaries[-2],
        "max_temporal_group_coverage_seconds": max(
            group["coverage_end_seconds"] - group["coverage_start_seconds"]
            for group in groups),
        "temporal_diagnostics_status": "diagnostic_only_no_prespecified_acceptance_threshold",
        "temporal_support_rule": "frame_midpoint_cells_clipped_to_video_duration_positive_overlap",
        "control_rule": "disjoint_equal_count_window_min_duration_difference_then_center_distance_then_earlier",
        "groups": groups,
        "interpretation_limit": (
            "supplied PTS/interval/duration/processor geometry only; no browser-to-decoder "
            "clock authentication, human-review authentication, semantic support, "
            "source provenance, or faithfulness; timing diagnostics are descriptive only, "
            "with no prespecified acceptance threshold or coverage guarantee"
        ),
    }
