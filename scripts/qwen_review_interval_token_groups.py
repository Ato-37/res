"""Map a reviewed time interval to complete Qwen temporal-grid key groups.

The helper establishes processor-layout geometry only. It cannot authenticate
reviewers, source clocks, decoded PTS, semantic support, or faithfulness.
Control selection is a deterministic design candidate that must be frozen in
the study protocol before scoring.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
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


def _finite_fraction(value: object, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}_invalid")
    try:
        if not math.isfinite(float(value)):
            raise ValueError(f"{label}_invalid")
        return value if isinstance(value, Fraction) else Fraction(str(value))
    except (OverflowError, TypeError, ValueError, ZeroDivisionError):
        raise ValueError(f"{label}_invalid") from None


def map_review_interval_to_qwen_groups(
    *, input_ids: Sequence[int], video_token_id: int,
    video_grid_thw: Sequence[int], spatial_merge_size: int,
    temporal_patch_size: int, sampled_pts_seconds: Sequence[Real],
    video_duration_seconds: Real, reviewed_interval_seconds: Sequence[Real],
    _select_control: bool = True,
) -> dict:
    """Return complete temporal-grid groups overlapping the reviewed time cells.

    Sampled-frame support cells use midpoint boundaries clipped to
    ``video_duration_seconds``; contiguous frames in one temporal patch form
    one cell span. The matched control is a contiguous, disjoint window with
    the same number of temporal groups, minimizing the difference in temporal
    duration, then center-time distance, then choosing the earlier start. If
    no such window exists, the helper fails closed. Time comparisons use exact
    rational values derived from their decimal representations. Positions are
    absolute token positions in the caller-validated unpadded, single-video
    sequence.
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
        pts = tuple(_finite_fraction(value, "sampled_pts") for value in sampled_pts_seconds)
        interval = tuple(_finite_fraction(value, "reviewed_interval")
                         for value in reviewed_interval_seconds)
        duration = _finite_fraction(video_duration_seconds, "video_duration")
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

    boundaries = [Fraction(0)]
    boundaries.extend((left + right) / 2 for left, right in zip(pts, pts[1:]))
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
            "sampled_pts_start_seconds": float(pts[frame_start]),
            "sampled_pts_end_seconds": float(pts[frame_stop - 1]),
            "coverage_start_seconds": float(group_start),
            "coverage_end_seconds": float(group_end),
            "token_positions": list(positions[token_start:token_stop]),
        })

    evidence = [group["temporal_grid_index"] for group in groups
                if boundaries[group["frame_start"]] < interval[1]
                and boundaries[group["frame_stop_exclusive"]] > interval[0]]
    if not evidence:
        raise ValueError("reviewed_interval_has_no_sampled_grid")
    evidence_count = len(evidence)
    control = []
    if _select_control:
        candidates = [list(range(start, start + evidence_count))
                      for start in range(temporal_steps - evidence_count + 1)
                      if not set(range(start, start + evidence_count)).intersection(evidence)]
        if not candidates:
            raise ValueError("equal_count_temporal_control_unavailable")
        evidence_start = boundaries[evidence[0] * temporal_patch_size]
        evidence_end = boundaries[(evidence[-1] + 1) * temporal_patch_size]
        evidence_center = (evidence_start + evidence_end) / 2
        evidence_duration = evidence_end - evidence_start
        control = min(candidates, key=lambda row: (
            abs((boundaries[(row[-1] + 1) * temporal_patch_size]
                 - boundaries[row[0] * temporal_patch_size]) - evidence_duration),
            abs((boundaries[row[0] * temporal_patch_size]
                 + boundaries[(row[-1] + 1) * temporal_patch_size]) / 2 - evidence_center),
            row[0]))
    evidence_positions = tuple(position for index in evidence
                               for position in groups[index]["token_positions"])
    control_positions = tuple(position for index in control
                               for position in groups[index]["token_positions"])
    if (_select_control and (len(evidence_positions) != len(control_positions)
            or set(evidence_positions).intersection(control_positions))):
        raise ValueError("internal_group_budget_or_overlap_invalid")

    return {
        "status": "processor_layout_geometry_only",
        "evidence_group_indices": evidence,
        "control_group_indices": control,
        "evidence_token_positions": list(evidence_positions),
        "control_token_positions": list(control_positions),
        "temporal_group_count": evidence_count,
        "token_budget_per_arm": len(evidence_positions),
        "max_sampled_pts_gap_seconds": float(max(sampled_gaps, default=Fraction(0))),
        "first_edge_cell_duration_seconds": float(boundaries[1] - boundaries[0]),
        "last_edge_cell_duration_seconds": float(boundaries[-1] - boundaries[-2]),
        "max_temporal_group_coverage_seconds": float(max(
            boundaries[(index + 1) * temporal_patch_size]
            - boundaries[index * temporal_patch_size]
            for index in range(temporal_steps))),
        "temporal_diagnostics_status": "diagnostic_only_no_prespecified_acceptance_threshold",
        "temporal_support_rule": "frame_midpoint_cells_clipped_to_video_duration_positive_overlap",
        "control_rule": ("disjoint_equal_count_window_min_duration_difference_then_center_distance_then_earlier"
                         if _select_control else "not_selected_evidence_mapping_only"),
        "groups": groups,
        "interpretation_limit": (
            "supplied PTS/interval/duration/processor geometry only; no browser-to-decoder "
            "clock authentication, human-review authentication, semantic support, "
            "source provenance, or faithfulness; timing diagnostics are descriptive only, "
            "with no prespecified acceptance threshold or coverage guarantee"
        ),
    }


def select_equal_count_rank_distance_control_groups(
    evidence_group_indices: Sequence[int], temporal_group_count: int,
) -> list[int]:
    """Select equal-count non-evidence ranks minimizing summed absolute distance.

    Evidence and control ranks are paired in sorted order. Dynamic programming
    finds the minimum-cost subset; equal-cost subsets resolve lexicographically
    to the earlier index tuple. This supports unions of disjoint reviewed
    occurrences without treating gaps between them as evidence.
    """
    if (not isinstance(evidence_group_indices, Sequence)
            or isinstance(evidence_group_indices, (str, bytes))
            or not evidence_group_indices
            or type(temporal_group_count) is not int or temporal_group_count <= 0
            or any(type(index) is not int for index in evidence_group_indices)):
        raise ValueError("evidence_group_indices_invalid")
    evidence = tuple(sorted(set(evidence_group_indices)))
    if (len(evidence) != len(evidence_group_indices)
            or evidence[0] < 0 or evidence[-1] >= temporal_group_count):
        raise ValueError("evidence_group_indices_invalid")
    candidates = [index for index in range(temporal_group_count) if index not in evidence]
    required = len(evidence)
    if len(candidates) < required:
        raise ValueError("equal_count_temporal_control_unavailable")

    # dp[count] = (cost, selected candidate tuple) after scanning candidates.
    dp: list[tuple[int, tuple[int, ...]] | None] = [None] * (required + 1)
    dp[0] = (0, ())
    for candidate in candidates:
        for count in range(required, 0, -1):
            previous = dp[count - 1]
            if previous is None:
                continue
            proposal = (previous[0] + abs(evidence[count - 1] - candidate),
                        previous[1] + (candidate,))
            current = dp[count]
            if current is None or proposal < current:
                dp[count] = proposal
    best = dp[required]
    if best is None:
        raise ValueError("equal_count_temporal_control_unavailable")
    return list(best[1])


def map_review_intervals_to_qwen_groups(
    *, input_ids: Sequence[int], video_token_id: int,
    video_grid_thw: Sequence[int], spatial_merge_size: int,
    temporal_patch_size: int, sampled_pts_seconds: Sequence[Real],
    video_duration_seconds: Real,
    reviewed_intervals_seconds: Sequence[Sequence[Real]],
    interval_boundary_guard_seconds: Real = 0.0,
    require_processor_uniform_grid_assignment: bool = False,
) -> dict:
    """Map occurrences and require stable assignment under a clock guard.

    This lower-level geometry helper does not check Qwen's scalar-FPS grid
    unless `require_processor_uniform_grid_assignment=True`; production Qwen
    callers must use `map_review_interval_from_processor_batch`, which always
    enables that check.

    Each interval is mapped with the same midpoint-cell positive-overlap rule
    as the single-interval mapper. Evidence is the union of all covered groups;
    intervening groups are not filled. The control contains the same number of
    groups outside that union, minimizes order-paired summed absolute index
    distance, and lexicographically prefers earlier ties. If no such control
    exists, the row is ineligible and this function fails closed. The interval
    guard compares possible membership (expanded intervals) with certain
    membership (contracted intervals); any difference rejects the row.
    When `require_processor_uniform_grid_assignment` is true, each interval's
    actual-PTS group set must also equal its group set on the uniform grid
    implied by endpoint-derived scalar FPS. A mismatch rejects the row before
    scoring because the processor receives scalar FPS rather than each source
    frame's actual PTS.
    """
    if (not isinstance(reviewed_intervals_seconds, Sequence)
            or isinstance(reviewed_intervals_seconds, (str, bytes))
            or not reviewed_intervals_seconds):
        raise ValueError("reviewed_intervals_invalid")
    guard = _finite_fraction(interval_boundary_guard_seconds,
                             "interval_boundary_guard")
    if guard < 0:
        raise ValueError("interval_boundary_guard_invalid")
    try:
        frozen_input_ids = tuple(input_ids)
    except (TypeError, ValueError):
        raise ValueError("input_ids_invalid") from None
    try:
        frozen_grid = tuple(video_grid_thw)
    except (TypeError, ValueError):
        raise ValueError("video_grid_thw_invalid") from None
    try:
        frozen_pts = tuple(sampled_pts_seconds)
    except (TypeError, ValueError):
        raise ValueError("sampled_pts_seconds_invalid") from None
    normalized_intervals = []
    for interval in reviewed_intervals_seconds:
        if (not isinstance(interval, Sequence)
                or isinstance(interval, (str, bytes))):
            raise ValueError("reviewed_intervals_invalid")
        try:
            normalized = tuple(_finite_fraction(value, "reviewed_interval")
                               for value in interval)
        except (TypeError, ValueError):
            raise ValueError("reviewed_intervals_invalid") from None
        if len(normalized) != 2 or normalized[0] >= normalized[1]:
            raise ValueError("reviewed_intervals_invalid")
        normalized_intervals.append(normalized)
    common = {
        "input_ids": frozen_input_ids,
        "video_token_id": video_token_id,
        "video_grid_thw": frozen_grid,
        "spatial_merge_size": spatial_merge_size,
        "temporal_patch_size": temporal_patch_size,
        "sampled_pts_seconds": frozen_pts,
        "video_duration_seconds": video_duration_seconds,
    }
    mapped = [map_review_interval_to_qwen_groups(
        **common, reviewed_interval_seconds=interval, _select_control=False)
        for interval in normalized_intervals]
    reference = mapped[0]
    processor_grid_max_deviation = None
    if require_processor_uniform_grid_assignment:
        actual_pts = tuple(_finite_fraction(value, "sampled_pts")
                           for value in frozen_pts)
        if (len(actual_pts) < 2 or actual_pts[0] != 0
                or actual_pts[-1] <= actual_pts[0]):
            raise ValueError("processor_implied_grid_invalid")
        frame_step = (actual_pts[-1] - actual_pts[0]) / (len(actual_pts) - 1)
        processor_pts = tuple(frame_step * index
                              for index in range(len(actual_pts)))
        processor_grid_max_deviation = max(
            abs(actual - implied)
            for actual, implied in zip(actual_pts, processor_pts))
        for interval, actual_row in zip(normalized_intervals, mapped):
            try:
                implied_row = map_review_interval_to_qwen_groups(
                    **{**common, "sampled_pts_seconds": processor_pts},
                    reviewed_interval_seconds=interval, _select_control=False)
            except ValueError as exc:
                if str(exc) == "reviewed_interval_has_no_sampled_grid":
                    raise ValueError(
                        "processor_implied_grid_changes_group_assignment") from None
                raise
            if (actual_row["evidence_group_indices"]
                    != implied_row["evidence_group_indices"]):
                raise ValueError("processor_implied_grid_changes_group_assignment")
    evidence = sorted({index for row in mapped for index in row["evidence_group_indices"]})
    temporal_group_count = len(reference["groups"])
    if guard:
        guard_pts = tuple(_finite_fraction(value, "sampled_pts")
                          for value in frozen_pts)
        guard_duration = _finite_fraction(video_duration_seconds, "video_duration")
        guard_boundaries = [Fraction(0)]
        guard_boundaries.extend((left + right) / 2
                                for left, right in zip(guard_pts, guard_pts[1:]))
        guard_boundaries.append(guard_duration)
        possible = set()
        certain = set()
        for start, end in normalized_intervals:
            possible.update(index for index in range(temporal_group_count)
                            if guard_boundaries[index * temporal_patch_size] < end + guard
                            and guard_boundaries[(index + 1) * temporal_patch_size] > start - guard)
            if end - start > 2 * guard:
                certain.update(index for index in range(temporal_group_count)
                               if guard_boundaries[index * temporal_patch_size] < end - guard
                               and guard_boundaries[(index + 1) * temporal_patch_size] > start + guard)
        if possible != certain:
            raise ValueError("calibration_guard_changes_group_assignment")
    control = select_equal_count_rank_distance_control_groups(
        evidence, temporal_group_count)
    groups = reference["groups"]
    evidence_positions = tuple(position for index in evidence
                               for position in groups[index]["token_positions"])
    control_positions = tuple(position for index in control
                              for position in groups[index]["token_positions"])
    if (len(evidence_positions) != len(control_positions)
            or set(evidence_positions).intersection(control_positions)):
        raise ValueError("internal_group_budget_or_overlap_invalid")
    return {
        **reference,
        "evidence_group_indices": evidence,
        "control_group_indices": control,
        "evidence_token_positions": list(evidence_positions),
        "control_token_positions": list(control_positions),
        "temporal_group_count": len(evidence),
        "evidence_group_count": len(evidence),
        "total_temporal_group_count": len(groups),
        "token_budget_per_arm": len(evidence_positions),
        "reviewed_interval_count": len(normalized_intervals),
        "interval_boundary_guard_seconds": float(guard),
        "processor_uniform_grid_assignment_check": (
            "passed_actual_pts_and_scalar_fps_group_sets_equal"
            if require_processor_uniform_grid_assignment else "not_requested"),
        "processor_uniform_grid_max_pts_deviation_seconds": (
            float(processor_grid_max_deviation)
            if processor_grid_max_deviation is not None else None),
        "reviewed_occurrence_group_indices": [
            row["evidence_group_indices"] for row in mapped],
        "control_rule": "equal_count_order_paired_rank_distance_sum_then_lexicographic_earlier",
        "temporal_group_count_alias": "evidence_group_count; total_temporal_group_count is all groups",
        "mapping_scope": "union_of_all_distinct_reviewed_occurrences",
    }
