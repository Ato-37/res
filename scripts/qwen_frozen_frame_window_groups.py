"""Map frozen sampled-frame windows to complete Qwen temporal token groups.

CPU-only geometry helper. It validates supplied processor geometry and PTS
coordinates; it does not establish source-clock accuracy or semantic support.
Frame windows are half-open sampled-frame index ranges.
"""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
import math


def _window(value: object, name: str, frame_count: int,
            temporal_patch_size: int) -> tuple[int, int]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or len(value) != 2
            or any(type(v) is not int for v in value)):
        raise ValueError(f"{name}_invalid")
    start, stop = value
    if (start < 0 or stop <= start or stop > frame_count
            or start % temporal_patch_size or stop % temporal_patch_size):
        raise ValueError(f"{name}_not_whole_temporal_groups")
    return start, stop


def map_frozen_frame_windows_to_qwen_groups(
    *, input_ids: Sequence[int], video_token_id: int,
    video_grid_thw: Sequence[int], spatial_merge_size: int,
    temporal_patch_size: int, sampled_pts_seconds: Sequence[Real],
    evidence_frame_window: Sequence[int], control_frame_window: Sequence[int],
) -> dict:
    """Map equal-size, disjoint half-open frame windows to token positions.

    Qwen's temporal grid has one group per ``temporal_patch_size`` sampled
    frames. Each temporal group contains ``(H / merge) * (W / merge)`` visual
    token positions, in the order of video placeholder tokens in ``input_ids``.
    The input token sequence must be the unpadded single-video sequence.
    """
    if (not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes))
            or not input_ids or any(type(token) is not int or token < 0
                                    for token in input_ids)):
        raise ValueError("input_ids_invalid")
    if type(video_token_id) is not int or video_token_id < 0:
        raise ValueError("video_token_id_invalid")
    if (not isinstance(video_grid_thw, Sequence)
            or isinstance(video_grid_thw, (str, bytes))
            or len(video_grid_thw) != 3
            or any(type(v) is not int or v <= 0 for v in video_grid_thw)
            or type(spatial_merge_size) is not int or spatial_merge_size <= 0
            or type(temporal_patch_size) is not int or temporal_patch_size <= 0):
        raise ValueError("video_grid_geometry_invalid")
    temporal_steps, height, width = video_grid_thw
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("video_spatial_grid_not_divisible")
    frame_count = temporal_steps * temporal_patch_size

    if (not isinstance(sampled_pts_seconds, Sequence)
            or isinstance(sampled_pts_seconds, (str, bytes))
            or len(sampled_pts_seconds) != frame_count):
        raise ValueError("sampled_pts_frame_geometry_mismatch")
    pts = []
    for value in sampled_pts_seconds:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("sampled_pts_invalid")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("sampled_pts_invalid")
        pts.append(number)
    if pts[0] < 0 or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("sampled_pts_order_invalid")

    evidence = _window(evidence_frame_window, "evidence_frame_window",
                       frame_count, temporal_patch_size)
    control = _window(control_frame_window, "control_frame_window",
                      frame_count, temporal_patch_size)
    if evidence[1] - evidence[0] != control[1] - control[0]:
        raise ValueError("frame_window_size_mismatch")
    if max(evidence[0], control[0]) < min(evidence[1], control[1]):
        raise ValueError("frame_windows_overlap")

    positions = tuple(i for i, token in enumerate(input_ids)
                      if token == video_token_id)
    tokens_per_group = ((height // spatial_merge_size)
                        * (width // spatial_merge_size))
    expected_tokens = temporal_steps * tokens_per_group
    if (len(positions) != expected_tokens or not positions
            or positions != tuple(range(positions[0], positions[0] + expected_tokens))):
        raise ValueError("video_token_block_geometry_mismatch")
    groups = [tuple(positions[i * tokens_per_group:(i + 1) * tokens_per_group])
              for i in range(temporal_steps)]

    def group_indices(window: tuple[int, int]) -> tuple[int, ...]:
        return tuple(range(window[0] // temporal_patch_size,
                           window[1] // temporal_patch_size))

    evidence_groups = group_indices(evidence)
    control_groups = group_indices(control)
    if set(evidence_groups).intersection(control_groups):
        raise ValueError("temporal_group_windows_overlap")
    evidence_positions = tuple(p for index in evidence_groups for p in groups[index])
    control_positions = tuple(p for index in control_groups for p in groups[index])
    if len(evidence_positions) != len(control_positions):
        raise ValueError("token_budget_mismatch")
    return {
        "status": "processor_layout_geometry_only",
        "frame_count": frame_count,
        "temporal_group_count": temporal_steps,
        "tokens_per_temporal_group": tokens_per_group,
        "evidence_frame_window": list(evidence),
        "control_frame_window": list(control),
        "evidence_group_indices": list(evidence_groups),
        "control_group_indices": list(control_groups),
        "evidence_token_positions": list(evidence_positions),
        "control_token_positions": list(control_positions),
        "token_budget_per_arm": len(evidence_positions),
        "evidence_pts_seconds": [pts[evidence[0]], pts[evidence[1] - 1]],
        "control_pts_seconds": [pts[control[0]], pts[control[1] - 1]],
        "temporal_groups": [list(group) for group in groups],
    }
