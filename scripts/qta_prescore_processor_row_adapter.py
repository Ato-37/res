"""Project a processor mapper result into a non-authorizing QTA row binding.

Caller-provided sampler and processor rows must agree on the fixed 32-frame
endpoint grid and exact PTS values. This check does not authenticate either.
"""
from __future__ import annotations

from fractions import Fraction
import re
from typing import Any

from necessity_choice_margin_utils import uniform_sample_indices

SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAMPLER_FRAMES = 32


def processor_binding_row(*, review_token: str, source_video_sha256: str,
                          mapping: dict[str, Any],
                          sampler_row: dict[str, Any]) -> dict[str, Any]:
    """Return only the mapper fields consumed by the three-row QTA join.

    This projection preserves no tensor or private path and authenticates no
    source, calibration, or human review. External pins are checked later by
    `audit_qta_prescore_mapping_bundle`.
    """
    if not isinstance(review_token, str) or not review_token:
        raise ValueError("processor_row_review_token_invalid")
    if (not isinstance(source_video_sha256, str)
            or not SHA256.fullmatch(source_video_sha256)):
        raise ValueError("processor_row_source_hash_invalid")
    if (not isinstance(mapping, dict)
            or mapping.get("status") != "processor_batch_snapshot_and_caller_metadata_bound"):
        raise ValueError("processor_mapping_status_invalid")
    _validate_sampler_match(source_video_sha256=source_video_sha256,
                            mapping=mapping, sampler_row=sampler_row)
    grid_check = mapping.get("processor_uniform_grid_assignment_check")
    if grid_check != "passed_actual_pts_and_scalar_fps_group_sets_equal":
        raise ValueError("processor_uniform_grid_assignment_check_invalid")
    batch_sha = _sha(mapping.get("typed_batch_sha256"), "typed_batch_sha256")
    sampling_sha = _sha(mapping.get("sampling_metadata_sha256"),
                        "sampling_metadata_sha256")
    grid = mapping.get("video_grid_thw")
    if (not isinstance(grid, list) or len(grid) != 3
            or any(type(value) is not int or value <= 0 for value in grid)):
        raise ValueError("processor_video_grid_invalid")
    intervals = mapping.get("reviewed_intervals_rational_seconds")
    if (not isinstance(intervals, list) or not intervals
            or any(not isinstance(pair, list) or len(pair) != 2
                   or any(not isinstance(value, str) or not value for value in pair)
                   for pair in intervals)):
        raise ValueError("processor_review_intervals_invalid")
    guard = mapping.get("interval_boundary_guard_rational_seconds")
    if not isinstance(guard, str) or not guard:
        raise ValueError("processor_clock_guard_invalid")
    evidence = _indices(mapping.get("evidence_group_indices"), "evidence_groups")
    control = _indices(mapping.get("control_group_indices"), "control_groups")
    raw_groups = mapping.get("groups")
    if (not isinstance(raw_groups, list) or len(raw_groups) != grid[0]
            or any(not isinstance(group, dict) for group in raw_groups)):
        raise ValueError("processor_row_temporal_groups_invalid")
    temporal_positions = [_indices(group.get("token_positions"), "temporal_group_positions")
                          for group in raw_groups]
    flattened = [position for group in temporal_positions for position in group]
    if (len(flattened) != len(set(flattened))
            or any(len(group) != len(temporal_positions[0]) for group in temporal_positions)
            or flattened != list(range(flattened[0], flattened[0] + len(flattened)))):
        raise ValueError("processor_row_temporal_groups_geometry_invalid")
    return {
        "review_token": review_token,
        "source_video_sha256": source_video_sha256,
        "typed_batch_sha256": batch_sha,
        "sampling_metadata_sha256": sampling_sha,
        "video_grid_thw": list(grid),
        "reviewed_intervals_rational_seconds": [list(pair) for pair in intervals],
        "interval_boundary_guard_rational_seconds": guard,
        "processor_uniform_grid_assignment_check": grid_check,
        "evidence_group_indices": evidence,
        "control_group_indices": control,
        "temporal_group_token_positions": temporal_positions,
    }


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"processor_row_{field}_invalid")
    return value


def _validate_sampler_match(*, source_video_sha256: str,
                            mapping: dict[str, Any],
                            sampler_row: dict[str, Any]) -> None:
    """Check caller-provided sampler row against processor frame metadata.

    This is a structural consistency check, not authentication of either source.
    """
    if not isinstance(sampler_row, dict):
        raise ValueError("processor_sampler_row_invalid")
    if sampler_row.get("source_sha256") != source_video_sha256:
        raise ValueError("processor_sampler_source_hash_mismatch")
    decoded_count = sampler_row.get("decoded_frame_count")
    count = sampler_row.get("sampled_frame_count")
    indices = sampler_row.get("sampled_indices")
    pts_pairs = sampler_row.get("sampled_pts")
    if (type(decoded_count) is not int or decoded_count < SAMPLER_FRAMES
            or type(count) is not int or count != SAMPLER_FRAMES
            or not isinstance(indices, list) or len(indices) != SAMPLER_FRAMES
            or any(type(index) is not int for index in indices)
            or not isinstance(pts_pairs, list) or len(pts_pairs) != SAMPLER_FRAMES):
        raise ValueError("processor_sampler_row_shape_invalid")
    if indices != uniform_sample_indices(decoded_count, SAMPLER_FRAMES):
        raise ValueError("processor_sampler_indices_not_frozen_endpoint_grid")
    mapping_indices = mapping.get("sampled_frame_indices")
    mapping_pts = mapping.get("sampled_pts_rational_seconds")
    if (not isinstance(mapping_indices, list)
            or len(mapping_indices) != SAMPLER_FRAMES
            or any(type(index) is not int for index in mapping_indices)
            or mapping_indices != indices or not isinstance(mapping_pts, list)
            or len(mapping_pts) != SAMPLER_FRAMES):
        raise ValueError("processor_sampler_indices_or_pts_count_mismatch")

    expected_pts = []
    for pair in pts_pairs:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(part) is not int for part in pair) or pair[1] <= 0):
            raise ValueError("processor_sampler_pts_invalid")
        expected_pts.append(Fraction(pair[0], pair[1]))
    if (expected_pts[0] != 0
            or any(right <= left for left, right in zip(expected_pts, expected_pts[1:]))):
        raise ValueError("processor_sampler_pts_invalid")
    observed_pts = []
    for value in mapping_pts:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("processor_mapping_pts_invalid")
        try:
            parsed = Fraction(value)
        except (ValueError, ZeroDivisionError, OverflowError):
            raise ValueError("processor_mapping_pts_invalid") from None
        if str(parsed) != value:
            raise ValueError("processor_mapping_pts_noncanonical")
        observed_pts.append(parsed)
    if expected_pts != observed_pts:
        raise ValueError("processor_sampler_pts_mismatch")


def _indices(value: Any, field: str) -> list[int]:
    if (not isinstance(value, list) or not value
            or any(type(index) is not int or index < 0 for index in value)
            or value != sorted(set(value))):
        raise ValueError(f"processor_row_{field}_invalid")
    return list(value)
