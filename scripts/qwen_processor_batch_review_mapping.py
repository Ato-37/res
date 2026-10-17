"""Bind temporal review mapping inputs to one cloned CPU Qwen batch snapshot.

This wrapper provides integrity bookkeeping, not source/reviewer/clock
authentication. The caller must provide the calibrated PTS-space guard
explicitly; the function fails if it is omitted. The caller remains
responsible for authenticating all external metadata before scoring.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from numbers import Real
from typing import Any

from nextgqa_atomic_batch_freeze import batch_sha256
from qwen_review_interval_token_groups import map_review_intervals_to_qwen_groups


_REQUIRED_BATCH_KEYS = {
    "input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw",
    "second_per_grid_ts",
}


def map_review_interval_from_processor_batch(
    *, batch: Mapping[str, Any], torch_module: Any, video_token_id: int,
    spatial_merge_size: int, temporal_patch_size: int,
    sampled_frame_indices: Sequence[int], sampled_pts_seconds: Sequence[Real | str],
    source_frame_count: int, video_duration_seconds: Real | str,
    sample_fps: Real, processor_revision: str,
    processor_fingerprint_sha256: str,
    reviewed_interval_seconds: Sequence[Real | str] | None = None,
    reviewed_intervals_seconds: Sequence[Sequence[Real | str]] | None = None,
    interval_boundary_guard_seconds: Real | str | None = None,
) -> dict:
    """Map a review window using grid/token data read from the same CPU batch.

    Enforces one unpadded video sequence, typed CPU tensors, pixel-row/grid
    agreement, sampled-frame count/order/range, and processor temporal spacing.
    The returned digests bind supplied caller metadata and the full typed batch
    without authenticating where those values came from.
    """
    if not isinstance(batch, Mapping) or not _REQUIRED_BATCH_KEYS.issubset(batch):
        raise ValueError("processor_batch_schema_invalid")
    if (torch_module is None or not callable(getattr(torch_module, "is_tensor", None))
            or not isinstance(processor_revision, str) or not processor_revision
            or type(video_token_id) is not int or video_token_id < 0
            or type(spatial_merge_size) is not int or spatial_merge_size <= 0
            or type(temporal_patch_size) is not int or temporal_patch_size <= 0
            or type(source_frame_count) is not int or source_frame_count <= 0
            or not isinstance(processor_fingerprint_sha256, str)
            or len(processor_fingerprint_sha256) != 64
            or any(char not in "0123456789abcdef" for char in processor_fingerprint_sha256)):
        raise ValueError("processor_batch_binding_metadata_invalid")
    fps = _positive_finite(sample_fps, "sample_fps")
    duration_fraction = _finite_fraction(video_duration_seconds, "video_duration")
    duration = float(duration_fraction)
    if duration <= 0:
        raise ValueError("video_duration_invalid")

    snapshot = _snapshot_tree(batch, torch_module)
    batch_digest = batch_sha256(snapshot)
    ids = snapshot["input_ids"]
    mask = snapshot["attention_mask"]
    grid = snapshot["video_grid_thw"]
    pixels = snapshot["pixel_values_videos"]
    time_steps = snapshot["second_per_grid_ts"]
    for name, tensor in (("input_ids", ids), ("attention_mask", mask),
                         ("video_grid_thw", grid),
                         ("pixel_values_videos", pixels),
                         ("second_per_grid_ts", time_steps)):
        if not torch_module.is_tensor(tensor) or str(tensor.device) != "cpu":
            raise ValueError(f"processor_{name}_cpu_tensor_required")
    if (ids.ndim != 2 or tuple(ids.shape[:1]) != (1,)
            or ids.dtype not in (torch_module.int32, torch_module.int64)):
        raise ValueError("processor_input_ids_shape_or_dtype_invalid")
    if (mask.shape != ids.shape or mask.dtype not in
            (torch_module.bool, torch_module.int32, torch_module.int64)
            or not bool(((mask == 0) | (mask == 1)).all())
            or not bool((mask == 1).all())):
        raise ValueError("processor_attention_mask_must_be_single_unpadded")
    if (grid.ndim != 2 or tuple(grid.shape) != (1, 3)
            or grid.dtype not in (torch_module.int32, torch_module.int64)):
        raise ValueError("processor_video_grid_shape_or_dtype_invalid")
    if (pixels.ndim != 2 or pixels.shape[0] <= 0 or pixels.shape[1] <= 0
            or not pixels.dtype.is_floating_point):
        raise ValueError("processor_video_pixels_shape_or_dtype_invalid")
    if (time_steps.ndim != 1 or tuple(time_steps.shape) != (1,)
            or not time_steps.dtype.is_floating_point
            or not bool(torch_module.isfinite(time_steps).all())
            or not bool((time_steps > 0).all())):
        raise ValueError("processor_temporal_spacing_invalid")

    grid_values = [int(value) for value in grid[0].tolist()]
    temporal_steps, height, width = grid_values
    expected_frames = temporal_steps * temporal_patch_size
    if (temporal_steps <= 0 or height <= 0 or width <= 0
            or height % spatial_merge_size or width % spatial_merge_size
            or int(pixels.shape[0]) != temporal_steps * height * width):
        raise ValueError("processor_pixel_grid_geometry_mismatch")
    expected_grid_spacing = temporal_patch_size / fps
    actual_grid_spacing = float(time_steps[0])
    if (not math.isfinite(expected_grid_spacing)
            or abs(actual_grid_spacing - expected_grid_spacing) > 1e-5):
        raise ValueError("processor_temporal_spacing_fps_mismatch")

    try:
        frame_indices = tuple(sampled_frame_indices)
        pts = tuple(_finite_fraction(value, "sampled_pts") for value in sampled_pts_seconds)
    except (TypeError, ValueError):
        raise ValueError("sampled_frame_metadata_invalid") from None
    if (reviewed_interval_seconds is None) == (reviewed_intervals_seconds is None):
        raise ValueError("reviewed_intervals_argument_invalid")
    if interval_boundary_guard_seconds is None:
        raise ValueError("clock_guard_required")
    guard = _finite_fraction(interval_boundary_guard_seconds, "interval_boundary_guard")
    if guard < 0:
        raise ValueError("interval_boundary_guard_invalid")
    try:
        if reviewed_interval_seconds is not None:
            intervals = (tuple(_finite_fraction(value, "reviewed_interval")
                               for value in reviewed_interval_seconds),)
        else:
            if (not isinstance(reviewed_intervals_seconds, Sequence)
                    or isinstance(reviewed_intervals_seconds, (str, bytes))
                    or not reviewed_intervals_seconds):
                raise ValueError("reviewed_intervals_invalid")
            intervals = tuple(
                tuple(_finite_fraction(value, "reviewed_interval") for value in interval)
                for interval in reviewed_intervals_seconds
            )
    except (TypeError, ValueError):
        raise ValueError("reviewed_intervals_invalid") from None
    if (len(frame_indices) != expected_frames
            or any(type(index) is not int or index < 0 or index >= source_frame_count
                   for index in frame_indices)
            or any(right <= left for left, right in zip(frame_indices, frame_indices[1:]))
            or len(pts) != expected_frames
            or any(len(interval) != 2 or interval[0] >= interval[1]
                   for interval in intervals)):
        raise ValueError("sampled_frame_metadata_grid_mismatch")

    geometry = map_review_intervals_to_qwen_groups(
        input_ids=ids[0].tolist(), video_token_id=video_token_id,
        video_grid_thw=grid_values, spatial_merge_size=spatial_merge_size,
        temporal_patch_size=temporal_patch_size, sampled_pts_seconds=pts,
        video_duration_seconds=duration_fraction,
        reviewed_intervals_seconds=intervals,
        interval_boundary_guard_seconds=guard,
    )
    nominal_frame_step = 1.0 / fps
    max_pts_cadence_deviation = max(
        (abs((right - left) - nominal_frame_step)
         for left, right in zip(pts, pts[1:])), default=0.0)
    metadata = {
        "processor_revision": processor_revision,
        "processor_fingerprint_sha256": processor_fingerprint_sha256,
        "video_token_id": video_token_id,
        "sampled_frame_indices": list(frame_indices),
        "sampled_pts_seconds": [float(value) for value in pts],
        "sampled_pts_rational_seconds": [str(value) for value in pts],
        "source_frame_count": source_frame_count,
        "video_duration_seconds": duration,
        "sample_fps": fps,
        "video_grid_thw": grid_values,
        "spatial_merge_size": spatial_merge_size,
        "temporal_patch_size": temporal_patch_size,
        "reviewed_intervals_seconds": [[float(value) for value in interval]
                                        for interval in intervals],
        "reviewed_intervals_rational_seconds": [
            [str(value) for value in interval] for interval in intervals],
        "video_duration_rational_seconds": str(duration_fraction),
        "interval_boundary_guard_rational_seconds": str(guard),
    }
    metadata_digest = hashlib.sha256(json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    return {
        **geometry,
        "status": "processor_batch_snapshot_and_caller_metadata_bound",
        "processor_revision": processor_revision,
        "processor_fingerprint_sha256": processor_fingerprint_sha256,
        "typed_batch_sha256": batch_digest,
        "sampling_metadata_sha256": metadata_digest,
        "sampled_pts_rational_seconds": [str(value) for value in pts],
        "reviewed_intervals_seconds": [[float(value) for value in interval]
                                        for interval in intervals],
        "reviewed_intervals_rational_seconds": [
            [str(value) for value in interval] for interval in intervals],
        "interval_boundary_guard_rational_seconds": str(guard),
        "interval_boundary_guard_seconds": float(guard),
        "reviewed_interval_count": len(intervals),
        **({"reviewed_interval_seconds": [float(value) for value in intervals[0]]}
           if len(intervals) == 1 else {}),
        "max_pts_cadence_deviation_from_nominal_seconds": max_pts_cadence_deviation,
        "source_authentication_status": "not_performed_by_this_helper",
        "_processor_batch_snapshot": snapshot,
        "binding_limit": (
            "digests bind supplied batch and metadata for integrity; they do not authenticate "
            "source video, reviewer, browser/decoder clock, frame sampling, or semantic support"
        ),
    }


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}_invalid")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label}_invalid") from None
    if not math.isfinite(number):
        raise ValueError(f"{label}_invalid")
    return number


def _finite_fraction(value: object, label: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{label}_invalid")
    try:
        if isinstance(value, str):
            if not value or value != value.strip() or len(value) > 128:
                raise ValueError(f"{label}_invalid")
            rational = Fraction(value)
            if str(rational) != value:
                raise ValueError(f"{label}_invalid")
            return rational
        if not isinstance(value, Real):
            raise ValueError(f"{label}_invalid")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label}_invalid")
        return value if isinstance(value, Fraction) else Fraction(str(value))
    except (OverflowError, TypeError, ValueError, ZeroDivisionError):
        raise ValueError(f"{label}_invalid") from None


def _snapshot_tree(value: Any, torch_module: Any) -> Any:
    """Clone the entire typed processor payload before mapping or hashing it."""
    if torch_module.is_tensor(value):
        if str(value.device) != "cpu":
            raise ValueError("processor_batch_cpu_tensor_required")
        try:
            return value.detach().clone()
        except Exception:
            raise ValueError("processor_batch_snapshot_failed") from None
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("processor_batch_mapping_key_invalid")
        return {key: _snapshot_tree(item, torch_module)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_tree(item, torch_module) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_tree(item, torch_module) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("processor_batch_value_type_unsupported")


def _positive_finite(value: object, label: str) -> float:
    number = _finite(value, label)
    if number <= 0:
        raise ValueError(f"{label}_invalid")
    return number
