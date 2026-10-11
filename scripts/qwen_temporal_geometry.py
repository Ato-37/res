"""Describe sampled-frame spans separately from Qwen temporal RoPE coordinates.

These helpers describe preprocessing geometry only. Endpoint spans and optional
nearest-sample coverage cells are not claims about which frames support an answer.
"""

from decimal import Decimal
import math
from numbers import Integral, Real

from video_time_mapping import effective_sample_fps


def _positive_integer(value, label, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    value = int(value)
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _finite_nonnegative(value, label, *, allow_zero=False):
    if isinstance(value, bool) or not (
        isinstance(value, Real) or isinstance(value, Decimal)
    ):
        raise ValueError(f"{label} must be a real number")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{label} must be finite") from None
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _timestamps(values):
    try:
        values = tuple(values)
    except (TypeError, ValueError):
        raise ValueError("timestamps must be an iterable of real numbers") from None
    if not values:
        raise ValueError("at least one sampled timestamp is required")
    normalized = tuple(
        _finite_nonnegative(value, f"timestamps[{index}]", allow_zero=True)
        for index, value in enumerate(values)
    )
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError("sample timestamps must be strictly increasing")
    return normalized


def _endpoint_spans(timestamps, temporal_patch_size):
    return tuple(
        (timestamps[start], timestamps[start + temporal_patch_size - 1])
        for start in range(0, len(timestamps), temporal_patch_size)
    )


def _duration_coverage_cells(timestamps, temporal_patch_size, duration_seconds):
    """Partition [0, duration] by nearest sampled timestamp, then group patches."""
    duration = _finite_nonnegative(duration_seconds, "duration_seconds")
    if timestamps[-1] > duration:
        raise ValueError("sample timestamps must not exceed duration_seconds")
    boundaries = [0.0]
    boundaries.extend(
        (left + right) / 2.0
        for left, right in zip(timestamps, timestamps[1:])
    )
    boundaries.append(duration)
    return tuple(
        (boundaries[start], boundaries[start + temporal_patch_size])
        for start in range(0, len(timestamps), temporal_patch_size)
    )


def qwen_temporal_geometry(
    sampled_timestamps,
    temporal_patch_size,
    *,
    tokens_per_second,
    second_per_grid=None,
    video_start_position=0,
    duration_seconds=None,
):
    """Return temporal endpoint spans and version-explicit Qwen RoPE anchors.

    ``sampled_frame_endpoint_spans_seconds`` records the first and last observed
    frame timestamps in each temporal patch. ``duration_coverage_cells_seconds``
    is an optional nearest-sample partition of the source duration; it is a
    geometric convention, not an annotation or evidence-validity judgment.

    The returned RoPE anchor fields describe two source-level formulas. In the
    installed Transformers 5.17 implementation, ``int(second_per_grid)`` is
    applied *before* multiplication by tokens-per-second. Transformers 4.49.0
    multiplied the fractional interval first and converted resulting IDs to
    integer. Keeping both fields explicit prevents either behavior from being
    silently treated as source timestamps or as validated temporal evidence.
    ``video_start_position`` is the model's sequence-position offset before the
    video block; it defaults to zero for relative geometry.

    Fractional ticks use Python float arithmetic. The historical version name
    identifies the formula, not bit-exact Torch float32 behavior near rounding
    boundaries. Inference callers must independently check the intended tensor
    arithmetic; the GPU regression runner refuses incompatible candidates.
    """
    times = _timestamps(sampled_timestamps)
    patch = _positive_integer(temporal_patch_size, "temporal_patch_size")
    tokens_per_second = _positive_integer(tokens_per_second, "tokens_per_second")
    start_position = _positive_integer(
        video_start_position, "video_start_position", allow_zero=True
    )
    if len(times) % patch:
        raise ValueError("sample count must be divisible by temporal_patch_size")

    sampled_fps = effective_sample_fps(times)
    derived_interval = patch / sampled_fps
    if second_per_grid is None:
        interval = derived_interval
    else:
        interval = _finite_nonnegative(second_per_grid, "second_per_grid")
        if not math.isclose(interval, derived_interval, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                "second_per_grid must match temporal_patch_size / sampled_fps"
            )

    temporal_steps = len(times) // patch
    installed_step = tokens_per_second * int(interval)
    installed_ticks = tuple(
        start_position + step * installed_step for step in range(temporal_steps)
    )
    fractional_ticks = tuple(
        start_position + int(step * interval * tokens_per_second)
        for step in range(temporal_steps)
    )

    coverage = None
    if duration_seconds is not None:
        coverage = _duration_coverage_cells(times, patch, duration_seconds)

    return {
        "sampled_fps": sampled_fps,
        "second_per_grid_seconds": interval,
        "sampled_frame_endpoint_spans_seconds": _endpoint_spans(times, patch),
        "duration_coverage_cells_seconds": coverage,
        "video_start_position": start_position,
        "tokens_per_second": tokens_per_second,
        "transformers_5_17_temporal_rope_ticks_pretruncate": installed_ticks,
        "transformers_4_49_0_temporal_rope_ticks_fractional": fractional_ticks,
    }


def replace_video_temporal_rope_row(
    native_position_ids,
    *,
    video_token_spans,
    image_token_spans,
    attention_mask,
    video_grid_thw,
    spatial_merge_size,
    second_per_grid_seconds,
    tokens_per_second,
):
    """Replace only one video's temporal RoPE IDs using fractional intervals.

    ``native_position_ids`` is the unbatched ``(3, sequence_length)`` result of
    the installed model's own ``get_rope_index``, converted to nested Python
    integer iterables (for a Torch tensor, pass ``tensor.tolist()``). The grid
    and token spans likewise use Python iterables. This keeps the helper free
    of a Torch dependency; tensor objects are not normalized implicitly. The
    returned spatial and text coordinates are copied unchanged. Fractional
    temporal ticks follow the official 4.49.0/main rule
    ``floor(t * seconds_per_grid * tokens_per_second)`` relative to the native
    first video tick. This adapter fails closed unless
    there is one contiguous video-token block, no image block, and an all-one
    attention mask. It does not decide whether the resulting coordinates are
    scientifically valid evidence times.
    """
    try:
        rows = tuple(tuple(row) for row in native_position_ids)
    except (TypeError, ValueError):
        raise ValueError("native_position_ids must have shape (3, sequence_length)") from None
    if len(rows) != 3 or not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("native_position_ids must have shape (3, sequence_length)")
    sequence_length = len(rows[0])
    if sequence_length == 0:
        raise ValueError("native position sequence must be nonempty")
    native = tuple(
        tuple(_positive_integer(value, "native position ID", allow_zero=True) for value in row)
        for row in rows
    )

    try:
        mask = tuple(attention_mask)
    except (TypeError, ValueError):
        raise ValueError("an all-one attention_mask is required") from None
    if len(mask) != sequence_length or any(value != 1 for value in mask):
        raise ValueError("padding or masked sequence positions are unsupported")

    try:
        image_spans = tuple(tuple(span) for span in image_token_spans)
        video_spans = tuple(tuple(span) for span in video_token_spans)
    except (TypeError, ValueError):
        raise ValueError("image/video token spans must be iterable start-stop pairs") from None
    if image_spans:
        raise ValueError("image inputs are unsupported by the single-video adapter")
    if len(video_spans) != 1 or len(video_spans[0]) != 2:
        raise ValueError("exactly one video token span is supported")
    start = _positive_integer(video_spans[0][0], "video token start", allow_zero=True)
    stop = _positive_integer(video_spans[0][1], "video token stop", allow_zero=True)
    if not (0 <= start < stop <= sequence_length):
        raise ValueError("video token span must be nonempty and inside the sequence")

    try:
        grid = tuple(video_grid_thw)
    except (TypeError, ValueError):
        raise ValueError("video_grid_thw must contain T, H, W") from None
    if len(grid) != 3:
        raise ValueError("video_grid_thw must contain T, H, W")
    temporal_steps, height, width = tuple(
        _positive_integer(value, f"video_grid_thw[{index}]")
        for index, value in enumerate(grid)
    )
    merge = _positive_integer(spatial_merge_size, "spatial_merge_size")
    tps = _positive_integer(tokens_per_second, "tokens_per_second")
    interval = _finite_nonnegative(second_per_grid_seconds, "second_per_grid_seconds")
    if height % merge or width % merge:
        raise ValueError("video spatial grid must be divisible by spatial_merge_size")
    tokens_per_step = (height // merge) * (width // merge)
    expected_tokens = temporal_steps * tokens_per_step
    if stop - start != expected_tokens:
        raise ValueError("video token span length does not match video_grid_thw")

    temporal = list(native[0])
    base_tick = native[0][start]
    for step in range(temporal_steps):
        begin = start + step * tokens_per_step
        end = begin + tokens_per_step
        native_chunk = native[0][begin:end]
        if len(set(native_chunk)) != 1:
            raise ValueError("native temporal IDs must be constant within each video grid slice")
        fractional_tick = base_tick + int(step * interval * tps)
        temporal[begin:end] = [fractional_tick] * tokens_per_step

    # Preserve axes 1/2 and every non-video temporal coordinate byte-for-byte
    # (as integer values) from the installed native get_rope_index result.
    return tuple(temporal), native[1], native[2]
