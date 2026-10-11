"""Pure time-axis helpers shared by video preprocessing and tests."""
from decimal import Decimal
from math import isfinite
from numbers import Integral, Real


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _timestamp_values(timestamps):
    try:
        raw_values = tuple(timestamps)
    except (TypeError, ValueError):
        raise ValueError("timestamps must be an iterable of real numbers") from None
    values = []
    for value in raw_values:
        if isinstance(value, bool) or not (
            isinstance(value, Real) or isinstance(value, Decimal)
        ):
            raise ValueError("timestamps must contain only real numbers")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("timestamps must contain finite real numbers") from None
        if not isfinite(number):
            raise ValueError("timestamps must contain finite real numbers")
        if number < 0:
            raise ValueError("Video sample timestamps must be nonnegative")
        values.append(number)
    return values

def uniform_frame_indices(total_frames, requested_frames):
    """Choose the largest constant integer frame stride within the source."""
    total_frames = _integer(total_frames, "total_frames")
    requested_frames = _integer(requested_frames, "requested_frames")
    if total_frames < 2:
        raise ValueError("At least two source frames are required")
    if requested_frames < 2:
        raise ValueError("At least two frames must be requested")
    count = min(total_frames, requested_frames)
    count -= count % 2
    if count < 2:
        raise ValueError("At least two effective frames are required")
    stride = max(1, (total_frames - 1) // (count - 1))
    unused = total_frames - 1 - stride * (count - 1)
    first = unused // 2
    return [first + index * stride for index in range(count)]

def effective_sample_fps(timestamps):
    """Return FPS only when timestamps are uniformly spaced."""
    values = _timestamp_values(timestamps)
    if len(values) < 2:
        raise ValueError("At least two finite sample timestamps are required")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("Sample timestamps must be strictly increasing")
    spacing = (values[-1] - values[0]) / (len(values) - 1)
    tolerance = max(1e-3, spacing * 0.01)
    if any(abs((right - left) - spacing) > tolerance
           for left, right in zip(values, values[1:])):
        raise ValueError("A single FPS value cannot represent nonuniform timestamps")
    return 1.0 / spacing


def chunk_timestamp_ranges(timestamps, temporal_patch_size):
    """Map each temporal patch to its first/last sampled-frame timestamps."""
    temporal_patch_size = _integer(temporal_patch_size, "temporal_patch_size")
    if temporal_patch_size < 1:
        raise ValueError("temporal_patch_size must be positive")
    values = _timestamp_values(timestamps)
    if not values:
        raise ValueError("At least one sample timestamp is required")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("Sample timestamps must be strictly increasing")
    if len(values) % temporal_patch_size:
        raise ValueError("Sample count must be divisible by temporal_patch_size")
    return [
        (values[start], values[start + temporal_patch_size - 1])
        for start in range(0, len(values), temporal_patch_size)
    ]
