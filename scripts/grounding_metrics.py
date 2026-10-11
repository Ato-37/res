"""Pure helpers for interpreting the coarse attention-overlap grounding proxy."""

import math
from decimal import Decimal
from numbers import Real

from video_time_mapping import chunk_timestamp_ranges


def _finite_real(value, label):
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{label} must contain real numeric values")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{label} must contain finite values") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must contain finite values")
    return number


def validate_temporal_intervals(intervals, label="annotated intervals"):
    """Validate numeric finite positive-duration pairs without coercing labels."""
    try:
        result = tuple((_finite_real(a,label),_finite_real(b,label)) for a,b in intervals)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite numeric positive-duration pairs") from None
    if not result or any(a >= b for a,b in result):
        raise ValueError(f"{label} must be nonempty finite positive-duration pairs; at least one is required")
    return result


def attention_overlap_proxy(attention, time_ranges, ground_truth):
    """Return normalized chunk weights, peak index, and top-chunk interval hit.

    This is only a coarse localization proxy: it checks whether the single
    highest-attention temporal chunk intersects any annotated interval. It does
    not test whether evidence is causally necessary for the model's answer.
    """
    weights = tuple(_finite_real(x,"attention weights") for x in attention)
    ranges = validate_temporal_intervals(time_ranges,"temporal ranges")
    gold = validate_temporal_intervals(ground_truth)

    if not weights or len(weights) != len(ranges):
        raise ValueError("attention and temporal ranges must be nonempty and aligned")
    if any(not (math.isfinite(a) and math.isfinite(b) and a < b) for a, b in ranges):
        raise ValueError("temporal ranges must be finite and have positive duration")
    if not gold:
        raise ValueError("at least one annotated interval is required")
    if any(not (math.isfinite(a) and math.isfinite(b) and a < b) for a, b in gold):
        raise ValueError("annotated intervals must be finite and have positive duration")
    if any(not (math.isfinite(x) and x >= 0) for x in weights):
        raise ValueError("attention weights must be finite and nonnegative")

    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("attention mass must be finite and positive")
    normalized = tuple(x / total for x in weights)
    peak = max(range(len(normalized)), key=normalized.__getitem__)
    lo, hi = ranges[peak]
    hit = any(lo < gold_hi and gold_lo < hi for gold_lo, gold_hi in gold)
    return normalized, peak, hit


def sampled_point_window_coverage(timestamps, temporal_patch_size, ground_truth):
    """Count sampled timestamp points inside annotated windows, by temporal chunk.

    Annotation windows use inclusive point membership (start <= t <= end).
    This is a geometric diagnostic only; point membership does not establish
    that a sampled frame contains semantically valid evidence.
    """
    try:
        raw_timestamps = tuple(timestamps)
    except (TypeError, ValueError):
        raise ValueError("timestamps must be an iterable of real numbers") from None
    timestamps = []
    for timestamp in raw_timestamps:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (Real, Decimal)):
            raise ValueError("timestamps must contain only real numbers")
        try:
            timestamp = float(timestamp)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("timestamps must contain finite real numbers") from None
        if not math.isfinite(timestamp):
            raise ValueError("timestamps must contain finite real numbers")
        timestamps.append(timestamp)
    timestamps = tuple(timestamps)
    ranges = chunk_timestamp_ranges(timestamps, temporal_patch_size)
    gold = validate_temporal_intervals(ground_truth)

    point_counts = []
    for chunk_index, (start, end) in enumerate(ranges):
        first = chunk_index * temporal_patch_size
        chunk_points = timestamps[first:first + temporal_patch_size]
        point_counts.append(sum(
            any(gold_start <= timestamp <= gold_end
                for gold_start, gold_end in gold)
            for timestamp in chunk_points
        ))
    total = sum(point_counts)
    return {
        "per_chunk_points_inside": tuple(point_counts),
        "points_inside": total,
        "maximum_points_inside_any_chunk": max(point_counts),
        "any_point_inside": total > 0,
    }
