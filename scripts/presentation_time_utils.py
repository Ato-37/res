"""Exact presentation-time helpers for decoded video frame sequences.

These dependency-light functions accept decoder PTS values and their rational
time base. They do not decode media or establish that the timestamps share the
annotation player's time origin; callers must bind and verify that provenance.
"""
from fractions import Fraction
import math

from necessity_choice_margin_utils import nearest_control_indices, uniform_sample_indices

FFMPEG_AV_NOPTS_VALUE = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def presentation_times_from_pts(pts_values, time_base):
    """Map display-order integer PTS values to exact seconds relative to frame 0.

    Variable positive frame intervals are allowed. Missing, duplicate, or
    non-monotonic timestamps fail closed. ``time_base`` must be rational (for
    example PyAV's ``Frame.time_base``), never a float approximation.
    """
    if isinstance(time_base, bool) or not isinstance(time_base, (Fraction, tuple)):
        raise ValueError("time_base must be a Fraction or integer numerator/denominator tuple")
    if isinstance(time_base, tuple) and (
            len(time_base) != 2
            or any(type(part) is not int for part in time_base)):
        raise ValueError("time_base tuple must contain two non-boolean integers")
    try:
        base = Fraction(*time_base) if isinstance(time_base, tuple) else time_base
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        raise ValueError("time_base must be a positive rational") from None
    if base <= 0:
        raise ValueError("time_base must be a positive rational")
    if not isinstance(pts_values, (list, tuple)) or not pts_values:
        raise ValueError("PTS sequence must be a nonempty list or tuple")
    if any(type(pts) is not int for pts in pts_values):
        raise ValueError("every decoded frame must have an integer PTS")
    if any(pts < FFMPEG_AV_NOPTS_VALUE or pts > INT64_MAX for pts in pts_values):
        raise ValueError("PTS value is outside the signed 64-bit decoder range")
    if FFMPEG_AV_NOPTS_VALUE in pts_values:
        raise ValueError("decoded frame PTS is missing")
    if any(right <= left for left, right in zip(pts_values, pts_values[1:])):
        raise ValueError("PTS values must be strictly increasing in display order")
    origin = pts_values[0]
    return tuple((pts - origin) * base for pts in pts_values)


def _interval_seconds(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be a finite nonnegative number")
    if value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    # Decimal text avoids importing a float's binary expansion into the
    # timestamp comparison while preserving the caller's declared precision.
    return Fraction(str(value))


def _validate_time_map(presentation_times):
    if not isinstance(presentation_times, (list, tuple)):
        raise ValueError("presentation_times must be a sequence")
    if not presentation_times:
        raise ValueError("presentation-time map must not be empty")
    if any(not isinstance(t, Fraction) or t < 0 for t in presentation_times):
        raise ValueError("presentation-time map must contain nonnegative Fractions")
    if presentation_times[0] != 0:
        raise ValueError("presentation-time map must be relative to the first decoded frame")
    if any(right <= left for left, right in zip(presentation_times, presentation_times[1:])):
        raise ValueError("presentation-time map must be strictly increasing")


def strict_evidence_indices_from_times(indices, presentation_times, intervals):
    """Select indices whose exact presentation times lie in inclusive spans."""
    if not isinstance(indices, (list, tuple)):
        raise ValueError("indices and presentation_times must be sequences")
    _validate_time_map(presentation_times)
    if any(type(index) is not int or index < 0 or index >= len(presentation_times)
           for index in indices):
        raise ValueError("sampled frame index is outside the presentation-time map")
    if len(set(indices)) != len(indices):
        raise ValueError("sampled frame indices must be unique")
    if not isinstance(intervals, list):
        raise ValueError("intervals must be a list")
    parsed = []
    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("each interval must have two endpoints")
        start, end = (_interval_seconds(interval[0], "interval start"),
                      _interval_seconds(interval[1], "interval end"))
        if end < start:
            raise ValueError("interval end must not precede its start")
        parsed.append((start, end))
    return [index for index in indices
            if any(start <= presentation_times[index] <= end for start, end in parsed)]


def assign_evidence_control_frames_from_times(total_frames, num_frames,
                                              presentation_times, intervals):
    """Build the claim/control assignment using a complete decoded PTS map.

    Sampling remains endpoint-inclusive and uniform by decoded frame index;
    only interval membership uses the exact presentation-time map. Controls
    retain the existing equal-count minimum sampled-rank-distance rule.
    """
    sampled = uniform_sample_indices(total_frames, num_frames)
    _validate_time_map(presentation_times)
    if len(presentation_times) != total_frames:
        raise ValueError("presentation-time count must equal decoded frame count")
    indices = sorted(set(sampled))
    base = {
        "requested_frame_count": num_frames,
        "sampled_frame_count": len(indices),
        "sampled_indices": indices,
        "evidence_indices": [],
        "control_indices": [],
        "matched_rank_distance": None,
    }
    if len(indices) != num_frames:
        return {"status": "insufficient_distinct_sample_frames", **base}
    evidence = strict_evidence_indices_from_times(indices, presentation_times, intervals)
    if not evidence:
        return {"status": "no_evidence", **base}
    control = nearest_control_indices(indices, evidence)
    if len(control) != len(evidence):
        return {"status": "insufficient_control", **base,
                "evidence_indices": evidence}
    ranks = {frame_index: position for position, frame_index in enumerate(indices)}
    distance = sum(abs(ranks[left] - ranks[right])
                   for left, right in zip(evidence, control))
    return {"status": "ok", **base, "evidence_indices": evidence,
            "control_indices": control, "matched_rank_distance": distance}
