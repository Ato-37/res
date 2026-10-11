"""Pure helpers for interpreting the coarse attention-overlap grounding proxy."""

import math


def attention_overlap_proxy(attention, time_ranges, ground_truth):
    """Return normalized chunk weights, peak index, and top-chunk interval hit.

    This is only a coarse localization proxy: it checks whether the single
    highest-attention temporal chunk intersects any annotated interval. It does
    not test whether evidence is causally necessary for the model's answer.
    """
    weights = tuple(float(x) for x in attention)
    ranges = tuple((float(a), float(b)) for a, b in time_ranges)
    gold = tuple((float(a), float(b)) for a, b in ground_truth)

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
    if total <= 0:
        raise ValueError("attention mass must be positive")
    normalized = tuple(x / total for x in weights)
    peak = max(range(len(normalized)), key=normalized.__getitem__)
    lo, hi = ranges[peak]
    hit = any(lo < gold_hi and gold_lo < hi for gold_lo, gold_hi in gold)
    return normalized, peak, hit
