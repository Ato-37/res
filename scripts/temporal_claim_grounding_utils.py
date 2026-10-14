"""Synthetic contract for single-interval claim-specific temporal grounding.

This helper scores already-reviewed claim-to-gold-span mappings. It does not
parse model text, inspect videos, or establish that a mapping is semantically
valid. Invalid or ungroundable predictions receive zero while remaining in
the caller's frozen denominator.
"""
from __future__ import annotations

import math
from numbers import Real


_ALIGNMENT_STATUSES = {"matched", "mismatch", "unverifiable"}


def _interval(value, label, duration):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two endpoints")
    start, end = value
    if any(isinstance(x, bool) or not isinstance(x, Real) for x in (start, end)):
        raise ValueError(f"{label} endpoints must be real numbers")
    try:
        start, end = float(start), float(end)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{label} endpoints must be finite") from None
    if not (math.isfinite(start) and math.isfinite(end)
            and 0 <= start < end <= duration):
        raise ValueError(f"{label} must satisfy 0 <= start < end <= duration")
    return start, end


def score_claim_interval(prediction, gold_intervals, duration, alignment_status):
    """Return max single-span tIoU and a generic status/failure code.

    gold_intervals must already be restricted to spans independently judged
    to support this claim. Multiple spans are treated as alternative acceptable
    locations, matching the official NExT-GQA single-prediction convention.
    """
    if isinstance(duration, bool) or not isinstance(duration, Real):
        raise ValueError("duration must be a finite positive number")
    try:
        duration = float(duration)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("duration must be a finite positive number") from None
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a finite positive number")
    if not isinstance(alignment_status, str) or alignment_status not in _ALIGNMENT_STATUSES:
        raise ValueError("claim alignment review must be resolved before scoring")

    if alignment_status == "mismatch":
        return {"temporal_iou": 0.0, "status": "claim_mismatch",
                "failure_code": "claim_alignment_mismatch"}
    if alignment_status == "unverifiable":
        return {"temporal_iou": 0.0, "status": "claim_unverifiable",
                "failure_code": "claim_alignment_unverifiable"}

    try:
        spans = tuple(_interval(span, "gold interval", duration)
                      for span in gold_intervals)
    except TypeError:
        raise ValueError("matched claims require at least one valid gold interval") from None
    if not spans:
        raise ValueError("matched claims require at least one valid gold interval")

    if prediction is None:
        return {"temporal_iou": 0.0, "status": "invalid_prediction",
                "failure_code": "timestamp_parse_failure"}
    try:
        start, end = _interval(prediction, "prediction interval", duration)
    except ValueError:
        return {"temporal_iou": 0.0, "status": "invalid_prediction",
                "failure_code": "prediction_interval_invalid"}

    ious = []
    for gold_start, gold_end in spans:
        intersection = max(0.0, min(end, gold_end) - max(start, gold_start))
        pred_length, gold_length = end - start, gold_end - gold_start
        scale = max(pred_length, gold_length)
        union_scaled = pred_length / scale + gold_length / scale - intersection / scale
        ious.append((intersection / scale) / union_scaled)
    return {"temporal_iou": max(ious), "status": "scored", "failure_code": None}
