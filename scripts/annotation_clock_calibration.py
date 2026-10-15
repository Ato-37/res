"""Exact-rational affine calibration utility for pre-frozen clock protocols.

This helper implements only ordinary least-squares affine mapping from decoder
PTS seconds to annotation seconds. It is not a default calibration method: use
it only when that exact method was fixed in the protocol before inspecting
anchors. Fit and held-out anchors are supplied separately; held-out values are
never used to estimate coefficients. It does not assess rights, source identity,
anchor validity/coverage, chronology, or mapping uncertainty, and cannot clear
the annotation-clock gate or authorize inference.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any


def _rational(value: Any, field: str) -> Fraction:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError(f"{field}_invalid")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError, TypeError, OverflowError):
        raise ValueError(f"{field}_invalid") from None
    return result


def _anchors(value: Any, field: str) -> list[tuple[str, Fraction, Fraction]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field}_invalid")
    result = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {"anchor_id", "pts_time", "annotation_time"}:
            raise ValueError(f"{field}_invalid")
        anchor_id = row["anchor_id"]
        if (not isinstance(anchor_id, str) or not anchor_id.strip()
                or anchor_id != anchor_id.strip() or anchor_id in seen):
            raise ValueError(f"{field}_identity_invalid")
        seen.add(anchor_id)
        result.append((anchor_id, _rational(row["pts_time"], field),
                       _rational(row["annotation_time"], field)))
    return result


def evaluate_affine_calibration(fit_anchors: Any, heldout_anchors: Any,
                                tolerance: Any) -> dict:
    """Fit affine PTS→annotation time on fit anchors and evaluate held-out errors.

    Anchor times and tolerance must be exact integers or rational strings (for
    example ``"3/2"``). The tolerance is caller-fixed; this function does not
    select it or compute uncertainty around the calibration.
    """
    fit = _anchors(fit_anchors, "fit_anchors")
    heldout = _anchors(heldout_anchors, "heldout_anchors")
    overlap = {row[0] for row in fit} & {row[0] for row in heldout}
    if overlap:
        raise ValueError("fit_heldout_identity_overlap")
    if len(fit) < 2:
        raise ValueError("fit_anchor_count_insufficient")
    limit = _rational(tolerance, "tolerance")
    if limit <= 0:
        raise ValueError("tolerance_not_positive")

    n = len(fit)
    sum_x = sum((x for _, x, _ in fit), Fraction(0))
    sum_y = sum((y for _, _, y in fit), Fraction(0))
    sum_xx = sum((x * x for _, x, _ in fit), Fraction(0))
    sum_xy = sum((x * y for _, x, y in fit), Fraction(0))
    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        raise ValueError("fit_pts_degenerate")
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    if slope <= 0:
        raise ValueError("fit_slope_not_positive")

    fit_errors = [abs(y - (slope * x + intercept)) for _, x, y in fit]
    heldout_errors = [abs(y - (slope * x + intercept)) for _, x, y in heldout]
    maximum = max(heldout_errors)
    return {
        "mapping": "annotation_time = slope * pts_time + intercept",
        "fit_method": "ordinary_least_squares_affine",
        "fit_anchor_count": len(fit),
        "heldout_anchor_count": len(heldout),
        "slope": str(slope),
        "intercept": str(intercept),
        "fit_max_abs_residual": str(max(fit_errors)),
        "heldout_max_abs_residual": str(maximum),
        "tolerance": str(limit),
        "heldout_within_tolerance": maximum <= limit,
        "uncertainty_status": "not_evaluated",
        "execution_authorization": "not_granted",
    }
