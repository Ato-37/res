"""Metrics for repeat-controlled question-swap selector diagnostics."""
from __future__ import annotations

import math
from statistics import median, stdev
from typing import Any


def group_jaccard(left: list[int] | tuple[int, ...],
                  right: list[int] | tuple[int, ...], *, group_count: int) -> float:
    if (type(group_count) is not int or group_count < 1
            or not isinstance(left, (list, tuple))
            or not isinstance(right, (list, tuple))
            or not left or not right
            or any(type(index) is not int or not 0 <= index < group_count
                   for index in list(left) + list(right))
            or len(set(left)) != len(left) or len(set(right)) != len(right)):
        raise ValueError("question_swap_selected_group_set_invalid")
    a, b = set(left), set(right)
    return len(a & b) / len(a | b)


def video_contrast(selections: dict[str, list[int] | tuple[int, ...]], *,
                   group_count: int) -> dict[str, float]:
    expected = {"A1", "A2", "B1", "B2"}
    if not isinstance(selections, dict) or set(selections) != expected:
        raise ValueError("question_swap_selection_conditions_invalid")
    j_repeat_a = group_jaccard(selections["A1"], selections["A2"],
                               group_count=group_count)
    j_repeat_b = group_jaccard(selections["B1"], selections["B2"],
                               group_count=group_count)
    cross = [group_jaccard(selections[a], selections[b], group_count=group_count)
             for a in ("A1", "A2") for b in ("B1", "B2")]
    j_repeat = (j_repeat_a + j_repeat_b) / 2
    j_cross = sum(cross) / len(cross)
    return {"j_repeat": j_repeat, "j_cross": j_cross,
            "d_repeat_minus_cross": j_repeat - j_cross}


def summarize_video_contrasts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("question_swap_contrast_rows_empty")
    contrasts = []
    for row in rows:
        if (not isinstance(row, dict)
                or set(row) != {"j_repeat", "j_cross", "d_repeat_minus_cross"}):
            raise ValueError("question_swap_contrast_row_schema_invalid")
        values = [row[key] for key in ("j_repeat", "j_cross", "d_repeat_minus_cross")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in values):
            raise ValueError("question_swap_contrast_nonfinite")
        if (not 0 <= row["j_repeat"] <= 1 or not 0 <= row["j_cross"] <= 1
                or not -1 <= row["d_repeat_minus_cross"] <= 1
                or abs(row["d_repeat_minus_cross"] - (row["j_repeat"] - row["j_cross"])) > 1e-12):
            raise ValueError("question_swap_contrast_range_or_arithmetic_invalid")
        contrasts.append(float(row["d_repeat_minus_cross"]))
    return {
        "n_video_pairs": len(contrasts),
        "mean_d_repeat_minus_cross": sum(contrasts) / len(contrasts),
        "sample_sd_d_repeat_minus_cross": stdev(contrasts) if len(contrasts) > 1 else None,
        "median_d_repeat_minus_cross": median(contrasts),
        "min_d_repeat_minus_cross": min(contrasts),
        "max_d_repeat_minus_cross": max(contrasts),
        "positive_count": sum(value > 1e-12 for value in contrasts),
        "negative_count": sum(value < -1e-12 for value in contrasts),
        "zero_within_tolerance_count": sum(abs(value) <= 1e-12 for value in contrasts),
        "inferential_test_performed": False,
        "population_interval_reported": False,
    }




def question_repeat_schedule(batch_a: Any, batch_b: Any,
                             question_a: str, question_b: str):
    """Return the exact repeated-batch four-forward order used by the runner."""
    if (batch_a is None or batch_b is None or batch_a is batch_b
            or not isinstance(question_a, str) or not question_a.strip()
            or not isinstance(question_b, str) or not question_b.strip()
            or question_a == question_b):
        raise ValueError("question_swap_repeat_schedule_invalid")
    return (("A1", batch_a, "A", question_a),
            ("A2", batch_a, "A", question_a),
            ("B1", batch_b, "B", question_b),
            ("B2", batch_b, "B", question_b))


def forward_accounting(requested: int, attempted: int,
                       completed: int) -> dict[str, int]:
    if (any(type(value) is not int for value in (requested, attempted, completed))
            or requested < 0 or attempted < 0 or completed < 0
            or attempted > requested or completed > attempted):
        raise ValueError("question_swap_forward_accounting_invalid")
    return {"requested_forwards": requested,
            "attempted_forwards": attempted,
            "completed_forwards": completed,
            "failed_forwards": attempted - completed,
            "not_attempted_forwards": requested - attempted}
