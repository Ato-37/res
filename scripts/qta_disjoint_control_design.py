"""Outcome-independent control selection for a future exploratory QTA proxy.

This module is deliberately separate from the completed v1 runner so the
historical run's source hash contract remains unchanged. It does not execute
model inference or authorize scoring.
"""
from __future__ import annotations

from collections.abc import Sequence


def select_qta_disjoint_rank_matched_control_groups(*,
        evidence_group_indices: Sequence[int], qta_group_indices: Sequence[int],
        temporal_group_count: int) -> list[int]:
    """Choose equal-count nearest-rank controls disjoint from evidence and QTA.

    Evidence and QTA groups must themselves be disjoint; otherwise the fixed
    treatment cannot be separated from evidence ablation. Candidate controls
    exclude both sets. Sorted evidence/control ranks are paired, minimizing
    summed absolute rank distance; ties resolve lexicographically. Call this
    before generating or scoring a target, and treat infeasibility as an
    eligibility failure rather than swapping groups after seeing outcomes.
    """
    if type(temporal_group_count) is not int or temporal_group_count <= 0:
        raise ValueError("temporal_group_count_invalid")

    def checked(name: str, value: Sequence[int]) -> tuple[int, ...]:
        if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
                or not value or any(type(index) is not int for index in value)):
            raise ValueError(f"{name}_group_indices_invalid")
        result = tuple(value)
        if (tuple(sorted(set(result))) != result
                or result[0] < 0 or result[-1] >= temporal_group_count):
            raise ValueError(f"{name}_group_indices_invalid")
        return result

    evidence = checked("evidence", evidence_group_indices)
    qta = checked("qta", qta_group_indices)
    if set(evidence) & set(qta):
        raise ValueError("qta_evidence_groups_overlap")
    required = len(evidence)
    forbidden = set(evidence) | set(qta)
    candidates = [index for index in range(temporal_group_count)
                  if index not in forbidden]
    if len(candidates) < required:
        raise ValueError("qta_disjoint_equal_count_control_unavailable")

    # dp[count] = (summed rank distance, lexicographically ordered selection).
    dp: list[tuple[int, tuple[int, ...]] | None] = [None] * (required + 1)
    dp[0] = (0, ())
    for candidate in candidates:
        for count in range(required, 0, -1):
            previous = dp[count - 1]
            if previous is None:
                continue
            proposal = (previous[0] + abs(evidence[count - 1] - candidate),
                        previous[1] + (candidate,))
            current = dp[count]
            if current is None or proposal < current:
                dp[count] = proposal
    best = dp[required]
    if best is None:
        raise ValueError("qta_disjoint_equal_count_control_unavailable")
    return list(best[1])
