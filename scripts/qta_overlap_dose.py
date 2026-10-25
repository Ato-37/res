"""Overlap dose-response sets for WP1D (experiments/2026-10-25_wp1_overlap_dose_plan.md): blocked-group sets of fixed size k whose overlap with the annotation is graded."""
from __future__ import annotations

import hashlib
from typing import Sequence

FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _ranked(items: Sequence[int], token: str, seed: str) -> list[int]:
    return sorted(items, key=lambda i: (hashlib.sha256(f"{seed}\0{token}\0{i}".encode()).digest(), i))


def overlap_sets(annotation: Sequence[int], sham: Sequence[int], group_count: int, token: str) -> list[dict]:
    """One blocked set per distinct achieved overlap m = round(f * k), f in FRACTIONS (k = |annotation|).

    The m evidence groups are the hash-first m of the annotation, the k-m others the hash-first of the groups outside the annotation
    (m=0 is exactly the project's disjoint sham; m=k is exactly the annotation). Duplicated m (small k) are run once.
    Returns [{"overlap": m, "fraction": m/k, "groups": sorted tuple}] ordered by m."""
    ann = tuple(sorted(set(annotation)))
    k = len(ann)
    if k < 1 or 2 * k > group_count or not token:
        raise ValueError("overlap_dose_geometry_invalid")
    outside = [g for g in range(group_count) if g not in set(ann)]
    out, seen = [], set()
    for f in FRACTIONS:
        m = int(f * k + 0.5)
        if m in seen:
            continue
        seen.add(m)
        if m == 0:
            groups = tuple(sorted(sham))
            if len(groups) != k or set(groups) & set(ann):
                raise ValueError("sham_not_disjoint_equal_count")
        else:
            inside = _ranked(ann, token, "wp1d-in-v1")[:m]
            rest = _ranked(outside, token, "wp1d-out-v1")[: k - m]
            groups = tuple(sorted(inside + rest))
        out.append({"overlap": m, "fraction": m / k, "groups": groups})
    return out
