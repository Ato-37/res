"""Pure-CPU contract for video-level claim-selective dependence contrasts.

This helper only checks score arithmetic and accounting on already-derived
records. It does not select claims, inspect video, or establish faithfulness.
"""
from __future__ import annotations

import math
from collections import Counter


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("records must not be empty")
    required = {"video_id", "target_sha256", "full_score", "selected_score", "control_score"}
    seen = set()
    values = []
    for row in records:
        if not required.issubset(row):
            raise ValueError("record missing required field")
        vid = str(row["video_id"])
        if not vid or vid in seen:
            raise ValueError("duplicate or empty video_id")
        seen.add(vid)
        digest = str(row["target_sha256"])
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError("invalid target digest")
        scores = [row[key] for key in ("full_score", "selected_score", "control_score")]
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores):
            raise ValueError("nonfinite score")
        # d = selected intervention drop - matched-control drop.
        d = (scores[0] - scores[1]) - (scores[0] - scores[2])
        values.append(d)
    mean = sum(values) / len(values)
    return {
        "schema_version": 1,
        "status": "synthetic_contract_only",
        "n_videos": len(values),
        "target_sha256_counts": dict(Counter(row["target_sha256"] for row in records)),
        "video_level_d": values,
        "mean_d": mean,
        "interpretation_limit": "arithmetic contract only; no claim extraction, semantic alignment, human review, GPU inference, or faithfulness efficacy",
    }
