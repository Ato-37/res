"""Pure-CPU contract for video-level claim-selective dependence contrasts.

This helper only checks score arithmetic and accounting on already-derived
records. It does not select claims, inspect video, or establish faithfulness.
"""
from __future__ import annotations

import math
import statistics

from scipy.stats import t as student_t


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("records must not be empty")
    required = {"video_id", "target_sha256", "full_score", "selected_score", "control_score"}
    seen = set()
    values = []
    for row in records:
        if not required.issubset(row):
            raise ValueError("record missing required field")
        vid = row["video_id"]
        if not isinstance(vid, str) or not vid or vid in seen:
            raise ValueError("duplicate or empty video_id")
        seen.add(vid)
        digest = row["target_sha256"]
        if not isinstance(digest, str):
            raise ValueError("target digest must be a string")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError("invalid target digest")
        scores = [row[key] for key in ("full_score", "selected_score", "control_score")]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores):
            raise ValueError("nonfinite score")
        # Algebraically cancel the shared full score before floating-point
        # evaluation to avoid catastrophic cancellation at large score offsets.
        d = scores[2] - scores[1]
        if not math.isfinite(d):
            raise ValueError("derived contrast is nonfinite")
        values.append(d)
    # The frozen gold answer/claim target is video-specific. Each triplet is
    # checked by its own hash; requiring one shared hash would incorrectly
    # reject a multi-video cohort with different answers.
    n = len(values)
    try:
        mean = statistics.mean(values)
    except OverflowError as exc:
        raise ValueError("nonfinite mean contrast") from exc
    if not math.isfinite(mean):
        raise ValueError("nonfinite mean contrast")
    if n >= 2:
        try:
            sd = statistics.stdev(values)
        except OverflowError as exc:
            raise ValueError("nonfinite contrast dispersion") from exc
        if not math.isfinite(sd):
            raise ValueError("nonfinite contrast dispersion")
        se = sd / math.sqrt(n)
        half_width = float(student_t.ppf(0.975, df=n - 1)) * se
        ci95 = [mean - half_width, mean + half_width]
        if not (math.isfinite(se) and all(math.isfinite(x) for x in ci95)):
            raise ValueError("nonfinite contrast interval")
    else:
        sd = se = None
        ci95 = None
    return {
        "schema_version": 2,
        "status": "synthetic_contract_only",
        "n_videos": n,
        "n_unique_target_hashes": len({row["target_sha256"] for row in records}),
        "video_level_d": values,
        "mean_d": mean,
        "sample_sd_d": sd,
        "standard_error_d": se,
        "ci95_t_d": ci95,
        "interval_method": "two-sided Student-t interval over independent video-level contrasts; df=n-1",
        "interpretation_limit": "synthetic arithmetic/interval contract only; no claim extraction, semantic alignment, human review, GPU inference, or faithfulness efficacy",
    }
