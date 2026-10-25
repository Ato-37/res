"""WP2 helpers (experiments/2026-10-26_wp2_self_cited_evidence_plan.md): parse the model's own cited evidence interval and map it to temporal groups."""
from __future__ import annotations

import math
import re
from typing import Sequence

_ANSWER = re.compile(r"answer\s*[:\-]?\s*\(?([A-E])\b", re.IGNORECASE)
_TIME = re.compile(r"(?:time|evidence|interval|seconds?)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?\s*(?:-|–|—|to)\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_citation(text: str) -> dict:
    """Return {'ok': bool, 'letter': str|None, 'start': float|None, 'end': float|None, 'reason': str}. Nothing but numbers/letter is kept."""
    a = _ANSWER.search(text or "")
    t = _TIME.search(text or "")
    if not t:
        return {"ok": False, "letter": a.group(1).upper() if a else None, "start": None, "end": None, "reason": "no_time_span"}
    start, end = float(t.group(1)), float(t.group(2))
    if not (math.isfinite(start) and math.isfinite(end)) or end < start:
        return {"ok": False, "letter": a.group(1).upper() if a else None, "start": start, "end": end, "reason": "invalid_span"}
    return {"ok": True, "letter": a.group(1).upper() if a else None, "start": start, "end": end, "reason": "ok"}


def cited_groups(sampled_seconds: Sequence[float], start: float, end: float, temporal_patch_size: int = 2) -> tuple[int, ...]:
    """Temporal groups whose frame-time span [t_first, t_last] intersects the cited [start, end] seconds."""
    n = len(sampled_seconds)
    if n % temporal_patch_size or n == 0:
        raise ValueError("sampled_times_not_divisible_by_patch")
    out = []
    for g in range(n // temporal_patch_size):
        lo = float(sampled_seconds[g * temporal_patch_size])
        hi = float(sampled_seconds[g * temporal_patch_size + temporal_patch_size - 1])
        if hi >= start and lo <= end:
            out.append(g)
    return tuple(out)
