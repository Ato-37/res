#!/usr/bin/env python3
"""Analysis of WP1D (experiments/2026-10-25_wp1_overlap_dose_plan.md).

Primary: per-video least-squares slope of the video->text knockout margin change (nat) on the achieved overlap fraction m/k (range 0..1),
so the slope is the change from 0% to 100% overlap along the fitted line; effect = -slope. Supported iff t, sign-flip and Wilcoxon p < 0.05
and mean effect >= 0.3 nat; else TOST at +/-0.3 nat or inconclusive. Secondary (descriptive): mean margin change by overlap level,
0% vs 100% contrast (replicates WP1 P1 on the same videos), middle-level deviation from the straight line between the ends,
positive control (whole-video video->text, mean relative change <= -20% on baseline-correct videos). Privacy-safe summary (no IDs).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from analyze_annotation_key_bias_confirmatory import analyze

DELTA_MIN = 0.3


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def dose_points(record: dict) -> list[tuple[float, float]]:
    """(overlap fraction, margin change vs baseline) for each dose arm of one video."""
    return sorted((k["overlap_fraction"], k["margin"] - record["baseline_margin"])
                  for k in record["knockouts"] if k.get("overlap_fraction") is not None)


def by_fraction(record: dict) -> dict[float, float]:
    """Mean margin change per achieved overlap fraction (several sliding windows can share a fraction)."""
    acc: dict[float, list[float]] = {}
    for f, y in dose_points(record):
        acc.setdefault(round(f, 4), []).append(y)
    return {f: st.fmean(v) for f, v in acc.items()}


def slope(points: list[tuple[float, float]]) -> float | None:
    xs = [p[0] for p in points]
    if len(set(xs)) < 2:
        return None
    mx, my = st.fmean(xs), st.fmean(p[1] for p in points)
    return sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)


def decide(res: dict) -> str:
    if (res["p_two_sided_t"] < 0.05 and res["p_sign_flip"] < 0.05 and res["p_wilcoxon"] < 0.05 and res["mean_nats"] >= DELTA_MIN):
        return "supported"
    return "equivalent_within_delta_min" if res["tost_p"] < 0.05 else "inconclusive"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--permutations", type=int, default=100_000)
    args = ap.parse_args(argv)
    records = load(args.receipt)
    used = [r for r in records if not r.get("excluded")]
    out = {"schema_version": 1, "purpose": "wp1d_overlap_dose_summary", "recorded": len(records), "excluded": len(records) - len(used),
           "analysed": len(used), "delta_min_nat": DELTA_MIN}
    slopes = [s for s in (slope(dose_points(r)) for r in used) if s is not None]
    out["n_with_slope"] = len(slopes)
    prim = analyze([-s for s in slopes], DELTA_MIN, permutations=args.permutations)
    prim["decision"] = decide(prim)
    out["primary_effect_minus_slope"] = prim
    levels = {}
    for r in used:
        for f, y in dose_points(r):
            levels.setdefault(round(f, 2), []).append(y)
    out["margin_change_by_overlap_level"] = {str(f): {"n": len(v), "mean": st.fmean(v), "sd": st.stdev(v) if len(v) > 1 else None}
                                              for f, v in sorted(levels.items())}
    ends = [(by_fraction(r).get(0.0), by_fraction(r).get(1.0)) for r in used]
    ends = [(a, b) for a, b in ends if a is not None and b is not None]
    out["zero_vs_full_overlap"] = analyze([a - b for a, b in ends], DELTA_MIN, permutations=args.permutations)
    mid = []
    for r in used:
        d = by_fraction(r)
        if 0.0 in d and 1.0 in d and 0.5 in d:
            mid.append(d[0.5] - (d[0.0] + d[1.0]) / 2)
    out["middle_minus_line_between_ends"] = ({"n": len(mid), "mean": st.fmean(mid), "sd": st.stdev(mid)} if len(mid) > 1 else None)
    pc = [k["relative_change_percent"] for r in used if r.get("baseline_correct") for k in r["knockouts"] if k["arm"] == "v2t_allvideo_band"]
    out["positive_control_mean_relative_change_percent"] = st.fmean(pc) if pc else None
    out["positive_control_passed"] = bool(pc) and st.fmean(pc) <= -20.0
    out["scope"] = "dependence of the answer margin on the annotated interval along the video->text pathway (label-free); not explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    p = out["primary_effect_minus_slope"]
    print(f"recorded {out['recorded']} (excluded {out['excluded']}), with slope {out['n_with_slope']}; positive control passed: {out['positive_control_passed']}")
    print(f"primary (effect = -slope, nat per 0->100% overlap): {p['mean_nats']:+.3f} [{p['ci95_t'][0]:+.3f},{p['ci95_t'][1]:+.3f}] "
          f"t p={p['p_two_sided_t']:.3g} Wilcoxon p={p['p_wilcoxon']:.3g} TOST p={p['tost_p']:.3g} -> {p['decision']}")
    print("margin change by overlap level:", {k: round(v["mean"], 3) for k, v in out["margin_change_by_overlap_level"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
