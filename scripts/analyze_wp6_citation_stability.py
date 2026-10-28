#!/usr/bin/env python3
"""Analysis of WP6 (experiments/2026-10-28_wp5_wp6_input_level_and_stability_plan.md): stability of the cited interval across option order.

Per video and shift: group-level IoU between the cited groups at shift s and at shift 0 (usable = parsed, non-empty). Primary: mean IoU over usable (video, shift) pairs minus a permutation null
(each video's shift-0 citation paired with the shift-s citation of a different random video, 1000 permutations); stable iff observed minus null mean >= 0.2 and the bootstrap 95% interval over
videos (mean per video over its usable shifts) has its lower end > 0.1. Secondary: mean pairwise IoU between all five orders, letter-consistency rate, splits by baseline overlap with the annotation,
Spearman of per-video mean IoU with the WP2 necessity effect (optional --wp2). Privacy-safe summary.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

DIFF_MIN = 0.2
LOWER_MIN = 0.1


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def iou(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def usable(s) -> bool:
    return bool(s["parse_ok"]) and bool(s["cited_group_indices"])


def per_video_iou(r) -> list[float]:
    return [iou(r["shift0_cited_group_indices"], s["cited_group_indices"]) for s in r["shifts"] if usable(s)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--wp2", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args(argv)
    rng = random.Random(20261028)
    records = load(args.receipt)
    used = [r for r in records if not r.get("excluded")]
    out = {"schema_version": 1, "purpose": "wp6_citation_stability_summary", "recorded": len(records), "excluded": len(records) - len(used), "analysed": len(used)}
    pv = [(r, per_video_iou(r)) for r in used]
    pv = [(r, v) for r, v in pv if v]
    out["videos_with_any_usable_shift"] = len(pv)
    out["usable_shift_rate"] = sum(usable(s) for r in used for s in r["shifts"]) / max(1, sum(len(r["shifts"]) for r in used))
    vid_means = [st.fmean(v) for _, v in pv]
    obs_pairs = [x for _, v in pv for x in v]
    obs = st.fmean(obs_pairs) if obs_pairs else float("nan")
    # permutation null: pair each video's shift-0 citation with a random other video's shift-s citation
    null_means = []
    base_idx = list(range(len(pv)))
    for _ in range(args.permutations):
        vals = []
        for i, (r, v) in enumerate(pv):
            j = rng.choice(base_idx)
            while j == i and len(pv) > 1:
                j = rng.choice(base_idx)
            other = [s for s in pv[j][0]["shifts"] if usable(s)]
            if not other:
                continue
            s = rng.choice(other)
            vals.append(iou(r["shift0_cited_group_indices"], s["cited_group_indices"]))
        if vals:
            null_means.append(st.fmean(vals))
    null = st.fmean(null_means) if null_means else float("nan")
    boots = sorted(st.fmean(rng.choices(vid_means, k=len(vid_means))) for _ in range(2000)) if vid_means else [float("nan")] * 2000
    lo, hi = boots[50], boots[1950]
    diff = obs - null
    stable = (diff >= DIFF_MIN) and (lo > LOWER_MIN)
    out["primary"] = {"observed_mean_iou": obs, "null_mean_iou": null, "difference": diff, "bootstrap_ci95_observed": [lo, hi],
                      "decision": "stable" if stable else "not_shown_stable"}
    allord = []
    for r, _ in pv:
        sets = [r["shift0_cited_group_indices"]] + [s["cited_group_indices"] for s in r["shifts"] if usable(s)]
        pairs = [iou(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
        if pairs:
            allord.append(st.fmean(pairs))
    out["mean_pairwise_iou_all_orders"] = st.fmean(allord) if allord else None
    lets = [s["letter_mapped_back"] == r["shift0_letter"] for r in used for s in r["shifts"] if s["letter_mapped_back"] is not None and r.get("shift0_letter") is not None]
    out["letter_consistency_rate"] = (sum(lets) / len(lets)) if lets else None
    for label, sel in (("overlap_gt0", lambda r: (r.get("cited_annotation_overlap") or 0) > 0), ("overlap_eq0", lambda r: (r.get("cited_annotation_overlap") or 0) == 0)):
        v = [st.fmean(x) for r, x in pv if sel(r)]
        out[f"mean_iou_{label}"] = {"n": len(v), "mean": st.fmean(v) if v else None}
    if args.wp2:
        by2 = {r["media_sha256"]: r for r in load(args.wp2) if not r.get("excluded")}
        pairs = []
        for r, v in pv:
            w = by2.get(r["media_sha256"])
            if w:
                k = {x["arm"]: x for x in w["knockouts"]}
                nec = -((k["v2t_cited_band"]["self_margin"] - w["baseline_self_margin"]) - (k["v2t_sham_band"]["self_margin"] - w["baseline_self_margin"]))
                pairs.append((st.fmean(v), nec))
        if len(pairs) > 5:
            def rank(x):
                o = sorted(range(len(x)), key=lambda i: x[i]); r_ = [0.0] * len(x)
                for p, i in enumerate(o): r_[i] = float(p)
                return r_
            a, b = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
            m = (len(a) - 1) / 2
            out["spearman_stability_vs_wp2_necessity"] = {"n": len(pairs), "rho": sum((x - m) * (y - m) for x, y in zip(a, b)) / sum((x - m) ** 2 for x in a)}
    out["scope"] = "stability of the generated citation under option-order changes; a reliability measurement, not a faithfulness result"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    p = out["primary"]
    print(f"recorded {out['recorded']} (excluded {out['excluded']}), analysed {out['analysed']}, usable shift rate {out['usable_shift_rate']:.2f}")
    print(f"primary: observed IoU {p['observed_mean_iou']:.3f} vs null {p['null_mean_iou']:.3f}, diff {p['difference']:+.3f}, 95% CI of observed [{lo:.3f},{hi:.3f}] -> {p['decision']}; letter consistency {out['letter_consistency_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
