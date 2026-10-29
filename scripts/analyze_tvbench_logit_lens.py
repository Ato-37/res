#!/usr/bin/env python3
"""Prespecified analysis of run_tvbench_logit_lens.py (plan: experiments/2026-10-29_map_the_flow_logit_lens_plan.md); derived counts only."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED, BOOT = 20261029, 10000
BANDS = {"early_1_4": range(0, 4), "mid_9_20": range(8, 20), "late_21_28": range(20, 28)}


def rates(rec, arm, words, kind):
    idx = [i for i, w in enumerate(words) if w["kind"] == kind]
    n = rec["n_video_tokens"]
    return [sum(row[i] for i in idx) / n for row in rec[arm]]


def band_mean(per_layer, band):
    return sum(per_layer[i] for i in band) / len(band)


def ci(vals, rng):
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT))
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]


def analyze(receipt):
    recs, words = receipt["records"], receipt["words"]
    rng = random.Random(SEED)
    mean = lambda v: sum(v) / len(v)
    R = {(arm, kind): [rates(r, arm, words, kind) for r in recs] for arm in ("counts_baseline", "counts_crossframe_blocked") for kind in ("temporal", "spatial")}
    out = {"n": len(recs), "mean_rate_per_layer": {f"{a}:{k}": [round(mean([v[i] for v in vs]), 5) for i in range(len(vs[0]))] for (a, k), vs in R.items()}}
    def stat(vals):
        lo, hi = ci(vals, rng)
        return {"point": mean(vals), "ci95": [lo, hi]}
    for kind in ("temporal", "spatial"):
        base = R[("counts_baseline", kind)]
        out[f"L1_mid_minus_early_{kind}"] = stat([band_mean(v, BANDS["mid_9_20"]) - band_mean(v, BANDS["early_1_4"]) for v in base])
        out[f"L1_late_minus_mid_{kind}"] = stat([band_mean(v, BANDS["late_21_28"]) - band_mean(v, BANDS["mid_9_20"]) for v in base])
        cf = R[("counts_crossframe_blocked", kind)]
        out[f"L2_crossframe_minus_baseline_mid_{kind}"] = stat([band_mean(c, BANDS["mid_9_20"]) - band_mean(b, BANDS["mid_9_20"]) for b, c in zip(base, cf)])
    dd = [(band_mean(c, BANDS["mid_9_20"]) - band_mean(b, BANDS["mid_9_20"])) - (band_mean(cs, BANDS["mid_9_20"]) - band_mean(bs, BANDS["mid_9_20"]))
          for b, c, bs, cs in zip(R[("counts_baseline", "temporal")], R[("counts_crossframe_blocked", "temporal")], R[("counts_baseline", "spatial")], R[("counts_crossframe_blocked", "spatial")])]
    out["L2_difference_of_differences_temporal_minus_spatial"] = stat(dd)
    ok = lambda s: s["ci95"][0] > 0
    out["criteria"] = {"L1_temporal_mid_gt_early": ok(out["L1_mid_minus_early_temporal"]),
                       "L2_crossframe_reduces_temporal_mid": out["L2_crossframe_minus_baseline_mid_temporal"]["ci95"][1] < 0,
                       "L2_reduction_larger_for_temporal_than_spatial": out["L2_difference_of_differences_temporal_minus_spatial"]["ci95"][1] < 0}
    tot = {}
    for w_i, w in enumerate(words):
        tot[w["word"]] = sum(row[w_i] for r in recs for row in r["counts_baseline"])
    out["total_hits_baseline_by_word"] = tot
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--receipt", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    receipt = json.loads(Path(a.receipt).read_text())
    res = analyze(receipt)
    res["receipt_name"] = Path(a.receipt).name
    res["cohort_sha256"] = receipt.get("cohort_sha256")
    out = Path(a.output)
    if out.exists():
        raise SystemExit("output_exists")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in res if k.startswith(("L1", "L2", "criteria", "n"))}, indent=1))


if __name__ == "__main__":
    main()
