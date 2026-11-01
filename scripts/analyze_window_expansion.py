#!/usr/bin/env python3
"""Analysis for WP2-TVB (plan: experiments/2026-10-31_tvbench_window_expansion_plan.md). One prespecified primary analysis; rest descriptive.

Per baseline-correct, non-excluded video and condition: OLS of the window relative change (%p) on [1, w/G, o/k] over windows with width < G.
Primary = mean over videos of the coefficient on o/k in condition 'eff'. Sensitivity adds an indicator for 'window includes group 0'.
Writes a derived-only JSON summary (no raw text)."""
import argparse, json, random
import numpy as np
from scipy import stats

DELTA_MIN = 6.7  # pp, 0.2 x SD of the per-video evidence-minus-sham %p contrast in WP1-TVB (plan)


def per_video(rec, cond, with_g0):
    ks = [k for k in rec["knockouts"] if k["condition"] == cond and k["width"] < rec["temporal_group_count"]]
    G, kk = rec["temporal_group_count"], len(rec["annotation_group_indices"])
    X = [[1.0, k["width"] / G, k["overlap"] / kk] + ([1.0 if k["start"] == 0 else 0.0] if with_g0 else []) for k in ks]
    y = [k["relative_change_percent"] for k in ks]
    X = np.array(X); y = np.array(y)
    if len(y) < X.shape[1] + 1 or np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta


def summarize(v, rng):
    v = np.array(v); n = len(v)
    boot = sorted(float(np.mean(v[[rng.randrange(n) for _ in range(n)]])) for _ in range(10000))
    se = v.std(ddof=1) / np.sqrt(n)
    t = stats.ttest_1samp(v, 0.0)
    signs = np.array([[rng.choice((-1, 1)) for _ in range(n)] for _ in range(20000)])
    p_sf = float(np.mean(np.abs((signs * v).mean(1)) >= abs(v.mean()) - 1e-12))
    w = stats.wilcoxon(v)
    tost_lo = stats.ttest_1samp(v, -DELTA_MIN, alternative="greater").pvalue
    tost_hi = stats.ttest_1samp(v, DELTA_MIN, alternative="less").pvalue
    return {"n": n, "mean": round(float(v.mean()), 3), "ci95_boot": [round(boot[249], 3), round(boot[9749], 3)], "t_p": float(t.pvalue),
            "signflip_p": p_sf, "wilcoxon_p": float(w.pvalue), "tost_p": float(max(tost_lo, tost_hi))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--receipt", required=True); ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = json.load(open(a.receipt)); rng = random.Random(0)
    rows = [r for r in d["records"] if not r.get("excluded") and r["baseline_correct"]]
    out = {"receipt_created_at": d["created_at_utc"], "requested": d["requested_videos"], "recorded": d["recorded_videos"], "excluded": d["excluded_videos"],
           "baseline_correct": len(rows), "delta_min_pp": DELTA_MIN, "regression": {}, "descriptive": {}}
    for cond in ("eff", "noneff", "all"):
        for g0 in (False, True):
            bs = [per_video(r, cond, g0) for r in rows]; bs = [b for b in bs if b is not None]
            if len(bs) < 5: continue
            key = f"{cond}{'+g0' if g0 else ''}"
            out["regression"][key] = {"videos_with_estimate": len(bs), "coef_width_per_full_video": summarize([b[1] for b in bs], rng),
                                      "coef_evidence_coverage_0to1": summarize([b[2] for b in bs], rng)}
            if g0: out["regression"][key]["coef_includes_group0"] = summarize([b[3] for b in bs], rng)
    G = 16
    for cond in ("eff", "noneff", "all"):
        tab = {}
        for r in rows:
            for k in r["knockouts"]:
                if k["condition"] != cond: continue
                tab.setdefault((k["width"] == r["temporal_group_count"], round(k["width"] / r["temporal_group_count"], 2), k["overlap"] > 0), []).append(k["relative_change_percent"])
        out["descriptive"][cond] = [{"whole_video": wv, "width_frac": wf, "covers_evidence": cv, "n_windows": len(v), "mean_pct": round(float(np.mean(v)), 2)}
                                    for (wv, wf, cv), v in sorted(tab.items())]
    json.dump(out, open(a.output, "w"), indent=1)
    for k, v in out["regression"].items(): print(k, "evidence-coverage coef:", v["coef_evidence_coverage_0to1"])


if __name__ == "__main__":
    main()
