#!/usr/bin/env python3
"""Analysis of run_tvbench_lens_evidence.py (plan: experiments/2026-10-31_tvbench_lens_evidence_plan.md). One primary analysis; rest descriptive. Derived-only output.
d(l, group) = mean over tokens of [log P_lens(gold-only words) - log P_lens(foil-only words)]. Per video C(l) = mean_{evidence groups} d - mean_{other groups} d.
Primary: C averaged over layers 9-20 (tuned linear lens); test mean(C) vs 0 (paired t, sign-flip, Wilcoxon, bootstrap CI), d_z, share of videos with C>0 (Wilson, sign test)."""
import argparse, json, random
import numpy as np
from scipy import stats
ap = argparse.ArgumentParser(); ap.add_argument("--receipt", required=True); ap.add_argument("--output", required=True); a = ap.parse_args()
d = json.load(open(a.receipt)); rng = random.Random(0)
recs = [r for r in d["records"] if not r.get("excluded")]
L = d["layer_count"]
def contrast(r, layers, drop_g0=False):
    g = np.array(r["log_mass_gold"]); f = np.array(r["log_mass_foil"]); x = g - f  # [L, G]
    G = x.shape[1]; ev = [i for i in r["annotation_group_indices"]]; ot = [i for i in range(G) if i not in ev]
    if not ev or not ot: return None
    if drop_g0 and 0 in ev: return None
    return float(x[layers][:, ev].mean() - x[layers][:, ot].mean())
def ci(v):
    v = np.array(v, float); n = len(v); m = sorted(float(v[[rng.randrange(n) for _ in range(n)]].mean()) for _ in range(10000)); return [round(m[249], 4), round(m[9749], 4)]
def summ(vals):
    v = np.array([x for x in vals if x is not None], float); n = len(v)
    t = stats.ttest_1samp(v, 0.0); w = stats.wilcoxon(v)
    signs = np.array([[rng.choice((-1, 1)) for _ in range(n)] for _ in range(20000)]); p_sf = float(np.mean(np.abs((signs * v).mean(1)) >= abs(v.mean()) - 1e-12))
    pos = int((v > 0).sum()); bt = stats.binomtest(pos, n, 0.5); pc = bt.proportion_ci(confidence_level=0.95, method="wilson")
    return {"n": n, "mean": round(float(v.mean()), 4), "ci95_boot": ci(v), "d_z": round(float(v.mean() / v.std(ddof=1)), 3), "t_p": float(t.pvalue), "signflip_p": p_sf, "wilcoxon_p": float(w.pvalue),
            "share_positive": round(pos / n, 3), "wilson95": [round(pc.low, 3), round(pc.high, 3)], "sign_test_p": float(bt.pvalue)}
mid = list(range(8, 20)); late = list(range(20, 28))
out = {"n_test": d["n_test"], "analysed": len(recs), "baseline_correct": sum(r["baseline_correct"] for r in recs)}
out["primary_mid_9_20"] = summ([contrast(r, mid) for r in recs])
out["late_21_28"] = summ([contrast(r, late) for r in recs])
out["mid_excluding_group0_evidence"] = summ([contrast(r, mid, True) for r in recs])
out["mid_baseline_correct_only"] = summ([contrast(r, mid) for r in recs if r["baseline_correct"]])
out["layer_profile"] = {str(l + 1): summ([contrast(r, [l]) for r in recs]) for l in range(L)}
out["last_token_margin_mean_by_layer"] = [round(float(np.mean([r["last_token_margin"][l] for r in recs])), 4) for l in range(L)]
out["mean_gold_foil_mass_by_layer_evidence_vs_other"] = {}
ev_m = [[] for _ in range(L)]; ot_m = [[] for _ in range(L)]
for r in recs:
    x = np.array(r["log_mass_gold"]) - np.array(r["log_mass_foil"]); G = x.shape[1]; ev = r["annotation_group_indices"]; ot = [i for i in range(G) if i not in ev]
    for l in range(L): ev_m[l].append(x[l][ev].mean()); ot_m[l].append(x[l][ot].mean())
out["mean_gold_foil_mass_by_layer_evidence_vs_other"] = {"evidence": [round(float(np.mean(v)), 4) for v in ev_m], "other": [round(float(np.mean(v)), 4) for v in ot_m]}
json.dump(out, open(a.output, "w"), indent=1)
p = out["primary_mid_9_20"]
print("primary", p)
print("late", out["late_21_28"]["mean"], out["late_21_28"]["ci95_boot"], "| excl g0", out["mid_excluding_group0_evidence"]["mean"], out["mid_excluding_group0_evidence"]["n"], "| baseline-correct", out["mid_baseline_correct_only"]["mean"], out["mid_baseline_correct_only"]["n"])
print("layer profile means", [out["layer_profile"][str(l + 1)]["mean"] for l in range(L)])
