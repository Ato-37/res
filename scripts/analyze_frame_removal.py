#!/usr/bin/env python3
"""Analysis of run_tvbench_frame_removal.py (plan: experiments/2026-10-31_tvbench_frame_removal_plan.md). One primary analysis; derived-only output.
Primary: per baseline-correct video, c = %p change(remove_evidence) - %p change(remove_sham); mean + bootstrap CI, paired t / sign-flip / Wilcoxon p, d_z,
proportion of videos with evidence removal hurting more (Wilson CI, sign test). Secondary: the masking contrast on the same videos (--masking-receipt, WP1-TVB) and the paired difference."""
import argparse, json, random
import numpy as np
from scipy import stats
ap = argparse.ArgumentParser(); ap.add_argument("--receipt", required=True); ap.add_argument("--masking-receipt"); ap.add_argument("--output", required=True); a = ap.parse_args()
rng = random.Random(0)
def load(p): return {r["media_sha256"]: r for r in json.load(open(p))["records"] if not r.get("excluded") and r["baseline_correct"]}
R = load(a.receipt)
def arm(r, n, key="arms"): return next(k["relative_change_percent"] for k in r[key] if k["arm"] == n)
def ci(v):
    v = np.array(v, float); n = len(v); m = sorted(float(v[[rng.randrange(n) for _ in range(n)]].mean()) for _ in range(10000)); return [round(m[249], 2), round(m[9749], 2)]
def stat(c):
    c = np.array(c, float); n = len(c); t = stats.ttest_1samp(c, 0.0); w = stats.wilcoxon(c)
    signs = np.array([[rng.choice((-1, 1)) for _ in range(n)] for _ in range(20000)]); p_sf = float(np.mean(np.abs((signs * c).mean(1)) >= abs(c.mean()) - 1e-12))
    neg = int((c < 0).sum()); bt = stats.binomtest(neg, n, 0.5); pc = bt.proportion_ci(confidence_level=0.95, method="wilson")
    return {"n": n, "mean": round(float(c.mean()), 2), "ci95_boot": ci(c), "d_z": round(float(c.mean() / c.std(ddof=1)), 2), "t_p": float(t.pvalue), "signflip_p": p_sf,
            "wilcoxon_p": float(w.pvalue), "n_evidence_hurts_more": neg, "share_evidence_hurts_more": round(neg / n, 3), "wilson95": [round(pc.low, 3), round(pc.high, 3)], "sign_test_p": float(bt.pvalue)}
out = {"requested": None, "baseline_correct": len(R)}
c = [arm(r, "remove_evidence") - arm(r, "remove_sham") for r in R.values()]
out["primary_remove_evidence_minus_remove_sham"] = stat(c)
out["remove_evidence_mean"] = round(float(np.mean([arm(r, "remove_evidence") for r in R.values()])), 2)
out["remove_sham_mean"] = round(float(np.mean([arm(r, "remove_sham") for r in R.values()])), 2)
g0 = [(int(0 in r["annotation_group_indices"]), arm(r, "remove_evidence") - arm(r, "remove_sham")) for r in R.values()]
out["by_group0_in_evidence"] = {str(k): {"n": sum(1 for x in g0 if x[0] == k), "mean": round(float(np.mean([x[1] for x in g0 if x[0] == k])), 2)} for k in (0, 1) if any(x[0] == k for x in g0)}
if a.masking_receipt:
    M = load(a.masking_receipt)
    common = [h for h in R if h in M]
    cm = [arm(M[h], "v2t_evidence_band", "knockouts") - arm(M[h], "v2t_sham_band", "knockouts") for h in common]
    cr = [arm(R[h], "remove_evidence") - arm(R[h], "remove_sham") for h in common]
    out["masking_vs_removal"] = {"n_common": len(common), "masking_contrast": stat(cm), "removal_contrast": stat(cr), "masking_minus_removal": {"mean": round(float(np.mean(np.array(cm) - np.array(cr))), 2), "ci95_boot": ci(np.array(cm) - np.array(cr))},
                                 "pearson_r": round(float(np.corrcoef(cm, cr)[0, 1]), 3) if len(common) > 2 else None}
json.dump(out, open(a.output, "w"), indent=1)
print(json.dumps({k: out[k] for k in out if k != "by_group0_in_evidence"}, indent=1)[:2500]); print(out["by_group0_in_evidence"])
