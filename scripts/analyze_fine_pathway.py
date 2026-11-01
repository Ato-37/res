#!/usr/bin/env python3
"""Aggregates for run_tvbench_pathway_fine.py (derived-only). Per arm: mean relative change of the predicted-option probability (%) over baseline-correct videos,
bootstrap 95% CI (video unit, 10k, seed 0). Also: temporal-group profile (v2t_band_g*), group-0 vs the mean of the other groups (paired per video)."""
import argparse, json, random
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--receipt", required=True); ap.add_argument("--output", required=True); a = ap.parse_args()
d = json.load(open(a.receipt)); rng = random.Random(0)
rows = [r for r in d["records"] if not r.get("excluded") and r["baseline_correct"]]
def ci(v):
    v = np.array(v); n = len(v); m = sorted(float(v[[rng.randrange(n) for _ in range(n)]].mean()) for _ in range(10000)); return [round(m[249], 2), round(m[9749], 2)]
arms = {}
for r in rows:
    for k in r["knockouts"]: arms.setdefault(k["arm"], []).append(k["relative_change_percent"])
out = {"n_videos": len(rows), "arms": {k: {"n": len(v), "mean_pct": round(float(np.mean(v)), 2), "ci95": ci(v)} for k, v in sorted(arms.items())}}
g0 = [next(k["relative_change_percent"] for k in r["knockouts"] if k["arm"] == "v2t_band_g0") for r in rows]
oth = [float(np.mean([k["relative_change_percent"] for k in r["knockouts"] if k["arm"].startswith("v2t_band_g") and k["arm"] != "v2t_band_g0"])) for r in rows]
diff = [x - y for x, y in zip(g0, oth)]
out["group0_minus_mean_other_groups_pct"] = {"mean": round(float(np.mean(diff)), 2), "ci95": ci(diff)}
json.dump(out, open(a.output, "w"), indent=1)
for k, v in out["arms"].items(): print(k, v["mean_pct"], v["ci95"])
print("group0 - others", out["group0_minus_mean_other_groups_pct"])
