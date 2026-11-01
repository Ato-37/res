#!/usr/bin/env python3
"""Analysis of run_tvbench_pruning_qwenranges.py (PATHSEARCH stage 2). Arms: baseline, LLaVA-scaled pathway (+matched random), Qwen-derived pathway (+matched random).
Primary (one analysis, video = unit, bootstrap 10k seed 20261101): (1) qwen_pathway - qwen_random, (2) qwen_pathway - baseline, (3) qwen_pathway - scaled_pathway (descriptive: the arms block different amounts).
Accuracy differences in percentage points with bootstrap 95% CI and exact McNemar for paired binary outcomes. Derived-only output."""
import argparse, json, math, random
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--receipt", required=True); ap.add_argument("--output", required=True); a = ap.parse_args()
d = json.load(open(a.receipt)); R = d["records"]; rng = random.Random(20261101)
def vec(key):
    out = []
    for r in R:
        p = r["predicted"][key]; g = r["gold"]
        out.append(sum(x == g for x in p) / len(p) if isinstance(p, list) else float(p == g))
    return np.array(out)
V = {k: vec(k) for k in ("baseline", "pathway", "random", "qwen_pathway", "qwen_random")}
def boot(x):
    n = len(x); m = sorted(float(x[[rng.randrange(n) for _ in range(n)]].mean()) for _ in range(10000)); return [round(100 * m[249], 2), round(100 * m[9749], 2)]
def mcn(x, y):
    b = int(((x == 1) & (y == 0)).sum()); c = int(((x == 0) & (y == 1)).sum()); n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n); return {"first_only": b, "second_only": c, "p_exact": round(p, 4)}
out = {"n": len(R), "accuracy_pct": {k: round(100 * float(v.mean()), 2) for k, v in V.items()},
       "blocked_fraction_mean": {"scaled": round(float(np.mean([r["blocked_fraction"] for r in R])), 3), "qwen": round(float(np.mean([r["qwen_blocked_fraction"] for r in R])), 3)}}
for name, x, y in (("qwen_pathway_minus_qwen_random", "qwen_pathway", "qwen_random"), ("qwen_pathway_minus_baseline", "qwen_pathway", "baseline"),
                   ("scaled_pathway_minus_scaled_random", "pathway", "random"), ("scaled_pathway_minus_baseline", "pathway", "baseline"),
                   ("qwen_pathway_minus_scaled_pathway", "qwen_pathway", "pathway")):
    dif = V[x] - V[y]; out[name] = {"mean_pt": round(100 * float(dif.mean()), 2), "ci95_pt": boot(dif)}
out["mcnemar_qwen_pathway_vs_baseline"] = mcn(V["baseline"], V["qwen_pathway"])
out["mcnemar_qwen_vs_scaled_pathway"] = mcn(V["pathway"], V["qwen_pathway"])
tasks = sorted({r["task"] for r in R}); per = {}
for t in tasks:
    idx = [i for i, r in enumerate(R) if r["task"] == t]
    per[t] = {"n": len(idx), **{k: round(100 * float(V[k][idx].mean()), 1) for k in V}}
out["per_task_accuracy_pct"] = per
out["qwen_layers_1based"] = d.get("qwen_layers_1based"); out["scaled_layers_1based"] = d.get("scaled_layers_1based")
if Path(a.output).exists(): raise SystemExit("output_exists")
json.dump(out, open(a.output, "w"), indent=1); print(json.dumps(out, indent=1))
