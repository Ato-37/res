#!/usr/bin/env python3
"""Post-hoc pooled summary of the two Action Sequence window-expansion batches (A and B, disjoint videos, same protocol), for the slides only.
Same per-video OLS as analyze_window_expansion.py (condition 'eff'), per-video coefficients concatenated; bootstrap 95% CI (10k, seed 0) and paired t p.
Not a prespecified analysis (done after seeing both batches). Derived-only output."""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_window_expansion import per_video

ap = argparse.ArgumentParser(); ap.add_argument("--receipts", nargs="+", required=True); ap.add_argument("--output", required=True); a = ap.parse_args()
rng = random.Random(0)
rows = []
for f in a.receipts:
    d = json.load(open(f)); rows += [r for r in d["records"] if not r.get("excluded") and r["baseline_correct"]]
def summ(v):
    v = np.array(v); n = len(v)
    m = sorted(float(v[[rng.randrange(n) for _ in range(n)]].mean()) for _ in range(10000))
    return {"n": n, "mean": round(float(v.mean()), 2), "ci95_boot": [round(m[249], 2), round(m[9749], 2)], "t_p": round(float(stats.ttest_1samp(v, 0).pvalue), 4)}
b0 = [x for x in (per_video(r, "eff", False) for r in rows) if x is not None]
b1 = [x for x in (per_video(r, "eff", True) for r in rows) if x is not None]
out = {"baseline_correct_videos": len(rows), "amount_blocked_per_full_video": summ([b[1] for b in b0]), "evidence_coverage_0to1": summ([b[2] for b in b0]),
       "includes_group0": summ([b[3] for b in b1]), "note": "post-hoc pooling of two disjoint batches; descriptive"}
if Path(a.output).exists(): raise SystemExit("output_exists")
json.dump(out, open(a.output, "w"), indent=1); print(json.dumps(out, indent=1))
