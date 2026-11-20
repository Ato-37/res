"""CPU analysis of flow_v2 JSONL: per (k, layer) mean d_margin real vs null with bootstrap CI (video = unit). Usage: analyze_pilot.py FILE [--task T]"""
import sys, json, numpy as np
from collections import defaultdict
f = sys.argv[1]; task = sys.argv[sys.argv.index("--task")+1] if "--task" in sys.argv else None
R = [json.loads(l) for l in open(f)][1:]
if task: R = [r for r in R if r["task"] == task]
D = defaultdict(dict)
for r in R: D[(r["k"], r["layer"], r["cond"])][r["video"]] = r["d_margin"]
rng = np.random.default_rng(0)
def ci(x):
    x = np.array(x); b = [rng.choice(x, len(x)).mean() for _ in range(1000)]; return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
conds = sorted({c for _, _, c in D}); ks = sorted({k for k, _, _ in D})
print("videos:", len({v for d in D.values() for v in d}), "conds:", conds)
for k in ks:
    print(f"k={k}")
    for L in range(32):
        row = []
        for c in conds:
            d = D.get((k, L, c), {})
            if d: m, lo, hi = ci(list(d.values())); row.append(f"{c}:{m:+.2f}[{lo:+.2f},{hi:+.2f}]n={len(d)}")
        print(f" L{L:2d} ", "  ".join(row))
