"""Prespecified layer-range rule (plan log 2026-11-27): pooled videos, per layer t 95% CI of confidence change d_margin*sign(base_margin)
(q2l: real - null); effective = upper bound < 0; range = longest contiguous run. Secondary: signed d_margin. Usage: analyze_paths.py FILE"""
import json, sys
import numpy as np
from scipy import stats
rows = [json.loads(l) for l in open(sys.argv[1])][1:]
D = {(r["task"], r["video"], r["path"], r["layer"]): r["d_margin"] for r in rows}
base = {(r["task"], r["video"]): r["base_margin"] for r in rows}
vids = sorted(base); layers = sorted({r["layer"] for r in rows})
def ci(x):
    n = len(x); m = x.mean(); h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / n ** .5
    return m, m - h, m + h
print("videos", len(vids), "baseline correct", sum(base[v] > 0 for v in vids))
for name in ["conf", "signed"]:
    s = np.array([np.sign(base[v]) if name == "conf" else 1.0 for v in vids])
    for path in ["v2q", "q2l"]:
        eff = []; line = []
        for l in layers:
            x = np.array([D[(*v, path, l)] - (D[(*v, "q2l_null", l)] if path == "q2l" else 0) for v in vids]) * s
            m, lo, hi = ci(x); line.append(f"{l}:{m:+.2f}{'*' if hi < 0 else ('^' if lo > 0 else '')}")
            if hi < 0: eff.append(l)
        runs = []
        for l in eff:
            if runs and l == runs[-1][-1] + 1: runs[-1].append(l)
            else: runs.append([l])
        best = max(runs, key=len) if runs else []
        print(f"\n[{name}] {path}: effective layers {eff}; longest run {best[:1]}..{best[-1:]}\n  " + " ".join(line))
        for t in sorted({v[0] for v in vids}):
            tv = [v for v in vids if v[0] == t]; ts = np.array([np.sign(base[v]) if name == "conf" else 1.0 for v in tv])
            print("  ", t, " ".join(f"{l}:{np.mean(np.array([D[(*v, path, l)] - (D[(*v, 'q2l_null', l)] if path == 'q2l' else 0) for v in tv]) * ts):+.2f}" for l in layers if l % 2 == 0))
