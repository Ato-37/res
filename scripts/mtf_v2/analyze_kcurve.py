"""real-null signed d_margin by window width k, mean over centre layers 6-15, on the videos of the first file given (k=3,7).
Usage: analyze_kcurve.py FILE_K3_7 FILE... ; later files only add k values; rows are deduplicated by (task, video, k, layer, cond)."""
import json, sys, collections
import numpy as np
from scipy import stats
rows = {}; base = set()
for i, f in enumerate(sys.argv[1:]):
    for l in open(f):
        try: d = json.loads(l)
        except ValueError: continue
        if "meta" in d or "d_margin" not in d: continue
        if i == 0: base.add((d["task"], d["video"]))
        if (d["task"], d["video"]) in base and 6 <= d["layer"] <= 15:
            rows.setdefault((d["task"], d["video"], d["k"], d["layer"], d["cond"]), d["d_margin"])
D = collections.defaultdict(list)
for (t, v, k, layer, c), x in rows.items(): D[(t, k, v, c)].append(x)
def ci(x):
    x = np.array(x); n = len(x); m = x.mean(); h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / n ** .5
    return f"{m:+.3f} [{m-h:+.3f},{m+h:+.3f}]"
for t in sorted({t for t, *_ in D}):
    for k in sorted({k for tt, k, *_ in D if tt == t}):
        vs = sorted({v for tt, kk, v, c in D if tt == t and kk == k and c == "real" and (t, k, v, "null") in D})
        r = [np.mean(D[(t, k, v, "real")]) for v in vs]; nu = [np.mean(D[(t, k, v, "null")]) for v in vs]
        nl = sorted({len(D[(t, k, v, "real")]) for v in vs})
        print(f"{t:18s} k={k:2d} n={len(vs)} layers/video={nl} real {np.mean(r):+.3f} null {np.mean(nu):+.3f} real-null {ci(np.array(r)-np.array(nu))}")
