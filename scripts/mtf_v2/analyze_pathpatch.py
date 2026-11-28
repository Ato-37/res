"""Prespecified analysis of flow_pathpatch.py. Unit = video. Per path: mean over all heads of |d_frozen| minus mean |d_free|;
t 95% CI and Holm over the paths. 'Detour hid the heads' = Holm p < .05 with a positive difference AND ratio of means >= 1.5."""
import sys, json, numpy as np
from collections import defaultdict
from scipy import stats
rows = {}
for f in sys.argv[1:]:
    for line in open(f):
        try: r = json.loads(line)
        except ValueError: continue
        if "path" in r: rows[(r["path"], r["task"], r["video"], r["layer"], r["head"])] = r
by = defaultdict(lambda: defaultdict(list)); whole = defaultdict(list)
for (p, t, v, l, h), r in rows.items():
    if l < 0: whole[(p, t)].append((r["d_free"], r["d_frozen"]))
    else: by[(p, t)][v].append((abs(r["d_free"]), abs(r["d_frozen"])))
res = {}
for k, vids in by.items():
    n_heads = {len(x) for x in vids.values()}
    m = np.array([np.mean(x, 0) for x in vids.values()]); mx = np.array([np.max(x, 0) for x in vids.values()])
    d = m[:, 1] - m[:, 0]; ci = stats.t.interval(.95, len(d) - 1, d.mean(), stats.sem(d))
    res[k] = (len(d), n_heads, m.mean(0), d.mean(), ci, stats.ttest_1samp(d, 0).pvalue, mx.mean(0), int((d > 0).sum()))
order = sorted(res, key=lambda k: res[k][5]); adj = 0
for i, k in enumerate(order):
    n, nh, m, d, ci, p, mx, pos = res[k]; adj = max(adj, min(1, p * (len(order) - i)))
    ratio = m[1] / m[0]
    print(f"{k[0]}:{k[1]} videos {n} heads {sorted(nh)} | mean|d| free {m[0]:.4f} frozen {m[1]:.4f} ratio {ratio:.2f} | diff {d:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] Holm p {adj:.4f} positive {pos}/{n} | max|d| free {mx[0]:.3f} frozen {mx[1]:.3f} | HID {adj < .05 and d > 0 and ratio >= 1.5}")
    w = np.array(whole[k]); print(f"    whole range: free {w[:, 0].mean():+.3f} frozen {w[:, 1].mean():+.3f} (mean|d| {np.abs(w[:, 0]).mean():.3f} / {np.abs(w[:, 1]).mean():.3f})")
