"""Top-10 heads from the search outputs (rule fixed 2026-11-27): score = mean over videos of (d_restore_one - d_full) * -sign(mean d_full)
(positive = restoring the head reduces the effect of the full block). Needs all 320 heads x all videos. Usage: select_loo.py OUT.json FILE..."""
import json, sys, collections
import numpy as np
D = {}; full = {}
for f in sys.argv[2:]:
    for line in open(f):
        try: r = json.loads(line)
        except ValueError: continue
        if "path" not in r: continue
        D[(r["path"], r["task"], r["video"], r["layer"], r["head"])] = r["d"]; full[(r["path"], r["task"], r["video"])] = r["d_full"]
res = {}
for path, task in sorted({k[:2] for k in D}):
    vids = sorted({k[2] for k in D if k[:2] == (path, task)}); heads = sorted({k[3:] for k in D if k[:2] == (path, task)})
    miss = sum((path, task, v, *h) not in D for v in vids for h in heads); assert miss == 0 and len(heads) == 320 and len(vids) == 30, (len(vids), len(heads), miss)
    mf = np.mean([full[(path, task, v)] for v in vids]); s = -np.sign(mf)
    score = {h: float(s * np.mean([D[(path, task, v, *h)] - full[(path, task, v)] for v in vids])) for h in heads}
    top = sorted(heads, key=lambda h: -score[h])[:10]
    x = np.array(list(score.values()))
    res[f"{path}:{task}"] = {"direction": float(s), "mean_d_full": float(mf), "n_videos": len(vids), "top": [list(h) for h in top], "top_scores": [score[h] for h in top],
                            "score_quantiles": np.quantile(x, [0, .05, .5, .95, 1]).tolist()}
json.dump(res, open(sys.argv[1], "x"), indent=1); print(json.dumps(res, indent=1))
