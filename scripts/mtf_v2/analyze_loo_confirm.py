"""Primary: per video recovery(top10) - mean recovery(random sets), recovery = (d_set - d_full) * direction; t 95% CI. Usage: analyze_loo_confirm.py FILE"""
import json, sys
import numpy as np
from scipy import stats
direction = {}; rows = []
for line in open(sys.argv[1]):
    r = json.loads(line)
    if "pair" in r: direction[r["pair"]] = r["direction"]
    elif "d" in r: rows.append(r)
def ci(x):
    n = len(x); m = x.mean(); h = stats.t.ppf(1 - .025 / max(1, len(direction)), n - 1) * x.std(ddof=1) / n ** .5
    return f"{m:+.4f} [{m-h:+.4f},{m+h:+.4f}]"
for key, s in direction.items():
    rs = [r for r in rows if f"{r['path']}:{r['task']}" == key]
    full = np.array([r["d"]["full"] for r in rs]); top = np.array([(r["d"]["top"] - r["d"]["full"]) * s for r in rs])
    rnd = np.array([np.mean([(v - r["d"]["full"]) * s for k, v in r["d"].items() if k.startswith("rand")]) for r in rs])
    print(key, "n", len(rs), "| full block", ci(full), "| recovery top10", ci(top), "| recovery random", ci(rnd), "| PRIMARY top-random", ci(top - rnd))
