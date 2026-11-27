"""Gate (fixed before the run): per (path, task) the full-range all-heads block must move the output on 30 videos: two-sided t interval at 97.5%
(Bonferroni over 2 endpoints) of signed d_full or of confidence change d_full*sign(base_margin) excludes 0. Prints passing pairs as 'path:task;...'.
Usage: gate_loo.py GATE_FILE"""
import json, sys
import numpy as np
from scipy import stats
rows = [json.loads(l) for l in open(sys.argv[1])][1:]
out = []
for path in ["v2q", "q2l", "ff"]:
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["path"] == path and r["task"] == task]
        if not rs: continue
        ok = False; msg = []
        for name in ["signed", "conf"]:
            x = np.array([r["d_full"] * (np.sign(r["base_margin"]) if name == "conf" else 1) for r in rs]); n = len(x)
            h = stats.t.ppf(1 - .025 / 2, n - 1) * x.std(ddof=1) / n ** .5; m = x.mean(); ok |= bool(abs(m) > h)
            msg.append(f"{name} {m:+.3f} [{m-h:+.3f},{m+h:+.3f}]")
        diff = max(abs(r["d_full"] - r["d_full_32head_mask"]) for r in rs)
        print(path, task, "n=%d" % len(rs), *msg, "| maxdiff 32-head vs 1-head mask %.4f" % diff, "| PASS" if ok else "| fail", file=sys.stderr)
        if ok: out.append(f"{path}:{task}")
print(";".join(out))
