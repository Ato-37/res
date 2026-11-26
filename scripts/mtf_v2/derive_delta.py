"""delta_min derivation and power from dev outputs (k=7, centre layers 6-15, row-matched null). endpoint = d_margin*sign(base_margin) (conf) or d_margin (signed).
Usage: derive_delta.py FILE"""
import json, sys, collections
import numpy as np
from scipy import stats
D = collections.defaultdict(list); base = {}
for l in open(sys.argv[1]):
    d = json.loads(l)
    if "meta" in d or "d_margin" not in d or d["k"] != 7 or not 6 <= d["layer"] <= 15: continue
    D[(d["task"], d["video"], d["cond"])].append(d["d_margin"]); base[(d["task"], d["video"])] = d["base_margin"]
sp = json.load(open("scripts/mtf_v2/tvbench_split_v1.json"))["tasks"]
def power(n, delta, sd):   # TOST power at true effect 0, alpha .05
    se = sd / n ** .5; return max(0.0, 2 * stats.t.cdf(delta / se - stats.t.ppf(.95, n - 1), n - 1) - 1)
for t in ["Action Antonym", "Scene Transition"]:
    vs = sorted(v for tt, v in base if tt == t); n_test = len(sp[t]["test"])
    b = np.array([abs(base[(t, v)]) for v in vs])
    for name in ["conf", "signed"]:
        s = [np.sign(base[(t, v)]) if name == "conf" else 1.0 for v in vs]
        r = np.array([np.mean(D[(t, v, "real")]) for v in vs]) * s; nu = np.array([np.mean(D[(t, v, "null")]) for v in vs]) * s
        x = r - nu; sd = x.std(ddof=1)
        print(f"{t} {name}: n_dev={len(vs)} real-null {x.mean():+.3f} sd {sd:.3f} | (b) 0.2*sd {0.2*sd:.3f} (c) null sd {nu.std(ddof=1):.3f} | n_test={n_test}")
        for dl in [0.05, 0.1, 0.2, 0.3]:
            print(f"    delta {dl}: (a) share |base|<=delta {np.mean(b<=dl):.2f}  TOST power {power(n_test, dl, sd):.2f}")
