"""Summarise a flow_v2.py dev run (exploratory; dev only, no test data).

Per task: baseline accuracy vs chance (exact binomial, one-sided), mean base margin.
Per task x k x layer band: mean d_margin for real and null blocking and real - null,
with video-level bootstrap 95% CI and a two-sided bootstrap p; BH-FDR over all real - null cells.
Usage: python analyze_dev40.py OUT.json run1.jsonl [run2.jsonl ...]
"""
import sys, json, math
from collections import defaultdict
import numpy as np

BANDS = {"L0-5": range(0, 6), "L6-15": range(6, 16), "L16-25": range(16, 26), "L26-31": range(26, 32)}
N_BOOT, SEED = 2000, 0


def binom_sf(x, n, p):  # P(X >= x)
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(x, n + 1))


def boot(vals, rng):
    vals = np.asarray(vals)
    idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
    bm = vals[idx].mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    p = min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean()))
    return float(vals.mean()), float(lo), float(hi), float(max(p, 1 / N_BOOT))


def bh(ps):
    ps = np.asarray(ps); n = len(ps); o = np.argsort(ps)
    q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, ps[i] * n / rank); q[i] = prev
    return q


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    metas, base = [], {}
    D = defaultdict(lambda: defaultdict(dict))  # (task,k,cond)[video][layer] = d_margin
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if "meta" in r: metas.append(r["meta"]); continue
            key = (r["task"], r["video"])
            base[key] = (r["base_margin"], r["gt"] == r["pred"], len(r["letters"]))
            D[(r["task"], r["k"], r["cond"])][r["video"]][r["layer"]] = r["d_margin"]
    rng = np.random.default_rng(SEED)
    tasks = sorted({t for t, _ in base})
    ks = sorted({k for _, k, _ in D})
    res = {"inputs": paths, "metas": metas, "n_boot": N_BOOT, "seed": SEED, "baseline": {}, "cells": []}
    for t in tasks:
        v = [b for (tt, _), b in base.items() if tt == t]
        n, c = len(v), sum(b[1] for b in v); ch = float(np.mean([1 / b[2] for b in v]))
        res["baseline"][t] = {"n": n, "correct": c, "acc": c / n, "chance": ch,
                              "p_above_chance": binom_sf(c, n, ch), "mean_base_margin": float(np.mean([b[0] for b in v]))}
    for t in tasks:
        for k in ks:
            vids = sorted(D[(t, k, "real")])
            for band, Ls in BANDS.items():
                per = {cond: [np.mean([D[(t, k, cond)][vd][L] for L in Ls]) for vd in vids] for cond in ("real", "null")}
                cell = {"task": t, "k": k, "band": band, "n_videos": len(vids)}
                for cond in ("real", "null"):
                    m, lo, hi, _ = boot(per[cond], rng); cell[cond] = [m, lo, hi]
                m, lo, hi, p = boot(np.array(per["real"]) - np.array(per["null"]), rng)
                cell["real_minus_null"] = [m, lo, hi]; cell["p_boot"] = p
                res["cells"].append(cell)
    for cell, q in zip(res["cells"], bh([c["p_boot"] for c in res["cells"]])):
        cell["q_bh"] = float(q)
    json.dump(res, open(out, "w"), indent=1)
    for t, b in res["baseline"].items():
        print(f"{t:17s} acc {b['correct']}/{b['n']}={b['acc']:.2f} chance {b['chance']:.2f} p>chance {b['p_above_chance']:.3g} base_margin {b['mean_base_margin']:+.2f}")
    print("task              k band    real            null            real-null [95%CI]        q")
    for c in res["cells"]:
        r, nl, d = c["real"], c["null"], c["real_minus_null"]
        print(f"{c['task']:17s} {c['k']:2d} {c['band']:6s} {r[0]:+.3f} {nl[0]:+.3f}  {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {c['q_bh']:.3f}{' *' if c['q_bh'] < 0.05 else ''}")


if __name__ == "__main__":
    main()
