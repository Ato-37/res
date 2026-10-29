#!/usr/bin/env python3
"""Prespecified analysis of run_tvbench_pruning.py (plan: experiments/2026-10-29_map_the_flow_pruning_plan.md). Writes a derived summary (no raw text)."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

SEED = 20261029
BOOT = 10000


def acc_vectors(records):
    base, path, rnd, tasks, chance = [], [], [], [], []
    for r in records:
        g = r["gold"]
        base.append(float(r["predicted"]["baseline"] == g))
        path.append(float(r["predicted"]["pathway"] == g))
        rnd.append(sum(p == g for p in r["predicted"]["random"]) / len(r["predicted"]["random"]))
        tasks.append(r["task"])
        chance.append(1.0 / r["n_options"])
    return base, path, rnd, tasks, chance


def boot_ci(diffs, rng):
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT))
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def analyze(receipt):
    recs = receipt["records"]
    base, path, rnd, tasks, chance = acc_vectors(recs)
    n = len(recs)
    rng = random.Random(SEED)
    mean = lambda v: sum(v) / len(v)
    d_path = [p - b for p, b in zip(path, base)]
    d_ctrl = [p - r for p, r in zip(path, rnd)]
    d_rand = [r - b for r, b in zip(rnd, base)]
    out = {"n": n, "accuracy": {"baseline": mean(base), "pathway": mean(path), "random_mean": mean(rnd), "chance": mean(chance)},
           "blocked_fraction_mean": mean([r["blocked_fraction"] for r in recs])}
    for name, d in (("pathway_minus_baseline", d_path), ("pathway_minus_random", d_ctrl), ("random_minus_baseline", d_rand)):
        lo, hi = boot_ci(d, rng)
        out[name] = {"point_pt": 100 * mean(d), "ci95_pt": [100 * lo, 100 * hi]}
    b = sum(1 for x, y in zip(base, path) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(base, path) if x == 0 and y == 1)
    out["mcnemar_baseline_vs_pathway"] = {"baseline_only_correct": b, "pathway_only_correct": c, "p_exact": exact_mcnemar(b, c)}
    out["criteria"] = {"P1_retention_ci_lower_gt_-3pt": out["pathway_minus_baseline"]["ci95_pt"][0] > -3.0,
                       "P2_control_ci_lower_gt_0_and_point_ge_5pt": out["pathway_minus_random"]["ci95_pt"][0] > 0 and out["pathway_minus_random"]["point_pt"] >= 5.0}
    per = {}
    for t in sorted(set(tasks)):
        idx = [i for i, x in enumerate(tasks) if x == t]
        per[t] = {"n": len(idx), "baseline": mean([base[i] for i in idx]), "pathway": mean([path[i] for i in idx]),
                  "random_mean": mean([rnd[i] for i in idx]), "chance": mean([chance[i] for i in idx])}
    out["per_task"] = per
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--receipt", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    receipt = json.loads(Path(a.receipt).read_text())
    res = analyze(receipt)
    res["receipt_name"] = Path(a.receipt).name
    res["cohort_sha256"] = receipt.get("cohort_sha256")
    res["scaled_layers_1based"] = receipt.get("scaled_layers_1based")
    res["interpretation_limit"] = receipt.get("interpretation_limit")
    out = Path(a.output)
    if out.exists():
        raise SystemExit("output_exists")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n", "accuracy", "blocked_fraction_mean", "pathway_minus_baseline", "pathway_minus_random", "criteria")}, indent=1))


if __name__ == "__main__":
    main()
