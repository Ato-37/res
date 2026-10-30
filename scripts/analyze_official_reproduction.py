#!/usr/bin/env python3
"""Derived-only summary of official Map-the-Flow outputs (plan: experiments/2026-10-29_official_code_reproduction_plan.md).

Reads <root>/<group>/tvbench/<target>/*/jsons/*.json, reports per task: n records, mean relative probability change (%),
paired-resample 95% CI over samples (video = unit approximated by data_id), and layer-band means. No raw text is written.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

SEED, BOOT = 20261029, 5000
GROUPS = {"I1_table2_first_half": "table2_first_half/tvbench/cross-frame", "I2_tableB_second_half": "tableB_second_half/tvbench/cross-frame",
          "A_cross_frame": "tvbench/cross-frame", "B_vql_to_ql": "tvbench/vql-to-ql", "C_q_opts_to_last": "tvbench/question-and-options-to-last",
          "D_vq_to_true_opt": "tvbench/vq-to-true-opt"}


def ci(vals, rng):
    n = len(vals)
    m = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT))
    return round(m[int(0.025 * BOOT)], 2), round(m[int(0.975 * BOOT) - 1], 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out, rng = {}, random.Random(SEED)
    for g, sub in GROUPS.items():
        for f in sorted(glob.glob(f"{a.root}/{sub}/*/jsons/*.json")):
            task = Path(f).stem.split("_", 1)[1]
            rows = json.loads(Path(f).read_text())
            by_layer = defaultdict(list)
            for r in rows:
                by_layer[r["layer"]].append(r["relative_diff"])
            allv = [r["relative_diff"] for r in rows]
            n_q = len({r["data_id"] for r in rows})
            bands = {}
            if len(by_layer) > 4:
                ls = sorted(by_layer)
                third = len(ls) // 3
                for name, sel in (("early", ls[:third]), ("mid", ls[third:2 * third]), ("late", ls[2 * third:])):
                    v = [x for l in sel for x in by_layer[l]]
                    bands[name] = round(sum(v) / len(v), 2)
            out.setdefault(g, {})[task] = {"n_questions": n_q, "n_records": len(rows), "mean_rel_diff_pct": round(sum(allv) / len(allv), 2),
                                           "ci95_records": ci(allv, rng), "layer_bands": bands}
    path = Path(a.output)
    if path.exists():
        raise SystemExit("output_exists")
    path.write_text(json.dumps(out, indent=1))
    for g, t in out.items():
        print(g)
        for k, v in t.items():
            print(f"  {k:18s} nq={v['n_questions']:3d} mean={v['mean_rel_diff_pct']:7.2f}% CI{v['ci95_records']} {v['layer_bands']}")


if __name__ == "__main__":
    main()
