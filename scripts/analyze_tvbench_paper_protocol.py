#!/usr/bin/env python3
"""Summaries of the paper-protocol receipt (run_tvbench_paper_protocol.py): the paper's Figs 2, 3, 7, 8, 9, Tables 2 and B, the window ablation and the failure cases.

Descriptive (no new hypothesis test): mean %p_change of the originally predicted option over baseline-correct videos per target and layer centre (Figs 2, 3, 7, 8); accuracy drop of cross-frame blocking in the first / second half of
the layers (share of baseline-correct videos whose prediction leaves the gold option; paper Table 2 / B); window size ablation; layer-wise true / false option probabilities from the last-token logit lens (Fig 9);
the failure-case traces (baseline-wrong videos, probability of the wrong prediction). Privacy-safe summary (no IDs).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def mean(v):
    v = list(v)
    return st.fmean(v) if v else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    R = load(args.receipt)
    correct = [r for r in R if r["baseline_correct"]]
    wrong = [r for r in R if not r["baseline_correct"]]
    out = {"schema_version": 1, "purpose": "tvbench_paper_protocol_summary", "videos": len(R), "baseline_correct": len(correct), "baseline_wrong": len(wrong)}
    tasks = sorted({r["task"] for r in R})
    out["accuracy_by_task"] = {t: mean(r["baseline_correct"] for r in R if r["task"] == t) for t in tasks}

    def sweep(recs, target, desc):
        by = defaultdict(list)
        for r in recs:
            for a in r["arms"]:
                if a["kind"] == "sweep" and a["target"] == target and a["desc"] == desc:
                    by[a["layers"][0] + (a["layers"][1] - a["layers"][0]) // 2].append(a["relative_change_percent"])
        return {c: mean(v) for c, v in sorted(by.items())}

    descs = sorted({(a["target"], a["desc"]) for r in R for a in r["arms"] if a["kind"] == "sweep"})
    out["sweeps_correct_mean_percent_change_by_center"] = {f"{t} | {d}": sweep(correct, t, d) for t, d in descs}
    out["sweeps_correct_by_task_cross_frame"] = {t: sweep([r for r in correct if r["task"] == t], "cross-frame", "No cross-frame interactions") for t in tasks}

    def half_drop(desc):
        res = {}
        for t in tasks:
            rs = [r for r in correct if r["task"] == t]
            flips = [any(a["kind"] == "half" and a["desc"] == desc and a["predicted_after"] != r["gold"] for a in r["arms"]) for r in rs]
            res[t] = {"n": len(rs), "accuracy_drop_percent_of_correct": 100.0 * mean(flips) if rs else None}
        return res
    out["table2_first_half_cross_frame"] = half_drop("first half of layers")
    out["tableB_second_half_cross_frame"] = half_drop("second half of layers")
    win = defaultdict(list)
    for r in correct:
        for a in r["arms"]:
            if a["kind"] == "window":
                win[(a["desc"], a["layers"][0] + (a["layers"][1] - a["layers"][0]) // 2)].append((a["relative_change_percent"], a["predicted_after"] != r["gold"]))
    out["window_ablation_cross_frame"] = {f"{d} | centre {c}": {"mean_percent_change": mean(x[0] for x in v), "flip_rate": mean(x[1] for x in v)} for (d, c), v in sorted(win.items())}
    L = len(R[0]["lens_choice_logprobs_per_layer"]) if R else 0
    tr, fl = [[] for _ in range(L)], [[] for _ in range(L)]
    for r in correct:
        for li, row in enumerate(r["lens_choice_logprobs_per_layer"]):
            p = [math.exp(x) for x in row]
            tr[li].append(p[r["gold"]])
            fl[li].append(mean(q for j, q in enumerate(p) if j != r["gold"]))
    out["fig9_layerwise_true_vs_false_option_probability_correct"] = [{"layer": li + 1, "true": mean(tr[li]), "false_mean": mean(fl[li])} for li in range(L)]
    fail = {}
    for target, desc in (("cross-frame", "No cross-frame interactions"), ("vql-to-ql", "Video -/-> Question"), ("vql-to-ql", "Question -/-> Last")):
        fail[f"{target} | {desc}"] = sweep(wrong, target, desc)
    out["failure_cases_wrong_prediction_probability_change"] = fail
    cf_wrong = []
    for r in wrong:
        v = [a["relative_change_percent"] for a in r["arms"] if a["target"] == "cross-frame" and a["kind"] == "sweep"]
        if v:
            cf_wrong.append((min(v), max(v)))
    out["failure_cross_frame_case_split"] = {"n": len(cf_wrong), "case1_wrong_probability_drops_below_minus10": sum(m < -10 for m, _ in cf_wrong),
                                              "case2_wrong_probability_rises_above_plus10": sum(M > 10 for _, M in cf_wrong)}
    out["scope"] = "descriptive reproduction of the paper's analyses on Qwen2.5-VL-7B and TVBench (4 of 5 tasks); not the paper's models"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    print(f"videos {out['videos']}, correct {out['baseline_correct']}, wrong {out['baseline_wrong']}")
    for k, v in out["sweeps_correct_mean_percent_change_by_center"].items():
        print(f"  {k}: " + " ".join(f"{c}:{x:+.1f}" for c, x in v.items()))
    for name in ("table2_first_half_cross_frame", "tableB_second_half_cross_frame"):
        print(name, {t: round(x["accuracy_drop_percent_of_correct"], 1) for t, x in out[name].items() if x["accuracy_drop_percent_of_correct"] is not None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
