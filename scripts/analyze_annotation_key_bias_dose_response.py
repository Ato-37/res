#!/usr/bin/env python3
"""Prespecified dose-response analysis (plan: experiments/2026-10-22_annotation_key_bias_dose_response_plan.md).

Per dose: two-sided t-test of annotation-minus-sham (Holm across doses), TOST at +/-delta_min (needs p < alpha/k),
manipulation-check means and changed-choice counts. Privacy-safe summary only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from analyze_annotation_key_bias_confirmatory import ALPHA, analyze


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, nargs="+", required=True, help="one completed full receipt per dose")
    ap.add_argument("--expected-n", type=int, required=True)
    ap.add_argument("--delta-min", type=float, default=0.025)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        raise SystemExit("output_exists")
    per = []
    media_sets = []
    design = set()
    for path in args.receipt:
        d = json.loads(path.read_text())
        rows = d.get("records") or d.get("rows")
        if d.get("status") != "completed" or d.get("effective_videos") != args.expected_n or len(rows) != args.expected_n:
            raise SystemExit("receipt_incomplete")
        media_sets.append([r["media_sha256"] for r in rows])
        design.add((d.get("query_scope", "last"), tuple(d.get("layers_1_based", ())), d.get("precision")))
        res = analyze([float(r["primary_annotation_minus_sham_nats"]) for r in rows], args.delta_min)
        changed = lambda c: sum(r["gold_prediction_correct_by_condition"][c] != r["gold_prediction_correct_by_condition"]["baseline"] for r in rows)
        res.update({"requested_bias_nats": d.get("requested_bias_nats"), "log2_dose": d.get("requested_bias_nats") / math.log(2),
                    "annotation_minus_baseline_mean": st.fmean(r["annotation_minus_baseline_nats"] for r in rows),
                    "sham_minus_baseline_mean": st.fmean(r["sham_minus_baseline_nats"] for r in rows),
                    "correctness_flips_annotation_vs_baseline": changed("annotation_interval_key_bias"),
                    "correctness_flips_sham_vs_baseline": changed("hash_sham_key_bias"),
                    "completed_forwards": d.get("completed_forwards")})
        per.append(res)
    if any(m != media_sets[0] for m in media_sets):
        raise SystemExit("cohort_differs_between_doses")
    if len(design) != 1:
        raise SystemExit("query_scope_layers_or_precision_differ_between_receipts")
    scope, layers, precision = next(iter(design))
    k = len(per)
    adj = holm([r["p_two_sided_t"] for r in per])
    for r, a in zip(per, adj):
        r["p_holm"] = a
        r["significant_after_holm"] = a < ALPHA
        r["equivalent_at_alpha_over_k"] = r["tost_p"] < ALPHA / k
    out = {"schema_version": 1, "status": "prespecified_dose_response", "doses": per, "n_doses": k,
           "query_scope": scope, "layers_1_based": list(layers), "precision": precision,
           "effective_bias_note": "mask dtype rounds the dose (e.g. bf16 -5.545 -> -5.53125); doses are labelled by requested bias",
           "scope": "answer-margin dependence on the QA time annotation (label-free); not explanation faithfulness",
           "faithfulness_status": "補助診断のみ"}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for r in per:
        print(json.dumps({x: r[x] for x in ("requested_bias_nats", "mean_nats", "ci95_t", "p_holm", "tost_p", "verdict")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
