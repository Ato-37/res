#!/usr/bin/env python3
"""Prespecified confirmatory analysis for the QA-annotation key-bias cohort (plan: experiments/2026-10-22_annotation_key_bias_confirmatory_val160_plan.md).

Primary: two-sided one-sample t-test of the per-video annotation-minus-sham margin contrast (nat); practical threshold
delta_min; TOST equivalence at +/- delta_min; sign-flip permutation as robustness. Writes a privacy-safe summary (no IDs).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

from scipy import stats

ALPHA = 0.05


def analyze(values: list[float], delta_min: float, *, permutations: int = 200_000, seed: int = 20261022) -> dict:
    n = len(values)
    if n < 3:
        raise ValueError("n_too_small")
    mean, sd = st.fmean(values), st.stdev(values)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(1 - ALPHA / 2, n - 1)
    t_stat = mean / se if se > 0 else float("inf") * (1 if mean > 0 else -1 if mean < 0 else 0)
    p_two = float(2 * stats.t.sf(abs(t_stat), n - 1)) if se > 0 else (1.0 if mean == 0 else 0.0)
    # TOST: H0a mean <= -delta, H0b mean >= +delta
    p_lower = float(stats.t.sf((mean + delta_min) / se, n - 1)) if se > 0 else float(mean + delta_min <= 0)
    p_upper = float(stats.t.cdf((mean - delta_min) / se, n - 1)) if se > 0 else float(mean - delta_min >= 0)
    p_tost = max(p_lower, p_upper)
    ci90 = (mean - stats.t.ppf(1 - ALPHA, n - 1) * se, mean + stats.t.ppf(1 - ALPHA, n - 1) * se)
    rng = random.Random(seed)
    abs_vals = [abs(v) for v in values]
    extreme = 0
    obs = abs(sum(values))
    for _ in range(permutations):
        s = sum(a if rng.random() < 0.5 else -a for a in abs_vals)
        if abs(s) >= obs - 1e-12:
            extreme += 1
    p_perm = (extreme + 1) / (permutations + 1)
    try:
        p_wilcoxon = float(stats.wilcoxon(values).pvalue)
    except ValueError:  # all differences zero
        p_wilcoxon = 1.0
    significant = p_two < ALPHA
    meaningful = significant and abs(mean) >= delta_min
    equivalent = p_tost < ALPHA
    verdict = ("practically_meaningful_effect" if meaningful else
               "statistically_significant_below_threshold" if significant else
               "no_meaningful_effect_equivalence" if equivalent else "inconclusive")
    return {"n": n, "mean_nats": mean, "sd_nats": sd, "median_nats": st.median(values),
            "ci95_t": [mean - tcrit * se, mean + tcrit * se], "ci90_t": list(ci90),
            "p_two_sided_t": p_two, "p_sign_flip": p_perm, "p_wilcoxon": p_wilcoxon, "permutations": permutations,
            "delta_min_nats": delta_min, "tost_p": p_tost, "tost_p_lower": p_lower, "tost_p_upper": p_upper,
            "positive": sum(v > 1e-9 for v in values), "negative": sum(v < -1e-9 for v in values),
            "verdict": verdict}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--delta-min", type=float, default=0.025)
    ap.add_argument("--expected-n", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        raise SystemExit("output_exists")
    d = json.loads(args.receipt.read_text())
    if d.get("status") != "completed":
        raise SystemExit("receipt_not_completed")
    rows = d.get("records") or d.get("rows")
    if d.get("effective_videos") != args.expected_n or len(rows) != args.expected_n:
        raise SystemExit("effective_n_mismatch")
    if len({r["media_sha256"] for r in rows}) != len(rows):
        raise SystemExit("duplicate_media")
    prim = [float(r["primary_annotation_minus_sham_nats"]) for r in rows]
    summary = analyze(prim, args.delta_min)
    sec = {}
    for key in ("annotation_minus_baseline_nats", "sham_minus_baseline_nats"):
        v = [float(r[key]) for r in rows]
        sec[key] = {"mean": st.fmean(v), "sd": st.stdev(v), "median": st.median(v)}
    acc = {}
    first = rows[0]["gold_prediction_correct_by_condition"]
    for c in (first if isinstance(first, dict) else {}):
        acc[c] = sum(bool(r["gold_prediction_correct_by_condition"][c]) for r in rows)
    out = {"schema_version": 1, "status": "confirmatory_prespecified_analysis", "dataset_split": d.get("dataset_split"),
           "requested_videos": d.get("requested_videos"), "effective_videos": len(rows),
           "completed_forwards": d.get("completed_forwards"), "readout": "fp32_lm_head_matmul_of_bf16_final_hidden_state",
           "primary": summary, "secondary_descriptive_no_inference": sec, "correct_counts_by_condition": acc,
           "scope": "dependence of the five-choice answer margin on the QA time annotation (label-free); not explanation faithfulness",
           "faithfulness_status": "補助診断のみ"}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("n", "mean_nats", "ci95_t", "p_two_sided_t", "tost_p", "verdict")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
