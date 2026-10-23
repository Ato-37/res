#!/usr/bin/env python3
"""Final confirmatory analysis of the evidence-focusing plan: pooled Stage 2 + Stage 2b contrast (keepAnn - keepRand), prespecified rule.

Confirmed iff t, sign-flip and Wilcoxon p < 0.05 and mean >= 0.3 nat; otherwise TOST at +/-0.3 or inconclusive. Also reports each cohort
alone, their Welch heterogeneity test and the decomposition versus baseline. Privacy-safe summary only.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from scipy import stats

from analyze_annotation_key_bias_confirmatory import analyze

DELTA_MIN = 0.3


def values(receipt: dict, key: str = "primary_annotation_minus_sham_nats") -> list[float]:
    rows = receipt.get("records") or receipt.get("rows")
    return [float(r[key]) for r in rows]


def decide(res: dict, delta_min: float = DELTA_MIN) -> str:
    if (res["p_two_sided_t"] < 0.05 and res["p_sign_flip"] < 0.05 and res["p_wilcoxon"] < 0.05 and res["mean_nats"] >= delta_min):
        return "confirmed"
    if res["tost_p"] < 0.05:
        return "equivalent_within_delta_min"
    return "not_confirmed_inconclusive"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, nargs="+", required=True, help="completed full receipts (cohorts to pool)")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--permutations", type=int, default=200_000)
    args = ap.parse_args(argv)
    receipts = [json.loads(p.read_text()) for p in args.receipt]
    for r in receipts:
        if r.get("status") != "completed" or r.get("arm_mode") != "keep":
            raise SystemExit("receipt_not_completed_keep_mode")
    media = [set(x["media_sha256"] for x in (r.get("records") or r["rows"])) for r in receipts]
    if sum(len(m) for m in media) != len(set().union(*media)):
        raise SystemExit("cohorts_overlap")
    per = [values(r) for r in receipts]
    pooled = [v for part in per for v in part]
    out = {"schema_version": 1, "purpose": "evidence_focusing_pooled_confirmation", "delta_min_nat": DELTA_MIN,
           "cohort_sizes": [len(p) for p in per], "pooled": analyze(pooled, DELTA_MIN, permutations=args.permutations)}
    out["pooled"]["decision"] = decide(out["pooled"])
    out["per_cohort"] = [dict(analyze(p, DELTA_MIN, permutations=20_000), decision_alone=None) for p in per]
    for o in out["per_cohort"]:
        o["decision_alone"] = decide(o)
    if len(per) == 2:
        out["heterogeneity_welch_p"] = float(stats.ttest_ind(per[0], per[1], equal_var=False).pvalue)
    for key, name in (("annotation_minus_baseline_nats", "keepAnn_minus_baseline"), ("sham_minus_baseline_nats", "keepRand_minus_baseline")):
        v = [x for r in receipts for x in values(r, key)]
        out[name] = {"mean": st.fmean(v), "sd": st.stdev(v), "median": st.median(v)}
    acc = {}
    for c in ("baseline", "annotation_interval_key_bias", "hash_sham_key_bias"):
        acc[c] = sum(bool(x["gold_prediction_correct_by_condition"][c]) for r in receipts for x in (r.get("records") or r["rows"]))
    out["correct_counts"] = acc
    out["scope"] = "oracle evidence-interval attention cropping, answer margin (label-free); not explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    p = out["pooled"]
    print(f"N pooled {len(pooled)} (cohorts {out['cohort_sizes']}): mean {p['mean_nats']:+.3f} [{p['ci95_t'][0]:+.3f},{p['ci95_t'][1]:+.3f}] "
          f"t p={p['p_two_sided_t']:.3g} sign-flip p={p['p_sign_flip']:.3g} Wilcoxon p={p['p_wilcoxon']:.3g} TOST(0.3) p={p['tost_p']:.3g} -> {p['decision']}")
    for i, o in enumerate(out["per_cohort"]):
        print(f"  cohort {i}: n={o['n']} mean {o['mean_nats']:+.3f} t p={o['p_two_sided_t']:.3g} Wilcoxon p={o['p_wilcoxon']:.3g} -> {o['decision_alone']}")
    print("heterogeneity Welch p:", out.get("heterogeneity_welch_p"), "| keepAnn-base %.3f keepRand-base %.3f" % (out["keepAnn_minus_baseline"]["mean"], out["keepRand_minus_baseline"]["mean"]), out["correct_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
