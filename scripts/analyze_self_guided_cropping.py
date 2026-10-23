#!/usr/bin/env python3
"""Analysis of SGC (experiments/2026-10-23_self_guided_cropping_plan.md).

Primary pair (Holm over 2): P1 = selfguided - random (matched budget), P2 = selfguided4 - random4. 'Supported' iff t, sign-flip and Wilcoxon
p < 0.05, mean >= 0.3 nat and Holm p < 0.05; else TOST at +/-0.3 nat or inconclusive. Reference contrasts: oracle - random, selfguided - oracle,
arm - baseline; localization of the self-guided groups vs chance. Privacy-safe summary (no IDs).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from analyze_annotation_key_bias_confirmatory import analyze
from analyze_annotation_key_bias_dose_response import holm

DELTA_MIN = 0.3


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def contrast(records: list[dict], a: str, b: str) -> list[float]:
    return [r["arms"][a]["margin"] - r["arms"][b]["margin"] for r in records]


def versus_baseline(records: list[dict], a: str) -> list[float]:
    return [r["arms"][a]["margin"] - r["baseline_margin"] for r in records]


def decide(res: dict, holm_p: float | None = None) -> str:
    ok = (res["p_two_sided_t"] < 0.05 and res["p_sign_flip"] < 0.05 and res["p_wilcoxon"] < 0.05 and res["mean_nats"] >= DELTA_MIN)
    if holm_p is not None:
        ok = ok and holm_p < 0.05
    if ok:
        return "supported"
    if res["tost_p"] < 0.05:
        return "equivalent_within_delta_min"
    return "inconclusive"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--permutations", type=int, default=100_000)
    args = ap.parse_args(argv)
    records = load(args.receipt)
    used = [r for r in records if not r.get("excluded")]
    out = {"schema_version": 1, "purpose": "self_guided_cropping_summary", "recorded": len(records), "excluded": len(records) - len(used),
           "analysed": len(used), "delta_min_nat": DELTA_MIN, "contrasts": {}, "arm_minus_baseline": {}}
    for label, (a, b) in {"P1_selfguided_minus_random": ("selfguided", "random"), "P2_selfguided4_minus_random4": ("selfguided4", "random4"),
                          "ref_oracle_minus_random": ("oracle", "random"), "ref_selfguided_minus_oracle": ("selfguided", "oracle")}.items():
        res = analyze(contrast(used, a, b), DELTA_MIN, permutations=args.permutations)
        out["contrasts"][label] = res
    adj = holm([out["contrasts"]["P1_selfguided_minus_random"]["p_two_sided_t"], out["contrasts"]["P2_selfguided4_minus_random4"]["p_two_sided_t"]])
    for label, p in zip(("P1_selfguided_minus_random", "P2_selfguided4_minus_random4"), adj):
        out["contrasts"][label]["p_holm"] = p
        out["contrasts"][label]["decision"] = decide(out["contrasts"][label], p)
    for label in ("ref_oracle_minus_random", "ref_selfguided_minus_oracle"):
        out["contrasts"][label]["decision"] = decide(out["contrasts"][label])
    for arm in ("oracle", "random", "selfguided", "selfguided4", "random4"):
        v = versus_baseline(used, arm)
        out["arm_minus_baseline"][arm] = {"mean": st.fmean(v), "median": st.median(v), "sd": st.stdev(v)}
    ov = [r["localization"]["selfguided_vs_annotation"] for r in used]
    out["localization"] = {"mean_overlap": st.fmean(o["overlap"] for o in ov), "mean_expected_by_chance": st.fmean(o["expected_by_chance"] for o in ov),
                           "top1_hit_rate": st.fmean(bool(r["localization"]["top1_in_annotation"]) for r in used),
                           "top1_chance_rate": st.fmean(len(r["annotation_group_indices"]) / r["temporal_group_count"] for r in used)}
    o_r = out["contrasts"]["ref_oracle_minus_random"]
    out["oracle_gain_reference_passed"] = bool(o_r["mean_nats"] >= DELTA_MIN)
    out["correct_counts"] = {"baseline": sum(bool(r["baseline_correct"]) for r in used)}
    for arm in ("oracle", "random", "selfguided", "selfguided4", "random4"):
        out["correct_counts"][arm] = sum(r["arms"][arm]["predicted_after"] == r["gold"] for r in used)
    out["scope"] = "answer-margin effect of attention-level cropping (label-free); not explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    print(f"recorded {out['recorded']} (excluded {out['excluded']}), analysed {out['analysed']}; oracle-minus-random >= 0.3: {out['oracle_gain_reference_passed']}")
    for label, r in out["contrasts"].items():
        print(f"{label}: mean {r['mean_nats']:+.3f} [{r['ci95_t'][0]:+.3f},{r['ci95_t'][1]:+.3f}] t p={r['p_two_sided_t']:.3g} Wilcoxon p={r['p_wilcoxon']:.3g} "
              f"TOST p={r['tost_p']:.3g} -> {r['decision']}" + (f" (Holm p={r['p_holm']:.3g})" if "p_holm" in r else ""))
    print("arm - baseline:", {k: round(v["mean"], 3) for k, v in out["arm_minus_baseline"].items()})
    print("localization:", {k: round(v, 3) for k, v in out["localization"].items()}, "correct", out["correct_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
