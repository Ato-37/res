#!/usr/bin/env python3
"""Analysis of WP1 (experiments/2026-10-23_evidence_pathway_knockout_plan.md): evidence vs sham pathway knockouts.

Contrast per video = margin change under the evidence-key knockout minus under the equal-count sham knockout (nat). P1 = video->text, band
layers 6-20; P2 = video->last, band. Effect rule (P1): t, sign-flip and Wilcoxon p < 0.05 and mean <= -delta_min. Holm over P1/P2 on the t-test.
Positive control: mean %p of the whole-video video->text knockout (band) <= -20% on baseline-correct videos. Writes a privacy-safe summary.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from analyze_annotation_key_bias_confirmatory import analyze
from analyze_annotation_key_bias_dose_response import holm

DELTA_MIN = 0.3
CONTROL_PERCENT = -20.0


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def margin_delta(rec: dict, arm: str) -> float:
    ko = next(k for k in rec["knockouts"] if k["arm"] == arm)
    return ko["margin"] - rec["baseline_margin"]


def contrast_values(records: list[dict], edge_short: str, band: str) -> list[float]:
    return [margin_delta(r, f"{edge_short}_evidence_{band}") - margin_delta(r, f"{edge_short}_sham_{band}")
            for r in records if not r.get("excluded")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--permutations", type=int, default=100_000)
    args = ap.parse_args(argv)
    records = load(args.receipt)
    used = [r for r in records if not r.get("excluded")]
    excluded = len(records) - len(used)
    out = {"schema_version": 1, "purpose": "evidence_pathway_knockout_wp1_summary", "recorded": len(records), "excluded": excluded,
           "analysed": len(used), "baseline_correct": sum(r["baseline_correct"] for r in used), "delta_min_nat": DELTA_MIN, "contrasts": {}}
    p_t = []
    for label, edge, band in (("P1_v2t_band", "v2t", "band"), ("P2_v2l_band", "v2l", "band"),
                              ("S1_v2t_all", "v2t", "all"), ("S2_v2l_all", "v2l", "all")):
        vals = contrast_values(records, edge, band)
        res = analyze(vals, DELTA_MIN, permutations=args.permutations)
        arm_e = [margin_delta(r, f"{edge}_evidence_{band}") for r in used]
        arm_s = [margin_delta(r, f"{edge}_sham_{band}") for r in used]
        res["evidence_minus_baseline_mean"] = st.fmean(arm_e)
        res["sham_minus_baseline_mean"] = st.fmean(arm_s)
        out["contrasts"][label] = res
        if label.startswith("P"):
            p_t.append(res["p_two_sided_t"])
    adj = holm(p_t)
    for label, a in zip(("P1_v2t_band", "P2_v2l_band"), adj):
        out["contrasts"][label]["p_holm"] = a
    p1 = out["contrasts"]["P1_v2t_band"]
    out["P1_effect"] = bool(p1["p_two_sided_t"] < 0.05 and p1["p_sign_flip"] < 0.05 and p1["p_wilcoxon"] < 0.05
                            and p1["mean_nats"] <= -DELTA_MIN and out["contrasts"]["P1_v2t_band"]["p_holm"] < 0.05)
    ctrl = [k["relative_change_percent"] for r in used if r["baseline_correct"] for k in r["knockouts"] if k["arm"] == "v2t_allvideo_band"]
    out["positive_control"] = {"n": len(ctrl), "mean_percent": st.fmean(ctrl) if ctrl else None,
                               "passed": bool(ctrl and st.fmean(ctrl) <= CONTROL_PERCENT)}
    out["scope"] = "pathway-level dependence on the annotated interval (label-free); not explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    print(f"recorded {out['recorded']} (excluded {excluded}), analysed {out['analysed']}, baseline-correct {out['baseline_correct']}")
    pc = out["positive_control"]
    print(f"positive control (whole-video v2t band): mean %p {pc['mean_percent']}, passed={pc['passed']}")
    for label, r in out["contrasts"].items():
        print(f"{label}: mean {r['mean_nats']:+.3f} [{r['ci95_t'][0]:+.3f},{r['ci95_t'][1]:+.3f}] t p={r['p_two_sided_t']:.3g} "
              f"sign-flip p={r['p_sign_flip']:.3g} Wilcoxon p={r['p_wilcoxon']:.3g} TOST(0.3) p={r['tost_p']:.3g} | "
              f"evid-base {r['evidence_minus_baseline_mean']:+.3f} sham-base {r['sham_minus_baseline_mean']:+.3f}")
    print("P1 effect (prespecified rule incl. Holm):", out["P1_effect"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
