#!/usr/bin/env python3
"""Analysis of WP2 (experiments/2026-10-26_wp2_self_cited_evidence_plan.md).

Primary (video = unit, label-free): contrast = self-margin change under the video->text knockout of the model's own cited groups minus the change under an
equal-count disjoint hash sham (self-margin = log-prob of the baseline-predicted option minus the best other option). Effect iff t, sign-flip and Wilcoxon
p < 0.05 and mean contrast <= -0.3 nat (effect = -contrast >= 0.3); else TOST at +/-0.3 nat or inconclusive. Positive control: whole-video video->text
mean relative change <= -20% on baseline-correct videos. Descriptive: citation parse/usable rates, agreement of the cited letter with the forced-choice
prediction, overlap of the cited groups with the annotation versus chance, annotation-knockout minus sham (reference), contrast split by overlap > 0 / = 0.
Privacy-safe summary (no IDs, no generated text).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from analyze_annotation_key_bias_confirmatory import analyze

DELTA_MIN = 0.3


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def arm_change(record: dict, arm: str) -> float:
    k = next(k for k in record["knockouts"] if k["arm"] == arm)
    return k["self_margin"] - record["baseline_self_margin"]


def decide(res: dict) -> str:
    if (res["p_two_sided_t"] < 0.05 and res["p_sign_flip"] < 0.05 and res["p_wilcoxon"] < 0.05 and res["mean_nats"] >= DELTA_MIN):
        return "supported"
    return "equivalent_within_delta_min" if res["tost_p"] < 0.05 else "inconclusive"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--permutations", type=int, default=100_000)
    args = ap.parse_args(argv)
    records = load(args.receipt)
    used = [r for r in records if not r.get("excluded")]
    parsed = [r for r in records if r.get("citation_parse_ok")]
    out = {"schema_version": 1, "purpose": "wp2_self_cited_evidence_summary", "recorded": len(records), "excluded": len(records) - len(used), "analysed": len(used),
           "citation_parse_rate_among_recorded": (len(parsed) / len(records)) if records else None, "delta_min_nat": DELTA_MIN}
    effect = [-(arm_change(r, "v2t_cited_band") - arm_change(r, "v2t_sham_band")) for r in used]
    prim = analyze(effect, DELTA_MIN, permutations=args.permutations)
    prim["decision"] = decide(prim)
    out["primary_effect_cited_vs_sham"] = prim
    ref = analyze([-(arm_change(r, "v2t_annotation_band") - arm_change(r, "v2t_sham_band")) for r in used], DELTA_MIN, permutations=args.permutations)
    ref["decision"] = decide(ref)
    out["reference_effect_annotation_vs_sham"] = ref
    pc = [k["relative_change_percent"] for r in used if r.get("baseline_correct") for k in r["knockouts"] if k["arm"] == "v2t_allvideo_band"]
    out["positive_control_mean_relative_change_percent"] = st.fmean(pc) if pc else None
    out["positive_control_passed"] = bool(pc) and st.fmean(pc) <= -20.0
    ov = [r["cited_annotation_overlap"] for r in used]
    chance = [len(r["cited_group_indices"]) * len(r["annotation_group_indices"]) / r["temporal_group_count"] for r in used] if used and "cited_group_indices" in used[0] else []
    out["grounding"] = {"mean_overlap": st.fmean(ov) if ov else None, "mean_expected_by_chance": st.fmean(chance) if chance else None,
                        "fraction_with_any_overlap": (sum(o > 0 for o in ov) / len(ov)) if ov else None}
    out["citation_letter_matches_prediction_rate"] = (st.fmean(bool(r.get("citation_letter_matches_prediction")) for r in used) if used else None)
    for label, sel in (("overlap_gt0", lambda r: r["cited_annotation_overlap"] > 0), ("overlap_eq0", lambda r: r["cited_annotation_overlap"] == 0)):
        v = [-(arm_change(r, "v2t_cited_band") - arm_change(r, "v2t_sham_band")) for r in used if sel(r)]
        out[f"effect_{label}"] = {"n": len(v), "mean": st.fmean(v) if v else None, "sd": st.stdev(v) if len(v) > 1 else None}
    out["scope"] = "causal dependence of the answer on the models own cited time interval along the video->text pathway; not general explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    p = out["primary_effect_cited_vs_sham"]
    print(f"recorded {out['recorded']} (excluded {out['excluded']}), analysed {out['analysed']}, parse rate {out['citation_parse_rate_among_recorded']:.2f}, positive control passed: {out['positive_control_passed']}")
    print(f"primary effect (cited minus sham, nat drop): {p['mean_nats']:+.3f} [{p['ci95_t'][0]:+.3f},{p['ci95_t'][1]:+.3f}] t p={p['p_two_sided_t']:.3g} TOST p={p['tost_p']:.3g} -> {p['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
