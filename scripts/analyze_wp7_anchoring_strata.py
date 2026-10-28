#!/usr/bin/env python3
"""Analysis of WP7 (experiments/2026-10-28_wp7_anchoring_strata_plan.md): necessity/sufficiency effect of the cited interval by citation stratum.

Strata: A = cited groups contain group 0 (start-anchored); B1 = not anchored and cited_annotation_overlap > 0; B0 = not anchored and no overlap.
Primary 1: WP2 necessity effect in B1 supported iff t, sign-flip and Wilcoxon p < 0.05 and mean >= 0.3 nat. Primary 2: stratum A effect small iff the upper end of its 95% interval < 0.3.
Primary 3: B1 minus A, Welch t p < 0.05 and difference >= 0.3. Holm over the three t-based p-values. Positive control as in WP2. Secondary: the same strata for WP3 sufficiency.
Privacy-safe summary (no IDs).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from scipy import stats

from analyze_annotation_key_bias_confirmatory import analyze
from analyze_annotation_key_bias_dose_response import holm

DELTA_MIN = 0.3


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def stratum(r: dict) -> str:
    if 0 in r["cited_group_indices"]:
        return "A"
    return "B1" if (r.get("cited_annotation_overlap") or 0) > 0 else "B0"


def nec(r: dict) -> float:
    k = {x["arm"]: x for x in r["knockouts"]}
    return -((k["v2t_cited_band"]["self_margin"] - r["baseline_self_margin"]) - (k["v2t_sham_band"]["self_margin"] - r["baseline_self_margin"]))


def suf(r: dict) -> float:
    k = {x["arm"]: x for x in r["knockouts"]}
    return k["keep_cited"]["gold_margin"] - k["keep_sham"]["gold_margin"]


def describe(v: list[float]) -> dict:
    if len(v) < 2:
        return {"n": len(v), "mean": v[0] if v else None}
    m, sd = st.fmean(v), st.stdev(v)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / len(v) ** 0.5
    return {"n": len(v), "mean": m, "sd": sd, "ci95": [m - h, m + h]}


def decide(res: dict) -> str:
    if (res["p_two_sided_t"] < 0.05 and res["p_sign_flip"] < 0.05 and res["p_wilcoxon"] < 0.05 and res["mean_nats"] >= DELTA_MIN):
        return "supported"
    return "equivalent_within_delta_min" if res["tost_p"] < 0.05 else "inconclusive"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wp2", type=Path, required=True)
    ap.add_argument("--wp3", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--permutations", type=int, default=100_000)
    args = ap.parse_args(argv)
    recs2 = load(args.wp2)
    used = [r for r in recs2 if not r.get("excluded")]
    out = {"schema_version": 1, "purpose": "wp7_anchoring_strata_summary", "recorded": len(recs2), "excluded": len(recs2) - len(used), "analysed": len(used), "delta_min_nat": DELTA_MIN}
    pc = [k["relative_change_percent"] for r in used if r.get("baseline_correct") for k in r["knockouts"] if k["arm"] == "v2t_allvideo_band"]
    out["positive_control_mean_relative_change_percent"] = st.fmean(pc) if pc else None
    out["positive_control_passed"] = bool(pc) and st.fmean(pc) <= -20.0
    out["pooled_necessity"] = analyze([nec(r) for r in used], DELTA_MIN, permutations=args.permutations)
    out["pooled_necessity"]["decision"] = decide(out["pooled_necessity"])
    strata = {s: [r for r in used if stratum(r) == s] for s in ("A", "B1", "B0")}
    out["strata_sizes"] = {s: len(v) for s, v in strata.items()}
    out["share_start_anchored"] = len(strata["A"]) / len(used) if used else None
    b1 = [nec(r) for r in strata["B1"]]
    a = [nec(r) for r in strata["A"]]
    p1 = analyze(b1, DELTA_MIN, permutations=args.permutations) if len(b1) > 3 else None
    out["primary1_B1_necessity"] = p1
    out["primary2_A_necessity"] = describe(a)
    p3 = None
    if len(b1) > 2 and len(a) > 2:
        t = stats.ttest_ind(b1, a, equal_var=False)
        p3 = {"difference": st.fmean(b1) - st.fmean(a), "welch_t_p": float(t.pvalue)}
    out["primary3_B1_minus_A"] = p3
    if p1 and p3 and len(a) > 2:
        hp = holm([p1["p_two_sided_t"], 0.0 if describe(a)["ci95"][1] < DELTA_MIN else 1.0, p3["welch_t_p"]])
        p1["decision"] = decide(p1) if (out["positive_control_passed"] and hp[0] < 0.05) else ("uninformative_positive_control_failed" if not out["positive_control_passed"] else "not_supported_after_holm")
        out["primary2_A_small"] = bool(describe(a)["ci95"][1] < DELTA_MIN)
        p3["supported"] = bool(out["positive_control_passed"] and p3["difference"] >= DELTA_MIN and hp[2] < 0.05)
        out["holm_adjusted_p"] = {"B1_effect": hp[0], "A_small_indicator": hp[1], "B1_minus_A": hp[2]}
    out["necessity_B0"] = describe([nec(r) for r in strata["B0"]])
    if args.wp3:
        by3 = {r["media_sha256"]: r for r in load(args.wp3) if not r.get("excluded")}
        sec = {}
        for s, v in strata.items():
            sec[s] = describe([suf(by3[r["media_sha256"]]) for r in v if r["media_sha256"] in by3])
        out["secondary_sufficiency_by_stratum"] = sec
        allv = [suf(by3[r["media_sha256"]]) for r in used if r["media_sha256"] in by3]
        out["secondary_pooled_sufficiency"] = describe(allv)
    out["scope"] = "dependence of the answer on the models own cited interval by citation stratum; not general explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    print(f"recorded {out['recorded']} (excluded {out['excluded']}), analysed {out['analysed']}, strata {out['strata_sizes']}, positive control passed: {out['positive_control_passed']}")
    pn = out["pooled_necessity"]
    print(f"pooled necessity {pn['mean_nats']:+.3f} [{pn['ci95_t'][0]:+.3f},{pn['ci95_t'][1]:+.3f}] -> {pn['decision']}")
    if p1:
        print(f"B1 necessity {p1['mean_nats']:+.3f} (n={len(b1)}) -> {p1.get('decision')}; A {describe(a).get('mean')} (n={len(a)}); B1-A {p3 and p3['difference']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
