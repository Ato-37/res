#!/usr/bin/env python3
"""Analysis of WP5 (experiments/2026-10-28_wp5_wp6_input_level_and_stability_plan.md).

Primary 1 (necessity, video = unit): effect = -( self-margin change under replace_cited - under replace_sham ). Primary 2 (sufficiency): gold margin under keep_cited minus keep_sham.
Each is supported iff t, sign-flip and Wilcoxon p < 0.05 and mean >= 0.3 nat and its positive control holds (necessity: allstatic mean relative change of the predicted option
probability <= -20% on baseline-correct videos; sufficiency: keep_annotation minus keep_sham mean >= 0.3 nat), with Holm over the two primaries on the t-test; otherwise TOST at
+-0.3 nat or inconclusive. Secondary: per-video Spearman against the attention-level effects (WP2 necessity, WP3 sufficiency receipts, optional), accuracy flips, splits.
Privacy-safe summary (no IDs).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from math import comb
from pathlib import Path

from analyze_annotation_key_bias_confirmatory import analyze
from analyze_annotation_key_bias_dose_response import holm

DELTA_MIN = 0.3
CONTROL_PERCENT = -20.0


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def arm(r: dict, name: str) -> dict:
    return next(a for a in r["arms"] if a["arm"] == name)


def chg(r: dict, name: str) -> float:
    return arm(r, name)["self_margin"] - r["baseline_self_margin"]


def decide(res: dict) -> str:
    if (res["p_two_sided_t"] < 0.05 and res["p_sign_flip"] < 0.05 and res["p_wilcoxon"] < 0.05 and res["mean_nats"] >= DELTA_MIN):
        return "supported"
    return "equivalent_within_delta_min" if res["tost_p"] < 0.05 else "inconclusive"


def mcnemar_exact(up: int, down: int) -> float:
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _rank(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    for p, i in enumerate(o):
        r[i] = float(p)
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    m = (n - 1) / 2
    den = sum((x - m) ** 2 for x in ra)
    return sum((x - m) * (y - m) for x, y in zip(ra, rb)) / den if den else float("nan")


def boot_ci(a, b, reps=2000, seed=0):
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(reps):
        ix = [rng.randrange(n) for _ in range(n)]
        vals.append(spearman([a[i] for i in ix], [b[i] for i in ix]))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--wp2", type=Path, help="WP2 receipt for the cross-method agreement")
    ap.add_argument("--wp3", type=Path, help="WP3 receipt for the cross-method agreement")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--permutations", type=int, default=100_000)
    args = ap.parse_args(argv)
    records = load(args.receipt)
    used = [r for r in records if not r.get("excluded")]
    out = {"schema_version": 1, "purpose": "wp5_input_level_summary", "recorded": len(records), "excluded": len(records) - len(used), "analysed": len(used), "delta_min_nat": DELTA_MIN}
    out["baseline_reproduction_max_abs_logprob_diff_vs_wp2"] = max((r["wp2_baseline_max_abs_logprob_diff"] for r in used), default=None)
    nec = analyze([-(chg(r, "replace_cited") - chg(r, "replace_sham")) for r in used], DELTA_MIN, permutations=args.permutations)
    suf = analyze([arm(r, "keep_cited")["gold_margin"] - arm(r, "keep_sham")["gold_margin"] for r in used], DELTA_MIN, permutations=args.permutations)
    pc_rel = [arm(r, "allstatic")["relative_change_percent"] for r in used if r.get("baseline_correct")]
    out["positive_control_necessity_allstatic_mean_relative_change_percent"] = st.fmean(pc_rel) if pc_rel else None
    pc_nec = bool(pc_rel) and st.fmean(pc_rel) <= CONTROL_PERCENT
    ctrl = analyze([arm(r, "keep_annotation")["gold_margin"] - arm(r, "keep_sham")["gold_margin"] for r in used], DELTA_MIN, permutations=args.permutations)
    out["positive_control_sufficiency_annotation_minus_sham"] = ctrl
    pc_suf = bool(used) and ctrl["mean_nats"] >= DELTA_MIN
    hp = holm([nec["p_two_sided_t"], suf["p_two_sided_t"]])
    nec["holm_p_t"], suf["holm_p_t"] = hp[0], hp[1]
    nec["decision"] = decide(nec) if (pc_nec and hp[0] < 0.05) else ("uninformative_positive_control_failed" if not pc_nec else (decide(nec) if decide(nec) != "supported" else "not_supported_after_holm"))
    suf["decision"] = decide(suf) if (pc_suf and hp[1] < 0.05) else ("uninformative_positive_control_failed" if not pc_suf else (decide(suf) if decide(suf) != "supported" else "not_supported_after_holm"))
    out["primary_necessity_replace_cited_vs_sham"], out["primary_sufficiency_keep_cited_vs_sham"] = nec, suf
    out["positive_control_necessity_passed"], out["positive_control_sufficiency_passed"] = pc_nec, pc_suf
    acc = {"baseline": sum(bool(r["baseline_correct"]) for r in used)}
    for name in ("replace_cited", "replace_sham", "keep_cited", "keep_sham", "replace_annotation", "keep_annotation", "allstatic"):
        corr = [arm(r, name)["predicted_after"] == r["gold"] for r in used]
        up = sum(c and not r["baseline_correct"] for c, r in zip(corr, used))
        down = sum((not c) and r["baseline_correct"] for c, r in zip(corr, used))
        acc[name] = {"correct": sum(corr), "wrong_to_correct": up, "correct_to_wrong": down, "mcnemar_exact_p": mcnemar_exact(up, down)}
    out["accuracy"] = acc
    for label, sel in (("overlap_gt0", lambda r: (r.get("cited_annotation_overlap") or 0) > 0), ("overlap_eq0", lambda r: (r.get("cited_annotation_overlap") or 0) == 0)):
        v = [-(chg(r, "replace_cited") - chg(r, "replace_sham")) for r in used if sel(r)]
        out[f"necessity_effect_{label}"] = {"n": len(v), "mean": st.fmean(v) if v else None}
    if args.wp2 and args.wp3:
        by2 = {r["media_sha256"]: r for r in load(args.wp2) if not r.get("excluded")}
        by3 = {r["media_sha256"]: r for r in load(args.wp3) if not r.get("excluded")}
        ks = [r for r in used if r["media_sha256"] in by2 and r["media_sha256"] in by3]
        def chg2(r, a): return next(k for k in r["knockouts"] if k["arm"] == a)["self_margin"] - r["baseline_self_margin"]
        def gm3(r, a): return next(k for k in r["knockouts"] if k["arm"] == a)["gold_margin"]
        n_in = [-(chg(r, "replace_cited") - chg(r, "replace_sham")) for r in ks]
        n_at = [-(chg2(by2[r["media_sha256"]], "v2t_cited_band") - chg2(by2[r["media_sha256"]], "v2t_sham_band")) for r in ks]
        s_in = [arm(r, "keep_cited")["gold_margin"] - arm(r, "keep_sham")["gold_margin"] for r in ks]
        s_at = [gm3(by3[r["media_sha256"]], "keep_cited") - gm3(by3[r["media_sha256"]], "keep_sham") for r in ks]
        out["cross_method_agreement"] = {"n": len(ks),
            "necessity_spearman": spearman(n_in, n_at) if ks else None, "necessity_ci95": boot_ci(n_in, n_at) if ks else None,
            "sufficiency_spearman": spearman(s_in, s_at) if ks else None, "sufficiency_ci95": boot_ci(s_in, s_at) if ks else None}
    out["scope"] = "input-level (frame replacement) dependence of the answer on the models own cited interval; not general explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    print(f"recorded {out['recorded']} (excluded {out['excluded']}), analysed {out['analysed']}, baseline repro max|dlogp| {out['baseline_reproduction_max_abs_logprob_diff_vs_wp2']}")
    print(f"necessity (replace cited vs sham, self-margin nat): {nec['mean_nats']:+.3f} [{nec['ci95_t'][0]:+.3f},{nec['ci95_t'][1]:+.3f}] t p={nec['p_two_sided_t']:.3g} holm={nec['holm_p_t']:.3g} control {pc_nec} -> {nec['decision']}")
    print(f"sufficiency (keep cited vs sham, gold margin nat): {suf['mean_nats']:+.3f} [{suf['ci95_t'][0]:+.3f},{suf['ci95_t'][1]:+.3f}] t p={suf['p_two_sided_t']:.3g} holm={suf['holm_p_t']:.3g} control {pc_suf} -> {suf['decision']}")
    if "cross_method_agreement" in out:
        c = out["cross_method_agreement"]
        print(f"cross-method Spearman necessity {c['necessity_spearman']:+.3f}, sufficiency {c['sufficiency_spearman']:+.3f} (n={c['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
