#!/usr/bin/env python3
"""Analysis of WP3 (experiments/2026-10-27_wp3_cited_sufficiency_plan.md).

Primary (video = unit): contrast = gold-minus-foil margin under keep-only-cited minus keep-only-equal-count-sham (positive = the cited interval carries more
answer-relevant information than a sham). Supported iff t, sign-flip and Wilcoxon p < 0.05 and mean >= 0.3 nat AND the positive control holds
(keep-annotation minus keep-sham mean >= 0.3 nat); if the positive control fails a null is uninformative; else TOST at +/-0.3 nat or inconclusive.
Secondary (descriptive): keep_cited minus the full-video baseline margin; accuracy of keep_cited / keep_sham / keep_annotation vs baseline with exact McNemar.
Privacy-safe summary (no IDs, no generated text).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from math import comb
from pathlib import Path

from analyze_annotation_key_bias_confirmatory import analyze

DELTA_MIN = 0.3


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def arm(record: dict, name: str) -> dict:
    return next(k for k in record["knockouts"] if k["arm"] == name)


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--permutations", type=int, default=100_000)
    args = ap.parse_args(argv)
    records = load(args.receipt)
    used = [r for r in records if not r.get("excluded")]
    out = {"schema_version": 1, "purpose": "wp3_cited_sufficiency_summary", "recorded": len(records), "excluded": len(records) - len(used), "analysed": len(used), "delta_min_nat": DELTA_MIN}
    prim = analyze([arm(r, "keep_cited")["gold_margin"] - arm(r, "keep_sham")["gold_margin"] for r in used], DELTA_MIN, permutations=args.permutations)
    ctrl = analyze([arm(r, "keep_annotation")["gold_margin"] - arm(r, "keep_sham")["gold_margin"] for r in used], DELTA_MIN, permutations=args.permutations)
    ctrl["decision"] = decide(ctrl)
    out["positive_control_annotation_minus_sham"] = ctrl
    out["positive_control_passed"] = bool(used) and ctrl["mean_nats"] >= DELTA_MIN
    prim["decision"] = decide(prim) if out["positive_control_passed"] else "uninformative_positive_control_failed"
    out["primary_cited_minus_sham"] = prim
    out["cited_minus_baseline"] = analyze([arm(r, "keep_cited")["gold_margin"] - r["baseline_gold_margin"] for r in used], DELTA_MIN, permutations=args.permutations)
    acc = {"baseline": sum(bool(r["baseline_correct"]) for r in used)}
    for name in ("keep_cited", "keep_sham", "keep_annotation"):
        corr = [arm(r, name)["predicted_after"] == r["gold"] for r in used]
        up = sum(c and not r["baseline_correct"] for c, r in zip(corr, used))
        down = sum((not c) and r["baseline_correct"] for c, r in zip(corr, used))
        acc[name] = {"correct": sum(corr), "wrong_to_correct": up, "correct_to_wrong": down, "mcnemar_exact_p": mcnemar_exact(up, down)}
    out["accuracy"] = acc
    out["scope"] = "sufficiency-type dependence of the answer on the models own cited interval under an out-of-distribution keep-mode crop; not general explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    p = prim
    print(f"recorded {out['recorded']} (excluded {out['excluded']}), analysed {out['analysed']}, positive control (annotation - sham) {ctrl['mean_nats']:+.3f} passed: {out['positive_control_passed']}")
    print(f"primary (keep cited minus keep sham, gold margin nat): {p['mean_nats']:+.3f} [{p['ci95_t'][0]:+.3f},{p['ci95_t'][1]:+.3f}] t p={p['p_two_sided_t']:.3g} TOST p={p['tost_p']:.3g} -> {p['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
