#!/usr/bin/env python3
"""Analysis of WP4 (experiments/2026-10-28_wp4_generated_answer_flip_plan.md).

Flip F = the generated letter under an arm differs from the baseline generated letter (an unparseable generation counts as a flip). Primary (video = unit): paired
difference in flip rate cited minus sham with the exact McNemar test on discordant pairs. Supported iff exact McNemar p < 0.05, difference >= 5 percentage points and
the positive control holds (allvideo flip rate minus sham flip rate >= 20 points); if the positive control fails a null is uninformative. Determinism check: among the
videos with a repeat generation the flip rate of the unmodified repeat must be <= 2%, else the endpoint is unusable. Descriptive: split by baseline correctness,
annotation-minus-sham, accuracy of the knockout generations. Privacy-safe summary (no IDs, no generated text).
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

MIN_DIFF = 0.05
PC_MIN = 0.20
MAX_REPEAT_FLIP = 0.02


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def flip(rec: dict, arm: str) -> bool:
    g = next(x for x in rec["generations"] if x["arm"] == arm)
    return g["letter"] is None or g["letter"] != rec["baseline_letter"]


def contrast(records: list[dict], a: str, b: str) -> dict:
    fa = [flip(r, a) for r in records]
    fb = [flip(r, b) for r in records]
    only_a = sum(x and not y for x, y in zip(fa, fb))
    only_b = sum(y and not x for x, y in zip(fa, fb))
    n = len(records)
    return {"n": n, "rate_a": sum(fa) / n if n else None, "rate_b": sum(fb) / n if n else None,
            "diff": (sum(fa) - sum(fb)) / n if n else None, "only_a": only_a, "only_b": only_b, "mcnemar_exact_p": mcnemar_exact(only_a, only_b)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    records = load(args.receipt)
    used = [r for r in records if not r.get("excluded")]
    out = {"schema_version": 1, "purpose": "wp4_generated_answer_flip_summary", "recorded": len(records), "excluded": len(records) - len(used), "analysed": len(used)}
    rep = [r for r in used if "repeat_letter" in r]
    rep_flip = (sum(r["repeat_letter"] != r["baseline_letter"] for r in rep) / len(rep)) if rep else None
    out["determinism"] = {"n": len(rep), "repeat_flip_rate": rep_flip, "usable": rep_flip is not None and rep_flip <= MAX_REPEAT_FLIP}
    prim = contrast(used, "cited", "sham")
    pc = contrast(used, "allvideo", "sham")
    out["positive_control_allvideo_vs_sham"] = pc
    out["positive_control_passed"] = bool(used) and pc["diff"] is not None and pc["diff"] >= PC_MIN
    if not out["determinism"]["usable"]:
        prim["decision"] = "unusable_endpoint_not_deterministic"
    elif not out["positive_control_passed"]:
        prim["decision"] = "uninformative_positive_control_failed"
    else:
        prim["decision"] = "supported" if (prim["mcnemar_exact_p"] < 0.05 and prim["diff"] >= MIN_DIFF) else "inconclusive_or_not_supported"
    out["primary_cited_vs_sham"] = prim
    out["annotation_vs_sham"] = contrast(used, "annotation", "sham")
    out["split_baseline_correct"] = contrast([r for r in used if r["baseline_correct"]], "cited", "sham")
    out["split_baseline_wrong"] = contrast([r for r in used if not r["baseline_correct"]], "cited", "sham")
    out["scope"] = "flips of the generated answer letter when the models own cited interval is removed from the video->text pathway at prefill; not general explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    d = out["determinism"]
    print(f"recorded {out['recorded']} (excluded {out['excluded']}), analysed {out['analysed']}, determinism repeat-flip {d['repeat_flip_rate']} (n={d['n']}), positive control passed: {out['positive_control_passed']}")
    print(f"primary flip rate cited {prim['rate_a']:.3f} vs sham {prim['rate_b']:.3f}, diff {prim['diff']:+.3f}, discordant {prim['only_a']}/{prim['only_b']}, exact McNemar p={prim['mcnemar_exact_p']:.3g} -> {prim['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
