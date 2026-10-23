#!/usr/bin/env python3
"""Analyze a pathway-knockout receipt (or its partial JSONL) per experiments/2026-10-23_pathway_knockout_replication_plan.md.

Per edge x centre: mean relative change of the predicted-answer probability (%p) with 95% t interval over baseline-correct videos;
prediction-change rate; criteria R1-R4 and the outcome label. Writes a privacy-safe summary (no IDs/paths). Descriptive.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from scipy import stats

EDGES = ("cross_frame", "video_to_text", "video_to_last", "text_to_last")
MEANINGFUL_PERCENT = -5.0


def load_records(path: Path) -> tuple[list[dict], dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()], {}
    d = json.loads(path.read_text())
    return d["records"], d


def interval(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    m = st.fmean(values)
    if n < 2:
        return m, float("nan"), float("nan")
    h = stats.t.ppf(0.975, n - 1) * st.stdev(values) / math.sqrt(n)
    return m, m - h, m + h


def profile(records: list[dict]) -> dict:
    correct = [r for r in records if r["baseline_correct"]]
    out = {}
    for edge in EDGES:
        out[edge] = {}
        centers = sorted({k["center"] for r in correct for k in r["knockouts"] if k["edge"] == edge})
        for c in centers:
            ks = [k for r in correct for k in r["knockouts"] if k["edge"] == edge and k["center"] == c]
            vals = [k["relative_change_percent"] for k in ks]
            m, lo, hi = interval(vals)
            flips = sum(k["predicted_after"] != r["predicted"] for r in correct for k in r["knockouts"]
                        if k["edge"] == edge and k["center"] == c)
            out[edge][c] = {"n": len(vals), "mean_percent": float(m), "ci95": [float(lo), float(hi)], "median_percent": float(st.median(vals)),
                            "prediction_change_rate": flips / len(vals), "meaningful": bool(m <= MEANINGFUL_PERCENT and hi < 0)}
    return out


def criteria(prof: dict) -> dict:
    def best(edge):
        items = prof[edge]
        c = min(items, key=lambda k: items[k]["mean_percent"])
        return c, items[c]
    res = {}
    cf = prof["cross_frame"]
    early = [v["mean_percent"] for c, v in cf.items() if c <= 14]
    late = [v["mean_percent"] for c, v in cf.items() if c >= 20]
    res["R1_cross_frame_early_vs_late"] = bool(early and late and min(early) <= st_mean(late) - 10)
    c, v = best("video_to_text")
    res["R2_video_to_text_mid"] = bool(5 <= c <= 20 and v["meaningful"])
    c3, v3 = best("text_to_last")
    res["R3_text_to_last_mid_late"] = bool(c3 >= 14 and v3["meaningful"])
    res["R4_video_to_last_smaller_than_video_to_text"] = bool(
        abs(best("video_to_last")[1]["mean_percent"]) < abs(v["mean_percent"]))
    n_ok = sum(res.values())
    res["criteria_met"] = n_ok
    res["outcome"] = "replicates" if n_ok >= 3 else "partially_replicates" if n_ok == 2 else "does_not_replicate"
    res["strongest_centres"] = {e: best(e)[0] for e in EDGES}
    return res


def st_mean(values):
    return sum(values) / len(values)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True, help="completed receipt .json or partial .jsonl")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    records, meta = load_records(args.receipt)
    prof = profile(records)
    crit = criteria(prof)
    summary = {"schema_version": 1, "purpose": "pathway_knockout_replication_summary", "status": meta.get("status", "partial"),
               "videos_recorded": len(records), "baseline_correct": sum(r["baseline_correct"] for r in records),
               "centres": sorted(next(iter(prof.values())).keys()) if prof.get("cross_frame") else [],
               "profile": prof, "criteria": crit, "scope": "attention-pathway dependence of the predicted answer; not explanation faithfulness"}
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(summary, indent=1))
    print(f"videos {summary['videos_recorded']}, baseline-correct {summary['baseline_correct']}; outcome: {crit['outcome']} ({crit['criteria_met']}/4)")
    print("%-14s" % "mean %p" + " ".join("%8s" % f"l={c}" for c in summary["centres"]))
    for e in EDGES:
        if prof.get(e):
            print("%-14s" % e + " ".join("%7.1f%%" % prof[e][c]["mean_percent"] for c in summary["centres"]))
    print("criteria:", {k: v for k, v in crit.items() if k.startswith("R")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
