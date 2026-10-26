#!/usr/bin/env python3
"""Analysis of the progressive knockout (experiments/2026-10-26_progressive_knockout_plan.md).

Per (edge, direction, N): mean relative change (%) of the originally predicted option's probability on baseline-correct videos, with a bootstrap 95% interval
(videos resampled). Prespecified descriptive criteria for the video->text edge (band 6-20 from WP0/WP1; any one failing is reported, none is a significance test):
C1 forward N=28 (all layers) mean <= -20% (the edge matters at all); C2 forward N=20 -> N=28 change smaller than 10 points (no extra dependence in layers 21-28);
C3 reverse N=24 (layers 24-28) mean >= -10% (late layers do not carry it); C4 forward N=4 (layers 1-4) mean >= -10% (early layers do not carry it).
Privacy-safe summary (no IDs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def curve(records: list[dict], draws: int = 5000, seed: int = 20261026) -> dict:
    cells: dict[tuple, dict[int, float]] = {}
    correct = [r for r in records if r.get("baseline_correct")]
    for vi, r in enumerate(correct):
        for k in r["knockouts"]:
            cells.setdefault((k["edge"], k["direction"], k["n"]), {})[vi] = k["relative_change_percent"]
    rng = np.random.default_rng(seed)
    out = {}
    for key, per_video in sorted(cells.items()):
        v = np.array(list(per_video.values()), float)
        boots = [rng.choice(v, len(v)).mean() for _ in range(draws)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["|".join(map(str, key))] = {"n": int(len(v)), "mean_percent": float(v.mean()), "ci95": [float(lo), float(hi)]}
    return out


def criteria(c: dict) -> dict:
    g = lambda e, d, n: c.get(f"{e}|{d}|{n}", {}).get("mean_percent")
    f28, f20, r24, f4 = (g("video_to_text", "forward", 28), g("video_to_text", "forward", 20),
                         g("video_to_text", "reverse", 24), g("video_to_text", "forward", 4))
    ok = lambda x: x is not None
    return {
        "C1_forward_all_layers_le_minus20": (f28 <= -20.0) if ok(f28) else None,
        "C2_extra_change_20_to_28_lt_10_points": (abs(f28 - f20) < 10.0) if ok(f28) and ok(f20) else None,
        "C3_reverse_24_ge_minus10": (r24 >= -10.0) if ok(r24) else None,
        "C4_forward_4_ge_minus10": (f4 >= -10.0) if ok(f4) else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    records = load(args.receipt)
    c = curve(records)
    out = {"schema_version": 1, "purpose": "progressive_knockout_summary", "recorded": len(records),
           "baseline_correct": sum(bool(r.get("baseline_correct")) for r in records), "curves": c, "criteria_video_to_text": criteria(c),
           "scope": "attention-pathway dependence of the predicted answer; not explanation faithfulness"}
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    print(f"recorded {out['recorded']}, baseline-correct {out['baseline_correct']}; criteria: {out['criteria_video_to_text']}")
    for key, v in c.items():
        if key.startswith("video_to_text"):
            print(f"  {key}: {v['mean_percent']:+.1f}% [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}] (n={v['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
