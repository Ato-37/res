#!/usr/bin/env python3
"""Analysis of the annotation-free random-crop ensemble (experiments/2026-10-26_crop_ensemble_compute_matched_plan.md).

Per receipt (one cohort): ensembles average option probabilities. Primary = A: baseline + the first 2 crops (3 forwards, compute-matched to the earlier
baseline+S4+R4 ensemble) versus the baseline answer, exact two-sided McNemar test; supported iff p < 0.05 and accuracy difference >= +0.02.
Descriptive: B crops 0-2 only (3 forwards), C baseline + all crops, D all crops only, single-crop accuracy, accuracy gain by number of crops.
Privacy-safe summary (no IDs). Not explanation faithfulness.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

MIN_DIFF = 0.02


def softmax(x):
    x = np.asarray(x, float)
    e = np.exp(x - x.max())
    return e / e.sum()


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def mcnemar_exact(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * sum(comb(n, k) for k in range(0, min(better, worse) + 1)) / 2 ** n))


def compare(correct_ens: np.ndarray, correct_base: np.ndarray) -> dict:
    w = int((correct_ens & ~correct_base).sum())
    l = int((~correct_ens & correct_base).sum())
    diff = float(correct_ens.mean() - correct_base.mean())
    p = mcnemar_exact(w, l)
    return {"accuracy": float(correct_ens.mean()), "baseline_accuracy": float(correct_base.mean()), "difference": diff,
            "wrong_to_correct": w, "correct_to_wrong": l, "mcnemar_p": p, "supported": bool(p < 0.05 and diff >= MIN_DIFF)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    rows = [r for r in load(args.receipt) if not r.get("excluded")]
    gold = np.array([r["gold"] for r in rows])
    base = np.array([softmax(r["baseline_choice_logprobs"]) for r in rows])
    crops = np.array([[softmax(c["choice_logprobs"]) for c in r["crops"]] for r in rows])  # N x K x 5
    k = crops.shape[1]
    cb = base.argmax(axis=1) == gold

    def ok(P):
        return P.argmax(axis=1) == gold

    out = {"schema_version": 1, "purpose": "crop_ensemble_summary", "analysed": int(len(rows)), "n_crops": int(k), "min_difference": MIN_DIFF}
    out["A_base_plus_2_crops_primary"] = compare(ok((base + crops[:, :2].sum(axis=1)) / 3.0), cb)
    out["B_3_crops_only"] = compare(ok(crops[:, :3].mean(axis=1)), cb)
    out["C_base_plus_all_crops"] = compare(ok((base + crops.sum(axis=1)) / (1 + k)), cb)
    out["D_all_crops_only"] = compare(ok(crops.mean(axis=1)), cb)
    out["single_crop_accuracy"] = [float(ok(crops[:, j]).mean()) for j in range(k)]
    out["accuracy_by_n_crops_with_baseline"] = {str(m): float(ok((base + crops[:, :m].sum(axis=1)) / (1 + m)).mean()) for m in range(0, k + 1)}
    out["scope"] = "annotation-free dataset-answer accuracy; not explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    a = out["A_base_plus_2_crops_primary"]
    print(f"analysed {out['analysed']}: baseline {a['baseline_accuracy']:.3f}; A (base+2 crops, 3 fwd) {a['accuracy']:.3f} ({a['difference']:+.3f}, +{a['wrong_to_correct']}/-{a['correct_to_wrong']}, McNemar p={a['mcnemar_p']:.3g}) supported={a['supported']}")
    for key in ("B_3_crops_only", "C_base_plus_all_crops", "D_all_crops_only"):
        v = out[key]
        print(f"{key}: {v['accuracy']:.3f} ({v['difference']:+.3f}, p={v['mcnemar_p']:.3g})")
    print("accuracy by number of crops added to baseline:", {m: round(x, 3) for m, x in out["accuracy_by_n_crops_with_baseline"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
