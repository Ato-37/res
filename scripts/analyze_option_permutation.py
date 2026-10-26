#!/usr/bin/env python3
"""Analysis of the option-order permutation ensemble (experiments/2026-10-26_option_permutation_plan.md).

Per receipt (one cohort): option probabilities (softmax over the 5 option letters) of the original order and 4 cyclic shifts, each mapped back to the original options.
Primary = P5: mean probability over the 5 orders versus the original-order answer, exact two-sided McNemar test; supported iff p < 0.05 and accuracy difference >= +0.02.
Secondary (descriptive): single-shift accuracies, geometric-mean ensemble, 3-order ensemble (original + shifts 1, 2), gated ensemble (use the ensemble only when the
original top probability < 0.6; fixed before any run), predicted-letter distribution of the original order (position bias). Privacy-safe summary (no IDs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_crop_ensemble import compare, load, softmax

GATE = 0.6


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    rows = [r for r in load(args.receipt) if not r.get("excluded")]
    gold = np.array([r["gold"] for r in rows])
    base = np.array([softmax(r["baseline_choice_logprobs"]) for r in rows])
    sh = np.array([[softmax(s["choice_logprobs"]) for s in r["shifts"]] for r in rows])  # N x 4 x 5
    ok = lambda P: P.argmax(axis=1) == gold
    cb = ok(base)
    out = {"schema_version": 1, "purpose": "option_permutation_summary", "analysed": int(len(rows)), "gate": GATE,
           "identity_prompt_match_rate": float(np.mean([bool(r.get("identity_prompt_matches")) for r in rows]))}
    p5 = (base + sh.sum(axis=1)) / 5.0
    out["P5_primary"] = compare(ok(p5), cb)
    out["P3_orig_plus_shifts_1_2"] = compare(ok((base + sh[:, :2].sum(axis=1)) / 3.0), cb)
    out["geometric_mean_5"] = compare(ok(np.exp((np.log(np.clip(base, 1e-12, 1)) + np.log(np.clip(sh, 1e-12, 1)).sum(axis=1)) / 5.0)), cb)
    use = base.max(axis=1) < GATE
    out["gated_P5"] = dict(compare(ok(np.where(use[:, None], p5, base)), cb), fraction_ensembled=float(use.mean()))
    out["single_shift_accuracy"] = [float(ok(sh[:, j]).mean()) for j in range(sh.shape[1])]
    out["original_order_accuracy"] = float(cb.mean())
    out["predicted_letter_distribution_original"] = [float(x) for x in np.bincount(base.argmax(axis=1), minlength=5) / len(rows)]
    out["gold_letter_distribution"] = [float(x) for x in np.bincount(gold, minlength=5) / len(rows)]
    out["scope"] = "annotation-free dataset-answer accuracy; not explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    a = out["P5_primary"]
    print(f"analysed {out['analysed']} (identity prompt match {out['identity_prompt_match_rate']:.2f}): original {a['baseline_accuracy']:.3f}; P5 {a['accuracy']:.3f} "
          f"({a['difference']:+.3f}, +{a['wrong_to_correct']}/-{a['correct_to_wrong']}, McNemar p={a['mcnemar_p']:.3g}) supported={a['supported']}")
    g = out["gated_P5"]
    print(f"gated P5 (ensemble on {g['fraction_ensembled']*100:.0f}%): {g['accuracy']:.3f} ({g['difference']:+.3f}, p={g['mcnemar_p']:.3g}); single shifts {[round(x, 3) for x in out['single_shift_accuracy']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
