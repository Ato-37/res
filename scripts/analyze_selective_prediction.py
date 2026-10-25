#!/usr/bin/env python3
"""Annotation-free selective prediction on SGC receipts (experiments/2026-10-26_crop_ensemble_prespec.md, updated 2026-10-26).

Answers are withheld in order of lowest confidence. Compared: BASE = baseline answer ordered by its top-option probability; ENS = answer and confidence
from the mean option probabilities of baseline, selfguided4 and random4 (all annotation-free). Primary (SGC test receipt, video = unit):
accuracy at 80% coverage, ENS minus BASE; supported iff the lower end of the paired bootstrap 95% interval (videos resampled, 20,000 draws) is > 0.
Co-primary (annotation-free warning works at all): BASE accuracy at 80% coverage minus BASE accuracy at 100%, same bootstrap rule. Secondary: accuracy at 100/90/70/60/50% coverage, AUROC of the confidence for correctness, area under the risk-coverage curve (lower is better),
errors withheld and correct answers lost at 80% coverage. Privacy-safe summary (no IDs). Not explanation faithfulness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
PRIMARY_COVERAGE = 0.8


def softmax(x):
    x = np.asarray(x, float)
    e = np.exp(x - x.max())
    return e / e.sum()


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def build(records: list[dict]):
    rows = [r for r in records if not r.get("excluded")]
    gold = np.array([r["gold"] for r in rows])
    base = np.array([softmax(r["baseline_choice_logprobs"]) for r in rows])
    s4 = np.array([softmax(r["arms"]["selfguided4"]["choice_logprobs"]) for r in rows])
    r4 = np.array([softmax(r["arms"]["random4"]["choice_logprobs"]) for r in rows])
    return gold, {"base": base, "ens": (base + s4 + r4) / 3.0}


def acc_at_coverage(correct: np.ndarray, conf: np.ndarray, coverage: float) -> float:
    order = np.argsort(-conf, kind="stable")
    k = max(1, int(round(coverage * len(correct))))
    return float(correct[order[:k]].mean())


def aurc(correct: np.ndarray, conf: np.ndarray) -> float:
    order = np.argsort(-conf, kind="stable")
    c = correct[order].astype(float)
    return float(np.mean(1.0 - np.cumsum(c) / np.arange(1, len(c) + 1)))


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def summarize(gold: np.ndarray, probs: dict) -> dict:
    out = {}
    for name, P in probs.items():
        correct = (P.argmax(axis=1) == gold).astype(int)
        conf = P.max(axis=1)
        out[name] = {"accuracy": float(correct.mean()), "auroc": auroc(conf, correct), "aurc": aurc(correct, conf),
                     "accuracy_at_coverage": {str(c): acc_at_coverage(correct, conf, c) for c in COVERAGES}}
    return out


def paired_bootstrap(gold: np.ndarray, probs: dict, coverage: float, draws: int = 20000, seed: int = 20261026) -> dict:
    rng = np.random.default_rng(seed)
    n = len(gold)
    cb = (probs["base"].argmax(axis=1) == gold).astype(int)
    ce = (probs["ens"].argmax(axis=1) == gold).astype(int)
    pb, pe = probs["base"].max(axis=1), probs["ens"].max(axis=1)
    diffs = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        diffs[i] = acc_at_coverage(ce[idx], pe[idx], coverage) - acc_at_coverage(cb[idx], pb[idx], coverage)
    point = acc_at_coverage(ce, pe, coverage) - acc_at_coverage(cb, pb, coverage)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"coverage": coverage, "difference": float(point), "ci95": [float(lo), float(hi)], "supported": bool(lo > 0)}


def warning_gain(gold: np.ndarray, probs: dict, coverage: float, draws: int = 20000, seed: int = 20261027) -> dict:
    """Does the baseline confidence warning work at all? accuracy at the coverage minus accuracy of answering everything (BASE)."""
    rng = np.random.default_rng(seed)
    n = len(gold)
    c = (probs["base"].argmax(axis=1) == gold).astype(int)
    p = probs["base"].max(axis=1)
    diffs = np.array([acc_at_coverage(c[i], p[i], coverage) - c[i].mean() for i in (rng.integers(0, n, n) for _ in range(draws))])
    point = acc_at_coverage(c, p, coverage) - c.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"coverage": coverage, "difference": float(point), "ci95": [float(lo), float(hi)], "supported": bool(lo > 0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val", type=Path, required=True, help="exploratory (hypothesis-generating) receipt")
    ap.add_argument("--test", type=Path, required=True, help="confirmatory receipt")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--draws", type=int, default=20000)
    args = ap.parse_args(argv)
    out = {"schema_version": 1, "purpose": "annotation_free_selective_prediction", "primary_coverage": PRIMARY_COVERAGE}
    for split, path in (("val_exploratory", args.val), ("test_confirmatory", args.test)):
        gold, probs = build(load(path))
        d = {"n": int(len(gold)), "models": summarize(gold, probs), "primary": paired_bootstrap(gold, probs, PRIMARY_COVERAGE, args.draws),
             "warning_works_base": warning_gain(gold, probs, PRIMARY_COVERAGE, args.draws)}
        pred_ok = (probs["base"].argmax(axis=1) == gold)
        order = np.argsort(-probs["base"].max(axis=1), kind="stable")
        k = int(round(PRIMARY_COVERAGE * len(gold)))
        d["baseline_at_80_coverage"] = {"errors_total": int((~pred_ok).sum()), "errors_withheld": int((~pred_ok[order[k:]]).sum()),
                                        "correct_lost": int(pred_ok[order[k:]].sum()), "withheld": int(len(gold) - k)}
        out[split] = d
    out["scope"] = "annotation-free dataset-answer selective prediction; not explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(out, indent=1))
    for split in ("val_exploratory", "test_confirmatory"):
        d = out[split]
        p = d["primary"]
        print(f"{split} n={d['n']}: acc@80% base {d['models']['base']['accuracy_at_coverage']['0.8']:.3f} ens {d['models']['ens']['accuracy_at_coverage']['0.8']:.3f} "
              f"diff {p['difference']:+.3f} [{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}] supported={p['supported']}; "
              f"warning (base acc@80% - base acc) {d['warning_works_base']['difference']:+.3f} [{d['warning_works_base']['ci95'][0]:+.3f},{d['warning_works_base']['ci95'][1]:+.3f}]; "
              f"AUROC base {d['models']['base']['auroc']:.3f} ens {d['models']['ens']['auroc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
