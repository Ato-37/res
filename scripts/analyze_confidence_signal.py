#!/usr/bin/env python3
"""Annotation-free confidence-signal analysis on SGC receipts (plan addendum in experiments/2026-10-23_self_guided_cropping_plan.md).

Target: baseline correctness. Signals: p_max, option entropy, and evidence-concentration features of the self-guided group scores.
Models (repeated 5-fold CV and val->test): raw p_max, temperature-scaled p_max, logistic on [logit p_max, entropy], and the same plus
concentration features. Metrics: AUROC, Brier, 10-bin ECE; primary = Brier(concentration model) - Brier(logistic on confidence only; same recalibration flexibility). Temperature scaling is reported as a reference, paired bootstrap.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def softmax(v):
    v = np.asarray(v, float)
    e = np.exp(v - v.max())
    return e / e.sum()


def features(rec: dict) -> dict:
    p = softmax(rec["baseline_choice_logprobs"])
    s = np.asarray(rec["group_scores"], float)
    share = s / s.sum()
    ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
    cent = float(-(share * np.log(np.clip(share, 1e-12, 1))).sum() / math.log(len(share)))
    return {"pmax": float(p.max()), "entropy": ent, "conc_entropy": cent, "conc_max": float(share.max())}


def build(records: list[dict]):
    rows = [r for r in records if not r.get("excluded")]
    f = [features(r) for r in rows]
    y = np.array([1.0 if r["baseline_correct"] else 0.0 for r in rows])
    X = {k: np.array([x[k] for x in f]) for k in f[0]}
    return X, y, [np.asarray(r["baseline_choice_logprobs"], float) for r in rows]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(F, y, l2=1e-2):
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Z = np.c_[np.ones(len(F)), (F - mu) / sd]

    def nll(w):
        z = Z @ w
        return float(np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(w[1:] ** 2))

    w = minimize(nll, np.zeros(Z.shape[1]), method="L-BFGS-B").x
    return lambda G: sigmoid(np.c_[np.ones(len(G)), (G - mu) / sd] @ w)


def fit_temperature(lps_train, y_train):
    def loss(t):
        p = np.array([softmax(np.asarray(lp) / t).max() for lp in lps_train])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p)))

    t = minimize_scalar(loss, bounds=(0.2, 20.0), method="bounded").x
    return lambda lps: np.array([softmax(np.asarray(lp) / t).max() for lp in lps])


def auroc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def ece(p, y, bins=10):
    out = 0.0
    for k in range(bins):
        m = (p >= k / bins) & (p < (k + 1) / bins + (1e-9 if k == bins - 1 else 0))
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


MODELS = ("raw_pmax", "temperature", "logistic_conf", "logistic_conf_plus_concentration")


def predict_models(Xtr, ytr, lptr, Xte, lpte):
    out = {"raw_pmax": Xte["pmax"], "temperature": fit_temperature(lptr, ytr)(lpte)}
    c = ["pmax", "entropy"]
    Ftr = np.c_[logit(Xtr["pmax"]), Xtr["entropy"]]
    Fte = np.c_[logit(Xte["pmax"]), Xte["entropy"]]
    out["logistic_conf"] = fit_logistic(Ftr, ytr)(Fte)
    Ftr2 = np.c_[Ftr, Xtr["conc_entropy"], Xtr["conc_max"]]
    Fte2 = np.c_[Fte, Xte["conc_entropy"], Xte["conc_max"]]
    out["logistic_conf_plus_concentration"] = fit_logistic(Ftr2, ytr)(Fte2)
    return out


def sub(X, idx):
    return {k: v[idx] for k, v in X.items()}


def cv_oof(X, y, lps, repeats=20, folds=5, seed=0):
    n = len(y)
    acc = {m: np.zeros(n) for m in MODELS}
    rng = random.Random(seed)
    for _ in range(repeats):
        order = list(range(n))
        rng.shuffle(order)
        for f in range(folds):
            te = np.array(sorted(order[f::folds]))
            tr = np.array(sorted(set(order) - set(te)))
            pred = predict_models(sub(X, tr), y[tr], [lps[i] for i in tr], sub(X, te), [lps[i] for i in te])
            for m in MODELS:
                acc[m][te] += pred[m] / repeats
    return acc


def summarize(preds, y):
    return {m: {"brier": float(np.mean((p - y) ** 2)), "auroc": auroc(np.asarray(p), y), "ece": ece(np.asarray(p), y)} for m, p in preds.items()}


def paired_bootstrap(loss_a, loss_b, reps=4000, seed=1):
    d = np.asarray(loss_a) - np.asarray(loss_b)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
    return float(d.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def load(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    Xv, yv, lv = build(load(args.val))
    res = {"schema_version": 1, "purpose": "confidence_signal_analysis", "n_val": int(len(yv)), "accuracy_val": float(yv.mean())}
    oof = cv_oof(Xv, yv, lv)
    res["val_cv"] = summarize(oof, yv)
    d, ci = paired_bootstrap((oof["logistic_conf_plus_concentration"] - yv) ** 2, (oof["logistic_conf"] - yv) ** 2)
    res["val_primary_brier_diff_concentration_minus_logistic_conf"] = {"mean": d, "ci95": ci, "useful_on_val": bool(ci[1] < 0)}
    if args.test:
        Xt, yt, lt = build(load(args.test))
        res["n_test"] = int(len(yt))
        pred = predict_models(Xv, yv, lv, Xt, lt)
        res["val_to_test"] = summarize(pred, yt)
        d2, ci2 = paired_bootstrap((pred["logistic_conf_plus_concentration"] - yt) ** 2, (pred["logistic_conf"] - yt) ** 2)
        res["test_primary_brier_diff_concentration_minus_logistic_conf"] = {"mean": d2, "ci95": ci2, "useful_on_test": bool(ci2[1] < 0)}
        res["signal_useful"] = bool(res["val_primary_brier_diff_concentration_minus_logistic_conf"]["useful_on_val"] and ci2[1] < 0)
    res["scope"] = "annotation-free confidence signal for baseline correctness; label = dataset answer; not explanation faithfulness"
    if args.output:
        if args.output.exists():
            raise SystemExit("output_exists")
        args.output.write_text(json.dumps(res, indent=1))
    for k in ("val_cv", "val_to_test"):
        if k in res:
            print(k, {m: {kk: round(vv, 3) for kk, vv in v.items()} for m, v in res[k].items()})
    print("primary val:", res["val_primary_brier_diff_concentration_minus_logistic_conf"], "| test:", res.get("test_primary_brier_diff_concentration_minus_logistic_conf"), "| useful:", res.get("signal_useful"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
