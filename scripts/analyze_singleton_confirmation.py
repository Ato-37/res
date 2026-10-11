"""Analyze fixed singleton candidates against same-layer head controls.

This is a CPU-only analysis of an independent confirmation run. For each
prespecified candidate it compares per-sample rescue with the mean rescue of
the other heads in that candidate's layer. A sign-flip max-|t| adjustment
controls the five candidate contrasts; it assumes symmetric sample-level nulls.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze_activation_patching import fmt, summarize


CANDIDATES = [(9, 19), (7, 26), (5, 23), (3, 4), (12, 21)]


def analyze(result, n_boot=10000, seed=0):
    records = result.get("records", [])
    layers = {int(layer) for layer in result.get("layers", [])}
    groups = [list(group) for group in result.get("groups", [])]
    keys = {"+".join(map(str, group)): group for group in groups}
    ids = [str(record["sample_id"]) for record in records]
    if n_boot < 100 or len(records) < 2 or len(set(ids)) != len(ids):
        raise ValueError("Need >=2 unique records and >=100 bootstrap draws")
    contrasts = []
    direct = []
    for layer, head in CANDIDATES:
        key = str(head)
        if layer not in layers or key not in keys or len(keys[key]) != 1:
            raise ValueError(f"Missing fixed candidate L{layer}/H{head}")
        peer_keys = [other for other, group in keys.items()
                     if len(group) == 1 and other != key]
        if not peer_keys:
            raise ValueError(f"No same-layer singleton controls for L{layer}")
        candidate_values, control_values = [], []
        for record in records:
            cell = record["layers"].get(str(layer), {})
            if set(cell) != set(keys):
                raise ValueError(f"Incomplete head coverage for sample {record['sample_id']}")
            candidate_values.append(float(cell[key]["patch_rescue"]))
            control_values.append(float(np.mean([cell[peer]["patch_rescue"]
                                                 for peer in peer_keys])))
        contrasts.append(np.asarray(candidate_values) - np.asarray(control_values))
        direct.append(candidate_values)

    matrix = np.column_stack(contrasts)
    means = matrix.mean(axis=0)
    se = matrix.std(axis=0, ddof=1) / math.sqrt(len(records))
    observed = np.divide(means, se, out=np.zeros_like(means), where=se > 0)
    rng = np.random.default_rng(seed)
    max_t = np.empty(n_boot)
    for start in range(0, n_boot, 128):
        stop = min(start + 128, n_boot)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(stop - start, len(records)))
        permuted = signs @ matrix / len(records)
        stats = np.divide(permuted, se, out=np.zeros_like(permuted), where=se > 0)
        max_t[start:stop] = np.max(np.abs(stats), axis=1)
    rows = []
    for i, (layer, head) in enumerate(CANDIDATES):
        contrast_stats = summarize(ids, contrasts[i], n_boot=n_boot, seed=seed)
        direct_stats = summarize(ids, direct[i], n_boot=n_boot, seed=seed)
        p_adjusted = float((1 + np.sum(max_t >= abs(observed[i]))) / (n_boot + 1))
        rows.append({"layer": layer, "head": head,
                     "candidate_rescue": direct_stats,
                     "candidate_minus_control": contrast_stats,
                     "t_stat": float(observed[i]), "p_max_t": p_adjusted,
                     "n_control_heads": len(keys) - 1})
    return {"schema_version": 1, "n_requested": result.get("n_requested"),
            "n_effective": len(records), "n_failures": len(result.get("failures", [])),
            "candidates_fixed_before_confirmation": [
                {"layer": layer, "head": head} for layer, head in CANDIDATES],
            "method": "candidate rescue minus same-layer mean of other singleton heads; "
                      "two-sided paired sample-level sign-flip max-|t| across five contrasts",
            "assumption": "sample-level null contrasts are symmetric around zero",
            "rows": rows, "failures": result.get("failures", [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = analyze(json.loads(Path(args.input).read_text()), args.n_boot, args.seed)
    print(report["method"])
    print(f"n_requested={report['n_requested']} n_effective={report['n_effective']} "
          f"failures={report['n_failures']}")
    print("layer head rescue ci95 candidate_minus_other_heads ci95 p_max_t")
    for row in report["rows"]:
        print(f"{row['layer']:5d} {row['head']:4d} "
              f"{fmt(row['candidate_rescue']['mean'])} "
              f"{fmt(row['candidate_rescue']['mean_ci95'])} "
              f"{fmt(row['candidate_minus_control']['mean'])} "
              f"{fmt(row['candidate_minus_control']['mean_ci95'])} "
              f"{row['p_max_t']:.6f}")
    for failure in report["failures"]:
        print(f"failure={failure}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
