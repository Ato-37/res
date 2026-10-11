"""Analyze sample-level soft-ablation deltas from a saved JSON result.

The primary contrast is the paired selective delta:
``correct_delta - control_delta``.  The input JSON is produced by
``soft_ablation_sweep.py``; this script does not load a model or require a GPU.

Usage:
    python scripts/analyze_soft_ablation.py \
        --input results/soft_ablation_positionmatched_heldout.json
"""

import argparse
import json
import math
from collections import defaultdict
from statistics import mean, stdev


# Two-sided 95% t critical values for the sample sizes used in this project.
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
                 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
                 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
                 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
                 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta function."""
    max_iterations = 200
    epsilon = 3.0e-7
    tiny = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = max(abs(d), tiny) * (1.0 if d >= 0 else -1.0)
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = max(abs(d), tiny) * (1.0 if d >= 0 else -1.0)
        c = 1.0 + aa / c
        c = max(abs(c), tiny) * (1.0 if c >= 0 else -1.0)
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = max(abs(d), tiny) * (1.0 if d >= 0 else -1.0)
        c = 1.0 + aa / c
        c = max(abs(c), tiny) * (1.0 if c >= 0 else -1.0)
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) + log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def two_tailed_p(t_value, df):
    if not math.isfinite(t_value):
        return 0.0
    return betai(df / 2.0, 0.5, df / (df + t_value * t_value))


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values in original order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--layer-a", type=int, default=6)
    parser.add_argument("--layer-b", type=int, default=12)
    args = parser.parse_args()

    with open(args.input) as file:
        result = json.load(file)

    groups = defaultdict(list)
    for record in result["records"]:
        selective = record["correct_delta"] - record["control_delta"]
        groups[(record["layer"], record["scale"])].append(selective)

    print(f"input={args.input}")
    rows = []
    print("layer scale n mean_selective sd t p holm_p ci95_low ci95_high positive_frac")
    for (layer, scale), values in sorted(groups.items()):
        n = len(values)
        avg = mean(values)
        sd = stdev(values) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else 0.0
        t_value = avg / se if se > 0 else float("inf")
        p_value = two_tailed_p(t_value, n - 1) if n > 1 else None
        critical = T_CRITICAL_95.get(n - 1)
        margin = critical * se if critical is not None else None
        ci_low = avg - margin if margin is not None else None
        ci_high = avg + margin if margin is not None else None
        positive_frac = sum(value > 0 for value in values) / n
        ci_text = (f"{ci_low:+.6f} {ci_high:+.6f}"
                   if margin is not None else "NA NA")
        rows.append({"layer": layer, "scale": scale, "n": n, "mean": avg,
                     "sd": sd, "t": t_value, "p": p_value,
                     "ci_text": ci_text, "positive_frac": positive_frac})

    test_rows = [row for row in rows if row["scale"] != 1.0 and row["p"] is not None]
    adjusted = holm_adjust([row["p"] for row in test_rows])
    for row, holm_p in zip(test_rows, adjusted):
        row["holm_p"] = holm_p
    for row in rows:
        holm_p = row.get("holm_p", 0.0 if row["scale"] == 1.0 else None)
        holm_text = f"{holm_p:.6f}" if holm_p is not None else "NA"
        print(f"{row['layer']:5d} {row['scale']:5.2f} {row['n']:2d} "
              f"{row['mean']:+.6f} {row['sd']:.6f} {row['t']:+.3f} "
              f"{row['p']:.6f} {holm_text} {row['ci_text']} "
              f"{row['positive_frac']:.3f}")

    by_layer_scale_sample = defaultdict(dict)
    for record in result["records"]:
        selective = record["correct_delta"] - record["control_delta"]
        key = (record["scale"], record["sample_id"])
        by_layer_scale_sample[(record["layer"], record["scale"])][record["sample_id"]] = selective

    contrast_rows = []
    for scale in sorted({record["scale"] for record in result["records"] if record["scale"] != 1.0}):
        values_a = by_layer_scale_sample[(args.layer_a, scale)]
        values_b = by_layer_scale_sample[(args.layer_b, scale)]
        paired = [values_a[sample_id] - values_b[sample_id]
                  for sample_id in values_a.keys() & values_b.keys()]
        if len(paired) < 2:
            continue
        n = len(paired)
        avg = mean(paired)
        sd = stdev(paired)
        se = sd / math.sqrt(n)
        t_value = avg / se if se > 0 else float("inf")
        p_value = two_tailed_p(t_value, n - 1)
        contrast_rows.append((scale, n, avg, sd, t_value, p_value))

    contrast_p = holm_adjust([row[-1] for row in contrast_rows]) if contrast_rows else []
    print(f"\npaired layer contrast: layer {args.layer_a} - layer {args.layer_b}")
    print("scale n mean_diff sd t p holm_p")
    for row, adjusted_p in zip(contrast_rows, contrast_p):
        scale, n, avg, sd, t_value, p_value = row
        print(f"{scale:5.2f} {n:2d} {avg:+.6f} {sd:.6f} {t_value:+.3f} "
              f"{p_value:.6f} {adjusted_p:.6f}")


if __name__ == "__main__":
    main()
