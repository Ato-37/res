"""Analyze saved temporal necessity-mask results.

The primary contrast is ``evidence_mask_logprob - control_mask_logprob``.
Negative values mean that masking the annotated evidence hurt the answer more
than masking a matched non-evidence window.
"""

import argparse
import json
import math
from statistics import mean, stdev


T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
                 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
                 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
                 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
                 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def betacf(a, b, x):
    tiny = 1.0e-30
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = (1.0 if d >= 0 else -1.0) * max(abs(d), tiny)
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a - 1.0 + m2) * (a + m2))
        d = (1.0 if 1.0 + aa * d >= 0 else -1.0) * max(abs(1.0 + aa * d), tiny)
        c = (1.0 if 1.0 + aa / c >= 0 else -1.0) * max(abs(1.0 + aa / c), tiny)
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2))
        d = (1.0 if 1.0 + aa * d >= 0 else -1.0) * max(abs(1.0 + aa * d), tiny)
        c = (1.0 if 1.0 + aa / c >= 0 else -1.0) * max(abs(1.0 + aa / c), tiny)
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3.0e-7:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        a * math.log(x) + b * math.log1p(-x)
        + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    )
    if x < (a + 1) / (a + b + 2):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1 - x) / b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    result = json.load(open(args.input))
    values = [record["necessity_contrast"] for record in result["records"]]
    n = len(values)
    avg = mean(values)
    sd = stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t_value = avg / se if se else float("inf")
    p_value = betai((n - 1) / 2, 0.5, (n - 1) / (n - 1 + t_value * t_value)) if n > 1 else None
    critical = T_CRITICAL_95.get(n - 1)
    margin = critical * se if critical is not None else None
    print(f"input={args.input}")
    print(f"n_requested={result['n_requested']} n_effective={n} failures={len(result['failures'])}")
    print(f"mean_necessity_contrast={avg:+.6f} sd={sd:.6f} t={t_value:+.3f} p={p_value:.6f}")
    if margin is not None:
        print(f"ci95=[{avg - margin:+.6f}, {avg + margin:+.6f}]")
    print(f"negative_fraction={sum(value < 0 for value in values) / n:.3f}")
    for failure in result["failures"]:
        print(f"failure id={failure['id']} error={failure['error']}")


if __name__ == "__main__":
    main()
