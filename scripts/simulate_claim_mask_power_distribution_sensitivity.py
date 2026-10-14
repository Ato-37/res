"""Monte Carlo stress test of paired-t power under standardized Student-t differences.

Planning sensitivity only: no empirical data are loaded and no SESOI is chosen.
"""
from __future__ import annotations

import argparse
import json
import math
import platform

import numpy as np
import scipy
from scipy.stats import t

from analyze_claim_mask_power_sensitivity import paired_t_lower_ci_power


def simulate_condition(n: int, excess: float, df: float, repetitions: int,
                       seed: int, confidence: float = 0.95,
                       batch_size: int = 5000) -> dict:
    if type(n) is not int or n < 2:
        raise ValueError("n must be an integer >= 2")
    if not isinstance(df, (int, float)) or isinstance(df, bool) or not math.isfinite(df) or df <= 2:
        raise ValueError("df must be finite and > 2 for unit-variance standardization")
    if not isinstance(excess, (int, float)) or isinstance(excess, bool) or not math.isfinite(excess) or excess < 0:
        raise ValueError("excess must be finite and >= 0")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be an integer >= 1")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an integer >= 0")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    critical = t.ppf((1 + confidence) / 2, n - 1)
    upper_count = two_sided_count = completed = 0
    while completed < repetitions:
        count = min(batch_size, repetitions - completed)
        # Student-t(df) has variance df/(df-2); scale to unit variance then
        # shift by the hypothetical mean excess above SESOI.
        samples = rng.standard_t(df=df, size=(count, n)) * math.sqrt((df - 2) / df) + excess
        statistic = samples.mean(axis=1) / (samples.std(axis=1, ddof=1) / math.sqrt(n))
        upper_count += int(np.count_nonzero(statistic > critical))
        two_sided_count += int(np.count_nonzero(np.abs(statistic) > critical))
        completed += count

    upper = upper_count / repetitions
    two_sided = two_sided_count / repetitions
    return {
        "effective_video_n": n,
        "true_mean_minus_sesoi_in_paired_sd": excess,
        "distribution": f"standardized_student_t_df_{df:g}",
        "repetitions": repetitions,
        "seed": seed,
        "upper_tail_probability": upper,
        "upper_tail_monte_carlo_se": math.sqrt(upper * (1 - upper) / repetitions),
        "two_sided_rejection_probability": two_sided,
        "two_sided_monte_carlo_se": math.sqrt(two_sided * (1 - two_sided) / repetitions),
        "normal_noncentral_t_reference": paired_t_lower_ci_power(n, excess, confidence),
        "confidence": confidence,
    }


def build_results(sample_sizes: list[int], excesses: list[float], df: float,
                  repetitions: int, seed: int, confidence: float = 0.95,
                  batch_size: int = 5000) -> dict:
    if not sample_sizes or not excesses:
        raise ValueError("sample_sizes and excesses must be non-empty")
    return {
        "purpose": "distributional stress test of the paired-t CI-lower-bound power rule",
        "empirical_data_used": False,
        "sesoi_selected": False,
        "assumption": "independent video-level paired differences; symmetric Student-t with finite variance, scaled to unit variance",
        "warning": "hypothetical sensitivity only; not evidence that future paired differences follow this distribution",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "rows": [simulate_condition(n, e, df, repetitions, seed + index, confidence, batch_size)
                 for index, (n, e) in enumerate((n, e) for n in sample_sizes for e in excesses)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--excesses", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--df", type=float, default=3.0)
    parser.add_argument("--repetitions", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_results(args.sample_sizes, args.excesses, args.df, args.repetitions,
                           args.seed, args.confidence, args.batch_size)
    with open(args.output, "x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": args.output, "rows": len(result["rows"]),
                      "empirical_data_used": False, "sesoi_selected": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
