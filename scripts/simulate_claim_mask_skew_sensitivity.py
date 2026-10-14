"""Stress-test the claim-mask paired-t rule with skewed synthetic effects.

Planning simulation only: observations are iid centered unit-variance
exponential variates shifted by a hypothetical mean excess over SESOI. No
project outcomes or SESOI are loaded or selected.
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


def _finite_real(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def simulate_condition(n: int, excess: float, repetitions: int, seed: int,
                       confidence: float = 0.95, batch_size: int = 5000) -> dict:
    if type(n) is not int or n < 2:
        raise ValueError("n must be an integer >= 2")
    if not _finite_real(excess) or excess < 0:
        raise ValueError("excess must be finite and >= 0")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be an integer >= 1")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an integer >= 0")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1")
    if not _finite_real(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and strictly between 0 and 1")

    # Fail before simulation if the normal-reference calculation cannot be
    # represented; otherwise extreme finite inputs can emit floating warnings.
    normal_reference = paired_t_lower_ci_power(n, excess, confidence)
    rng = np.random.default_rng(seed)
    critical = t.ppf((1 + confidence) / 2, n - 1)
    reject_count = completed = 0
    while completed < repetitions:
        count = min(batch_size, repetitions - completed)
        # Exp(1) has mean and variance 1; subtracting 1 yields mean 0,
        # variance 1, and skewness 2. `excess` is the hypothetical mean.
        centered = rng.exponential(scale=1.0, size=(count, n)) - 1.0
        statistic = ((centered.mean(axis=1) + excess)
                     / (centered.std(axis=1, ddof=1) / math.sqrt(n)))
        reject_count += int(np.count_nonzero(statistic > critical))
        completed += count

    probability = reject_count / repetitions
    return {
        "effective_video_n": n,
        "true_mean_minus_sesoi_in_paired_sd": excess,
        "distribution": "centered_unit_variance_exponential_skewness_2",
        "repetitions": repetitions,
        "seed": seed,
        "success_probability": probability,
        "monte_carlo_se": math.sqrt(probability * (1 - probability) / repetitions),
        "normal_noncentral_t_reference": normal_reference,
        "confidence": confidence,
    }


def build_results(sample_sizes: list[int], excesses: list[float], repetitions: int,
                  seed: int, confidence: float = 0.95, batch_size: int = 5000) -> dict:
    if not sample_sizes or not excesses:
        raise ValueError("sample_sizes and excesses must be non-empty")
    rows = []
    for index, (n, excess) in enumerate((n, excess) for n in sample_sizes for excess in excesses):
        rows.append(simulate_condition(n, excess, repetitions, seed + index,
                                       confidence, batch_size))
    return {
        "purpose": "paired-t CI-lower-bound sensitivity to a skewed effect distribution",
        "empirical_data_used": False,
        "sesoi_selected": False,
        "assumption": "iid centered unit-variance exponential differences with skewness 2; hypothetical mean excess over SESOI",
        "warning": "stress-test only; no evidence that project video-level differences follow this distribution",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--excesses", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--repetitions", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_results(args.sample_sizes, args.excesses, args.repetitions,
                           args.seed, args.confidence, args.batch_size)
    with open(args.output, "x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": args.output, "rows": len(result["rows"]),
                      "empirical_data_used": False, "sesoi_selected": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
