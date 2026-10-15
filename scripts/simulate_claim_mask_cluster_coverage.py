"""Synthetic sensitivity audit of the planned video-level Student-t interval.

No project outcomes are read. The simulation keeps the target at the
video-weighted mean and measures nominal 95% interval coverage under a
random-intercept cluster model; it does not select an inferential method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t


TARGET_COVERAGE = 0.95
ICC_GRID = (0.0, 0.05, 0.10, 0.20, 0.40, 0.60)
LAYOUTS = {
    "balanced_5": [16] * 5,
    "balanced_10": [8] * 10,
    "balanced_20": [4] * 20,
    "balanced_40": [2] * 40,
    "balanced_80": [1] * 80,
    "unbalanced_10": [2, 4, 6, 8, 10, 10, 10, 10, 10, 10],
}


def _finite_numeric(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _validate(cluster_sizes, icc, repetitions, seed, batch_size):
    if (not isinstance(cluster_sizes, (list, tuple)) or len(cluster_sizes) < 2
            or any(type(size) is not int or size < 1 for size in cluster_sizes)):
        raise ValueError("at least two clusters with positive integer sizes are required")
    if not _finite_numeric(icc) or not 0 <= icc <= 1:
        raise ValueError("ICC must be finite and in [0, 1]")
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    n = sum(cluster_sizes)
    if n < 2:
        raise ValueError("at least two videos are required")


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    p = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def simulate_condition(cluster_sizes, icc, *, repetitions=50_000, seed=20261015,
                       batch_size=2_500):
    """Estimate coverage of xbar ± t(.975,n−1) s/sqrt(n) for true mean zero."""
    _validate(cluster_sizes, icc, repetitions, seed, batch_size)
    sizes = list(cluster_sizes)
    n = sum(sizes)
    group_index = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    rng = np.random.default_rng(seed)
    critical = float(t.ppf(0.975, n - 1))
    residual_scale = math.sqrt(1 - icc)
    cluster_scale = math.sqrt(icc)

    covered = 0
    sum_mean = sum_mean_sq = sum_estimated_se = sum_width = sum_sample_variance = 0.0
    completed = 0
    while completed < repetitions:
        count = min(batch_size, repetitions - completed)
        random_intercepts = rng.standard_normal((count, len(sizes))) * cluster_scale
        residuals = rng.standard_normal((count, n)) * residual_scale
        values = random_intercepts[:, group_index] + residuals
        means = values.mean(axis=1)
        sample_sd = values.std(axis=1, ddof=1)
        estimated_se = sample_sd / math.sqrt(n)
        half_width = critical * estimated_se
        covered += int(np.count_nonzero(np.abs(means) <= half_width))
        sum_mean += float(means.sum())
        sum_mean_sq += float(np.square(means).sum())
        sum_estimated_se += float(estimated_se.sum())
        sum_width += float((2 * half_width).sum())
        sum_sample_variance += float(np.square(sample_sd).sum())
        completed += count

    coverage = covered / repetitions
    empirical_variance = max(0.0, sum_mean_sq / repetitions - (sum_mean / repetitions) ** 2)
    design_effect = 1 + icc * (n * sum(size * size for size in sizes) / (n * n) - 1)
    analytic_variance = design_effect / n  # marginal video variance is fixed at one
    expected_sample_variance = (n - design_effect) / (n - 1)
    return {
        "cluster_sizes": sizes,
        "cluster_count": len(sizes),
        "video_count": n,
        "icc": float(icc),
        "repetitions": repetitions,
        "seed": seed,
        "batch_size": batch_size,
        "target_coverage": TARGET_COVERAGE,
        "empirical_coverage": coverage,
        "coverage_monte_carlo_se": math.sqrt(coverage * (1 - coverage) / repetitions),
        "coverage_wilson_95_interval": _wilson_interval(covered, repetitions),
        "mean_estimator_bias": sum_mean / repetitions,
        "empirical_sampling_sd": math.sqrt(empirical_variance),
        "analytic_sampling_sd": math.sqrt(analytic_variance),
        "mean_naive_estimated_se": sum_estimated_se / repetitions,
        "mean_ci_width": sum_width / repetitions,
        "analytic_design_effect": design_effect,
        "design_effect_effective_video_count": n / design_effect,
        "analytic_expected_naive_sample_variance": expected_sample_variance,
        "empirical_mean_naive_sample_variance": sum_sample_variance / repetitions,
    }


def simulate_grid(*, repetitions=50_000, seed=20261015, batch_size=2_500,
                  iccs=ICC_GRID, layout_names=tuple(LAYOUTS)):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    if (not isinstance(layout_names, (list, tuple)) or not layout_names
            or any(type(name) is not str or name not in LAYOUTS for name in layout_names)
            or len(set(layout_names)) != len(layout_names)):
        raise ValueError("unknown or empty cluster-layout set")
    if (not isinstance(iccs, (list, tuple)) or not iccs
            or any(not _finite_numeric(value)
                   or not 0 <= value <= 1 for value in iccs)
            or len({float(value) for value in iccs}) != len(iccs)):
        raise ValueError("ICC grid must not be empty")
    conditions = []
    for layout_name in layout_names:
        for icc in iccs:
            sizes = LAYOUTS[layout_name]
            cell_key = json.dumps([seed, layout_name, sizes, float(icc)], separators=(",", ":"))
            cell_seed = int.from_bytes(hashlib.sha256(cell_key.encode("ascii")).digest()[:8], "big") % (2**63)
            conditions.append(simulate_condition(
                sizes, icc, repetitions=repetitions,
                seed=cell_seed, batch_size=batch_size))
            conditions[-1]["layout_name"] = layout_name
    return {
        "schema_version": 1,
        "purpose": "synthetic_video_cluster_dependence_coverage_sensitivity",
        "generator": "balanced or fixed-unequal Gaussian random-intercept model",
        "estimand": "video-weighted population mean of one contrast per video",
        "data_generation": {
            "true_mean": 0.0,
            "marginal_video_variance": 1.0,
            "model": "Y_gj = mean + sqrt(ICC) U_g + sqrt(1-ICC) epsilon_gj",
            "random_effects_and_residuals": "independent standard normal variates",
            "cluster_sizes_are_fixed": True,
            "target_interval": "two-sided 95% one-sample Student-t, df=N-1",
        },
        "grid_config": {
            "layout_names": list(layout_names), "icc_values": [float(value) for value in iccs],
            "base_seed": seed, "repetitions_per_cell": repetitions,
            "batch_size": batch_size, "nominal_confidence": 0.95,
        },
        "software_versions": {"python": platform.python_version(),
                              "numpy": np.__version__, "scipy": scipy.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": conditions,
        "limitations": [
            "Synthetic sensitivity only; ICC and cluster counts are not estimated from project data.",
            "Coverage evaluates the planned video-level t interval under the specified Gaussian random-intercept model only.",
            "No alternative interval is evaluated or selected; no empirical faithfulness score, power, or SESOI is estimated.",
            "Fixed cluster sizes preserve the video-weighted mean target; informative cluster size and non-Gaussian outcomes are outside this grid.",
            "Wilson intervals quantify simulation error, not uncertainty about the real dataset's dependence structure.",
            "The displayed plug-in Monte Carlo SE is zero if the observed coverage is exactly zero or one; inspect the Wilson interval in those boundary cases.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--batch-size", type=int, default=2_500)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = simulate_grid(repetitions=args.repetitions, seed=args.seed,
                           batch_size=args.batch_size)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in result["conditions"]:
        print(f"{row['layout_name']} n={row['video_count']} K={row['cluster_count']} "
              f"ICC={row['icc']:.2f} coverage={row['empirical_coverage']:.4f} "
              f"MCSE={row['coverage_monte_carlo_se']:.4f} "
              f"analytic_SE={row['analytic_sampling_sd']:.4f} "
              f"mean_naive_SE={row['mean_naive_estimated_se']:.4f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
