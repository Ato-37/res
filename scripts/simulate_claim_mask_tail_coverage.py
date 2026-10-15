"""Synthetic tail/dependence sensitivity for the planned video-level t interval.

No project outcomes are read. Both cluster and residual components are varied
independently while preserving zero means and unit variances. This is an
assumption sensitivity analysis, not a method-selection or faithfulness test.
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

from simulate_claim_mask_cluster_coverage import (
    ICC_GRID,
    LAYOUTS,
    TARGET_COVERAGE,
    _wilson_interval,
)


DISTRIBUTIONS = ("normal", "student_t5_unit_variance", "student_t3_unit_variance")


def _draw(rng: np.random.Generator, shape: tuple[int, ...], distribution: str) -> np.ndarray:
    if distribution == "normal":
        return rng.standard_normal(shape)
    if distribution == "student_t5_unit_variance":
        return rng.standard_t(5, size=shape) / math.sqrt(5 / 3)
    if distribution == "student_t3_unit_variance":
        return rng.standard_t(3, size=shape) / math.sqrt(3)
    raise ValueError("unknown component distribution")


def simulate_condition(cluster_sizes: list[int], icc: float, cluster_distribution: str,
                       residual_distribution: str, *, repetitions: int, seed: int,
                       batch_size: int) -> dict:
    if (not isinstance(cluster_sizes, list) or len(cluster_sizes) < 2
            or any(type(size) is not int or size < 1 for size in cluster_sizes)):
        raise ValueError("at least two clusters with positive integer sizes are required")
    if type(icc) not in (int, float) or not math.isfinite(icc) or not 0 <= icc <= 1:
        raise ValueError("ICC must be finite and in [0, 1]")
    if cluster_distribution not in DISTRIBUTIONS or residual_distribution not in DISTRIBUTIONS:
        raise ValueError("unknown component distribution")
    if icc == 0 and cluster_distribution != "normal":
        raise ValueError("inactive cluster component must use the canonical normal label")
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")

    sizes = list(cluster_sizes)
    n = sum(sizes)
    group_index = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    rng = np.random.default_rng(seed)
    critical = float(t.ppf(0.975, n - 1))
    cluster_scale, residual_scale = math.sqrt(icc), math.sqrt(1 - icc)
    covered = 0
    sum_mean = sum_mean_sq = sum_se = sum_width = 0.0
    completed = 0
    while completed < repetitions:
        count = min(batch_size, repetitions - completed)
        intercept = _draw(rng, (count, len(sizes)), cluster_distribution) * cluster_scale
        residual = _draw(rng, (count, n), residual_distribution) * residual_scale
        values = intercept[:, group_index] + residual
        means = values.mean(axis=1)
        estimated_se = values.std(axis=1, ddof=1) / math.sqrt(n)
        half_width = critical * estimated_se
        covered += int(np.count_nonzero(np.abs(means) <= half_width))
        sum_mean += float(means.sum())
        sum_mean_sq += float(np.square(means).sum())
        sum_se += float(estimated_se.sum())
        sum_width += float((2 * half_width).sum())
        completed += count

    coverage = covered / repetitions
    empirical_variance = max(0.0, sum_mean_sq / repetitions - (sum_mean / repetitions) ** 2)
    design_effect = 1 + icc * (n * sum(size * size for size in sizes) / n**2 - 1)
    return {
        "cluster_sizes": sizes,
        "cluster_count": len(sizes),
        "video_count": n,
        "icc": float(icc),
        "cluster_component_distribution": cluster_distribution,
        "residual_component_distribution": residual_distribution,
        "repetitions": repetitions,
        "seed": seed,
        "batch_size": batch_size,
        "target_coverage": TARGET_COVERAGE,
        "coverage_count": covered,
        "empirical_coverage": coverage,
        "coverage_monte_carlo_se": math.sqrt(coverage * (1 - coverage) / repetitions),
        "coverage_wilson_95_interval": _wilson_interval(covered, repetitions),
        "mean_estimator_bias": sum_mean / repetitions,
        "empirical_sampling_sd": math.sqrt(empirical_variance),
        "analytic_sampling_sd": math.sqrt(design_effect / n),
        "mean_naive_estimated_se": sum_se / repetitions,
        "mean_ci_width": sum_width / repetitions,
        "analytic_design_effect": design_effect,
    }


def simulate_grid(*, repetitions: int = 20_000, seed: int = 20261015,
                  batch_size: int = 1_000, iccs=ICC_GRID,
                  layout_names=tuple(LAYOUTS)) -> dict:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    if (not isinstance(layout_names, (tuple, list)) or not layout_names
            or any(type(name) is not str or name not in LAYOUTS for name in layout_names)
            or len(set(layout_names)) != len(layout_names)):
        raise ValueError("unknown, duplicate, or empty layout set")
    if (not isinstance(iccs, (tuple, list)) or not iccs
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or not 0 <= value <= 1 for value in iccs)
            or len({float(value) for value in iccs}) != len(iccs)):
        raise ValueError("ICC grid must contain unique finite values in [0,1]")

    conditions = []
    for layout_name in layout_names:
        sizes = LAYOUTS[layout_name]
        for icc in iccs:
            component_pairs = (("normal", residual) for residual in DISTRIBUTIONS) if icc == 0 else (
                (cluster_dist, residual_dist)
                for cluster_dist in DISTRIBUTIONS for residual_dist in DISTRIBUTIONS
            )
            for cluster_dist, residual_dist in component_pairs:
                key = json.dumps([seed, layout_name, sizes, float(icc), cluster_dist, residual_dist],
                                 separators=(",", ":"))
                cell_seed = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:8], "big") % (2**63)
                row = simulate_condition(
                    sizes, icc, cluster_dist, residual_dist,
                    repetitions=repetitions, seed=cell_seed, batch_size=batch_size,
                )
                row["layout_name"] = layout_name
                conditions.append(row)

    return {
        "schema_version": 1,
        "purpose": "synthetic_video_tail_and_cluster_coverage_sensitivity",
        "generator": "fixed-size random-intercept model with independently varied unit-variance components",
        "estimand": "video-weighted population mean of one contrast per video",
        "data_generation": {
            "true_mean": 0.0,
            "marginal_video_variance": 1.0,
            "model": "Y_gj = sqrt(ICC) U_g + sqrt(1-ICC) epsilon_gj",
            "component_distributions": {
                "normal": "N(0,1)",
                "student_t5_unit_variance": "Student-t(5)/sqrt(5/3), variance 1",
                "student_t3_unit_variance": "Student-t(3)/sqrt(3), variance 1, infinite fourth moment",
            },
            "cluster_sizes_fixed": True,
            "target_interval": "two-sided 95% one-sample Student-t, df=N-1",
        },
        "grid_config": {
            "layout_names": list(layout_names), "icc_values": [float(x) for x in iccs],
            "component_distributions": list(DISTRIBUTIONS), "base_seed": seed,
            "repetitions_per_cell": repetitions, "batch_size": batch_size,
            "nominal_confidence": 0.95, "condition_count": len(conditions),
            "total_simulated_datasets": len(conditions) * repetitions,
        },
        "software_versions": {"python": platform.python_version(),
                              "numpy": np.__version__, "scipy": scipy.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": conditions,
        "limitations": [
            "Synthetic sensitivity only; component distributions, ICCs and cluster counts are not estimated from project data.",
            "All cells use fixed cluster sizes and the same video-weighted mean target; informative cluster size is not modeled.",
            "The nominal t interval is the only interval measured; no alternative method is selected.",
            "The t(3) component has finite variance but infinite fourth moment; it is a mathematical stress condition, not a plausibility claim.",
            "Wilson intervals and Monte Carlo SE describe simulation error, not uncertainty about real data or model adequacy.",
            "No p-values, SESOI, power, empirical faithfulness scores, or result-driven scenario exclusions are used.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = simulate_grid(repetitions=args.repetitions, seed=args.seed,
                           batch_size=args.batch_size)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = result["grid_config"]
    print(f"complete: cells={config['condition_count']} "
          f"repetitions_per_cell={config['repetitions_per_cell']} "
          f"datasets={config['total_simulated_datasets']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
