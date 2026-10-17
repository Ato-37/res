#!/usr/bin/env python3
"""Outcome-blind paired-video sample-size sensitivity for the QTA DeltaF.

This is a planning calculation, not an empirical result or preregistered
design. It assumes one independent video-cluster-level paired DeltaF per
sampling unit and a two-sided one-sample t test at the supplied alpha. Multiple
questions from one source video must not be treated as independent rows.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Iterable

from scipy.stats import nct, t


def _finite_number(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}_must_be_finite_number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field}_must_be_finite_number")
    return result


def paired_t_power(n: int, delta_nats: float, sd_nats: float,
                   alpha: float = 0.05) -> float:
    """Exact noncentral-t power under iid Normal paired differences."""
    if type(n) is not int or n < 2:
        raise ValueError("n_must_be_integer_at_least_2")
    delta_nats = _finite_number(delta_nats, "delta")
    sd_nats = _finite_number(sd_nats, "sd")
    alpha = _finite_number(alpha, "alpha")
    if sd_nats <= 0:
        raise ValueError("sd_must_be_positive_finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha_must_be_between_zero_and_one")
    critical = float(t.ppf(1.0 - alpha / 2.0, n - 1))
    ncp = delta_nats * math.sqrt(n) / sd_nats
    return float(nct.sf(critical, n - 1, ncp)
                 + nct.cdf(-critical, n - 1, ncp))


def required_n(delta_nats: float, sd_nats: float, target_power: float,
               alpha: float = 0.05, max_n: int = 10000) -> int | None:
    """Smallest n meeting target power; None if max_n is insufficient."""
    target_power = _finite_number(target_power, "target_power")
    alpha = _finite_number(alpha, "alpha")
    if not 0 < alpha < target_power < 1:
        raise ValueError("target_power_must_be_greater_than_alpha_and_less_than_one")
    if type(max_n) is not int or max_n < 2:
        raise ValueError("max_n_must_be_integer_at_least_2")
    for n in range(2, max_n + 1):
        if paired_t_power(n, delta_nats, sd_nats, alpha) >= target_power:
            return n
    return None


def sensitivity_grid(deltas: Iterable[float], sds: Iterable[float],
                     powers: Iterable[float] = (0.8, 0.9),
                     alpha: float = 0.05) -> list[dict]:
    """Return fully labeled assumptions and exact noncentral-t calculations."""
    try:
        deltas, sds, powers = tuple(deltas), tuple(sds), tuple(powers)
    except TypeError:
        raise ValueError("scenario_iterables_invalid") from None
    if not deltas or not sds or not powers:
        raise ValueError("scenario_iterables_must_be_nonempty")
    alpha = _finite_number(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha_must_be_between_zero_and_one")
    deltas = tuple(_finite_number(value, "delta") for value in deltas)
    sds = tuple(_finite_number(value, "sd") for value in sds)
    powers = tuple(_finite_number(value, "target_power") for value in powers)
    if any(value <= 0 for value in sds):
        raise ValueError("sd_must_be_positive_finite")
    if any(not alpha < value < 1 for value in powers):
        raise ValueError("target_power_must_be_greater_than_alpha_and_less_than_one")
    rows = []
    for delta in deltas:
        for sd in sds:
            rows.append({
                "assumed_mean_delta_nats": float(delta),
                "assumed_sd_paired_delta_nats": float(sd),
                "assumed_standardized_effect_dz": float(delta) / float(sd),
                "two_sided_alpha": float(alpha),
                "power_at_n3": paired_t_power(3, float(delta), float(sd), alpha),
                "required_n_by_power": {
                    str(float(power)): required_n(float(delta), float(sd), float(power), alpha)
                    for power in powers
                },
                "unit": "independent_source_video_cluster_mean_of_paired_deltaF",
                "cluster_aggregation": "one prespecified mean per source video; target cohort rule remains unfrozen",
                "scope": "assumption_sensitivity_not_empirical_or_preregistered",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltas-nats", type=float, nargs="+",
                        default=(0.10, 0.25, 0.40))
    parser.add_argument("--sd-nats", type=float, nargs="+",
                        default=(0.20, 0.40, 0.60))
    parser.add_argument("--powers", type=float, nargs="+", default=(0.80, 0.90))
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps({
        "schema_version": 1,
        "purpose": "qta_paired_deltaf_sample_size_sensitivity",
        "method": "two_sided_one_sample_noncentral_t_iid_normal_paired_differences",
        "alpha": args.alpha,
        "assumptions_are_not_sesoi_or_empirical_estimates": True,
        "rows": sensitivity_grid(args.deltas_nats, args.sd_nats, args.powers, args.alpha),
        "limitations": [
            "paired-difference SD and target effects are scenario inputs, not measured",
            "normality and independent video clusters are assumed, not established",
            "source-video clusters may themselves share source identity; higher-level dependence is unaudited",
            "within-video question aggregation is not frozen; current n=3 has one row per selected video",
            "does not account for unusable rows, multiplicity, or model/data selection",
            "the n=3 QTA feasibility cohort cannot confirm efficacy",
        ],
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
