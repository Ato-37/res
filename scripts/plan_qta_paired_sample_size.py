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

from scipy.stats import binomtest, nct, t


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


def exact_sign_p_value(positive: int, negative: int) -> float:
    """Two-sided exact sign-test p-value; zero differences are omitted."""
    if (type(positive) is not int or positive < 0
            or type(negative) is not int or negative < 0):
        raise ValueError("sign_counts_must_be_nonnegative_integers")
    n = positive + negative
    if n < 1:
        raise ValueError("nonzero_sign_count_must_be_positive")
    return float(binomtest(positive, n=n, p=0.5, alternative="two-sided").pvalue)


def exact_sign_test_power(n: int, probability_positive: float,
                          alpha: float = 0.05) -> float:
    """Exact power under an assumed independent per-video sign probability.

    Here n counts nonzero video-level signs and probability_positive is
    conditional on a nonzero effect. This is a planning sensitivity for effect
    direction, not a test or estimator of the primary mean-DeltaF estimand;
    the zero-effect rate is not modeled.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n_must_be_positive_integer")
    probability_positive = _finite_number(probability_positive, "probability_positive")
    alpha = _finite_number(alpha, "alpha")
    if not 0 <= probability_positive <= 1:
        raise ValueError("probability_positive_must_be_between_zero_and_one")
    if not 0 < alpha < 1:
        raise ValueError("alpha_must_be_between_zero_and_one")
    from scipy.stats import binom

    return float(sum(
        binom.pmf(positive, n, probability_positive)
        for positive in range(n + 1)
        if exact_sign_p_value(positive, n - positive) <= alpha
    ))


def required_sign_test_n(probability_positive: float, target_power: float,
                         alpha: float = 0.05,
                         max_n: int = 10000) -> int | None:
    """Smallest nonzero-sign n attaining planning power; scenario only."""
    target_power = _finite_number(target_power, "target_power")
    alpha = _finite_number(alpha, "alpha")
    if not 0 < alpha < target_power < 1:
        raise ValueError("target_power_must_be_greater_than_alpha_and_less_than_one")
    if type(max_n) is not int or max_n < 1:
        raise ValueError("max_n_must_be_positive_integer")
    for n in range(1, max_n + 1):
        if exact_sign_test_power(n, probability_positive, alpha) >= target_power:
            return n
    return None


def sign_test_sensitivity(probabilities: Iterable[float] = (0.7, 0.8, 0.9),
                          powers: Iterable[float] = (0.8, 0.9),
                          alpha: float = 0.05) -> list[dict]:
    """Outcome-blind exact sign-test resolution and power scenarios."""
    try:
        probabilities, powers = tuple(probabilities), tuple(powers)
    except TypeError:
        raise ValueError("scenario_iterables_invalid") from None
    if not probabilities or not powers:
        raise ValueError("scenario_iterables_must_be_nonempty")
    alpha = _finite_number(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha_must_be_between_zero_and_one")
    probabilities = tuple(_finite_number(p, "probability_positive") for p in probabilities)
    powers = tuple(_finite_number(p, "target_power") for p in powers)
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("probability_positive_must_be_between_zero_and_one")
    if any(not alpha < power < 1 for power in powers):
        raise ValueError("target_power_must_be_greater_than_alpha_and_less_than_one")
    return [{
        "assumed_probability_video_deltaF_positive": probability,
        "assumptions_are_scenarios_not_empirical_estimates": True,
        "two_sided_exact_sign_test_alpha": alpha,
        "power_at_n3": exact_sign_test_power(3, probability, alpha),
        "required_n_by_power": {
            str(power): required_sign_test_n(probability, power, alpha)
            for power in powers
        },
        "unit": "independent_source_video_level_deltaF_sign",
        "n_counts": "nonzero_video_level_deltaF_signs",
        "positive_sign_probability_is_conditional_on_nonzero_deltaF": True,
        "zero_effect_rate_modeled": False,
        "scope": "directional_robustness_sensitivity_not_primary_mean_deltaF_test",
    } for probability in probabilities]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltas-nats", type=float, nargs="+",
                        default=(0.10, 0.25, 0.40))
    parser.add_argument("--sd-nats", type=float, nargs="+",
                        default=(0.20, 0.40, 0.60))
    parser.add_argument("--powers", type=float, nargs="+", default=(0.80, 0.90))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--positive-sign-probabilities", type=float, nargs="+",
                        default=(0.7, 0.8, 0.9),
                        help="scenario probabilities for a positive video-level DeltaF sign")
    args = parser.parse_args()
    print(json.dumps({
        "schema_version": 1,
        "purpose": "qta_paired_deltaf_sample_size_sensitivity",
        "method": "two_sided_one_sample_noncentral_t_iid_normal_paired_differences",
        "alpha": args.alpha,
        "assumptions_are_not_sesoi_or_empirical_estimates": True,
        "rows": sensitivity_grid(args.deltas_nats, args.sd_nats, args.powers, args.alpha),
        "two_sided_exact_sign_test_sensitivity": {
            "minimum_attainable_p_at_n3_if_all_nonzero_effects_share_a_sign":
                exact_sign_p_value(3, 0),
            "minimum_sample_size_with_any_rejection_at_alpha": next(
                n for n in range(1, 10001)
                if min(exact_sign_p_value(k, n - k) for k in range(n + 1)) <= args.alpha
            ),
            "scenarios": sign_test_sensitivity(
                args.positive_sign_probabilities, args.powers, args.alpha),
        },
        "limitations": [
            "paired-difference SD and target effects are scenario inputs, not measured",
            "normality and independent video clusters are assumed, not established",
            "source-video clusters may themselves share source identity; higher-level dependence is unaudited",
            "within-video question aggregation is not frozen; current n=3 has one row per selected video",
            "does not account for unusable rows, multiplicity, or model/data selection",
            "the n=3 QTA feasibility cohort cannot confirm efficacy",
            "the exact sign test targets per-video effect direction/median, not the primary mean DeltaF",
            "sign-test n counts nonzero video-level effects; the zero-effect rate is not modeled",
        ],
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
