"""Conditional power sensitivity for the claim-mask paired-t success rule.

This is planning math only. It uses no dataset outcomes and does not choose a
SESOI. The input is the assumed standardized distance (true mean minus SESOI,
divided by SD of video-level paired differences).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import scipy
from scipy.stats import nct, t


def _finite_real(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def paired_t_lower_ci_power(n: int, excess_over_sesoi_sd: float, confidence: float = 0.95) -> float:
    """Power that a two-sided t CI lower bound exceeds SESOI under normal paired effects."""
    if type(n) is not int or n < 2:
        raise ValueError("n must be an integer >= 2")
    if not _finite_real(excess_over_sesoi_sd) or excess_over_sesoi_sd < 0:
        raise ValueError("excess_over_sesoi_sd must be finite and >= 0")
    if not _finite_real(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    try:
        critical = float(t.ppf((1 + confidence) / 2, n - 1))
        noncentrality = excess_over_sesoi_sd * math.sqrt(n)
        power = float(nct.sf(critical, n - 1, noncentrality))
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise ValueError("noncentral-t power calculation was numerically unstable") from exc
    # SciPy's noncentral-t survival function can return NaN for finite but
    # extreme noncentralities. Never serialize that as a power estimate.
    if not math.isfinite(power) or not 0.0 <= power <= 1.0:
        raise ValueError("noncentral-t power calculation was numerically unstable")
    return power


def minimum_n_for_target_power(excess_over_sesoi_sd: float, target_power: float = 0.8,
                               confidence: float = 0.95, max_n: int = 10000) -> int | None:
    if not _finite_real(target_power) or not 0 < target_power < 1:
        raise ValueError("target_power must be strictly between 0 and 1")
    if type(max_n) is not int or max_n < 2:
        raise ValueError("max_n must be an integer >= 2")
    for n in range(2, max_n + 1):
        if paired_t_lower_ci_power(n, excess_over_sesoi_sd, confidence) >= target_power:
            return n
    return None


def build_grid(sample_sizes: list[int], excesses: list[float], confidence: float = 0.95,
               target_power: float = 0.8) -> dict:
    if not sample_sizes or not excesses:
        raise ValueError("sample_sizes and excesses must be non-empty")
    rows = []
    for n in sample_sizes:
        for excess in excesses:
            rows.append({
                "effective_video_n": n,
                "true_mean_minus_sesoi_in_paired_sd": excess,
                "confidence": confidence,
                "power": paired_t_lower_ci_power(n, excess, confidence),
            })
    return {
        "analysis": "one_sample_t_on_video_level_paired_differences",
        "success_rule": "two_sided_ci_lower_bound_exceeds_sesoi",
        "assumption": "independent normally distributed video-level paired differences with common variance",
        "power_inputs_are_hypothetical": True,
        "sesoi_selected": False,
        "empirical_data_used": False,
        "software": {"python": platform.python_version(), "scipy": scipy.__version__},
        "target_power": target_power,
        "rows": rows,
        "minimum_n_by_hypothetical_excess": [
            {"true_mean_minus_sesoi_in_paired_sd": excess,
             "minimum_effective_video_n": minimum_n_for_target_power(excess, target_power, confidence)}
            for excess in excesses
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[20, 40, 60, 80])
    parser.add_argument("--excesses", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_grid(args.sample_sizes, args.excesses, args.confidence, args.target_power)
    with open(args.output, "x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": args.output, "rows": len(result["rows"]), "sesoi_selected": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
