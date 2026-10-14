"""Conditional power sensitivity for the claim-mask paired-t success rule.

This is planning math only. It uses no dataset outcomes and does not choose a
SESOI. The input is the assumed standardized distance (true mean minus SESOI,
divided by SD of video-level paired differences).
"""
from __future__ import annotations

import argparse
import json
import math
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
    critical = t.ppf((1 + confidence) / 2, n - 1)
    noncentrality = excess_over_sesoi_sd * math.sqrt(n)
    return float(nct.sf(critical, n - 1, noncentrality))


def build_grid(sample_sizes: list[int], excesses: list[float], confidence: float = 0.95) -> dict:
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
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[20, 40, 60, 80])
    parser.add_argument("--excesses", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_grid(args.sample_sizes, args.excesses, args.confidence)
    with open(args.output, "x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": args.output, "rows": len(result["rows"]), "sesoi_selected": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
