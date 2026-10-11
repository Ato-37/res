"""Describe patching rescue and sensitivity to individual samples (CPU only).

Usage:
    python scripts/analyze_activation_patching.py \
        --input results/activation_patching_confirm_heldout.json \
        --output results/activation_patching_analysis.json

Bootstrap intervals resample whole samples within the displayed subset, are
pointwise (not multiplicity-adjusted), and do not account for selection of the
harmed subset or layers. Outputs are exploratory. Harmed-subset recovery is
sum(rescue)/sum(damage), not a mean of unstable per-sample ratios.
"""

import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean, median, stdev


def quantile(values, fraction):
    position = (len(values) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (position - left) * (values[right] - values[left])


def summarize(ids, values, damage=None, n_boot=10000, seed=0):
    n = len(values)
    if len(ids) != n or (damage is not None and len(damage) != n):
        raise ValueError("Sample/value/damage lengths differ")
    if any(not math.isfinite(x) for x in values):
        raise ValueError("Nonfinite values")
    if damage is not None and any(not math.isfinite(x) or x <= 0 for x in damage):
        raise ValueError("Recovery ratio requires strictly positive damage")
    result = dict(n=n, mean=None, sd=None, median=None, positive_fraction=None,
                  mean_ci95=None, aggregate_recovery=None, recovery_ci95=None,
                  leave_one_out_mean_range=None, largest_abs_sample_id=None,
                  largest_abs_value=None, mean_without_largest_abs=None)
    if not n:
        return result
    result.update(mean=mean(values), sd=stdev(values) if n > 1 else None,
                  median=median(values), positive_fraction=sum(x > 0 for x in values) / n)
    largest = max(range(n), key=lambda i: abs(values[i]))
    result.update(largest_abs_sample_id=ids[largest], largest_abs_value=values[largest])
    if damage is not None:
        result["aggregate_recovery"] = math.fsum(values) / math.fsum(damage)
    if n < 2:
        return result
    total = math.fsum(values)
    leave_one_out = [(total - x) / (n - 1) for x in values]
    result["leave_one_out_mean_range"] = [min(leave_one_out), max(leave_one_out)]
    result["mean_without_largest_abs"] = leave_one_out[largest]
    rng = random.Random(seed)
    means, ratios = [], []
    for _ in range(n_boot):
        draw = [rng.randrange(n) for _ in range(n)]
        numerator = math.fsum(values[i] for i in draw)
        means.append(numerator / n)
        if damage is not None:
            ratios.append(numerator / math.fsum(damage[i] for i in draw))
    means.sort()
    result["mean_ci95"] = [quantile(means, p) for p in (0.025, 0.975)]
    if ratios:
        ratios.sort()
        result["recovery_ci95"] = [quantile(ratios, p) for p in (0.025, 0.975)]
    return result


def analyze(result, n_boot=10000, seed=0):
    if n_boot < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    records = result["records"]
    layers = sorted({int(layer) for layer in result.get("layers", [])}
                    | {int(layer) for r in records for layer in r["layers"]})
    ids = [str(r["sample_id"]) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample_id: cannot establish sample pairing")
    if result.get("n_effective", len(records)) != len(records):
        raise ValueError("n_effective disagrees with records")
    for r in records:
        if not math.isfinite(r["swap_drop"]):
            raise ValueError("Nonfinite swap_drop")
        if any(str(layer) not in r["layers"] for layer in layers):
            raise ValueError("Incomplete layer coverage; refusing silent sample exclusion")
    harmed = [r for r in records if r["swap_drop"] < 0]
    rows = []
    for scope, subset in (("all", records), ("harmed", harmed)):
        for layer in layers:
            values = [r["layers"][str(layer)]["patch_rescue"] for r in subset]
            stats = summarize([str(r["sample_id"]) for r in subset], values,
                              [-r["swap_drop"] for r in subset] if scope == "harmed" else None,
                              n_boot=n_boot, seed=seed)
            rows.append(dict(scope=scope, layer=layer, **stats))
    # A nearest-outside-frame fallback is not an actual annotated evidence frame.
    outside, unknown = [], []
    for r in records:
        if all(k in r for k in ("sampled_indices", "sampled_timestamps", "grounding", "evidence_indices")):
            timestamps = dict(zip(r["sampled_indices"], r["sampled_timestamps"]))
            if any(not any(a <= timestamps[idx] <= b for a, b in r["grounding"])
                   for idx in r["evidence_indices"]):
                outside.append(str(r["sample_id"]))
        else:
            unknown.append(str(r["sample_id"]))
    return dict(schema_version=2, n_requested=result.get("n_requested"),
                n_effective=len(records), n_harmed=len(harmed),
                failures=result.get("failures", []),
                outside_grounding_evidence_ids=outside, coverage_unknown_ids=unknown,
                seed=seed, n_boot=n_boot,
                interpretation="Exploratory; pointwise percentile bootstrap CIs; selected harmed subset; no multiplicity correction.",
                swap_drop=summarize(ids, [r["swap_drop"] for r in records], n_boot=n_boot, seed=seed),
                rows=rows)


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, list):
        return "[" + ",".join(fmt(x) for x in value) + "]"
    return f"{value:+.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="Optional analysis JSON (must differ from input)")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output and Path(args.output).resolve() == Path(args.input).resolve():
        parser.error("--output must not overwrite the input")
    with open(args.input) as file:
        report = analyze(json.load(file), args.n_boot, args.seed)
    report["input"] = args.input
    print(f"input={args.input}")
    print(f"n_requested={report['n_requested']} n_effective={report['n_effective']} "
          f"n_harmed={report['n_harmed']} failures={len(report['failures'])}")
    print(report["interpretation"])
    print(f"swap_drop_mean={fmt(report['swap_drop']['mean'])} "
          f"ci95={fmt(report['swap_drop']['mean_ci95'])}")
    print("scope layer n mean_rescue ci95 median positive_fraction aggregate_recovery recovery_ci95")
    for row in report["rows"]:
        print(f"{row['scope']} {row['layer']} {row['n']} {fmt(row['mean'])} "
              f"{fmt(row['mean_ci95'])} {fmt(row['median'])} {fmt(row['positive_fraction'])} "
              f"{fmt(row['aggregate_recovery'])} {fmt(row['recovery_ci95'])}")
        print(f"  largest_abs_id={row['largest_abs_sample_id']} "
              f"mean_without_it={fmt(row['mean_without_largest_abs'])} "
              f"leave_one_out_range={fmt(row['leave_one_out_mean_range'])}")
    print(f"outside_grounding_evidence_ids={report['outside_grounding_evidence_ids']}")
    print(f"coverage_unknown_ids={report['coverage_unknown_ids']}")
    for failure in report["failures"]:
        print(f"failure={failure}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as file:
            json.dump(report, file, indent=2, allow_nan=False)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
