"""Validate and summarize the fixed scalar-only Cycle16 BNB result.

This CPU-only analyzer checks the manifest and comparison plan, verifies the
reported zero-difference controls, and emits compact per-condition M1-to-long
row maxima. The source artifact contains no output vectors, so the analyzer
cannot recompute the original tensor metrics from arrays.

Usage::

    .venv/bin/python scripts/analyze_bnb_row_count_gpu_diagnostic.py \
      results/bnb_row_count_gpu_diagnostic_20261012_01.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


CONDITIONS = (
    "original_bf16_f_linear",
    "original_fp32_f_linear",
    "bnb_fp4_bf16_compute",
    "bnb_dequantized_bf16_f_linear",
)
ROW_COUNTS = (1, 584, 585, 586)
EXPECTED_CALLS = 20
EXPECTED_COMPARISONS = 32
ZERO_METRICS = ("raw_max_abs", "raw_mean_abs", "raw_rms", "relative_l2_to_left")


def _expected_plan() -> Counter:
    plan = Counter()
    for condition in CONDITIONS:
        for index, left_m in enumerate(ROW_COUNTS):
            for right_m in ROW_COUNTS[index + 1:]:
                plan[("row_count_within_condition", condition, left_m, right_m)] += 1
        plan[("same_shape_repeat", condition, 584, 584)] += 1
    for count in ROW_COUNTS:
        plan[("bnb_vs_dequantized_same_quantized_weight",
              "bnb_fp4_bf16_compute_vs_bnb_dequantized_bf16_f_linear",
              count, count)] += 1
    return plan


def _observed_plan(comparisons: list[dict]) -> Counter:
    plan = Counter()
    for item in comparisons:
        kind = item.get("comparison_type")
        condition = item.get("condition")
        if kind == "row_count_within_condition":
            key = (kind, condition, item.get("left_m"), item.get("right_m"))
        elif kind == "same_shape_repeat":
            key = (kind, condition, item.get("m"), item.get("m"))
        elif kind == "bnb_vs_dequantized_same_quantized_weight":
            key = (kind, condition, item.get("m"), item.get("m"))
        else:
            raise ValueError("unexpected_comparison_type")
        plan[key] += 1
    return plan


def _metric_item(item: dict) -> None:
    for key in ("raw_max_abs", "raw_mean_abs", "raw_rms", "torch64_numpy_max_abs_error"):
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("comparison_metric_missing_or_nonfinite")
    relative = item.get("relative_l2_to_left")
    relative_status = item.get("relative_l2_status")
    if relative_status == "defined":
        if isinstance(relative, bool) or not isinstance(relative, (int, float)) or not math.isfinite(relative):
            raise ValueError("relative_l2_status_value_mismatch")
    elif relative_status in ("zero_reference_equal", "zero_reference_different"):
        if relative is not None:
            raise ValueError("relative_l2_status_value_mismatch")
    else:
        raise ValueError("relative_l2_status_missing_or_unexpected")
    if item.get("numeric_crosscheck_within_1e_10") is not True:
        raise ValueError("numeric_crosscheck_failed")


def summarize_record(record: dict, artifact_sha256: str) -> dict:
    """Validate a completed artifact and return compact, provenance-bound scalars."""
    if record.get("schema_version") != 1 or record.get("status") != "complete":
        raise ValueError("artifact_not_complete_supported_schema")
    if record.get("failure_count") != 0 or record.get("failure_code") is not None:
        raise ValueError("artifact_reports_failures")
    progress = record.get("progress")
    if not isinstance(progress, dict) or progress.get("attempted_forward_calls") != EXPECTED_CALLS \
            or progress.get("completed_forward_calls") != EXPECTED_CALLS \
            or progress.get("completed_comparisons") != EXPECTED_COMPARISONS:
        raise ValueError("fixed_progress_count_mismatch")
    comparisons = record.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != EXPECTED_COMPARISONS \
            or record.get("comparison_count") != EXPECTED_COMPARISONS:
        raise ValueError("fixed_comparison_count_mismatch")
    if record.get("numeric_crosscheck_all_32_passed") is not True:
        raise ValueError("artifact_crosscheck_summary_failed")
    if _observed_plan(comparisons) != _expected_plan():
        raise ValueError("fixed_comparison_plan_mismatch")

    indexed = {}
    for item in comparisons:
        _metric_item(item)
        kind, condition = item["comparison_type"], item["condition"]
        if kind == "row_count_within_condition":
            key = (kind, condition, item["left_m"], item["right_m"])
        elif kind == "same_shape_repeat":
            key = (kind, condition, item["m"], item["m"])
        else:
            key = (kind, condition, item["m"], item["m"])
        indexed[key] = item

    conditions = {}
    for condition in CONDITIONS:
        m1_to_long = {}
        for count in ROW_COUNTS[1:]:
            item = indexed[("row_count_within_condition", condition, 1, count)]
            m1_to_long[str(count)] = item["raw_max_abs"]
        long_pairs = ((584, 585), (584, 586), (585, 586))
        long_row_differences = {}
        for left_m, right_m in long_pairs:
            item = indexed[("row_count_within_condition", condition, left_m, right_m)]
            if any(item[key] != 0.0 for key in ZERO_METRICS):
                raise ValueError("long_row_difference_not_exact_zero")
            long_row_differences[f"{left_m}_vs_{right_m}"] = item["raw_max_abs"]
        repeat = indexed[("same_shape_repeat", condition, 584, 584)]
        if any(repeat[key] != 0.0 for key in ZERO_METRICS):
            raise ValueError("same_shape_repeat_not_exact_zero")
        conditions[condition] = {
            "m1_to_long_raw_max_abs_by_m": m1_to_long,
            "m1_to_long_max_raw_abs": max(m1_to_long.values()),
            "long_row_pair_raw_max_abs": long_row_differences,
            "m584_repeat_raw_max_abs": repeat["raw_max_abs"],
        }

    return {
        "analysis": "fixed_bnb_row_count_scalar_artifact_validation",
        "artifact_sha256": artifact_sha256,
        "artifact_status": record["status"],
        "source_commit": record.get("source_commit"),
        "runner_sha256": record.get("runner_sha256"),
        "verified_forward_calls": EXPECTED_CALLS,
        "verified_comparisons": EXPECTED_COMPARISONS,
        "all_numeric_crosschecks_passed": True,
        "all_long_row_pairs_and_m584_repeats_exact_zero": True,
        "conditions": conditions,
        "array_recalculation": "not_possible_outputs_not_in_artifact",
    }


def analyze_path(path: Path) -> dict:
    raw = path.read_bytes()
    record = json.loads(raw)
    return summarize_record(record, hashlib.sha256(raw).hexdigest())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="existing scalar-only Cycle16 JSON")
    args = parser.parse_args(argv)
    print(json.dumps(analyze_path(args.artifact), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
