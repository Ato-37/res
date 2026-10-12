"""CPU contract tests for the Cycle16 scalar artifact analyzer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import analyze_bnb_row_count_gpu_diagnostic as analyzer


def comparison(kind, condition, m=None, left=None, right=None, maximum=0.0):
    item = {
        "comparison_type": kind,
        "condition": condition,
        "raw_max_abs": maximum,
        "raw_mean_abs": maximum / 2,
        "raw_rms": maximum / 3,
        "relative_l2_to_left": maximum,
        "relative_l2_status": "defined",
        "torch64_numpy_max_abs_error": 0.0,
        "numeric_crosscheck_within_1e_10": True,
    }
    if left is not None:
        item.update(left_m=left, right_m=right)
    if m is not None:
        item["m"] = m
    return item


def fixture_record():
    comparisons = []
    for index, condition in enumerate(analyzer.CONDITIONS):
        for left_i, left in enumerate(analyzer.ROW_COUNTS):
            for right in analyzer.ROW_COUNTS[left_i + 1:]:
                maximum = (index + 1) * 0.001 if left == 1 else 0.0
                comparisons.append(comparison("row_count_within_condition", condition,
                                              left=left, right=right, maximum=maximum))
        comparisons.append(comparison("same_shape_repeat", condition, m=584))
    bnb_dequant = "bnb_fp4_bf16_compute_vs_bnb_dequantized_bf16_f_linear"
    for count in analyzer.ROW_COUNTS:
        comparisons.append(comparison("bnb_vs_dequantized_same_quantized_weight", bnb_dequant, m=count))
    return {
        "schema_version": 1,
        "status": "complete",
        "failure_count": 0,
        "failure_code": None,
        "progress": {
            "attempted_forward_calls": 20,
            "completed_forward_calls": 20,
            "completed_comparisons": 32,
        },
        "comparison_count": 32,
        "comparisons": comparisons,
        "numeric_crosscheck_all_32_passed": True,
        "source_commit": "fixed-source-commit",
        "runner_sha256": "fixed-runner-sha256",
    }


class BNBRowCountArtifactAnalyzerTests(unittest.TestCase):
    def test_valid_plan_and_zero_controls_produce_compact_summary(self):
        result = analyzer.summarize_record(fixture_record(), "a" * 64)
        self.assertEqual(result["artifact_sha256"], "a" * 64)
        self.assertEqual(result["verified_forward_calls"], 20)
        self.assertEqual(result["verified_comparisons"], 32)
        self.assertTrue(result["all_numeric_crosschecks_passed"])
        self.assertTrue(result["all_long_row_pairs_and_m584_repeats_exact_zero"])
        self.assertEqual(result["conditions"][analyzer.CONDITIONS[0]]["m1_to_long_max_raw_abs"], 0.001)
        self.assertEqual(result["array_recalculation"], "not_possible_outputs_not_in_artifact")

    def test_rejects_changed_comparison_plan_and_missing_call(self):
        record = fixture_record()
        record["comparisons"].pop()
        with self.assertRaisesRegex(ValueError, "comparison_count"):
            analyzer.summarize_record(record, "a" * 64)
        record = fixture_record()
        record["progress"]["completed_forward_calls"] = 19
        with self.assertRaisesRegex(ValueError, "progress_count"):
            analyzer.summarize_record(record, "a" * 64)

    def test_rejects_nonfinite_values_and_failed_status(self):
        record = fixture_record()
        record["comparisons"][0]["raw_max_abs"] = float("inf")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            analyzer.summarize_record(record, "a" * 64)
        record = fixture_record()
        record["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "not_complete"):
            analyzer.summarize_record(record, "a" * 64)

    def test_rejects_nonzero_long_row_or_repeat_controls(self):
        record = fixture_record()
        target = next(item for item in record["comparisons"]
                      if item["comparison_type"] == "row_count_within_condition"
                      and item["condition"] == analyzer.CONDITIONS[0]
                      and item["left_m"] == 584 and item["right_m"] == 586)
        target["raw_max_abs"] = 1e-12
        with self.assertRaisesRegex(ValueError, "long_row_difference"):
            analyzer.summarize_record(record, "a" * 64)
        record = fixture_record()
        repeat = next(item for item in record["comparisons"]
                      if item["comparison_type"] == "same_shape_repeat")
        repeat["raw_rms"] = 1e-12
        with self.assertRaisesRegex(ValueError, "same_shape_repeat"):
            analyzer.summarize_record(record, "a" * 64)

    def test_path_analysis_hashes_exact_existing_artifact_bytes(self):
        encoded = (json.dumps(fixture_record()) + "\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_bytes(encoded)
            result = analyzer.analyze_path(path)
        self.assertEqual(result["artifact_sha256"], hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    unittest.main()
