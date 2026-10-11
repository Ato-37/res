"""Fixed CPU tests for Python/float32 Qwen temporal-ID quantization differences."""

import json
from pathlib import Path
import tempfile
import unittest

from audit_qwen_fractional_precision import (
    BASE_POSITION,
    FIXED_INTERVALS_SECONDS,
    TEMPORAL_STEPS,
    _write_exclusive,
    audit_interval,
)


class QwenFractionalPrecisionTests(unittest.TestCase):
    def test_fixed_case_list_is_frozen_and_complete(self):
        self.assertEqual(FIXED_INTERVALS_SECONDS, (
            0.49999999, 0.5, 0.50000001,
            0.99999999, 1.0, 1.00000001,
            199.9998016357422, 0.75,
        ))
        self.assertEqual(TEMPORAL_STEPS, 4)

    def test_subsecond_fixture_matches_both_float32_source_orders(self):
        result = audit_interval(0.75, steps=2)
        expected = [BASE_POSITION, BASE_POSITION + 1]
        self.assertEqual(result["helper_python_float_fractional_ticks"], expected)
        self.assertEqual(result["transformers_main_torch_float32_ticks"], expected)
        self.assertEqual(result["transformers_449_torch_float32_ticks"], expected)
        self.assertTrue(result["helper_matches_transformers_main_torch_float32"])
        self.assertTrue(result["helper_matches_transformers_449_torch_float32"])

    def test_fixed_near_boundary_values_expose_rounding_differences(self):
        below_half = audit_interval(0.49999999)
        self.assertEqual(below_half["float32_represented_seconds"], 0.5)
        self.assertEqual(below_half["helper_python_float_fractional_ticks"], [15, 15, 16, 17])
        self.assertEqual(below_half["transformers_main_torch_float32_ticks"], [15, 16, 17, 18])
        self.assertEqual(below_half["helper_vs_main_mismatch_step_indices"], [1, 2, 3])

        below_one = audit_interval(0.99999999)
        self.assertEqual(below_one["float32_represented_seconds"], 1.0)
        self.assertEqual(below_one["helper_python_float_fractional_ticks"], [15, 16, 18, 20])
        self.assertEqual(below_one["transformers_main_torch_float32_ticks"], [15, 17, 19, 21])
        self.assertEqual(below_one["helper_vs_main_mismatch_step_indices"], [1, 2, 3])

    def test_recorded_noaa_interval_helper_matches_runtime_candidate(self):
        result = audit_interval(199.9998016357422, steps=2)
        self.assertEqual(result["helper_python_float_fractional_ticks"], [15, 414])
        self.assertEqual(result["transformers_main_torch_float32_ticks"], [15, 414])
        self.assertEqual(result["transformers_449_torch_float32_ticks"], [15, 414])

    def test_output_must_be_unique_json_directly_in_results(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            report = {"purpose": "test"}
            output = _write_exclusive(Path("results/out.json"), repo, report)
            self.assertEqual(json.loads(output.read_text()), report)
            with self.assertRaises(FileExistsError):
                _write_exclusive(Path("results/out.json"), repo, report)
            with self.assertRaises(ValueError):
                _write_exclusive(Path("other/out.json"), repo, report)


if __name__ == "__main__":
    unittest.main()
