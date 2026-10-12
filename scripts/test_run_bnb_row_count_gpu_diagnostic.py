"""CPU contract tests for the fixed BNB row-count diagnostic."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import types
import unittest
import uuid
from unittest import mock

import torch

import run_bnb_row_count_gpu_diagnostic as diagnostic


class BNBRowCountDiagnosticTests(unittest.TestCase):
    def test_fixed_inputs_are_deterministic_and_do_not_touch_global_rng(self):
        torch.manual_seed(701)
        before = torch.random.get_rng_state().clone()
        w1, x1, ext1 = diagnostic.build_fixed_inputs(width=16, seed=20261012)
        after = torch.random.get_rng_state().clone()
        w2, x2, ext2 = diagnostic.build_fixed_inputs(width=16, seed=20261012)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(w1, w2))
        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(torch.equal(ext1, ext2))
        self.assertEqual(tuple(w1.shape), (16, 16))
        self.assertEqual(tuple(x1.shape), (584, 16))
        self.assertEqual(tuple(ext1.shape), (586, 16))
        self.assertTrue(torch.equal(ext1[584], x1[0]))
        self.assertTrue(torch.equal(ext1[585], x1[0]))

    def test_all_row_counts_select_the_same_declared_source_vector(self):
        _, x, ext = diagnostic.build_fixed_inputs(width=8)
        rows = diagnostic.make_row_inputs(x, ext)
        self.assertTrue(torch.equal(rows[1][0][0], x[583]))
        for count in (584, 585, 586):
            self.assertTrue(torch.equal(rows[count][0][583], x[583]))
            self.assertEqual(rows[count][1], 583)
        self.assertEqual(rows[1][1], 0)
        with self.assertRaises(ValueError):
            diagnostic.make_row_inputs(x, ext, counts=(1, 583))

    def test_fixed_comparison_matrix_has_32_rows_and_repeat_controls(self):
        self.assertEqual(diagnostic.EXPECTED_FORWARD_CALLS, 20)
        self.assertEqual(diagnostic.EXPECTED_COMPARISONS, 32)
        pairs = diagnostic.build_comparison_plan()
        self.assertEqual(len(pairs), 32)
        self.assertEqual(sum(p["comparison_type"] == "row_count_within_condition" for p in pairs), 24)
        repeats = [p for p in pairs if p["comparison_type"] == "same_shape_repeat"]
        self.assertEqual(len(repeats), 4)
        self.assertEqual({p["condition"] for p in repeats}, set(diagnostic.CONDITIONS))
        self.assertTrue(all(p["m"] == 584 for p in repeats))
        self.assertEqual(sum(p["comparison_type"] == "bnb_vs_dequantized_same_quantized_weight" for p in pairs), 4)
        self.assertEqual([p["m"] for p in pairs[-4:]], [1, 584, 585, 586])
        with self.assertRaises(ValueError):
            diagnostic.build_comparison_plan((1, 584, 585))

    def test_source_dispatch_prediction_distinguishes_floor_and_fallback(self):
        self.assertEqual(diagnostic.source_dispatch_prediction(1)["route_prediction"],
                         "custom_kernel_candidate_small_m_floor")
        for m in (584, 585, 586):
            item = diagnostic.source_dispatch_prediction(m)
            self.assertEqual(item["route_prediction"], "dequantize_then_f_linear_by_non_small_weight_m_gt_512")
            self.assertFalse(item["actual_kernel_observed"])

    def test_scalar_crosscheck_and_nonfinite_rejection(self):
        left = torch.tensor([1.25, -3.0, 5.5], dtype=torch.float32)
        right = torch.tensor([1.0, -2.5, 5.25], dtype=torch.float32)
        result = diagnostic.summarize_pair(left, right)
        self.assertTrue(result["numeric_crosscheck_within_1e_10"])
        self.assertAlmostEqual(result["raw_max_abs"], 0.5)
        self.assertGreater(result["relative_l2_to_left"], 0.0)
        with self.assertRaises(ValueError):
            diagnostic.summarize_pair(torch.tensor([1.0, float("nan")]), torch.tensor([0.0, 1.0]))
        with self.assertRaises(ValueError):
            diagnostic.summarize_pair(torch.ones(2), torch.ones(3))

    def test_bfloat16_numpy_crosscheck_converts_before_numpy(self):
        left = torch.tensor([1.25, -3.0, 5.5], dtype=torch.bfloat16)
        right = torch.tensor([1.0, -2.5, 5.25], dtype=torch.bfloat16)
        result = diagnostic.summarize_pair(left, right)
        self.assertTrue(result["numeric_crosscheck_within_1e_10"])
        self.assertAlmostEqual(result["raw_max_abs"], 0.5)

    def test_zero_reference_relative_l2_is_explicit(self):
        equal = diagnostic.summarize_pair(torch.zeros(4), torch.zeros(4))
        self.assertIsNone(equal["relative_l2_to_left"])
        self.assertEqual(equal["relative_l2_status"], "zero_reference_equal")
        different = diagnostic.summarize_pair(torch.zeros(4), torch.tensor([1.0, 0, 0, 0]))
        self.assertIsNone(different["relative_l2_to_left"])
        self.assertEqual(different["relative_l2_status"], "zero_reference_different")

    def test_scalar_crosscheck_rejects_float64_aggregate_overflow(self):
        with self.assertRaisesRegex(ValueError, "overflowed float64"):
            diagnostic.summarize_pair(torch.tensor([1e308, -1e308], dtype=torch.float64),
                                      torch.tensor([-1e308, 1e308], dtype=torch.float64))

    def test_dispatch_trace_captures_real_python_route_and_fallback_weight(self):
        expected = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).contiguous()

        def dequant(data, absmax=None, blocksize=None, quant_type=None, dtype=None, out=None):
            out.copy_(expected)
            return out

        fake_ops = types.SimpleNamespace(
            _gemm_4bit_kernel_impl=lambda *args, **kwargs: torch.zeros((1, 1)),
            _dequant_linear_fallback=lambda *args, **kwargs: torch.zeros((1, 1)),
            _dequantize_4bit_impl=dequant,
        )
        trace = diagnostic.BNBDispatchTrace(fake_ops, expected)
        originals = {key: getattr(fake_ops, key) for key in (
            "_gemm_4bit_kernel_impl", "_dequant_linear_fallback", "_dequantize_4bit_impl")}
        with trace:
            x = torch.zeros((584, 4), dtype=torch.bfloat16)
            trace.active_m = 584
            trace.active_label = "584"
            fake_ops._dequant_linear_fallback(x)
            destination = torch.empty_like(expected)
            fake_ops._dequantize_4bit_impl(None, None, 64, "fp4", torch.bfloat16, out=destination)
        self.assertEqual(trace.python_routes, [{"m": 584, "python_path": "dequantize_then_f_linear"}])
        self.assertTrue(trace.fallback_weights["584"]["bitwise_equal_public_dequant"])
        self.assertEqual(trace.fallback_weights["584"]["shape_out_in"], [3, 4])
        self.assertTrue(all(getattr(fake_ops, key) is value for key, value in originals.items()))

    def test_result_writer_is_private_exclusive_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            diagnostic.write_exclusive_private_json(path, {"status": "complete", "failures": 0})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text())["status"], "complete")
            with self.assertRaises(FileExistsError):
                diagnostic.write_exclusive_private_json(path, {"status": "other"})
            self.assertEqual(json.loads(path.read_text())["status"], "complete")

    def test_reserved_output_exists_before_diagnostic_starts(self):
        repo = Path(__file__).resolve().parents[1]
        output = repo / "results" / f"test_reserved_{uuid.uuid4().hex}.json"
        relative = output.relative_to(repo)
        def observe_reserved_output(*args, **kwargs):
            self.assertTrue(output.exists())
            raise RuntimeError("synthetic_cpu_test_stop")

        try:
            with mock.patch.object(diagnostic, "run_diagnostic", side_effect=observe_reserved_output):
                status = diagnostic.main(["--output", str(relative)])
            self.assertEqual(status, 1)
            self.assertEqual(json.loads(output.read_text())["failure_count"], 1)
        finally:
            output.unlink(missing_ok=True)

    def test_quant_state_witness_hashes_quant_tensors_and_packed_weight(self):
        packed = torch.tensor([1, 2, 3], dtype=torch.uint8)
        absmax = torch.tensor([0.5, 0.25], dtype=torch.float32)
        code = torch.arange(4, dtype=torch.float32)
        state = types.SimpleNamespace(
            quant_type="fp4", dtype=torch.bfloat16, blocksize=64, nested=False,
            absmax=absmax, code=code,
        )
        weight = types.SimpleNamespace(data=packed, quant_state=state, dtype=torch.uint8,
                                       shape=(3,), __class__=type("Params4bit", (), {}))
        layer = types.SimpleNamespace(compute_dtype=torch.bfloat16, weight=weight)
        witness = diagnostic.quant_state_witness(layer, weight)
        self.assertEqual(witness["state_tensors"]["absmax"]["sha256"], diagnostic.tensor_sha256(absmax))
        self.assertEqual(witness["packed_weight_sha256"], diagnostic.tensor_sha256(packed))

    def test_output_must_be_direct_results_json(self):
        repo = Path(__file__).resolve().parents[1]
        valid = diagnostic.exclusive_result_path(repo, "results/bnb_diagnostic_test.json")
        self.assertEqual(valid.parent, (repo / "results").resolve())
        for invalid in ("results/nested/x.json", "results/x.txt", "scripts/x.json"):
            with self.assertRaises(ValueError):
                diagnostic.exclusive_result_path(repo, invalid)


if __name__ == "__main__":
    unittest.main()
