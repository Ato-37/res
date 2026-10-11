"""Tests for the fixed native teacher-forcing/cache CPU diagnostic."""

import random
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_qwen_teacher_forcing_cache_cpu as audit  # noqa: E402


class QwenTeacherForcingCacheCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit.run_diagnostic()

    def test_native_teacher_forcing_matches_sequential_fresh_cache(self):
        result = self.result
        observations = result["observations"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(observations["model_forward_calls_attempted"], 6)
        self.assertEqual(observations["model_forward_calls_completed"], 6)
        self.assertEqual(observations["model_forward_calls_by_stage_completed"], {
            "teacher_forced_forward": 1,
            "cache_prefill": 2,
            "sequential_cached_decode": 2,
            "negative_control_decode": 1,
        })
        self.assertEqual(observations["cache_lengths_expected"], [36, 37, 38])
        self.assertEqual(observations["cache_lengths_after_prefill_and_decode"], [36, 37, 38])
        self.assertTrue(observations["captured_decode_positions_match_native_slices"])
        self.assertTrue(observations["all_scores_finite"])
        self.assertEqual(
            observations["teacher_forced_vs_cache_allclose_each_suffix_prediction"],
            [True, True, True],
        )
        self.assertEqual(len(observations["teacher_forced_vs_cache_max_abs_error_each_suffix_prediction"]), 3)
        self.assertEqual(result["protocol"]["forward_api"],
                         "direct model.forward with explicit native three-row positions; no GenerationMixin packed text row")

    def test_fixed_wrong_position_control_is_discriminating(self):
        control = self.result["observations"]
        self.assertEqual(audit.WRONG_POSITION_SHIFT, 1)
        self.assertTrue(control["wrong_offset_control_not_allclose"])
        self.assertGreater(control["wrong_offset_control_max_abs_error"], audit.ATOL)
        self.assertEqual(self.result["protocol"]["negative_control"]["sensitivity_failure_action"],
                         "record as diagnostic failure; no offset/token/seed search")

    def test_report_keeps_synthetic_token_ids_private_and_scope_bounded(self):
        result = self.result
        self.assertTrue(result["fixture"]["synthetic_token_ids_omitted"])
        self.assertEqual(result["fixture"]["fixed_suffix_length"], 3)
        self.assertIn("no visual pixels, QA", result["scope"])
        self.assertIn("does not establish 7B", result["limitations"])

    def test_diagnostic_restores_rng_and_determinism_settings(self):
        old_enabled = torch.are_deterministic_algorithms_enabled()
        old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        old_threads = torch.get_num_threads()
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.set_num_threads(2)
            random.seed(9021)
            torch.manual_seed(8132)
            python_state = random.getstate()
            torch_state = torch.get_rng_state().clone()
            expected_python = random.random()
            random.setstate(python_state)
            expected_torch = torch.rand(4)
            torch.set_rng_state(torch_state)

            audit.run_diagnostic()

            self.assertEqual(random.random(), expected_python)
            self.assertTrue(torch.equal(torch.rand(4), expected_torch))
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertEqual(torch.get_num_threads(), 2)
        finally:
            torch.set_num_threads(old_threads)
            torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn)

    def test_diagnostic_restores_rng_and_settings_after_failure(self):
        random.seed(9031)
        torch.manual_seed(8142)
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        old_enabled = torch.are_deterministic_algorithms_enabled()
        old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        old_threads = torch.get_num_threads()
        try:
            with mock.patch.object(audit, "_run_diagnostic_impl", side_effect=RuntimeError("expected")):
                failed = audit.run_diagnostic()
            self.assertEqual(failed["status"], "execution_failed")
            self.assertEqual(failed["failure_stage"], "model_setup")
            self.assertEqual(failed["failure_exception_type"], "RuntimeError")
            self.assertEqual(failed["observations"]["model_forward_calls_attempted"], 0)
            self.assertNotIn("exception_message", failed)
            self.assertNotIn('"expected"', json.dumps(failed))
            self.assertEqual(random.getstate(), python_state)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), old_enabled)
            self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), old_warn)
            self.assertEqual(torch.get_num_threads(), old_threads)
        finally:
            torch.set_num_threads(old_threads)
            torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn)

    def test_cli_persists_sanitized_failure_manifest_and_exits_nonzero(self):
        failure = {
            "schema_version": 1,
            "purpose": "native_teacher_forcing_vs_fresh_cache_cpu_api_diagnostic",
            "status": "diagnostic_gate_failed",
            "failure_stage": "gate_validation",
            "seed": audit.SEED,
            "observations": {
                "model_forward_calls_attempted": 6,
                "model_forward_calls_completed": 6,
                "cache_lengths_after_prefill_and_decode": [36, 37, 38],
                "teacher_forced_vs_cache_allclose_each_suffix_prediction": [True, True, True],
                "wrong_offset_control_not_allclose": False,
                "wrong_offset_control_max_abs_error": 0.0,
            },
            "model_config": {"config_sha256": "c" * 64},
            "protocol": audit._protocol_summary(),
            "scope": "synthetic CPU-only diagnostic",
            "failure_exception_type": None,
            "elapsed_seconds": 0.001,
        }
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / "results").mkdir()
            with mock.patch.object(audit, "run_diagnostic", return_value=failure), \
                 mock.patch.object(audit, "_sha256", return_value="a" * 64), \
                 mock.patch.object(audit, "_source_commit", return_value="fixture-commit"):
                with self.assertRaises(SystemExit) as raised:
                    audit.main(["--output", "results/failure.json"], repo_path=repo)
            self.assertEqual(raised.exception.code, 1)
            output = repo / "results" / "failure.json"
            self.assertTrue(output.is_file())
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "diagnostic_gate_failed")
            self.assertEqual(saved["source_commit"], "fixture-commit")
            self.assertEqual(saved["source_sha256"]["runner"], "a" * 64)
            self.assertEqual(saved["observations"]["model_forward_calls_completed"], 6)
            self.assertEqual(saved["protocol"]["atol"], 1e-5)
            self.assertEqual(saved["protocol"]["rtol"], 1e-4)
            self.assertFalse(saved["observations"]["wrong_offset_control_not_allclose"])
            self.assertNotIn("input_ids", json.dumps(saved))
            self.assertNotIn("exception_message", saved)


if __name__ == "__main__":
    unittest.main()
