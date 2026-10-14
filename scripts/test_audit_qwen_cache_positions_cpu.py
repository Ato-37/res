"""CPU-only tests for the installed Qwen cache-position diagnostic."""

import sys
from pathlib import Path
import random
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_qwen_cache_positions_cpu as audit  # noqa: E402
from audit_qwen_cache_positions_cpu import run_diagnostic  # noqa: E402


class QwenCachePositionCpuAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_diagnostic()

    def test_fixed_fixture_discriminates_video_temporal_and_cache_delta(self):
        result = self.result
        geometry = result["position_geometry"]
        self.assertEqual(result["fixture"]["video_grid_thw"], [8, 4, 4])
        self.assertEqual(result["fixture"]["second_per_grid_seconds"], 0.75)
        self.assertEqual(geometry["candidate_temporal_anchors"], [2, 3, 5, 6, 8, 9, 11, 12])
        if geometry["fractional_candidate_equals_native"]:
            self.assertEqual(
                geometry["native_temporal_anchors"],
                geometry["candidate_temporal_anchors"],
            )
            self.assertFalse(geometry["candidate_delta_differs"])
        else:
            self.assertEqual(geometry["native_temporal_anchors"], [2] * 8)
            self.assertTrue(geometry["candidate_delta_differs"])
        self.assertTrue(geometry["non_temporal_positions_preserved"])
        self.assertIn("not semantic validation", geometry["delta_convention"])

    def test_uninterrupted_generation_noop_and_candidate_use_last_text_policy(self):
        generated = self.result["uninterrupted_generate"]
        self.assertTrue(generated["native_auto_prefill_matches_get_rope_index"])
        self.assertTrue(generated["explicit_native_prefill_matches_get_rope_index"])
        self.assertTrue(generated["explicit_candidate_prefill_matches_helper"])
        self.assertTrue(generated["native_auto_vs_explicit_native_scores_bitwise_equal"])
        self.assertTrue(generated["native_auto_vs_explicit_native_sequences_equal"])
        self.assertTrue(generated["candidate_prefill_changes_only_video_temporal_row"])
        self.assertEqual(generated["decode_positions_are_last_prompt_coordinate_plus_one"], 6)
        self.assertTrue(generated["candidate_decode_equals_native_decode"])
        self.assertTrue(generated["explicit_native_generation_left_rope_delta_none"])
        self.assertTrue(generated["explicit_candidate_generation_left_rope_delta_none"])
        self.assertTrue(generated["native_auto_populated_rope_delta"])

    def test_explicit_prefill_preserves_cleared_or_stale_rope_delta_state(self):
        state = self.result["explicit_prefill_state"]
        self.assertTrue(state["cleared_rope_delta_stayed_cleared"])
        self.assertTrue(state["stale_rope_delta_survived_explicit_prefill"])
        if self.result["position_geometry"]["fractional_candidate_equals_native"]:
            self.assertEqual(
                state["native_delta_from_actual_get_rope_index"],
                state["candidate_delta_manually_set_for_cache_resume"],
            )
        else:
            self.assertNotEqual(
                state["native_delta_from_actual_get_rope_index"],
                state["candidate_delta_manually_set_for_cache_resume"],
            )

    def test_cache_resume_slices_full_mask_and_exposes_delta_policy(self):
        cache = self.result["cache_resume_generate"]
        self.assertTrue(cache["one_token_positions_after_generation_slice"])
        self.assertTrue(cache["finite_scores_all_cases"])
        self.assertEqual(cache["cleared_delta_decode_position"], 36)
        if self.result["position_geometry"]["fractional_candidate_equals_native"]:
            self.assertEqual(
                cache["native_delta_decode_position"],
                cache["candidate_mechanical_delta_decode_position"],
            )
        else:
            self.assertEqual(cache["native_delta_decode_position"], 6)
            self.assertEqual(cache["candidate_mechanical_delta_decode_position"], 13)
        self.assertIn("not claimed semantically correct", cache["interpretation"])

    def test_direct_full_mask_cached_forward_shape_is_reported_before_any_fix(self):
        report = self.result["direct_cached_forward"]
        self.assertEqual(report["past_length"], 36)
        self.assertEqual(report["current_input_length"], 1)
        self.assertEqual(report["computed_position_ids_length_from_full_mask"], 37)
        self.assertTrue(report["shape_mismatch_observed"])
        # The installed source's direct path is expected to error when the
        # full-mask positions reach RoPE for a one-token query. Keep the test
        # explicit so a library update prompts a fresh audit.
        self.assertEqual(report["outcome"], "raised")
        self.assertIsNotNone(report["error_type"])
        paired = report["paired_explicit_one_token_positions"]
        self.assertEqual(paired["position_ids_length"], 1)
        self.assertEqual(paired["position_coordinate"], 6)
        self.assertEqual(paired["outcome"], "returned")
        self.assertTrue(paired["finite_logits"])
        self.assertIn("shape contract only", paired["interpretation"])

    def test_long_interval_native_cache_and_continuous_policies_are_distinct(self):
        report = self.result["long_interval_native_policy"]
        self.assertEqual(report["second_per_grid_seconds"], 5.0)
        self.assertTrue(report["positions_differ"])
        self.assertNotEqual(
            report["cache_resume_position_from_native_max_plus_one"],
            report["continuous_generate_position_from_last_packed_coordinate_plus_one"],
        )
        self.assertIn("neither is endorsed", report["interpretation"])

    def test_run_diagnostic_restores_global_rng_and_determinism_state(self):
        old_enabled = torch.are_deterministic_algorithms_enabled()
        old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            random.seed(9981)
            torch.manual_seed(7712)
            random_state = random.getstate()
            torch_state = torch.get_rng_state().clone()
            expected_py_next = random.random()
            random.setstate(random_state)
            expected_torch_next = torch.rand(5)
            torch.set_rng_state(torch_state)

            run_diagnostic()

            self.assertEqual(random.random(), expected_py_next)
            self.assertTrue(torch.equal(torch.rand(5), expected_torch_next))
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        finally:
            torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn)

    def test_run_diagnostic_restores_state_after_failure(self):
        random.seed(331)
        torch.manual_seed(441)
        random_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        old_enabled = torch.are_deterministic_algorithms_enabled()
        old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            with mock.patch.object(audit, "_run_diagnostic_impl", side_effect=RuntimeError("expected")):
                with self.assertRaisesRegex(RuntimeError, "expected"):
                    audit.run_diagnostic()
            self.assertEqual(random.getstate(), random_state)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), old_enabled)
            self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), old_warn)
        finally:
            torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn)


if __name__ == "__main__":
    unittest.main()
