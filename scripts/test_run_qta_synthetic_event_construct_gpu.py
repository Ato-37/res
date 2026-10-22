from __future__ import annotations

import math
import unittest

from run_qta_synthetic_event_construct_gpu import (
    _aggregate_pair_deltas, baseline_foil_index, clip_primary_delta,
    run_nine_branch_adapter, _load_attention_components, validate_preflight_receipt,
    validate_smoke_receipt,
)


class SyntheticConstructScoringTests(unittest.TestCase):
    def test_runner_loads_attention_selector_and_capture_from_correct_modules(self):
        question_map, aggregate, capture = _load_attention_components()
        self.assertEqual(question_map.__module__, "qwen_question_temporal_attention_selector")
        self.assertEqual(aggregate.__module__, "qta_attention_forward_capture")
        self.assertEqual(capture.__module__, "qta_attention_forward_capture")

    def test_baseline_foil_is_highest_non_gold_with_alphabetical_tie_break(self):
        self.assertEqual(baseline_foil_index([-1.0, -2.0, -1.0, -3.0, -4.0],
                                             gold_index=1), 0)

    def test_delta_uses_same_gold_and_baseline_non_gold_foil_for_every_branch(self):
        scores = {
            "baseline_full": [-5, -2, -1, -3, -4],
            "baseline_evidence_ablation": [-5, -4, -1, -3, -4],
            "baseline_control_ablation": [-5, -3, -1, -3, -4],
            "qta_full": [-5, -2, -1, -3, -4],
            "qta_evidence_ablation": [-5, -3, -1, -3, -4],
            "qta_control_ablation": [-5, -1, -1, -3, -4],
            "sham_full": [-5, -2, -1, -3, -4],
            "sham_evidence_ablation": [-5, -4, -1, -3, -4],
            "sham_control_ablation": [-5, -2, -1, -3, -4],
        }
        qta_delta, sham_delta = clip_primary_delta(scores, gold_index=1)
        self.assertAlmostEqual(qta_delta, 1.0)
        self.assertAlmostEqual(sham_delta, 1.0)

    def test_adapter_executes_nine_frozen_branches_and_captures_only_baseline_full(self):
        groups = tuple((2 * index, 2 * index + 1) for index in range(16))
        calls = []
        values = {
            "baseline_full": [-5, -2, -1, -3, -4],
            "baseline_evidence_ablation": [-5, -4, -1, -3, -4],
            "baseline_control_ablation": [-5, -3, -1, -3, -4],
            "qta_full": [-5, -2, -1, -3, -4],
            "qta_evidence_ablation": [-5, -3, -1, -3, -4],
            "qta_control_ablation": [-5, -1, -1, -3, -4],
            "sham_full": [-5, -2, -1, -3, -4],
            "sham_evidence_ablation": [-5, -4, -1, -3, -4],
            "sham_control_ablation": [-5, -2, -1, -3, -4],
        }

        def forward(*, name, biased_positions, ablated_positions,
                    capture_question_attention):
            calls.append((name, tuple(biased_positions), tuple(ablated_positions),
                          capture_question_attention))
            masses = ([0.0] * 5 + [1.0, 0.9] + [0.0] * 9
                      if capture_question_attention else None)
            return values[name], masses

        result = run_nine_branch_adapter(
            gold_index=1, evidence_positions=(2, 3), control_positions=(4, 5),
            temporal_groups=groups, opaque_token="qta-synth-event-v1:seed:circle",
            forward=forward)
        expected_order = [
            "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
            "qta_full", "qta_evidence_ablation", "qta_control_ablation",
            "sham_full", "sham_evidence_ablation", "sham_control_ablation",
        ]
        self.assertEqual([row[0] for row in calls], expected_order)
        self.assertEqual([row[3] for row in calls], [True] + [False] * 8)
        self.assertEqual(calls[0][1:3], ((), ()))
        self.assertEqual(calls[1][2], groups[2] + groups[3])
        self.assertEqual(calls[2][2], groups[4] + groups[5])
        self.assertEqual(calls[3][1], groups[5] + groups[6])
        self.assertEqual(result["target_group_indices"], (5, 6))
        self.assertEqual(result["forward_count"], 9)
        self.assertAlmostEqual(result["qta_delta_f"], 1.0)

    def test_adapter_rejects_invalid_masks_and_nonfinite_scores(self):
        groups = tuple((2 * index, 2 * index + 1) for index in range(16))

        def forward(**_kwargs):
            return ([-1.0] * 5, [0.0] * 16 if _kwargs["capture_question_attention"] else None)

        with self.assertRaises(ValueError):
            run_nine_branch_adapter(
                gold_index=1, evidence_positions=(1, 1), control_positions=(4, 5),
                temporal_groups=groups, opaque_token="token", forward=forward)

        def bad_forward(*, name, capture_question_attention, **_kwargs):
            scores = [float("nan")] * 5 if name == "baseline_full" else [-1.0] * 5
            return scores, [0.0] * 16 if capture_question_attention else None

        with self.assertRaisesRegex(ValueError, "nine_branch_forward_score_nonfinite"):
            run_nine_branch_adapter(
                gold_index=1, evidence_positions=(2, 3), control_positions=(4, 5),
                temporal_groups=groups, opaque_token="token", forward=bad_forward)

    def test_preflight_receipt_binds_source_protocol_assignment_and_zero_forwards(self):
        receipt = {
            "schema_version": 1,
            "status": "passed_cpu_processor_preflight_no_model_forward",
            "purpose": "qta_synthetic_event_construct_processor_preflight",
            "scope": "synthetic_processor_geometry_only_no_qta_overlap",
            "requested_pairs": 40, "effective_pairs": 40,
            "requested_clips": 80, "effective_clips": 80,
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "model_weights_loaded": False, "gpu_used": False,
            "model_forwards": 0, "faithfulness_result": False,
            "protocol_sha256": "a" * 64,
            "source_sha256": {"fixture": "b" * 64},
            "assignment_manifest_sha256": "c" * 64,
            "authorization": "processor_preflight_only_no_scoring_authorization",
            "qta_selector_overlap_measured": False,
            "checks": {
                "paired_rendered_qa_equal": 40,
                "paired_choice_lists_equal": 40,
                "paired_processor_geometry_equal": 40,
                "clips_with_exactly_two_event_groups": 80,
                "clips_with_two_disjoint_nearest_controls": 80,
            },
            "design_balance": {
                "onset_pair_counts": [
                    {"start_frame": onset, "pairs": 10} for onset in (4, 10, 18, 24)],
                "orientation_pair_counts_by_onset": [
                    {"start_frame": onset, "circle_left": 5, "circle_right": 5}
                    for onset in (4, 10, 18, 24)],
                "sampled_cohort_seeds_unique": 40,
                "orientation_rotation_combinations_per_onset": 10,
                "circle_target_correct_label_counts": {key: 8 for key in "ABCDE"},
                "triangle_target_correct_label_counts": {key: 8 for key in "ABCDE"},
            },
            "aggregate_processor_geometry": {
                "grid_counts": [{"video_grid_thw": [16, 28, 40], "clips": 80}],
            },
        }
        validate_preflight_receipt(
            receipt, expected_protocol_sha256="a" * 64,
            expected_source_sha256={"fixture": "b" * 64},
            expected_assignment_sha256="c" * 64, mode="full")
        for key, value in (("model_forwards", 1), ("effective_clips", 79),
                           ("assignment_manifest_sha256", "d" * 64)):
            with self.subTest(key=key):
                tampered = dict(receipt)
                tampered[key] = value
                with self.assertRaises(ValueError):
                    validate_preflight_receipt(
                        tampered, expected_protocol_sha256="a" * 64,
                        expected_source_sha256={"fixture": "b" * 64},
                        expected_assignment_sha256="c" * 64, mode="full")

    def test_full_mode_requires_successful_external_smoke_binding(self):
        receipt = {
            "schema_version": 1,
            "status": "passed_external_engineering_smoke_only", "mode": "smoke",
            "requested_pairs": 1, "effective_pairs": 1,
            "requested_clips": 2, "effective_clips": 2,
            "requested_forwards": 18, "completed_forwards": 18,
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "protocol_sha256": "a" * 64,
            "source_sha256": {"runner": "b" * 64},
            "assignment_manifest_sha256": "c" * 64,
            "checkpoint_sha256": "d" * 64,
            "model_config_sha256": "9" * 64,
            "external_smoke_seed_sha256": "e" * 64,
            "preflight_receipt_sha256": "f" * 64,
            "source_commit": "deadbeef",
            "attempted_forwards": 18,
            "failure_count": 0,
            "model_weights_loaded": True,
            "gpu_used": True,
            "cohort_included": False, "faithfulness_result": False,
        }
        validate_smoke_receipt(
            receipt, expected_protocol_sha256="a" * 64,
            expected_sources={"runner": "b" * 64},
            expected_assignment_sha256="c" * 64,
            expected_checkpoint_sha256="d" * 64,
            expected_model_config_sha256="9" * 64,
            expected_smoke_seed_sha256="e" * 64,
            expected_preflight_receipt_sha256="f" * 64,
            expected_source_commit="deadbeef")
        tampered = dict(receipt, cohort_included=True)
        with self.assertRaises(ValueError):
            validate_smoke_receipt(
                tampered, expected_protocol_sha256="a" * 64,
                expected_sources={"runner": "b" * 64},
                expected_assignment_sha256="c" * 64,
                expected_checkpoint_sha256="d" * 64,
                expected_model_config_sha256="9" * 64,
                expected_smoke_seed_sha256="e" * 64,
                expected_preflight_receipt_sha256="f" * 64,
                expected_source_commit="deadbeef")

    def test_aggregate_keeps_only_onset_sufficient_statistics(self):
        pairs = [(onset, float(index), float(-index))
                 for onset in (4, 10, 18, 24) for index in range(10)]
        result = _aggregate_pair_deltas(pairs)
        self.assertEqual(list(result), ["4", "10", "18", "24"])
        self.assertEqual(result["4"]["n_pairs"], 10)
        self.assertEqual(result["4"]["qta_delta_sum"], 45.0)
        self.assertEqual(result["4"]["qta_delta_sum_squares"], 285.0)
        self.assertNotIn("pair_deltas", result)
        with self.assertRaises(ValueError):
            _aggregate_pair_deltas(pairs[:-1])


if __name__ == "__main__":
    unittest.main()
