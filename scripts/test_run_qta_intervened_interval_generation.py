from __future__ import annotations

import unittest
import torch

from run_qta_intervened_interval_generation import (
    BRANCHES, allowed_interval_next_tokens, frame_cell_edges,
    generation_eos_ids, hash_sham_group_indices, interval_code_token_sequences, interval_iou,
    compare_generation_logit_diagnostics, interval_prefix_allowed_tokens,
    summarize_generation_logits,
    validate_processor_video_metadata,
    paired_summary, parse_interval_generation, selected_group_indices,
    validate_dynamic_shape_trace, validate_generation_mask_audit,
    validate_packet, validate_receipt,
)


class IntervalGenerationRunnerTests(unittest.TestCase):
    def test_top_attention_selection_is_fixed_fraction_with_stable_ties(self):
        self.assertEqual(selected_group_indices([0.1] * 8, 8), (0,))
        self.assertEqual(selected_group_indices([0.1, 0.2, 0.9, 0.4], 4), (2,))
        with self.assertRaisesRegex(ValueError, "temporal_selector_count_mismatch"):
            selected_group_indices([0.1, 0.2], 4)

    def test_hash_sham_is_deterministic_equal_size_and_disjoint(self):
        selected = (2, 5)
        first = hash_sham_group_indices(group_count=8, top_k=2,
            selected=selected, record_token="a" * 64)
        second = hash_sham_group_indices(group_count=8, top_k=2,
            selected=selected, record_token="a" * 64)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(selected))
        self.assertFalse(set(first) & set(selected))
        self.assertEqual(tuple(sorted(first)), first)
        with self.assertRaisesRegex(ValueError, "hash_sham_selection_inputs_invalid"):
            hash_sham_group_indices(group_count=4, top_k=2,
                selected=(0, 1, 2), record_token="x")

    def test_qa_tiou_uses_best_alternative_gold_interval(self):
        score = interval_iou([2, 4], [[0, 1], [1, 5]], 10)
        self.assertEqual(score, {"temporal_iou": 0.5, "status": "scored"})

    def test_invalid_generation_has_zero_score(self):
        for prediction in (None, [4, 4], [-1, 2], [2, 11], [float("nan"), 3]):
            with self.subTest(prediction=prediction):
                self.assertEqual(interval_iou(prediction, [[2, 4]], 10),
                                 {"temporal_iou": 0.0,
                                  "status": "invalid_prediction"})
        pts = [0.5 + i for i in range(16)]
        self.assertEqual(parse_interval_generation("not-json", 16, pts),
                         {"prediction": None, "status": "invalid_interval_code"})
        self.assertEqual(parse_interval_generation("16,16", 16, pts),
                         {"prediction": None, "status": "interval_code_out_of_range"})
        self.assertEqual(parse_interval_generation("03,07 trailing", 16, pts),
                         {"prediction": None, "status": "invalid_interval_code"})

    def test_interval_code_grammar_is_complete_and_prefix_constrained(self):
        class CharacterTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return [ord(char) for char in text]

            @staticmethod
            def decode(ids, skip_special_tokens=True):
                return "".join(chr(token) for token in ids)

        grammar = interval_code_token_sequences(CharacterTokenizer(), 16)
        self.assertEqual(len(grammar), 136)
        self.assertEqual(len(set(grammar)), 136)
        eos = 999
        candidate = next(ids for ids, code in grammar.items() if code == "03,07")
        prefix = ()
        for token in candidate:
            allowed = allowed_interval_next_tokens(grammar, prefix, eos)
            self.assertIn(token, allowed)
            self.assertNotIn(eos, allowed)
            prefix += (token,)
        self.assertEqual(allowed_interval_next_tokens(grammar, candidate, eos), [eos])
        with self.assertRaisesRegex(ValueError, "prefix_not_in_grammar"):
            allowed_interval_next_tokens(grammar, (0,), eos)

    def test_interval_codes_bind_to_text_template_boundary(self):
        class CharacterTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return [ord(char) for char in text]

            @staticmethod
            def decode(ids, skip_special_tokens=True):
                return "".join(chr(token) for token in ids)

        prefix = "<|im_start|>assistant\\n"
        grammar = interval_code_token_sequences(
            CharacterTokenizer(), 16, rendered_prompt=prefix)
        self.assertEqual(len(grammar), 136)
        self.assertEqual({len(ids) for ids in grammar}, {5})

    def test_transformers_scalar_or_multiple_eos_configuration_is_normalized(self):
        self.assertEqual(generation_eos_ids(151645), (151645,))
        self.assertEqual(generation_eos_ids([151645, 151643]), (151645, 151643))
        for invalid in (None, [], [151645, 151645], [True]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                generation_eos_ids(invalid)

    def test_transformers_prefix_callback_consumes_one_dimensional_sequence(self):
        class CharacterTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return [ord(char) for char in text]

            @staticmethod
            def decode(ids, skip_special_tokens=True):
                return "".join(chr(token) for token in ids)

        grammar = interval_code_token_sequences(CharacterTokenizer(), 16)
        code = next(ids for ids, text in grammar.items() if text == "03,07")
        prompt = [17, 18, 19]
        self.assertIn(code[0], interval_prefix_allowed_tokens(
            0, prompt, prompt_length=3, sequences=grammar, eos_token_id=999))
        self.assertEqual(interval_prefix_allowed_tokens(
            0, prompt + list(code), prompt_length=3,
            sequences=grammar, eos_token_id=999), [999])
        with self.assertRaisesRegex(ValueError, "prefix_invalid"):
            interval_prefix_allowed_tokens(0, [prompt], prompt_length=3,
                                           sequences=grammar, eos_token_id=999)

    def test_frame_indices_map_to_actual_sampled_pts_cells(self):
        pts = [0.5 + index for index in range(16)]
        edges = frame_cell_edges(pts, 16.0)
        self.assertEqual(edges, [float(index) for index in range(17)])
        self.assertEqual(parse_interval_generation("03,07", 16.0, pts), {
            "prediction": [3.0, 8.0],
            "status": "valid_constrained_interval_code",
            "start_frame_index": 3, "end_frame_index": 7})
        self.assertAlmostEqual(
            frame_cell_edges([0.2 + i * 0.9 for i in range(16)], 14.2)[4],
            3.35)
        with self.assertRaisesRegex(ValueError, "sampled_frame_pts_invalid"):
            frame_cell_edges([0.5] * 16, 16.0)

    def test_video_metadata_accepts_float_roundoff_but_rejects_wrong_frame_count(self):
        metadata = {"frames_indices": list(range(16)),
            "total_num_frames": 15.999999999999998, "fps": 0.25}
        validate_processor_video_metadata(metadata, 0.25)
        with self.assertRaisesRegex(ValueError, "metadata_mismatch"):
            validate_processor_video_metadata({**metadata, "total_num_frames": 15.9}, 0.25)

    def test_raw_logits_diagnostics_preserve_compact_scores_and_same_prefix_delta(self):
        baseline = summarize_generation_logits((torch.tensor([[1.0, 0.0, -1.0]]),), [0], torch)
        treatment = summarize_generation_logits((torch.tensor([[1.5, 0.0, -1.0]]),), [0], torch)
        sham = summarize_generation_logits((torch.tensor([[1.25, 0.0, -1.0]]),), [0], torch)
        self.assertTrue(baseline[0]["greedy_top1_matches_generated"])
        self.assertGreater(treatment[0]["generated_token_log_probability"],
                           baseline[0]["generated_token_log_probability"])
        audits = {"baseline": {"raw_logit_diagnostics": baseline},
            "selected_temporal_bias": {"raw_logit_diagnostics": treatment},
            "hash_sham_bias": {"raw_logit_diagnostics": sham}}
        comparison = compare_generation_logit_diagnostics(audits)
        self.assertEqual(comparison["common_generated_prefix_tokens"], 1)
        self.assertGreater(comparison["compared_log_probability_deltas"][
            "selected_minus_baseline"][0], 0)
        with self.assertRaisesRegex(ValueError, "step_count_mismatch"):
            summarize_generation_logits([], [0], torch)

    def test_generation_mask_audit_requires_exact_treatment_and_sham_bias_cells(self):
        trace = [[5, 5], [1, 6], [1, 7]]
        observations = [
            {"layer_index": layer, "modified_cells": 2,
             "effective_bias": 0.6931471805599453,
             "max_abs_delta_error": 0.0, "verified": True}
            for layer in (4, 5) for _ in trace
        ]
        result = validate_generation_mask_audit(observations, trace, (4, 5),
                                                expected_applied=True)
        self.assertEqual(result["verified_cells"], 12)
        self.assertTrue(result["all_selected_cells_match_bias"])
        with self.assertRaisesRegex(ValueError, "value_mismatch"):
            validate_generation_mask_audit(
                [{**observations[0], "max_abs_delta_error": 0.01}, *observations[1:]], trace,
                (4, 5), expected_applied=True)
        with self.assertRaisesRegex(ValueError, "baseline_intervention_mask_observed"):
            validate_generation_mask_audit(observations, trace, (4, 5),
                                           expected_applied=False)

    def test_paired_summary_keeps_zero_iou_invalid_rows_in_denominator(self):
        records = [self._row(0.25, 0.0, 0.125), self._row(0.0, 0.0, 0.0),
                   self._row(0.5, 0.25, 0.25)]
        summary = paired_summary(records, 3)
        self.assertEqual(summary["effective_n"], 3)
        self.assertEqual(summary["paired_delta_iou"], [-0.25, 0.0, -0.25])
        self.assertAlmostEqual(summary["mean_paired_delta_iou"], -1 / 6)
        self.assertEqual(summary["mean_iou_by_condition"]["baseline"], 0.25)
        self.assertAlmostEqual(summary["mean_iou_by_condition"]["selected_temporal_bias"], 1 / 12)
        incomplete = paired_summary(records[:2], 3)
        self.assertIsNone(incomplete["mean_iou_by_condition"])

    def test_dynamic_cache_trace_requires_square_prefill_and_incremental_decode(self):
        good = {index: [[5, 5], [1, 6], [1, 7]] for index in (4, 5)}
        self.assertEqual(validate_dynamic_shape_trace(good, (4, 5), 5),
                         [[5, 5], [1, 6], [1, 7]])
        for bad in (
            {4: [[5, 9], [1, 10]], 5: [[5, 9], [1, 10]]},
            {4: [[5, 5], [1, 7]], 5: [[5, 5], [1, 7]]},
            {4: [[5, 5], [2, 7]], 5: [[5, 5], [2, 7]]},
            {4: [[5, 5], [1, 6]], 5: [[5, 5], [1, 7]]},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_dynamic_shape_trace(bad, (4, 5), 5)

    def test_receipt_schema_checks_frozen_denominator_and_arithmetic(self):
        records = [self._row(0.4, 0.2, 0.1)]
        payload = {"requested_n": 1, "packet_sha256": "a" * 64,
            "direct_faithfulness": False,
            "explanation_claim_support_established": False,
            "endpoint": "paired generated QA interval temporal IoU vs NExT-GQA QA interval",
            "records": records, "summary": paired_summary(records, 1),
            "branch_counts": {name: {"requested": 1, "attempted": 1,
                "completed": 1} for name in BRANCHES},
            "status": "completed_exploratory"}
        validate_receipt(payload, requested_n=1, packet_sha256="a" * 64)
        payload["records"][0]["generation_audit"]["baseline"]["forward_calls"] = 2
        with self.assertRaisesRegex(ValueError, "completed_audit_invalid"):
            validate_receipt(payload, requested_n=1, packet_sha256="a" * 64)
        payload["records"][0]["generation_audit"]["baseline"]["forward_calls"] = 6
        payload["summary"]["mean_paired_delta_iou"] = 0.2
        with self.assertRaisesRegex(ValueError, "summary_mismatch"):
            validate_receipt(payload, requested_n=1, packet_sha256="a" * 64)

    def test_packet_contract_requires_exactly_three_unique_wellformed_rows(self):
        def row(media):
            return {"question": "q", "choices": ["a", "b", "c", "d", "e"],
                "local_video_path": "/private/video.mp4", "video_sha256": media,
                "generated_parsed": {"valid": True}, "gold_interval_seconds": [[1, 2]]}
        packet = {"status": "passed", "dataset_split": "test",
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "claim_review_completed": False, "requested_videos": 3,
            "effective_videos": 3, "fully_structured_claim_records": 3,
            "records": [row("1" * 64), row("2" * 64), row("3" * 64)]}
        self.assertEqual(len(validate_packet(packet)), 3)
        packet["records"].pop()
        with self.assertRaisesRegex(ValueError, "packet_contract_invalid"):
            validate_packet(packet)

    @staticmethod
    def _row(baseline, treatment, sham):
        return {"condition_scores": {
            "baseline": {"temporal_iou": baseline,
                "parse_status": "valid_constrained_interval_code",
                "score_status": "scored", "prediction": [0.0, 1.0]},
            "selected_temporal_bias": {"temporal_iou": treatment,
                "parse_status": "valid_constrained_interval_code",
                "score_status": "scored", "prediction": [0.0, 1.0]},
            "hash_sham_bias": {"temporal_iou": sham,
                "parse_status": "valid_constrained_interval_code",
                "score_status": "scored", "prediction": [0.0, 1.0]}},
            "generation_audit": {
                "baseline": {"constraint_code_count": 136,
                    "constraint_eos_terminated": True,
                    "generated_token_count": 6, "constraint_code_token_count": 5,
                    "forward_calls": 6,
                    "shape_trace": [[5, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10]],
                    "raw_logit_diagnostics": IntervalGenerationRunnerTests._logit_diagnostics(),
                    "intervention_mask_audit": {"applied": False, "hook_calls": 0,
                        "verified_cells": 0, "all_selected_cells_match_bias": True,
                        "logit_bias": 0.0}},
                **{name: {"constraint_code_count": 136,
                    "constraint_eos_terminated": True,
                    "generated_token_count": 6, "constraint_code_token_count": 5,
                    "forward_calls": 6,
                    "shape_trace": [[5, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10]],
                    "raw_logit_diagnostics": IntervalGenerationRunnerTests._logit_diagnostics(),
                    "intervention_mask_audit": {"applied": True, "hook_calls": 2,
                        "verified_cells": 2, "all_selected_cells_match_bias": True,
                        "effective_mask_bias": 0.6931471805599453,
                        "logit_bias": 0.6931471805599453}}
                    for name in ("selected_temporal_bias", "hash_sham_bias")}},
            "mechanistic_logits_comparison": {
                "scope": "post hoc mechanistic diagnostic; not an efficacy or faithfulness endpoint",
                "common_generated_prefix_tokens": 6,
                "compared_log_probability_deltas": {
                    "selected_minus_baseline": [0.0] * 6,
                    "selected_minus_sham": [0.0] * 6},
                "argmax_changed_steps_vs_baseline": 0,
                "entropy_deltas_selected_minus_baseline": [0.0] * 6}}

    @staticmethod
    def _logit_diagnostics():
        return [{"generated_token_id": 1, "greedy_top1_token_id": 1,
            "greedy_top1_matches_generated": True, "top1_minus_top2_logit": 1.0,
            "generated_token_log_probability": -0.2, "entropy_nats": 0.5}
            for _ in range(6)]


if __name__ == "__main__":
    unittest.main()
