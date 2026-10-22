from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from qta_multiquery_video_key_intervention import (
    apply_multiquery_video_key_intervention,
    append_target_to_unpadded_batch,
    aggregate_captured_question_group_attention,
    compute_text_annotation_selective_dependence,
    freeze_generated_continuation,
    install_multiquery_video_key_intervention_hooks,
    report_qta_key_overlap,
    remove_hooks,
    score_target_logits_window,
    validate_evidence_control_key_sets,
)


class MultiqueryVideoKeyInterventionTests(unittest.TestCase):
    def setUp(self):
        # Four causal query rows and six keys; visual keys are 1 and 2.
        self.mask = torch.full((1, 1, 6, 6), float("-inf"), dtype=torch.float32)
        for q in range(6):
            self.mask[0, 0, q, :q + 1] = 0.0

    def test_bias_and_annotation_ablation_touch_only_target_rows(self):
        result = apply_multiquery_video_key_intervention(
            attention_mask=self.mask, biased_positions=(1,), ablated_positions=(2,),
            query_positions=(3, 4), logit_bias=0.6931471805599453,
            torch_module=torch)
        self.assertTrue(torch.allclose(result[0, 0, 3:5, 1], torch.tensor([0.6931472] * 2)))
        self.assertTrue(torch.isneginf(result[0, 0, 3:5, 2]).all())
        self.assertTrue(torch.equal(result[0, 0, :3], self.mask[0, 0, :3]))
        self.assertTrue(torch.equal(result[0, 0, 5], self.mask[0, 0, 5]))
        self.assertEqual(float(result[0, 0, 3, 3]), 0.0)

    def test_ablation_takes_precedence_on_overlap(self):
        result = apply_multiquery_video_key_intervention(
            attention_mask=self.mask, biased_positions=(1, 2), ablated_positions=(2,),
            query_positions=(3, 4), logit_bias=0.6931471805599453,
            torch_module=torch)
        self.assertAlmostEqual(float(result[0, 0, 3, 1]), 0.69314718, places=6)
        self.assertTrue(torch.isneginf(result[0, 0, 3, 2]))

    def test_negative_bias_suppresses_only_selected_keys_and_target_rows(self):
        result = apply_multiquery_video_key_intervention(
            attention_mask=self.mask, biased_positions=(1,), ablated_positions=(),
            query_positions=(3, 4), logit_bias=-0.6931471805599453,
            torch_module=torch)
        self.assertTrue(torch.allclose(result[0, 0, 3:5, 1],
                                       torch.tensor([-0.6931472] * 2)))
        self.assertTrue(torch.equal(result[0, 0, :3], self.mask[0, 0, :3]))
        self.assertTrue(torch.equal(result[0, 0, 5], self.mask[0, 0, 5]))

    def test_rejects_noncausal_or_unavailable_keys(self):
        with self.assertRaisesRegex(ValueError, "precede_every_query"):
            apply_multiquery_video_key_intervention(
                attention_mask=self.mask, biased_positions=(3,), ablated_positions=(),
                query_positions=(3, 4), logit_bias=0.7, torch_module=torch)
        bad = self.mask.clone()
        bad[0, 0, 4, 1] = float("-inf")
        with self.assertRaisesRegex(ValueError, "visible_zero"):
            apply_multiquery_video_key_intervention(
                attention_mask=bad, biased_positions=(1,), ablated_positions=(),
                query_positions=(3, 4), logit_bias=0.7, torch_module=torch)

    def test_rejects_duplicate_or_unsorted_positions(self):
        for queries in ((3, 3), (4, 3)):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                apply_multiquery_video_key_intervention(
                    attention_mask=self.mask, biased_positions=(1,), ablated_positions=(),
                    query_positions=queries, logit_bias=0.7, torch_module=torch)

    def test_scores_exact_causal_target_rows_and_mean(self):
        # Prompt P=5: y0 comes from absolute row 4, y1 from row 5; row 6 is extra.
        logits = torch.tensor([[[9.0, 0.0, 0.0], [0.0, 9.0, 0.0],
                                [0.0, 0.0, 9.0]]])
        scored = score_target_logits_window(logits=logits, target_token_ids=(0, 1),
            prompt_length=5, first_query_position=4, torch_module=torch)
        self.assertEqual(scored["last_target_query_position"], 5)
        expected = torch.log_softmax(logits[0, :2], dim=-1)[
            range(2), torch.tensor([0, 1])].mean()
        self.assertAlmostEqual(scored["target_logprob_mean"], float(expected), places=6)
        with self.assertRaisesRegex(ValueError, "offset_invalid"):
            score_target_logits_window(logits=logits, target_token_ids=(0, 1),
                prompt_length=5, first_query_position=5, torch_module=torch)

    def test_rejects_wrong_logits_window(self):
        logits = torch.zeros((1, 2, 3))
        with self.assertRaisesRegex(ValueError, "window_length"):
            score_target_logits_window(logits=logits, target_token_ids=(0, 1),
                prompt_length=5, first_query_position=4, torch_module=torch)

    def test_selective_dependence_difference_in_differences(self):
        result = compute_text_annotation_selective_dependence(
            baseline_evidence_logprob=-3.0, baseline_control_logprob=-2.0,
            qta_evidence_logprob=-4.5, qta_control_logprob=-2.5)
        self.assertEqual(result["baseline_annotation_selective_text_dependence"], 1.0)
        self.assertEqual(result["qta_annotation_selective_text_dependence"], 2.0)
        self.assertEqual(result["delta_annotation_selective_text_dependence"], 1.0)
        null = compute_text_annotation_selective_dependence(
            baseline_evidence_logprob=-2.0, baseline_control_logprob=-1.0,
            qta_evidence_logprob=-5.0, qta_control_logprob=-4.0)
        self.assertEqual(null["delta_annotation_selective_text_dependence"], 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_text_annotation_selective_dependence(
                baseline_evidence_logprob=float("nan"), baseline_control_logprob=0,
                qta_evidence_logprob=0, qta_control_logprob=0)

    def test_registered_hook_applies_same_multiquery_contract_and_cleans_up(self):
        class SelfAttention(torch.nn.Module):
            def forward(self, *, attention_mask):
                return attention_mask

        attention = SelfAttention()
        model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)])))
        handles = install_multiquery_video_key_intervention_hooks(
            model=model, layer_indices=(0,), biased_positions=(1,), ablated_positions=(2,),
            query_positions=(3, 4), logit_bias=0.6931471805599453, torch_module=torch)
        try:
            changed = attention(attention_mask=self.mask)
            self.assertTrue(torch.isneginf(changed[0, 0, 3:5, 2]).all())
            self.assertTrue(torch.allclose(changed[0, 0, 3:5, 1],
                                           torch.tensor([0.6931472] * 2)))
        finally:
            remove_hooks(handles)
        restored = attention(attention_mask=self.mask)
        self.assertTrue(torch.equal(restored, self.mask))

    def test_twelve_production_layer_indices_receive_identical_fixed_rows(self):
        class SelfAttention(torch.nn.Module):
            def forward(self, *, attention_mask):
                return attention_mask

        attentions = [SelfAttention() for _ in range(16)]
        model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention) for attention in attentions])))
        handles = install_multiquery_video_key_intervention_hooks(
            model=model, layer_indices=tuple(i - 1 for i in range(5, 17)),
            biased_positions=(1,), ablated_positions=(2,), query_positions=(3, 4),
            logit_bias=0.6931471805599453, torch_module=torch)
        try:
            outputs = [attentions[index](attention_mask=self.mask)
                       for index in range(16)]
        finally:
            remove_hooks(handles)
        for index in range(16):
            if 4 <= index <= 15:
                self.assertTrue(torch.isneginf(outputs[index][0, 0, 3:5, 2]).all())
                self.assertTrue(torch.allclose(outputs[index][0, 0, 3:5, 1],
                                               torch.tensor([0.6931472] * 2)))
            else:
                self.assertTrue(torch.equal(outputs[index], self.mask))

    def test_freezes_raw_generated_tail_only_after_exact_prompt_prefix(self):
        frozen = freeze_generated_continuation(
            prompt_token_ids=(10, 11, 12), generated_sequence_ids=(10, 11, 12, 19, 20, 99),
            eos_token_id=99)
        self.assertEqual(frozen["target_token_ids"], (19, 20, 99))
        self.assertEqual(frozen["target_token_count"], 3)
        self.assertTrue(frozen["terminal_seen"])
        with self.assertRaisesRegex(ValueError, "prefix_mismatch"):
            freeze_generated_continuation(
                prompt_token_ids=(10, 11, 12), generated_sequence_ids=(10, 11, 13, 19),
                eos_token_id=99)

    def test_appends_frozen_ids_and_returns_exact_target_query_rows(self):
        batch = {"input_ids": torch.tensor([[10, 11, 12, 13]]),
                 "attention_mask": torch.ones((1, 4), dtype=torch.long),
                 "mm_token_type_ids": torch.tensor([[0, 0, 1, 0]]),
                 "position_ids": torch.arange(4).view(1, 1, 4)}
        extended, queries = append_target_to_unpadded_batch(
            batch=batch, target_token_ids=(20, 21, 99), torch_module=torch)
        self.assertEqual(extended["input_ids"].tolist(), [[10, 11, 12, 13, 20, 21, 99]])
        self.assertEqual(queries, (3, 4, 5))
        self.assertEqual(extended["mm_token_type_ids"].tolist(), [[0, 0, 1, 0, 0, 0, 0]])
        self.assertNotIn("position_ids", extended)
        self.assertEqual(batch["input_ids"].shape[1], 4)
        padded = dict(batch, attention_mask=torch.tensor([[1, 1, 1, 0]]))
        with self.assertRaisesRegex(ValueError, "unpadded"):
            append_target_to_unpadded_batch(batch=padded, target_token_ids=(2,), torch_module=torch)

    def test_hook_is_removed_after_forward_exception(self):
        class FailingAttention(torch.nn.Module):
            def forward(self, *, attention_mask):
                raise RuntimeError("synthetic failure")

        attention = FailingAttention()
        model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)])))
        handles = install_multiquery_video_key_intervention_hooks(
            model=model, layer_indices=(0,), biased_positions=(1,), ablated_positions=(),
            query_positions=(3, 4), logit_bias=0.6931471805599453, torch_module=torch)
        try:
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                attention(attention_mask=self.mask)
        finally:
            remove_hooks(handles)
        self.assertEqual(len(attention._forward_pre_hooks), 0)

    def test_equal_flattened_key_budget_and_independent_qta_overlap_report(self):
        valid = validate_evidence_control_key_sets(
            evidence_positions=(1, 2), control_positions=(5, 6))
        self.assertEqual(valid["evidence_key_count"], valid["control_key_count"])
        with self.assertRaisesRegex(ValueError, "budget_mismatch"):
            validate_evidence_control_key_sets(
                evidence_positions=(1, 2), control_positions=(5,))
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_evidence_control_key_sets(
                evidence_positions=(1, 2), control_positions=(2, 3))
        overlap = report_qta_key_overlap(qta_positions=(1, 4),
            evidence_positions=(1, 2), control_positions=(5, 6))
        self.assertEqual(overlap["qta_evidence_overlap_count"], 1)
        self.assertEqual(overlap["qta_control_overlap_count"], 0)

    def test_aggregates_compact_question_attention_rows_by_group(self):
        rows = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                              [0.3, 0.1, 0.4, 0.2]]])
        scores = aggregate_captured_question_group_attention(
            attention_by_layer={4: rows, 5: rows.clone()}, layer_indices=(4, 5),
            group_positions=((0, 1), (2, 3)), question_token_count=2,
            torch_module=torch)
        self.assertAlmostEqual(scores[0], 0.35, places=6)
        self.assertAlmostEqual(scores[1], 0.65, places=6)
        with self.assertRaisesRegex(ValueError, "shape_invalid"):
            aggregate_captured_question_group_attention(
                attention_by_layer={4: rows, 5: rows[:, :1, :]},
                layer_indices=(4, 5), group_positions=((0, 1), (2, 3)),
                question_token_count=2, torch_module=torch)


if __name__ == "__main__":
    unittest.main()
