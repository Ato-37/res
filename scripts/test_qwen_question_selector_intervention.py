from __future__ import annotations

import math
import json
import hashlib
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import torch

from qwen_internal_attention_ablation import compose_video_key_bias_and_ablation
from smoke_qwen_question_selector_intervention import (
    _choice_log_probabilities, _close_private_output, _prepare_private_output,
    _resolve_choice_continuation_ids, _score_treatment_branches,
    _validate_processed_choice_suffix, _write_private_json,
    temporal_group_overlap,
)
from qta_nine_branch_plan import build_qta_nine_branch_plan, validate_qta_nine_branch_plan


class QuestionSelectorInterventionCompositionTests(unittest.TestCase):
    def setUp(self):
        self.mask = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        upper = torch.triu(torch.ones((5, 5), dtype=torch.bool), diagonal=1)
        self.mask[0, 0][upper] = float("-inf")

    def test_bias_then_same_key_ablation_leaves_key_blocked(self):
        combined = compose_video_key_bias_and_ablation(
            attention_mask=self.mask, biased_positions=(1, 2),
            ablated_positions=(1, 2), query_position=4,
            logit_bias=math.log(2.0), torch_module=torch)
        self.assertTrue(torch.isneginf(combined[0, 0, 4, 1:3]).all())
        self.assertTrue(torch.equal(combined[0, 0, 4, [0, 3, 4]],
                                    self.mask[0, 0, 4, [0, 3, 4]]))

    def test_bias_is_retained_when_a_disjoint_key_is_ablated(self):
        combined = compose_video_key_bias_and_ablation(
            attention_mask=self.mask, biased_positions=(1,),
            ablated_positions=(2,), query_position=4,
            logit_bias=math.log(2.0), torch_module=torch)
        self.assertAlmostEqual(float(combined[0, 0, 4, 1]), math.log(2.0))
        self.assertTrue(torch.isneginf(combined[0, 0, 4, 2]))
        self.assertEqual(float(combined[0, 0, 4, 0]), 0.0)

    def test_partial_overlap_gives_ablation_precedence(self):
        combined = compose_video_key_bias_and_ablation(
            attention_mask=self.mask, biased_positions=(1, 2),
            ablated_positions=(2, 3), query_position=4,
            logit_bias=math.log(2.0), torch_module=torch)
        self.assertAlmostEqual(float(combined[0, 0, 4, 1]), math.log(2.0))
        self.assertTrue(torch.isneginf(combined[0, 0, 4, 2:4]).all())

    def test_condition_plan_has_nine_unique_reusable_branches(self):
        plan = build_qta_nine_branch_plan(
            row_binding_sha256="9" * 64,
            temporal_groups=tuple((2 * i, 2 * i + 1) for i in range(8)),
            target_group_indices=(2,), sham_group_indices=(6,),
            evidence_group_indices=(0, 1), control_group_indices=(6, 7),
            typed_batch_sha256="a" * 64, score_binding_sha256="b" * 64,
            intervention_layers_1_based=(5, 6), logit_bias=math.log(2.0),
            sham_seed="test-sham-seed")
        branches = plan["branches"]
        self.assertEqual(len(branches), 9)
        self.assertEqual(len({branch["name"] for branch in branches}), 9)
        self.assertEqual(branches[0]["biased_visual_key_positions"], [])
        self.assertEqual([branch["name"].split("_")[0] for branch in branches],
                         ["baseline"] * 3 + ["target"] * 3 + ["sham"] * 3)
        self.assertEqual(branches[3]["biased_visual_key_positions"],
                         branches[4]["biased_visual_key_positions"])
        self.assertEqual(branches[4]["ablated_visual_key_positions"], [0, 1, 2, 3])
        self.assertEqual(branches[5]["ablated_visual_key_positions"], [12, 13, 14, 15])
        self.assertEqual(branches[6]["biased_visual_key_positions"],
                         branches[7]["biased_visual_key_positions"])
        self.assertEqual(branches[7]["ablated_visual_key_positions"], [0, 1, 2, 3])
        self.assertEqual(branches[8]["ablated_visual_key_positions"], [12, 13, 14, 15])
        self.assertTrue(all(branch["score_binding_sha256"] == "b" * 64
                            for branch in branches))
        self.assertTrue(all(branch["typed_batch_sha256"] == "a" * 64
                            for branch in branches))
        self.assertEqual(plan["execution_authorization"], "not_granted")

    def test_condition_plan_digest_detects_branch_mutation_and_omission(self):
        plan = build_qta_nine_branch_plan(
            row_binding_sha256="8" * 64,
            temporal_groups=((0, 1), (2, 3), (4, 5), (6, 7)),
            target_group_indices=(0,), sham_group_indices=(3,),
            evidence_group_indices=(0,), control_group_indices=(2,),
            typed_batch_sha256="c" * 64, score_binding_sha256="d" * 64,
            intervention_layers_1_based=(5, 6), logit_bias=math.log(2.0),
            sham_seed="test-sham-seed")
        changed = json.loads(json.dumps(plan))
        changed["branches"][4]["ablated_visual_key_positions"] = [999]
        with self.assertRaisesRegex(ValueError, "qta_plan_branch_semantics_invalid"):
            validate_qta_nine_branch_plan(changed)
        missing = json.loads(json.dumps(plan))
        missing["branches"].pop()
        with self.assertRaisesRegex(ValueError, "qta_plan_contract_invalid"):
            validate_qta_nine_branch_plan(missing)

    def test_condition_plan_rejects_contradictory_rehashed_metadata(self):
        plan = build_qta_nine_branch_plan(
            row_binding_sha256="7" * 64,
            temporal_groups=((0, 1), (2, 3), (4, 5), (6, 7)),
            target_group_indices=(0,), sham_group_indices=(3,),
            evidence_group_indices=(0,), control_group_indices=(2,),
            typed_batch_sha256="1" * 64, score_binding_sha256="2" * 64,
            intervention_layers_1_based=(5, 6), logit_bias=math.log(2.0),
            sham_seed="test-sham-seed")
        changed = json.loads(json.dumps(plan))
        changed["selected_groups"]["target"] = [1]
        changed.pop("plan_sha256")
        changed["plan_sha256"] = hashlib.sha256(json.dumps(
            changed, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
        with self.assertRaisesRegex(ValueError, "qta_plan_branch_semantics_invalid"):
            validate_qta_nine_branch_plan(changed)

        changed = json.loads(json.dumps(plan))
        changed["branches"][3]["treatment"] = "sham"
        descriptor = {key: value for key, value in changed["branches"][3].items()
                      if key != "branch_sha256"}
        changed["branches"][3]["branch_sha256"] = hashlib.sha256(json.dumps(
            descriptor, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
        changed.pop("plan_sha256")
        changed["plan_sha256"] = hashlib.sha256(json.dumps(
            changed, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
        with self.assertRaisesRegex(ValueError, "qta_plan_branch_semantics_invalid"):
            validate_qta_nine_branch_plan(changed)

    def test_condition_plan_rejects_invalid_group_assignment(self):
        common = dict(
            row_binding_sha256="6" * 64,
            temporal_groups=((0, 1), (2, 3), (4, 5), (6, 7)),
            target_group_indices=(0,), sham_group_indices=(3,),
            evidence_group_indices=(0,), control_group_indices=(2,),
            typed_batch_sha256="e" * 64, score_binding_sha256="f" * 64,
            intervention_layers_1_based=(5, 6), logit_bias=math.log(2.0),
            sham_seed="test-sham-seed")
        with self.assertRaisesRegex(ValueError, "evidence_control_group_contract_invalid"):
            build_qta_nine_branch_plan(**dict(common, control_group_indices=(0,)))
        with self.assertRaisesRegex(ValueError, "target_group_invalid"):
            build_qta_nine_branch_plan(**dict(common, target_group_indices=(4,)))
        with self.assertRaisesRegex(ValueError, "intervention_layers_invalid"):
            build_qta_nine_branch_plan(**dict(common, intervention_layers_1_based=(0, 5)))
        with self.assertRaisesRegex(ValueError, "logit_bias_invalid"):
            build_qta_nine_branch_plan(**dict(common, logit_bias=True))
        with self.assertRaisesRegex(ValueError, "selector_group_budget_mismatch"):
            build_qta_nine_branch_plan(**dict(
                common, target_group_indices=(0, 1), sham_group_indices=(2, 3)))

    def test_condition_plan_validator_rejects_rehashed_oversized_selector(self):
        plan = build_qta_nine_branch_plan(
            row_binding_sha256="5" * 64,
            temporal_groups=((0, 1), (2, 3), (4, 5), (6, 7)),
            target_group_indices=(0,), sham_group_indices=(3,),
            evidence_group_indices=(0,), control_group_indices=(2,),
            typed_batch_sha256="4" * 64, score_binding_sha256="3" * 64,
            intervention_layers_1_based=(5, 6), logit_bias=math.log(2.0),
            sham_seed="test-sham-seed")
        changed = json.loads(json.dumps(plan))
        changed["selected_groups"]["target"] = [0, 1]
        changed["selected_groups"]["sham"] = [2, 3]
        changed.pop("plan_sha256")
        changed["plan_sha256"] = hashlib.sha256(json.dumps(
            changed, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
        with self.assertRaisesRegex(ValueError, "qta_plan_selector_budget_invalid"):
            validate_qta_nine_branch_plan(changed)

    def test_overlap_report_binds_group_and_visual_key_intersections(self):
        overlap = temporal_group_overlap(
            selector_groups=(1, 2), comparison_groups=(0, 1),
            temporal_groups=((10, 11), (12, 13), (14, 15), (16, 17)))
        self.assertEqual(overlap["overlap_group_indices"], [1])
        self.assertEqual(overlap["overlap_group_count"], 1)
        self.assertEqual(overlap["selector_group_fraction"], 0.5)
        self.assertEqual(overlap["comparison_group_fraction"], 0.5)
        self.assertEqual(overlap["overlap_visual_key_count"], 2)
        self.assertEqual(overlap["selector_visual_key_fraction"], 0.5)
        self.assertEqual(overlap["comparison_visual_key_fraction"], 0.5)

    def test_overlap_report_rejects_empty_duplicate_or_unsorted_inputs(self):
        valid = {
            "selector_groups": (1,), "comparison_groups": (2,),
            "temporal_groups": ((10,), (20,), (30,)),
        }
        for field, bad_values in (
                ("selector_groups", ()),
                ("comparison_groups", (1, 1)),
                ("temporal_groups", ((10,), (10,)))):
            arguments = dict(valid)
            arguments[field] = bad_values
            with self.subTest(field=field), self.assertRaises(ValueError):
                temporal_group_overlap(**arguments)
        with self.assertRaisesRegex(ValueError, "temporal_groups_invalid"):
            temporal_group_overlap(
                selector_groups=(4,), comparison_groups=(2,),
                temporal_groups=((10,), (20,), (30,)))
        for invalid_group in ((11, 10), (10, 10), (-1,)):
            with self.subTest(invalid_group=invalid_group), self.assertRaises(ValueError):
                temporal_group_overlap(
                    selector_groups=(0,), comparison_groups=(1,),
                    temporal_groups=(invalid_group, (20,)))
        for invalid_group in ((11, 10), (10, 10), (-1,)):
            with self.subTest(invalid_group=invalid_group), self.assertRaises(ValueError):
                temporal_group_overlap(
                    selector_groups=(0,), comparison_groups=(1,),
                    temporal_groups=(invalid_group, (20,)))

    def test_choice_scorer_uses_full_vocab_log_softmax_at_single_final_row(self):
        logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
        actual = _choice_log_probabilities(logits, [0, 1, 2, 3, 4], torch)
        expected = torch.log_softmax(logits[0, 0], dim=-1)[:5].tolist()
        self.assertEqual(len(actual), 5)
        for got, want in zip(actual, expected, strict=True):
            self.assertAlmostEqual(got, want, places=6)

    def test_choice_scorer_rejects_bad_or_duplicate_option_ids(self):
        logits = torch.zeros((1, 1, 6))
        for ids in ([0, 1, 2, 3, 6], [0, 1, 2, 3, 3], [0, 1, 2, 3]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                _choice_log_probabilities(logits, ids, torch)

    def test_choice_ids_come_from_exact_prompt_continuation(self):
        class ContextTokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                prefix = "assistant\n"
                if text == prefix:
                    return [100, 101]
                if text.startswith(prefix) and len(text) == len(prefix) + 1:
                    return [100, 101, ord(text[-1])]
                raise ValueError("unexpected_text")

        self.assertEqual(_resolve_choice_continuation_ids(
            ContextTokenizer(), "assistant\n"), list(map(ord, "ABCDE")))

        class RetokenizingTokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [100] if text == "prefix" else [99, 102]

        with self.assertRaisesRegex(ValueError, "prefix_retokenized"):
            _resolve_choice_continuation_ids(RetokenizingTokenizer(), "prefix")

        class MultiTokenTokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [100] if text == "prefix" else [100, 102, 103]

        with self.assertRaisesRegex(ValueError, "single_continuation_token"):
            _resolve_choice_continuation_ids(MultiTokenTokenizer(), "prefix")

    def test_multimodal_processor_must_preserve_prefix_and_append_expected_id(self):
        base = torch.tensor([[4, 5, 6]])
        _validate_processed_choice_suffix(
            base, torch.tensor([[4, 5, 6, 10]]), 10, torch)
        with self.assertRaisesRegex(ValueError, "prefix_mismatch"):
            _validate_processed_choice_suffix(
                base, torch.tensor([[4, 5, 7, 10]]), 10, torch)
        with self.assertRaisesRegex(ValueError, "token_mismatch"):
            _validate_processed_choice_suffix(
                base, torch.tensor([[4, 5, 6, 11]]), 10, torch)
        with self.assertRaisesRegex(ValueError, "shape_invalid"):
            _validate_processed_choice_suffix(
                base, torch.tensor([[4, 5, 6, 10, 12]]), 10, torch)

    def test_target_branch_projection_reads_branch_bindings_and_maps_names(self):
        row = {
            "answer_label_order": list("ABCDE"), "answer_token_ids": [0, 1, 2, 3, 4],
            "input_ids_sha256": "a" * 64, "attention_mask_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64, "model_revision": "revision",
            "prompt_sha256": "d" * 64, "input_length": 4, "score_position": 3,
        }
        names = (
            "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
            "intervention_full", "intervention_evidence_ablation",
            "intervention_control_ablation",
        )
        binding = {"branch_bindings": {name: dict(row) for name in names}}
        branch_logits = {}
        for index, name in enumerate((
                "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
                "target_full", "target_evidence_ablation", "target_control_ablation")):
            logits = torch.zeros((1, 1, 8))
            logits[0, 0, :5] = torch.tensor([4.0 - index, 3.0, 2.0, 1.0, 0.0])
            branch_logits[name] = {"logits": logits}
        result = _score_treatment_branches(binding, branch_logits, torch)
        self.assertEqual(result["status"], "bound_logits_adapter_contract_only")
        self.assertEqual(set(result["choice_log_probabilities_by_condition"]), set(names))

        with self.assertRaisesRegex(ValueError, "target_score_branch_set_invalid"):
            _score_treatment_branches(
                binding, {key: value for key, value in branch_logits.items()
                          if key != "target_control_ablation"}, torch)
        with self.assertRaisesRegex(ValueError, "six_condition_branch_bindings_missing"):
            _score_treatment_branches(
                {"bindings": binding["branch_bindings"]}, branch_logits, torch)

    def test_sham_branches_use_the_same_bound_fixed_prediction_scorer(self):
        row = {
            "answer_label_order": list("ABCDE"), "answer_token_ids": [0, 1, 2, 3, 4],
            "input_ids_sha256": "a" * 64, "attention_mask_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64, "model_revision": "revision",
            "prompt_sha256": "d" * 64, "input_length": 4, "score_position": 3,
        }
        binding = {"branch_bindings": {
            name: dict(row) for name in (
                "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
                "intervention_full", "intervention_evidence_ablation",
                "intervention_control_ablation")}}
        branch_logits = {}
        for index, name in enumerate((
                "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
                "sham_full", "sham_evidence_ablation", "sham_control_ablation")):
            logits = torch.zeros((1, 1, 8))
            logits[0, 0, :5] = torch.tensor([4.0 - index, 3.0, 2.0, 1.0, 0.0])
            branch_logits[name] = {"logits": logits}
        result = _score_treatment_branches(
            binding, branch_logits, torch, treatment_prefix="sham")
        self.assertEqual(result["status"], "bound_logits_adapter_contract_only")
        self.assertEqual(set(result["choice_log_probabilities_by_condition"]), {
            "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
            "intervention_full", "intervention_evidence_ablation",
            "intervention_control_ablation"})
        with self.assertRaisesRegex(ValueError, "sham_score_branch_set_invalid"):
            _score_treatment_branches(
                binding, {key: value for key, value in branch_logits.items()
                          if key != "sham_control_ablation"},
                torch, treatment_prefix="sham")


class PrivateOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.private = self.root / "private_nextgqa"
        self.private.mkdir(mode=0o700)
        os.chmod(self.private, 0o700)

    def tearDown(self):
        self.temp.cleanup()

    def test_new_dedicated_directory_and_file_are_private_at_creation(self):
        old_umask = os.umask(0o022)
        try:
            target = _prepare_private_output(
                self.private / "run-1" / "summary.json", self.private)
            output = target.display_path
            self.assertEqual(stat.S_IMODE(output.parent.stat().st_mode), 0o700)
            _write_private_json(target, {"ok": True})
            _close_private_output(target)
        finally:
            os.umask(old_umask)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        self.assertEqual(json.loads(output.read_text()), {"ok": True})

    def test_outside_root_symlink_and_nested_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "output_must_be_under_private"):
            _prepare_private_output(self.root / "outside" / "summary.json", self.private)
        outside = self.root / "outside_dir"
        outside.mkdir()
        (self.private / "linked").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "already_exists"):
            _prepare_private_output(
                self.private / "linked" / "summary.json", self.private)
        with self.assertRaisesRegex(ValueError, "fresh_private_run_directory"):
            _prepare_private_output(
                self.private / "nested" / "run" / "summary.json", self.private)

    def test_private_root_symlink_is_rejected(self):
        real = self.root / "real_private"
        real.mkdir(mode=0o700)
        os.chmod(real, 0o700)
        linked_root = self.root / "private_link"
        linked_root.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "private_output_root_mode_invalid"):
            _prepare_private_output(linked_root / "run" / "summary.json", linked_root)

    def test_existing_run_directory_is_not_mutated_or_overwritten(self):
        shared = self.private / "existing"
        shared.mkdir(mode=0o755)
        os.chmod(shared, 0o755)
        with self.assertRaisesRegex(ValueError, "already_exists"):
            _prepare_private_output(shared / "summary.json", self.private)
        self.assertEqual(stat.S_IMODE(shared.stat().st_mode), 0o755)

    def test_serialization_failure_removes_partial_private_output(self):
        target = _prepare_private_output(
            self.private / "run-fail" / "summary.json", self.private)
        with self.assertRaises(TypeError):
            _write_private_json(target, {"not_json": object()})
        _close_private_output(target)
        output = target.display_path
        self.assertFalse(output.exists())
        self.assertFalse(output.parent.exists())

    def test_output_open_failure_removes_empty_run_directory(self):
        target = _prepare_private_output(
            self.private / "run-open-fail" / "summary.json", self.private)
        with patch("smoke_qwen_question_selector_intervention.os.open",
                   side_effect=OSError("synthetic_open_failure")):
            with self.assertRaisesRegex(OSError, "synthetic_open_failure"):
                _write_private_json(target, {"ok": True})
        _close_private_output(target)
        self.assertFalse(target.display_path.parent.exists())

    def test_existing_output_is_refused_without_mutation(self):
        target = _prepare_private_output(
            self.private / "run-existing" / "summary.json", self.private)
        output = target.display_path
        output.write_text("preserve", encoding="utf-8")
        os.chmod(output, 0o600)
        with self.assertRaises(FileExistsError):
            _write_private_json(target, {"new": True})
        self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
        _close_private_output(target)


if __name__ == "__main__":
    unittest.main()
