from __future__ import annotations

import math
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import torch

from qwen_internal_attention_ablation import compose_video_key_bias_and_ablation
from smoke_qwen_question_selector_intervention import (
    _choice_log_probabilities, _close_private_output, _prepare_private_output, _write_private_json,
    build_smoke_condition_plan,
)


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
        target, sham = (1, 2), (7, 8)
        evidence, control = (10, 11), (20, 21)
        plan = build_smoke_condition_plan(
            target_positions=target, sham_positions=sham,
            evidence_positions=evidence, control_positions=control)
        self.assertEqual(len(plan), 9)
        self.assertEqual(len({branch["name"] for branch in plan}), 9)
        self.assertEqual(plan[0], {"name": "baseline_full", "bias": (), "ablation": ()})
        self.assertEqual([branch["name"].split("_")[0] for branch in plan],
                         ["baseline"] * 3 + ["target"] * 3 + ["sham"] * 3)
        self.assertEqual(plan[3]["bias"], plan[4]["bias"])
        self.assertEqual(plan[4]["ablation"], evidence)
        self.assertEqual(plan[5]["ablation"], control)
        self.assertEqual(plan[6]["bias"], plan[7]["bias"])

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
