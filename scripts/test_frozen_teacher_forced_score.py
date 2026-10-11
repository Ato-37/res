"""Tests for raw-ID frozen teacher-forced scoring (no model/GPU required)."""

import math
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_teacher_forced_score import (  # noqa: E402
    freeze_generation_tail,
    score_frozen_target,
)


class FrozenTeacherForcedScoreTests(unittest.TestCase):
    PROMPT_LENGTH = 3
    RAW_BASELINE_SEQUENCE = [1, 2, 3, 7, 8, 9]
    EOS_ID = 9
    VOCAB_SIZE = 12

    def test_finite_logits_with_unrepresentable_logprob_fail_closed(self):
        rows = [[1e308] * self.VOCAB_SIZE for _ in range(6)]
        rows[2][7] = -1e308
        with self.assertRaisesRegex(ValueError, "finite scalar precision"):
            score_frozen_target([rows], self.target)

    def test_float64_tensor_preserves_resolvable_logit_differences(self):
        logits = torch.full((1, 6, 12), 1e16, dtype=torch.float64)
        logits[0, 2, 0] += 2
        result = score_frozen_target(logits, self.target)
        self.assertAlmostEqual(result["token_logprobs_nats"][0], -math.log(11 + math.exp(2)))

    @classmethod
    def setUpClass(cls):
        # Correct target predictions are at rows P-1, P, P+1 = 2, 3, 4.
        cls.logits = [[0.0] * cls.VOCAB_SIZE for _ in range(6)]
        cls.logits[2][7] = 5.0
        cls.logits[3][8] = 5.0
        cls.logits[4][9] = 5.0
        cls.target = freeze_generation_tail(
            cls.RAW_BASELINE_SEQUENCE,
            prompt_length=cls.PROMPT_LENGTH,
            eos_token_id=cls.EOS_ID,
        )
        cls.result = score_frozen_target([cls.logits], cls.target)

    def test_raw_tail_is_frozen_once_and_provenance_limit_is_explicit(self):
        self.assertEqual(self.target.token_ids, (7, 8, 9))
        self.assertEqual(self.target.source_policy, "generation_sequence_tail")
        self.assertTrue(self.target.caller_provenance_unverified)
        self.assertEqual(self.target.target_sha256, freeze_generation_tail(
            tuple(self.RAW_BASELINE_SEQUENCE),
            prompt_length=self.PROMPT_LENGTH,
            eos_token_id=self.EOS_ID,
        ).target_sha256)
        mutated_source = list(self.RAW_BASELINE_SEQUENCE)
        frozen = freeze_generation_tail(
            mutated_source, prompt_length=self.PROMPT_LENGTH, eos_token_id=self.EOS_ID
        )
        old_hash = frozen.target_sha256
        mutated_source[-1] = 10
        self.assertEqual(frozen.token_ids, (7, 8, 9))
        self.assertEqual(frozen.target_sha256, old_hash)
        with self.assertRaises(FrozenInstanceError):
            frozen.token_ids = (7, 8)

    def test_teacher_forced_alignment_and_both_mechanical_score_units(self):
        result = self.result
        self.assertEqual(result["prediction_logit_rows"], (2, 3, 4))
        for value in result["token_logprobs_nats"]:
            self.assertAlmostEqual(value, self._preferred_logprob())
        self.assertAlmostEqual(result["all_generated_sum_nats"], 3 * self._preferred_logprob())
        self.assertAlmostEqual(
            result["all_generated_mean_nats_per_token"], self._preferred_logprob()
        )
        self.assertEqual(result["score_unit"], "nats")

    def test_torch_ids_and_logits_match_nested_list_inputs(self):
        tensor_target = freeze_generation_tail(
            torch.tensor(self.RAW_BASELINE_SEQUENCE, dtype=torch.int64),
            prompt_length=self.PROMPT_LENGTH,
            eos_token_id=self.EOS_ID,
        )
        tensor_result = score_frozen_target(
            torch.tensor([self.logits], dtype=torch.float32), tensor_target
        )
        self.assertEqual(tensor_target.token_ids, self.target.token_ids)
        self.assertEqual(tensor_target.target_sha256, self.target.target_sha256)
        self.assertEqual(tensor_result["prediction_logit_rows"], self.result["prediction_logit_rows"])
        for key in (
            "all_generated_sum_nats",
            "all_generated_mean_nats_per_token",
            "content_sum_nats",
            "content_mean_nats_per_token",
        ):
            self.assertAlmostEqual(tensor_result[key], self.result[key])

    def test_extreme_equal_logits_use_stable_log_softmax(self):
        extreme_logits = [[[1e16] * self.VOCAB_SIZE for _ in range(6)]]
        result = score_frozen_target(extreme_logits, self.target)
        expected = -math.log(self.VOCAB_SIZE)
        for value in result["token_logprobs_nats"]:
            self.assertAlmostEqual(value, expected, places=12)
        self.assertAlmostEqual(result["all_generated_mean_nats_per_token"], expected, places=12)

    def test_terminal_token_provenance_reports_full_and_content_scores(self):
        result = self.result
        self.assertEqual(result["eos_token_id"], self.EOS_ID)
        self.assertTrue(result["terminal_seen"])
        self.assertEqual(result["termination_status"], "terminal_seen")
        self.assertTrue(result["all_generated_score_includes_terminal"])
        self.assertTrue(result["content_score_excludes_terminal"])
        self.assertEqual(result["content_token_count"], 2)
        self.assertAlmostEqual(result["content_sum_nats"], 2 * self._preferred_logprob())
        self.assertAlmostEqual(
            result["content_mean_nats_per_token"], self._preferred_logprob()
        )

    def test_wrong_offset_rows_change_score(self):
        correct = self.result["all_generated_sum_nats"]
        # Explicit negative-control arithmetic: starting at P instead of P-1
        # shifts all target IDs one prediction row later.
        wrong_rows = (3, 4, 5)
        wrong = sum(
            self._logprob(self.logits[row], token_id)
            for row, token_id in zip(wrong_rows, self.target.token_ids)
        )
        self.assertNotEqual(wrong_rows, self.result["prediction_logit_rows"])
        self.assertNotAlmostEqual(wrong, correct)

    def test_truncated_teacher_forcing_rows_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "sequence length"):
            score_frozen_target([self.logits[:-1]], self.target)
        truncated_target = freeze_generation_tail(
            self.RAW_BASELINE_SEQUENCE[:-1],
            prompt_length=self.PROMPT_LENGTH,
            eos_token_id=self.EOS_ID,
        )
        with self.assertRaisesRegex(ValueError, "sequence length"):
            score_frozen_target([self.logits], truncated_target)

    def test_selected_nonfinite_logits_and_bad_shapes_fail_closed(self):
        nonfinite = [row[:] for row in self.logits]
        nonfinite[3][8] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            score_frozen_target([nonfinite], self.target)
        nonselected_nonfinite = [row[:] for row in self.logits]
        nonselected_nonfinite[0][0] = float("nan")
        self.assertEqual(
            score_frozen_target([nonselected_nonfinite], self.target)["all_generated_sum_nats"],
            self.result["all_generated_sum_nats"],
        )
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            score_frozen_target([self.logits, self.logits], self.target)
        wrong_width = [row[:] for row in self.logits]
        wrong_width[2].pop()
        with self.assertRaisesRegex(ValueError, "vocabulary width"):
            score_frozen_target([wrong_width], self.target)

    def test_nonterminal_generation_is_marked_termination_unverified(self):
        no_eos = freeze_generation_tail(
            [1, 2, 3, 7, 8], prompt_length=self.PROMPT_LENGTH, eos_token_id=self.EOS_ID
        )
        result = score_frozen_target([self.logits[:5]], no_eos)
        self.assertFalse(result["terminal_seen"])
        self.assertEqual(result["termination_status"], "termination_unverified")
        self.assertFalse(result["all_generated_score_includes_terminal"])
        self.assertEqual(result["all_generated_sum_nats"], result["content_sum_nats"])
        self.assertEqual(
            result["all_generated_mean_nats_per_token"],
            result["content_mean_nats_per_token"],
        )

    def test_terminal_only_target_has_no_content_mean_and_empty_target_rejected(self):
        terminal_only = freeze_generation_tail(
            [1, 2, 3, self.EOS_ID], prompt_length=self.PROMPT_LENGTH,
            eos_token_id=self.EOS_ID,
        )
        result = score_frozen_target([self.logits[:4]], terminal_only)
        self.assertEqual(result["target_token_count"], 1)
        self.assertEqual(result["content_token_count"], 0)
        self.assertEqual(result["content_sum_nats"], 0.0)
        self.assertIsNone(result["content_mean_nats_per_token"])
        with self.assertRaisesRegex(ValueError, "at least one token"):
            freeze_generation_tail([1, 2, 3], prompt_length=3, eos_token_id=self.EOS_ID)

    def test_malformed_ids_and_multi_eos_configuration_are_not_guessed(self):
        with self.assertRaises(TypeError):
            freeze_generation_tail([1, 2, 3, True], prompt_length=3)
        with self.assertRaises(TypeError):
            freeze_generation_tail(self.RAW_BASELINE_SEQUENCE, prompt_length=True)
        with self.assertRaises(TypeError):
            freeze_generation_tail(self.RAW_BASELINE_SEQUENCE, prompt_length=3, eos_token_id=[9, 10])
        with self.assertRaises(TypeError):
            freeze_generation_tail(torch.tensor([1, 2, 3, 4.0]), prompt_length=3)
        with self.assertRaises(ValueError):
            freeze_generation_tail(torch.tensor([[1, 2, 3, 4]]), prompt_length=3)
        with self.assertRaises(ValueError):
            freeze_generation_tail(self.RAW_BASELINE_SEQUENCE, prompt_length=0)

    @staticmethod
    def _logprob(row, token_id):
        maximum = max(row)
        normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in row))
        return row[token_id] - normalizer

    def _preferred_logprob(self):
        return self._logprob(self.logits[2], 7)


if __name__ == "__main__":
    unittest.main()
