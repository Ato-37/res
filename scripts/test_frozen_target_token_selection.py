"""CPU tests for fixed target-relative token selection arithmetic."""

from dataclasses import FrozenInstanceError, replace
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_target_token_selection import (  # noqa: E402
    freeze_token_selection,
    score_frozen_selection,
)
from frozen_teacher_forced_score import (  # noqa: E402
    freeze_generation_tail,
    score_frozen_target,
)


class FrozenTargetTokenSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = freeze_generation_tail(
            [1, 2, 3, 7, 8, 9], prompt_length=3, eos_token_id=9
        )
        rows = [[0.0] * 12 for _ in range(6)]
        rows[2][7] = 5.0
        rows[3][8] = 4.0
        rows[4][9] = 3.0
        cls.full_score = score_frozen_target([rows], cls.target)

    def test_fixed_selection_aggregates_exact_target_relative_positions(self):
        frozen = freeze_token_selection(self.target, (0, 1))
        selected = score_frozen_selection(self.full_score, frozen)
        values = self.full_score["token_logprobs_nats"]
        self.assertEqual(frozen.indices, (0, 1))
        self.assertEqual(selected.count, 2)
        self.assertAlmostEqual(selected.sum_nats, values[0] + values[1])
        self.assertAlmostEqual(selected.mean_nats_per_token, (values[0] + values[1]) / 2)
        self.assertEqual(selected.target_sha256, self.full_score["target_sha256"])
        self.assertEqual(selected.target_length, 3)
        self.assertFalse(selected.semantic_selection_verified)
        # Selection does not mutate or replace the full frozen-target scoring.
        self.assertEqual(self.full_score["target_token_count"], 3)
        self.assertTrue(self.full_score["terminal_seen"])

    def test_wrong_selection_control_changes_selected_score_without_changing_target(self):
        first = score_frozen_selection(
            self.full_score, freeze_token_selection(self.target, (0,))
        )
        second = score_frozen_selection(
            self.full_score, freeze_token_selection(self.target, (1,))
        )
        self.assertEqual(first.target_sha256, second.target_sha256)
        self.assertEqual(first.target_length, second.target_length)
        self.assertNotAlmostEqual(first.mean_nats_per_token, second.mean_nats_per_token)
        self.assertEqual(self.full_score["target_token_count"], 3)

    def test_selection_digest_is_stable_and_selection_is_immutable(self):
        first = freeze_token_selection(self.target, (0, 1))
        second = freeze_token_selection(self.target, [0, 1])
        self.assertEqual(first, second)
        with self.assertRaises(FrozenInstanceError):
            first.indices = (0,)
        with self.assertRaises(ValueError):
            score_frozen_selection(
                self.full_score, replace(first, selection_sha256="0" * 64)
            )

    def test_freeze_rejects_empty_unsorted_duplicate_out_of_range_and_noninteger(self):
        for values in ((), (1, 0), (0, 0), (-1,), (3,)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                freeze_token_selection(self.target, values)
        for values in ((True,), (1.0,), ("1",)):
            with self.subTest(values=values), self.assertRaises(TypeError):
                freeze_token_selection(self.target, values)
        with self.assertRaises(TypeError):
            freeze_token_selection(self.target, "0")

    def test_score_rejects_target_digest_or_length_mismatch(self):
        frozen = freeze_token_selection(self.target, (0, 1))
        altered_digest = dict(self.full_score, target_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "digest"):
            score_frozen_selection(altered_digest, frozen)
        altered_length = dict(self.full_score, target_token_count=2)
        with self.assertRaisesRegex(ValueError, "length"):
            score_frozen_selection(altered_length, frozen)

    def test_score_rejects_truncated_or_invalid_selected_per_token_values(self):
        frozen = freeze_token_selection(self.target, (0, 1))
        with self.assertRaisesRegex(ValueError, "per-token score length"):
            score_frozen_selection(
                dict(self.full_score, token_logprobs_nats=(-1.0, -2.0)), frozen
            )
        for invalid in (float("nan"), float("inf"), True, "-1"):
            bad_values = list(self.full_score["token_logprobs_nats"])
            bad_values[1] = invalid
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                score_frozen_selection(
                    dict(self.full_score, token_logprobs_nats=tuple(bad_values)), frozen
                )
        # Non-finite scores at unselected target positions are irrelevant to this
        # explicitly selected arithmetic and do not get silently included.
        unselected_bad = dict(
            self.full_score,
            token_logprobs_nats=(
                self.full_score["token_logprobs_nats"][0],
                self.full_score["token_logprobs_nats"][1],
                math.nan,
            ),
        )
        self.assertTrue(math.isfinite(score_frozen_selection(unselected_bad, frozen).sum_nats))

    def test_malformed_frozen_selection_object_fails_closed(self):
        frozen = freeze_token_selection(self.target, (0, 1))
        with self.assertRaises(ValueError):
            score_frozen_selection(self.full_score, replace(frozen, indices=(1, 0)))
        with self.assertRaises(ValueError):
            score_frozen_selection(self.full_score, replace(frozen, target_length=2))
        with self.assertRaises(TypeError):
            score_frozen_selection(self.full_score, replace(frozen, indices=(False, 1)))


if __name__ == "__main__":
    unittest.main()
