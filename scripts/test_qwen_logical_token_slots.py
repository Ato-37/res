"""CPU tests for logical token-slot mapping across append-only cache calls."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_logical_token_slots import (  # noqa: E402
    logical_slot_to_local,
    replace_logical_slot,
)


class _TrackedList(list):
    def __init__(self, values):
        super().__init__(values)
        self.writes = 0

    def __setitem__(self, index, value):
        self.writes += 1
        super().__setitem__(index, value)


class QwenLogicalTokenSlotTests(unittest.TestCase):
    # Fixed, outcome-independent toy: prompt slots 0..4, visual span [1,4),
    # answer slots 5 and 6. Values stand for activations at logical positions.
    PROMPT_LENGTH = 5
    VISUAL_SPAN = (1, 4)
    ANSWER_SLOT = 5
    DONOR = "patched-answer-activation"

    def test_full_and_split_decode_patch_same_logical_slot_once(self):
        self.assertLess(0, self.VISUAL_SPAN[0])
        self.assertLess(self.VISUAL_SPAN[1], self.PROMPT_LENGTH)
        self.assertEqual(self.PROMPT_LENGTH, self.ANSWER_SLOT)
        full = _TrackedList(range(7))
        full_local = replace_logical_slot(
            full,
            self.DONOR,
            logical_slot=self.ANSWER_SLOT,
            query_origin=0,
            query_length=7,
            actual_cache_length=0,
            expected_cache_length=0,
        )
        self.assertEqual(full_local, self.ANSWER_SLOT)
        self.assertEqual(full.writes, 1)

        # On prompt prefill, target slot 5 is not in query [0,5): skip this
        # call instead of treating absolute slot 5 as a local tensor offset.
        prefill = _TrackedList(range(self.PROMPT_LENGTH))
        with self.assertRaisesRegex(ValueError, "outside this query window"):
            replace_logical_slot(
                prefill,
                self.DONOR,
                logical_slot=self.ANSWER_SLOT,
                query_origin=0,
                query_length=self.PROMPT_LENGTH,
                actual_cache_length=0,
                expected_cache_length=0,
            )
        self.assertEqual(prefill, list(range(self.PROMPT_LENGTH)))
        self.assertEqual(prefill.writes, 0)

        # Once decoder query [5,6) arrives with a cache of length 5, the same
        # logical answer slot maps to its sole local element 0.
        decode = _TrackedList([self.ANSWER_SLOT])
        decode_local = replace_logical_slot(
            decode,
            self.DONOR,
            logical_slot=self.ANSWER_SLOT,
            query_origin=self.PROMPT_LENGTH,
            query_length=1,
            actual_cache_length=self.PROMPT_LENGTH,
            expected_cache_length=self.PROMPT_LENGTH,
        )
        self.assertEqual(decode_local, 0)
        self.assertEqual(decode.writes, 1)
        self.assertEqual(prefill + decode + [6], full)

    def test_invalid_prompt_absolute_slot_on_decode_fails_before_mutation(self):
        decode = _TrackedList([5])
        with self.assertRaisesRegex(ValueError, "outside this query window"):
            replace_logical_slot(
                decode,
                self.DONOR,
                logical_slot=4,
                query_origin=5,
                query_length=1,
                actual_cache_length=5,
                expected_cache_length=5,
            )
        self.assertEqual(decode, [5])
        self.assertEqual(decode.writes, 0)

    def test_zero_origin_with_nonempty_cache_and_stale_length_fail_before_mutation(self):
        for origin, actual, expected, message in (
            (0, 5, 5, "query origin"),
            (5, 6, 5, "actual cache length"),
        ):
            with self.subTest(origin=origin, actual=actual):
                decode = _TrackedList([5])
                with self.assertRaisesRegex(ValueError, message):
                    replace_logical_slot(
                        decode,
                        self.DONOR,
                        logical_slot=5,
                        query_origin=origin,
                        query_length=1,
                        actual_cache_length=actual,
                        expected_cache_length=expected,
                    )
                self.assertEqual(decode, [5])
                self.assertEqual(decode.writes, 0)

    def test_strict_integer_and_nonempty_query_validation(self):
        base = {
            "query_origin": 5,
            "query_length": 1,
            "actual_cache_length": 5,
            "expected_cache_length": 5,
        }
        for field, value in (("logical_slot", True), ("query_origin", 5.0), ("actual_cache_length", False)):
            args = {"logical_slot": 5, **base}
            args[field] = value
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    logical_slot_to_local(**args)
        for query_length in (0, -1):
            with self.subTest(query_length=query_length):
                with self.assertRaisesRegex(ValueError, "query_length"):
                    logical_slot_to_local(5, **{**base, "query_length": query_length})
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            logical_slot_to_local(-1, **base)

    def test_local_minus_one_has_distinct_full_and_decode_identity(self):
        # Legacy negative indexing denotes last-in-this-call, not a stable
        # full-sequence logical token identity.
        full_origin, full_query_length = 0, 7
        decode_origin, decode_query_length = 5, 1
        full_minus_one_logical = full_origin + full_query_length - 1
        decode_minus_one_logical = decode_origin + decode_query_length - 1
        self.assertEqual(full_minus_one_logical, 6)
        self.assertEqual(decode_minus_one_logical, 5)
        self.assertNotEqual(full_minus_one_logical, decode_minus_one_logical)

    def test_recipient_shape_fails_before_mutation(self):
        wrong_length = _TrackedList([5, 6])
        with self.assertRaisesRegex(ValueError, "sequence length"):
            replace_logical_slot(
                wrong_length,
                self.DONOR,
                logical_slot=5,
                query_origin=5,
                query_length=1,
                actual_cache_length=5,
                expected_cache_length=5,
            )
        self.assertEqual(wrong_length, [5, 6])
        self.assertEqual(wrong_length.writes, 0)


if __name__ == "__main__":
    unittest.main()
