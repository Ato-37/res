from __future__ import annotations

import hashlib
import json
import math
import unittest

from qta_v4_mask_specificity_utils import (
    FIXED_SEED, PROTOCOL_ID, MaskSpecificityInputError,
    choose_translation, eligible_translations, finite_cohort_summary,
    specificity_descriptives,
)


class MaskSpecificityUtilityTests(unittest.TestCase):
    def test_nonwrapping_translation_preserves_run_shape_and_token_count(self):
        mask = (0, 1)
        counts = (1, 2, 3, 1, 2, 3, 1, 2, 3)
        offsets = eligible_translations(mask, counts)
        self.assertEqual(offsets, (3, 6))
        for offset in offsets:
            translated = tuple(group + offset for group in mask)
            self.assertEqual(len(translated), len(mask))
            self.assertEqual(tuple(b-a for a,b in zip(translated, translated[1:])),
                             tuple(b-a for a,b in zip(mask, mask[1:])))
            self.assertEqual(sum(counts[i] for i in translated),
                             sum(counts[i] for i in mask))
            self.assertGreaterEqual(min(translated), 0)
            self.assertLess(max(translated), len(counts))

    def test_full_mask_has_no_go_and_no_eligible_offset(self):
        self.assertEqual(eligible_translations((0, 1, 2), (4, 4, 4)), ())
        self.assertIsNone(choose_translation((0, 1, 2), (4, 4, 4),
                                             video_id="synthetic-video", qid="synthetic-qid"))

    def test_no_equal_token_cardinality_shift_is_structurally_unavailable(self):
        counts = (2, 1, 3, 4)
        self.assertEqual(eligible_translations((1,), counts), ())
        self.assertIsNone(choose_translation((1,), counts,
                                             video_id="synthetic-video", qid="synthetic-qid"))

    def test_hash_min_selection_is_fixed_and_input_order_independent(self):
        mask = (0, 1)
        counts = (1, 2, 3, 1, 2, 3, 1, 2, 3)
        offsets = eligible_translations(mask, counts)
        self.assertGreaterEqual(len(offsets), 2)
        video_id, qid = "synthetic-video-17", "synthetic-qid-4"
        got = choose_translation(mask, counts, video_id=video_id, qid=qid)
        expected = min(offsets, key=lambda s: (hashlib.sha256(json.dumps(
            [PROTOCOL_ID, str(FIXED_SEED), video_id, qid, str(s)],
            ensure_ascii=False, separators=(",", ":")).encode("utf-8")).digest(), s))
        self.assertEqual(got, expected)
        self.assertEqual(got, choose_translation(tuple(reversed(mask)), counts,
                                                  video_id=video_id, qid=qid))
        self.assertEqual(got, choose_translation(mask, counts,
                                                  video_id=video_id, qid=qid,
                                                  protocol_id=PROTOCOL_ID,
                                                  fixed_seed=FIXED_SEED))

    def test_geometry_rejects_bool_duplicate_empty_and_out_of_range_inputs(self):
        invalid_cases = [
            ((True,), (1, 1)), ((0, 0), (1,)), ((), (1,)), ((2,), (1, 1)),
            ((0,), ()), ((0,), (1, True)), ((0,), (1, 0)),
        ]
        for mask, counts in invalid_cases:
            with self.subTest(mask=mask, counts=counts):
                with self.assertRaises(MaskSpecificityInputError):
                    eligible_translations(mask, counts)

    def test_selection_rejects_ambiguous_identifiers_and_boolean_seed(self):
        kw = {"mask_groups": (0,), "group_token_counts": (1, 1),
              "video_id": "synthetic-video", "qid": "synthetic-qid"}
        for args in ({**kw, "video_id": ""}, {**kw, "qid": "bad\0id"},
                     {**kw, "fixed_seed": True}, {**kw, "protocol_id": "bad\0protocol"}):
            with self.subTest(args=args), self.assertRaises(MaskSpecificityInputError):
                choose_translation(**args)

    def test_finite_cohort_summary_is_descriptive_only(self):
        report = finite_cohort_summary((-1.0, 0.0, 3.0))
        self.assertEqual(report["n"], 3)
        self.assertAlmostEqual(report["mean"], 2/3)
        self.assertAlmostEqual(report["sample_sd"], math.sqrt(13/3))
        self.assertEqual((report["minimum"], report["maximum"]), (-1.0, 3.0))
        self.assertEqual((report["positive"], report["negative"], report["zero"]), (1, 1, 1))
        self.assertIsNone(finite_cohort_summary((2.5,))["sample_sd"])

    def test_paired_specificity_arithmetic(self):
        report = specificity_descriptives((1.0, 2.0, -1.0), (0.0, 1.0, -2.0))
        self.assertEqual(report["D_annotation_union"]["mean"], 2/3)
        self.assertEqual(report["D_placebo_translation"]["mean"], -1/3)
        self.assertEqual(report["specificity_difference"]["mean"], 1.0)
        self.assertEqual(report["specificity_difference"]["sample_sd"], 0.0)
        with self.assertRaisesRegex(MaskSpecificityInputError, "length_mismatch"):
            specificity_descriptives((1.0,), (1.0, 2.0))

    def test_finite_arithmetic_rejects_missing_nonfinite_bool_and_overflow(self):
        for values in ((), (math.nan,), (math.inf,), (True,), ("1",), (1e308, -1e308, 1e308)):
            with self.subTest(values=values), self.assertRaises(MaskSpecificityInputError):
                finite_cohort_summary(values)
        with self.assertRaises(MaskSpecificityInputError):
            specificity_descriptives((1.0, math.nan), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
