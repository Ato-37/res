import unittest
from fractions import Fraction

from build_nextgqa_clock_anchor_plan import build_anchor_plan, select_anchor_points


class BuildNextgqaClockAnchorPlanTests(unittest.TestCase):
    def test_fixed_fraction_anchors_are_disjoint_and_split_before_observations(self):
        fit, heldout = select_anchor_points(list(range(100, 121)), Fraction(1, 1))
        self.assertEqual([row["anchor_id"] for row in fit],
                         ["fit_begin", "fit_midpoint", "fit_end"])
        self.assertEqual([row["decoder_pts_seconds"] for row in fit], ["1", "10", "19"])
        self.assertEqual([row["anchor_id"] for row in heldout],
                         ["heldout_quarter", "heldout_three_quarter"])
        self.assertEqual([row["decoder_pts_seconds"] for row in heldout], ["5", "15"])
        self.assertEqual({row["decoder_pts_seconds"] for row in fit}.intersection(
            row["decoder_pts_seconds"] for row in heldout), set())

    def test_ties_choose_earlier_frame_deterministically(self):
        fit, heldout = select_anchor_points(list(range(0, 101, 10)), Fraction(1, 10))
        by_id = {row["anchor_id"]: row["decoder_pts_seconds"] for row in fit + heldout}
        self.assertEqual(by_id["fit_begin"], "0")
        self.assertEqual(by_id["heldout_quarter"], "2")
        self.assertEqual(by_id["heldout_three_quarter"], "7")

    def test_insufficient_frames_or_duplicate_pts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "anchor_frame_count_invalid"):
            select_anchor_points([0, 1, 2, 3], Fraction(1, 1))
        with self.assertRaisesRegex(ValueError, "anchor_pts_not_strictly_increasing"):
            select_anchor_points([0, 1, 2, 2, 4, 5], Fraction(1, 1))

    def test_nearest_targets_must_map_to_five_distinct_frames(self):
        with self.assertRaisesRegex(ValueError, "anchor_targets_not_distinct"):
            select_anchor_points([0, 10, 11, 12, 100], Fraction(1, 1))

    def test_zero_time_base_denominator_fails_with_sanitized_error(self):
        rows = [{"review_token": f"tok{i:02d}" + "x" * 19,
                 "source_media_sha256": f"{i + 1:064x}",
                 "pts": list(range(100, 121)), "time_base": (1, 0)}
                for i in range(3)]
        with self.assertRaisesRegex(ValueError, "anchor_time_base_invalid"):
            build_anchor_plan(rows)

    def test_builder_emits_capture_converter_contract_without_authority(self):
        rows = [{"review_token": f"tok{i:02d}" + "x" * 19,
                 "source_media_sha256": f"{i + 1:064x}",
                 "pts": list(range(100, 121)), "time_base": (1, 10)}
                for i in range(3)]
        plan = build_anchor_plan(rows)
        self.assertEqual(set(plan), {"schema_version", "purpose", "status",
                                     "execution_authorization", "records"})
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(plan["purpose"], "predeclared_html5_clock_pts_anchor_plan")
        self.assertEqual(plan["status"], "predeclared_frozen")  # schedule only; see design note
        self.assertEqual(plan["execution_authorization"], "not_granted")
        self.assertTrue(all(len(row["fit_anchors"]) == 3
                            and len(row["heldout_anchors"]) == 2
                            for row in plan["records"]))


if __name__ == "__main__":
    unittest.main()
