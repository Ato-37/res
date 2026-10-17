import unittest

from annotation_clock_calibration import (
    evaluate_affine_calibration,
    map_annotation_intervals_to_pts,
)


class AnnotationClockCalibrationTests(unittest.TestCase):
    def test_review_intervals_are_inverse_mapped_to_pts_exactly(self):
        # player_time = 3/2 * pts_time + 1/4
        self.assertEqual(map_annotation_intervals_to_pts(
            [["7/4", "13/4"], [4, 5]], "3/2", "1/4"),
            [["1", "2"], ["5/2", "19/6"]])

    def test_interval_mapping_rejects_bad_clock_or_ranges(self):
        for intervals, slope, intercept in (
                ([], "1", "0"), ([[1, 1]], "1", "0"),
                ([[-1, 1]], "1", "0"), ([[1, 2]], "0", "0"),
                ([[1, 2]], True, "0"), ([[float("nan"), 2]], "1", "0"),
                ([["1/0", 2]], "1", "0"), ([[1, 2]], "1", "3"),
                ([["1/3", "2/3"]], "1", "0")):
            with self.subTest(intervals=intervals, slope=slope, intercept=intercept):
                with self.assertRaises(ValueError):
                    map_annotation_intervals_to_pts(intervals, slope, intercept)

    def test_affine_offset_and_drift_fit_then_independent_heldout_check(self):
        fit = [
            {"anchor_id": "fit-1", "pts_time": "0", "annotation_time": "5/4"},
            {"anchor_id": "fit-2", "pts_time": "2", "annotation_time": "13/4"},
            {"anchor_id": "fit-3", "pts_time": "4", "annotation_time": "21/4"},
        ]
        heldout = [
            {"anchor_id": "hold-1", "pts_time": "1", "annotation_time": "9/4"},
            {"anchor_id": "hold-2", "pts_time": "3", "annotation_time": "17/4"},
        ]
        result = evaluate_affine_calibration(fit, heldout, "1/100")
        self.assertEqual(result["slope"], "1")
        self.assertEqual(result["intercept"], "5/4")
        self.assertEqual(result["heldout_max_abs_residual"], "0")
        self.assertTrue(result["heldout_within_tolerance"])
        self.assertEqual(result["uncertainty_status"], "not_evaluated")
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_heldout_outlier_is_reported_without_refitting(self):
        fit = [
            {"anchor_id": "fit-1", "pts_time": "0", "annotation_time": "0"},
            {"anchor_id": "fit-2", "pts_time": "1", "annotation_time": "1"},
            {"anchor_id": "fit-3", "pts_time": "2", "annotation_time": "2"},
        ]
        heldout = [{"anchor_id": "hold-1", "pts_time": "3", "annotation_time": "4"}]
        result = evaluate_affine_calibration(fit, heldout, "1/2")
        self.assertEqual(result["slope"], "1")
        self.assertEqual(result["intercept"], "0")
        self.assertEqual(result["heldout_max_abs_residual"], "1")
        self.assertFalse(result["heldout_within_tolerance"])

    def test_noncollinear_fit_uses_all_fit_anchors_and_inclusive_tolerance(self):
        fit = [
            {"anchor_id": "fit-1", "pts_time": "0", "annotation_time": "0"},
            {"anchor_id": "fit-2", "pts_time": "1", "annotation_time": "1"},
            {"anchor_id": "fit-3", "pts_time": "2", "annotation_time": "3"},
        ]
        heldout = [{"anchor_id": "hold-1", "pts_time": "3", "annotation_time": "9/2"}]
        at_boundary = evaluate_affine_calibration(fit, heldout, "1/6")
        self.assertEqual(at_boundary["slope"], "3/2")
        self.assertEqual(at_boundary["intercept"], "-1/6")
        self.assertEqual(at_boundary["fit_max_abs_residual"], "1/3")
        self.assertEqual(at_boundary["heldout_max_abs_residual"], "1/6")
        self.assertTrue(at_boundary["heldout_within_tolerance"])
        below_boundary = evaluate_affine_calibration(fit, heldout, "1/7")
        self.assertFalse(below_boundary["heldout_within_tolerance"])

    def test_degenerate_or_reused_anchors_fail_closed(self):
        same_pts = [
            {"anchor_id": "fit-1", "pts_time": "2", "annotation_time": "1"},
            {"anchor_id": "fit-2", "pts_time": "2", "annotation_time": "3"},
        ]
        heldout = [{"anchor_id": "hold-1", "pts_time": "3", "annotation_time": "4"}]
        with self.assertRaisesRegex(ValueError, "fit_pts_degenerate"):
            evaluate_affine_calibration(same_pts, heldout, "1")
        valid_fit = [
            {"anchor_id": "shared", "pts_time": "0", "annotation_time": "0"},
            {"anchor_id": "fit-2", "pts_time": "1", "annotation_time": "1"},
        ]
        with self.assertRaisesRegex(ValueError, "fit_heldout_identity_overlap"):
            evaluate_affine_calibration(
                valid_fit, [{"anchor_id": "shared", "pts_time": "2", "annotation_time": "2"}], "1")

    def test_nonmonotonic_clock_mapping_fails_closed(self):
        reversed_fit = [
            {"anchor_id": "fit-1", "pts_time": "0", "annotation_time": "2"},
            {"anchor_id": "fit-2", "pts_time": "1", "annotation_time": "1"},
        ]
        heldout = [{"anchor_id": "hold-1", "pts_time": "2", "annotation_time": "0"}]
        with self.assertRaisesRegex(ValueError, "fit_slope_not_positive"):
            evaluate_affine_calibration(reversed_fit, heldout, "1/10")

    def test_invalid_tolerance_and_anchor_values_fail_closed(self):
        fit = [
            {"anchor_id": "fit-1", "pts_time": "0", "annotation_time": "0"},
            {"anchor_id": "fit-2", "pts_time": "1", "annotation_time": "1"},
        ]
        heldout = [{"anchor_id": "hold-1", "pts_time": "2", "annotation_time": "2"}]
        for tolerance in ("0", "-1/10", True, "1/0"):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                evaluate_affine_calibration(fit, heldout, tolerance)
        malformed = [dict(fit[0], pts_time=True), fit[1]]
        with self.assertRaisesRegex(ValueError, "fit_anchors_invalid"):
            evaluate_affine_calibration(malformed, heldout, "1")


if __name__ == "__main__":
    unittest.main()
