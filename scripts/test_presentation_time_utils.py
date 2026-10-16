"""Synthetic tests for exact decoder-PTS to evidence-time mapping."""
from fractions import Fraction
import unittest

from presentation_time_utils import (FFMPEG_AV_NOPTS_VALUE,
                                     INT64_MAX, presentation_times_from_pts,
                                     assign_evidence_control_frames_from_times,
                                     strict_evidence_indices_from_times,
                                     uniform_sampling_rate_from_times)


class PresentationTimeUtilsTests(unittest.TestCase):
    def test_cfr_pts_map_uses_first_displayed_frame_as_zero(self):
        self.assertEqual(
            presentation_times_from_pts([300, 301, 302, 303], Fraction(1, 30)),
            (Fraction(0), Fraction(1, 30), Fraction(1, 15), Fraction(1, 10)))

    def test_vfr_positive_nonuniform_pts_are_preserved_exactly(self):
        self.assertEqual(
            presentation_times_from_pts([100, 103, 108, 109], (1, 10)),
            (Fraction(0), Fraction(3, 10), Fraction(4, 5), Fraction(9, 10)))

    def test_inclusive_interval_membership_uses_exact_presentation_times(self):
        times = presentation_times_from_pts([0, 2, 5, 9], Fraction(1, 10))
        self.assertEqual(
            strict_evidence_indices_from_times([0, 1, 2, 3], times, [[0.2, 0.5]]),
            [1, 2])

    def test_fraction_interval_endpoints_remain_exact(self):
        times = presentation_times_from_pts([0, 1, 2], Fraction(1, 3))
        self.assertEqual(
            strict_evidence_indices_from_times(
                [0, 1, 2], times, [[Fraction(1, 3), Fraction(2, 3)]]),
            [1, 2])

    def test_integrated_assignment_uses_pts_but_preserves_rank_control(self):
        times = presentation_times_from_pts(
            [0, 1, 2, 4, 5, 6, 14, 15], Fraction(1, 10))
        result = assign_evidence_control_frames_from_times(
            8, 4, times, [[0.2, 0.6]])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["sampled_indices"], [0, 2, 5, 7])
        self.assertEqual(result["evidence_indices"], [2, 5])
        self.assertEqual(result["control_indices"], [0, 7])
        self.assertEqual(result["matched_rank_distance"], 2)

    def test_integrated_assignment_rejects_incomplete_pts_map(self):
        with self.assertRaisesRegex(ValueError, "count must equal"):
            assign_evidence_control_frames_from_times(
                8, 4, presentation_times_from_pts([0, 1], Fraction(1, 30)),
                [[0, 1]])

    def test_multiple_intervals_form_union_without_reordering_frames(self):
        times = presentation_times_from_pts([0, 1, 2, 3], Fraction(1, 10))
        self.assertEqual(
            strict_evidence_indices_from_times([3, 0, 1, 2], times,
                                               [[0.3, 0.3], [0.0, 0.0]]),
            [3, 0])

    def test_rejects_empty_or_missing_pts(self):
        for pts in ([], [0, None], None):
            with self.subTest(pts=pts), self.assertRaises(ValueError):
                presentation_times_from_pts(pts, Fraction(1, 25))
        for pts in ([FFMPEG_AV_NOPTS_VALUE, FFMPEG_AV_NOPTS_VALUE + 1],
                    [0, FFMPEG_AV_NOPTS_VALUE]):
            with self.subTest(pts=pts), self.assertRaisesRegex(ValueError, "PTS is missing"):
                presentation_times_from_pts(pts, Fraction(1, 25))
        with self.assertRaisesRegex(ValueError, "signed 64-bit"):
            presentation_times_from_pts([0, INT64_MAX + 1], Fraction(1, 25))

    def test_rejects_duplicate_or_nonmonotonic_pts(self):
        for pts in ([0, 1, 1], [0, 2, 1]):
            with self.subTest(pts=pts), self.assertRaisesRegex(ValueError, "strictly increasing"):
                presentation_times_from_pts(pts, Fraction(1, 25))

    def test_rejects_invalid_time_bases(self):
        for time_base in (Fraction(0), Fraction(-1, 10), 0.04, True, (1, 0),
                          (True, 10), (1, True)):
            with self.subTest(time_base=time_base), self.assertRaises(ValueError):
                presentation_times_from_pts([0, 1], time_base)

    def test_rejects_bad_sample_indices_and_inconsistent_time_maps(self):
        valid = (Fraction(0), Fraction(1, 10))
        for indices, times in (([2], valid), ([-1], valid), ([0, 0], valid), ([0], ()),
                               ([0], (Fraction(1, 10),)),
                               ([0], (Fraction(0), Fraction(0)))):
            with self.subTest(indices=indices, times=times), self.assertRaises(ValueError):
                strict_evidence_indices_from_times(indices, times, [[0, 1]])

    def test_rejects_malformed_intervals(self):
        times = (Fraction(0), Fraction(1, 10))
        for intervals in (None, [[0]], [[1, 0]], [[float("nan"), 1]], [[False, 1]]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                strict_evidence_indices_from_times([0, 1], times, intervals)

    def test_uniform_sampled_times_return_exact_qwen_rate(self):
        rate, deviation = uniform_sampling_rate_from_times(
            [Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2)], 0)
        self.assertEqual(rate, Fraction(2))
        self.assertEqual(deviation, Fraction(0))

    def test_rounding_jitter_is_accepted_only_with_frozen_tolerance(self):
        times = [Fraction(0), Fraction(7, 10), Fraction(14, 10), Fraction(22, 10)]
        rate, deviation = uniform_sampling_rate_from_times(times, Fraction(1, 10))
        self.assertEqual(rate, Fraction(15, 11))
        self.assertEqual(deviation, Fraction(1, 15))
        with self.assertRaisesRegex(ValueError, "uniform-grid tolerance"):
            uniform_sampling_rate_from_times(times, Fraction(1, 40))

    def test_nonuniform_vfr_times_fail_uniform_qwen_rate_gate(self):
        times = [Fraction(0), Fraction(1, 10), Fraction(5), Fraction(10)]
        with self.assertRaisesRegex(ValueError, "uniform-grid tolerance"):
            uniform_sampling_rate_from_times(times, Fraction(1, 100))

    def test_uniform_rate_gate_rejects_missing_or_invalid_times(self):
        for times, tolerance in (([], 0), ([Fraction(0)], 0),
                                 ([Fraction(0), Fraction(0)], 0),
                                 ([Fraction(0), Fraction(1, 2)], True),
                                 ([Fraction(0), Fraction(1, 2)], Fraction(-1, 10)),
                                 ([0, Fraction(1, 2)], 0)):
            with self.subTest(times=times, tolerance=tolerance), self.assertRaises(ValueError):
                uniform_sampling_rate_from_times(times, tolerance)


if __name__ == "__main__":
    unittest.main()
