"""Boundary tests for the counts-only NOAA sampling geometry audit."""

import math
import unittest

from audit_noaa_sampling_coverage import (
    _endpoint_span_summary,
    _window_frame_bounds,
    audit_window_bounds,
)


class NoaaSamplingCoverageTests(unittest.TestCase):
    def test_noninteger_fps_uses_ceil_start_floor_end_and_clamped_inclusive_end(self):
        fps = 29.97
        frame_count = 300
        duration = frame_count / fps
        self.assertEqual(
            _window_frame_bounds(
                0.1, 0.2, fps=fps, frame_count=frame_count,
                duration_seconds=duration,
            ),
            (3, 5),
        )
        self.assertEqual(
            _window_frame_bounds(
                duration - (2 / fps), duration, fps=fps,
                frame_count=frame_count, duration_seconds=duration,
            ),
            (298, 299),
        )

    def test_nonfinite_and_out_of_source_bounds_are_distinct(self):
        common = dict(fps=10.0, frame_count=100, duration_seconds=10.0)
        self.assertEqual(_window_frame_bounds(math.nan, 1.0, **common), "nonfinite_window_bounds")
        self.assertEqual(_window_frame_bounds(-0.1, 1.0, **common), "window_bounds_outside_source")
        self.assertEqual(_window_frame_bounds(1.0, 1.0, **common), "window_bounds_outside_source")
        self.assertEqual(_window_frame_bounds(9.0, 10.1, **common), "window_bounds_outside_source")

    def test_partial_failures_preserve_window_denominator_and_report_actual_counts(self):
        audit = audit_window_bounds(
            [
                (0.1, 0.2),
                (math.nan, 0.5),
                (-1.0, 1.0),
                (0.5, 0.5),
            ],
            frame_count=300,
            fps=29.97,
            duration_seconds=300 / 29.97,
        )
        self.assertEqual(audit["windows_requested"], 4)
        self.assertEqual(audit["windows_valid"], 1)
        self.assertEqual(audit["nonfinite_window_bounds_failures"], 1)
        self.assertEqual(audit["window_bounds_failures"], 2)
        self.assertEqual(audit["windows_without_source_frame_presentation_timestamps"], 0)
        for requested in ("4", "8"):
            per_window = audit["sampling"][requested]["per_question_window"]
            self.assertEqual(per_window["frames_inside_windows"], 2)
            self.assertEqual(per_window["windows_with_zero_sampled_frames"], 3)
            self.assertEqual(per_window["windows_with_fewer_than_two_sampled_frames"], 3)
            self.assertEqual(per_window["windows_with_at_least_two_sampled_frames"], 1)

    def test_endpoint_span_can_overlap_while_no_sampled_frame_point_is_inside(self):
        summary = _endpoint_span_summary(
            [(0, 10)], [(4.0, 6.0)], fps=1.0,
        )
        self.assertEqual(summary["windows_with_any_sampled_point"], 0)
        self.assertEqual(summary["windows_with_any_positive_paired_endpoint_span_overlap"], 1)
        self.assertEqual(summary["positive_paired_endpoint_span_intersections"], 1)
        self.assertEqual(summary["span_intersects_but_zero_actual_sample_points_windows"], 1)
        self.assertEqual(summary["max_nearest_sample_gap_seconds"], 4.0)

    def test_endpoint_span_boundary_touch_is_not_positive_overlap(self):
        summary = _endpoint_span_summary(
            [(0, 4)], [(4.0, 6.0)], fps=1.0,
        )
        self.assertEqual(summary["windows_with_any_sampled_point"], 1)
        self.assertEqual(summary["windows_with_any_positive_paired_endpoint_span_overlap"], 0)
        self.assertEqual(summary["positive_paired_endpoint_span_intersections"], 0)
        self.assertEqual(summary["span_intersects_but_zero_actual_sample_points_windows"], 0)


if __name__ == "__main__":
    unittest.main()
