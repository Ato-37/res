import unittest
from fractions import Fraction
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qta_question_swap_gsub_alignment import _score_one_selection

from qta_question_swap_gsub_alignment import (
    exact_two_group_mean_iou, interval_union_duration, intervals_within_media,
    map_intervals_to_temporal_groups, set_iou, validate_selected_groups,
    validate_source_clock, validate_video_question_pair, video_support_end,
)


class QtaQuestionSwapGsubAlignmentTests(unittest.TestCase):
    def test_maps_one_interval_to_overlapping_pairwise_groups(self):
        pts = [Fraction(i, 31) * 10 for i in range(32)]
        self.assertEqual(
            map_intervals_to_temporal_groups(
                sampled_pts=pts, temporal_patch_size=2, temporal_groups=16,
                support_duration=10, intervals=[[2.0, 3.0]]),
            (3, 4),
        )

    def test_endpoint_inclusive_sample_at_right_censored_support_is_valid(self):
        pts = [Fraction(i, 31) * 10 for i in range(32)]
        self.assertEqual(pts[-1], Fraction(10))
        self.assertEqual(map_intervals_to_temporal_groups(
            sampled_pts=pts, temporal_patch_size=2, temporal_groups=16,
            support_duration=pts[-1], intervals=[[9.9, 10.0]]), (15,))

    def test_score_uses_fraction_pts_and_respects_right_censored_end(self):
        pts = [Fraction(i, 31) * 10 for i in range(32)]
        result = _score_one_selection(selected=(15,), evidence=(15,),
            sampled_pts=pts, intervals=[[9.9, 10.0]], duration=float(pts[-1]))
        self.assertEqual(result["group_iou"], 1.0)
        self.assertAlmostEqual(result["interval_time_coverage"], 1.0, places=6)

    def test_disjoint_occurrences_form_union_without_filling_gap(self):
        pts = [Fraction(i, 31) * 16 for i in range(32)]
        groups = map_intervals_to_temporal_groups(
            sampled_pts=pts, temporal_patch_size=2, temporal_groups=16,
            support_duration=16, intervals=[[1.0, 2.0], [14.0, 15.0]])
        self.assertEqual(groups, (1, 2, 13, 14))

    def test_source_clock_can_start_after_zero_and_extend_to_media_end(self):
        pts = [Fraction(1, 30) + Fraction(i, 31) * 16 for i in range(32)]
        first = map_intervals_to_temporal_groups(
            sampled_pts=pts, temporal_patch_size=2, temporal_groups=16,
            support_duration=Fraction(16) + Fraction(1, 30),
            intervals=[[0.0, 0.2]])
        last = map_intervals_to_temporal_groups(
            sampled_pts=pts, temporal_patch_size=2, temporal_groups=16,
            support_duration=Fraction(16) + Fraction(1, 30),
            intervals=[[15.9, 16.0]])
        self.assertEqual(first, (0,))
        self.assertEqual(last, (15,))

    def test_source_clock_requires_ordered_in_bounds_pts_matches(self):
        pts = [Fraction(i, 10) for i in range(5)]
        self.assertEqual(validate_source_clock(
            source_times=[0.0, 0.2, 0.4], decoded_pts=pts,
            media_start=0, media_end=0.4, tolerance=0), (0.0, 0.0, 0.0))
        for values in ([0.0, 0.4, 0.2], [0.0, 0.5], [0.0, 0.22]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_source_clock(source_times=values, decoded_pts=pts,
                    media_start=0, media_end=0.4, tolerance=Fraction(1, 100))
        with self.assertRaisesRegex(ValueError, "pts_mismatch"):
            validate_source_clock(source_times=[0.04, 0.049], decoded_pts=pts,
                media_start=0, media_end=0.4, tolerance=Fraction(1, 10))

    def test_all_temporal_groups_are_not_a_localized_interval(self):
        pts = [Fraction(i, 31) * 10 for i in range(32)]
        groups = map_intervals_to_temporal_groups(
            sampled_pts=pts, temporal_patch_size=2, temporal_groups=16,
            support_duration=10, intervals=[[0, 10]])
        self.assertEqual(groups, tuple(range(16)))

    def test_plan_video_and_question_rows_must_join_exactly(self):
        qa = {("v1", "q1"): {"question": "what happens"},
              ("v1", "q2"): {"question": "where"}}
        a = {"video_id": "v1", "qid": "q1", "question": "what happens"}
        b = {"video_id": "v1", "qid": "q2", "question": "where"}
        self.assertEqual(validate_video_question_pair(video_id="v1",
            relative_video_path="shard/v1.mp4", mapped_video_path="shard/v1",
            question_a=a, question_b=b, question_map=qa), (qa[("v1", "q1")], qa[("v1", "q2")]))
        with self.assertRaises(ValueError):
            validate_video_question_pair(video_id="v1", relative_video_path="shard/v1.mp4",
                mapped_video_path="shard/v1", question_a=a,
                question_b={**b, "video_id": "v2"}, question_map=qa)
        with self.assertRaises(ValueError):
            validate_video_question_pair(video_id="v1", relative_video_path="shard/v1.mp4",
                mapped_video_path="shard/v1", question_a=a,
                question_b={**b, "qid": "q1"}, question_map=qa)

    def test_annotation_intervals_are_not_silently_clipped(self):
        self.assertTrue(intervals_within_media([[1, 2], [3, 4]], 4))
        self.assertFalse(intervals_within_media([[1, 2], [3, 5]], 4))
        self.assertFalse(intervals_within_media([[5, 6]], 4))

    def test_video_end_uses_stream_not_longer_container_duration(self):
        end, method = video_support_end(container_start=0,
            container_duration=10, stream_start=0, stream_duration=5,
            last_pts=4.9, frame_period=0.1)
        self.assertEqual(end, 5)
        self.assertEqual(method, "video_stream_start_plus_duration")
        with self.assertRaisesRegex(ValueError, "bounds_disagree"):
            video_support_end(container_start=0, container_duration=5,
                stream_start=0, stream_duration=7, last_pts=4.9, frame_period=0.1)
        fallback, fallback_method = video_support_end(container_start=0,
            container_duration=5.1, stream_start=None, stream_duration=None,
            last_pts=5, frame_period=0.1)
        self.assertEqual(fallback, Fraction(51, 10))
        self.assertEqual(fallback_method,
                         "last_decoded_pts_plus_median_frame_period_fallback")

    def test_group_iou_and_exact_equal_budget_reference(self):
        self.assertEqual(set_iou((1, 2), (2, 3, 4)), 0.25)
        expected = sum(set_iou(pair, (0, 1)) for pair in
                       ((i, j) for i in range(4) for j in range(i + 1, 4))) / 6
        self.assertEqual(exact_two_group_mean_iou(temporal_groups=4,
                                                   evidence_groups=(0, 1)), expected)

    def test_interval_union_does_not_double_count_overlap(self):
        self.assertEqual(interval_union_duration([[1, 3], [2, 4], [6, 7]]), 4)

    def test_selected_groups_must_be_unique_ordered_and_in_range(self):
        self.assertEqual(validate_selected_groups((1, 4), temporal_groups=16), (1, 4))
        for value in ((4, 1), (1, 1), (1, 16), (1,)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_selected_groups(value, temporal_groups=16)

    def test_invalid_intervals_and_pts_fail_closed(self):
        pts = [Fraction(i, 31) * 10 for i in range(32)]
        for intervals in ([], [[1, 1]], [[-1, 2]], [[9, 11]]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                map_intervals_to_temporal_groups(
                    sampled_pts=pts, temporal_patch_size=2, temporal_groups=16,
                    support_duration=10, intervals=intervals)


if __name__ == "__main__":
    unittest.main()
