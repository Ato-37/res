import unittest
from decimal import Decimal

from video_time_mapping import (
    chunk_timestamp_ranges, effective_sample_fps, uniform_frame_indices,
)


class VideoTimeMappingTests(unittest.TestCase):
    def test_effective_sample_fps_uses_endpoint_span(self):
        self.assertAlmostEqual(effective_sample_fps([0, 2, 4, 6]), 0.5)
        self.assertAlmostEqual(effective_sample_fps([1.0, 2.5, 4.0]), 2 / 3)
        self.assertAlmostEqual(effective_sample_fps([Decimal("0"), Decimal("2")]), 0.5)

    def test_effective_sample_fps_rejects_invalid_times(self):
        for timestamps in (
            [0], [0, 0], [0, 2, 1], [0, float("inf")], [0, 1, 3], [-1, 0],
            [True, 1], [0, "1"], [0, None], [0, float("nan")], [], None, 1.0,
        ):
            with self.subTest(timestamps=timestamps):
                with self.assertRaises(ValueError):
                    effective_sample_fps(timestamps)

    def test_uniform_frame_indices_have_constant_stride_and_stay_in_bounds(self):
        self.assertEqual(
            uniform_frame_indices(24, 8), [1, 4, 7, 10, 13, 16, 19, 22]
        )
        indices = uniform_frame_indices(7, 8)
        self.assertEqual(indices, [0, 1, 2, 3, 4, 5])
        self.assertEqual(len(set(b - a for a, b in zip(indices, indices[1:]))), 1)
        self.assertTrue(all(0 <= index < 7 for index in indices))

    def test_uniform_frame_indices_properties_for_small_integer_domain(self):
        for total_frames in range(2, 25):
            for requested_frames in range(2, 11):
                with self.subTest(total_frames=total_frames,
                                  requested_frames=requested_frames):
                    indices = uniform_frame_indices(total_frames, requested_frames)
                    expected_count = min(total_frames, requested_frames)
                    expected_count -= expected_count % 2
                    self.assertEqual(len(indices), expected_count)
                    self.assertGreaterEqual(len(indices), 2)
                    self.assertLessEqual(len(indices), requested_frames)
                    self.assertLessEqual(len(indices), total_frames)
                    self.assertEqual(indices, sorted(set(indices)))
                    self.assertTrue(all(0 <= index < total_frames for index in indices))
                    strides = [right - left for left, right in zip(indices, indices[1:])]
                    self.assertTrue(strides)
                    self.assertEqual(len(set(strides)), 1)

    def test_uniform_frame_indices_rejects_noninteger_and_invalid_counts(self):
        invalid_pairs = (
            (True, 4), (4, True), (4.0, 4), (4, 4.0), (4, 2.5),
            ("4", 4), (4, "4"), (None, 4), (4, None),
            (1, 4), (4, 1), (0, 4), (4, 0), (-1, 4), (4, -1),
        )
        for total_frames, requested_frames in invalid_pairs:
            with self.subTest(total_frames=total_frames,
                              requested_frames=requested_frames):
                with self.assertRaises(ValueError):
                    uniform_frame_indices(total_frames, requested_frames)

    def test_chunk_ranges_follow_temporal_patch_size(self):
        self.assertEqual(
            chunk_timestamp_ranges([0, 1, 2, 3, 4, 5], 2),
            [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)],
        )
        self.assertEqual(
            chunk_timestamp_ranges([Decimal("0"), Decimal("1")], 2),
            [(0.0, 1.0)],
        )
        with self.assertRaises(ValueError):
            chunk_timestamp_ranges([0, 1, 2], 2)
        with self.assertRaises(ValueError):
            chunk_timestamp_ranges([0, float("nan")], 2)

    def test_chunk_ranges_rejects_invalid_patch_sizes(self):
        for patch_size in (True, 0, -1, 1.0, 1.5, "2", None):
            with self.subTest(patch_size=patch_size):
                with self.assertRaises(ValueError):
                    chunk_timestamp_ranges([0, 1], patch_size)

    def test_chunk_ranges_rejects_empty_and_invalid_timestamps(self):
        for timestamps in (
            [], [True, 1], [0, "1"], [0, None], [0, float("nan")],
            [0, float("inf")], [-1, 0], [1, 1], [0, 2, 1], None,
        ):
            with self.subTest(timestamps=timestamps):
                with self.assertRaises(ValueError):
                    chunk_timestamp_ranges(timestamps, 1)


if __name__ == "__main__":
    unittest.main()
