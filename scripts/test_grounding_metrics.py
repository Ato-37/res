import unittest

from grounding_metrics import attention_overlap_proxy, sampled_point_window_coverage


class AttentionOverlapProxyTests(unittest.TestCase):
    def test_peak_chunk_intersects_any_gold_interval(self):
        weights, peak, hit = attention_overlap_proxy(
            [0.1, 0.8, 0.1], [(0, 1), (1, 2), (2, 3)], [(1.7, 2.2)]
        )
        self.assertEqual(peak, 1)
        self.assertTrue(hit)
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_peak_chunk_outside_gold_is_miss_even_if_other_chunk_overlaps(self):
        _, peak, hit = attention_overlap_proxy(
            [0.8, 0.1, 0.1], [(0, 1), (1, 2), (2, 3)], [(1.1, 1.9)]
        )
        self.assertEqual(peak, 0)
        self.assertFalse(hit)

    def test_touching_boundary_is_not_positive_duration_intersection(self):
        _, _, hit = attention_overlap_proxy([1.0], [(0, 1)], [(1, 2)])
        self.assertFalse(hit)

    def test_empty_gold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            attention_overlap_proxy([1], [(0, 1)], [])

    def test_invalid_weights_are_rejected(self):
        for values in ([0], [-1], [float("nan")], [float("inf")]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    attention_overlap_proxy(values, [(0, 1)], [(0, 1)])

    def test_finite_weights_with_overflowing_sum_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'finite and positive'):
            attention_overlap_proxy([1e308,1e308],[(0,1),(1,2)],[(0,1)])

    def test_misaligned_and_invalid_intervals_are_rejected(self):
        with self.assertRaises(ValueError):
            attention_overlap_proxy([1, 2], [(0, 1)], [(0, 1)])
        with self.assertRaises(ValueError):
            attention_overlap_proxy([1], [(0, 0)], [(0, 1)])
        with self.assertRaises(ValueError):
            attention_overlap_proxy([1], [(0, 1)], [(2, 1)])


class SampledPointWindowCoverageTests(unittest.TestCase):
    def test_bool_and_string_annotations_are_rejected_before_numeric_coercion(self):
        for gold in ([('0','1')], [(False,True)], [(0,10**400)]):
            with self.subTest(gold=gold), self.assertRaises(ValueError):
                sampled_point_window_coverage([0,1],2,gold)
            with self.subTest(gold=gold), self.assertRaises(ValueError):
                attention_overlap_proxy([1],[(0,1)],gold)
        for attention in ([True],['1']):
            with self.subTest(attention=attention), self.assertRaises(ValueError):
                attention_overlap_proxy(attention,[(0,1)],[(0,1)])
        for ranges in ([('0','1')],[(False,True)]):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                attention_overlap_proxy([1],ranges,[(0,1)])

    def test_span_proxy_can_hit_when_no_sampled_point_is_inside(self):
        timestamps = [0, 10]
        window = [(4, 6)]
        chunk_ranges = [(timestamps[0], timestamps[-1])]
        _, _, span_hit = attention_overlap_proxy([1], chunk_ranges, window)
        coverage = sampled_point_window_coverage(timestamps, 2, window)
        self.assertTrue(span_hit)
        self.assertEqual(coverage['points_inside'], 0)
        self.assertEqual(coverage['per_chunk_points_inside'], (0,))
        self.assertEqual(coverage['maximum_points_inside_any_chunk'], 0)
        self.assertFalse(coverage['any_point_inside'])

    def test_attention_peak_count_is_distinct_from_maximum_coverage_chunk(self):
        timestamps = [0, 10, 11, 12]
        windows = [(4, 6), (11, 12)]
        ranges = [(0, 10), (11, 12)]
        coverage = sampled_point_window_coverage(timestamps, 2, windows)
        _, attention_peak, hit = attention_overlap_proxy(
            [0.9, 0.1], ranges, windows)
        self.assertEqual(coverage['per_chunk_points_inside'], (0, 2))
        self.assertEqual(coverage['maximum_points_inside_any_chunk'], 2)
        self.assertEqual(attention_peak, 0)
        self.assertTrue(hit)
        self.assertEqual(coverage['per_chunk_points_inside'][attention_peak], 0)

    def test_point_membership_is_inclusive_at_annotation_boundaries(self):
        timestamps = [0, 1]
        window = [(1, 2)]
        coverage = sampled_point_window_coverage(timestamps, 1, window)
        self.assertEqual(coverage['points_inside'], 1)
        self.assertEqual(coverage['per_chunk_points_inside'], (0, 1))
        _, _, strict_span_hit = attention_overlap_proxy([1], [(0, 1)], window)
        self.assertFalse(strict_span_hit)

    def test_reports_counts_for_each_temporal_patch(self):
        coverage = sampled_point_window_coverage(
            [0, 1, 2, 3, 4, 5], 2, [(1, 1.2), (4, 4.5)])
        self.assertEqual(coverage['per_chunk_points_inside'], (1, 0, 1))
        self.assertEqual(coverage['points_inside'], 2)
        self.assertEqual(coverage['maximum_points_inside_any_chunk'], 1)
        self.assertTrue(coverage['any_point_inside'])

    def test_rejects_invalid_timestamps_order_patch_size_and_gold(self):
        for timestamps in ([0, float('nan')], [0, float('inf')], [1, 0],
                           [0, '1'], [0, True]):
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                sampled_point_window_coverage(timestamps, 1, [(0, 1)])
        for patch_size in (0, -1, 1.5, 2.0, True):
            with self.subTest(patch_size=patch_size), self.assertRaises(ValueError):
                sampled_point_window_coverage([0, 1], patch_size, [(0, 1)])
        with self.assertRaises(ValueError):
            sampled_point_window_coverage([0, 1], 2, [])
        for intervals in ([(2, 1)], [(0, float('inf'))], [(0, float('nan'))]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                sampled_point_window_coverage([0, 1], 2, intervals)


if __name__ == "__main__":
    unittest.main()
