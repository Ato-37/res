import unittest

from grounding_metrics import attention_overlap_proxy


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

    def test_misaligned_and_invalid_intervals_are_rejected(self):
        with self.assertRaises(ValueError):
            attention_overlap_proxy([1, 2], [(0, 1)], [(0, 1)])
        with self.assertRaises(ValueError):
            attention_overlap_proxy([1], [(0, 0)], [(0, 1)])
        with self.assertRaises(ValueError):
            attention_overlap_proxy([1], [(0, 1)], [(2, 1)])


if __name__ == "__main__":
    unittest.main()
