import unittest
from analyze_claim_likelihood_window_gold_overlap import (
    union_intervals,overlap_metrics,sampled_point_coverage,
)

class GoldOverlapTests(unittest.TestCase):
    def test_unions_overlapping_and_touching_annotation_spans(self):
        self.assertEqual(union_intervals([[1,3],[2,4],[5,6],[6,7]]),((1.0,4.0),(5.0,7.0)))

    def test_overlap_metrics_use_temporal_union_and_interval_lengths(self):
        m=overlap_metrics((2,6),[[1,3],[3,5]])
        self.assertAlmostEqual(m["intersection_seconds"],3)
        self.assertAlmostEqual(m["intersection_over_candidate"],.75)
        self.assertAlmostEqual(m["intersection_over_annotation"],.75)
        self.assertAlmostEqual(m["temporal_iou"],.6)

    def test_invalid_or_empty_candidates_fail_closed(self):
        with self.assertRaises(ValueError):overlap_metrics((1,1),[[0,2]])
        with self.assertRaises(ValueError):union_intervals([[2,1]])

    def test_sampled_point_coverage_uses_union_and_inclusive_interval_bounds(self):
        self.assertAlmostEqual(sampled_point_coverage(
            [0,1,2,3,4],[[1,2],[2,3]]),.6)
        with self.assertRaisesRegex(ValueError,"sampled_points_empty"):
            sampled_point_coverage([],[[0,1]])

if __name__=="__main__":unittest.main()
