import unittest

from audit_nextgqa_pilot_frame2time_alignment import uniform_grid_coverage


class UniformGridCoverageTests(unittest.TestCase):
    def test_endpoint_inclusive_sampling_covers_only_sampled_instants(self):
        result = uniform_grid_coverage(
            list(map(float, range(16))), [(5.0, 5.0), (4.5, 4.5)], 4)
        self.assertTrue(result["compliant"])
        self.assertEqual(result["covered_spans"], 1)
        self.assertEqual(result["total_spans"], 2)

    def test_nonuniform_grid_over_tolerance_is_not_counted_as_valid_coverage(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.2, 6.0, 7.0]
        result = uniform_grid_coverage(times, [(5.2, 5.2)], 4, 0.05)
        self.assertFalse(result["compliant"])
        self.assertEqual(result["covered_spans"], 0)

    def test_insufficient_decoded_frames_fail_closed(self):
        result = uniform_grid_coverage([0.0, 1.0], [(0.0, 1.0)], 4)
        self.assertFalse(result["compliant"])
        self.assertIsNone(result["max_deviation_seconds"])
        self.assertEqual(result["total_spans"], 1)


if __name__ == "__main__":
    unittest.main()
