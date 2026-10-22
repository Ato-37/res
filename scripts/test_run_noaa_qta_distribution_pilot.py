import unittest

from run_noaa_qta_distribution_pilot import (
    choose_equal_budget_sham, restricted_choice_probabilities,
    temporal_token_groups, total_variation,
)


class NoaaQtaDistributionPilotTests(unittest.TestCase):
    def test_sham_is_deterministic_equal_size_and_disjoint(self):
        first = choose_equal_budget_sham(temporal_group_count=8,
            qta_groups=(2, 5), opaque_row_token="opaque-row")
        second = choose_equal_budget_sham(temporal_group_count=8,
            qta_groups=(2, 5), opaque_row_token="opaque-row")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertFalse(set(first).intersection((2, 5)))

    def test_sham_rejects_no_disjoint_budget(self):
        with self.assertRaisesRegex(ValueError, "sham_inputs_invalid"):
            choose_equal_budget_sham(temporal_group_count=2,
                qta_groups=(0, 1), opaque_row_token="opaque-row")

    def test_maps_equal_complete_temporal_groups(self):
        groups = temporal_token_groups(
            [7, 99, 99, 99, 99, 99, 99, 99, 99, 8], video_token_id=99,
            grid=(2, 2, 2), spatial_merge_size=1)
        self.assertEqual(groups, [(1, 2, 3, 4), (5, 6, 7, 8)])

    def test_rejects_noncontiguous_video_tokens(self):
        with self.assertRaisesRegex(ValueError, "video_token_group_geometry_invalid"):
            temporal_token_groups([7, 99, 8, 99], video_token_id=99,
                grid=(2, 1, 1), spatial_merge_size=1)

    def test_candidate_choice_probabilities_and_tvd(self):
        p = restricted_choice_probabilities([-1.0, -2.0, -3.0, -4.0, -5.0])
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertAlmostEqual(total_variation(p, p), 0.0)
        q = restricted_choice_probabilities([-5.0, -4.0, -3.0, -2.0, -1.0])
        self.assertGreater(total_variation(p, q), 0.0)
        self.assertLessEqual(total_variation(p, q), 1.0)


if __name__ == "__main__":
    unittest.main()
