"""CPU contract tests for review-interval to Qwen temporal-grid positions."""
from pathlib import Path
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_review_interval_token_groups import (  # noqa: E402
    map_review_interval_to_qwen_groups,
)


class ReviewIntervalTokenGroupTests(unittest.TestCase):
    def setUp(self):
        self.ids = [9] * 7 + [99] * 24 + [9] * 5
        self.kwargs = {
            "input_ids": self.ids,
            "video_token_id": 99,
            "video_grid_thw": [6, 4, 4],
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "sampled_pts_seconds": [Fraction(i) for i in range(12)],
            "video_duration_seconds": Fraction(12),
            "reviewed_interval_seconds": [Fraction(4), Fraction(7)],
        }

    def test_maps_complete_grid_groups_and_nearest_equal_control(self):
        result = map_review_interval_to_qwen_groups(**self.kwargs)
        self.assertEqual(result["status"], "processor_layout_geometry_only")
        self.assertEqual(result["evidence_group_indices"], [2, 3])
        self.assertEqual(result["control_group_indices"], [0, 1])
        self.assertEqual(result["temporal_group_count"], 2)
        self.assertEqual(result["token_budget_per_arm"], 8)
        self.assertEqual(result["evidence_token_positions"], list(range(15, 23)))
        self.assertEqual(result["control_token_positions"], list(range(7, 15)))
        self.assertFalse(set(result["evidence_token_positions"]) &
                         set(result["control_token_positions"]))

    def test_outcome_free_duration_and_center_matching_tie_chooses_earlier(self):
        first = map_review_interval_to_qwen_groups(**self.kwargs)
        changed_only_question_context = {
            **self.kwargs,
            "reviewed_interval_seconds": [4, 7],
        }
        second = map_review_interval_to_qwen_groups(**changed_only_question_context)
        self.assertEqual(first["evidence_group_indices"], second["evidence_group_indices"])
        self.assertEqual(first["control_group_indices"], [0, 1])
        self.assertEqual(second["control_group_indices"], [0, 1])

    def test_control_duration_precedes_center_and_center_precedes_start(self):
        # Evidence group 1 has duration 1.0. Group 0 is closer in center,
        # but group 2 has the matching duration and therefore wins.
        duration_first = {
            "input_ids": [99] * 24, "video_token_id": 99,
            "video_grid_thw": [6, 4, 4], "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "sampled_pts_seconds": [0, 1, 2, 3, 4, 5],
            "video_duration_seconds": 14,
            "reviewed_interval_seconds": [0.51, 1.49],
        }
        result = map_review_interval_to_qwen_groups(**duration_first)
        self.assertEqual(result["evidence_group_indices"], [1])
        self.assertEqual(result["control_group_indices"], [2])

        # In the regular grid, equal-duration candidates [0,1] and [4,5]
        # are equally distant in center from evidence [2,3]; earlier wins.
        center_then_start = map_review_interval_to_qwen_groups(**self.kwargs)
        self.assertEqual(center_then_start["control_group_indices"], [0, 1])

    def test_midpoint_cells_cover_unsampled_intervals_and_assign_boundaries_once(self):
        kwargs = {
            "input_ids": [9] * 4 + [99] * 16,
            "video_token_id": 99,
            "video_grid_thw": [4, 4, 4],
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "sampled_pts_seconds": [0, 1, 10, 11, 20, 21, 30, 31],
            "video_duration_seconds": 32,
            "reviewed_interval_seconds": [4, 6],
        }
        result = map_review_interval_to_qwen_groups(**kwargs)
        self.assertEqual(result["evidence_group_indices"], [0, 1])
        self.assertEqual(result["control_group_indices"], [2, 3])
        self.assertEqual(result["max_sampled_pts_gap_seconds"], 9.0)
        self.assertEqual(result["first_edge_cell_duration_seconds"], 0.5)
        self.assertEqual(result["last_edge_cell_duration_seconds"], 1.5)
        self.assertEqual(result["max_temporal_group_coverage_seconds"], 10.0)
        self.assertEqual(
            result["temporal_diagnostics_status"],
            "diagnostic_only_no_prespecified_acceptance_threshold")
        boundary = map_review_interval_to_qwen_groups(
            **{**kwargs, "reviewed_interval_seconds": [5.5, 6]})
        self.assertEqual(boundary["evidence_group_indices"], [1])

    def test_rejects_missing_geometry_pts_or_matched_control(self):
        cases = [
            ({**self.kwargs, "video_grid_thw": [5, 4, 4]}, "sampled_pts_grid_mismatch"),
            ({**self.kwargs, "input_ids": [9] * 31}, "video_token_block_geometry_mismatch"),
            ({**self.kwargs, "reviewed_interval_seconds": [20, 21]},
             "reviewed_interval_invalid"),
            ({**self.kwargs, "reviewed_interval_seconds": [0, 11]},
             "equal_count_temporal_control_unavailable"),
            ({**self.kwargs, "sampled_pts_seconds": [0, 1, 2, 3, 4, 5,
                                                        6, 7, 8, 9, 10, 10]},
             "sampled_pts_grid_mismatch"),
            ({**self.kwargs, "reviewed_interval_seconds": [0, 13]},
             "reviewed_interval_invalid"),
            ({**self.kwargs, "video_duration_seconds": 10},
             "video_duration_or_sampled_pts_invalid"),
        ]
        for values, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                map_review_interval_to_qwen_groups(**values)


if __name__ == "__main__":
    unittest.main()
