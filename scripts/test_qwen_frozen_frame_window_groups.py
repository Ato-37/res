"""Focused CPU contract tests for frozen Qwen sampled-frame windows."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_frozen_frame_window_groups import (  # noqa: E402
    map_frozen_frame_windows_to_qwen_groups,
)


class FrozenFrameWindowGroupsTests(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "input_ids": [7] * 3 + [99] * 24 + [8],
            "video_token_id": 99,
            "video_grid_thw": [6, 4, 4],
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "sampled_pts_seconds": list(range(12)),
            "evidence_frame_window": [4, 8],
            "control_frame_window": [0, 4],
        }

    def test_maps_whole_windows_and_flattens_token_positions(self):
        result = map_frozen_frame_windows_to_qwen_groups(**self.kwargs)
        self.assertEqual(result["evidence_group_indices"], [2, 3])
        self.assertEqual(result["control_group_indices"], [0, 1])
        self.assertEqual(result["evidence_token_positions"], list(range(11, 19)))
        self.assertEqual(result["control_token_positions"], list(range(3, 11)))
        self.assertEqual(result["token_budget_per_arm"], 8)
        self.assertFalse(set(result["evidence_token_positions"]) &
                         set(result["control_token_positions"]))

    def test_rejects_partial_temporal_groups_and_overlapping_windows(self):
        cases = [
            ({"evidence_frame_window": [3, 7]}, "evidence_frame_window_not_whole"),
            ({"control_frame_window": [0, 3]}, "control_frame_window_not_whole"),
            ({"control_frame_window": [2, 6]}, "frame_windows_overlap"),
        ]
        for update, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                map_frozen_frame_windows_to_qwen_groups(**{**self.kwargs, **update})

    def test_rejects_unequal_window_sizes(self):
        with self.assertRaisesRegex(ValueError, "frame_window_size_mismatch"):
            map_frozen_frame_windows_to_qwen_groups(
                **{**self.kwargs, "control_frame_window": [0, 2]})

    def test_rejects_mismatched_frame_patch_and_pts_geometry(self):
        cases = [
            ({"video_grid_thw": [5, 4, 4]}, "sampled_pts_frame_geometry_mismatch"),
            ({"sampled_pts_seconds": list(range(11))}, "sampled_pts_frame_geometry_mismatch"),
            ({"temporal_patch_size": 3}, "sampled_pts_frame_geometry_mismatch"),
            ({"sampled_pts_seconds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]},
             "sampled_pts_order_invalid"),
        ]
        for update, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                map_frozen_frame_windows_to_qwen_groups(**{**self.kwargs, **update})

    def test_rejects_incorrect_visual_token_geometry(self):
        with self.assertRaisesRegex(ValueError, "video_token_block_geometry_mismatch"):
            map_frozen_frame_windows_to_qwen_groups(
                **{**self.kwargs, "input_ids": [7] * 3 + [99] * 23 + [8]})


if __name__ == "__main__":
    unittest.main()
