"""CPU contract tests for review-interval to Qwen temporal-grid positions."""
from pathlib import Path
from itertools import combinations
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_review_interval_token_groups import (  # noqa: E402
    map_review_intervals_to_qwen_groups,
    map_review_interval_to_qwen_groups,
    select_single_annotation_anchor_group,
    select_equal_count_rank_distance_control_groups,
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

    def test_rational_exact_touch_does_not_create_float_overlap(self):
        kwargs = {
            "input_ids": [99] * 8, "video_token_id": 99,
            "video_grid_thw": [2, 2, 2], "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "sampled_pts_seconds": [Fraction(1, 10), Fraction(1, 5)],
            "video_duration_seconds": Fraction(3, 10),
            "reviewed_interval_seconds": [Fraction(3, 20), Fraction(1, 5)],
        }
        result = map_review_interval_to_qwen_groups(**kwargs)
        self.assertEqual(result["evidence_group_indices"], [1])

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

    def test_unions_disjoint_occurrences_without_filling_the_gap(self):
        result = map_review_intervals_to_qwen_groups(
            **{k: v for k, v in self.kwargs.items() if k != "reviewed_interval_seconds"},
            reviewed_intervals_seconds=[[0, 0.5], [11.5, 12]])
        self.assertEqual(result["evidence_group_indices"], [0, 5])
        self.assertEqual(result["control_group_indices"], [1, 4])
        self.assertEqual(result["temporal_group_count"], 2)
        self.assertEqual(result["evidence_group_count"], 2)
        self.assertEqual(result["total_temporal_group_count"], 6)
        self.assertIn("evidence_group_count", result["temporal_group_count_alias"])
        self.assertEqual(result["token_budget_per_arm"], 8)
        self.assertEqual(result["reviewed_interval_count"], 2)
        self.assertEqual(result["reviewed_occurrence_group_indices"], [[0], [5]])
        self.assertEqual(result["mapping_scope"], "union_of_all_distinct_reviewed_occurrences")
        self.assertFalse(set(result["evidence_group_indices"]) &
                         set(result["control_group_indices"]))

    def test_guard_accepts_stable_group_union_and_rejects_ambiguous_boundary(self):
        common = {k: v for k, v in self.kwargs.items()
                  if k != "reviewed_interval_seconds"}
        stable = map_review_intervals_to_qwen_groups(
            **common, reviewed_intervals_seconds=[[4, 7]],
            interval_boundary_guard_seconds=0.4)
        self.assertEqual(stable["evidence_group_indices"], [2, 3])
        self.assertEqual(stable["interval_boundary_guard_seconds"], 0.4)
        with self.assertRaisesRegex(ValueError,
                                    "calibration_guard_changes_group_assignment"):
            map_review_intervals_to_qwen_groups(
                **common, reviewed_intervals_seconds=[[4, 7]],
                interval_boundary_guard_seconds=1.0)

    def test_guard_must_be_finite_and_nonnegative(self):
        common = {k: v for k, v in self.kwargs.items()
                  if k != "reviewed_interval_seconds"}
        for guard in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(guard=guard), self.assertRaises(ValueError):
                map_review_intervals_to_qwen_groups(
                    **common, reviewed_intervals_seconds=[[4, 7]],
                    interval_boundary_guard_seconds=guard)

    def test_union_mapping_does_not_require_legacy_contiguous_control_per_occurrence(self):
        kwargs = {
            "input_ids": [99] * 28, "video_token_id": 99,
            "video_grid_thw": [7, 2, 2], "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "sampled_pts_seconds": [0, 2, 4, 6, 8, 10, 12],
            "video_duration_seconds": 14,
        }
        result = map_review_intervals_to_qwen_groups(
            **kwargs, reviewed_intervals_seconds=[[3, 9]])
        self.assertEqual(result["evidence_group_indices"], [2, 3, 4])
        self.assertEqual(result["control_group_indices"], [0, 1, 5])

    def test_overlapping_occurrences_union_duplicate_groups_once(self):
        result = map_review_intervals_to_qwen_groups(
            **{k: v for k, v in self.kwargs.items() if k != "reviewed_interval_seconds"},
            reviewed_intervals_seconds=[[0, 0.5], [0.25, 1.25]])
        self.assertEqual(result["evidence_group_indices"], [0])
        self.assertEqual(result["temporal_group_count"], 1)
        self.assertEqual(result["total_temporal_group_count"], 6)
        self.assertEqual(result["reviewed_interval_count"], 2)
        self.assertEqual(result["reviewed_occurrence_group_indices"], [[0], [0]])

    def test_rank_distance_control_matches_exhaustive_oracle(self):
        for count in range(2, 9):
            for size in range(1, count // 2 + 1):
                for evidence in combinations(range(count), size):
                    candidates = [i for i in range(count) if i not in evidence]
                    expected = min(
                        combinations(candidates, size),
                        key=lambda control: (
                            sum(abs(left - right)
                                for left, right in zip(evidence, control)),
                            control))
                    with self.subTest(count=count, evidence=evidence):
                        self.assertEqual(
                            select_equal_count_rank_distance_control_groups(
                                evidence, count), list(expected))

    def test_union_mapper_fails_closed_if_no_equal_count_control_exists(self):
        with self.assertRaisesRegex(ValueError, "equal_count_temporal_control_unavailable"):
            select_equal_count_rank_distance_control_groups([0, 1, 2], 5)
        with self.assertRaisesRegex(ValueError, "reviewed_intervals_invalid"):
            map_review_intervals_to_qwen_groups(
                **{k: v for k, v in self.kwargs.items()
                   if k != "reviewed_interval_seconds"},
                reviewed_intervals_seconds=[])
        duplicate_rows = map_review_intervals_to_qwen_groups(
            **{k: v for k, v in self.kwargs.items()
               if k != "reviewed_interval_seconds"},
            reviewed_intervals_seconds=[[0, 0.5], [0, 0.5]])
        self.assertEqual(duplicate_rows["reviewed_interval_count"], 2)
        self.assertEqual(duplicate_rows["reviewed_occurrence_group_indices"], [[0], [0]])
        self.assertEqual(duplicate_rows["evidence_group_indices"], [0])

    def test_union_mapper_freezes_shared_iterables_once(self):
        result = map_review_intervals_to_qwen_groups(
            input_ids=iter(self.kwargs["input_ids"]),
            video_token_id=self.kwargs["video_token_id"],
            video_grid_thw=iter(self.kwargs["video_grid_thw"]),
            spatial_merge_size=self.kwargs["spatial_merge_size"],
            temporal_patch_size=self.kwargs["temporal_patch_size"],
            sampled_pts_seconds=iter(self.kwargs["sampled_pts_seconds"]),
            video_duration_seconds=self.kwargs["video_duration_seconds"],
            reviewed_intervals_seconds=[[0, 0.5], [11.5, 12]],
        )
        self.assertEqual(result["evidence_group_indices"], [0, 5])

    def test_annotation_bound_is_separate_from_sampled_video_support_end(self):
        from qwen_review_interval_token_groups import map_review_intervals_to_qwen_groups
        support_end = Fraction(50417, 1000)
        kwargs = {
            "input_ids": [99] * 16,
            "video_token_id": 99,
            "video_grid_thw": [4, 2, 2],
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "sampled_pts_seconds": [0, 10, 20, support_end],
            "video_duration_seconds": support_end,
            "annotation_duration_seconds": 50,
            "reviewed_intervals_seconds": [[17, Fraction(35, 2)]],
        }
        result = map_review_intervals_to_qwen_groups(**kwargs)
        self.assertEqual(result["evidence_group_indices"], [2])
        self.assertEqual(result["support_duration_seconds"], float(support_end))
        self.assertEqual(result["annotation_duration_seconds"], 50.0)
        self.assertEqual(kwargs["reviewed_intervals_seconds"], [[17, Fraction(35, 2)]])
        with self.assertRaisesRegex(ValueError, "reviewed_interval_invalid"):
            map_review_intervals_to_qwen_groups(
                **{**kwargs, "reviewed_intervals_seconds": [[Fraction(501, 10), 50.2]]})

    def test_processor_scalar_fps_gate_rejects_group_boundary_disagreement(self):
        kwargs = {
            "input_ids": [99] * 16,
            "video_token_id": 99,
            "video_grid_thw": [4, 2, 2],
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "sampled_pts_seconds": [0, Fraction(101, 100), 2, 3],
            "video_duration_seconds": 4,
            "reviewed_intervals_seconds": [[Fraction(502, 1000), Fraction(504, 1000)]],
            "interval_boundary_guard_seconds": 0,
        }
        actual_only = map_review_intervals_to_qwen_groups(**kwargs)
        self.assertEqual(actual_only["evidence_group_indices"], [0])
        with self.assertRaisesRegex(
                ValueError, "processor_implied_grid_changes_group_assignment"):
            map_review_intervals_to_qwen_groups(
                **kwargs, require_processor_uniform_grid_assignment=True)

    def test_processor_scalar_fps_gate_records_matching_uniform_grid(self):
        kwargs = {
            "input_ids": [99] * 16,
            "video_token_id": 99,
            "video_grid_thw": [4, 2, 2],
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "sampled_pts_seconds": [0, 1, 2, 3],
            "video_duration_seconds": 4,
            "reviewed_intervals_seconds": [[Fraction(52, 100), Fraction(54, 100)]],
            "interval_boundary_guard_seconds": 0,
            "require_processor_uniform_grid_assignment": True,
        }
        result = map_review_intervals_to_qwen_groups(**kwargs)
        self.assertEqual(
            result["processor_uniform_grid_assignment_check"],
            "passed_actual_pts_and_scalar_fps_group_sets_equal")
        self.assertEqual(result["processor_uniform_grid_max_pts_deviation_seconds"], 0.0)

    def test_processor_scalar_fps_gate_supports_temporal_patches(self):
        result = map_review_intervals_to_qwen_groups(
            input_ids=[99] * 12, video_token_id=99,
            video_grid_thw=[3, 2, 2], spatial_merge_size=1,
            temporal_patch_size=2, sampled_pts_seconds=[0, 1, 2, 3, 4, 5],
            video_duration_seconds=6,
            reviewed_intervals_seconds=[[Fraction(21, 10), Fraction(29, 10)]],
            interval_boundary_guard_seconds=0,
            require_processor_uniform_grid_assignment=True,
        )
        self.assertEqual(result["evidence_group_indices"], [1])
        self.assertEqual(
            result["processor_uniform_grid_assignment_check"],
            "passed_actual_pts_and_scalar_fps_group_sets_equal")

    def test_union_mapper_classifies_bad_shared_inputs(self):
        base = {k: v for k, v in self.kwargs.items()
                if k != "reviewed_interval_seconds"}
        for field, expected in (
            ("input_ids", "input_ids_invalid"),
            ("video_grid_thw", "video_grid_thw_invalid"),
            ("sampled_pts_seconds", "sampled_pts_seconds_invalid"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, expected):
                map_review_intervals_to_qwen_groups(
                    **{**base, field: None},
                    reviewed_intervals_seconds=[[0, 0.5]],
                )

    def test_single_anchor_merges_overlapping_annotations_without_double_counting(self):
        result = select_single_annotation_anchor_group(
            sampled_pts_seconds=[0, 1, 5, 9], support_duration_seconds=10,
            temporal_patch_size=1, evidence_group_indices=[1, 2],
            reviewed_intervals_seconds=[[Fraction(1, 2), 7],
                                        [Fraction(1, 2), 3]],
            annotation_duration_seconds=10)
        # Midpoint support cells are [0,1/2], [1/2,3], [3,7], [7,10].
        # The merged union overlaps group 2 by 4 seconds and group 1 by 5/2;
        # summing the overlapping annotation rows separately would incorrectly
        # give group 1 the larger total (5 versus 4).
        self.assertEqual(result["representative_group_index"], 2)
        self.assertEqual(result["control_group_index"], 3)
        self.assertEqual(result["overlap_duration_fraction"], "4/1")
        self.assertEqual(result["merged_annotation_union_seconds"], "13/2")

    def test_single_anchor_overlap_tie_chooses_earliest_group(self):
        result = select_single_annotation_anchor_group(
            sampled_pts_seconds=[0, 1, 2, 3, 4], support_duration_seconds=5,
            temporal_patch_size=1, evidence_group_indices=[1, 2],
            reviewed_intervals_seconds=[[Fraction(1, 2), Fraction(5, 2)]],
            annotation_duration_seconds=5)
        self.assertEqual(result["representative_group_index"], 1)
        self.assertEqual(result["control_group_index"], 0)
        self.assertEqual(result["overlap_duration_fraction"], "1/1")
        self.assertEqual(result["merged_annotation_union_seconds"], "2/1")

    def test_single_anchor_handles_support_edge_exactly(self):
        result = select_single_annotation_anchor_group(
            sampled_pts_seconds=[0, 1, 2], support_duration_seconds=3,
            temporal_patch_size=1, evidence_group_indices=[0],
            reviewed_intervals_seconds=[[0, Fraction(1, 4)]],
            annotation_duration_seconds=3)
        self.assertEqual(result["representative_group_index"], 0)
        self.assertEqual(result["control_group_index"], 1)
        self.assertEqual(result["overlap_duration_fraction"], "1/4")
        self.assertEqual(result["merged_annotation_union_seconds"], "1/4")

    def test_single_anchor_allows_one_control_when_evidence_exceeds_half(self):
        with self.assertRaisesRegex(ValueError, "equal_count_temporal_control_unavailable"):
            select_equal_count_rank_distance_control_groups([0, 1, 2], 5)
        result = select_single_annotation_anchor_group(
            sampled_pts_seconds=[0, 1, 2, 3, 4], support_duration_seconds=5,
            temporal_patch_size=1, evidence_group_indices=[0, 1, 2],
            reviewed_intervals_seconds=[[0, Fraction(5, 2)]],
            annotation_duration_seconds=5)
        self.assertEqual(result["representative_group_index"], 1)
        self.assertEqual(result["control_group_index"], 3)
        self.assertEqual(result["overlap_duration_fraction"], "1/1")

    def test_single_anchor_rejects_short_pts_empty_evidence_and_invalid_intervals(self):
        common = {"sampled_pts_seconds": [0, 1, 2],
                  "support_duration_seconds": 3, "temporal_patch_size": 1,
                  "evidence_group_indices": [1],
                  "reviewed_intervals_seconds": [[0, 2]],
                  "annotation_duration_seconds": 3}
        invalid = [
            ({**common, "sampled_pts_seconds": [0]}, "anchor_requires_at_least_two_groups"),
            ({**common, "evidence_group_indices": []}, "evidence_group_indices_invalid"),
            ({**common, "reviewed_intervals_seconds": [[2, 4]]}, "reviewed_interval_invalid"),
            ({**common, "reviewed_intervals_seconds": []}, "reviewed_intervals_invalid"),
        ]
        for kwargs, error_code in invalid:
            with self.subTest(error_code=error_code), self.assertRaisesRegex(
                    ValueError, error_code):
                select_single_annotation_anchor_group(**kwargs)

    def test_single_anchor_rejects_omitted_or_uncovered_evidence_groups(self):
        common = {"sampled_pts_seconds": [0, 1, 2, 3],
                  "support_duration_seconds": 4, "temporal_patch_size": 1,
                  "reviewed_intervals_seconds": [[0, Fraction(12, 5)]],
                  "annotation_duration_seconds": 4}
        # The interval has positive overlap with groups 0, 1, and 2, but not 3.
        for supplied in ([0, 1], [0, 1, 2, 3]):
            with self.subTest(supplied=supplied), self.assertRaisesRegex(
                    ValueError, "annotation_evidence_group_set_mismatch"):
                select_single_annotation_anchor_group(
                    **common, evidence_group_indices=supplied)

    def test_single_anchor_policy_maps_over_half_evidence_to_one_disjoint_control(self):
        kwargs = {
            "input_ids": [99] * 5, "video_token_id": 99,
            "video_grid_thw": [5, 1, 1], "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "sampled_pts_seconds": [0, 1, 2, 3, 4],
            "video_duration_seconds": 5, "annotation_duration_seconds": 5,
            "reviewed_intervals_seconds": [[0, Fraction(5, 2)]],
        }
        with self.assertRaisesRegex(ValueError, "equal_count_temporal_control_unavailable"):
            map_review_intervals_to_qwen_groups(**kwargs)

        first = map_review_intervals_to_qwen_groups(
            **kwargs, evidence_selection_policy="single_overlap_anchor")
        second = map_review_intervals_to_qwen_groups(
            **kwargs, evidence_selection_policy="single_overlap_anchor")
        self.assertEqual(first["annotation_covered_group_indices"], [0, 1, 2])
        self.assertEqual(first["evidence_group_indices"], [1])
        self.assertEqual(first["control_group_indices"], [3])
        self.assertEqual(first["representative_annotation_group_index"], 1)
        self.assertEqual(first["evidence_selection_policy"], "single_overlap_anchor")
        self.assertNotIn(first["control_group_indices"][0],
                         first["annotation_covered_group_indices"])
        self.assertEqual(first["evidence_group_indices"], second["evidence_group_indices"])
        self.assertEqual(first["control_group_indices"], second["control_group_indices"])
        self.assertEqual(first["representative_annotation_group_index"],
                         second["representative_annotation_group_index"])
        with self.assertRaisesRegex(ValueError, "evidence_selection_policy_invalid"):
            map_review_intervals_to_qwen_groups(
                **kwargs, evidence_selection_policy="unknown_policy")

    def test_annotation_union_only_returns_exact_union_and_explicit_empty_control(self):
        kwargs = {
            "input_ids": [99] * 5, "video_token_id": 99,
            "video_grid_thw": [5, 1, 1], "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "sampled_pts_seconds": [0, 1, 2, 3, 4],
            "video_duration_seconds": 5, "annotation_duration_seconds": 5,
            "reviewed_intervals_seconds": [[0, Fraction(5, 2)]],
        }
        result = map_review_intervals_to_qwen_groups(
            **kwargs, evidence_selection_policy="annotation_union_only")
        self.assertEqual(result["annotation_covered_group_indices"], [0, 1, 2])
        self.assertEqual(result["evidence_group_indices"], [0, 1, 2])
        self.assertEqual(result["control_group_indices"], [])
        self.assertEqual(result["control_token_positions"], [])
        self.assertEqual(result["control_rule"], "none_one_sided_annotation_union_ablation")
        # The no-control estimand remains geometrically defined even if every
        # temporal group is annotation covered.
        full = map_review_intervals_to_qwen_groups(
            **{**kwargs, "reviewed_intervals_seconds": [[0, 5]]},
            evidence_selection_policy="annotation_union_only")
        self.assertEqual(full["evidence_group_indices"], [0, 1, 2, 3, 4])
        self.assertEqual(full["control_group_indices"], [])


if __name__ == "__main__":
    unittest.main()
