import unittest

from minerva_ego_anchor_utils import (
    evidence_temporal_groups,
    build_uniform_sample_indices,
    build_anchor_centered_sample_indices,
    matched_control_groups,
    parse_timestamp_bins,
    parse_rationale_time_intervals,
    select_rationale_mask_anchor,
    supported_rationale_lines,
    score_generated_movement_anchors,
    track_overlapping_temporal_groups,
    timestamp_bins_overlap_segment,
    validate_sampled_evidence_alignment,
    bbox_to_merged_patch_offsets,
    bbox_to_video_visual_key_positions,
    include_exact_anchor_sample,
    matched_spatial_sham_offsets,
)
from run_minerva_ego_generated_grounding_pilot import build_grounding_prompt
from run_minerva_ego_anchor_gpu_pilot import (
    select_supported_target, validate_cohort_binding,
    validate_spatial_token_preflight, _video_groups)
from preflight_minerva_grounding_cohort import process_anchor_frame_pair


class TimestampAnchorTests(unittest.TestCase):
    def test_video_token_groups_accept_dynamic_56_frame_batch(self):
        import torch
        class VideoProcessor:
            temporal_patch_size = 2
        class ImageProcessor:
            merge_size = 2
        class Processor:
            video_processor = VideoProcessor()
            image_processor = ImageProcessor()
            video_token_id = 42
        group_size = 11 * 11
        ids = [10] + [42] * (28 * group_size) + [11]
        groups, grid = _video_groups({
            "input_ids": torch.tensor([ids]),
            "video_grid_thw": torch.tensor([[28, 22, 22]])},
            Processor(), frame_count=56)
        self.assertEqual(grid, [28, 22, 22])
        self.assertEqual(len(groups), 28)
        self.assertTrue(all(len(group) == 121 for group in groups))
        with self.assertRaisesRegex(ValueError, "processor_video_group_geometry_mismatch"):
            _video_groups({"input_ids": torch.tensor([ids]),
                "video_grid_thw": torch.tensor([[28, 22, 22]])},
                Processor(), frame_count=64)

    def test_spatial_sham_is_deterministic_equal_dose_and_disjoint(self):
        args = dict(merged_cell_count=121,
            support_offsets=(71, 72, 82, 83, 93, 94),
            seed="spatial-pilot-v1", record_token="row-a")
        first = matched_spatial_sham_offsets(**args)
        self.assertEqual(first, matched_spatial_sham_offsets(**args))
        self.assertEqual(len(first), 6)
        self.assertEqual(len(set(first)), 6)
        self.assertTrue(set(first).isdisjoint(args["support_offsets"]))
        self.assertTrue(all(0 <= cell < 121 for cell in first))

    def test_spatial_sham_fails_when_equal_dose_is_impossible(self):
        with self.assertRaisesRegex(ValueError, "spatial_sham_dose_unavailable"):
            matched_spatial_sham_offsets(merged_cell_count=5,
                support_offsets=(0, 1, 2), seed="s", record_token="r")

    def test_bbox_maps_to_row_major_merged_patch_cells(self):
        # 1408px -> 22x22 14px patches -> 11x11 merged 28px cells.
        result = bbox_to_merged_patch_offsets(
            bbox_xyxy=[128, 128, 256, 256], source_width=1408,
            source_height=1408, grid_height=22, grid_width=22,
            patch_size=14, spatial_merge_size=2)
        self.assertEqual(result["merged_grid_hw"], [11, 11])
        self.assertEqual(result["patch_cell_rc"], ((1, 1),))
        self.assertEqual(result["patch_cell_offsets"], (12,))
        self.assertEqual(result["clipped_area_fraction"], 0.0)

    def test_bbox_maps_exact_multirow_offsets_on_rectangular_grid(self):
        # 448x224 stays unchanged: 32x16 raw patches -> 16x8 merged cells.
        result = bbox_to_merged_patch_offsets(
            bbox_xyxy=[28, 28, 84, 84], source_width=448,
            source_height=224, grid_height=16, grid_width=32,
            patch_size=14, spatial_merge_size=2)
        self.assertEqual(result["merged_grid_hw"], [8, 16])
        self.assertEqual(result["patch_cell_rc"],
                         ((1, 1), (1, 2), (2, 1), (2, 2)))
        self.assertEqual(result["patch_cell_offsets"], (17, 18, 33, 34))

    def test_bbox_half_open_right_bottom_boundary_does_not_add_next_cell(self):
        result = bbox_to_merged_patch_offsets(
            bbox_xyxy=[0, 0, 128, 128], source_width=1408,
            source_height=1408, grid_height=22, grid_width=22,
            patch_size=14, spatial_merge_size=2)
        self.assertEqual(result["patch_cell_rc"], ((0, 0),))

    def test_full_frame_bbox_maps_to_every_merged_cell(self):
        result = bbox_to_merged_patch_offsets(
            bbox_xyxy=[0, 0, 1408, 1408], source_width=1408,
            source_height=1408, grid_height=22, grid_width=22,
            patch_size=14, spatial_merge_size=2)
        self.assertEqual(len(result["patch_cell_offsets"]), 121)
        self.assertEqual(result["patch_cell_offsets"][0], 0)
        self.assertEqual(result["patch_cell_offsets"][-1], 120)

    def test_bbox_small_clipping_is_recorded_and_large_clipping_rejected(self):
        result = bbox_to_merged_patch_offsets(
            bbox_xyxy=[-5, 0, 95, 100], source_width=1408,
            source_height=1408, grid_height=22, grid_width=22,
            patch_size=14, spatial_merge_size=2,
            maximum_clipped_fraction=0.051)
        self.assertAlmostEqual(result["clipped_area_fraction"], 0.05)
        with self.assertRaisesRegex(ValueError, "clipped_fraction_above_threshold"):
            bbox_to_merged_patch_offsets(
                bbox_xyxy=[-20, 0, 80, 100], source_width=1408,
                source_height=1408, grid_height=22, grid_width=22,
                patch_size=14, spatial_merge_size=2,
                maximum_clipped_fraction=0.05)

    def test_bbox_mapper_rejects_nonfinite_degenerate_and_bad_grid(self):
        kwargs = {"source_width": 1408, "source_height": 1408,
                  "grid_height": 22, "grid_width": 22,
                  "patch_size": 14, "spatial_merge_size": 2}
        for bbox in ([0, 0, 0, 4], [0, 0, float("nan"), 4],
                     [0, 4, 4, 3]):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                bbox_to_merged_patch_offsets(bbox_xyxy=bbox, **kwargs)
        with self.assertRaisesRegex(ValueError, "bbox_patch_geometry_invalid"):
            bbox_to_merged_patch_offsets(
                bbox_xyxy=[0, 0, 128, 128], **{**kwargs, "grid_width": 21})

    def test_bbox_maps_to_video_keys_in_temporal_pair_group(self):
        groups = [tuple(range(100, 221)), tuple(range(300, 421))]
        result = bbox_to_video_visual_key_positions(
            bbox_xyxy=[128, 128, 256, 256], frame_rank=3,
            frame_count=4, temporal_patch_size=2,
            visual_token_groups=groups, source_width=1408,
            source_height=1408, grid_height=22, grid_width=22,
            patch_size=14, spatial_merge_size=2)
        self.assertEqual(result["temporal_group_index"], 1)
        self.assertEqual(result["visual_key_positions"], (312,))

    def test_both_frames_in_temporal_pair_map_to_same_visual_keys(self):
        groups = [tuple(range(100, 221))]
        args = dict(bbox_xyxy=[128, 128, 256, 256], frame_count=2,
            temporal_patch_size=2, visual_token_groups=groups,
            source_width=1408, source_height=1408, grid_height=22,
            grid_width=22, patch_size=14, spatial_merge_size=2)
        first = bbox_to_video_visual_key_positions(frame_rank=0, **args)
        second = bbox_to_video_visual_key_positions(frame_rank=1, **args)
        self.assertEqual(first["visual_key_positions"], (112,))
        self.assertEqual(second["visual_key_positions"], (112,))

    def test_bbox_video_mapper_rejects_missing_or_malformed_group_positions(self):
        args = {"bbox_xyxy": [0, 0, 128, 128], "frame_rank": 0,
                "frame_count": 2, "temporal_patch_size": 2,
                "source_width": 1408, "source_height": 1408,
                "grid_height": 22, "grid_width": 22,
                "patch_size": 14, "spatial_merge_size": 2}
        with self.assertRaisesRegex(ValueError, "bbox_video_group_geometry_invalid"):
            bbox_to_video_visual_key_positions(
                visual_token_groups=[], **args)
        with self.assertRaisesRegex(ValueError, "bbox_visual_group_positions_invalid"):
            bbox_to_video_visual_key_positions(
                visual_token_groups=[tuple(range(120))], **args)
        with self.assertRaisesRegex(ValueError, "bbox_visual_group_positions_invalid"):
            bbox_to_video_visual_key_positions(
                visual_token_groups=[tuple(reversed(range(121)))], **args)

    def test_cohort_binding_requires_exact_row_video_and_hash(self):
        row_key = "v1:key"
        import hashlib
        cohort = {"selection": [{"row_key": row_key, "video_id": "v1",
                                 "row_key_sha256": hashlib.sha256(row_key.encode()).hexdigest()}]}
        self.assertEqual(validate_cohort_binding(
            cohort=cohort, row_key=row_key, video_id="v1")["video_id"], "v1")
        with self.assertRaisesRegex(ValueError, "cohort_row_video_binding_mismatch"):
            validate_cohort_binding(cohort=cohort, row_key=row_key, video_id="v2")

    def test_gpu_token_receipt_binds_prompt_positions_and_runtime(self):
        import json
        from run_minerva_ego_anchor_gpu_pilot import sha
        positions = [("baseline", ()), ("object_bbox_support_ablation", (1, 2)),
                     ("equal_dose_same_pair_spatial_sham_ablation", (3, 4)),
                     ("equal_dose_temporal_bbox_sham_ablation", (5, 6))]
        receipt = {"status": "completed_full_prompt_cpu_spatial_token_preflight",
            "cohort_sha256": "c", "preflight_receipt_sha256": "p",
            "row_key_sha256": "r", "video_sha256": "v", "time_map_sha256": "t",
            "annotation_source_sha256": {"x": "y"}, "bias_sign": None,
            "bias_nats": None, "intervention_operation": "key_ablation",
            "processor_video_grid_thw": [28, 22, 22],
            "rendered_prompt_sha256": "prompt", "runtime_source_sha256": {"s": "h"},
            "condition_key_counts": {name: len(keys) for name, keys in positions},
            "condition_key_position_sha256": {name: sha(json.dumps(list(keys)).encode())
                for name, keys in positions}}
        validate_spatial_token_preflight(receipt, cohort_sha256="c",
            preflight_sha256="p", row_key_sha256="r", video_sha256="v",
            time_map_sha256="t", annotation_source_sha256={"x": "y"},
            bias_sign=None, bias_nats=None,
            operation="key_ablation", grid=[28, 22, 22],
            condition_positions=positions, rendered_prompt_sha256="prompt",
            runtime_source_sha256={"s": "h"})
        altered = dict(receipt, rendered_prompt_sha256="other")
        with self.assertRaisesRegex(ValueError, "spatial_token_preflight_binding_mismatch"):
            validate_spatial_token_preflight(altered, cohort_sha256="c",
                preflight_sha256="p", row_key_sha256="r", video_sha256="v",
                time_map_sha256="t", annotation_source_sha256={"x": "y"},
                bias_sign=None, bias_nats=None,
                operation="key_ablation", grid=[28, 22, 22],
                condition_positions=positions, rendered_prompt_sha256="prompt",
                runtime_source_sha256={"s": "h"})

    def test_selects_exact_mask_frame_inside_rationale_timestamp_bin(self):
        anchor = select_rationale_mask_anchor(
            reasoning="At 00:08 the ladle moves (abcdef01).",
            object_id="abcdef01",
            tracks=[{"time_segment": [7.5, 9.0], "masks": ["m1", "m2"]}],
            mask_frame_numbers={"m1": 225, "m2": 240},
            mp4_frame_times_seconds=[i / 30 for i in range(300)])
        self.assertEqual(anchor["mask_frame_number"], 240)
        self.assertEqual(anchor["timestamp_support_interval_seconds"], [8, 9])
        self.assertAlmostEqual(anchor["timestamp_support_center_error_seconds"], 0.5)

    def test_rationale_timestamp_range_selects_interior_mask(self):
        text = "washed chickpeas from 01:17-01:46 (abcdef01)"
        self.assertEqual(parse_rationale_time_intervals(text), [(77, 107)])
        times = [i / 30 for i in range(3300)]
        anchor = select_rationale_mask_anchor(
            reasoning=text, object_id="abcdef01",
            tracks=[{"time_segment": [68.5, 114.0], "masks": ["m1"]}],
            mask_frame_numbers={"m1": 2674},
            mp4_frame_times_seconds=times)
        self.assertEqual(anchor["mask_frame_number"], 2674)
        self.assertEqual(anchor["timestamp_support_interval_seconds"], [77, 107])
        self.assertAlmostEqual(anchor["mask_time_seconds"], 2674 / 30)

    def test_timestamp_range_and_standalone_timestamp_are_both_parsed(self):
        self.assertEqual(parse_rationale_time_intervals(
            "from 01:17–01:46; then at 02:05"), [(77, 107), (125, 126)])

    def test_invalid_timestamp_ranges_do_not_degrade_to_endpoint_bins(self):
        self.assertEqual(parse_rationale_time_intervals("01:46-01:17"), [])
        self.assertEqual(parse_rationale_time_intervals("01:60-02:00"), [])

    def test_rationale_anchor_fails_without_mask_in_timestamp_bin(self):
        with self.assertRaisesRegex(ValueError, "missing_in_timestamp_bin"):
            select_rationale_mask_anchor(
                reasoning="At 00:08 the ladle moves (abcdef01).",
                object_id="abcdef01",
                tracks=[{"time_segment": [7.5, 9.0], "masks": ["m1"]}],
                mask_frame_numbers={"m1": 225},
                mp4_frame_times_seconds=[i / 30 for i in range(300)])

    def test_rationale_anchor_allows_single_frame_track_boundary_rounding(self):
        anchor = select_rationale_mask_anchor(
            reasoning="At 00:07 the ladle moves (abcdef01).",
            object_id="abcdef01",
            tracks=[{"time_segment": [7.51, 9.0], "masks": ["m1"]}],
            mask_frame_numbers={"m1": 225},
            mp4_frame_times_seconds=[i / 30 for i in range(300)])
        self.assertEqual(anchor["mask_frame_number"], 225)
        self.assertEqual(anchor["track_boundary_tolerance_seconds"], 0.05)

    def test_range_anchor_does_not_exceed_track_boundary_tolerance(self):
        with self.assertRaisesRegex(ValueError, "missing_in_timestamp_bin"):
            select_rationale_mask_anchor(
                reasoning="from 00:07-00:09 the ladle moves (abcdef01)",
                object_id="abcdef01",
                tracks=[{"time_segment": [7.0, 9.0], "masks": ["m1"]}],
                mask_frame_numbers={"m1": 272},  # 9.0667s exceeds track end + 0.05s
                mp4_frame_times_seconds=[i / 30 for i in range(300)])

    def test_selects_and_redacts_track_supported_dataset_line(self):
        target, count = select_supported_target(
            reasoning=("1. At 00:05 it stays put (second ladle-abcdef01).\n"
                       "2. At 00:08 I move the ladle (second ladle-abcdef01)."),
            object_name="second ladle", object_id="abcdef01",
            track_segments=[[7.5, 9.0]])
        self.assertEqual(count, 1)
        self.assertEqual(target, "At 00:08 I move the ladle.")

    def test_generation_prompt_uses_row_target_alias_not_fixed_mug(self):
        row = {"question": "What happens to the tool?",
               **{f"answer_choice_{i}": f"choice {i}" for i in range(5)}}
        prompt = build_grounding_prompt(row, "stainless steel ladle")
        self.assertIn("movements of the stainless steel ladle", prompt)
        self.assertIn("Repeat 'ladle'", prompt)
        self.assertNotIn("Repeat 'mug'", prompt)

    def test_generated_anchor_precision_and_segment_recall(self):
        score = score_generated_movement_anchors(
            text=("At 00:05 the mug sits on the counter. "
                  "At 00:07 the mug is carried to the sink. "
                  "At 01:27 the cup is rinsed."),
            track_segments=[[7.881, 11.0729], [87.043, 94.31355]])
        self.assertEqual(score["timestamp_anchor_count"], 3)
        self.assertEqual(score["aligned_anchor_count"], 2)
        self.assertAlmostEqual(score["anchor_precision"], 2/3)
        self.assertEqual(score["covered_segment_count"], 2)
        self.assertEqual(score["segment_recall"], 1.0)

    def test_timestamp_is_a_one_second_bin(self):
        self.assertEqual(parse_timestamp_bins("at 01:27 and 00:07"),
                         [(7, 8), (87, 88)])

    def test_generated_anchor_scorer_accepts_configured_object_name(self):
        score = score_generated_movement_anchors(
            text="At 00:08 the ladle is moved.",
            track_segments=[[7.5, 9.0]], object_names=["ladle"])
        self.assertEqual(score["timestamp_anchor_count"], 1)
        self.assertEqual(score["aligned_anchor_count"], 1)
        self.assertEqual(score["covered_segment_count"], 1)

    def test_generated_anchor_scorer_rejects_object_name_substring(self):
        score = score_generated_movement_anchors(
            text="At 00:08 the mallet is moved.",
            track_segments=[[7.5, 9.0]], object_names=["mug"])
        self.assertEqual(score["aligned_anchor_count"], 0)
        self.assertEqual(score["covered_segment_count"], 0)

    def test_rejects_invalid_seconds(self):
        self.assertEqual(parse_timestamp_bins("at 01:67"), [])

    def test_rejects_hour_timestamp_substring(self):
        self.assertEqual(parse_timestamp_bins("at 1:02:03"), [])

    def test_bin_overlaps_track_even_when_displayed_second_starts_early(self):
        self.assertTrue(timestamp_bins_overlap_segment([(7, 8)], [7.881, 11.0729]))
        self.assertTrue(timestamp_bins_overlap_segment([(87, 88)], [87.043, 94.31355]))
        self.assertFalse(timestamp_bins_overlap_segment([(5, 6)], [7.881, 11.0729]))
        self.assertFalse(timestamp_bins_overlap_segment([(6, 7)], [7, 8]))

    def test_only_object_linked_lines_with_overlapping_timestamps_survive(self):
        oid = "opaque-object-id"
        lines = supported_rationale_lines(
            reasoning=(f"At 00:05 the object ({oid}) is still.\n"
                       f"At 00:07 I move it ({oid}).\n"
                       "At 00:08 another object moves."),
            object_tracks={oid: [[7.881, 11.0729]]})
        self.assertEqual(lines, [f"At 00:07 I move it ({oid})."])

    def test_object_id_requires_hex_token_boundaries(self):
        oid = "abcdef0123456789"
        self.assertEqual(supported_rationale_lines(
            reasoning=f"00:07 ({oid}f)", object_tracks={oid: [[7, 8]]}), [])


class TemporalGroupTests(unittest.TestCase):
    def test_cpu_preflight_processes_exact_qwen_temporal_pair(self):
        import numpy as np
        from PIL import Image
        from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

        cap = 128 * 28 * 28
        processor = Qwen2VLVideoProcessor(min_pixels=cap, max_pixels=cap)
        frames = [Image.fromarray(np.zeros((224, 448, 3), dtype=np.uint8) + index)
                  for index in range(4)]
        pair_start, grid, pair = process_anchor_frame_pair(
            frames=frames, anchor_rank=3, video_processor=processor)
        self.assertEqual(pair_start, 2)
        self.assertEqual(len(pair), 2)
        self.assertEqual(grid, [1, 16, 32])

    def test_anchor_centered_clip_has_fixed_rate_frames_and_exact_anchor(self):
        times = [i / 30 for i in range(3000)]
        clip = build_anchor_centered_sample_indices(
            mp4_frame_times_seconds=times, anchor_frame_number=1200)
        self.assertEqual(len(clip["sample_frame_indices"]), 64)
        self.assertEqual(clip["sample_frame_indices"],
                         sorted(set(clip["sample_frame_indices"])) )
        self.assertEqual(clip["anchor_sample_rank"], 32)
        self.assertAlmostEqual(clip["anchor_time_error_seconds"], 0.0)
        self.assertLessEqual(clip["maximum_grid_time_error_seconds"], 1 / 30)

    def test_anchor_centered_clip_shifts_at_start_of_video(self):
        times = [i / 30 for i in range(3000)]
        clip = build_anchor_centered_sample_indices(
            mp4_frame_times_seconds=times, anchor_frame_number=10)
        self.assertEqual(clip["clip_start_source_seconds"], 0.0)
        self.assertLess(clip["anchor_sample_rank"], 32)

    def test_exact_anchor_sample_inserts_annotation_frame_and_keeps_count(self):
        times = [i / 30 for i in range(3000)]
        initial = build_anchor_centered_sample_indices(
            mp4_frame_times_seconds=times, anchor_frame_number=1505,
            sample_count=64, sample_fps=2.0)
        result = include_exact_anchor_sample(
            sample_frame_indices=initial["sample_frame_indices"],
            mp4_frame_times_seconds=times, anchor_frame_number=1505)
        self.assertEqual(len(result["sample_frame_indices"]), 64)
        self.assertEqual(result["sample_frame_indices"],
                         sorted(set(result["sample_frame_indices"])))
        self.assertEqual(result["sample_frame_indices"][result["anchor_sample_rank"]],
                         1505)
        self.assertEqual(result["anchor_time_error_seconds"], 0.0)
        self.assertGreater(result["maximum_sample_gap_seconds"], 0.5)

    def test_exact_anchor_sample_rejects_unreplaceable_or_invalid_schedules(self):
        with self.assertRaisesRegex(ValueError, "exact_anchor_outside_sampled_clip"):
            include_exact_anchor_sample(
                sample_frame_indices=[10, 12],
                mp4_frame_times_seconds=list(range(20)), anchor_frame_number=5)
        with self.assertRaisesRegex(ValueError, "exact_anchor_sampling_geometry_invalid"):
            include_exact_anchor_sample(
                sample_frame_indices=[1, [2]],
                mp4_frame_times_seconds=list(range(5)), anchor_frame_number=3)

    def test_uniform_indices_preserve_exact_annotation_anchor(self):
        indices = build_uniform_sample_indices(
            total_frames=1000, sample_count=64, anchor_frames=[240])
        self.assertEqual(len(indices), 64)
        self.assertEqual(len(set(indices)), 64)
        self.assertEqual(indices, sorted(indices))
        self.assertIn(240, indices)

    def test_uniform_indices_reject_out_of_range_anchor(self):
        with self.assertRaisesRegex(ValueError, "sampling_geometry_invalid"):
            build_uniform_sample_indices(total_frames=100, sample_count=64,
                                        anchor_frames=[100])

    def test_sampled_evidence_alignment_accepts_close_track_samples(self):
        result = validate_sampled_evidence_alignment(
            mapping={"mapped_mask_frames": [
                {"absolute_time_error_seconds": 0.25}],
                "evidence_temporal_groups": [2]},
            track_groups=[1, 2], maximum_error_seconds=1.0)
        self.assertEqual(result["track_overlapping_evidence_group_count"], 1)

    def test_sampled_evidence_alignment_rejects_large_sampling_error(self):
        with self.assertRaisesRegex(ValueError, "time_error_above_threshold"):
            validate_sampled_evidence_alignment(
                mapping={"mapped_mask_frames": [
                    {"absolute_time_error_seconds": 1.01}],
                    "evidence_temporal_groups": [2]},
                track_groups=[2])

    def test_sampled_evidence_alignment_rejects_evidence_outside_track_samples(self):
        with self.assertRaisesRegex(ValueError, "not_sampled_inside_track"):
            validate_sampled_evidence_alignment(
                mapping={"mapped_mask_frames": [
                    {"absolute_time_error_seconds": 0.1}],
                    "evidence_temporal_groups": [2]},
                track_groups=[3])

    def test_track_group_exclusion_uses_any_sample_not_group_center(self):
        times = [0, 5, 10, 15, 83, 88, 100, 105]
        self.assertEqual(track_overlapping_temporal_groups(
            sampled_times_seconds=times, temporal_patch_size=2,
            track_segments=[[7.881, 11.0729], [87.043, 94.31355]]),
            [1, 2])

    def test_track_groups_allow_single_frame_pts_rounding_at_boundaries(self):
        self.assertEqual(track_overlapping_temporal_groups(
            sampled_times_seconds=[7.5, 8.0], temporal_patch_size=2,
            track_segments=[[7.51, 8.5]]), [0])

    def test_mask_frame_maps_by_official_row_and_decoded_pts(self):
        frame_times = [i / 30 for i in range(120)]
        sampled = [
            {"decoded_frame_index": i, "relative_presentation_time": [i, 30]}
            for i in (0, 30, 60, 90)
        ]
        result = evidence_temporal_groups(
            mask_frame_numbers=[31, 89], mp4_frame_times_seconds=frame_times,
            sampled_frames=sampled, temporal_patch_size=2, temporal_step_count=2)
        self.assertEqual(result["evidence_temporal_groups"], [0, 1])
        self.assertEqual([r["sampled_frame_index"] for r in result["mapped_mask_frames"]],
                         [30, 90])

    def test_rejects_pts_mapping_drift(self):
        with self.assertRaisesRegex(ValueError, "sampled_frame_pts_map_mismatch"):
            evidence_temporal_groups(
                mask_frame_numbers=[1], mp4_frame_times_seconds=[0, 1, 2],
                sampled_frames=[{"decoded_frame_index": 0,
                                 "relative_presentation_time": [2, 1]}],
                temporal_patch_size=1, temporal_step_count=1)

    def test_control_groups_are_stable_and_exclude_tracks(self):
        args = {"temporal_step_count": 8, "evidence_groups": [2, 5],
                "excluded_groups": [1, 3, 4], "seed": "pilot-v1",
                "record_token": "row-token"}
        first = matched_control_groups(**args)
        self.assertEqual(first, matched_control_groups(**args))
        self.assertEqual(len(first), 2)
        self.assertFalse(set(first) & {1, 2, 3, 4, 5})

    def test_control_fails_when_no_equal_dose_set_exists(self):
        with self.assertRaisesRegex(ValueError, "insufficient_control_groups"):
            matched_control_groups(temporal_step_count=3, evidence_groups=[0, 1],
                                   excluded_groups=[2], seed="s", record_token="r")


if __name__ == "__main__":
    unittest.main()
