import unittest
from types import SimpleNamespace

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

from qwen_temporal_geometry import (
    qwen_temporal_geometry,
    replace_video_temporal_rope_row,
)


class QwenTemporalGeometryTests(unittest.TestCase):
    def test_endpoint_spans_are_not_nearest_sample_duration_cells(self):
        geometry = qwen_temporal_geometry(
            [10.0, 10.375, 10.75, 11.125],
            2,
            tokens_per_second=2,
            duration_seconds=12.0,
        )
        self.assertEqual(
            geometry["sampled_frame_endpoint_spans_seconds"],
            ((10.0, 10.375), (10.75, 11.125)),
        )
        self.assertEqual(
            geometry["duration_coverage_cells_seconds"],
            ((0.0, 10.5625), (10.5625, 12.0)),
        )

    def test_fractional_interval_exposes_installed_truncation_vs_449(self):
        timestamps = [index * 0.375 for index in range(8)]
        geometry = qwen_temporal_geometry(
            timestamps, 2, tokens_per_second=2, video_start_position=5
        )
        self.assertAlmostEqual(geometry["second_per_grid_seconds"], 0.75)
        self.assertEqual(
            geometry["transformers_5_17_temporal_rope_ticks_pretruncate"],
            (5, 5, 5, 5),
        )
        self.assertEqual(
            geometry["transformers_4_49_0_temporal_rope_ticks_fractional"],
            (5, 6, 8, 9),
        )

    def test_integral_interval_matches_both_rope_formulas(self):
        geometry = qwen_temporal_geometry(
            [0.0, 1.0, 2.0, 3.0],
            2,
            tokens_per_second=2,
            video_start_position=7,
        )
        expected = (7, 11)
        self.assertEqual(
            geometry["transformers_5_17_temporal_rope_ticks_pretruncate"], expected
        )
        self.assertEqual(
            geometry["transformers_4_49_0_temporal_rope_ticks_fractional"],
            expected,
        )

    def test_rope_ticks_are_model_coordinates_not_absolute_timestamps(self):
        geometry = qwen_temporal_geometry(
            [100.0, 101.0, 102.0, 103.0], 2, tokens_per_second=2
        )
        self.assertEqual(
            geometry["sampled_frame_endpoint_spans_seconds"],
            ((100.0, 101.0), (102.0, 103.0)),
        )
        self.assertEqual(
            geometry["transformers_4_49_0_temporal_rope_ticks_fractional"],
            (0, 4),
        )

    def test_rejects_unrepresentable_sampling_and_processor_interval_mismatch(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            qwen_temporal_geometry([0.0, 1.0, 1.0, 2.0], 2, tokens_per_second=2)
        with self.assertRaisesRegex(ValueError, "single FPS"):
            qwen_temporal_geometry([0.0, 1.0, 2.1, 3.0], 2, tokens_per_second=2)
        with self.assertRaisesRegex(ValueError, "second_per_grid"):
            qwen_temporal_geometry(
                [0.0, 1.0, 2.0, 3.0],
                2,
                tokens_per_second=2,
                second_per_grid=1.5,
            )

    def test_rejects_bad_patch_divisibility_and_duration_bounds(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            qwen_temporal_geometry([0.0, 1.0, 2.0], 2, tokens_per_second=2)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            qwen_temporal_geometry(
                [0.0, 1.0, 2.0, 3.0],
                2,
                tokens_per_second=2,
                duration_seconds=2.5,
            )

    def test_installed_qwen_positions_match_grid_with_fractional_time_interval(self):
        # Exercise the installed 5.17 get_rope_index implementation on CPU with
        # the same 2x2 spatial merge and 2 tokens/second as the pinned checkpoint.
        class ModelGeometryMethods:
            get_rope_index = Qwen2_5_VLModel.get_rope_index
            get_vision_position_ids = Qwen2_5_VLModel.get_vision_position_ids

        model = ModelGeometryMethods()
        model.config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=2)
        )
        prefix_tokens = 3
        temporal_steps = 4
        tokens_per_step = 4  # (4 // 2) * (4 // 2) spatial positions
        video_tokens = temporal_steps * tokens_per_step
        total = prefix_tokens + video_tokens + 1
        input_ids = torch.zeros((1, total), dtype=torch.long)
        mm_types = torch.zeros((1, total), dtype=torch.int32)
        mm_types[:, prefix_tokens:prefix_tokens + video_tokens] = 2

        positions, _ = model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_types,
            video_grid_thw=torch.tensor([[temporal_steps, 4, 4]]),
            second_per_grid_ts=torch.tensor([0.75]),
        )
        actual_video_time_ticks = positions[0, 0,
            prefix_tokens:prefix_tokens + video_tokens].reshape(
                temporal_steps, tokens_per_step
            )
        self.assertTrue(torch.equal(
            actual_video_time_ticks,
            torch.full((temporal_steps, tokens_per_step), prefix_tokens),
        ))

        # The companion helper makes the discrepancy with 4.49.0/main explicit;
        # these are implementation diagnostics, not seconds or evidence labels.
        geometry = qwen_temporal_geometry(
            [index * 0.375 for index in range(8)],
            2,
            tokens_per_second=2,
            video_start_position=prefix_tokens,
        )
        self.assertEqual(
            geometry["transformers_5_17_temporal_rope_ticks_pretruncate"],
            tuple(int(row[0].item()) for row in actual_video_time_ticks),
        )
        self.assertEqual(
            geometry["transformers_4_49_0_temporal_rope_ticks_fractional"],
            (3, 4, 6, 7),
        )

        native_rows = positions[:, 0, :].tolist()
        adjusted_rows = replace_video_temporal_rope_row(
            native_rows,
            video_token_spans=((prefix_tokens, prefix_tokens + video_tokens),),
            image_token_spans=(),
            attention_mask=[1] * total,
            video_grid_thw=(temporal_steps, 4, 4),
            spatial_merge_size=2,
            second_per_grid_seconds=0.75,
            tokens_per_second=2,
        )
        expected_adjusted = torch.tensor(adjusted_rows)
        self.assertEqual(
            tuple(int(x) for x in expected_adjusted[0, prefix_tokens:
                prefix_tokens + video_tokens].reshape(temporal_steps, tokens_per_step)[:, 0]),
            (3, 4, 6, 7),
        )
        self.assertTrue(torch.equal(expected_adjusted[1:], positions[1:, 0, :]))
        self.assertTrue(torch.equal(
            expected_adjusted[0, :prefix_tokens], positions[0, 0, :prefix_tokens]
        ))
        self.assertTrue(torch.equal(
            expected_adjusted[0, prefix_tokens + video_tokens:],
            positions[0, 0, prefix_tokens + video_tokens:],
        ))

    def test_temporal_replacement_fails_closed_on_unsupported_sequences(self):
        native = (
            (0, 1, 2, 2, 2, 2, 5),
            (0, 1, 2, 2, 3, 3, 5),
            (0, 1, 2, 3, 2, 3, 5),
        )
        common = dict(
            native_position_ids=native,
            video_grid_thw=(1, 2, 2),
            spatial_merge_size=1,
            second_per_grid_seconds=1.0,
            tokens_per_second=2,
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            replace_video_temporal_rope_row(
                **common, video_token_spans=((2, 6), (6, 7)),
                image_token_spans=(), attention_mask=[1] * 7,
            )
        with self.assertRaisesRegex(ValueError, "image inputs"):
            replace_video_temporal_rope_row(
                **common, video_token_spans=((2, 6),),
                image_token_spans=((1, 2),), attention_mask=[1] * 7,
            )
        with self.assertRaisesRegex(ValueError, "padding"):
            replace_video_temporal_rope_row(
                **common, video_token_spans=((2, 6),),
                image_token_spans=(), attention_mask=[1, 1, 1, 1, 1, 1, 0],
            )
        with self.assertRaisesRegex(ValueError, "span length"):
            replace_video_temporal_rope_row(
                **common, video_token_spans=((2, 5),),
                image_token_spans=(), attention_mask=[1] * 7,
            )

    def test_temporal_replacement_requires_python_iterables_and_positive_grid_axes(self):
        native = (
            (0, 1, 2, 2, 2, 2, 5),
            (0, 1, 2, 2, 3, 3, 5),
            (0, 1, 2, 3, 2, 3, 5),
        )
        common = dict(
            video_token_spans=((2, 6),), image_token_spans=(),
            attention_mask=[1] * 7, spatial_merge_size=1,
            second_per_grid_seconds=1.0, tokens_per_second=2,
        )
        # The public API stays Torch-free: callers convert model tensors with
        # .tolist(), as the GPU comparison runner does.
        with self.assertRaisesRegex(ValueError, "native position ID must be an integer"):
            replace_video_temporal_rope_row(
                **common, native_position_ids=torch.tensor(native),
                video_grid_thw=(1, 2, 2),
            )

        for grid in ((0, 2, 2), (1, 0, 2), (1, 2, 0)):
            with self.subTest(grid=grid):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    replace_video_temporal_rope_row(
                        **common, native_position_ids=native, video_grid_thw=grid,
                    )

    def test_temporal_replacement_rejects_varying_ids_within_grid_slice(self):
        native = (
            (0, 1, 2, 3, 2, 2, 5),
            (0, 1, 2, 2, 3, 3, 5),
            (0, 1, 2, 3, 2, 3, 5),
        )
        with self.assertRaisesRegex(ValueError, "constant within each video grid slice"):
            replace_video_temporal_rope_row(
                native,
                video_token_spans=((2, 6),), image_token_spans=(),
                attention_mask=[1] * 7, video_grid_thw=(1, 2, 2),
                spatial_merge_size=1, second_per_grid_seconds=1.0,
                tokens_per_second=2,
            )


if __name__ == "__main__":
    unittest.main()
