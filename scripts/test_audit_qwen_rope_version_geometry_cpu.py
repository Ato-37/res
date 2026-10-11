"""CPU tests for the frozen synthetic Qwen RoPE-version geometry fixtures."""

from __future__ import annotations

import unittest

from audit_qwen_rope_version_geometry_cpu import (
    FIXED_CASES,
    LEGACY_SOURCE,
    LEGACY_SOURCE_SHA256,
    _case_geometry,
    _simple_model,
    extract_legacy_get_rope_index,
)


class RopeVersionGeometryCpuTests(unittest.TestCase):
    def test_fixed_cases_are_exact_and_predeclared(self):
        self.assertEqual(
            [(case["grid_thw"], case["prefix_tokens_including_vision_start"], case["suffix_tokens"], case["seconds_per_grid"])
             for case in FIXED_CASES],
            [
                ((8, 4, 4), 2, 2, 0.75),
                ((8, 4, 4), 2, 2, 5.0),
                ((2, 24, 46), 15, 17, 199.9998016357422),
            ],
        )
        fixture_lengths = [
            case["prefix_tokens_including_vision_start"]
            + case["grid_thw"][0]
            * (case["grid_thw"][1] // 2)
            * (case["grid_thw"][2] // 2)
            + case["suffix_tokens"]
            for case in FIXED_CASES
        ]
        self.assertEqual(fixture_lengths, [36, 36, 584])
        self.assertEqual(
            [case["prefix_tokens_including_vision_start"] for case in FIXED_CASES], [2, 2, 15]
        )

    def test_hash_mismatch_is_rejected_before_source_parsing(self):
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            extract_legacy_get_rope_index(b"this is not Python source", expected_sha256=LEGACY_SOURCE_SHA256)

    def test_hash_verified_legacy_temporal_ticks_match_manual_formula(self):
        import torch

        legacy_method = extract_legacy_get_rope_index(
            LEGACY_SOURCE.read_bytes(), expected_sha256=LEGACY_SOURCE_SHA256
        )
        expected = (
            ([2, 3, 5, 6, 8, 9, 11, 12], 4),
            ([2, 12, 22, 32, 42, 52, 62, 72], 4),
        )
        for case, (ticks, copies_per_tick) in zip(FIXED_CASES[:2], expected):
            t, h, w = case["grid_thw"]
            video_count = t * (h // 2) * (w // 2)
            ids = torch.tensor([[
                *range(101, 101 + case["prefix_tokens_including_vision_start"] - 1),
                900_001,
                *([900_002] * video_count),
                *range(201, 201 + case["suffix_tokens"]),
            ]], dtype=torch.long)
            grid = torch.tensor([case["grid_thw"]], dtype=torch.long)
            seconds = torch.tensor([case["seconds_per_grid"]], dtype=torch.float32)
            self_obj = type("LegacyConfig", (), {
                "config": type("Config", (), {
                    "vision_config": type("VisionConfig", (), {
                        "spatial_merge_size": 2,
                        "tokens_per_second": 2,
                    })(),
                    "image_token_id": 900_003,
                    "video_token_id": 900_002,
                    "vision_start_token_id": 900_001,
                })()
            })()
            with torch.device("cpu"):
                position_ids, _ = legacy_method(
                    self_obj, input_ids=ids, video_grid_thw=grid, second_per_grid_ts=seconds
                )
            video_start = case["prefix_tokens_including_vision_start"]
            observed = position_ids[0, 0, video_start:video_start + video_count].tolist()
            manual = [tick for tick in ticks for _ in range(copies_per_tick)]
            self.assertEqual(observed, manual)

    def test_integer_control_temporal_matches_while_text_cursor_differs(self):
        import torch
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

        legacy_method = extract_legacy_get_rope_index(
            LEGACY_SOURCE.read_bytes(), expected_sha256=LEGACY_SOURCE_SHA256
        )
        installed_model = _simple_model(torch, Qwen2_5_VLModel)
        with torch.device("cpu"):
            record = _case_geometry(
                FIXED_CASES[1], legacy_method=legacy_method, installed_model=installed_model, torch=torch
            )
        self.assertEqual(record["video_changed_count_by_axis_legacy_vs_installed"]["temporal"], 0)
        self.assertGreater(
            record["suffix_text_changed_count_by_axis_legacy_vs_installed"]["temporal"], 0
        )
        self.assertFalse(record["candidate_temporal_only_matches_full_legacy_positions"])


if __name__ == "__main__":
    unittest.main()
