"""CPU-only integration test for current legacy video metadata reconstruction."""

import os
from pathlib import Path
import tempfile
import unittest

from audit_legacy_video_time_metadata import (
    MODEL_REVISION,
    compare_metadata_paths,
    make_synthetic_jpegs,
    MAX_PIXELS,
    MIN_PIXELS,
)


MODEL_PATH = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"
    / MODEL_REVISION
)


@unittest.skipUnless(MODEL_PATH.is_dir(), "pinned processor/config are not cached")
class LegacyVideoMetadataAuditTests(unittest.TestCase):
    def test_legacy_and_explicit_paths_keep_pixels_tokens_and_grid_but_change_time_ids(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoConfig, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(MODEL_PATH), local_files_only=True,
            min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
        config = AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
        with tempfile.TemporaryDirectory(prefix="legacy_video_metadata_test_") as temp:
            paths, fixture_hash = make_synthetic_jpegs(Path(temp) / "frames")
            duplicate_paths, duplicate_hash = make_synthetic_jpegs(Path(temp) / "duplicate")
            self.assertEqual(fixture_hash, duplicate_hash)
            self.assertEqual(
                [Path(path).read_bytes() for path in paths],
                [Path(path).read_bytes() for path in duplicate_paths],
            )
            comparison = compare_metadata_paths(processor, config, paths)

        legacy = comparison["legacy_build_inputs"]
        explicit = comparison["explicit_qwen_metadata"]
        identity = comparison["identity_checks"]
        self.assertEqual(legacy["grid_thw"], [8, 32, 32])
        self.assertEqual(explicit["grid_thw"], legacy["grid_thw"])
        self.assertAlmostEqual(legacy["second_per_grid_seconds"], 2 / 24, places=6)
        self.assertAlmostEqual(explicit["second_per_grid_seconds"], 1.0, places=6)
        self.assertAlmostEqual(identity["explicit_video_metadata_fps"], 2.0)
        self.assertEqual(identity["explicit_video_metadata_frame_count"], 16)
        self.assertFalse(identity["explicit_do_sample_frames"])
        for key in (
            "input_token_ids_equal", "video_grid_thw_equal", "video_token_count_equal",
            "processor_pixel_value_keys_equal", "processor_pixel_values_equal",
            "position_id_tensor_shapes_equal",
        ):
            self.assertTrue(identity[key], key)
        self.assertEqual(len(legacy["temporal_anchors"]), 8)
        self.assertEqual(len(set(legacy["temporal_anchors"])), 1)
        explicit_steps = [b - a for a, b in zip(
            explicit["temporal_anchors"], explicit["temporal_anchors"][1:]
        )]
        self.assertEqual(explicit_steps, [int(config.vision_config.tokens_per_second)] * 7)
        self.assertGreater(comparison["changed_position_id_entries"], 0)


if __name__ == "__main__":
    unittest.main()
