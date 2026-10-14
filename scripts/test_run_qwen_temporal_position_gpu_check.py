"""CPU-only integration check for the deterministic subsecond processor fixture."""

import os
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

from qwen_temporal_geometry import qwen_temporal_geometry, replace_video_temporal_rope_row
from run_qwen_temporal_position_gpu_check import (
    make_synthetic_subsecond_fixture,
    validate_synthetic_subsecond_geometry,
    assert_fractional_float32_compatibility,
)


REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_PATH = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"
    / REVISION
)


class FractionalPrecisionGateTests(unittest.TestCase):
    def test_declared_runtime_fixtures_are_compatible(self):
        assert_fractional_float32_compatibility([15,16],.75,2)
        assert_fractional_float32_compatibility([15,414],199.9998016357422,2)

    def test_rounding_boundary_refuses_incompatible_python_math(self):
        with self.assertRaisesRegex(RuntimeError,'arithmetic differ'):
            assert_fractional_float32_compatibility([15,15,16,17],.49999999,2)


@unittest.skipUnless(MODEL_PATH.is_dir(), "pinned processor/config are not cached")
class SyntheticSubsecondProcessorTests(unittest.TestCase):
    def test_actual_processor_and_installed_rope_method_on_fixed_subsecond_fixture(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from qwen_vl_utils import process_vision_info
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

        frames, timestamps, fps, pixel_sha256 = make_synthetic_subsecond_fixture()
        duplicate_frames, duplicate_times, duplicate_fps, duplicate_hash = make_synthetic_subsecond_fixture()
        self.assertEqual(pixel_sha256, duplicate_hash)
        self.assertEqual(timestamps, duplicate_times)
        self.assertEqual(fps, duplicate_fps)
        self.assertEqual(len(pixel_sha256), 64)
        self.assertEqual([frame.tobytes() for frame in frames],
                         [frame.tobytes() for frame in duplicate_frames])

        processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        config = AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames, "sample_fps": fps},
            {"type": "text", "text": "Describe the visible scene briefly. Use only visual information."},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, return_video_metadata=True)
        self.assertEqual(len(videos), 1)
        video_tensor, video_metadata = videos[0]
        self.assertEqual(video_metadata['frames_indices'], [0,1,2,3])
        self.assertEqual(video_metadata['fps'], fps)
        self.assertEqual([i/video_metadata['fps'] for i in video_metadata['frames_indices']],timestamps)
        batch = processor(text=[text], images=images, videos=[video_tensor],
            video_metadata=[video_metadata], do_sample_frames=False,
            cap_pixels_per_frame=True, padding=True, return_tensors="pt")

        grid = batch["video_grid_thw"][0].tolist()
        seconds = float(batch["second_per_grid_ts"][0])
        self.assertEqual(int(grid[0]), 2)
        self.assertAlmostEqual(seconds, 0.75, places=6)

        class ModelGeometryMethods:
            get_rope_index = Qwen2_5_VLModel.get_rope_index
            get_vision_position_ids = Qwen2_5_VLModel.get_vision_position_ids

        model = ModelGeometryMethods()
        model.config = SimpleNamespace(vision_config=config.vision_config)
        native, _ = model.get_rope_index(
            input_ids=batch["input_ids"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            video_grid_thw=batch["video_grid_thw"],
            second_per_grid_ts=batch["second_per_grid_ts"],
            attention_mask=batch["attention_mask"],
        )
        video_token_id = processor.video_token_id
        positions = (batch["input_ids"][0] == video_token_id).nonzero(as_tuple=True)[0].tolist()
        self.assertTrue(positions)
        self.assertEqual(positions, list(range(positions[0], positions[-1] + 1)))
        merge = int(processor.video_processor.merge_size)
        candidate_rows = replace_video_temporal_rope_row(
            native[:, 0, :].tolist(),
            video_token_spans=((positions[0], positions[-1] + 1),),
            image_token_spans=(),
            attention_mask=batch["attention_mask"][0].tolist(),
            video_grid_thw=grid,
            spatial_merge_size=merge,
            second_per_grid_seconds=seconds,
            tokens_per_second=int(config.vision_config.tokens_per_second),
        )
        candidate = torch.tensor(candidate_rows, dtype=native.dtype).unsqueeze(1)
        tokens_per_step = (grid[1] // merge) * (grid[2] // merge)
        native_anchors = [int(native[0, 0, positions[0] + i * tokens_per_step]) for i in range(2)]
        fractional_anchors = [int(candidate[0, 0, positions[0] + i * tokens_per_step]) for i in range(2)]
        validate_synthetic_subsecond_geometry(
            timestamps, fps, grid, seconds, native_anchors, fractional_anchors,
            int(config.vision_config.tokens_per_second),
        )

        # Check the exact fixture against float32 arithmetic only; 0.75 is
        # exactly representable, so this test does not imply universal precision.
        torch_ticks = (torch.arange(2, dtype=torch.float32)
            * torch.tensor(seconds, dtype=torch.float32)
            * int(config.vision_config.tokens_per_second)).long() + native_anchors[0]
        self.assertEqual(tuple(torch_ticks.tolist()), tuple(fractional_anchors))
        geometry = qwen_temporal_geometry(
            timestamps, int(processor.video_processor.temporal_patch_size),
            tokens_per_second=int(config.vision_config.tokens_per_second),
            second_per_grid=seconds, video_start_position=native_anchors[0],
        )
        self.assertEqual(
            geometry["transformers_4_49_0_temporal_rope_ticks_fractional"],
            tuple(int(value) for value in torch_ticks.tolist()),
        )
        self.assertTrue(torch.equal(native[1:], candidate[1:]))


if __name__ == "__main__":
    unittest.main()
