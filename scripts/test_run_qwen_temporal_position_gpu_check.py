"""CPU-only integration check for the deterministic subsecond processor fixture."""

import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from qwen_temporal_geometry import qwen_temporal_geometry, replace_video_temporal_rope_row
from run_qwen_temporal_position_gpu_check import (
    legacy_pretruncate_position_ids,
    make_synthetic_subsecond_fixture,
    validate_synthetic_subsecond_geometry,
    validate_synthetic_grid,
    assert_fractional_float32_compatibility,
    host_memory_snapshot,
    process_peak_rss_bytes,
)


REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_PATH = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"
    / REVISION
)


class FractionalPrecisionGateTests(unittest.TestCase):
    def test_linux_host_memory_snapshot_reads_kib_as_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            path.write_text(
                "MemTotal:       15728640 kB\nMemFree: 1024 kB\nMemAvailable: 12582912 kB\n",
                encoding="ascii",
            )
            self.assertEqual(host_memory_snapshot(path), {
                "total_bytes": 15 * 1024**3,
                "available_bytes": 12 * 1024**3,
            })

    def test_linux_host_memory_snapshot_rejects_malformed_counters(self):
        invalid = (
            "MemTotal: 1 kB\nMemTotal: 2 kB\nMemAvailable: 1 kB\n",
            "MemTotal: 1 MB\nMemAvailable: 1 kB\n",
            "MemTotal: 1 kB\nMemAvailable: 1\n",
            "MemTotal: 1 kB\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            for contents in invalid:
                path.write_text(contents, encoding="ascii")
                with self.subTest(contents=contents), self.assertRaises(RuntimeError):
                    host_memory_snapshot(path)

    def test_process_peak_rss_kib_is_converted_to_bytes(self):
        with patch("run_qwen_temporal_position_gpu_check.resource.getrusage",
                   return_value=SimpleNamespace(ru_maxrss=123)):
            self.assertEqual(process_peak_rss_bytes(), 123 * 1024)

    def test_synthetic_16_frame_geometry_is_deterministic_and_complete(self):
        frames, timestamps, fps, digest = make_synthetic_subsecond_fixture(16)
        again = make_synthetic_subsecond_fixture(16)
        self.assertEqual((len(frames), len(timestamps)), (16, 16))
        self.assertEqual([frame.getpixel((0, 0)) for frame in frames[:4]],
                         [(17, 43, 89), (61, 97, 131), (109, 151, 173), (197, 211, 229)])
        self.assertEqual(timestamps, [i / fps for i in range(16)])
        self.assertEqual((digest, timestamps, fps), (again[3], again[1], again[2]))
        anchors = [15 + int(i * 0.75 * 2) for i in range(8)]
        validate_synthetic_subsecond_geometry(
            timestamps, fps, (8, 32, 32), 0.75, anchors, anchors, 2,
            expected_frames=16,
        )

    def test_synthetic_32_frame_geometry_is_deterministic_and_complete(self):
        frames, timestamps, fps, digest = make_synthetic_subsecond_fixture(32)
        again = make_synthetic_subsecond_fixture(32)
        self.assertEqual((len(frames), len(timestamps)), (32, 32))
        self.assertEqual(timestamps, [i / fps for i in range(32)])
        self.assertEqual((digest, timestamps, fps), (again[3], again[1], again[2]))
        anchors = [15 + int(i * 0.75 * 2) for i in range(16)]
        validate_synthetic_subsecond_geometry(
            timestamps, fps, (16, 32, 32), 0.75, anchors, anchors, 2,
            expected_frames=32,
        )

    def test_synthetic_40_frame_geometry_is_deterministic_and_complete(self):
        frames, timestamps, fps, digest = make_synthetic_subsecond_fixture(40)
        again = make_synthetic_subsecond_fixture(40)
        self.assertEqual((len(frames), len(timestamps)), (40, 40))
        self.assertEqual(timestamps, [i / fps for i in range(40)])
        self.assertEqual((digest, timestamps, fps), (again[3], again[1], again[2]))
        anchors = [15 + int(i * 0.75 * 2) for i in range(20)]
        validate_synthetic_subsecond_geometry(
            timestamps, fps, (20, 32, 32), 0.75, anchors, anchors, 2,
            expected_frames=40,
        )

    def test_synthetic_fixture_rejects_unsupported_frame_count(self):
        with self.assertRaisesRegex(ValueError, 'must be 4, 16, 32, or 40'):
            make_synthetic_subsecond_fixture(8)

    def test_synthetic_grid_requires_fixed_spatial_geometry(self):
        validate_synthetic_grid([20, 32, 32], 40)
        for grid in ([20, 31, 32], [20, 32, 33], [19, 32, 32], [True, 32, 32]):
            with self.subTest(grid=grid), self.assertRaisesRegex(RuntimeError, "grid must be"):
                validate_synthetic_grid(grid, 40)

    def test_legacy_reference_reconstructs_integer_pretruncate_without_touching_other_rows(self):
        native = (
            (0, 1, 5, 6, 7),
            (0, 1, 5, 5, 5),
            (0, 1, 5, 5, 5),
        )
        rows, anchors = legacy_pretruncate_position_ids(
            native,
            video_token_start=2,
            video_token_stop=4,
            video_grid_thw=(2, 2, 2),
            spatial_merge_size=2,
            seconds_per_grid=0.75,
            tokens_per_second=2,
        )
        self.assertEqual(anchors, [5, 5])
        self.assertEqual(rows[0], [0, 1, 5, 5, 7])
        self.assertEqual(rows[1:], [list(native[1]), list(native[2])])

    def test_legacy_reference_rejects_invalid_grid_and_token_span(self):
        native = ((0, 1, 2), (0, 1, 2), (0, 1, 2))
        with self.assertRaisesRegex(ValueError, "span does not match"):
            legacy_pretruncate_position_ids(
                native, video_token_start=1, video_token_stop=2,
                video_grid_thw=(2, 2, 2), spatial_merge_size=2,
                seconds_per_grid=0.75, tokens_per_second=2,
            )
        with self.assertRaisesRegex(ValueError, "invalid temporal/spatial"):
            legacy_pretruncate_position_ids(
                native, video_token_start=1, video_token_stop=3,
                video_grid_thw=(2, 3, 2), spatial_merge_size=2,
                seconds_per_grid=0.75, tokens_per_second=2,
            )

    def test_declared_runtime_fixtures_are_compatible(self):
        assert_fractional_float32_compatibility([15,16],.75,2)
        assert_fractional_float32_compatibility([15,414],199.9998016357422,2)

    def test_rounding_boundary_refuses_incompatible_python_math(self):
        with self.assertRaisesRegex(RuntimeError,'arithmetic differ'):
            assert_fractional_float32_compatibility([15,15,16,17],.49999999,2)


@unittest.skipUnless(MODEL_PATH.is_dir(), "pinned processor/config are not cached")
class SyntheticSubsecondProcessorTests(unittest.TestCase):
    def test_actual_processor_40_frame_geometry_without_weights(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from qwen_vl_utils import process_vision_info
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

        frames, timestamps, fps, _ = make_synthetic_subsecond_fixture(40)
        processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        config = AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames, "sample_fps": fps},
            {"type": "text", "text": "Describe the visible scene briefly."},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, return_video_metadata=True)
        tensor, metadata = videos[0]
        self.assertEqual(metadata["frames_indices"], list(range(40)))
        self.assertEqual([i / metadata["fps"] for i in metadata["frames_indices"]], timestamps)
        batch = processor(text=[text], images=images, videos=[tensor], video_metadata=[metadata],
            do_sample_frames=False, cap_pixels_per_frame=True, padding=True, return_tensors="pt")
        grid = batch["video_grid_thw"][0].tolist()
        seconds = float(batch["second_per_grid_ts"][0])
        self.assertEqual(grid, [20, 32, 32])
        self.assertAlmostEqual(seconds, 0.75, places=6)

        class ModelGeometryMethods:
            get_rope_index = Qwen2_5_VLModel.get_rope_index
            get_vision_position_ids = Qwen2_5_VLModel.get_vision_position_ids

        model = ModelGeometryMethods()
        model.config = SimpleNamespace(vision_config=config.vision_config)
        native, _ = model.get_rope_index(input_ids=batch["input_ids"],
            mm_token_type_ids=batch["mm_token_type_ids"], video_grid_thw=batch["video_grid_thw"],
            second_per_grid_ts=batch["second_per_grid_ts"], attention_mask=batch["attention_mask"])
        positions = (batch["input_ids"][0] == processor.video_token_id).nonzero(as_tuple=True)[0].tolist()
        merge = int(processor.video_processor.merge_size)
        tokens_per_step = (grid[1] // merge) * (grid[2] // merge)
        anchors = [int(native[0, 0, positions[0] + i * tokens_per_step]) for i in range(grid[0])]
        expected = [anchors[0] + int(i * seconds * int(config.vision_config.tokens_per_second))
                    for i in range(20)]
        self.assertEqual(anchors, expected)
        validate_synthetic_subsecond_geometry(timestamps, fps, grid, seconds,
            anchors, anchors, int(config.vision_config.tokens_per_second), expected_frames=40)

    def test_actual_processor_32_frame_geometry_without_weights(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from qwen_vl_utils import process_vision_info
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

        frames, timestamps, fps, _ = make_synthetic_subsecond_fixture(32)
        processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        config = AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames, "sample_fps": fps},
            {"type": "text", "text": "Describe the visible scene briefly."},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, return_video_metadata=True)
        tensor, metadata = videos[0]
        self.assertEqual(metadata["frames_indices"], list(range(32)))
        self.assertEqual([i / metadata["fps"] for i in metadata["frames_indices"]], timestamps)
        batch = processor(text=[text], images=images, videos=[tensor], video_metadata=[metadata],
            do_sample_frames=False, cap_pixels_per_frame=True, padding=True, return_tensors="pt")
        grid = batch["video_grid_thw"][0].tolist()
        seconds = float(batch["second_per_grid_ts"][0])
        self.assertEqual(grid, [16, 32, 32])
        self.assertAlmostEqual(seconds, 0.75, places=6)

        class ModelGeometryMethods:
            get_rope_index = Qwen2_5_VLModel.get_rope_index
            get_vision_position_ids = Qwen2_5_VLModel.get_vision_position_ids

        model = ModelGeometryMethods()
        model.config = SimpleNamespace(vision_config=config.vision_config)
        native, _ = model.get_rope_index(input_ids=batch["input_ids"],
            mm_token_type_ids=batch["mm_token_type_ids"], video_grid_thw=batch["video_grid_thw"],
            second_per_grid_ts=batch["second_per_grid_ts"], attention_mask=batch["attention_mask"])
        video_token_id = processor.video_token_id
        positions = (batch["input_ids"][0] == video_token_id).nonzero(as_tuple=True)[0].tolist()
        merge = int(processor.video_processor.merge_size)
        tokens_per_step = (grid[1] // merge) * (grid[2] // merge)
        anchors = [int(native[0, 0, positions[0] + i * tokens_per_step]) for i in range(grid[0])]
        expected = [anchors[0] + int(i * seconds * int(config.vision_config.tokens_per_second))
                    for i in range(16)]
        self.assertEqual(anchors, expected)
        validate_synthetic_subsecond_geometry(timestamps, fps, grid, seconds,
            anchors, anchors, int(config.vision_config.tokens_per_second), expected_frames=32)

    def test_actual_processor_16_frame_geometry_without_weights(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from qwen_vl_utils import process_vision_info
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

        frames, timestamps, fps, _ = make_synthetic_subsecond_fixture(16)
        processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        config = AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames, "sample_fps": fps},
            {"type": "text", "text": "Describe the visible scene briefly."},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, return_video_metadata=True)
        tensor, metadata = videos[0]
        self.assertEqual(metadata["frames_indices"], list(range(16)))
        self.assertEqual([i / metadata["fps"] for i in metadata["frames_indices"]], timestamps)
        batch = processor(text=[text], images=images, videos=[tensor], video_metadata=[metadata],
            do_sample_frames=False, cap_pixels_per_frame=True, padding=True, return_tensors="pt")
        grid = batch["video_grid_thw"][0].tolist()
        seconds = float(batch["second_per_grid_ts"][0])
        self.assertEqual(grid, [8, 32, 32])
        self.assertAlmostEqual(seconds, 0.75, places=6)

        class ModelGeometryMethods:
            get_rope_index = Qwen2_5_VLModel.get_rope_index
            get_vision_position_ids = Qwen2_5_VLModel.get_vision_position_ids

        model = ModelGeometryMethods()
        model.config = SimpleNamespace(vision_config=config.vision_config)
        native, _ = model.get_rope_index(input_ids=batch["input_ids"],
            mm_token_type_ids=batch["mm_token_type_ids"], video_grid_thw=batch["video_grid_thw"],
            second_per_grid_ts=batch["second_per_grid_ts"], attention_mask=batch["attention_mask"])
        video_token_id = processor.video_token_id
        positions = (batch["input_ids"][0] == video_token_id).nonzero(as_tuple=True)[0].tolist()
        merge = int(processor.video_processor.merge_size)
        tokens_per_step = (grid[1] // merge) * (grid[2] // merge)
        anchors = [int(native[0, 0, positions[0] + i * tokens_per_step]) for i in range(grid[0])]
        expected = [anchors[0] + int(i * seconds * int(config.vision_config.tokens_per_second))
                    for i in range(8)]
        self.assertEqual(anchors, expected)
        validate_synthetic_subsecond_geometry(timestamps, fps, grid, seconds,
            anchors, anchors, int(config.vision_config.tokens_per_second), expected_frames=16)

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
