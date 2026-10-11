"""CPU preflight tests for the GPU video-cache smoke (no model weights/GPU)."""

import os
import json
from pathlib import Path
import sys
import uuid
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_qwen_video_cache_gpu_smoke as runner  # noqa: E402
from run_qwen_video_cache_gpu_smoke import (  # noqa: E402
    GenerationCacheTrace,
    _generation_outputs_finite,
    _write_result,
    kv_cache_payload_estimate,
    pack_generation_positions,
    normalize_rope_positions,
    resolve_output_path,
    validate_generation_trace,
)
from run_qwen_temporal_position_gpu_check import assert_fractional_float32_compatibility  # noqa: E402


REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_PATH = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"
    / REVISION
)


class _FakeCache:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length

    def get_seq_length(self):
        return self.sequence_length


class _TinyLanguageModel(nn.Module):
    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, past_key_values=None):
        input_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        current = 0 if past_key_values is None else past_key_values.get_seq_length()
        return SimpleNamespace(past_key_values=_FakeCache(current + input_length))


class VideoCacheGpuSmokeCpuTests(unittest.TestCase):
    def test_position_packing_and_normalization_are_shape_strict(self):
        rope = torch.arange(3 * 1 * 5).reshape(3, 1, 5)
        packed = pack_generation_positions(rope, 5)
        self.assertEqual(tuple(packed.shape), (4, 1, 5))
        self.assertTrue(torch.equal(normalize_rope_positions(packed), rope))
        self.assertTrue(torch.equal(normalize_rope_positions(rope), rope))
        with self.assertRaises(ValueError):
            pack_generation_positions(rope[:, :, :-1], 5)
        with self.assertRaises(ValueError):
            normalize_rope_positions(torch.zeros((2, 1, 5), dtype=torch.long))

    def test_fake_text_model_hooks_verify_prefill_and_cached_decode_lengths(self):
        lm = _TinyLanguageModel()
        trace = GenerationCacheTrace(lm)
        prefill_pids = torch.arange(15).reshape(3, 1, 5)
        decode_pids = torch.full((3, 1, 1), 55, dtype=torch.long)
        prefill = lm(
            input_ids=torch.ones((1, 5), dtype=torch.long),
            position_ids=prefill_pids,
            past_key_values=_FakeCache(0),
        )
        lm(
            input_ids=torch.ones((1, 1), dtype=torch.long),
            position_ids=decode_pids,
            past_key_values=prefill.past_key_values,
        )
        trace.close()
        report = validate_generation_trace(trace, prefill_pids, 5, 55)
        self.assertEqual(report["model_calls"], 2)
        self.assertEqual(report["input_lengths"], [5, 1])
        self.assertEqual(report["past_cache_lengths"], [0, 5])
        self.assertEqual(report["cache_lengths_after"], [5, 6])

    def test_trace_rejects_wrong_decode_coordinate(self):
        lm = _TinyLanguageModel()
        trace = GenerationCacheTrace(lm)
        prefill_pids = torch.zeros((3, 1, 5), dtype=torch.long)
        prefill = lm(input_ids=torch.ones((1, 5)), position_ids=prefill_pids,
                     past_key_values=_FakeCache(0))
        lm(input_ids=torch.ones((1, 1)), position_ids=torch.full((3, 1, 1), 54),
           past_key_values=prefill.past_key_values)
        trace.close()
        with self.assertRaisesRegex(RuntimeError, "expected coordinate"):
            validate_generation_trace(trace, prefill_pids, 5, 55)

    def test_generation_output_check_requires_finite_scores_and_raw_logits(self):
        output = SimpleNamespace(
            scores=(torch.zeros((1, 7)), torch.ones((1, 7))),
            logits=(torch.zeros((1, 7)), torch.ones((1, 7))),
        )
        self.assertTrue(_generation_outputs_finite(output))
        output.logits = (torch.zeros((1, 7)), torch.full((1, 7), float("nan")))
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            _generation_outputs_finite(output)
        with self.assertRaisesRegex(RuntimeError, "two score and raw-logit"):
            _generation_outputs_finite(SimpleNamespace(scores=(), logits=None))

    def test_output_guard_refuses_outside_results_and_existing_path(self):
        repo = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            resolve_output_path(repo, "../outside.json")
        with self.assertRaises(ValueError):
            resolve_output_path(repo, "results/nested/new.json")
        existing = repo / "results" / "noaa_temporal_position_gpu_check_20261012_01.json"
        if existing.exists():
            with self.assertRaises(ValueError):
                resolve_output_path(repo, str(existing.relative_to(repo)))

    def test_summary_file_is_exclusive_and_private_and_kv_estimate_is_bounded(self):
        repo = Path(__file__).resolve().parents[1]
        output = repo / "results" / f".tmp_cache_smoke_{uuid.uuid4().hex}.json"
        try:
            _write_result(output, {"ok": True})
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                _write_result(output, {"overwrite": True})
        finally:
            output.unlink(missing_ok=True)

        config = SimpleNamespace(
            num_hidden_layers=28, num_key_value_heads=4,
            num_attention_heads=28, hidden_size=3584, head_dim=128,
        )
        estimate = kv_cache_payload_estimate(config, 584)
        self.assertEqual(estimate["estimated_prefill_kv_payload_bytes"], 33_488_896)
        self.assertAlmostEqual(estimate["estimated_prefill_kv_payload_mib"], 31.94, places=2)

    def test_failure_summary_preserves_partial_effective_counts_without_exception_text(self):
        repo = Path(__file__).resolve().parents[1]
        output = repo / "results" / f".tmp_cache_failure_{uuid.uuid4().hex}.json"
        original = dict(runner.RUN_PROGRESS)

        def fail_after_partial_progress(_output):
            runner.RUN_PROGRESS.update(
                stage="generate_explicit_native_noop", completed_model_calls=2,
                effective_conditions=1, effective_frames=4,
            )
            raise RuntimeError("must not be serialized")

        try:
            with mock.patch.object(runner, "run_gpu_smoke", side_effect=fail_after_partial_progress):
                with self.assertRaises(SystemExit) as caught:
                    runner.main(["--output", str(output.relative_to(repo))])
            self.assertEqual(caught.exception.code, 1)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["effective_conditions"], 1)
            self.assertEqual(record["completed_model_calls"], 2)
            self.assertEqual(record["effective_frames"], 4)
            self.assertEqual(record["failures"], 1)
            self.assertNotIn("must not be serialized", output.read_text(encoding="utf-8"))
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        finally:
            output.unlink(missing_ok=True)
            runner.RUN_PROGRESS.clear()
            runner.RUN_PROGRESS.update(original)

    @unittest.skipUnless(MODEL_PATH.is_dir(), "pinned cached processor is unavailable")
    def test_cached_processor_preflight_matches_fixed_four_frame_geometry_without_weights(self):
        """Exercise the cached processor, not the 7B checkpoint or CUDA path."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        fps = 0.010000010000009999
        colors = ((18, 47, 83), (71, 103, 137), (129, 163, 197), (211, 223, 239))
        # The existing extractor thumbnails the 640x360 source to 224x126.
        frames = [Image.new("RGB", (224, 126), color) for color in colors]
        processor = AutoProcessor.from_pretrained(
            str(MODEL_PATH), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
        )
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames, "sample_fps": fps},
            {"type": "text", "text": "Describe the visible scene briefly. Use only visual information."},
        ]}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages, return_video_metadata=True)
        self.assertEqual(len(videos), 1)
        video_tensor, metadata = videos[0]
        batch = processor(
            text=[prompt_text], images=images, videos=[video_tensor],
            video_metadata=[metadata], do_sample_frames=False,
            cap_pixels_per_frame=True, padding=True, return_tensors="pt",
        )
        grid = tuple(int(v) for v in batch["video_grid_thw"][0].tolist())
        self.assertEqual(grid, (2, 24, 46))
        self.assertEqual(int(batch["input_ids"].shape[1]), 584)
        self.assertAlmostEqual(float(batch["second_per_grid_ts"][0]), 199.9998, places=3)
        assert_fractional_float32_compatibility(
            [15, 414], float(batch["second_per_grid_ts"][0]), 2
        )
        self.assertEqual(int((batch["mm_token_type_ids"] == 1).sum()), 0)
        self.assertEqual(int((batch["mm_token_type_ids"] == 2).sum()), 552)
        self.assertTrue(torch.equal(
            batch["mm_token_type_ids"][0] == 2,
            batch["input_ids"][0] == processor.video_token_id,
        ))
        self.assertTrue(bool(batch["attention_mask"].bool().all()))


if __name__ == "__main__":
    unittest.main()
