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
    GpuTelemetrySampler,
    pack_generation_positions,
    normalize_rope_positions,
    resolve_output_path,
    summarize_gpu_telemetry,
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
    def test_gpu_telemetry_summary_contains_only_sample_statistics(self):
        samples = [
            {"gpu_util_percent": 40, "memory_used_mib": 9000,
             "memory_free_mib": 7000, "power_watts": 100.0},
            {"gpu_util_percent": 80, "memory_used_mib": 11000,
             "memory_free_mib": 5000, "power_watts": 200.0},
        ]
        summary = summarize_gpu_telemetry(samples, errors=1, elapsed_seconds=1.5)
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["sample_errors"], 1)
        self.assertEqual(summary["gpu_util_percent"],
                         {"mean": 60.0, "median": 60.0, "max": 80})
        self.assertEqual(summary["power_watts"]["mean"], 150.0)
        self.assertNotIn("process_name", summary)
        self.assertIsNone(summarize_gpu_telemetry([], 2, 0.2)["gpu_util_percent"]["mean"])

    def test_gpu_telemetry_sampler_parses_only_four_public_counters(self):
        sampler = GpuTelemetrySampler()
        sampler._started = 1.0
        with mock.patch.object(runner.subprocess, "check_output",
                               return_value="55, 10000, 6000, 175.5\n"):
            sampler._sample()
        self.assertEqual(len(sampler.samples), 1)
        sample = sampler.samples[0]
        self.assertEqual(sample["gpu_util_percent"], 55)
        self.assertEqual(sample["memory_used_mib"], 10000)
        self.assertEqual(sample["memory_free_mib"], 6000)
        self.assertEqual(sample["power_watts"], 175.5)
        self.assertNotIn("pid", sample)

    @unittest.skipUnless(MODEL_PATH.is_dir(), "pinned cached config is unavailable")
    def test_generation_padding_uses_generation_config_not_multimodal_config(self):
        from transformers import AutoConfig, GenerationConfig
        config = AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
        generation = GenerationConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
        self.assertFalse(hasattr(config, "pad_token_id"))
        self.assertEqual(generation.pad_token_id, 151643)

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

    def test_32_frame_fractional_anchor_sequence_is_checked_across_all_temporal_steps(self):
        seconds = 19.3526668548584
        tokens_per_second = 2
        interval = torch.tensor(seconds, dtype=torch.float32) * tokens_per_second
        anchors = (
            torch.arange(16, dtype=torch.float32) * interval
        ).long().add_(15).tolist()
        self.assertEqual(
            anchors,
            [15, 53, 92, 131, 169, 208, 247, 285,
             324, 363, 402, 440, 479, 518, 556, 595],
        )
        assert_fractional_float32_compatibility(anchors, seconds, tokens_per_second)
        anchors[-1] += 1
        with self.assertRaisesRegex(RuntimeError, "float32"):
            assert_fractional_float32_compatibility(anchors, seconds, tokens_per_second)

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
        with self.assertRaisesRegex(RuntimeError, "every decode step"):
            validate_generation_trace(trace, prefill_pids, 5, 55)

    def test_sixteen_cached_decode_steps_advance_every_position_and_cache_length(self):
        lm = _TinyLanguageModel()
        trace = GenerationCacheTrace(lm)
        prefill_pids = torch.arange(15).reshape(3, 1, 5)
        output = lm(input_ids=torch.ones((1, 5)), position_ids=prefill_pids,
                    past_key_values=_FakeCache(0))
        for step in range(16):
            lm(input_ids=torch.ones((1, 1)),
               position_ids=torch.full((3, 1, 1), 55 + step),
               past_key_values=output.past_key_values)
            output = SimpleNamespace(past_key_values=_FakeCache(6 + step))
        trace.close()
        report = validate_generation_trace(
            trace, prefill_pids, 5, 55, cached_decode_steps=16
        )
        self.assertEqual(report["model_calls"], 17)
        self.assertEqual(report["decode_start_coordinate"], 55)
        self.assertEqual(report["decode_end_coordinate"], 70)
        self.assertEqual(report["past_cache_lengths"], [0, *range(5, 21)])
        self.assertEqual(report["cache_lengths_after"], [5, *range(6, 22)])
        trace.calls[8]["position_ids"] = torch.full((3, 1, 1), 999)
        with self.assertRaisesRegex(RuntimeError, "every decode step"):
            validate_generation_trace(
                trace, prefill_pids, 5, 55, cached_decode_steps=16
            )
        trace.calls[8]["position_ids"] = torch.full((3, 1, 1), 62)
        trace.calls[8]["cache_length_after"] += 1
        with self.assertRaisesRegex(RuntimeError, "expected KV lengths"):
            validate_generation_trace(
                trace, prefill_pids, 5, 55, cached_decode_steps=16
            )

    def test_128_cached_decode_steps_check_interior_and_final_cache_positions(self):
        lm = _TinyLanguageModel()
        trace = GenerationCacheTrace(lm)
        prefill_pids = torch.arange(15).reshape(3, 1, 5)
        output = lm(input_ids=torch.ones((1, 5)), position_ids=prefill_pids,
                    past_key_values=_FakeCache(0))
        for step in range(128):
            lm(input_ids=torch.ones((1, 1)),
               position_ids=torch.full((3, 1, 1), 55 + step),
               past_key_values=output.past_key_values)
            output = SimpleNamespace(past_key_values=_FakeCache(6 + step))
        trace.close()
        report = validate_generation_trace(
            trace, prefill_pids, 5, 55, cached_decode_steps=128
        )
        self.assertEqual(report["model_calls"], 129)
        self.assertEqual(report["decode_end_coordinate"], 182)
        self.assertEqual(report["past_cache_lengths"], [0, *range(5, 133)])
        self.assertEqual(report["cache_lengths_after"], [5, *range(6, 134)])
        for index in (64, 128):
            trace.calls[index]["position_ids"] = torch.full((3, 1, 1), 999)
            with self.assertRaisesRegex(RuntimeError, "every decode step"):
                validate_generation_trace(
                    trace, prefill_pids, 5, 55, cached_decode_steps=128
                )
            trace.calls[index]["position_ids"] = torch.full(
                (3, 1, 1), 54 + index, dtype=torch.long
            )

    def test_generation_output_check_requires_finite_scores_and_raw_logits(self):
        output = SimpleNamespace(
            scores=(torch.zeros((1, 7)), torch.ones((1, 7))),
            logits=(torch.zeros((1, 7)), torch.ones((1, 7))),
        )
        self.assertTrue(_generation_outputs_finite(output))
        output.logits = (torch.zeros((1, 7)), torch.full((1, 7), float("nan")))
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            _generation_outputs_finite(output)
        with self.assertRaisesRegex(RuntimeError, "requested score and raw-logit"):
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

        def fail_after_partial_progress(_output, _frame_count, _cached_decode_steps):
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
    def test_cached_processor_preflight_matches_both_fixed_frame_profiles_without_weights(self):
        """Exercise the cached processor, not the 7B checkpoint or CUDA path."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        processor = None
        for frame_count, fps, size, max_pixels, expected_grid, expected_tokens in (
            (4, 0.010000010000009999, (224, 126), 256 * 28 * 28, (2, 24, 46), 584),
            (32, 0.10334493093113782, (640, 360), 200704, (16, 24, 44), 4256),
        ):
            with self.subTest(frame_count=frame_count):
                processor = AutoProcessor.from_pretrained(
                    str(MODEL_PATH), local_files_only=True,
                    min_pixels=256 * 28 * 28, max_pixels=max_pixels,
                )
                frames = [Image.new("RGB", size, color=(18 + i, 47 + i, 83 + i))
                          for i in range(frame_count)]
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
                self.assertEqual(grid, expected_grid)
                self.assertEqual(int(batch["input_ids"].shape[1]), expected_tokens)
                if frame_count == 4:
                    self.assertAlmostEqual(float(batch["second_per_grid_ts"][0]), 199.9998, places=3)
                    assert_fractional_float32_compatibility(
                        [15, 414], float(batch["second_per_grid_ts"][0]), 2
                    )
                self.assertEqual(int((batch["mm_token_type_ids"] == 1).sum()), 0)
                expected_video_tokens = expected_grid[0] * (expected_grid[1] // 2) * (expected_grid[2] // 2)
                self.assertEqual(int((batch["mm_token_type_ids"] == 2).sum()), expected_video_tokens)
                self.assertTrue(torch.equal(
                    batch["mm_token_type_ids"][0] == 2,
                    batch["input_ids"][0] == processor.video_token_id,
                ))
                self.assertTrue(bool(batch["attention_mask"].bool().all()))


if __name__ == "__main__":
    unittest.main()
