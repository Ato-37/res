"""CPU-only contract tests for the bounded Qwen batch throughput smoke."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qwen_video_batch_throughput_gpu_smoke as runner  # noqa: E402


VIDEO_ID = 99
GRID = (2, 24, 46)
PATCHES = GRID[0] * GRID[1] * GRID[2]
VIDEO_TOKENS = GRID[0] * (GRID[1] // 2) * (GRID[2] // 2)


def processor_batch(batch_size, profile=runner.FRAME_PROFILE):
    seq = profile["input_tokens"]
    grid = profile["grid"]
    patches = grid[0] * grid[1] * grid[2]
    video_tokens = grid[0] * (grid[1] // 2) * (grid[2] // 2)
    ids = torch.ones((batch_size, seq), dtype=torch.long)
    ids[:, 10:10 + video_tokens] = VIDEO_ID
    types = torch.zeros_like(ids)
    types[:, 10:10 + video_tokens] = 2
    pixels = torch.arange(patches * 4, dtype=torch.float32).reshape(patches, 4)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "mm_token_type_ids": types,
        "pixel_values_videos": pixels.repeat(batch_size, 1),
        "video_grid_thw": torch.tensor([grid] * batch_size, dtype=torch.long),
        "second_per_grid_ts": torch.full((batch_size,), 0.5),
    }


class _Layer:
    def __init__(self, batch, length):
        self.keys = torch.zeros((batch, 2, length, 4))


class _Cache:
    def __init__(self, batch, length):
        self.layers = [_Layer(batch, length)] if length else []
        self.length = length

    def get_seq_length(self):
        return self.length


class _TraceModel(nn.Module):
    def forward(self, input_ids=None, position_ids=None, past_key_values=None):
        length = input_ids.shape[-1]
        batch = input_ids.shape[0]
        prior = 0 if past_key_values is None else past_key_values.get_seq_length()
        return SimpleNamespace(past_key_values=_Cache(batch, prior + length))


class BatchThroughputSmokeCpuTests(unittest.TestCase):
    def test_processor_batch_contract_and_duplicate_items(self):
        one, two = processor_batch(1), processor_batch(2)
        report = runner.validate_duplicate_processor_batches(one, two, VIDEO_ID)
        self.assertTrue(report["duplicate_items_match"])
        self.assertEqual(report["batch2"]["patch_rows_per_item"], [PATCHES, PATCHES])
        self.assertEqual(report["batch2"]["video_tokens_per_item"], [VIDEO_TOKENS] * 2)

    def test_processor_contract_rejects_patch_count_or_token_mapping_errors(self):
        batch = processor_batch(2)
        batch["pixel_values_videos"] = batch["pixel_values_videos"][:-1]
        with self.assertRaisesRegex(ValueError, "patch count"):
            runner.validate_processor_batch(batch, 2, VIDEO_ID)
        batch = processor_batch(2)
        batch["mm_token_type_ids"][1, 10] = 0
        with self.assertRaisesRegex(ValueError, "token/type/grid"):
            runner.validate_processor_batch(batch, 2, VIDEO_ID)

    def test_32_frame_profile_contract_is_supported_without_gpu(self):
        profile = runner.PROFILES[32]
        batch = processor_batch(2, profile)
        report = runner.validate_processor_batch(
            batch, 2, VIDEO_ID, profile["grid"], profile["input_tokens"]
        )
        self.assertEqual(report["grid_thw_per_item"], [list(profile["grid"])] * 2)
        self.assertEqual(report["input_tokens_per_item"], 4256)
        self.assertEqual(report["video_tokens_per_item"], [4224, 4224])

    def test_batched_cache_trace_checks_each_item_and_coordinate(self):
        model = _TraceModel()
        trace = runner.BatchGenerationTrace(model, 2)
        prefill = torch.arange(12).reshape(3, 2, 2)
        model(input_ids=torch.ones((2, 2), dtype=torch.long), position_ids=prefill,
              past_key_values=_Cache(2, 0))
        for step in range(runner.DECODE_STEPS):
            coordinate, length = 20 + step, 2 + step
            pos = torch.full((3, 2, 1), coordinate)
            model(input_ids=torch.ones((2, 1), dtype=torch.long), position_ids=pos,
                  past_key_values=_Cache(2, length))
        trace.close()
        report = runner.validate_batch_trace(trace, prefill, [20, 20], sequence_length=2)
        self.assertEqual(report["cache_lengths_after"], list(range(2, 2 + 1 + runner.DECODE_STEPS)))
        self.assertEqual(report["cache_batch_size_after"], [2] * (1 + runner.DECODE_STEPS))

    def test_128_step_trace_and_cache_growth(self):
        decode_steps, sequence_length = 128, 4256
        model = _TraceModel()
        trace = runner.BatchGenerationTrace(model, 2)
        prefill = torch.zeros((3, 2, sequence_length), dtype=torch.long)
        model(input_ids=torch.ones((2, sequence_length), dtype=torch.long),
              position_ids=prefill, past_key_values=_Cache(2, 0))
        for step in range(decode_steps):
            model(input_ids=torch.ones((2, 1), dtype=torch.long),
                  position_ids=torch.full((3, 2, 1), 100 + step),
                  past_key_values=_Cache(2, sequence_length + step))
        trace.close()
        report = runner.validate_batch_trace(
            trace, prefill, [100, 100], sequence_length, decode_steps
        )
        self.assertEqual(report["model_calls"], 129)
        self.assertEqual(report["cache_lengths_after"][-1], 4384)
        self.assertEqual(report["cache_batch_size_after"][-1], 2)

    def test_batched_trace_rejects_wrong_cache_batch_size(self):
        model = _TraceModel()
        trace = runner.BatchGenerationTrace(model, 2)
        prefill = torch.zeros((3, 2, 2), dtype=torch.long)
        model(input_ids=torch.ones((2, 2), dtype=torch.long), position_ids=prefill,
              past_key_values=_Cache(2, 0))
        for step in range(runner.DECODE_STEPS):
            model(input_ids=torch.ones((2, 1), dtype=torch.long),
                  position_ids=torch.full((3, 2, 1), 5 + step),
                  past_key_values=_Cache(1 if step == 0 else 2, 2 + step))
        trace.close()
        with self.assertRaisesRegex(RuntimeError, "cache batch correctly"):
            runner.validate_batch_trace(trace, prefill, [5, 5], sequence_length=2)

    def test_output_comparison_exports_only_scalar_summaries(self):
        one = SimpleNamespace(
            logits=tuple(torch.tensor([[1.0, 2.0]]) for _ in range(runner.MAX_NEW_TOKENS)),
            sequences=torch.tensor([[1] + [2] * runner.MAX_NEW_TOKENS]),
        )
        two = SimpleNamespace(
            logits=tuple(torch.tensor([[1.0, 2.0], [1.0, 2.0]])
                         for _ in range(runner.MAX_NEW_TOKENS)),
            sequences=torch.tensor([[1] + [2] * runner.MAX_NEW_TOKENS,
                                    [1] + [2] * runner.MAX_NEW_TOKENS]),
        )
        report = runner.compare_generation_outputs(one, two, 2)
        self.assertTrue(report["per_generated_step_logits"][0]["b1_vs_b2_item1"]["bitwise_equal"])
        compared = report["per_generated_step_logits"][0]["b1_vs_b2_item0"]
        self.assertEqual(compared["top1_margin_left"], 1.0)
        self.assertEqual(compared["top1_margin_right"], 1.0)
        self.assertTrue(compared["top2_set_equal"])
        self.assertIsNone(report["first_top1_divergence_step"])
        self.assertTrue(report["sequence_equality"]["b2_duplicate_sequences_equal"])
        self.assertTrue(runner.shared_context_top1_gate(report)["passed"])
        self.assertNotIn('"logits":', json.dumps(report))
        self.assertIsInstance(report["per_generated_step_logits"][0]["b1_vs_b2_item0"][
            "maximum_absolute_difference"], float)

    def test_shared_context_gate_fails_on_top1_or_prefix_mismatch(self):
        one = SimpleNamespace(
            logits=tuple(torch.tensor([[1.0, 2.0]]) for _ in range(runner.MAX_NEW_TOKENS)),
            sequences=torch.tensor([[1] + [2] * runner.MAX_NEW_TOKENS]),
        )
        two = SimpleNamespace(
            logits=tuple(torch.tensor([[1.0, 2.0], [2.0, 1.0]])
                         for _ in range(runner.MAX_NEW_TOKENS)),
            sequences=torch.tensor([[1] + [2] * runner.MAX_NEW_TOKENS,
                                    [1] + [3] * runner.MAX_NEW_TOKENS]),
        )
        report = runner.compare_generation_outputs(one, two, 2)
        gate = runner.shared_context_top1_gate(report)
        self.assertEqual(report["first_top1_divergence_step"], 1)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["forced_shared_sequences_equal"])
        self.assertFalse(gate["per_step_top1_equal"])

    def test_force_prefix_processor_forces_same_tokens_for_each_batch_row(self):
        processor = runner.make_force_prefix_logits_processor([3, 1], prompt_length=5)
        ids0 = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.zeros((2, 6))
        out0 = processor(ids0, scores)
        self.assertEqual(out0.argmax(-1).tolist(), [3, 3])
        self.assertTrue(torch.isfinite(out0).all())
        ids1 = torch.tensor([[1, 2, 3, 4, 5, 3]])
        out1 = processor(ids1, scores)
        self.assertEqual(out1.argmax(-1).tolist(), [1, 1])
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            processor(torch.ones((1, 7), dtype=torch.long), scores)

    def test_completed_gate_mismatch_is_a_completed_cli_run(self):
        self.assertEqual(runner.runner_exit_code({"status": "completed_same_context_argmax_mismatch"}), 0)
        self.assertEqual(runner.runner_exit_code({"status": "completed_same_context_top1_match"}), 0)
        self.assertEqual(runner.runner_exit_code({"status": "failed"}), 1)

    def test_swapped_top1_can_be_reported_without_exporting_token_ids(self):
        one = SimpleNamespace(
            logits=(torch.tensor([[10.0, 9.0, 0.0]]), torch.tensor([[10.0, 9.0, 0.0]])),
            sequences=torch.tensor([[7, 8, 8]]),
        )
        two = SimpleNamespace(
            logits=(torch.tensor([[9.0, 10.0, 0.0], [9.0, 10.0, 0.0]]),
                    torch.tensor([[9.0, 10.0, 0.0], [9.0, 10.0, 0.0]])),
            sequences=torch.tensor([[7, 9, 9], [7, 9, 9]]),
        )
        report = runner.compare_generation_outputs(one, two, 2)
        comparison = report["per_generated_step_logits"][0]["b1_vs_b2_item0"]
        self.assertFalse(comparison["top1_equal"])
        self.assertTrue(comparison["top2_set_equal"])
        self.assertEqual(comparison["top1_margin_left"], 1.0)
        self.assertEqual(comparison["top1_margin_right"], 1.0)
        self.assertNotIn("indices", str(report))

    def test_cpu_comparison_projection_drops_scores_and_cache_fields(self):
        output = SimpleNamespace(
            logits=(torch.ones((1, 3)),),
            sequences=torch.ones((1, 5), dtype=torch.long),
            scores=(torch.ones((1, 3)),),
            past_key_values=_Cache(1, 5),
        )
        projected = runner._copy_logits_to_cpu(output)
        self.assertEqual(set(vars(projected)), {"logits", "sequences"})
        self.assertEqual(projected.logits[0].device.type, "cpu")

    def test_output_path_is_exclusive_results_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            path = runner.resolve_output_path(root, "results/new.json")
            self.assertEqual(path.parent, root / "results")
            path.touch()
            with self.assertRaises(ValueError):
                runner.resolve_output_path(root, "results/new.json")
            with self.assertRaises(ValueError):
                runner.resolve_output_path(root, "results/sub/new.json")

    def test_failure_codes_are_sanitized(self):
        error = RuntimeError("CUDA out of memory at /private/video.mp4")
        self.assertEqual(runner._failure_code(error, "generate_batch_2"), "cuda_oom")
        self.assertEqual(runner._failure_code(RuntimeError("secret"), "prepare_batch_2"),
                         "processor_batch2_failed")


if __name__ == "__main__":
    unittest.main()
