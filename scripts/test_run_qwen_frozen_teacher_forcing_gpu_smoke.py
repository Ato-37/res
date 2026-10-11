"""CPU-only checks for the frozen-target teacher-forcing GPU smoke."""

import json
from pathlib import Path
import sys
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_qwen_frozen_teacher_forcing_gpu_smoke as runner  # noqa: E402
from frozen_teacher_forced_score import freeze_generation_tail  # noqa: E402
from run_qwen_frozen_teacher_forcing_gpu_smoke import (  # noqa: E402
    NativeGenerationTrace,
    compare_generation_to_teacher_forcing,
    prediction_rows,
    resolve_output_path,
    validate_teacher_forcing_trace,
    validate_native_trace,
    _write_result,
    _write_gate_result,
)


class _Cache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class _TinyLM(nn.Module):
    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None,
                past_key_values=None, use_cache=True):
        length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        previous = 0 if past_key_values is None else past_key_values.get_seq_length()
        cache = _Cache(previous + length) if use_cache else None
        return SimpleNamespace(past_key_values=cache)


class FrozenTeacherForcingGpuSmokeTests(unittest.TestCase):
    def setUp(self):
        self.prompt_length = 3
        self.target_ids = (1, 2)
        self.target = freeze_generation_tail(
            [10, 11, 12, *self.target_ids], prompt_length=self.prompt_length
        )
        # Both generated-token distributions have distinct maxima. The full
        # teacher-forced positions at rows P-1 and P exactly reproduce them.
        self.generated_logits = torch.tensor(
            [[[0.0, 5.0, 0.0, -1.0], [0.0, 0.0, 4.0, -1.0]]]
        )
        self.full_logits = torch.zeros((1, 5, 4))
        self.full_logits[:, 2:4, :] = self.generated_logits
        self.full_logits[:, 4, :] = torch.tensor([[-2.0, 1.0, 0.0, 3.0]])

    def test_teacher_forcing_rows_match_generation_and_wrong_offset_is_discriminating(self):
        selected = prediction_rows(self.full_logits, self.prompt_length, 2)
        self.assertTrue(torch.equal(selected, self.full_logits[:, 2:4, :]))
        result = compare_generation_to_teacher_forcing(
            self.generated_logits, self.full_logits, self.target, self.prompt_length
        )
        self.assertEqual(result["prediction_logit_rows"], [2, 3])
        self.assertEqual(result["wrong_offset_prediction_rows"], [3, 4])
        self.assertTrue(result["raw_logits_allclose_atol_0_1_rtol_0_01"])
        self.assertEqual(result["maximum_absolute_raw_logit_difference"], 0.0)
        self.assertEqual(result["frozen_target_score_absolute_difference_nats"], 0.0)
        self.assertTrue(result["wrong_offset_control_discriminates"])
        self.assertGreater(result["generation_vs_wrong_offset_maximum_absolute_raw_logit_difference"], 1e-6)
        self.assertGreater(
            result["teacher_correct_vs_wrong_offset_maximum_absolute_per_token_logprob_difference_nats"],
            1e-6,
        )
        self.assertEqual(result["target_sha256"], self.target.target_sha256)
        self.assertNotIn("token_ids", result)
        self.assertNotIn("target_score", result)

    def test_wrong_offset_score_control_compares_teacher_rows_independent_of_generation(self):
        generation_logits = self.full_logits[:, 3:5, :].clone()
        result = compare_generation_to_teacher_forcing(
            generation_logits, self.full_logits, self.target, self.prompt_length
        )
        self.assertEqual(
            result["generation_vs_wrong_offset_maximum_absolute_raw_logit_difference"], 0.0
        )
        self.assertTrue(result["wrong_offset_control_discriminates"])
        self.assertGreater(
            result["teacher_correct_vs_wrong_offset_maximum_absolute_per_token_logprob_difference_nats"],
            1e-6,
        )

    def test_fp64_supplied_differences_survive_large_common_logit_offset(self):
        full = torch.full((1, 5, 4), 1e16, dtype=torch.float64)
        generated = full[:, 2:4, :].clone()
        generated[0, 0, 1] += 2.0
        result = compare_generation_to_teacher_forcing(
            generated, full, self.target, self.prompt_length
        )
        self.assertEqual(result["maximum_absolute_raw_logit_difference"], 2.0)
        self.assertGreater(result["maximum_absolute_per_token_logprob_difference_nats"], 0.02)
        self.assertFalse(result["max_per_token_logprob_within_0_02_nats"])

    def test_fixed_point_02_gate_uses_max_per_token_difference_not_sum(self):
        balanced_teacher = torch.zeros((1, 5, 4))
        balanced_teacher[0, 2, 1] = 0.5
        balanced_teacher[0, 3, 2] = 0.5
        balanced_teacher[0, 4, 1] = 0.5
        near_generation = balanced_teacher[:, 2:4, :].clone()
        near_generation[0, 0, 1] -= 0.04
        near_generation[0, 1, 2] += 0.04
        result = compare_generation_to_teacher_forcing(
            near_generation, balanced_teacher, self.target, self.prompt_length
        )
        self.assertTrue(result["raw_logits_allclose_atol_0_1_rtol_0_01"])
        self.assertGreater(result["maximum_absolute_per_token_logprob_difference_nats"], 0.02)
        self.assertFalse(result["max_per_token_logprob_within_0_02_nats"])
        self.assertLess(result["frozen_target_score_absolute_difference_nats"], 0.02)

    def test_alignment_rejects_wrong_shapes_lengths_and_nonfinite_logits(self):
        with self.assertRaisesRegex(ValueError, "exact prompt-plus-target"):
            prediction_rows(self.full_logits[:, :-1], self.prompt_length, 2)
        with self.assertRaisesRegex(ValueError, "target length"):
            compare_generation_to_teacher_forcing(
                self.generated_logits[:, :1], self.full_logits, self.target, self.prompt_length
            )
        nonfinite = self.full_logits.clone()
        nonfinite[0, 2, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            compare_generation_to_teacher_forcing(
                self.generated_logits, nonfinite, self.target, self.prompt_length
            )

    def test_native_trace_checks_prefill_positions_cache_and_decode_coordinate(self):
        lm = _TinyLM()
        trace = NativeGenerationTrace(lm)
        prompt_positions = torch.arange(15, dtype=torch.long).reshape(3, 1, 5)
        prefill = lm(
            input_ids=torch.ones((1, 5), dtype=torch.long),
            position_ids=prompt_positions,
            past_key_values=_Cache(0),
        )
        lm(
            input_ids=torch.ones((1, 1), dtype=torch.long),
            position_ids=torch.full((3, 1, 1), 55, dtype=torch.long),
            past_key_values=prefill.past_key_values,
        )
        trace.close()
        summary = validate_native_trace(trace, prompt_positions, 5)
        self.assertEqual(summary["input_lengths"], [5, 1])
        self.assertEqual(summary["past_cache_lengths"], [0, 5])
        self.assertEqual(summary["cache_lengths_after"], [5, 6])
        self.assertEqual(summary["first_generated_token_position"], 55)

    def test_trace_rejects_decode_position_mismatch(self):
        lm = _TinyLM()
        trace = NativeGenerationTrace(lm)
        pids = torch.zeros((3, 1, 4), dtype=torch.long)
        prefill = lm(input_ids=torch.ones((1, 4)), position_ids=pids, past_key_values=_Cache(0))
        lm(input_ids=torch.ones((1, 1)), position_ids=torch.full((3, 1, 1), 54),
           past_key_values=prefill.past_key_values)
        trace.close()
        with self.assertRaisesRegex(RuntimeError, "coordinate 55"):
            validate_native_trace(trace, pids, 4)

    def test_teacher_forward_trace_checks_actual_three_row_positions(self):
        lm = _TinyLM()
        trace = NativeGenerationTrace(lm)
        positions = torch.tensor([[[0, 1, 2, 55, 56]],
                                  [[0, 1, 2, 55, 56]],
                                  [[0, 1, 2, 55, 56]]], dtype=torch.long)
        lm(input_ids=torch.ones((1, 5), dtype=torch.long), position_ids=positions,
           past_key_values=_Cache(0), use_cache=False)
        trace.close()
        summary = validate_teacher_forcing_trace(trace, positions, 5)
        self.assertEqual(summary["model_calls"], 1)
        self.assertTrue(summary["positions_match_at_text_model"])

    def test_output_guard_and_exclusive_private_result(self):
        repo = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            resolve_output_path(repo, "results/subdir/new.json")
        with self.assertRaises(ValueError):
            resolve_output_path(repo, "../outside.json")
        out = repo / "results" / f".tmp_frozen_teacher_{uuid.uuid4().hex}.json"
        try:
            _write_result(out, {"status": "ok"})
            self.assertEqual(out.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                _write_result(out, {"overwrite": True})
        finally:
            out.unlink(missing_ok=True)

    def test_gate_failure_writes_complete_comparison_evidence(self):
        repo = Path(__file__).resolve().parents[1]
        out = repo / "results" / f".tmp_frozen_gate_{uuid.uuid4().hex}.json"
        record = {
            "status": "pending",
            "source_sha256": {"runner": "a" * 64},
            "checkpoint_config_sha256": "b" * 64,
            "media_sha256": "c" * 64,
            "frozen_target_sha256": "d" * 64,
            "completed_model_calls": 3,
            "comparison": {
                "maximum_absolute_raw_logit_difference": 0.4,
                "maximum_absolute_per_token_logprob_difference_nats": 0.03,
            },
        }
        try:
            written = _write_gate_result(out, record, ["max_per_token_logprob_difference_outside_0_02_nats"])
            persisted = json.loads(out.read_text(encoding="utf-8"))
            self.assertIs(written, record)
            self.assertEqual(persisted["status"], "gate_failed")
            self.assertEqual(persisted["failures"], 1)
            self.assertEqual(persisted["completed_model_calls"], 3)
            self.assertEqual(persisted["comparison"], record["comparison"])
            self.assertEqual(persisted["source_sha256"], record["source_sha256"])
            self.assertEqual(persisted["frozen_target_sha256"], "d" * 64)
        finally:
            out.unlink(missing_ok=True)

    def test_failure_artifact_records_partial_calls_without_exception_text(self):
        repo = Path(__file__).resolve().parents[1]
        out = repo / "results" / f".tmp_frozen_failure_{uuid.uuid4().hex}.json"
        old_progress = dict(runner.RUN_PROGRESS)

        def fail_after_one_call(_path):
            runner.RUN_PROGRESS.update(
                stage="native_full_teacher_forcing",
                completed_model_calls=2,
                effective_frames=4,
                generation_tokens_frozen=2,
                partial_evidence={
                    "source_sha256": {"runner": "e" * 64},
                    "media_sha256": "f" * 64,
                    "frozen_target_sha256": "a" * 64,
                    "comparison": {"maximum_absolute_raw_logit_difference": 0.5},
                },
            )
            raise RuntimeError("private terminal detail must not leak")

        try:
            with mock.patch.object(runner, "run_gpu_smoke", side_effect=fail_after_one_call):
                with self.assertRaises(SystemExit) as caught:
                    runner.main(["--output", str(out.relative_to(repo))])
            self.assertEqual(caught.exception.code, 1)
            body = out.read_text(encoding="utf-8")
            failure = json.loads(body)
            self.assertEqual(failure["completed_model_calls"], 2)
            self.assertEqual(failure["effective_frames"], 4)
            self.assertEqual(failure["generation_tokens_frozen"], 2)
            self.assertEqual(failure["failures"], 1)
            self.assertEqual(failure["evidence"]["source_sha256"]["runner"], "e" * 64)
            self.assertEqual(failure["evidence"]["comparison"]["maximum_absolute_raw_logit_difference"], 0.5)
            self.assertNotIn("private terminal detail", body)
            self.assertEqual(out.stat().st_mode & 0o777, 0o600)
        finally:
            out.unlink(missing_ok=True)
            runner.RUN_PROGRESS.clear()
            runner.RUN_PROGRESS.update(old_progress)


if __name__ == "__main__":
    unittest.main()
