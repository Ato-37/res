from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

import run_qta_synthetic_event_readability_v2 as runner
from qta_synthetic_event_video_fixture import build_synthetic_event_pair
from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        if text.endswith("Answer:"):
            return [1, 2]
        if len(text) > 1 and text[-2:-1] == ":" and text[-1] in "ABCDE":
            return [1, 2, ord(text[-1]) - 60]
        raise ValueError("unexpected fake tokenizer input")

    class _Backend:
        @staticmethod
        def to_str():
            return '{"type":"fake-tokenizer"}'

    backend_tokenizer = _Backend()


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.image_processor = SimpleNamespace(
            to_dict=lambda: {"fake": True}, merge_size=1)
        self.video_token_id = 99
        self.calls = 0
        self.templates = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        self.templates.append(messages)
        text = messages[0]["content"][1]["text"]
        return text + "Answer:"

    def __call__(self, *, text, images, videos, video_metadata,
                 do_sample_frames, cap_pixels_per_frame, padding, return_tensors):
        self.calls += 1
        self.assertions = (len(text) == 1 and videos == ["video-payload"]
                           and video_metadata == ["video-meta"]
                           and do_sample_frames is False and padding is True
                           and return_tensors == "pt")
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.int64),
            "pixel_values_videos": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
            "video_grid_thw": torch.tensor([[16, 1, 1]], dtype=torch.int64),
        }


class SyntheticEventReadabilityRunnerTests(unittest.TestCase):
    def _private_root(self, root: Path) -> Path:
        private = root / "data" / "private_nextgqa"
        private.mkdir(parents=True, mode=0o700)
        private.chmod(0o700)
        return private

    def test_output_path_is_private_direct_child_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = self._private_root(root)
            output = runner.validate_output_path(
                Path("data/private_nextgqa/event.json"), root)
            self.assertEqual(output, private / "event.json")
            with self.assertRaises(ValueError):
                runner.validate_output_path(root / "results" / "event.json", root)
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                runner.validate_output_path(output, root)

    def test_exclusive_report_is_0600_aggregate_only_and_refuses_sensitive_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            private = self._private_root(Path(temporary))
            output = private / "summary.json"
            report = {
                "status": "passed",
                "faithfulness_result": False,
                "top_choice_correct_count": 2,
                "target_vs_best_foil_margin_nats": {
                    "n": 2, "mean": 0.25, "min": 0.1, "max": 0.4,
                },
            }
            runner.write_report_exclusive(output, report)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(output.read_text()), report)
            with self.assertRaises(FileExistsError):
                runner.write_report_exclusive(output, report)
            with self.assertRaises(ValueError):
                runner._assert_aggregate_only({"question": "private prompt"})

    def test_cpu_processor_preflight_builds_exact_pair_and_binds_A_to_E(self):
        pair = build_synthetic_event_pair()
        processor = _Processor()
        calls = []

        def vision_info(messages, return_video_metadata):
            calls.append((messages, return_video_metadata))
            self.assertEqual(return_video_metadata, True)
            video = messages[0]["content"][0]["video"]
            self.assertEqual(len(video), 32)
            self.assertTrue(all(frame.mode == "RGB" for frame in video))
            return None, [("video-payload", "video-meta")]

        prepared = runner.prepare_pair_inputs(
            pair=pair,
            processor=processor,
            torch_module=torch,
            vision_info_fn=vision_info,
            build_prompt_fn=build_qa_prompt,
            process_video_fn=process_qa_video,
        )
        self.assertEqual(len(prepared), 2)
        self.assertEqual(processor.calls, 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(processor.assertions)
        self.assertEqual([row["target_index"] for row in prepared], [1, 3])
        self.assertEqual(prepared[0]["choice_token_ids"],
                         prepared[1]["choice_token_ids"])
        self.assertEqual(prepared[0]["prompt_sha256"], prepared[1]["prompt_sha256"])
        self.assertTrue(all(row["batch"]["input_ids"].device.type == "cpu"
                            for row in prepared))
        summaries = runner.processor_shape_summary(prepared, merge_size=1)
        self.assertEqual(summaries, [
            {"sequence_tokens": 2, "video_grid_thw": [16, 1, 1],
             "merged_visual_tokens": 16},
            {"sequence_tokens": 2, "video_grid_thw": [16, 1, 1],
             "merged_visual_tokens": 16},
        ])

    def test_scoring_calls_model_exactly_twice_without_generation(self):
        prepared = [
            {"batch": {"input_ids": torch.tensor([[1, 2]])},
             "choice_token_ids": [5, 6, 7, 8, 9], "target_index": 1},
            {"batch": {"input_ids": torch.tensor([[1, 2]])},
             "choice_token_ids": [5, 6, 7, 8, 9], "target_index": 3},
        ]

        class Model:
            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                logits = torch.zeros((1, 1, 16), dtype=torch.float32)
                logits[0, 0, 6 if len(self.calls) == 1 else 8] = 3.0
                return SimpleNamespace(logits=logits)

        model = Model()
        guards = []
        lifecycle = []

        def before_forward():
            guards.append({"ok": True})
            lifecycle.append("guard")

        fake_cuda = SimpleNamespace(
            synchronize=lambda device: lifecycle.append("synchronize"),
            empty_cache=lambda: lifecycle.append("empty_cache"),
        )
        fake_torch_cuda = SimpleNamespace(cuda=fake_cuda)

        class TrackingModel(Model):
            def __call__(self, **kwargs):
                lifecycle.append("forward")
                return super().__call__(**kwargs)

        model = TrackingModel()
        result = runner.score_two_inputs(
            model=model, prepared=prepared, torch_module=torch, device="cpu",
            before_forward=before_forward,
            after_forward_cleanup=lambda: runner._release_cuda_after_forward(
                fake_torch_cuda, 0),
        )
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(guards), 2)
        self.assertEqual(result["attempted_forwards"], 2)
        self.assertEqual(result["completed_forwards"], 2)
        self.assertEqual(result["top_choice_correct_count"], 2)
        self.assertEqual(result["top_choice_accuracy"], 1.0)
        self.assertEqual(result["target_vs_best_foil_margin_nats"]["n"], 2)
        self.assertEqual(lifecycle, [
            "guard", "forward", "synchronize", "empty_cache", "guard",
            "forward", "synchronize", "empty_cache",
        ])
        for call in model.calls:
            self.assertFalse(call["use_cache"])
            self.assertEqual(call["logits_to_keep"], 1)
        self.assertFalse(hasattr(model, "generate"))

    def test_failed_forward_keeps_attempt_and_completed_call_counts(self):
        prepared = [{
            "batch": {"input_ids": torch.tensor([[1, 2]])},
            "choice_token_ids": [5, 6, 7, 8, 9],
            "target_index": 1,
        }, {
            "batch": {"input_ids": torch.tensor([[1, 2]])},
            "choice_token_ids": [5, 6, 7, 8, 9],
            "target_index": 3,
        }]

        class MalformedModel:
            calls = 0

            def __call__(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(logits=torch.zeros((1, 0, 16)))

        progress = []
        with self.assertRaisesRegex(ValueError, "model_logits_schema_invalid"):
            runner.score_two_inputs(
                model=MalformedModel(), prepared=prepared,
                torch_module=torch, device="cpu",
                progress=lambda attempted, completed: progress.append(
                    (attempted, completed)),
            )
        self.assertEqual(progress, [(1, 0), (1, 1)])

    def test_safe_report_has_required_scope_and_denies_faithfulness(self):
        report = runner._base_report(
            started_utc="2026-10-19T00:00:00+00:00", dry_run=True,
            pair_sha256="a" * 64,
            source_sha256={"runner": "b" * 64},
        )
        self.assertEqual(report["requested_videos"], 2)
        self.assertEqual(report["planned_model_forwards"], 2)
        self.assertFalse(report["faithfulness_result"])
        self.assertFalse(report["faithfulness_scoring_authorized"])
        self.assertEqual(report["model_revision"], runner.MODEL_REVISION)
        self.assertEqual(report["minimum_free_vram_gib_before_model_load"], 13.5)
        self.assertEqual(report["minimum_free_vram_gib_before_each_forward"],
                         runner._memory_gib(runner.MIN_FREE_VRAM_BEFORE_FORWARD))
        runner._assert_aggregate_only(report)

    def test_source_manifest_hashes_frozen_protocol(self):
        repo_root = Path(__file__).resolve().parents[1]
        hashes = runner._source_hashes(repo_root)
        self.assertIn("experiment_protocol", hashes)
        self.assertRegex(hashes["experiment_protocol"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
