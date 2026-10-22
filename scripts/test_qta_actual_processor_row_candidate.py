"""CPU-only composition tests for the non-authorizing processor row candidate."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import unittest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from qta_actual_processor_row_candidate import (  # noqa: E402
    build_actual_processor_row_candidate,
    verify_actual_processor_row_candidate,
    verify_actual_processor_row_candidate_self_consistency,
)
from qta_prescore_mapping_binding import interval_list_sha256  # noqa: E402


VIDEO_PLACEHOLDER = "<|video_pad|>"
VIDEO_ID = 99


class FakeTokenizer:
    def _tokens(self, text):
        ids, offsets = [], []
        cursor = 0
        while cursor < len(text):
            if text.startswith(VIDEO_PLACEHOLDER, cursor):
                stop = cursor + len(VIDEO_PLACEHOLDER)
                ids.append(VIDEO_ID)
                offsets.append((cursor, stop))
            else:
                stop = cursor + 1
                ids.append(ord(text[cursor]) + 100)
                offsets.append((cursor, stop))
            cursor = stop
        return ids, offsets

    def encode(self, text, add_special_tokens=False):
        return self._tokens(text)[0]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = self._tokens(text)
        return {"input_ids": ids, "offset_mapping": offsets}


class FakeProcessor:
    def __init__(self, tokenizer=None, fail_on_call=None):
        self.calls = 0
        self.tokenizer = tokenizer or FakeTokenizer()
        self.fail_on_call = fail_on_call

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        text = next(part["text"] for part in messages[0]["content"]
                    if part["type"] == "text")
        return VIDEO_PLACEHOLDER + "\n" + text

    def __call__(self, *, text, images, videos, video_metadata, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("synthetic_processor_failure")
        raw_ids = FakeTokenizer().encode(text[0], add_special_tokens=False)
        ids = []
        for value in raw_ids:
            ids.extend([VIDEO_ID] * 32 if value == VIDEO_ID else [value])
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "pixel_values_videos": videos[0],
            "video_grid_thw": torch.tensor([[32, 1, 1]], dtype=torch.long),
            "second_per_grid_ts": torch.tensor([1.0], dtype=torch.float32),
        }


def _fixtures():
    rows, expected, sampler_rows = [], [], []
    tokenizer_sha = hashlib.sha256(b"fake-tokenizer").hexdigest()
    for index in range(3):
        token = f"opaque-token-{index + 1}"
        source_sha = hashlib.sha256(f"video-{index}".encode()).hexdigest()
        question = f"Which event occurs in clip {index}?"
        choices = [f"choice {index}-{letter}" for letter in "ABCDE"]
        claim = f"A supported claim for video {index}."
        interval = [0.1, 0.9]
        frames = [Image.new("RGB", (2, 2), (index, i, 20)) for i in range(32)]
        rows.append({
            "review_token": token,
            "source_video_sha256": source_sha,
            "question": question,
            "choices": choices,
            "claim": claim,
            "frames": frames,
            "video_duration_seconds": 32.0,
            "reviewed_intervals_seconds": [interval],
            "interval_boundary_guard_seconds": 0.0,
            "processor_revision": "cpu-fake-processor-v1",
            "processor_fingerprint_sha256": hashlib.sha256(b"fake-processor").hexdigest(),
            "video_token_id": VIDEO_ID,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
        })
        sampler = {
            "source_sha256": source_sha,
            "decoded_frame_count": 32,
            "sampled_frame_count": 32,
            "sampled_indices": list(range(32)),
            "sampled_pts": [[i, 1] for i in range(32)],
            "effective_sample_fps": [1, 1],
            "sampled_rgb_sha256": [hashlib.sha256(
                frame.convert("RGB").tobytes()).hexdigest() for frame in frames],
        }
        sampler_rows.append(sampler)
        expected.append({
            "review_token": token,
            "source_video_sha256": source_sha,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "options_sha256": hashlib.sha256(
                json.dumps(choices, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode()).hexdigest(),
            "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
            "reviewed_intervals_sha256": interval_list_sha256([["1/10", "9/10"]]),
            "tokenizer_sha256": tokenizer_sha,
            "processor_revision": "cpu-fake-processor-v1",
            "processor_fingerprint_sha256": hashlib.sha256(b"fake-processor").hexdigest(),
            "sampler_row_sha256": hashlib.sha256(json.dumps(
                sampler, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
            "video_duration_seconds": "32",
            "interval_boundary_guard_seconds": "0",
        })
    return rows, expected, sampler_rows


def _build(rows=None, expected=None, processor=None, sampler_rows=None,
           tokenizer=None):
    rows0, expected0, sampler0 = _fixtures()
    tokenizer = tokenizer or (processor.tokenizer if processor is not None
                              else FakeTokenizer())
    processor = processor or FakeProcessor(tokenizer)
    candidate = build_actual_processor_row_candidate(
        rows=rows0 if rows is None else rows,
        expected_rows=expected0 if expected is None else expected,
        sampler_rows=sampler0 if sampler_rows is None else sampler_rows,
        processor=processor, tokenizer=tokenizer,
        tokenizer_sha256=hashlib.sha256(b"fake-tokenizer").hexdigest(),
        vision_info_fn=lambda _messages, return_video_metadata: (
            None, [(torch.zeros((32, 3)), {"frames_indices": list(range(32)), "fps": 1.0})]),
        torch_module=torch)
    return candidate, processor, rows0, expected0, sampler0


class ActualProcessorRowCandidateTests(unittest.TestCase):
    def test_builds_three_same_prompt_processor_candidates_non_authorizing(self):
        candidate, processor, _rows, _pins, _sampler = _build()
        verify_actual_processor_row_candidate(candidate, expected_rows=_pins)
        self.assertEqual(processor.calls, 3)
        self.assertEqual(candidate["requested_rows"], 3)
        self.assertEqual(len(candidate["candidate_rows"]), 3)
        for output in candidate["candidate_rows"]:
            self.assertEqual(len(output["answer_token_ids"]), 5)
            self.assertEqual(len(set(output["answer_token_ids"])), 5)
            self.assertTrue(output["question_positions"])
            self.assertEqual(output["tokenizer_sha256"], _pins[0]["tokenizer_sha256"])
            self.assertEqual(output["processor_revision"], "cpu-fake-processor-v1")
            self.assertIn("question_sha256", output)
            self.assertIn("options_sha256", output)
            self.assertIn("claim_sha256", output)
        for field in ("model_load_permitted", "scoring_authorized",
                      "human_identity_verified", "semantic_review_authenticated"):
            self.assertIs(candidate[field], False)
        self.assertEqual(candidate["execution_authorization"], "not_granted")
        self.assertIs(candidate["faithfulness_scored"], False)

    def test_preprocessor_rejects_source_interval_token_and_order_mutations(self):
        for mutation in ("source", "interval", "token", "order"):
            with self.subTest(mutation=mutation):
                rows, pins, sampler_rows = _fixtures()
                if mutation == "source":
                    rows[0]["source_video_sha256"] = "f" * 64
                elif mutation == "interval":
                    rows[0]["reviewed_intervals_seconds"] = [[1.1, 1.9]]
                elif mutation == "token":
                    rows[0]["review_token"] = "swapped-token"
                else:
                    rows[0], rows[1] = rows[1], rows[0]
                processor = FakeProcessor()
                with self.assertRaises(ValueError):
                    _build(rows, pins, processor, sampler_rows)
                self.assertEqual(processor.calls, 0)

    def test_rejects_processor_batch_tamper_after_candidate_build(self):
        candidate, _processor, _rows, _pins, _sampler = _build()
        candidate["candidate_rows"][1]["batch"]["input_ids"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "candidate_batch_digest_mismatch"):
            verify_actual_processor_row_candidate(candidate, expected_rows=_pins)

    def test_external_verifier_rejects_tampered_expected_pin(self):
        candidate, _processor, _rows, pins, _sampler = _build()
        pins[1] = {**pins[1], "claim_sha256": "f" * 64}
        with self.assertRaisesRegex(ValueError, "candidate_external_expected_pin_mismatch"):
            verify_actual_processor_row_candidate(candidate, expected_rows=pins)

    def test_self_consistency_verifier_rejects_faithfulness_claim(self):
        candidate, *_ = _build()
        candidate["faithfulness_scored"] = True
        with self.assertRaisesRegex(ValueError, "candidate_status_or_authority_invalid"):
            verify_actual_processor_row_candidate_self_consistency(candidate)

    def test_self_consistency_rejects_spliced_prompt_binding_artifact(self):
        candidate, *_ = _build()
        candidate["prompt_answer_token_bindings_raw"] += b" "
        with self.assertRaisesRegex(ValueError, "candidate_prompt_answer_binding_mismatch"):
            verify_actual_processor_row_candidate_self_consistency(candidate)

    def test_rejects_duplicate_source_hashes_before_processor_calls(self):
        rows, pins, sampler_rows = _fixtures()
        rows[1]["source_video_sha256"] = rows[0]["source_video_sha256"]
        pins[1]["source_video_sha256"] = pins[0]["source_video_sha256"]
        processor = FakeProcessor()
        with self.assertRaisesRegex(ValueError, "candidate_order_or_source_binding_mismatch"):
            _build(rows, pins, processor, sampler_rows)
        self.assertEqual(processor.calls, 0)

    def test_rejects_processor_tokenizer_mismatch_before_first_call(self):
        rows, pins, sampler_rows = _fixtures()
        tokenizer = FakeTokenizer()
        processor = FakeProcessor(FakeTokenizer())
        with self.assertRaisesRegex(ValueError, "candidate_processor_tokenizer_identity_mismatch"):
            _build(rows, pins, processor, sampler_rows, tokenizer=tokenizer)
        self.assertEqual(processor.calls, 0)

    def test_rejects_sampler_row_splice_before_processor_call(self):
        rows, pins, sampler_rows = _fixtures()
        sampler_rows[0] = dict(sampler_rows[1])
        processor = FakeProcessor()
        with self.assertRaises(ValueError):
            _build(rows, pins, processor, sampler_rows)
        self.assertEqual(processor.calls, 0)

    def test_rejects_source_text_splice_before_processor_call(self):
        rows, pins, sampler_rows = _fixtures()
        rows[1]["claim"] += " altered"
        processor = FakeProcessor()
        with self.assertRaisesRegex(ValueError, "candidate_source_text_binding_mismatch"):
            _build(rows, pins, processor, sampler_rows)
        self.assertEqual(processor.calls, 0)

    def test_rejects_rgb_frame_splice_before_processor_call(self):
        rows, pins, sampler_rows = _fixtures()
        rows[0]["frames"][4], rows[1]["frames"][4] = (
            rows[1]["frames"][4], rows[0]["frames"][4])
        processor = FakeProcessor()
        with self.assertRaisesRegex(ValueError, "candidate_rgb_frame_sampler_hash_mismatch"):
            _build(rows, pins, processor, sampler_rows)
        self.assertEqual(processor.calls, 0)

    def test_rejects_duration_or_guard_mutation_before_processor_call(self):
        for field, value in (("video_duration_seconds", 31.0),
                             ("interval_boundary_guard_seconds", 0.02)):
            with self.subTest(field=field):
                rows, pins, sampler_rows = _fixtures()
                rows[0][field] = value
                processor = FakeProcessor()
                with self.assertRaisesRegex(ValueError, "candidate_duration_or_guard_pin_mismatch"):
                    _build(rows, pins, processor, sampler_rows)
                self.assertEqual(processor.calls, 0)

    def test_second_row_processor_failure_returns_no_candidate(self):
        rows, pins, sampler_rows = _fixtures()
        processor = FakeProcessor(fail_on_call=2)
        with self.assertRaisesRegex(RuntimeError, "synthetic_processor_failure"):
            _build(rows, pins, processor, sampler_rows)
        self.assertEqual(processor.calls, 2)

    def test_verifier_rejects_authority_flip_and_row_reordering(self):
        candidate, _processor, _rows, pins, _sampler = _build()
        candidate["model_load_permitted"] = True
        with self.assertRaisesRegex(ValueError, "candidate_status_or_authority_invalid"):
            verify_actual_processor_row_candidate(candidate, expected_rows=pins)
        candidate, _processor, _rows, pins, _sampler = _build()
        candidate["candidate_rows"].reverse()
        with self.assertRaisesRegex(ValueError, "candidate_order_or_row_commitment_mismatch"):
            verify_actual_processor_row_candidate(candidate, expected_rows=pins)


if __name__ == "__main__":
    unittest.main()
