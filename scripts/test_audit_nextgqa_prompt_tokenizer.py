from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - test environment dependent
    torch = None

from audit_nextgqa_prompt_tokenizer import (
    MODEL_REVISION,
    DEFAULT_FRAME_COUNT,
    validate_answer_label_tokens,
    validate_target_appended_batch,
    safe_failure_code,
    _write_exclusive,
)
from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video


class RecordingProcessor:
    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "opaque-template-output"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return "opaque-processor-output"


class PromptTokenizerAuditTests(unittest.TestCase):
    def test_answer_labels_must_be_distinct_single_tokens(self):
        class Tokenizer:
            def __init__(self, rows):
                self.rows = rows

            def __call__(self, label, *, add_special_tokens):
                if add_special_tokens is not False:
                    raise AssertionError("special_tokens_must_be_disabled")
                return {"input_ids": self.rows[ord(label) - ord("A")]}

        self.assertEqual(validate_answer_label_tokens(
            Tokenizer([[11], [12], [13], [14], [15]])), [11, 12, 13, 14, 15])
        with self.assertRaisesRegex(ValueError, "one_token"):
            validate_answer_label_tokens(
                Tokenizer([[11], [12, 99], [13], [14], [15]]))
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_answer_label_tokens(
                Tokenizer([[11], [12], [13], [14], [11]]))

    def test_shared_production_prompt_has_video_then_text_shape(self):
        processor = RecordingProcessor()
        fake_frames = [object() for _ in range(32)]
        item = {"question": "private question", "choices": ["a", "b", "c", "d", "e"]}
        text, messages = build_qa_prompt(
            processor,
            item,
            fake_frames, [30000, 1001])
        self.assertEqual(text, "opaque-template-output")
        self.assertEqual(processor.kwargs,
                         {"tokenize": False, "add_generation_prompt": True})
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual([part["type"] for part in content], ["video", "text"])
        self.assertIs(content[0]["video"], fake_frames)
        self.assertAlmostEqual(content[0]["sample_fps"], 30000 / 1001)
        self.assertIn("Choose the best answer. Respond with one letter only.\nAnswer:",
                      content[1]["text"])
        self.assertIn("private question", content[1]["text"])
        self.assertEqual(DEFAULT_FRAME_COUNT, 16)

    def test_shared_production_processor_contract(self):
        processor = RecordingProcessor()
        pixels, metadata, images = object(), {"frames_indices": list(range(16))}, None
        result = process_qa_video(
            processor, "opaque prompt", images, pixels, metadata)
        self.assertEqual(result, "opaque-processor-output")
        self.assertEqual(processor.call_kwargs, {
            "text": ["opaque prompt"], "images": None, "videos": [pixels],
            "video_metadata": [metadata], "do_sample_frames": False,
            "cap_pixels_per_frame": True, "padding": True,
            "return_tensors": "pt",
        })

    def test_revision_is_pinned(self):
        self.assertEqual(MODEL_REVISION,
                         "cc594898137f460bfe9f0759e9844b3ce807cfb5")

    def test_private_output_is_exclusive_and_mode_0600(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            _write_exclusive(path, {"diagnostic_only": True,
                                    "execution_authorization": "not_granted"})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            saved = json.loads(path.read_text())
            self.assertTrue(saved["diagnostic_only"])
            self.assertEqual(saved["execution_authorization"], "not_granted")
            with self.assertRaises(FileExistsError):
                _write_exclusive(path, {})

    def test_failure_reporting_only_exposes_allowlisted_boundary_codes(self):
        self.assertEqual(safe_failure_code(
            ValueError("target_appended_prefix_or_token_mismatch")),
            "target_appended_prefix_or_token_mismatch")
        self.assertEqual(safe_failure_code(
            RuntimeError("/private/path/and/question")), "audit_failed")


@unittest.skipIf(torch is None, "PyTorch unavailable")
class TargetAppendedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.prefix = {
            "input_ids": torch.tensor([[10, 20, 30]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 3), dtype=torch.int64),
            "mm_token_type_ids": torch.zeros((1, 3), dtype=torch.int64),
            "pixel_values_videos": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "video_grid_thw": torch.tensor([[2, 1, 2]], dtype=torch.int64),
            "second_per_grid_ts": torch.tensor([0.5], dtype=torch.float32),
        }
        self.token_ids = [101, 102, 103, 104, 105]
        self.targets = {}
        for label, token_id in zip("ABCDE", self.token_ids, strict=True):
            self.targets[label] = {
                "input_ids": torch.tensor([[10, 20, 30, token_id]], dtype=torch.int64),
                "attention_mask": torch.ones((1, 4), dtype=torch.int64),
                "mm_token_type_ids": torch.zeros((1, 4), dtype=torch.int64),
                "pixel_values_videos": self.prefix["pixel_values_videos"].clone(),
                "video_grid_thw": self.prefix["video_grid_thw"].clone(),
                "second_per_grid_ts": self.prefix["second_per_grid_ts"].clone(),
            }

    def test_all_five_labels_preserve_processor_expanded_prefix(self):
        results = [validate_target_appended_batch(
            prefix_batch=self.prefix, target_batch=self.targets[label],
            token_id=token_id)
            for label, token_id in zip("ABCDE", self.token_ids, strict=True)]
        self.assertEqual(len(results), 5)
        self.assertTrue(all(item["contextual_target_token_id_matches_standalone"]
                            for item in results))
        self.assertEqual({item["prefix_token_count"] for item in results}, {3})

    def test_rejects_contextual_token_mismatch(self):
        self.targets["C"]["input_ids"][0, -1] = 999
        with self.assertRaisesRegex(ValueError, "prefix_or_token_mismatch"):
            validate_target_appended_batch(
                prefix_batch=self.prefix, target_batch=self.targets["C"],
                token_id=self.token_ids[2])

    def test_rejects_changed_prefix_after_video_expansion(self):
        self.targets["B"]["input_ids"][0, 1] = 999
        with self.assertRaisesRegex(ValueError, "prefix_or_token_mismatch"):
            validate_target_appended_batch(
                prefix_batch=self.prefix, target_batch=self.targets["B"],
                token_id=self.token_ids[1])

    def test_rejects_changed_pixel_or_video_metadata(self):
        self.targets["D"]["pixel_values_videos"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "nontext_processor_input_changed"):
            validate_target_appended_batch(
                prefix_batch=self.prefix, target_batch=self.targets["D"],
                token_id=self.token_ids[3])

    def test_rejects_extra_processor_key(self):
        self.targets["A"]["unexpected"] = torch.tensor([1])
        with self.assertRaisesRegex(ValueError, "processor_key_set_changed"):
            validate_target_appended_batch(
                prefix_batch=self.prefix, target_batch=self.targets["A"],
                token_id=self.token_ids[0])

    def test_rejects_noninteger_and_dtype_changed_target_ids(self):
        self.targets["A"]["input_ids"] = self.targets["A"]["input_ids"].float()
        with self.assertRaisesRegex(ValueError, "input_ids_length_mismatch"):
            validate_target_appended_batch(
                prefix_batch=self.prefix, target_batch=self.targets["A"],
                token_id=self.token_ids[0])

    def test_rejects_inactive_target_attention_mask(self):
        self.targets["A"]["attention_mask"][0, -1] = 0
        with self.assertRaisesRegex(ValueError, "attention_mask_invalid"):
            validate_target_appended_batch(
                prefix_batch=self.prefix, target_batch=self.targets["A"],
                token_id=self.token_ids[0])

    def test_rejects_changed_appended_modality_type(self):
        self.targets["A"]["mm_token_type_ids"][0, -1] = 1
        with self.assertRaisesRegex(ValueError, "token_type_suffix_mismatch"):
            validate_target_appended_batch(
                prefix_batch=self.prefix, target_batch=self.targets["A"],
                token_id=self.token_ids[0])


if __name__ == "__main__":
    unittest.main()
