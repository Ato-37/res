from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from audit_nextgqa_prompt_tokenizer import (
    MODEL_REVISION,
    DEFAULT_FRAME_COUNT,
    validate_answer_label_tokens,
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


if __name__ == "__main__":
    unittest.main()
