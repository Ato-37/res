"""CPU-only structural tests for the fixed synthetic tokenizer audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from audit_qwen_claim_tokenization_cpu import (
    FIXED_PROMPT,
    FIXED_TARGETS,
    audit_fixed_cases,
    audit_fixed_choice_labels,
    build_report,
    ids_list,
    MODEL_REVISION,
    safe_output_path,
    write_report,
)


class FakeTokenizer:
    """Tiny deterministic tokenizer with a fixed prefix/target boundary merge."""

    def __call__(self, text: str, *, add_special_tokens: bool):
        if add_special_tokens is not False:
            raise AssertionError("fixture must not add tokenizer special tokens")
        encoded = []
        index = 0
        while index < len(text):
            if text.startswith(":A", index):
                encoded.append(900_001)
                index += 2
            else:
                encoded.append(ord(text[index]))
                index += 1
        return {"input_ids": encoded}


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    @staticmethod
    def _render(messages, add_generation_prompt):
        user = next(message["content"] for message in messages if message["role"] == "user")
        prefix = f"USER:{user}<EOT>ASSISTANT:"
        assistant = [message["content"] for message in messages if message["role"] == "assistant"]
        if assistant:
            return prefix + assistant[-1] + "<END>"
        if add_generation_prompt:
            return prefix
        return f"USER:{user}<EOT>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=False):
        rendered = self._render(messages, add_generation_prompt)
        if not tokenize:
            return rendered
        ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        return {"input_ids": [ids]} if return_dict else ids


class BoundarySafeProcessor(FakeProcessor):
    """Model a chat template whose assistant-generation marker is a special token."""

    @staticmethod
    def _render(messages, add_generation_prompt):
        user = next(message["content"] for message in messages if message["role"] == "user")
        prefix = f"USER:{user}<EOT>ASSISTANT⟨GEN⟩"
        assistant = [message["content"] for message in messages if message["role"] == "assistant"]
        if assistant:
            return prefix + assistant[-1] + "<END>"
        if add_generation_prompt:
            return prefix
        return f"USER:{user}<EOT>"


class ClaimTokenizationAuditTests(unittest.TestCase):
    def test_fixed_fixture_is_exactly_12_prospective_synthetic_cases(self):
        self.assertEqual(len(FIXED_TARGETS), 12)
        self.assertEqual(FIXED_PROMPT, "Describe the visible scene briefly. Use only visual information.")
        self.assertEqual(FIXED_TARGETS[0:4], ("A", " A", "The object moves.", " The object moves."))
        self.assertEqual(
            FIXED_TARGETS[9:12], ("\nThe object moves.", "水中の物体。", "é")
        )

    def test_boundary_audit_detects_fixture_merge_without_emitting_text_or_ids(self):
        report = audit_fixed_cases(FakeProcessor())
        self.assertEqual(report["fixed_case_count"], 12)
        self.assertEqual(report["literal_concat_match_count"], 11)
        self.assertEqual(report["completed_chat_prefix_stable_count"], 11)
        self.assertFalse(report["cases"][0]["literal_concat_equals_prefix_plus_standalone"])
        self.assertFalse(report["cases"][0]["completed_chat_prefix_stable"])
        self.assertTrue(report["cases"][1]["literal_concat_equals_prefix_plus_standalone"])
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("The object moves", serialized)
        self.assertNotIn("水中の物体", serialized)
        self.assertNotIn("900001", serialized)
        self.assertIn("literal_concat_token_ids_sha256", serialized)

    def test_ids_list_normalizes_a_single_batch_row(self):
        self.assertEqual(ids_list({"input_ids": [[1, 2, 3]]}), [1, 2, 3])

    def test_choice_labels_are_checked_at_chat_generation_boundary(self):
        report = audit_fixed_choice_labels(BoundarySafeProcessor())
        self.assertTrue(report["distinct_single_token_labels"])
        self.assertTrue(report["gate_pass"])
        self.assertEqual(report["label_count"], 5)
        self.assertEqual(report["stable_prefix_count"], 5)
        self.assertEqual(report["contextual_label_match_count"], 5)
        self.assertEqual([row["label_index"] for row in report["labels"]], list(range(5)))
        serialized = json.dumps(report)
        self.assertNotIn("Synthetic question", serialized)
        self.assertNotIn("token_ids\"", serialized)

    def test_choice_audit_rejects_non_distinct_or_multitoken_labels(self):
        class BadTokenizer(FakeTokenizer):
            def __call__(self, text, *, add_special_tokens):
                if text in {"A", "B", "C", "D", "E"}:
                    return {"input_ids": [ord(text[0]), ord(text[0])]} if text == "E" else {
                        "input_ids": [42] if text == "D" else [ord(text[0])]
                    }
                return super().__call__(text, add_special_tokens=add_special_tokens)

        class BadProcessor(BoundarySafeProcessor):
            def __init__(self):
                self.tokenizer = BadTokenizer()

        report = audit_fixed_choice_labels(BadProcessor())
        self.assertFalse(report["distinct_single_token_labels"])
        self.assertFalse(report["gate_pass"])
        self.assertEqual(report["contextual_label_match_count"], 3)

    def test_artifact_is_private_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "results" / "audit.json"
            artifact = write_report({"synthetic": True}, output)
            self.assertEqual(json.loads(artifact.read_text()), {"synthetic": True})
            self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_report({"different": True}, output)
            with self.assertRaises(ValueError):
                safe_output_path(str(root / "elsewhere" / "x.json"), root)

    def test_report_has_complete_counts_timing_and_hashes_without_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / MODEL_REVISION
            snapshot.mkdir()
            fake_tokenizers = types.SimpleNamespace(__version__="test-tokenizers")
            fake_transformers = types.SimpleNamespace(__version__="test-transformers")
            with mock.patch.dict(
                sys.modules,
                {"tokenizers": fake_tokenizers, "transformers": fake_transformers},
            ):
                report = build_report(
                    FakeProcessor(), runner=Path(__file__), repo_root=root, snapshot=snapshot
                )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["requested_cases"], 12)
        self.assertEqual(report["effective_cases"], 12)
        self.assertEqual(report["failures"], 0)
        self.assertGreaterEqual(report["timing"]["tokenizer_comparison_core_seconds"], 0)
        self.assertIn("excludes_imports_processor_loading", report["timing"]["scope"])
        hashes = report["source_sha256"]["installed_implementation_files"]
        self.assertEqual(set(hashes), {
            "processor_class_source", "tokenizer_class_source", "chat_template_method_source",
            "chat_template_utils_module_source",
        })
        self.assertNotIn(str(root), json.dumps(report))


if __name__ == "__main__":
    unittest.main()
