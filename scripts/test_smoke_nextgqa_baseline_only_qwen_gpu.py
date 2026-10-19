from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from smoke_nextgqa_baseline_only_qwen_gpu import (
    _assert_aggregate_only,
    build_arg_parser,
    derive_answer_token_ids,
    validate_output_path,
    write_manifest_exclusive,
)


class BaselineOnlyQwenRunnerSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        (self.repo / "results").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_parser_requires_output_and_recognizes_dry_run(self):
        with self.assertRaises(SystemExit):
            build_arg_parser().parse_args([])
        parsed = build_arg_parser().parse_args([
            "--output", "results/preflight.json", "--dry-run"])
        self.assertTrue(parsed.dry_run)
        self.assertEqual(parsed.output, Path("results/preflight.json"))

    def test_answer_ids_are_resolved_as_exact_prompt_continuations(self):
        class ContinuationTokenizer:
            def __init__(self):
                self.calls = []

            def encode(self, text, add_special_tokens=False):
                self.calls.append(text)
                prefix = "rendered QA Answer:"
                if text == prefix:
                    return [101, 102]
                if text.startswith(prefix) and text[len(prefix):] in "ABCDE":
                    return [101, 102, 200 + "ABCDE".index(text[len(prefix):])]
                raise ValueError("unexpected_input")

        tokenizer = ContinuationTokenizer()
        ids = derive_answer_token_ids(tokenizer, "rendered QA Answer:")
        self.assertEqual(ids, [200, 201, 202, 203, 204])
        self.assertEqual(tokenizer.calls[0], "rendered QA Answer:")
        self.assertTrue(all(call.startswith("rendered QA Answer:")
                            for call in tokenizer.calls))

    def test_output_must_be_under_results_and_must_not_exist(self):
        path = validate_output_path(Path("results/run.json"), self.repo)
        self.assertEqual(path, (self.repo / "results/run.json").resolve())
        for invalid in (Path("/tmp/out.json"), Path("experiments/out.json"),
                        Path("results/out.txt")):
            with self.subTest(path=invalid):
                with self.assertRaisesRegex(ValueError,
                                            "output_must_be_json_under_results"):
                    validate_output_path(invalid, self.repo)
        existing = self.repo / "results/existing.json"
        existing.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "output_already_exists"):
            validate_output_path(Path("results/existing.json"), self.repo)

    def test_manifest_is_aggregate_only_private_and_exclusive(self):
        path = self.repo / "results/aggregate.json"
        report = {
            "status": "passed", "requested_forwards": 3,
            "completed_forwards": 3, "faithfulness_result": False,
            "aggregate_pq_log_probability_margin_nats": {
                "n": 3, "mean": -0.25, "min": -0.5, "max": 0.0},
        }
        write_manifest_exclusive(path, report)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            write_manifest_exclusive(path, report)

    def test_sensitive_nested_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError,
                                    "aggregate_report_contains_sensitive_field"):
            _assert_aggregate_only({"summary": {"frames": ["raw"]}})
        with self.assertRaisesRegex(ValueError,
                                    "aggregate_report_contains_sensitive_field"):
            _assert_aggregate_only({"records": []})
        with self.assertRaisesRegex(ValueError,
                                    "aggregate_report_contains_sensitive_field"):
            _assert_aggregate_only({"summary": {"question_text": "not allowed"}})

    def test_source_hash_inventory_names_pass_aggregate_field_filter(self):
        _assert_aggregate_only({"source_sha256": {
            "synthetic_fixture": "a" * 64,
            "qa_video_batch_builder": "b" * 64,
            "answer_token_binding": "c" * 64,
            "atomic_batch_freeze": "d" * 64,
        }})

    def test_symlink_escape_is_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        link = self.repo / "results/escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable")
        with self.assertRaisesRegex(ValueError, "output_must_be_json_under_results"):
            validate_output_path(Path("results/escape/out.json"), self.repo)


if __name__ == "__main__":
    unittest.main()
