"""CPU contract tests for the fixed sequence-length numeric diagnostic."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest import mock

import numpy as np
import torch

import run_qwen_sequence_length_numerics_gpu_diagnostic as diagnostic


class SequenceLengthNumericsTests(unittest.TestCase):
    @staticmethod
    def valid_checkpoint_audit():
        shards = []
        for name in diagnostic.CHECKPOINT_AUDIT_SHARDS:
            size, sha = diagnostic.CHECKPOINT_AUDIT_EXPECTED_SHARD_METADATA[name]
            shards.append({
                "filename": name,
                "expected_size_bytes": size,
                "observed_size_bytes": size,
                "bytes_read": size,
                "expected_sha256": sha,
                "observed_sha256": sha,
                "sha256_matches": True,
                "stable_during_read": True,
                "failure_code": None,
            })
        return {
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "revision": diagnostic.REVISION,
            "byte_authentication_status": "verified_current_local_bytes",
            "historical_gpu_runs_byte_verified": False,
            "requested_shards": 5,
            "effective_shards": 5,
            "expected_shard_bytes": diagnostic.CHECKPOINT_AUDIT_EXPECTED_BYTES,
            "bytes_read": diagnostic.CHECKPOINT_AUDIT_EXPECTED_BYTES,
            "failure_count": 0,
            "failure_codes": [],
            "api_metadata_hash_matches_pinned_artifact": True,
            "api_metadata_sha256": diagnostic.CHECKPOINT_AUDIT_API_SHA256,
            "source_commit": diagnostic.CHECKPOINT_AUDIT_SOURCE_COMMIT,
            "runner_sha256": diagnostic.CHECKPOINT_AUDIT_RUNNER_SHA256,
            "shards": shards,
            "index": {
                "git_blob_sha1_matches": True,
                "failure_code": None,
                "expected_git_blob_sha1": diagnostic.CHECKPOINT_AUDIT_INDEX_BLOB_SHA1,
                "observed_git_blob_sha1": diagnostic.CHECKPOINT_AUDIT_INDEX_BLOB_SHA1,
                "expected_size_bytes": diagnostic.CHECKPOINT_AUDIT_INDEX_BYTES,
                "observed_size_bytes": diagnostic.CHECKPOINT_AUDIT_INDEX_BYTES,
            },
            "index_weight_map_entries": 729,
            "index_tensor_payload_bytes": diagnostic.CHECKPOINT_AUDIT_INDEX_TENSOR_BYTES,
        }

    def test_checkpoint_audit_fixed_provenance_gate_and_strict_types(self):
        report = self.valid_checkpoint_audit()
        diagnostic.validate_checkpoint_audit(report)
        for mutate in (
            lambda r: r.update(revision="0" * 40),
            lambda r: r.update(api_metadata_sha256="0" * 64),
            lambda r: r.update(requested_shards=True),
            lambda r: r.update(bytes_read=diagnostic.CHECKPOINT_AUDIT_EXPECTED_BYTES - 1),
            lambda r: r["shards"][0].update(observed_sha256="0" * 64),
            lambda r: r["shards"][0].update(observed_size_bytes=True),
            lambda r: r["index"].update(observed_git_blob_sha1="0" * 40),
            lambda r: r.update(index_weight_map_entries=True),
        ):
            candidate = self.valid_checkpoint_audit()
            mutate(candidate)
            with self.assertRaises(ValueError):
                diagnostic.validate_checkpoint_audit(candidate)

    def test_checkpoint_audit_loader_binds_report_to_artifact_sha(self):
        source_repo = Path(__file__).resolve().parents[1]
        raw = (source_repo / diagnostic.CHECKPOINT_AUDIT_PATH).read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            path = repo / diagnostic.CHECKPOINT_AUDIT_PATH
            path.parent.mkdir(parents=True)
            path.write_bytes(raw)
            report, digest = diagnostic.load_checkpoint_audit(repo)
            self.assertEqual(report["revision"], diagnostic.REVISION)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

    def test_same_valid_json_with_different_bytes_is_rejected_before_parse(self):
        source_repo = Path(__file__).resolve().parents[1]
        raw = (source_repo / diagnostic.CHECKPOINT_AUDIT_PATH).read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            path = repo / diagnostic.CHECKPOINT_AUDIT_PATH
            path.parent.mkdir(parents=True)
            path.write_bytes(raw + b" ")
            with mock.patch.object(diagnostic.json, "loads") as loads:
                with self.assertRaisesRegex(ValueError, "artifact SHA"):
                    diagnostic.load_checkpoint_audit(repo)
            loads.assert_not_called()

    def test_fixed_matrix_has_nine_expected_rows(self):
        pairs = diagnostic.build_comparison_plan(584)
        self.assertEqual(len(pairs), 9)
        self.assertEqual(
            [(p["left"], p["right"]) for p in pairs[:5]],
            [
                ("generation_token_0", "full_584_row_583"),
                ("generation_token_0", "full_585_row_583"),
                ("generation_token_0", "full_586_row_583"),
                ("generation_token_1", "full_585_row_584"),
                ("generation_token_1", "full_586_row_584"),
            ],
        )
        self.assertEqual(
            [(p["left"], p["right"]) for p in pairs[5:]],
            [
                ("full_584_row_583", "full_585_row_583"),
                ("full_584_row_583", "full_586_row_583"),
                ("full_585_row_583", "full_586_row_583"),
                ("full_585_row_584", "full_586_row_584"),
            ],
        )
        self.assertEqual([p["prediction_row"] for p in pairs], [583, 583, 583, 584, 584, 583, 583, 583, 584])
        captured_full_rows = {
            name for pair in pairs for name in (pair["left"], pair["right"])
            if name.startswith("full_")
        }
        self.assertEqual(captured_full_rows, {
            "full_584_row_583", "full_585_row_583", "full_585_row_584",
            "full_586_row_583", "full_586_row_584",
        })

    def test_projection_matrix_has_five_full_rows_and_ten_scalar_comparisons(self):
        specs = diagnostic.projection_comparison_plan(584)
        self.assertEqual(
            [(item["full_length"], item["row"], item["generation_name"]) for item in specs],
            [
                (584, 583, "generation_token_0"),
                (585, 583, "generation_token_0"),
                (585, 584, "generation_token_1"),
                (586, 583, "generation_token_0"),
                (586, 584, "generation_token_1"),
            ],
        )
        self.assertEqual(2 * len(specs), 10)

    def test_projection_hook_records_full_and_single_row_inputs_without_mutation(self):
        torch.manual_seed(17)
        head = torch.nn.Linear(5, 7, bias=True)
        hidden = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5)
        original = hidden.clone()
        trace = diagnostic.LMHeadInputTrace(head)
        try:
            with torch.inference_mode():
                full_logits = head(hidden)
                chosen = hidden[:, 2:3, :].clone()
                one_logits = head(chosen)
        finally:
            trace.close()
        self.assertEqual([call["shape"] for call in trace.calls], [(1, 4, 5), (1, 1, 5)])
        self.assertTrue(torch.equal(hidden, original))
        self.assertTrue(torch.equal(trace.calls[0]["hidden_cpu"], original))
        self.assertTrue(torch.equal(trace.calls[1]["hidden_cpu"], chosen))
        self.assertTrue(torch.equal(full_logits[:, 2:3, :], one_logits))
        pair = {
            "left": "full_row",
            "right": "single_row_projection",
            "comparison_type": "projection_shape_test",
            "token_index": 0,
            "prediction_row": 2,
        }
        summary = diagnostic.summarize_pair(
            pair, full_logits[0, 2].detach(), one_logits[0, 0].detach(), target_token_id=3
        )
        self.assertEqual(summary["raw_max_abs"], 0.0)
        self.assertEqual(summary["centered_max_abs"], 0.0)
        self.assertEqual(summary["kl_symmetric_mean_nats"], 0.0)
        self.assertTrue(summary["numeric_crosscheck_within_1e_10"])

    def test_projection_hook_captures_only_prespecified_hidden_rows(self):
        head = torch.nn.Linear(4, 3, bias=False)
        hidden = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4)
        trace = diagnostic.LMHeadInputTrace(head, selected_rows=(7, 9))
        try:
            head(hidden)
        finally:
            trace.close()
        self.assertEqual(trace.calls[0]["shape"], (1, 10, 4))
        self.assertEqual(trace.calls[0]["captured_rows"], (7, 9))
        self.assertEqual(tuple(trace.calls[0]["hidden_cpu"].shape), (1, 2, 4))
        self.assertTrue(torch.equal(trace.calls[0]["hidden_cpu"], hidden[:, [7, 9], :]))

    def test_hidden_comparison_is_scalar_only_and_validates_dtype_shape_finiteness(self):
        left = torch.tensor([1.0, 2.0, 4.0], dtype=torch.bfloat16)
        right = torch.tensor([1.0, 3.0, 4.0], dtype=torch.bfloat16)
        pair = {"left": "a", "right": "b", "prediction_row": 5}
        result = diagnostic.summarize_hidden_pair(pair, left, right)
        self.assertEqual(result["raw_max_abs"], 1.0)
        self.assertAlmostEqual(result["raw_mean_abs"], 1 / 3, places=7)
        self.assertTrue(result["numeric_crosscheck_within_1e_10"])
        self.assertNotIn("hidden_values", result)
        self.assertNotIn("hidden_tensor", result)
        with self.assertRaises(ValueError):
            diagnostic.summarize_hidden_pair(pair, left, right.float())
        with self.assertRaises(ValueError):
            diagnostic.summarize_hidden_pair(pair, left, torch.ones(4, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            diagnostic.summarize_hidden_pair(pair, left, torch.tensor([1.0, float("inf"), 4.0], dtype=torch.bfloat16))

    def test_outer_forward_trace_records_explicit_and_default_projection_width(self):
        class Dummy(torch.nn.Module):
            def forward(self, *, logits_to_keep=0):
                return logits_to_keep

        model = Dummy()
        trace = diagnostic.OuterForwardConfigTrace(model)
        try:
            model(logits_to_keep=1)
            model()
        finally:
            trace.close()
        self.assertEqual(trace.calls, [
            {"effective_logits_to_keep": 1, "explicit_kwarg": True},
            {"effective_logits_to_keep": 0, "explicit_kwarg": False},
        ])

    def test_common_additive_offset_changes_raw_only(self):
        left = np.array([-3.0, 0.5, 7.0, 1.25], dtype=np.float64)
        right = left + 1.0
        pair = diagnostic.build_comparison_plan()[0]
        out = diagnostic.summarize_pair(pair, left, right, target_token_id=2)
        self.assertEqual(out["raw_max_abs"], 1.0)
        self.assertAlmostEqual(out["raw_mean_abs"], 1.0, places=14)
        self.assertAlmostEqual(out["vocab_mean_offset_delta"], 1.0, places=14)
        self.assertAlmostEqual(out["centered_max_abs"], 0.0, places=14)
        self.assertAlmostEqual(out["logprob_max_abs"], 0.0, places=14)
        self.assertAlmostEqual(out["selected_token_logprob_abs_difference_nats"], 0.0, places=14)
        self.assertAlmostEqual(out["kl_symmetric_mean_nats"], 0.0, places=14)
        self.assertTrue(out["numeric_crosscheck_within_1e_10"])

    def test_independent_numpy_torch_and_python_scoring_agree(self):
        left = [0.0, -2.5, 13.0, 1.75, -11.0]
        right = [0.25, -2.0, 12.5, 1.5, -10.0]
        pair = diagnostic.build_comparison_plan()[8]
        out = diagnostic.summarize_pair(pair, left, right, target_token_id=3)
        self.assertTrue(out["numeric_crosscheck_within_1e_10"])
        self.assertLessEqual(out["torch64_numpy_python_max_abs_error"], 1e-10)
        self.assertGreater(out["kl_left_to_right_nats"], 0.0)
        self.assertGreater(out["kl_right_to_left_nats"], 0.0)

    def test_invalid_vectors_and_target_indices_fail_closed(self):
        with self.assertRaises(ValueError):
            diagnostic._torch_metrics([1.0], [1.0, 2.0], 0)
        with self.assertRaises(ValueError):
            diagnostic._numpy_metrics([1.0, float("nan")], [1.0, 2.0], 0)
        with self.assertRaises(ValueError):
            diagnostic._torch_metrics([1.0, 2.0], [2.0, 3.0], True)
        with self.assertRaises(ValueError):
            diagnostic.build_comparison_plan(0)

    def test_fixed_target_hash_rejects_a_substituted_target(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            diagnostic.require_expected_target_hash("0" * 64)
        diagnostic.require_expected_target_hash(diagnostic.EXPECTED_TARGET_SHA256)

    def test_failure_path_persists_sanitized_partial_counts_privately(self):
        repo = Path(__file__).resolve().parents[1]
        output = repo / "results" / f".test_sequence_length_failure_{uuid.uuid4().hex}.json"

        def fail_after_partial(path):
            diagnostic.RUN_PROGRESS.update(
                stage="teacher_forcing_prefix_585",
                completed_model_calls=3,
                effective_frames=4,
                completed_teacher_prefixes=1,
                frozen_target_sha256=diagnostic.EXPECTED_TARGET_SHA256,
                partial_evidence={"safe_count": 3},
            )
            raise RuntimeError("must not export prompt or exception contents")

        try:
            with mock.patch.object(diagnostic, "run_gpu_diagnostic", fail_after_partial), \
                 mock.patch.object(diagnostic, "_failure_provenance", return_value={"source_commit": "fixture"}), \
                 self.assertRaises(SystemExit) as raised:
                diagnostic.main(["--output", str(output.relative_to(repo))])
            self.assertEqual(raised.exception.code, 1)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["completed_model_calls"], 3)
            self.assertEqual(record["completed_teacher_prefixes"], 1)
            self.assertEqual(record["effective_frames"], 4)
            self.assertEqual(record["evidence"]["safe_count"], 3)
            self.assertEqual(record["evidence"]["source_commit"], "fixture")
            self.assertNotIn("exception_message", record)
            self.assertNotIn("must not export", output.read_text(encoding="utf-8"))
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        finally:
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
