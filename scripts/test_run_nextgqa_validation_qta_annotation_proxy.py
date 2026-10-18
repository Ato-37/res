"""CPU-only synthetic contract tests for the exploratory QTA proxy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from fractions import Fraction
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nextgqa_validation_qta_annotation_proxy as runner


class ChoiceTokenKernelTests(unittest.TestCase):
    def test_runner_uses_strict_shared_choice_continuation_contract(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                if text == "prompt":
                    return [10, 11]
                if text.startswith("prompt") and len(text) == 7:
                    return [10, 11, ord(text[-1])]
                raise ValueError("unexpected prompt")

        self.assertEqual(runner._choice_token_ids(Tokenizer(), "prompt"),
                         [ord(label) for label in "ABCDE"])

        class NegativeIdTokenizer(Tokenizer):
            def encode(self, text, add_special_tokens=False):
                result = super().encode(text, add_special_tokens)
                return [10, 11, -1] if text == "promptA" else result

        with self.assertRaisesRegex(ValueError, "choice_label_not_single_continuation_token"):
            runner._choice_token_ids(NegativeIdTokenizer(), "prompt")


def inventory_fixture():
    roles = [f"review_package_media_{i:02d}" for i in (1, 2, 3)]
    inventory = {"artifacts": [{"role": role, "sha256": f"{i:064x}"}
                               for i, role in enumerate(roles, 1)]}
    provenance = {"inventory_path": runner.V6_INVENTORY,
                  "inventory_sha256": runner.V6_INVENTORY_SHA256,
                  "package_dir": "data/private_nextgqa/reviewer_package_v6_20261018_0243",
                  "execution_authorization": "not_granted"}
    return inventory, provenance


class SourceAndCohortTests(unittest.TestCase):
    def test_support_extent_can_exceed_but_never_mutates_annotation_duration(self):
        annotation_duration = Fraction(50)
        pts = [Fraction(0), Fraction(50417, 1000)]
        self.assertEqual(runner.temporal_support_end(annotation_duration, pts), pts[-1])
        self.assertEqual(annotation_duration, Fraction(50))
        self.assertEqual(runner.temporal_support_end(Fraction(60), pts), Fraction(60))
        with self.assertRaisesRegex(ValueError, "sampled_pts_empty"):
            runner.temporal_support_end(annotation_duration, [])

    def test_qids_may_repeat_across_distinct_videos(self):
        rows = [{"video_id": f"video-{index}", "qid": f"q{index % 11}"}
                for index in range(runner.GOLD_COHORT_N)]
        runner.validate_pilot_cohort_identity(rows)
        rows[-1]["video_id"] = rows[0]["video_id"]
        with self.assertRaisesRegex(ValueError, "historical_pilot_cohort_not_one_row_per_video"):
            runner.validate_pilot_cohort_identity(rows)

    def test_default_manifest_is_resolved_relative_to_checkout(self):
        repo = Path("/workspace/res")
        self.assertEqual(runner.resolve_pilot_manifest(repo, None),
                         repo / runner.PRIVATE_PILOT_COPY)
        supplied = "/private/custom/pilot.json"
        self.assertEqual(runner.resolve_pilot_manifest(repo, supplied), Path(supplied))

    def test_protocol_and_selection_docs_are_in_executed_source_manifest(self):
        self.assertIn("docs/qwen_question_attention_faithfulness_protocol_v1_draft.md",
                      runner.SOURCE_FILES)
        self.assertIn("experiments/2026-10-18_qta_treatment_selection.md",
                      runner.SOURCE_FILES)

    def test_processor_provenance_binds_tokenizer_processor_and_config(self):
        class Tokenizer:
            class Backend:
                @staticmethod
                def to_str():
                    return '{"synthetic_tokenizer":true}'
            backend_tokenizer = Backend()

        class ImageProcessor:
            @staticmethod
            def to_dict():
                return {"size": 224, "synthetic": True}

        class Processor:
            tokenizer = Tokenizer()
            image_processor = ImageProcessor()
            video_token_id = 42

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text('{"model_type":"qwen2_5_vl"}')
            result = runner.processor_provenance(Processor(), model_path)
        self.assertEqual(result["model_revision"], runner.MODEL_REVISION)
        self.assertEqual(result["tokenizer_revision"], runner.MODEL_REVISION)
        self.assertEqual(len(result["processor_fingerprint_sha256"]), 64)
        self.assertEqual(result["model_config_sha256"],
                         hashlib.sha256(b'{"model_type":"qwen2_5_vl"}').hexdigest())

    def test_hash_bound_receipt_rejects_forgery(self):
        with self.assertRaisesRegex(ValueError, "historical_pilot_run_manifest_mismatch"):
            runner.validate_pilot_run_receipt({"samples_sha256": runner.PILOT_MANIFEST_SHA256,
                                               "n_requested": 30, "n_effective": 29,
                                               "n_failures": 0})

    def test_run_manifest_digest_mismatch_rejected_before_receipt_fields(self):
        with self.assertRaisesRegex(ValueError, "historical_pilot_receipt_hash_mismatch"):
            runner.validate_pilot_run_manifest_bytes(b'{"n_requested":30}')

    def test_inventory_join_derives_denylist_without_embedded_media_hashes(self):
        inventory, provenance = inventory_fixture()
        result = runner.validate_v6_inventory_metadata(inventory, provenance)
        self.assertEqual(len(result), 3)
        self.assertNotIn("V6_MEDIA_SHA256", vars(runner))
        provenance["inventory_path"] = "wrong"
        with self.assertRaisesRegex(ValueError, "v6_source_provenance_binding_invalid"):
            runner.validate_v6_inventory_metadata(inventory, provenance)

    def test_inventory_requires_three_distinct_sha256_roles(self):
        inventory, provenance = inventory_fixture()
        inventory["artifacts"][2]["sha256"] = inventory["artifacts"][1]["sha256"]
        with self.assertRaisesRegex(ValueError, "v6_review_media_exclusion_set_mismatch"):
            runner.validate_v6_inventory_metadata(inventory, provenance)


class PreflightAndScoringTests(unittest.TestCase):
    def test_safe_failure_detail_keeps_only_bounded_runtime_or_fixed_value_codes(self):
        self.assertEqual(runner.safe_failure_detail(
            ValueError("historical_pilot_gsub_interval_missing")),
            "historical_pilot_gsub_interval_missing")
        self.assertIsNone(runner.safe_failure_detail(
            ValueError("question text or a filesystem path")))
        self.assertEqual(runner.safe_failure_detail(
            RuntimeError("per_forward_free_vram_guard_failed")),
            "per_forward_free_vram_guard_failed")
        self.assertIsNone(runner.safe_failure_detail(OSError("private path")))

    def test_failure_receipt_binds_selected_evidence_policy_without_estimate(self):
        counts = {name: {"requested": 30, "attempted": 0, "completed": 0}
                  for name in runner.BRANCH_NAMES}
        receipt = runner.failure_receipt(
            requested_n=30, stage="full_cpu_preflight", error_type="ValueError",
            preflight_attempted=9, preflight_completed=8,
            branch_counts=counts, failure_detail="annotation_anchor_control_unavailable",
            evidence_selection_policy="single_overlap_anchor")
        self.assertEqual(receipt["evidence_selection_policy"], "single_overlap_anchor")
        self.assertEqual((receipt["requested_n"], receipt["effective_n"]), (30, 0))
        self.assertIsNone(receipt["finite_cohort_delta_f"])

    def test_checkpoint_audit_requires_pinned_revision_and_all_five_verified_shards(self):
        shards = [{"filename": f"model-{index:05d}-of-00005.safetensors",
                   "observed_size_bytes": 100 + index, "observed_sha256": f"{index:064x}",
                   "sha256_matches": True, "stable_during_read": True,
                   "failure_code": None} for index in range(1, 6)]
        report = {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                  "revision": runner.MODEL_REVISION,
                  "byte_authentication_status": "verified_current_local_bytes",
                  "requested_shards": 5, "effective_shards": 5,
                  "failure_count": 0, "failure_codes": [],
                  "api_metadata_hash_matches_pinned_artifact": True,
                  "shards": shards}
        summary = runner.validate_checkpoint_audit(report)
        self.assertEqual(summary["effective_shards"], 5)
        report["shards"][-1]["sha256_matches"] = False
        with self.assertRaisesRegex(ValueError, "pinned_checkpoint_shard_audit_incomplete"):
            runner.validate_checkpoint_audit(report)

    def test_baseline_accuracy_has_explicit_finite_denominator(self):
        rows = [{"baseline_full_correct": value} for value in [1, 0] * 15]
        summary = runner.summarize_baseline_accuracy(rows, 30, smoke=False)
        self.assertEqual((summary["n"], summary["correct"], summary["accuracy"]),
                         (30, 15, .5))
        self.assertIsNone(runner.summarize_baseline_accuracy(
            [{"baseline_full_correct": 1}], 1, smoke=True))
        with self.assertRaisesRegex(ValueError, "baseline_full_accuracy_denominator_invalid"):
            runner.summarize_baseline_accuracy(rows[:29], 30, smoke=False)

    def test_exact_denominator_and_equal_disjoint_controls(self):
        plans = [{"geometry_sha256": "a" * 64, "evidence_count": 2,
                  "control_count": 2, "group_count": 10} for _ in range(30)]
        self.assertTrue(runner.validate_preflight_plans(plans, 30))
        plans[0]["control_count"] = 1
        with self.assertRaisesRegex(ValueError, "preflight_effective_denominator_or_control_invalid"):
            runner.validate_preflight_plans(plans, 30)
        with self.assertRaises(ValueError):
            runner.validate_preflight_plans(plans[:29], 29)

    def test_six_branch_contract_and_same_input_fingerprint(self):
        self.assertEqual(len(runner.BRANCH_NAMES), 6)
        hashes = {name: "a" * 64 for name in runner.BRANCH_NAMES}
        self.assertEqual(runner.validate_shared_batch_fingerprints(hashes), "a" * 64)
        hashes[runner.BRANCH_NAMES[-1]] = "b" * 64
        with self.assertRaisesRegex(ValueError, "six_branch_original_batch_binding_mismatch"):
            runner.validate_shared_batch_fingerprints(hashes)

    def test_six_condition_scorer_freezes_p_and_q_from_baseline_full(self):
        result = runner.score_six_calls(
            baseline_full=[-.1, -.2, -.3, -.4, -.5],
            baseline_evidence=[-.4, -.2, -.3, -.4, -.5],
            baseline_control=[-.15, -.2, -.3, -.4, -.5],
            qta_full=[-.1, -.2, -.3, -.4, -.5],
            qta_evidence=[-.5, -.2, -.3, -.4, -.5],
            qta_control=[-.1, -.2, -.3, -.4, -.5])
        self.assertEqual((result["answer_index"], result["foil_index"]), (0, 1))
        self.assertAlmostEqual(result["baseline_selective_dependence"], .25)
        self.assertAlmostEqual(result["intervention_selective_dependence"], .4)
        self.assertAlmostEqual(result["delta_f"], .15)

    def test_failure_receipt_has_no_estimate_and_preserves_branch_denominators(self):
        counts = {name: {"requested": 30, "attempted": 2, "completed": 1}
                  for name in runner.BRANCH_NAMES}
        receipt = runner.failure_receipt(requested_n=30, stage="six_branch_score",
                                         error_type="ValueError", preflight_attempted=30,
                                         preflight_completed=30, branch_counts=counts,
                                         failure_detail="synthetic bounded diagnostic")
        self.assertEqual(receipt["effective_n"], 0)
        self.assertIsNone(receipt["finite_cohort_delta_f"])
        self.assertEqual({name: row["requested"] for name, row in receipt["branch_counts"].items()},
                         {name: 30 for name in runner.BRANCH_NAMES})
        self.assertTrue(all(row["effective"] == 0 for row in receipt["branch_counts"].values()))
        self.assertEqual(receipt["execution_authorization"], "local_only_user_authorized")
        self.assertEqual(receipt["failure_detail"], "synthetic bounded diagnostic")
        long_receipt = runner.failure_receipt(
            requested_n=30, stage="six_branch_score", error_type="RuntimeError",
            preflight_attempted=30, preflight_completed=30, branch_counts=counts,
            failure_detail="x" * 300)
        self.assertEqual(len(long_receipt["failure_detail"]), 240)

    def test_summary_is_descriptive_only(self):
        summary = runner.summarize([1.0, -1.0])
        self.assertEqual(summary["mean"], 0)
        self.assertFalse({"p_value", "ci", "confidence_interval"} & set(summary))


class PrivateOutputTests(unittest.TestCase):
    def test_writes_only_under_mode_700_private_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / "data").mkdir()
            private = repo / "data/private_nextgqa"
            private.mkdir(mode=0o700)
            private.chmod(0o700)
            runner.write_private_output(repo, "result.json", {"status": "synthetic"})
            self.assertEqual((private / "result.json").stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                runner.write_private_output(repo, "result.json", {"status": "overwrite"})
            self.assertEqual(json.loads((private / "result.json").read_text())["status"],
                             "synthetic")

    def test_rejects_symlinked_private_root_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / "data").mkdir()
            external = repo / "external"
            external.mkdir(mode=0o700)
            (repo / "data/private_nextgqa").symlink_to(external, target_is_directory=True)
            with self.assertRaises(OSError):
                runner.write_private_output(repo, "x.json", {})
            with tempfile.TemporaryDirectory() as second:
                with self.assertRaisesRegex(ValueError, "private_output_filename_invalid"):
                    runner.write_private_output(repo, "../escape", {})


if __name__ == "__main__":
    unittest.main()
