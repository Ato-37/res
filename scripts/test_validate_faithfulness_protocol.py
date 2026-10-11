"""Adversarial CPU tests for the structural faithfulness protocol validator."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from validate_faithfulness_protocol import main, validate_protocol


H = "a" * 64
H2 = "b" * 64
H3 = "c" * 64
H4 = "d" * 64
H5 = "e" * 64


def valid_protocol() -> dict:
    return {
        "schema_version": 1,
        "protocol_stage": "confirmatory",
        "model_revision": "pinned-model-revision",
        "tokenizer_revision": "pinned-tokenizer-revision",
        "cohort": {
            "primary_unit": "video_id",
            "video_count": 2,
            "record_count": 2,
            "analysis_unit_count": 2,
            "records_aggregated_within_video": False,
            "split_manifest_sha256": H,
            "overlap_audit_sha256": H2,
            "split_overlap_count": 0,
        },
        "annotation": {
            "locator_kind": "interval",
            "source_manifest_sha256": H3,
            "time_unit": "seconds",
            "time_origin": "video_start",
            "evaluation_metric": "interval_iou",
            "semantic_review_record_sha256": H4,
            "point_conversion": None,
        },
        "sampling": {
            "policy": "whole_video_fixed",
            "config_sha256": H5,
            "uses_annotation_for_selection": False,
        },
        "estimand": {
            "primary_estimand": "claim_selective_dependence",
            "primary_score_target": "claim",
            "score_kind": "teacher_forced_logprob",
            "target_text_sha256": H,
            "gold_role": "evaluation_only",
            "claim_intervention_mapping_sha256": H4,
        },
        "intervention": {
            "kind": "internal_activation",
            "selector_sha256": H2,
            "selector_gold_access": False,
            "control_protocol_sha256": H3,
            "control_match_dimensions": ["layer", "token_count", "magnitude"],
        },
        "cache": {
            "policy": "no_cache_full_sequence",
            "use_cache": False,
            "decode_position_policy": None,
            "runtime_evidence_sha256": None,
            "runtime_source_version": None,
        },
    }


class ProtocolValidatorTests(unittest.TestCase):
    def assert_invalid(self, protocol: dict, code: str) -> dict:
        result = validate_protocol(protocol)
        self.assertFalse(result["structural_valid"])
        self.assertIn(code, result["failure_codes"])
        self.assertEqual(result["semantic_validation"], "not_machine_verifiable")
        self.assertEqual(result["execution_authorization"], "not_granted")
        return result

    def test_valid_minimal_claim_protocol_is_structural_only(self):
        result = validate_protocol(valid_protocol(), input_sha256=H5)
        self.assertTrue(result["structural_valid"])
        self.assertEqual(result["failure_codes"], [])
        self.assertEqual(result["protocol_sha256"], H5)
        self.assertEqual(result["semantic_validation"], "not_machine_verifiable")
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_unknown_top_level_and_nested_keys_fail_closed(self):
        protocol = valid_protocol()
        protocol["human_reviewed"] = True
        self.assert_invalid(protocol, "protocol_schema_keys_invalid")

        protocol = valid_protocol()
        protocol["annotation"]["semantic_match"] = True
        self.assert_invalid(protocol, "annotation_schema_invalid")

    def test_boolean_schema_version_is_not_integer_one(self):
        protocol = valid_protocol()
        protocol["schema_version"] = True
        self.assert_invalid(protocol, "schema_version_invalid")

    def test_video_is_analysis_unit_and_repeated_records_need_aggregation(self):
        protocol = valid_protocol()
        protocol["cohort"]["primary_unit"] = "qid"
        self.assert_invalid(protocol, "primary_unit_must_be_video")

        protocol = valid_protocol()
        protocol["cohort"].update(record_count=4, analysis_unit_count=4)
        self.assert_invalid(protocol, "analysis_units_not_video_level")

        protocol = valid_protocol()
        protocol["cohort"].update(record_count=4, records_aggregated_within_video=True)
        self.assertTrue(validate_protocol(protocol)["structural_valid"])

    def test_point_cannot_be_silently_scored_as_interval(self):
        protocol = valid_protocol()
        protocol["annotation"].update(
            locator_kind="point", evaluation_metric="interval_iou", point_conversion=None,
            semantic_review_record_sha256=None,
        )
        self.assert_invalid(protocol, "point_used_as_interval_without_review_refs")

    def test_point_conversion_requires_explicit_artifact_hash_references(self):
        protocol = valid_protocol()
        protocol["annotation"].update(
            locator_kind="point",
            evaluation_metric="interval_iou",
            point_conversion={
                "rule_sha256": H,
                "derived_interval_artifact_sha256": H2,
                "review_record_sha256": H3,
            },
        )
        result = validate_protocol(protocol)
        self.assertTrue(result["structural_valid"])
        self.assertEqual(result["semantic_validation"], "not_machine_verifiable")
        self.assertEqual(result["execution_authorization"], "not_granted")

        protocol["annotation"]["point_conversion"]["review_record_sha256"] = "bad"
        self.assert_invalid(protocol, "point_used_as_interval_without_review_refs")

    def test_annotation_conditioned_sampling_cannot_be_confirmatory(self):
        protocol = valid_protocol()
        protocol["sampling"].update(policy="annotation_conditioned", uses_annotation_for_selection=True)
        self.assert_invalid(protocol, "confirmatory_sampling_uses_annotation")

    def test_conditioned_sampling_must_be_disclosed(self):
        protocol = valid_protocol()
        protocol["sampling"].update(policy="annotation_conditioned", uses_annotation_for_selection=False)
        self.assert_invalid(protocol, "conditioned_sampling_not_declared")

    def test_answer_score_cannot_be_labeled_claim_faithfulness(self):
        protocol = valid_protocol()
        protocol["estimand"].update(primary_score_target="answer")
        self.assert_invalid(protocol, "score_target_estimand_mismatch")

        protocol = valid_protocol()
        protocol["estimand"].update(
            primary_estimand="qa_accuracy", primary_score_target="answer", score_kind="accuracy",
            claim_intervention_mapping_sha256=None,
        )
        result = validate_protocol(protocol)
        self.assertTrue(result["structural_valid"])
        self.assertEqual(result["semantic_validation"], "not_machine_verifiable")

    def test_selector_cannot_access_gold_and_control_matching_is_recorded(self):
        protocol = valid_protocol()
        protocol["intervention"]["selector_gold_access"] = True
        self.assert_invalid(protocol, "selector_uses_gold")

        protocol = valid_protocol()
        protocol["intervention"]["control_match_dimensions"] = ["temporal_position"]
        self.assert_invalid(protocol, "claim_control_matching_incomplete")

    def test_cache_mode_must_match_explicit_runtime_policy(self):
        protocol = valid_protocol()
        protocol["cache"].update(
            policy="cached_prefill_decode",
            use_cache=True,
            decode_position_policy="explicit_native",
            runtime_evidence_sha256=None,
            runtime_source_version=None,
        )
        self.assert_invalid(protocol, "cached_runtime_evidence_missing")

        protocol["cache"].update(runtime_evidence_sha256=H4, runtime_source_version="transformers-x")
        result = validate_protocol(protocol)
        self.assertTrue(result["structural_valid"])
        self.assertEqual(result["semantic_validation"], "not_machine_verifiable")

        protocol["cache"]["decode_position_policy"] = None
        self.assert_invalid(protocol, "cached_decode_position_policy_missing")

    def test_no_cache_mode_rejects_cache_claims(self):
        protocol = valid_protocol()
        protocol["cache"]["use_cache"] = True
        self.assert_invalid(protocol, "no_cache_policy_inconsistent")

    def test_cli_output_contains_no_private_path_or_protocol_text(self):
        protocol = valid_protocol()
        protocol["model_revision"] = "PRIVATE-MODEL-NAME"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "private-protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([str(path)])
        self.assertEqual(code, 0)
        emitted = stdout.getvalue()
        self.assertNotIn(str(path), emitted)
        self.assertNotIn("PRIVATE-MODEL-NAME", emitted)
        payload = json.loads(emitted)
        self.assertTrue(payload["structural_valid"])
        self.assertEqual(payload["execution_authorization"], "not_granted")

    def test_cli_rejects_duplicate_json_keys_without_echoing_input(self):
        raw = '{"schema_version":1,"schema_version":1,"private_marker":"DO_NOT_ECHO"}'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate-private.json"
            path.write_text(raw, encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([str(path)])
        self.assertEqual(code, 2)
        emitted = stdout.getvalue()
        self.assertNotIn(str(path), emitted)
        self.assertNotIn("DO_NOT_ECHO", emitted)
        payload = json.loads(emitted)
        self.assertFalse(payload["structural_valid"])
        self.assertEqual(payload["failure_codes"], ["protocol_unreadable_or_invalid_json"])
        self.assertEqual(payload["execution_authorization"], "not_granted")

    def test_cli_rejects_nonfinite_json_without_echoing_input(self):
        raw = '{"protocol_stage":"private_marker","score":NaN}'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nonfinite-private.json"
            path.write_text(raw, encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([str(path)])
        self.assertEqual(code, 2)
        emitted = stdout.getvalue()
        self.assertNotIn(str(path), emitted)
        self.assertNotIn("private_marker", emitted)
        payload = json.loads(emitted)
        self.assertFalse(payload["structural_valid"])
        self.assertEqual(payload["failure_codes"], ["protocol_unreadable_or_invalid_json"])
        self.assertEqual(payload["semantic_validation"], "not_machine_verifiable")

    def test_hash_validation_rejects_boolean_counts_and_bad_hashes(self):
        protocol = valid_protocol()
        protocol["cohort"]["video_count"] = True
        self.assert_invalid(protocol, "cohort_counts_invalid")

        protocol = valid_protocol()
        protocol["sampling"]["config_sha256"] = "not-a-hash"
        self.assert_invalid(protocol, "sampling_config_hash_invalid")

    def test_arbitrary_json_enum_values_fail_without_raising(self):
        mutations = (
            ("protocol_stage", []),
            ("annotation.locator_kind", {}),
            ("annotation.time_origin", []),
            ("annotation.evaluation_metric", []),
            ("sampling.policy", {}),
            ("estimand.primary_estimand", []),
            ("estimand.primary_score_target", {}),
            ("estimand.score_kind", []),
            ("intervention.kind", {}),
            ("cache.policy", []),
            ("cache.decode_position_policy", {}),
        )
        for dotted, replacement in mutations:
            with self.subTest(field=dotted):
                protocol = valid_protocol()
                target = protocol
                parts = dotted.split(".")
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = replacement
                result = validate_protocol(protocol)
                self.assertFalse(result["structural_valid"])
                self.assertTrue(result["failure_codes"])
                self.assertEqual(result["semantic_validation"], "not_machine_verifiable")


if __name__ == "__main__":
    unittest.main()
