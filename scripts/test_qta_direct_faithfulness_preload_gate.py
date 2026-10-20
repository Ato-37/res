from __future__ import annotations

import hashlib
import unittest
from unittest import mock

import qta_actual_candidate_input_bridge as bridge
import qta_direct_faithfulness_preload_gate as preload_gate
from qta_direct_faithfulness_preload_gate import (
    evaluate_qta_direct_faithfulness_preload,
)
from test_qta_actual_candidate_input_bridge import QtaActualCandidateInputBridgeTests


class QtaDirectFaithfulnessPreloadGateTests(unittest.TestCase):
    def setUp(self):
        fixture = QtaActualCandidateInputBridgeTests()
        fixture.setUp()
        self.f = fixture

    def call(self, **overrides):
        f = self.f
        kwargs = {
            "sampler_receipt_raw": f.sampler_raw,
            "candidate_inventory_raw": b"synthetic inventory bytes",
            "media_sha256_by_role": {},
            "actual_source_inputs": f.source_inputs,
            "stage1_audit_inputs": f.stage1_inputs,
            "actual_qa_source_inputs": f.qa_inputs,
            "clock_inputs_by_token": f.clock_inputs,
            "processor_pins_by_token": f.processor,
            "tokenizer_sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            "video_duration_seconds_by_token": f.durations,
        }
        kwargs.update(overrides)
        with (mock.patch.object(bridge, "preflight_actual_media_source_join", return_value={
                    "status": "source_video_hash_joined_unvalidated", "joined_rows": 3,
                    "human_identity_verified": False, "semantic_review_authenticated": False})
              as media,
              mock.patch.object(bridge, "derive_stage1_source_pins", return_value=f.stage1_pins)
              as stage1,
              mock.patch.object(bridge, "_cross_bind_stage1_to_actual_source") as qa,
              mock.patch.object(bridge, "adapt_review_bundle_to_qta_rows", return_value=[
                    {"review_token": token, "support_decision": "supports_claim",
                     "evidence_intervals_seconds": [[1, 2]]} for token in f.tokens]) as review,
              mock.patch.object(bridge, "adapt_calibration_to_clock_row", return_value={
                    "guard_pts_seconds": "1/10",
                    "clock_correspondence_authenticated": False}) as clock):
            result = evaluate_qta_direct_faithfulness_preload(**kwargs)
        return result, (media, stage1, qa, review, clock), kwargs

    def test_revalidates_raw_inputs_and_composes_only_structural_status(self):
        result, validators, kwargs = self.call()
        self.assertEqual(result["status"], "structural_inputs_composed_unvalidated")
        self.assertEqual(result["validated_rows"], 0)
        self.assertEqual(result["structurally_composed_rows"], 3)
        self.assertEqual(result["structural_source_composition"], "passed_unvalidated")
        self.assertEqual(result["requested_clock_frame_anchors"], 15)
        self.assertEqual(result["validated_clock_frame_anchors"], 0)
        self.assertEqual(result["trusted_human_review"], "blocked")
        self.assertEqual(result["semantic_review"], "blocked")
        self.assertEqual(result["clock_frame_correspondence"], "blocked")
        self.assertEqual(result["phase"], "pre_baseline_candidate")
        self.assertEqual(result["baseline_prediction_manifest"], "not_applicable_pre_baseline")
        self.assertEqual(result["prediction_link_review"],
                         "not_yet_applicable_baseline_required")
        self.assertEqual(result["processor_batch_provenance"], "blocked")
        self.assertFalse(result["direct_faithfulness_ready"])
        self.assertFalse(result["model_load_permitted"])
        self.assertFalse(result["cuda_used"])
        self.assertFalse(result["model_loaded"])
        self.assertEqual(result["forward_calls"], 0)
        self.assertEqual(result["score_calls"], 0)
        self.assertFalse(result["faithfulness_scored"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(set(result), {
            "schema_version", "purpose", "phase", "status", "requested_rows", "validated_rows",
            "structurally_composed_rows",
            "requested_clock_frame_anchors", "validated_clock_frame_anchors",
            "signature_bound_frame_pair_decisions", "clock_frame_audit_status",
            "clock_frame_trust_root_origin_authenticated",
            "structural_source_composition", "trusted_human_review", "semantic_review",
            "clock_frame_correspondence", "baseline_prediction_manifest",
            "prediction_link_review",
            "processor_batch_provenance", "direct_faithfulness_ready",
            "model_load_permitted", "cuda_used", "model_loaded", "forward_calls",
            "score_calls", "faithfulness_scored", "execution_authorization",
        })
        for validator in validators[:4]:
            validator.assert_called_once()
        self.assertEqual(validators[4].call_count, 3)
        self.assertIs(validators[0].call_args.kwargs["sampler_receipt_raw"],
                      kwargs["sampler_receipt_raw"])
        self.assertIs(validators[1].call_args.kwargs["stage1_audit_inputs"],
                      kwargs["stage1_audit_inputs"])

    def test_missing_or_tampered_raw_gate_input_fails_closed(self):
        f = self.f
        changed = dict(f.qa_inputs)
        changed["csv_raw"] = b"tampered exact source bytes"
        result, validators, _ = self.call(actual_qa_source_inputs=changed)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["validated_rows"], 0)
        self.assertEqual(result["structurally_composed_rows"], 0)
        self.assertEqual(result["structural_source_composition"], "blocked")
        self.assertEqual(result["forward_calls"], 0)
        self.assertEqual(result["score_calls"], 0)
        self.assertFalse(result["model_load_permitted"])
        self.assertFalse(result["direct_faithfulness_ready"])
        # Byte mismatch is rejected before any source validator is called.
        for validator in validators:
            validator.assert_not_called()

        result, _, _ = self.call(clock_inputs_by_token={})
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["validated_rows"], 0)
        self.assertFalse(result["model_loaded"])
        self.assertEqual(result["forward_calls"], 0)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_clock_frame_audit_is_rerun_and_only_counts_untrusted_signed_pairs(self):
        raw_inputs = {"evidence_raw": b"exact raw evidence bytes"}
        report = {
            "status": "candidate_signed_evidence_bytes_consistent",
            "requested_records": 3, "requested_anchors": 15,
            "verified_anchors": 15, "match_decisions": 15,
            "ambiguous_decisions": 0, "mismatch_decisions": 0,
            "screenshot_and_decoder_bytes_checked": True,
            "reviewer_signature_verified": True,
            "frame_reviewer_role_key_verified": True,
            "capture_operator_signature_verified": True,
            "capture_operator_human_identity_verified": False,
            "execution_authorization": "not_granted",
        }
        with mock.patch.object(preload_gate, "audit_clock_frame_evidence",
                               return_value=report) as audit:
            result, _, _ = self.call(clock_frame_audit_inputs=raw_inputs)
        audit.assert_called_once_with(**raw_inputs)
        self.assertEqual(result["signature_bound_frame_pair_decisions"], 15)
        self.assertEqual(result["clock_frame_audit_status"],
                         "signed_pairs_consistent_untrusted_root")
        self.assertFalse(result["clock_frame_trust_root_origin_authenticated"])
        self.assertEqual(result["validated_clock_frame_anchors"], 0)
        self.assertEqual(result["clock_frame_correspondence"], "blocked")
        self.assertFalse(result["model_load_permitted"])
        self.assertEqual(result["forward_calls"], 0)

    def test_caller_supplied_clock_summary_cannot_fill_signed_pair_count(self):
        result, _, _ = self.call(clock_frame_audit_inputs={"report": {
            "status": "candidate_signed_evidence_bytes_consistent",
            "requested_anchors": 15, "verified_anchors": 15,
        }})
        self.assertEqual(result["signature_bound_frame_pair_decisions"], 0)
        self.assertEqual(result["clock_frame_audit_status"], "blocked_or_incomplete")
        self.assertFalse(result["clock_frame_trust_root_origin_authenticated"])
        self.assertFalse(result["model_load_permitted"])

    def test_incomplete_or_legacy_clock_audit_never_counts_decisions(self):
        passing = {
            "status": "candidate_signed_evidence_bytes_consistent",
            "requested_records": 3, "requested_anchors": 15,
            "verified_anchors": 15, "match_decisions": 15,
            "ambiguous_decisions": 0, "mismatch_decisions": 0,
            "screenshot_and_decoder_bytes_checked": True,
            "reviewer_signature_verified": True,
            "frame_reviewer_role_key_verified": True,
            "capture_operator_signature_verified": True,
            "capture_operator_human_identity_verified": False,
            "execution_authorization": "not_granted",
        }
        mutations = (
            ("status", "legacy_v1_unrostered_key_candidate"),
            ("requested_records", 2), ("requested_anchors", 14),
            ("verified_anchors", 14), ("match_decisions", 14),
            ("ambiguous_decisions", 1), ("mismatch_decisions", 1),
            ("screenshot_and_decoder_bytes_checked", False),
            ("reviewer_signature_verified", False),
            ("frame_reviewer_role_key_verified", False),
            ("capture_operator_signature_verified", False),
            ("capture_operator_human_identity_verified", True),
            ("execution_authorization", "granted"),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                altered = dict(passing)
                altered[key] = value
                with mock.patch.object(preload_gate, "audit_clock_frame_evidence",
                                       return_value=altered):
                    result, _, _ = self.call(clock_frame_audit_inputs={
                        "evidence_raw": b"exact raw evidence bytes"})
                self.assertEqual(result["signature_bound_frame_pair_decisions"], 0)
                self.assertEqual(result["clock_frame_audit_status"],
                                 "blocked_or_incomplete")
                self.assertFalse(result["model_load_permitted"])


if __name__ == "__main__":
    unittest.main()
