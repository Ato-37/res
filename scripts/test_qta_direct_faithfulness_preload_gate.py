from __future__ import annotations

import hashlib
import unittest
from unittest import mock

import qta_actual_candidate_input_bridge as bridge
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
        self.assertEqual(result["frozen_baseline_prediction_link"], "blocked")
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
            "schema_version", "purpose", "status", "requested_rows", "validated_rows",
            "structurally_composed_rows",
            "requested_clock_frame_anchors", "validated_clock_frame_anchors",
            "structural_source_composition", "trusted_human_review", "semantic_review",
            "clock_frame_correspondence", "frozen_baseline_prediction_link",
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


if __name__ == "__main__":
    unittest.main()
