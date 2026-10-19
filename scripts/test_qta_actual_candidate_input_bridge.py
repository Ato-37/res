from __future__ import annotations

import hashlib
import json
import unittest
from unittest import mock

import qta_actual_candidate_input_bridge as bridge
from qta_prescore_mapping_binding import interval_list_sha256
from test_qta_stage1_source_pin_adapter import QtaStage1SourcePinAdapterTests
from test_qta_actual_processor_row_candidate import (
    FakeProcessor, FakeTokenizer, _fixtures as processor_fixtures,
)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canon(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


class QtaActualCandidateInputBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tokens = ["A" * 23 + str(i) for i in range(1, 4)]
        self.sources = [sha(f"source-{i}".encode()) for i in range(3)]
        self.receipt = {
            "schema_version": 1,
            "purpose": "local_qta_real_video_sampler_eligibility_preflight",
            "status": "passed", "execution_authorization": "not_granted",
            "gpu_used": False, "model_loaded": False,
            "faithfulness_scored": False, "requested_records": 3,
            "effective_records": 3, "frames_per_video": 32,
            "rows": [],
        }
        self.sampler_rows = []
        for i, source in enumerate(self.sources):
            row = {
                "record_index": i + 1, "source_sha256": source,
                "decoded_frame_count": 32, "sampled_frame_count": 32,
                "sampled_indices": list(range(32)),
                "sampled_pts": [[j, 1] for j in range(32)],
                "effective_sample_fps": [31, 31],
                "sampled_rgb_sha256": [sha(f"{i}-{j}".encode()) for j in range(32)],
            }
            self.sampler_rows.append(row)
        self.receipt["rows"] = self.sampler_rows
        self.sampler_raw = canon(self.receipt)
        self.pins = []
        for token, source in zip(self.tokens, self.sources, strict=True):
            self.pins.append({
                "review_token": token, "source_video_sha256": source,
                "question_sha256": sha(f"q-{token}".encode()),
                "options_sha256": sha(f"o-{token}".encode()),
                "claim_sha256": sha(f"c-{token}".encode()),
                "reviewed_intervals_sha256": interval_list_sha256([[1, 2]]),
                "video_support_decision": "supports_claim",
            })
        self.stage1_pins = {
            "status": "three_row_source_pins_structurally_derived",
            "effective_records": 3,
            "source_rows": self.pins,
            "expected_video_claim_review_receipt_sha256": sha(b"review-receipt"),
            "reviewer_identity_verified": False,
            "semantic_review_authenticated": False,
        }
        self.clock_inputs = {token: {"calibration": {}, "capture_raw": b"capture"}
                             for token in self.tokens}
        self.processor = {token: {"processor_revision": "local-revision",
                                  "processor_fingerprint_sha256": sha(b"processor")}
                          for token in self.tokens}
        self.durations = {token: "32" for token in self.tokens}
        self.source_inputs = {
            "csv_raw": b"csv", "lineage_manifest_raw": b"lineage",
            "qta_protocol_raw": b"protocol", "qta_config_raw": b"config",
        }
        self.qa_inputs = dict(self.source_inputs)
        self.stage1_inputs = {
            "worksheet_raw": b"worksheet", "lineage_raw": b"lineage",
            "review_raw": b"review", "review_index_raw": b"review-index",
            "expected_lineage_sha256": sha(b"lineage"),
        }

    def invoke(self, **overrides):
        args = {
            "sampler_receipt_raw": self.sampler_raw,
            "candidate_inventory_raw": b"inventory",
            "media_sha256_by_role": {},
            "actual_source_inputs": self.source_inputs,
            "stage1_audit_inputs": self.stage1_inputs,
            "actual_qa_source_inputs": self.qa_inputs,
            "clock_inputs_by_token": self.clock_inputs,
            "processor_pins_by_token": self.processor,
            "tokenizer_sha256": sha(b"tokenizer"),
            "video_duration_seconds_by_token": self.durations,
        }
        args.update(overrides)
        with (mock.patch.object(bridge, "preflight_actual_media_source_join", return_value={
                    "status": "source_video_hash_joined_unvalidated", "joined_rows": 3,
                    "human_identity_verified": False, "semantic_review_authenticated": False}),
              mock.patch.object(bridge, "derive_stage1_source_pins", return_value=self.stage1_pins),
              mock.patch.object(bridge, "_cross_bind_stage1_to_actual_source"),
              mock.patch.object(bridge, "adapt_review_bundle_to_qta_rows", return_value=[
                    {"review_token": token, "support_decision": "supports_claim",
                     "evidence_intervals_seconds": [[1, 2]]} for token in self.tokens]),
              mock.patch.object(bridge, "adapt_calibration_to_clock_row", return_value={
                    "guard_pts_seconds": "1/10",
                    "clock_correspondence_authenticated": False})):
            return bridge.build_qta_actual_candidate_input_material(**args)

    def test_builds_ordered_local_candidate_inputs_and_denies_authority(self):
        result = self.invoke()
        self.assertEqual(result["status"], "three_row_candidate_inputs_derived_unvalidated")
        self.assertEqual(result["ordered_review_tokens"], self.tokens)
        self.assertEqual(len(result["expected_rows"]), 3)
        self.assertEqual(result["expected_rows"][0]["sampler_row_sha256"],
                         sha(canon(self.sampler_rows[0])))
        self.assertEqual(result["expected_rows"][0]["interval_boundary_guard_seconds"], "1/10")
        self.assertEqual(result["sampler_rows"], self.sampler_rows)
        self.assertFalse(result["human_identity_verified"])
        self.assertFalse(result["semantic_review_authenticated"])
        self.assertFalse(result["clock_frame_correspondence_verified"])
        self.assertFalse(result["model_load_permitted"])
        self.assertFalse(result["scoring_authorized"])
        self.assertFalse(result["faithfulness_scored"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertTrue(result["human_review_artifacts_present"])
        self.assertTrue(result["clock_calibration_artifacts_present"])

    def test_missing_clock_inputs_refuse_before_any_validator_call(self):
        with mock.patch.object(bridge, "preflight_actual_media_source_join") as preflight:
            with self.assertRaisesRegex(ValueError, "bridge_three_clock_inputs_required"):
                bridge.build_qta_actual_candidate_input_material(
                    sampler_receipt_raw=self.sampler_raw,
                    candidate_inventory_raw=b"inventory", media_sha256_by_role={},
                    actual_source_inputs=self.source_inputs, stage1_audit_inputs=self.stage1_inputs,
                    actual_qa_source_inputs=self.qa_inputs, clock_inputs_by_token={},
                    processor_pins_by_token=self.processor, tokenizer_sha256=sha(b"t"),
                    video_duration_seconds_by_token=self.durations)
            preflight.assert_not_called()

    def test_missing_review_artifact_or_decision_fails_closed(self):
        pins = {**self.stage1_pins, "source_rows": [dict(row) for row in self.pins]}
        pins["source_rows"][1]["video_support_decision"] = "needs_review"
        with mock.patch.object(bridge, "preflight_actual_media_source_join", return_value={
                    "status": "source_video_hash_joined_unvalidated", "joined_rows": 3,
                    "human_identity_verified": False, "semantic_review_authenticated": False}), \
             mock.patch.object(bridge, "derive_stage1_source_pins", return_value=pins), \
             mock.patch.object(bridge, "_cross_bind_stage1_to_actual_source"), \
             mock.patch.object(bridge, "adapt_review_bundle_to_qta_rows", return_value=[
                    {"review_token": token, "support_decision": "supports_claim",
                     "evidence_intervals_seconds": [[1, 2]]} for token in self.tokens]), \
             mock.patch.object(bridge, "adapt_calibration_to_clock_row", return_value={
                    "guard_pts_seconds": "1/10",
                    "clock_correspondence_authenticated": False}):
            with self.assertRaisesRegex(ValueError, "bridge_human_support_review_missing"):
                bridge.build_qta_actual_candidate_input_material(
                    sampler_receipt_raw=self.sampler_raw, candidate_inventory_raw=b"inventory",
                    media_sha256_by_role={}, actual_source_inputs=self.source_inputs,
                    stage1_audit_inputs=self.stage1_inputs, actual_qa_source_inputs=self.qa_inputs,
                    clock_inputs_by_token=self.clock_inputs,
                    processor_pins_by_token=self.processor,
                    tokenizer_sha256=sha(b"tokenizer"),
                    video_duration_seconds_by_token=self.durations)

    def test_source_bytes_and_cohort_order_are_pinned(self):
        with self.assertRaisesRegex(ValueError, "bridge_source_byte_sets_differ"):
            self.invoke(actual_qa_source_inputs={**self.qa_inputs, "csv_raw": b"other"})
        other_clock = dict(self.clock_inputs)
        other_clock.pop(self.tokens[0])
        other_clock["wrong-token"] = {"calibration": {}, "capture_raw": b"capture"}
        with self.assertRaisesRegex(ValueError, "bridge_review_clock_processor_cohort_mismatch"):
            self.invoke(clock_inputs_by_token=other_clock)

    def test_calibration_adapter_cannot_claim_verified_correspondence(self):
        with mock.patch.object(bridge, "preflight_actual_media_source_join", return_value={
                    "status": "source_video_hash_joined_unvalidated", "joined_rows": 3,
                    "human_identity_verified": False, "semantic_review_authenticated": False}), \
             mock.patch.object(bridge, "derive_stage1_source_pins", return_value=self.stage1_pins), \
             mock.patch.object(bridge, "_cross_bind_stage1_to_actual_source"), \
             mock.patch.object(bridge, "adapt_review_bundle_to_qta_rows", return_value=[
                    {"review_token": token, "support_decision": "supports_claim",
                     "evidence_intervals_seconds": [[1, 2]]} for token in self.tokens]), \
             mock.patch.object(bridge, "adapt_calibration_to_clock_row", return_value={
                    "guard_pts_seconds": "0", "clock_correspondence_authenticated": True}):
            with self.assertRaisesRegex(ValueError, "bridge_clock_correspondence_must_remain_unverified"):
                bridge.build_qta_actual_candidate_input_material(
                    sampler_receipt_raw=self.sampler_raw, candidate_inventory_raw=b"inventory",
                    media_sha256_by_role={}, actual_source_inputs=self.source_inputs,
                    stage1_audit_inputs=self.stage1_inputs, actual_qa_source_inputs=self.qa_inputs,
                    clock_inputs_by_token=self.clock_inputs,
                    processor_pins_by_token=self.processor,
                    tokenizer_sha256=sha(b"tokenizer"),
                    video_duration_seconds_by_token=self.durations)

    def test_actual_qa_claim_bridge_rejects_text_splice(self):
        fixture = QtaStage1SourcePinAdapterTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        stage1_pins = fixture.invoke()
        qa = fixture.qa_inputs
        claims = json.loads(qa["claim_manifest_raw"])["records"]
        claim_doc = {
            "schema_version": 1,
            "eligible_dataset_id": "synthetic-nextgqa",
            "eligible_manifest_sha256": "a" * 64,
            "claim_policy_sha256": "b" * 64,
            "occurrence_policy": "all_relevant_occurrences",
            "claims": [{**row, "model_revision": "synthetic-model@" + "c" * 40,
                        "generation_run_sha256": "d" * 64} for row in claims],
        }
        worksheet = json.loads(fixture.stage1_fixture.worksheet_raw)
        target = json.loads(qa["target_manifest_raw"])
        source = {
            "csv_raw": qa["csv_raw"],
            "claim_input_manifest_raw": canon(claim_doc),
            "worksheet_raw": canon({"records": [
                {key: row[key] for key in ("video_id", "claim_id", "review_token")}
                for row in worksheet["records"]]}),
            "target_candidate_raw": canon({
                "purpose": "private_nextgqa_diagnostic_target_candidate_unvalidated",
                "status": "diagnostic_candidate_unvalidated",
                "requested_records": 3, "effective_records": 3,
                "records": target["records"],
            }),
            "lineage_manifest_raw": qa["lineage_manifest_raw"],
            "qta_protocol_raw": qa["qta_protocol_raw"],
            "qta_config_raw": qa["qta_config_raw"],
        }
        bridge._cross_bind_stage1_to_actual_source(
            source_inputs=source, stage1_pins=stage1_pins)
        changed_claims = json.loads(source["claim_input_manifest_raw"])
        changed_claims["claims"][0]["claim_text"] += " splice"
        source["claim_input_manifest_raw"] = canon(changed_claims)
        with self.assertRaisesRegex(ValueError, "bridge_stage1_actual_qa_claim_mismatch"):
            bridge._cross_bind_stage1_to_actual_source(
                source_inputs=source, stage1_pins=stage1_pins)
        changed_token_source = dict(source)
        changed_worksheet = json.loads(changed_token_source["worksheet_raw"])
        changed_worksheet["records"][0]["review_token"] = "Z" * 24
        changed_token_source["worksheet_raw"] = canon(changed_worksheet)
        with self.assertRaisesRegex(ValueError, "bridge_stage1_actual_qa_claim_mismatch"):
            bridge._cross_bind_stage1_to_actual_source(
                source_inputs=changed_token_source, stage1_pins=stage1_pins)

    def test_passing_binder_then_stage1_question_pin_mismatch_is_rejected(self):
        fixture = QtaStage1SourcePinAdapterTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        stage1_pins = fixture.invoke()
        qa = fixture.qa_inputs
        claims = json.loads(qa["claim_manifest_raw"])["records"]
        claim_doc = {
            "schema_version": 1, "eligible_dataset_id": "synthetic-nextgqa",
            "eligible_manifest_sha256": "a" * 64, "claim_policy_sha256": "b" * 64,
            "occurrence_policy": "all_relevant_occurrences",
            "claims": [{**row, "model_revision": "synthetic-model@" + "c" * 40,
                        "generation_run_sha256": "d" * 64} for row in claims],
        }
        worksheet = json.loads(fixture.stage1_fixture.worksheet_raw)
        target = json.loads(qa["target_manifest_raw"])
        source = {
            "csv_raw": qa["csv_raw"],
            "claim_input_manifest_raw": canon(claim_doc),
            "worksheet_raw": canon({"records": [
                {key: row[key] for key in ("video_id", "claim_id", "review_token")}
                for row in worksheet["records"]]}),
            "target_candidate_raw": canon({
                "purpose": "private_nextgqa_diagnostic_target_candidate_unvalidated",
                "status": "diagnostic_candidate_unvalidated",
                "requested_records": 3, "effective_records": 3,
                "records": target["records"],
            }),
            "lineage_manifest_raw": qa["lineage_manifest_raw"],
            "qta_protocol_raw": qa["qta_protocol_raw"],
            "qta_config_raw": qa["qta_config_raw"],
        }
        altered_pins = {**stage1_pins,
                        "source_rows": [dict(row) for row in stage1_pins["source_rows"]]}
        altered_pins["source_rows"][0]["question_sha256"] = sha(b"different-question")
        real_binder = bridge.bind_qta_actual_qa_source
        binder_statuses = []

        def observe_binder(**kwargs):
            result = real_binder(**kwargs)
            binder_statuses.append(result["summary"]["status"])
            return result

        with mock.patch.object(bridge, "bind_qta_actual_qa_source",
                               side_effect=observe_binder) as binder:
            with self.assertRaisesRegex(ValueError, "bridge_stage1_actual_qa_claim_mismatch"):
                bridge._cross_bind_stage1_to_actual_source(
                    source_inputs=source, stage1_pins=altered_pins)
        binder.assert_called_once()
        self.assertEqual(binder_statuses, ["bound_unvalidated"])

    def test_bridge_output_composes_into_processor_candidate_and_external_verifier(self):
        import torch
        from qta_actual_processor_row_candidate import (
            build_actual_processor_row_candidate,
            verify_actual_processor_row_candidate,
        )

        rows, processor_pins, sampler_rows = processor_fixtures()
        tokens = [row["review_token"] for row in rows]
        stage1_rows = [{
            "review_token": expected["review_token"],
            "source_video_sha256": expected["source_video_sha256"],
            "question_sha256": expected["question_sha256"],
            "options_sha256": expected["options_sha256"],
            "claim_sha256": expected["claim_sha256"],
            "reviewed_intervals_sha256": expected["reviewed_intervals_sha256"],
            "video_support_decision": "supports_claim",
        } for expected in processor_pins]
        sampler_raw = canon({
            "purpose": "local_qta_real_video_sampler_eligibility_preflight",
            "status": "passed", "execution_authorization": "not_granted",
            "gpu_used": False, "model_loaded": False, "faithfulness_scored": False,
            "requested_records": 3, "effective_records": 3, "frames_per_video": 32,
            "rows": sampler_rows,
        })
        source_inputs = {
            "csv_raw": b"same-synthetic-csv", "lineage_manifest_raw": b"same-lineage",
            "qta_protocol_raw": b"same-protocol", "qta_config_raw": b"same-config",
        }
        tokenizer_sha = processor_pins[0]["tokenizer_sha256"]
        clock_inputs = {token: {"calibration": {}, "capture_raw": b"fixture-capture"}
                        for token in tokens}
        processor_by_token = {row["review_token"]: {
            "processor_revision": row["processor_revision"],
            "processor_fingerprint_sha256": row["processor_fingerprint_sha256"],
        } for row in rows}
        durations = {row["review_token"]: row["video_duration_seconds"] for row in rows}
        stage1_pins = {
            "status": "three_row_source_pins_structurally_derived",
            "effective_records": 3, "source_rows": stage1_rows,
            "expected_video_claim_review_receipt_sha256": sha(b"synthetic-receipt"),
            "reviewer_identity_verified": False, "semantic_review_authenticated": False,
        }
        review_rows = [{
            "review_token": row["review_token"], "support_decision": "supports_claim",
            "evidence_intervals_seconds": row["reviewed_intervals_seconds"],
        } for row in rows]
        with mock.patch.object(bridge, "preflight_actual_media_source_join", return_value={
                    "status": "source_video_hash_joined_unvalidated", "joined_rows": 3,
                    "human_identity_verified": False, "semantic_review_authenticated": False}), \
             mock.patch.object(bridge, "derive_stage1_source_pins", return_value=stage1_pins), \
             mock.patch.object(bridge, "_cross_bind_stage1_to_actual_source"), \
             mock.patch.object(bridge, "adapt_review_bundle_to_qta_rows", return_value=review_rows), \
             mock.patch.object(bridge, "adapt_calibration_to_clock_row", return_value={
                    "guard_pts_seconds": "0",
                    "clock_correspondence_authenticated": False}):
            material = bridge.build_qta_actual_candidate_input_material(
                sampler_receipt_raw=sampler_raw, candidate_inventory_raw=b"synthetic-inventory",
                media_sha256_by_role={}, actual_source_inputs=source_inputs,
                stage1_audit_inputs={"worksheet_raw": b"x", "lineage_raw": b"x",
                                     "review_raw": b"x", "review_index_raw": b"x",
                                     "expected_lineage_sha256": sha(b"x")},
                actual_qa_source_inputs=dict(source_inputs),
                clock_inputs_by_token=clock_inputs,
                processor_pins_by_token=processor_by_token,
                tokenizer_sha256=tokenizer_sha,
                video_duration_seconds_by_token=durations)
        self.assertEqual(set(material["ordered_review_tokens"]), set(tokens))
        processor = FakeProcessor(FakeTokenizer())
        candidate = build_actual_processor_row_candidate(
            rows=rows, expected_rows=material["expected_rows"],
            sampler_rows=material["sampler_rows"], processor=processor,
            tokenizer=processor.tokenizer, tokenizer_sha256=tokenizer_sha,
            vision_info_fn=lambda _messages, return_video_metadata: (
                None, [(torch.zeros((32, 3)),
                        {"frames_indices": list(range(32)), "fps": 1.0})]),
            torch_module=torch)
        verify_actual_processor_row_candidate(
            candidate, expected_rows=material["expected_rows"])
        self.assertEqual(processor.calls, 3)
        self.assertFalse(material["scoring_authorized"])
        self.assertFalse(material["clock_frame_correspondence_verified"])
        self.assertFalse(candidate["scoring_authorized"])
        self.assertFalse(candidate["faithfulness_scored"])

    def test_interval_splice_is_rejected_after_real_review_row_adapter(self):
        pins = {**self.stage1_pins, "source_rows": [dict(row) for row in self.pins]}
        pins["source_rows"][0]["reviewed_intervals_sha256"] = sha(b"different interval")
        with mock.patch.object(bridge, "preflight_actual_media_source_join", return_value={
                    "status": "source_video_hash_joined_unvalidated", "joined_rows": 3,
                    "human_identity_verified": False, "semantic_review_authenticated": False}), \
             mock.patch.object(bridge, "derive_stage1_source_pins", return_value=pins), \
             mock.patch.object(bridge, "_cross_bind_stage1_to_actual_source"), \
             mock.patch.object(bridge, "adapt_review_bundle_to_qta_rows", return_value=[
                    {"review_token": token, "support_decision": "supports_claim",
                     "evidence_intervals_seconds": [[1, 2]]} for token in self.tokens]), \
             mock.patch.object(bridge, "adapt_calibration_to_clock_row", return_value={
                    "guard_pts_seconds": "1/10",
                    "clock_correspondence_authenticated": False}):
            with self.assertRaisesRegex(ValueError, "bridge_review_interval_pin_mismatch"):
                bridge.build_qta_actual_candidate_input_material(
                    sampler_receipt_raw=self.sampler_raw, candidate_inventory_raw=b"inventory",
                    media_sha256_by_role={}, actual_source_inputs=self.source_inputs,
                    stage1_audit_inputs=self.stage1_inputs, actual_qa_source_inputs=self.qa_inputs,
                    clock_inputs_by_token=self.clock_inputs,
                    processor_pins_by_token=self.processor,
                    tokenizer_sha256=sha(b"tokenizer"),
                    video_duration_seconds_by_token=self.durations)


if __name__ == "__main__":
    unittest.main()
