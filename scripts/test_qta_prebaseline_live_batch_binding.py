import hashlib
import json
import unittest

import torch

from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256
from qta_actual_qa_source_contract import bind_qta_actual_qa_source
from qta_actual_prebaseline_diagnostic import diagnose_qta_prebaseline
from test_qta_actual_prebaseline_diagnostic import combined_inputs
import test_qta_phase_freeze_receipts as phase_test_module
from qta_phase_freeze_receipts import build_external_cohort_pin_candidate
from qta_signed_prebaseline_live_batches import verify_and_bind_signed_phase1_batches


class LiveProcessorBatchBindingTests(unittest.TestCase):
    def setUp(self):
        self.source, self.review, _ = combined_inputs()
        bound = bind_qta_actual_qa_source(**self.source)
        self.crosswalk = json.loads(
            bound["private_bundle"]["raw_json"].decode("utf-8")
        )["review_source_crosswalk"]
        source_lineage = json.loads(self.source["lineage_manifest_raw"])["records"]
        self.batches = [
            FrozenBatch({
                "input_ids": torch.tensor([[index + 1, index + 2]], dtype=torch.int64),
                "pixel_values_videos": torch.arange(12, dtype=torch.float32).reshape(6, 2)
                                      + index,
            })
            for index in range(3)
        ]
        self.expected = []
        self.live = []
        for row, line, batch in zip(self.crosswalk, source_lineage, self.batches, strict=True):
            digest = batch_sha256(batch)
            identity = {"review_token": row["review_token"],
                        "source_video_sha256": line["video_sha256"]}
            self.expected.append({**identity, "batch_sha256": digest})
            self.live.append({**identity, "batch_sha256": digest, "batch": batch})

    def diagnose(self, expected=None, live=None):
        return diagnose_qta_prebaseline(
            actual_source_inputs=self.source,
            review_bundle_inputs=self.review,
            expected_processor_rows=self.expected if expected is None else expected,
            live_processor_rows=self.live if live is None else live,
        )

    def test_exact_live_batches_join_all_three_rows_but_remain_non_authorizing(self):
        report = self.diagnose()
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["validated_rows"], 3)
        self.assertIs(report["model_load_permitted"], False)
        self.assertIs(report["scoring_authorized"], False)
        self.assertIs(report["faithfulness_scored"], False)
        serialized = json.dumps(report, ensure_ascii=False)
        for row in self.expected:
            self.assertNotIn(row["review_token"], serialized)
            self.assertNotIn(row["batch_sha256"], serialized)

    def test_reordered_live_rows_fail_atomically(self):
        report = self.diagnose(live=list(reversed(self.live)))
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["status"], "blocked")
        self.assertEqual(gate["validated_rows"], 0)

    def test_mutated_batch_tensor_fails_live_digest_check(self):
        self.live[1]["batch"]["pixel_values_videos"][0, 0] += 1
        report = self.diagnose()
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["failure_code"], "processor_batch_live_digest_mismatch")
        self.assertEqual(gate["validated_rows"], 0)

    def test_nonfinite_batch_is_rejected_before_hash_acceptance(self):
        self.live[1]["batch"]["pixel_values_videos"][0, 0] = float("inf")
        report = self.diagnose()
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["failure_code"], "processor_batch_nonfinite_tensor")
        self.assertEqual(gate["validated_rows"], 0)

    def test_duplicate_source_token_is_rejected_atomically(self):
        duplicate = [dict(row) for row in self.expected]
        duplicate[2]["review_token"] = duplicate[1]["review_token"]
        report = self.diagnose(expected=duplicate)
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["failure_code"], "processor_batch_duplicate_source_or_token")
        self.assertEqual(gate["validated_rows"], 0)

    def test_upstream_mismatch_blocks_processor_binding(self):
        changed_review = dict(self.review)
        changed_review["expected_lineage_sha256"] = "f" * 64
        report = diagnose_qta_prebaseline(
            actual_source_inputs=self.source,
            review_bundle_inputs=changed_review,
            expected_processor_rows=self.expected,
            live_processor_rows=self.live,
        )
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["failure_code"], "upstream_source_review_join_blocked")
        self.assertEqual(gate["validated_rows"], 0)


class SignedReceiptLiveBatchCompositionTests(unittest.TestCase):
    def setUp(self):
        self.phase = phase_test_module.QtaPhaseFreezeReceiptTests()
        self.phase.setUp()
        self.source, _review, _ = combined_inputs()
        source_result = bind_qta_actual_qa_source(**self.source)
        crosswalk = json.loads(
            source_result["private_bundle"]["raw_json"].decode("utf-8")
        )["review_source_crosswalk"]
        source_lines = json.loads(self.source["lineage_manifest_raw"])["records"]
        self.crosswalk = [
            {"review_token": row["review_token"],
             "video_sha256": line["video_sha256"]}
            for row, line in zip(crosswalk, source_lines, strict=True)
        ]
        self.batches = [
            FrozenBatch({"input_ids": torch.tensor([[i + 1, i + 2]]),
                         "pixels": torch.arange(8, dtype=torch.float32) + i})
            for i in range(3)
        ]
        self.live = []
        for row, batch in zip(self.crosswalk, self.batches, strict=True):
            self.live.append({
                "review_token": row["review_token"],
                "source_video_sha256": row["video_sha256"],
                "batch_sha256": batch_sha256(batch), "batch": batch,
            })

        artifacts = dict(self.phase.p1_artifacts)
        stage1 = json.loads(artifacts["stage1_review_and_lineage"])
        tokens = [row["review_token"] for row in self.crosswalk]
        stage1["expected_cohort_tokens"] = tokens
        for item, crosswalk_row in zip(stage1["source_rows"], self.crosswalk, strict=True):
            token = crosswalk_row["review_token"]
            item["review_token"] = token
            item["source_video_sha256"] = crosswalk_row["video_sha256"]
        artifacts["stage1_review_and_lineage"] = json.dumps(
            stage1, sort_keys=True, separators=(",", ":")).encode("utf-8")
        artifacts["external_cohort_pin"] = build_external_cohort_pin_candidate(
            artifacts["stage1_review_and_lineage"])
        self.expected_external_pin = hashlib.sha256(
            artifacts["external_cohort_pin"]).hexdigest()
        records = [{
            "record_token": hashlib.sha256(row["review_token"].encode("ascii")).hexdigest(),
            "processor_batch_sha256": live["batch_sha256"],
        } for row, live in zip(self.crosswalk, self.live, strict=True)]
        self.artifacts = artifacts
        self.receipt = self.phase._create("pre_baseline", records, artifacts)

    def tearDown(self):
        self.phase.tearDown()

    def call(self, *, receipt=None, artifacts=None, live=None):
        return verify_and_bind_signed_phase1_batches(
            receipt_raw=self.receipt if receipt is None else receipt,
            artifact_bytes=self.artifacts if artifacts is None else artifacts,
            integrity_roster_raw=self.phase.integrity,
            release_roster_raw=self.phase.release,
            root_public_key_pem=self.phase.keys["root"][1],
            expected_root_public_der_sha256=self.phase.root_pin,
            expected_external_cohort_pin_sha256=self.expected_external_pin,
            live_rows=self.live if live is None else live,
        )

    def test_signed_receipt_rows_bind_exact_live_batches_and_stay_non_authorizing(self):
        report = self.call()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["validated_rows"], 3)
        self.assertIs(report["phase_receipt_signature_valid"], True)
        self.assertIs(report["artifact_bytes_match"], True)
        self.assertIs(report["external_cohort_pin_origin_authenticated"], False)
        self.assertIs(report["model_load_permitted"], False)
        self.assertIs(report["scoring_authorized"], False)
        self.assertEqual(report["authorized_forwards"], 0)

    def test_signed_but_stale_row_hash_is_rejected_against_live_tensor(self):
        rows = [{"record_token": hashlib.sha256(
                    item["review_token"].encode("ascii")).hexdigest(),
                 "processor_batch_sha256": "f" * 64}
                for item in self.crosswalk]
        receipt = self.phase._create("pre_baseline", rows, self.artifacts)
        stale_live = [dict(row) for row in self.live]
        for row in stale_live:
            row["batch_sha256"] = "f" * 64
        report = self.call(receipt=receipt, live=stale_live)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["failure_code"], "processor_batch_live_digest_mismatch")
        self.assertEqual(report["validated_rows"], 0)

    def test_live_reorder_or_artifact_mutation_fails_closed(self):
        report = self.call(live=list(reversed(self.live)))
        self.assertEqual(report["failure_code"],
                         "processor_batch_source_order_or_pin_mismatch")
        self.assertEqual(report["validated_rows"], 0)
        changed_artifacts = dict(self.artifacts)
        changed_artifacts["full_video_processor_batches"] += b"tamper"
        report = self.call(artifacts=changed_artifacts)
        self.assertEqual(report["status"], "blocked")
        self.assertIs(report["phase_receipt_signature_valid"], False)
        self.assertIs(report["model_load_permitted"], False)

if __name__ == "__main__":
    unittest.main()
