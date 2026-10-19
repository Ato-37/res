from __future__ import annotations

import hashlib
import json
import unittest
from unittest import mock

import torch

import qta_prebaseline_transaction_preflight as transaction
from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256
from qta_phase_freeze_receipts import build_external_cohort_pin_candidate
from test_qta_phase_freeze_receipts import QtaPhaseFreezeReceiptTests
from test_qta_stage1_source_pin_adapter import QtaStage1SourcePinAdapterTests


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class QtaPrebaselineTransactionPreflightTests(unittest.TestCase):
    def setUp(self):
        self.source_fixture = QtaStage1SourcePinAdapterTests()
        self.source_fixture.setUp()
        self.addCleanup(self.source_fixture.doCleanups)
        self.stage1_pins = self.source_fixture.invoke()
        self.actual_source = self.source_fixture.qa_inputs
        self.actual_qa = self.source_fixture.qa_inputs
        stage1 = self.source_fixture.stage1_inputs
        self.review = {
            "worksheet_raw": stage1["worksheet_raw"],
            "lineage_raw": stage1["lineage_raw"],
            "review_raw": stage1["review_raw"],
            "review_index_raw": stage1["review_index_raw"],
            "expected_lineage_sha256": stage1["expected_lineage_sha256"],
        }
        self.phase = QtaPhaseFreezeReceiptTests()
        self.phase.setUp()
        self.addCleanup(self.phase.tearDown)

        self.batches = [
            FrozenBatch({
                "input_ids": torch.tensor([[index + 1, index + 2]], dtype=torch.int64),
                "pixels": torch.arange(8, dtype=torch.float32) + index,
            })
            for index in range(3)
        ]
        self.live_rows = []
        receipt_rows = []
        for row, batch in zip(self.stage1_pins["source_rows"], self.batches, strict=True):
            digest = batch_sha256(batch)
            self.live_rows.append({
                "review_token": row["review_token"],
                "source_video_sha256": row["source_video_sha256"],
                "batch_sha256": digest,
                "batch": batch,
            })
            receipt_rows.append({
                "record_token": hashlib.sha256(
                    row["review_token"].encode("ascii")).hexdigest(),
                "processor_batch_sha256": digest,
            })

        self.stage1_raw = canonical(self.stage1_pins)
        self.artifacts = dict(self.phase.p1_artifacts)
        self.artifacts["stage1_review_and_lineage"] = self.stage1_raw
        external_pin = build_external_cohort_pin_candidate(self.stage1_raw)
        self.artifacts["external_cohort_pin"] = external_pin
        external_pin_sha = hashlib.sha256(external_pin).hexdigest()
        receipt_raw = self.phase._create("pre_baseline", receipt_rows, self.artifacts)
        self.kwargs = {
            "actual_source_inputs": self.actual_source,
            "review_bundle_inputs": self.review,
            "stage1_audit_inputs": self.source_fixture.stage1_inputs,
            "actual_qa_source_inputs": self.actual_qa,
            "phase1_receipt_raw": receipt_raw,
            "phase1_artifact_bytes": self.artifacts,
            "integrity_roster_raw": self.phase.integrity,
            "release_roster_raw": self.phase.release,
            "root_public_key_pem": self.phase.keys["root"][1],
            "expected_root_public_der_sha256": self.phase.root_pin,
            "expected_external_cohort_pin_sha256": external_pin_sha,
            "live_processor_rows": self.live_rows,
        }

    def test_signed_and_source_review_gates_bind_the_same_live_batch_objects(self):
        with (mock.patch.object(transaction, "verify_and_bind_signed_phase1_batches",
                                wraps=transaction.verify_and_bind_signed_phase1_batches) as signed,
              mock.patch.object(transaction, "diagnose_qta_prebaseline",
                                wraps=transaction.diagnose_qta_prebaseline) as diagnostic):
            report = transaction.preflight_qta_phase1_batch_transaction(**self.kwargs)
        self.assertEqual(report["status"], "structural_bindings_consistent_unvalidated")
        self.assertEqual(report["structurally_bound_rows"], 3)
        self.assertTrue(report["same_live_batch_objects_rechecked"])
        self.assertFalse(report["external_trust_root_origin_authenticated"])
        self.assertFalse(report["human_identity_authenticated"])
        self.assertFalse(report["semantic_review_authenticated"])
        self.assertFalse(report["clock_frame_correspondence_authenticated"])
        self.assertFalse(report["model_load_permitted"])
        self.assertFalse(report["scoring_authorized"])
        self.assertEqual(report["authorized_forwards"], 0)
        self.assertFalse(report["faithfulness_scored"])
        signed_rows = signed.call_args.kwargs["live_rows"]
        diagnostic_rows = diagnostic.call_args.kwargs["live_processor_rows"]
        self.assertIs(signed_rows, diagnostic_rows)
        self.assertEqual([id(row["batch"]) for row in signed_rows],
                         [id(batch) for batch in self.batches])
        serialized = json.dumps(report)
        for row in self.live_rows:
            self.assertNotIn(row["review_token"], serialized)
            self.assertNotIn(row["batch_sha256"], serialized)

    def test_reordered_live_rows_fail_closed(self):
        kwargs = dict(self.kwargs)
        kwargs["live_processor_rows"] = list(reversed(self.live_rows))
        report = transaction.preflight_qta_phase1_batch_transaction(**kwargs)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["structurally_bound_rows"], 0)
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["authorized_forwards"], 0)

    def test_tensor_change_after_receipt_freeze_fails_closed(self):
        self.live_rows[1]["batch"]["pixels"][0] += 1
        report = transaction.preflight_qta_phase1_batch_transaction(**self.kwargs)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["same_live_batch_objects_rechecked"])
        self.assertFalse(report["scoring_authorized"])

    def test_stage1_bytes_must_match_the_derived_source_review_cohort(self):
        kwargs = dict(self.kwargs)
        artifacts = dict(self.artifacts)
        changed = json.loads(self.stage1_raw)
        changed["source_rows"][0]["claim_sha256"] = "f" * 64
        changed_raw = canonical(changed)
        artifacts["stage1_review_and_lineage"] = changed_raw
        artifacts["external_cohort_pin"] = build_external_cohort_pin_candidate(changed_raw)
        kwargs["phase1_artifact_bytes"] = artifacts
        kwargs["expected_external_cohort_pin_sha256"] = hashlib.sha256(
            artifacts["external_cohort_pin"]).hexdigest()
        report = transaction.preflight_qta_phase1_batch_transaction(**kwargs)
        self.assertEqual(report["failure_code"],
                         "phase1_stage1_bytes_do_not_match_derived_pins")
        self.assertFalse(report["model_load_permitted"])

    def test_tampered_signed_artifact_fails_before_structural_success(self):
        kwargs = dict(self.kwargs)
        artifacts = dict(self.artifacts)
        artifacts["full_video_processor_batches"] += b"tampered"
        kwargs["phase1_artifact_bytes"] = artifacts
        report = transaction.preflight_qta_phase1_batch_transaction(**kwargs)
        self.assertEqual(report["failure_code"],
                         "signed_phase1_batch_binding_incomplete")
        self.assertFalse(report["scoring_authorized"])
        self.assertEqual(report["authorized_forwards"], 0)

    def test_qa_and_diagnostic_source_maps_must_be_the_same_bytes(self):
        kwargs = dict(self.kwargs)
        changed = dict(self.actual_qa)
        changed["csv_raw"] += b" "
        kwargs["actual_qa_source_inputs"] = changed
        report = transaction.preflight_qta_phase1_batch_transaction(**kwargs)
        self.assertEqual(report["failure_code"], "actual_qa_source_byte_maps_differ")
        self.assertFalse(report["model_load_permitted"])

    def test_review_bytes_must_match_the_stage1_audit_snapshot(self):
        kwargs = dict(self.kwargs)
        changed = dict(self.review)
        changed["review_raw"] += b" "
        kwargs["review_bundle_inputs"] = changed
        report = transaction.preflight_qta_phase1_batch_transaction(**kwargs)
        self.assertEqual(report["failure_code"], "stage1_review_byte_maps_differ")
        self.assertFalse(report["scoring_authorized"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
