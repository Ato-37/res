from __future__ import annotations

import hashlib
import json
import unittest

from qta_phase_freeze_receipts import ReceiptError, build_external_cohort_pin_candidate
from qta_phase2_plan_manifest import _phase1_stage1_source_pins
from qta_prescore_mapping_binding import interval_list_sha256
from qta_stage1_source_pin_adapter import Stage1SourcePinError
from test_qta_phase_freeze_receipts import QtaPhaseFreezeReceiptTests
from test_qta_stage1_source_pin_adapter import QtaStage1SourcePinAdapterTests
from nextgqa_completed_review_attestation import create_attestation
import test_audit_qta_stage1_bundle_cli as stage1_cli_test


class Stage1Phase1CompositionTests(unittest.TestCase):
    """Exercise the real synthetic stage1 adapter output through phase-1 freeze."""

    def setUp(self):
        self.adapter = QtaStage1SourcePinAdapterTests()
        self.adapter.setUp()
        self.addCleanup(self.adapter.doCleanups)

        self.phase = QtaPhaseFreezeReceiptTests()
        self.phase.setUp()
        self.addCleanup(self.phase.tearDown)

        pins = self.adapter.invoke()
        self.stage1_raw = json.dumps(
            pins, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        self.external_pin_raw = build_external_cohort_pin_candidate(self.stage1_raw)
        # Phase-1 uses the SHA-256 of each opaque Stage-1 review token as its
        # record identity. The receipt fixture's default cohort is deliberately
        # separate, so rebind its synthetic rows to this adapter output before
        # testing composition.
        self.phase.records = [
            {
                "record_token": hashlib.sha256(token.encode("ascii")).hexdigest(),
                "processor_batch_sha256": self.phase.records[index][
                    "processor_batch_sha256"],
            }
            for index, token in enumerate(pins["expected_cohort_tokens"])
        ]

    def _phase1_artifacts(self):
        artifacts = dict(self.phase.p1_artifacts)
        artifacts["stage1_review_and_lineage"] = self.stage1_raw
        artifacts["external_cohort_pin"] = self.external_pin_raw
        return artifacts

    def test_actual_adapter_output_composes_with_non_authorizing_phase1_receipt(self):
        pins = json.loads(self.stage1_raw)
        phase2_source = _phase1_stage1_source_pins(self.stage1_raw)
        artifacts = self._phase1_artifacts()
        self.phase.expected_external_cohort_pin_sha256 = hashlib.sha256(
            self.external_pin_raw
        ).hexdigest()
        receipt = self.phase._create("pre_baseline", self.phase.records, artifacts)

        report = self.phase._verify(receipt, "pre_baseline", artifacts)

        self.assertEqual(pins["expected_cohort_tokens"], [
            row["review_token"] for row in pins["source_rows"]
        ])
        self.assertEqual(len(pins["source_rows"]), 3)
        self.assertEqual(phase2_source["records"], pins["source_rows"])
        self.assertEqual(
            phase2_source["video_claim_review_receipt_sha256"],
            pins["expected_video_claim_review_receipt_sha256"],
        )
        self.assertTrue(report["receipt_signature_valid"])
        self.assertTrue(report["artifact_bytes_match"])
        self.assertTrue(report["external_cohort_pin_matches_caller_pin"])
        self.assertFalse(report["external_cohort_pin_origin_authenticated"])
        self.assertFalse(report["semantic_review_authenticated"])
        self.assertFalse(report["model_load_permitted"])
        self.assertFalse(report["scoring_authorized"])
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertEqual(report["authorized_forwards"], 0)

    def test_external_pin_cannot_be_reused_after_source_row_mutation(self):
        artifacts = self._phase1_artifacts()
        changed = json.loads(self.stage1_raw)
        changed["source_rows"][0]["claim_sha256"] = "0" * 64
        artifacts["stage1_review_and_lineage"] = json.dumps(
            changed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

        with self.assertRaisesRegex(
            ReceiptError, "external_cohort_pin_cohort_binding_mismatch"
        ):
            self.phase._create("pre_baseline", self.phase.records, artifacts)

    def test_rendered_review_export_composes_through_source_pins_into_phase1(self):
        # Execute the existing real rendered-page export/index handlers against
        # this same synthetic Stage-1 fixture; no response bytes are fabricated
        # or translated in this composition test.
        helper = stage1_cli_test.Stage1BundleCliTests()
        helper.fixture = self.adapter.stage1_fixture
        review_raw, review_index_raw = helper._render_synthetic_browser_export()
        review = json.loads(review_raw)
        review_index = json.loads(review_index_raw)
        stage1_inputs = dict(self.adapter.stage1_inputs)
        stage1_inputs.update(review_raw=review_raw,
                             review_index_raw=review_index_raw)
        attestation_raw = create_attestation(
            worksheet=self.adapter.stage1_fixture.worksheet,
            worksheet_raw=self.adapter.stage1_fixture.worksheet_raw,
            lineage=self.adapter.stage1_fixture.lineage,
            lineage_raw=self.adapter.stage1_fixture.lineage_raw,
            review=review, review_raw=review_raw,
            review_index=review_index, review_index_raw=review_index_raw,
            expected_lineage_sha256=self.adapter.stage1_fixture.expected_lineage_sha256,
            pre_review_receipt_sha256=hashlib.sha256(
                self.adapter.stage1_fixture.receipt_raw).hexdigest(),
            integrity_roster_raw=self.adapter.stage1_fixture.f.integrity_roster_raw,
            root_public_key_pem=self.adapter.stage1_fixture.f.keys["root"][1],
            expected_root_public_der_sha256=self.adapter.stage1_fixture.f.root_pin,
            reviewer_private_key_pem=self.adapter.stage1_fixture.f.keys["claim"][0],
            reviewer_public_key_pem=self.adapter.stage1_fixture.f.keys["claim"][1],
        )
        stage1_inputs["reviewer_attestation_raw"] = attestation_raw

        pins = self.adapter.invoke(stage1=stage1_inputs)
        self.assertEqual(pins["status"], "three_row_source_pins_structurally_derived")
        self.assertEqual(len(pins["source_rows"]), 3)
        review_sha = hashlib.sha256(review_raw).hexdigest()
        review_index_sha = hashlib.sha256(review_index_raw).hexdigest()
        self.assertEqual(pins["expected_review_payload_sha256"], review_sha)
        self.assertEqual(pins["stage1_source_bytes_sha256"]["review"], review_sha)
        self.assertEqual(pins["stage1_source_bytes_sha256"]["review_index"],
                         review_index_sha)
        self.assertEqual(pins["expected_cohort_tokens"], [
            row["review_token"] for row in review["records"]])
        for row, exported in zip(pins["source_rows"], review["records"], strict=True):
            intervals = exported["review"]["evidence_intervals_seconds"]
            self.assertEqual(row["reviewed_intervals_sha256"],
                             interval_list_sha256(intervals))
            self.assertEqual(row["video_support_decision"],
                             exported["review"]["support_decision"])
        self.assertEqual(pins["execution_authorization"], "not_granted")
        self.assertFalse(pins["scoring_authorized"])

        self.stage1_raw = json.dumps(
            pins, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        self.external_pin_raw = build_external_cohort_pin_candidate(self.stage1_raw)
        self.phase.records = [
            {
                "record_token": hashlib.sha256(token.encode("ascii")).hexdigest(),
                "processor_batch_sha256": self.phase.records[index][
                    "processor_batch_sha256"],
            }
            for index, token in enumerate(pins["expected_cohort_tokens"])
        ]
        artifacts = self._phase1_artifacts()
        self.phase.expected_external_cohort_pin_sha256 = hashlib.sha256(
            self.external_pin_raw).hexdigest()
        receipt = self.phase._create("pre_baseline", self.phase.records, artifacts)
        report = self.phase._verify(receipt, "pre_baseline", artifacts)
        phase2_source = _phase1_stage1_source_pins(self.stage1_raw)
        self.assertEqual(phase2_source["records"], pins["source_rows"])
        self.assertTrue(report["receipt_signature_valid"])
        self.assertTrue(report["artifact_bytes_match"])
        self.assertFalse(report["scoring_authorized"])
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["authorized_forwards"], 0)
        self.assertEqual(report["execution_authorization"], "not_granted")

        tampered = dict(stage1_inputs)
        tampered["review_raw"] = review_raw + b" "
        with self.assertRaisesRegex(Stage1SourcePinError,
                                    "stage1_audit_not_complete"):
            self.adapter.invoke(stage1=tampered)


if __name__ == "__main__":
    unittest.main()
