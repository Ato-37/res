from __future__ import annotations

import hashlib
import json
import unittest

from qta_phase_freeze_receipts import ReceiptError, build_external_cohort_pin_candidate
from test_qta_phase_freeze_receipts import QtaPhaseFreezeReceiptTests
from test_qta_stage1_source_pin_adapter import QtaStage1SourcePinAdapterTests


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

    def _phase1_artifacts(self):
        artifacts = dict(self.phase.p1_artifacts)
        artifacts["stage1_review_and_lineage"] = self.stage1_raw
        artifacts["external_cohort_pin"] = self.external_pin_raw
        return artifacts

    def test_actual_adapter_output_composes_with_non_authorizing_phase1_receipt(self):
        pins = json.loads(self.stage1_raw)
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


if __name__ == "__main__":
    unittest.main()
