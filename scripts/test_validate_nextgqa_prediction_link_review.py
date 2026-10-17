from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from nextgqa_prediction_link_review import displayed_fields_sha256
from validate_nextgqa_prediction_link_review import validate_private_files


class PrivatePredictionLinkCliTests(unittest.TestCase):
    def setUp(self):
        self.protocol_hash = "b" * 64
        self.video_review_hash = "c" * 64
        self.interval_hash = "d" * 64
        self.row = {
            "review_token": "opaque-1", "question": "Synthetic question?",
            "options": ["A", "B", "C", "D", "E"],
            "claim": "Synthetic claim.",
            "option_log_probabilities": [-0.1, -0.2, -1, -2, -3],
            "p_index": 0, "q_index": 1,
            "video_support_decision": "supports_claim",
            "reviewed_intervals_sha256": self.interval_hash,
        }
        self.baseline_raw = json.dumps(
            {"schema_version": 1,
             "purpose": "qta_baseline_prediction_link_manifest",
             "video_claim_review_receipt_sha256": self.video_review_hash,
             "records": [self.row]}, sort_keys=True,
            separators=(",", ":")).encode()
        self.baseline_hash = hashlib.sha256(self.baseline_raw).hexdigest()
        self.cohort_raw = json.dumps(
            {"schema_version": 1,
             "purpose": "qta_prediction_link_frozen_cohort_sources",
             "status": "frozen_pre_score",
             "protocol_sha256": self.protocol_hash,
             "baseline_manifest_sha256": self.baseline_hash,
             "video_claim_review_receipt_sha256": self.video_review_hash,
             "records": [{
                 "review_token": self.row["review_token"],
                 "question_sha256": hashlib.sha256(self.row["question"].encode()).hexdigest(),
                 "options_sha256": hashlib.sha256(json.dumps(
                     self.row["options"], ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode()).hexdigest(),
                 "claim_sha256": hashlib.sha256(self.row["claim"].encode()).hexdigest(),
                 "video_support_decision": "supports_claim",
                 "reviewed_intervals_sha256": self.interval_hash,
             }], "execution_authorization": "not_granted"}, sort_keys=True,
            separators=(",", ":")).encode()
        self.cohort_hash = hashlib.sha256(self.cohort_raw).hexdigest()
        self.receipt_raw = json.dumps({
            "schema_version": 1, "purpose": "qta_prediction_link_adjudication",
            "status": "prediction_link_review_recorded_unvalidated",
            "baseline_manifest_sha256": self.baseline_hash,
            "protocol_sha256": self.protocol_hash,
            "prior_video_review_receipt_sha256": self.video_review_hash,
            "reviewer_role": "independent_prediction_link_reviewer",
            "reviewer_private_id": "synthetic-reviewer",
            "independence_attestation": "independent_from_generation_scoring_custody_and_video_review",
            "gold_and_branch_outcomes_hidden": True,
            "reviewed_at_utc": "2026-10-18T10:00:00Z",
            "records": [{
                "review_token": self.row["review_token"],
                "displayed_fields_sha256": displayed_fields_sha256(
                    self.row, video_claim_review_receipt_sha256=self.video_review_hash),
                "judgment": "supports_p_over_q", "rationale": "Synthetic rationale.",
            }], "execution_authorization": "not_granted",
        }, sort_keys=True, separators=(",", ":")).encode()

    def _run(self, **overrides):
        temp = tempfile.TemporaryDirectory()
        try:
            root = Path(temp.name) / "private"
            root.mkdir(mode=0o700)
            os.chmod(root, 0o700)
            for name, raw in (("baseline.json", self.baseline_raw),
                              ("cohort.json", self.cohort_raw),
                              ("receipt.json", self.receipt_raw)):
                path = root / name
                path.write_bytes(raw)
                os.chmod(path, 0o600)
            args = {
                "private_root": root, "baseline_name": "baseline.json",
                "receipt_name": "receipt.json", "cohort_source_name": "cohort.json",
                "output_name": "validation.json",
                "expected_baseline_sha256": self.baseline_hash,
                "expected_cohort_source_sha256": self.cohort_hash,
                "expected_protocol_sha256": self.protocol_hash,
            }
            args.update(overrides)
            return validate_private_files(**args), root, temp
        except Exception:
            temp.cleanup()
            raise

    def test_valid_input_writes_private_summary_without_authority(self):
        report, root, temp = self._run()
        try:
            self.assertEqual(report["status"], "prediction_link_review_structurally_valid_unvalidated")
            self.assertEqual(report["support_judgment_count"], 1)
            self.assertFalse(report["other_score_gates_verified"])
            self.assertEqual(report["execution_authorization"], "not_granted")
            output = root / "validation.json"
            stored = json.loads(output.read_text())
            self.assertEqual(stored, report)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        finally:
            temp.cleanup()

    def test_changed_source_digest_or_cohort_pin_fails_before_output(self):
        with self.assertRaisesRegex(ValueError, "cohort_source_digest_mismatch"):
            self._run(expected_cohort_source_sha256="c" * 64)
        with self.assertRaisesRegex(ValueError, "prediction_link_manifest_binding_mismatch"):
            changed = "d" * 64
            raw = json.loads(self.receipt_raw)
            raw["baseline_manifest_sha256"] = changed
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "private"
                root.mkdir(mode=0o700)
                os.chmod(root, 0o700)
                for name, data in (("baseline.json", self.baseline_raw),
                                   ("cohort.json", self.cohort_raw),
                                   ("receipt.json", json.dumps(raw).encode())):
                    path = root / name
                    path.write_bytes(data)
                    os.chmod(path, 0o600)
                validate_private_files(
                    private_root=root, baseline_name="baseline.json",
                    receipt_name="receipt.json", cohort_source_name="cohort.json",
                    output_name="validation.json",
                    expected_baseline_sha256=self.baseline_hash,
                    expected_cohort_source_sha256=self.cohort_hash,
                    expected_protocol_sha256=self.protocol_hash)

    def test_private_names_and_exclusive_output_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "private_filename_invalid"):
            self._run(baseline_name="../baseline.json")
        report, root, temp = self._run()
        try:
            self.assertTrue(report["all_expected_rows_have_supporting_self_reported_judgment"])
            with self.assertRaisesRegex(ValueError, "output_already_exists"):
                validate_private_files(
                    private_root=root, baseline_name="baseline.json",
                    receipt_name="receipt.json", cohort_source_name="cohort.json",
                    output_name="validation.json",
                    expected_baseline_sha256=self.baseline_hash,
                    expected_cohort_source_sha256=self.cohort_hash,
                    expected_protocol_sha256=self.protocol_hash)
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
