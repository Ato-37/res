from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import audit_claim_mapping_manifest as gate

H = "a" * 64
VIDEO = "private-video-id"
INTERVALS = [[1.0, 2.0]]


def make_row(**changes):
    row = {
        "video_id": VIDEO,
        "claim_id": "claim-1",
        "claim_text_sha256": H,
        "claim_origin": "model_generated_frozen",
        "model_revision": "org/model@" + "b" * 40,
        "generation_run_sha256": H,
        "mapping_method": "independent_claim_review",
        "qa_seed": {"usage": "seed_only", "intervals": [[1.0, 2.0]]},
        "evidence_intervals": INTERVALS,
        "review": {
            "status": "completed",
            "reviewer_role": "independent_human",
            "review_record_sha256": H,
            "reviewed_at_utc": "2026-10-15T00:00:00Z",
            "claim_text_sha256_seen": H,
            "evidence_intervals_sha256_seen": gate._intervals_sha256(INTERVALS),
            "video_reviewed": True,
            "answer_and_gold_hidden": True,
            "outcome_scores_seen": False,
            "claim_support_decision": "supports_claim",
            "occurrence_policy_applied": "one_sufficient_occurrence",
        },
    }
    row.update(changes)
    return row


def make_ledger(eligible_raw: bytes, rows=None, **changes):
    ledger = {
        "schema_version": 1,
        "eligible_dataset_id": "synthetic-dataset",
        "eligible_manifest_sha256": hashlib.sha256(eligible_raw).hexdigest(),
        "claim_policy_sha256": H,
        "occurrence_policy": "one_sufficient_occurrence",
        "scoring_started_at_utc": None,
        "claims": [make_row()] if rows is None else rows,
    }
    ledger.update(changes)
    return ledger


class ClaimMappingManifestAuditTests(unittest.TestCase):
    def setUp(self):
        self.eligible = [{"video_id": VIDEO}]
        self.eligible_raw = json.dumps(self.eligible, separators=(",", ":")).encode()

    def run_audit(self, ledger):
        raw = json.dumps(ledger, separators=(",", ":")).encode()
        return gate.audit(self.eligible, self.eligible_raw, ledger, raw)

    def test_seed_interval_can_equal_reviewed_interval_but_pass_is_only_declarative(self):
        result = self.run_audit(make_ledger(self.eligible_raw))
        self.assertTrue(result["structurally_valid"])
        self.assertEqual(result["semantic_validation"], "not_machine_verifiable")
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_qa_span_copy_is_rejected(self):
        row = make_row(mapping_method="qa_span_copy")
        result = self.run_audit(make_ledger(self.eligible_raw, rows=[row]))
        self.assertFalse(result["structurally_valid"])
        self.assertIn("claim_specific_remapping_missing", result["failure_codes"])

    def test_missing_frozen_generated_claim_is_rejected(self):
        row = make_row(claim_origin="human_or_dataset_text")
        result = self.run_audit(make_ledger(self.eligible_raw, rows=[row]))
        self.assertIn("frozen_model_claim_missing", result["failure_codes"])

    def test_claim_and_interval_must_match_review_record(self):
        row = make_row()
        row["review"]["claim_text_sha256_seen"] = "b" * 64
        row["review"]["evidence_intervals_sha256_seen"] = "b" * 64
        result = self.run_audit(make_ledger(self.eligible_raw, rows=[row]))
        self.assertIn("independent_human_review_incomplete", result["failure_codes"])

    def test_unsupported_claim_fails_full_cohort_gate_instead_of_dropping_item(self):
        row = make_row()
        row["review"]["claim_support_decision"] = "contradicts_claim"
        result = self.run_audit(make_ledger(self.eligible_raw, rows=[row]))
        self.assertFalse(result["structurally_valid"])
        self.assertIn("claim_support_not_established", result["failure_codes"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_non_support_decision_is_reported_even_if_review_is_incomplete(self):
        row = make_row()
        row["review"]["claim_support_decision"] = "contradicts_claim"
        row["review"].pop("reviewed_at_utc")
        result = self.run_audit(make_ledger(self.eligible_raw, rows=[row]))
        self.assertIn("independent_human_review_incomplete", result["failure_codes"])
        self.assertIn("claim_support_not_established", result["failure_codes"])

    def test_post_scoring_review_is_rejected(self):
        ledger = make_ledger(self.eligible_raw, scoring_started_at_utc="2026-10-15T00:00:00Z")
        result = self.run_audit(ledger)
        self.assertIn("review_not_before_scoring", result["failure_codes"])

    def test_repeated_occurrence_policy_requires_each_interval(self):
        ledger = make_ledger(self.eligible_raw, occurrence_policy="all_relevant_occurrences")
        ledger["claims"][0]["review"]["occurrence_policy_applied"] = "all_relevant_occurrences"
        ledger["claims"][0]["evidence_intervals"] = [[1.0, 2.0], [5.0, 6.0]]
        ledger["claims"][0]["review"]["evidence_intervals_sha256_seen"] = gate._intervals_sha256(
            ledger["claims"][0]["evidence_intervals"]
        )
        result = self.run_audit(ledger)
        self.assertTrue(result["structurally_valid"])
        ledger["claims"][0]["evidence_intervals"] = []
        result = self.run_audit(ledger)
        self.assertIn("claim_evidence_intervals_invalid", result["failure_codes"])

    def test_unknown_occurrence_policy_is_rejected(self):
        result = self.run_audit(make_ledger(self.eligible_raw, occurrence_policy="after_results"))
        self.assertIn("occurrence_policy_missing_or_invalid", result["failure_codes"])

    def test_nonfinite_or_reversed_intervals_fail_without_raising(self):
        for intervals in ([[float("nan"), 2.0]], [[3.0, 2.0]], [[1.0, 1.0]],
                          [[0, 10**1000]], [[2**53, 2**53 + 1]]):
            with self.subTest(intervals=intervals):
                row = make_row(evidence_intervals=intervals)
                result = self.run_audit(make_ledger(self.eligible_raw, rows=[row]))
                self.assertFalse(result["structurally_valid"])
                self.assertIn("claim_evidence_intervals_invalid", result["failure_codes"])

    def test_nonfinite_values_anywhere_in_input_fail(self):
        eligible = [{"video_id": VIDEO, "unused_metadata": float("inf")}]
        eligible_raw = json.dumps(eligible, allow_nan=True, separators=(",", ":")).encode()
        result = gate.audit(eligible, eligible_raw,
                            make_ledger(eligible_raw), json.dumps(make_ledger(eligible_raw)).encode())
        self.assertIn("nonfinite_json_number", result["failure_codes"])

    def test_json_exponent_overflow_is_rejected_on_read(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "overflow.json"
            path.write_text('[{"video_id":"private-video-id","ignored":1e9999}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                gate._read_json(path)

    def test_whitespace_padded_video_ids_are_rejected(self):
        padded_eligible = [{"video_id": " " + VIDEO}]
        padded_raw = json.dumps(padded_eligible, separators=(",", ":")).encode()
        result = gate.audit(padded_eligible, padded_raw,
                            make_ledger(padded_raw, rows=[make_row(video_id=VIDEO + " ")]), b"{}")
        self.assertIn("eligible_manifest_invalid", result["failure_codes"])
        self.assertIn("claim_row_identifier_invalid", result["failure_codes"])

    def test_parsed_inputs_must_match_their_raw_hash_bytes(self):
        different_raw = b'[{"video_id":"different-id"}]'
        ledger = make_ledger(different_raw)
        ledger_raw = json.dumps(ledger, separators=(",", ":")).encode()
        result = gate.audit(self.eligible, different_raw, ledger, ledger_raw)
        self.assertIn("parsed_input_raw_bytes_mismatch", result["failure_codes"])

    def test_non_bytes_provenance_inputs_fail_without_raising(self):
        ledger = make_ledger(self.eligible_raw)
        result = gate.audit(self.eligible, self.eligible_raw.decode(), ledger, b"{}")
        self.assertIn("raw_input_bytes_invalid", result["failure_codes"])

    def test_unhashable_occurrence_policy_fails_closed(self):
        ledger = make_ledger(self.eligible_raw, occurrence_policy=[])
        result = self.run_audit(ledger)
        self.assertIn("occurrence_policy_missing_or_invalid", result["failure_codes"])

    def test_mutable_model_revision_alias_is_rejected(self):
        row = make_row(model_revision="org/model@latest")
        result = self.run_audit(make_ledger(self.eligible_raw, rows=[row]))
        self.assertIn("frozen_model_claim_missing", result["failure_codes"])

    def test_duplicate_or_missing_video_rows_are_rejected(self):
        row = make_row()
        duplicate = self.run_audit(make_ledger(self.eligible_raw, rows=[row, dict(row)]))
        self.assertIn("duplicate_video_claim_rows", duplicate["failure_codes"])
        missing = self.run_audit(make_ledger(self.eligible_raw, rows=[]))
        self.assertIn("claim_rows_missing", missing["failure_codes"])

    def test_cli_summary_never_echoes_private_identifiers_or_intervals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            eligible_path, ledger_path = root / "eligible.json", root / "ledger.json"
            eligible_path.write_bytes(self.eligible_raw)
            ledger_path.write_text(json.dumps(make_ledger(self.eligible_raw)), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(Path(gate.__file__)), "--eligible-manifest", str(eligible_path),
                 "--claim-ledger", str(ledger_path)], capture_output=True, text=True, check=False,
            )
        self.assertEqual(proc.returncode, 0)
        self.assertNotIn(VIDEO, proc.stdout)
        self.assertNotIn("1.0", proc.stdout)
        summary = json.loads(proc.stdout)
        self.assertEqual(summary["execution_authorization"], "not_granted")


if __name__ == "__main__":
    unittest.main()
