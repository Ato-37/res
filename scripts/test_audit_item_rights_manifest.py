"""Synthetic tests for privacy-safe item-rights manifest auditing."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_item_rights_manifest import audit, main  # noqa: E402

H = "a" * 64
USES = ["local_inference", "local_retention", "aggregate_publication"]


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def fixture():
    eligible = [{"video_id": "private-a"}, {"video_id": "private-b"}]
    eligible_raw = raw(eligible)
    decisions = {use: "permitted" for use in USES}
    annotation = {
        "dataset_id": "synthetic annotations",
        "license_id": "synthetic annotation terms",
        "evidence_locator": "private://synthetic/annotation-license",
        "evidence_sha256": H,
        "review_record_sha256": H,
        "reviewer_role": "synthetic independent reviewer",
        "review_status": "completed",
        "reviewed_at_utc": "2026-10-14T08:00:00Z",
        "decisions": decisions.copy(),
    }
    items = []
    for video_id, source_id in (("private-a", "source-a"), ("private-b", "source-b")):
        items.append({
            "video_id": video_id,
            "source_dataset_id": "synthetic source collection",
            "source_video_id": source_id,
            "source_lineage": [
                {"from_dataset_id": "synthetic annotations", "from_record_id": video_id,
                 "to_dataset_id": "synthetic video QA", "to_record_id": source_id,
                 "mapping_evidence_locator": "private://synthetic/qa-map",
                 "mapping_evidence_sha256": H},
                {"from_dataset_id": "synthetic video QA", "from_record_id": source_id,
                 "to_dataset_id": "synthetic source collection", "to_record_id": source_id,
                 "mapping_evidence_locator": "private://synthetic/source-map",
                 "mapping_evidence_sha256": H},
            ],
            "media_license_id": "CC-BY-2.0",
            "media_evidence_locator": "private://synthetic/source-license",
            "media_evidence_sha256": H,
            "media_review_record_sha256": H,
            "media_reviewer_role": "synthetic independent reviewer",
            "media_review_status": "completed",
            "media_reviewed_at_utc": "2026-10-14T08:00:00Z",
            "conditions_record_sha256": H,
            "conditions_status": "reviewed_resolved",
            "decisions": decisions.copy(),
        })
    ledger = {
        "schema_version": 2,
        "eligible_dataset_id": "synthetic annotations",
        "eligible_manifest_sha256": hashlib.sha256(eligible_raw).hexdigest(),
        "intended_use_scope": "synthetic noncommercial local inference and aggregate results",
        "intended_uses": USES.copy(),
        "annotation_rights": annotation,
        "items": items,
    }
    return eligible, eligible_raw, ledger, raw(ledger)


class ItemRightsManifestAuditTests(unittest.TestCase):
    def test_complete_synthetic_ledger_reconciles_but_never_authorizes(self):
        result = audit(*fixture())
        self.assertTrue(result["structurally_valid"])
        self.assertTrue(result["declared_item_and_annotation_use_gates_structurally_complete"])
        self.assertEqual(result["dataset_access_agreement_gate"], "not_machine_verified")
        self.assertEqual(result["eligible_video_count"], 2)
        self.assertEqual(result["rights_row_count"], 2)
        self.assertEqual(result["eligible_rows_with_complete_permitted_declarations"], 2)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(result["semantic_validation"], "not_machine_verifiable")

    def test_stale_eligible_manifest_hash_fails(self):
        eligible, eligible_raw, ledger, ledger_raw = fixture()
        ledger["eligible_manifest_sha256"] = H
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertIn("eligible_manifest_hash_mismatch", result["failure_codes"])
        self.assertFalse(result["declared_item_and_annotation_use_gates_structurally_complete"])
        self.assertEqual(result["eligible_rows_with_complete_permitted_declarations"], 0)

    def test_parsed_objects_must_match_hash_bound_raw_bytes(self):
        eligible, eligible_raw, ledger, ledger_raw = fixture()
        mismatched_eligible = [{"video_id": "private-other-a"}, {"video_id": "private-other-b"}]
        result = audit(mismatched_eligible, eligible_raw, ledger, ledger_raw)
        self.assertIn("parsed_input_raw_bytes_mismatch", result["failure_codes"])
        self.assertFalse(result["declared_item_and_annotation_use_gates_structurally_complete"])
        self.assertEqual(result["execution_authorization"], "not_granted")

        mismatched_ledger = json.loads(ledger_raw)
        mismatched_ledger["items"][0]["decisions"]["local_inference"] = "unresolved"
        result = audit(eligible, eligible_raw, mismatched_ledger, ledger_raw)
        self.assertIn("parsed_input_raw_bytes_mismatch", result["failure_codes"])
        self.assertFalse(result["declared_item_and_annotation_use_gates_structurally_complete"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_nonfinite_or_duplicate_key_inputs_are_rejected(self):
        eligible, eligible_raw, ledger, ledger_raw = fixture()
        with_nonfinite = json.loads(eligible_raw)
        with_nonfinite[0]["ignored"] = float("inf")
        result = audit(with_nonfinite, eligible_raw, ledger, ledger_raw)
        self.assertIn("parsed_input_raw_bytes_mismatch", result["failure_codes"])

        duplicate_key_raw = b'{"schema_version":2,"schema_version":2}'
        result = audit(eligible, eligible_raw, ledger, duplicate_key_raw)
        self.assertIn("parsed_input_raw_bytes_mismatch", result["failure_codes"])

        result = audit(eligible, eligible_raw, ledger, ledger_raw.decode())
        self.assertIn("parsed_input_raw_bytes_mismatch", result["failure_codes"])

        eligible_with_non_json_extra = [{"video_id": "private-a", 1: "number-key"}, eligible[1]]
        eligible_with_non_json_raw = b'[{"video_id":"private-a","1":"number-key"},{"video_id":"private-b"}]'
        result = audit(eligible_with_non_json_extra, eligible_with_non_json_raw, ledger, ledger_raw)
        self.assertIn("parsed_input_raw_bytes_mismatch", result["failure_codes"])

        eligible_with_tuple = [("private-a",), {"video_id": "private-b"}]
        tuple_raw = b'[["private-a"],{"video_id":"private-b"}]'
        result = audit(eligible_with_tuple, tuple_raw, ledger, ledger_raw)
        self.assertIn("parsed_input_raw_bytes_mismatch", result["failure_codes"])

        for invalid_raw in (
            b'{"schema_version":2,"x":1e999}',
            b'{"schema_version":2,"schema_version":2}',
            b'{not-json',
            b'\xff',
        ):
            with self.subTest(invalid_raw=invalid_raw):
                result = audit(eligible, eligible_raw, ledger, invalid_raw)
                self.assertIn("parsed_input_raw_bytes_mismatch", result["failure_codes"])
                self.assertFalse(result["declared_item_and_annotation_use_gates_structurally_complete"])
                self.assertEqual(result["execution_authorization"], "not_granted")

    def test_lineage_identifiers_reject_padding_at_each_link(self):
        for field in ("from_dataset_id", "from_record_id", "to_dataset_id", "to_record_id"):
            with self.subTest(field=field):
                eligible, eligible_raw, ledger, _ = fixture()
                ledger["items"][0]["source_lineage"][0][field] += " "
                result = audit(eligible, eligible_raw, ledger, raw(ledger))
                self.assertFalse(result["declared_item_and_annotation_use_gates_structurally_complete"])
                self.assertEqual(result["execution_authorization"], "not_granted")

    def test_identifier_whitespace_is_rejected_instead_of_normalized(self):
        eligible, eligible_raw, ledger, _ = fixture()
        eligible[0]["video_id"] = "private-a "
        padded_eligible_raw = raw(eligible)
        result = audit(eligible, padded_eligible_raw, ledger, raw(ledger))
        self.assertIn("eligible_manifest_invalid", result["failure_codes"])

        eligible, eligible_raw, ledger, _ = fixture()
        ledger["items"][0]["video_id"] = "private-a "
        ledger["items"][0]["source_lineage"][0]["from_record_id"] = "private-a "
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertIn("item_rights_video_id_invalid", result["failure_codes"])
        self.assertFalse(result["declared_item_and_annotation_use_gates_structurally_complete"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_missing_duplicate_and_out_of_cohort_rows_fail_without_echoing_ids(self):
        eligible, eligible_raw, ledger, _ = fixture()
        ledger["items"] = [ledger["items"][0], ledger["items"][0], {
            **ledger["items"][1], "video_id": "private-extra"}]
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertIn("rights_video_ids_duplicate", result["failure_codes"])
        self.assertIn("rights_rows_missing", result["failure_codes"])
        self.assertIn("rights_rows_outside_eligible_cohort", result["failure_codes"])
        self.assertNotIn("private-a", json.dumps(result))
        self.assertNotIn("private-extra", json.dumps(result))

    def test_unresolved_use_and_conditions_keep_gate_closed(self):
        eligible, eligible_raw, ledger, _ = fixture()
        ledger["items"][0]["decisions"]["local_retention"] = "unresolved"
        ledger["items"][1]["conditions_status"] = "unresolved"
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertIn("item_use_unresolved", result["failure_codes"])
        self.assertIn("item_rights_conditions_unresolved", result["failure_codes"])
        self.assertFalse(result["declared_item_and_annotation_use_gates_structurally_complete"])
        self.assertEqual(result["item_rows_with_complete_permitted_decisions"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_not_permitted_annotation_use_fails_gate(self):
        eligible, eligible_raw, ledger, _ = fixture()
        ledger["annotation_rights"]["decisions"]["aggregate_publication"] = "not_permitted"
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertIn("annotation_use_not_permitted", result["failure_codes"])
        self.assertFalse(result["declared_item_and_annotation_use_gates_structurally_complete"])

    def test_duplicate_eligible_ids_are_invalid(self):
        _, eligible_raw, ledger, _ = fixture()
        eligible = [{"video_id": "private-a"}, {"video_id": "private-a"}]
        result = audit(eligible, raw(eligible), ledger, raw(ledger))
        self.assertIn("eligible_manifest_invalid", result["failure_codes"])

    def test_source_lineage_must_be_connected_and_match_endpoints(self):
        eligible, eligible_raw, ledger, _ = fixture()
        ledger["items"][0]["source_lineage"][1]["from_record_id"] = "wrong-middle-record"
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertIn("item_source_lineage_disconnected", result["failure_codes"])
        self.assertIn("item_rights_source_lineage_invalid", result["failure_codes"])

        eligible, eligible_raw, ledger, _ = fixture()
        ledger["items"][0]["source_lineage"][0]["from_record_id"] = "other-video"
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertIn("item_source_lineage_endpoint_mismatch", result["failure_codes"])

    def test_source_lineage_requires_mapping_evidence_per_transition(self):
        eligible, eligible_raw, ledger, _ = fixture()
        ledger["items"][0]["source_lineage"][1]["mapping_evidence_sha256"] = "invalid"
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertIn("item_source_lineage_evidence_invalid", result["failure_codes"])

    def test_lineage_allows_upstream_reuse_of_source_video_identifier(self):
        eligible, eligible_raw, ledger, _ = fixture()
        item = ledger["items"][0]
        item["source_dataset_id"] = "YFCC100M"
        item["source_video_id"] = "yfcc-id-123"
        item["source_lineage"] = [
            {"from_dataset_id": "synthetic annotations", "from_record_id": item["video_id"],
             "to_dataset_id": "NExT-QA", "to_record_id": "nextqa-key",
             "mapping_evidence_locator": "private://synthetic/edge-1", "mapping_evidence_sha256": H},
            {"from_dataset_id": "NExT-QA", "from_record_id": "nextqa-key",
             "to_dataset_id": "VidOR", "to_record_id": "yfcc-id-123",
             "mapping_evidence_locator": "private://synthetic/edge-2", "mapping_evidence_sha256": H},
            {"from_dataset_id": "VidOR", "from_record_id": "yfcc-id-123",
             "to_dataset_id": "YFCC100M", "to_record_id": "yfcc-id-123",
             "mapping_evidence_locator": "private://synthetic/edge-3", "mapping_evidence_sha256": H},
        ]
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertTrue(result["structurally_valid"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_empty_or_missing_source_lineage_keeps_gate_closed(self):
        eligible, eligible_raw, ledger, _ = fixture()
        ledger["items"][0]["source_lineage"] = []
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertIn("item_source_lineage_missing", result["failure_codes"])

    def test_cli_output_contains_no_private_ids_or_evidence_locators(self):
        _, eligible_raw, ledger, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            eligible_path = Path(directory) / "eligible.json"
            ledger_path = Path(directory) / "rights.json"
            eligible_path.write_bytes(eligible_raw)
            ledger_path.write_bytes(raw(ledger))
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["--eligible-manifest", str(eligible_path),
                                  "--rights-ledger", str(ledger_path)])
        self.assertEqual(exit_code, 0)
        self.assertNotIn("private-a", output.getvalue())
        self.assertNotIn("private://", output.getvalue())
        self.assertNotIn("source-a", output.getvalue())
        summary = json.loads(output.getvalue())
        self.assertTrue(summary["declared_item_and_annotation_use_gates_structurally_complete"])
        self.assertEqual(summary["dataset_access_agreement_gate"], "not_machine_verified")
        self.assertEqual(summary["execution_authorization"], "not_granted")


if __name__ == "__main__":
    unittest.main()
