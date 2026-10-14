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
        "schema_version": 1,
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
        self.assertTrue(result["declared_use_gates_structurally_complete"])
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
        self.assertFalse(result["declared_use_gates_structurally_complete"])
        self.assertEqual(result["eligible_rows_with_complete_permitted_declarations"], 0)

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
        self.assertFalse(result["declared_use_gates_structurally_complete"])
        self.assertEqual(result["item_rows_with_complete_permitted_decisions"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_not_permitted_annotation_use_fails_gate(self):
        eligible, eligible_raw, ledger, _ = fixture()
        ledger["annotation_rights"]["decisions"]["aggregate_publication"] = "not_permitted"
        result = audit(eligible, eligible_raw, ledger, raw(ledger))
        self.assertIn("annotation_use_not_permitted", result["failure_codes"])
        self.assertFalse(result["declared_use_gates_structurally_complete"])

    def test_duplicate_eligible_ids_are_invalid(self):
        _, eligible_raw, ledger, _ = fixture()
        eligible = [{"video_id": "private-a"}, {"video_id": "private-a"}]
        result = audit(eligible, raw(eligible), ledger, raw(ledger))
        self.assertIn("eligible_manifest_invalid", result["failure_codes"])

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
        self.assertTrue(summary["declared_use_gates_structurally_complete"])
        self.assertEqual(summary["execution_authorization"], "not_granted")


if __name__ == "__main__":
    unittest.main()
