import copy
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from audit_data_rights_evidence_index import audit, main, _read

H = "a" * 64
USES = ["local_inference", "local_retention", "aggregate_publication"]


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def fixture(applicability="applicable"):
    eligible = [{"video_id": "opaque-test-a"}, {"video_id": "opaque-test-b"}]
    eligible_raw = raw(eligible)
    ledger = {
        "schema_version": 2,
        "eligible_manifest_sha256": hashlib.sha256(eligible_raw).hexdigest(),
        "intended_use_scope": "synthetic study scope",
        "intended_uses": USES,
        "annotation_rights": {
            "evidence_sha256": H, "review_status": "completed",
            "decisions": {use: "permitted" for use in USES},
        },
        "items": [
            {"video_id": row["video_id"], "media_evidence_sha256": "b" * 64,
             "conditions_record_sha256": ("c" if row["video_id"] == "opaque-test-a" else "d") * 64,
             "media_review_status": "completed",
             "conditions_status": "reviewed_resolved",
             "decisions": {use: "permitted" for use in USES}}
            for row in eligible
        ],
    }
    ledger_raw = raw(ledger)
    agreement = {
        "applicability": applicability, "sha256": H,
        "private_locator": "private://synthetic/agreement-evidence",
        "review_record_sha256": H, "reviewer_role": "synthetic reviewer",
        "reviewed_at_utc": "2026-10-15T00:00:00Z",
        "planned_use_scope": "synthetic study scope",
        "decisions": {use: "permitted" for use in USES} if applicability == "applicable" else None,
        "non_applicability_basis_sha256": H if applicability == "not_applicable" else None,
    }
    index = {
        "schema_version": 2,
        "eligible_manifest": {"sha256": hashlib.sha256(eligible_raw).hexdigest(),
                              "private_locator": "private://synthetic/eligible", "eligible_video_count": 2},
        "item_rights_manifest": {"sha256": hashlib.sha256(ledger_raw).hexdigest(),
                                 "private_locator": "private://synthetic/ledger"},
        "annotation_terms": {"sha256": H, "private_locator": "private://synthetic/annotation"},
        "item_media_conditions": {
            "sha256": H, "private_locator": "private://synthetic/media", "covered_video_count": 2,
            "items": [
                {"video_id": row["video_id"], "media_evidence_sha256": "b" * 64,
                 "conditions_record_sha256": ("c" if row["video_id"] == "opaque-test-a" else "d") * 64}
                for row in eligible
            ],
        },
        "dataset_access_agreement": agreement,
    }
    index_raw = raw(index)
    protocol = {
        "schema_version": 13,
        "cohort": {"eligible_manifest_sha256": hashlib.sha256(eligible_raw).hexdigest(),
                   "eligible_video_count": 2},
        "data_rights": {"scope": "synthetic study scope", "status": "verified",
                        "evidence_sha256": hashlib.sha256(index_raw).hexdigest(),
                        "item_rights_manifest_sha256": hashlib.sha256(ledger_raw).hexdigest(),
                        "covered_eligible_video_count": 2, "unresolved_eligible_video_count": 0},
    }
    return protocol, eligible, eligible_raw, ledger, ledger_raw, index, index_raw


def run_audit(parts):
    protocol, eligible, eligible_raw, ledger, ledger_raw, index, index_raw = parts
    return audit(protocol, raw(protocol), eligible, eligible_raw,
                 ledger, ledger_raw, index, index_raw)


class EvidenceIndexAuditTests(unittest.TestCase):
    def test_matching_synthetic_index_passes_but_never_authorizes_execution(self):
        report = run_audit(fixture())
        self.assertEqual(report["structural_status"], "passed")
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertEqual(report["requested_eligible_video_count"], 2)
        self.assertNotIn("private://", json.dumps(report))
        self.assertNotIn("opaque-test", json.dumps(report))

    def test_arbitrary_protocol_hash_does_not_pass_without_matching_index(self):
        parts = list(fixture())
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["data_rights"]["evidence_sha256"] = "0" * 64
        report = run_audit(parts)
        self.assertIn("evidence_index_hash_mismatch", report["failure_codes"])

    def test_index_byte_change_breaks_protocol_hash_binding(self):
        parts = list(fixture())
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["item_media_conditions"]["covered_video_count"] = 1
        parts[6] = raw(parts[5])
        report = run_audit(parts)
        self.assertIn("evidence_index_hash_mismatch", report["failure_codes"])
        self.assertIn("index_media_coverage_incomplete", report["failure_codes"])

    def test_each_media_and_conditions_hash_is_bound_to_its_ledger_video(self):
        parts = list(fixture())
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["item_media_conditions"]["items"].reverse()
        parts[6] = raw(parts[5])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["data_rights"]["evidence_sha256"] = hashlib.sha256(parts[6]).hexdigest()
        self.assertEqual(run_audit(parts)["structural_status"], "passed")
        parts = list(fixture())
        parts[5] = copy.deepcopy(parts[5])
        records = parts[5]["item_media_conditions"]["items"]
        records[0]["media_evidence_sha256"], records[1]["media_evidence_sha256"] = (
            records[1]["media_evidence_sha256"], records[0]["media_evidence_sha256"])
        records[0]["conditions_record_sha256"], records[1]["conditions_record_sha256"] = (
            records[1]["conditions_record_sha256"], records[0]["conditions_record_sha256"])
        parts[6] = raw(parts[5])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["data_rights"]["evidence_sha256"] = hashlib.sha256(parts[6]).hexdigest()
        self.assertIn("index_media_item_evidence_mismatch", run_audit(parts)["failure_codes"])
        parts = list(fixture())
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["item_media_conditions"]["items"][0]["media_evidence_sha256"] = "e" * 64
        parts[6] = raw(parts[5])
        self.assertIn("index_media_item_evidence_mismatch", run_audit(parts)["failure_codes"])
        parts = list(fixture())
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["item_media_conditions"]["items"][0]["conditions_record_sha256"] = "e" * 64
        parts[6] = raw(parts[5])
        report = run_audit(parts)
        self.assertIn("index_media_item_evidence_mismatch", report["failure_codes"])
        parts = list(fixture())
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["item_media_conditions"]["items"].pop()
        parts[6] = raw(parts[5])
        self.assertIn("index_media_item_coverage_mismatch", run_audit(parts)["failure_codes"])

    def test_protocol_and_index_must_bind_exact_ledger_and_cohort(self):
        parts = list(fixture())
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["item_rights_manifest"]["sha256"] = "0" * 64
        parts[6] = raw(parts[5])
        report = run_audit(parts)
        self.assertIn("index_ledger_hash_mismatch", report["failure_codes"])
        parts = list(fixture())
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["cohort"]["eligible_video_count"] = 3
        self.assertIn("protocol_eligible_manifest_mismatch", run_audit(parts)["failure_codes"])

    def test_applicable_agreement_must_clear_every_declared_use(self):
        parts = list(fixture())
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["dataset_access_agreement"]["decisions"]["local_retention"] = "unresolved"
        parts[6] = raw(parts[5])
        report = run_audit(parts)
        self.assertIn("dataset_access_use_not_cleared", report["failure_codes"])

    def test_non_applicability_requires_substantiation(self):
        parts = list(fixture("not_applicable"))
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["dataset_access_agreement"]["non_applicability_basis_sha256"] = None
        parts[6] = raw(parts[5])
        report = run_audit(parts)
        self.assertIn("dataset_access_non_applicability_unsubstantiated", report["failure_codes"])
        parts = list(fixture("not_applicable"))
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["dataset_access_agreement"]["non_applicability_basis_sha256"] = "b" * 64
        parts[6] = raw(parts[5])
        self.assertIn("dataset_access_non_applicability_unsubstantiated", run_audit(parts)["failure_codes"])

    def test_unresolved_rights_or_ledger_decision_blocks(self):
        parts = list(fixture("unresolved"))
        self.assertIn("dataset_access_agreement_unresolved", run_audit(parts)["failure_codes"])
        parts = list(fixture())
        parts[3] = copy.deepcopy(parts[3])
        parts[3]["items"][0]["decisions"]["local_inference"] = "unresolved"
        parts[4] = raw(parts[3])
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["item_rights_manifest"]["sha256"] = hashlib.sha256(parts[4]).hexdigest()
        parts[6] = raw(parts[5])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["data_rights"]["item_rights_manifest_sha256"] = hashlib.sha256(parts[4]).hexdigest()
        report = run_audit(parts)
        self.assertIn("ledger_item_use_not_cleared", report["failure_codes"])

    def test_duplicate_json_keys_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
            _read_from_bytes(b'{"schema_version":1,"schema_version":1}')

    def test_malformed_untrusted_types_fail_without_raising(self):
        parts = list(fixture())
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["schema_version"] = True
        parts[5]["dataset_access_agreement"]["applicability"] = []
        parts[6] = raw(parts[5])
        report = run_audit(parts)
        self.assertEqual(report["structural_status"], "failed")
        self.assertIn("evidence_index_schema_invalid", report["failure_codes"])
        self.assertIn("dataset_access_agreement_entry_invalid", report["failure_codes"])

    def test_boolean_coverage_count_is_not_an_integer_declaration(self):
        parts = list(fixture())
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["data_rights"]["covered_eligible_video_count"] = True
        self.assertIn("protocol_rights_coverage_incomplete", run_audit(parts)["failure_codes"])

    def test_wrong_nested_json_types_fail_closed(self):
        parts = list(fixture())
        parts[3] = copy.deepcopy(parts[3])
        parts[3]["annotation_rights"] = []
        parts[4] = raw(parts[3])
        parts[5] = copy.deepcopy(parts[5])
        parts[5]["item_rights_manifest"]["sha256"] = hashlib.sha256(parts[4]).hexdigest()
        parts[6] = raw(parts[5])
        parts[0] = copy.deepcopy(parts[0])
        parts[0]["data_rights"]["item_rights_manifest_sha256"] = hashlib.sha256(parts[4]).hexdigest()
        report = run_audit(parts)
        self.assertEqual(report["structural_status"], "failed")
        self.assertIn("index_annotation_hash_mismatch", report["failure_codes"])

    def test_parsed_objects_must_match_their_hashed_raw_bytes(self):
        parts = list(fixture())
        parts[1] = copy.deepcopy(parts[1])
        parts[1][0]["video_id"] = "different-a"
        parts[1][1]["video_id"] = "different-b"
        parts[3] = copy.deepcopy(parts[3])
        parts[3]["items"][0]["video_id"] = "different-a"
        parts[3]["items"][1]["video_id"] = "different-b"
        report = run_audit(parts)
        self.assertEqual(report["structural_status"], "failed")
        self.assertIn("eligible_manifest_byte_binding_invalid", report["failure_codes"])
        self.assertIn("item_rights_ledger_byte_binding_invalid", report["failure_codes"])

    def test_non_byte_hash_inputs_fail_closed(self):
        parts = list(fixture())
        report = audit(parts[0], raw(parts[0]), parts[1], parts[2], parts[3],
                       None, parts[5], parts[6])
        self.assertIn("item_rights_ledger_byte_binding_invalid", report["failure_codes"])
        self.assertIn("protocol_ledger_hash_mismatch", report["failure_codes"])

    def test_cli_end_to_end_reports_counts_without_private_values(self):
        protocol, eligible, eligible_raw, ledger, ledger_raw, index, index_raw = fixture()
        contents = {
            "protocol.json": raw(protocol),
            "eligible.json": eligible_raw,
            "ledger.json": ledger_raw,
            "index.json": index_raw,
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name, data in contents.items():
                path = Path(tmp) / name
                path.write_bytes(data)
                paths[name] = str(path)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([
                    "--protocol", paths["protocol.json"],
                    "--eligible-manifest", paths["eligible.json"],
                    "--item-rights-ledger", paths["ledger.json"],
                    "--evidence-index", paths["index.json"],
                ])
        self.assertEqual(code, 0)
        self.assertIn("status=passed", stdout.getvalue())
        self.assertIn("execution_authorization=not_granted", stdout.getvalue())
        self.assertNotIn("private://", stdout.getvalue())
        self.assertNotIn("opaque-test", stdout.getvalue())


def _read_from_bytes(data):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "duplicate.json"
        path.write_bytes(data)
        return _read(path)


if __name__ == "__main__":
    unittest.main()
