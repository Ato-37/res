import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from preflight_claim_mask_gates import main, preflight
from require_execution_preflight import require_execution_preflight
from build_claim_input_manifest import build as build_claim_input
from test_audit_claim_review_evidence_index import fixture as claim_review_fixture, REVISION
from test_audit_item_rights_manifest import fixture as item_rights_fixture
from test_validate_claim_mask_protocol import (fixture as protocol_fixture,
                                               compute_primary_estimand_sha256,
                                               compute_inference_plan_sha256)


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return raw


class ClaimMaskGatePreflightTests(unittest.TestCase):
    def paths_and_protocol(self, root):
        protocol = protocol_fixture()
        names = {
            "protocol": "protocol.json", "eligible_manifest": "eligible.json",
            "claim_input_manifest": "claim-input.json", "claim_mapping_ledger": "claim-ledger.json",
            "claim_review_index": "claim-review-index.json", "item_rights_ledger": "item-rights.json",
            "data_rights_index": "data-rights-index.json",
        }
        values = {key: {} for key in names}
        values["protocol"] = protocol
        paths = {}
        for key, name in names.items():
            path = Path(root) / name
            write_json(path, values[key])
            paths[key] = path
        return paths, protocol

    def fully_bound_synthetic_paths(self, root):
        claim_parts = list(claim_review_fixture())
        protocol = protocol_fixture()
        eligible, _eligible_raw, rights, _rights_raw = item_rights_fixture()
        eligible_raw = write_json(Path(root) / "eligible-work.json", eligible)
        rights["eligible_manifest_sha256"] = hashlib.sha256(eligible_raw).hexdigest()
        rights_raw = write_json(Path(root) / "rights-work.json", rights)
        for index, row in enumerate(eligible):
            video_id = row["video_id"]
            claim_parts[1][index]["video_id"] = video_id
            claim_parts[3]["claims"][index]["video_id"] = video_id
            claim_parts[5]["claims"][index]["video_id"] = video_id
            claim_parts[7]["records"][index]["video_id"] = video_id
        # Exercise UTF-8 builder bytes through every downstream hash binding.
        unicode_claim = "Café, synthetic evidence statement."
        unicode_claim_sha = hashlib.sha256(unicode_claim.encode("utf-8")).hexdigest()
        claim_parts[3]["claims"][0]["claim_text"] = unicode_claim
        claim_parts[3]["claims"][0]["claim_text_sha256"] = unicode_claim_sha
        claim_parts[5]["claims"][0]["claim_text_sha256"] = unicode_claim_sha
        claim_parts[5]["claims"][0]["review"]["claim_text_sha256_seen"] = unicode_claim_sha
        eligible_sha = hashlib.sha256(eligible_raw).hexdigest()
        claim_parts[3]["eligible_manifest_sha256"] = eligible_sha
        claim_parts[5]["eligible_manifest_sha256"] = eligible_sha
        generated_claims = {"schema_version": 1, "claims": [
            {key: row[key] for key in ("video_id", "claim_id", "claim_text", "model_revision",
                                       "generation_run_sha256")}
            for row in claim_parts[3]["claims"]
        ]}
        claim_input, claim_input_raw = build_claim_input(
            eligible, eligible_raw, generated_claims, claim_parts[3]["eligible_dataset_id"],
            claim_parts[3]["claim_policy_sha256"], claim_parts[3]["occurrence_policy"],
        )
        claim_ledger_raw = write_json(Path(root) / "claim-ledger-work.json", claim_parts[5])
        claim_parts[7]["mapping_ledger_sha256"] = hashlib.sha256(claim_ledger_raw).hexdigest()
        claim_review_raw = write_json(Path(root) / "claim-review-work.json", claim_parts[7])
        protocol["model"]["revision"] = REVISION
        protocol["cohort"]["eligible_manifest_sha256"] = eligible_sha
        protocol["claim_review"].update(
            claim_manifest_sha256=hashlib.sha256(claim_input_raw).hexdigest(),
            mapping_sha256=hashlib.sha256(claim_ledger_raw).hexdigest(),
            review_record_sha256=hashlib.sha256(claim_review_raw).hexdigest(),
        )
        agreement = {
            "applicability": "applicable", "sha256": "a" * 64,
            "private_locator": "private://synthetic/agreement",
            "review_record_sha256": "a" * 64, "reviewer_role": "synthetic reviewer",
            "reviewed_at_utc": "2026-10-15T00:00:00Z",
            "planned_use_scope": rights["intended_use_scope"],
            "decisions": {use: "permitted" for use in rights["intended_uses"]},
            "non_applicability_basis_sha256": None,
        }
        evidence_index = {
            "schema_version": 1,
            "eligible_manifest": {"sha256": eligible_sha, "private_locator": "private://synthetic/eligible",
                                  "eligible_video_count": len(eligible)},
            "item_rights_manifest": {"sha256": hashlib.sha256(rights_raw).hexdigest(),
                                     "private_locator": "private://synthetic/rights"},
            "annotation_terms": {"sha256": rights["annotation_rights"]["evidence_sha256"],
                                  "private_locator": "private://synthetic/annotation"},
            "item_media_conditions": {"sha256": "a" * 64, "private_locator": "private://synthetic/media",
                                      "covered_video_count": len(eligible)},
            "dataset_access_agreement": agreement,
        }
        evidence_raw = write_json(Path(root) / "data-rights-work.json", evidence_index)
        protocol["data_rights"] = {
            "scope": rights["intended_use_scope"], "status": "verified",
            "evidence_sha256": hashlib.sha256(evidence_raw).hexdigest(),
            "item_rights_manifest_sha256": hashlib.sha256(rights_raw).hexdigest(),
            "covered_eligible_video_count": len(eligible), "unresolved_eligible_video_count": 0,
        }
        protocol["estimand"]["primary_estimand_sha256"] = compute_primary_estimand_sha256(protocol)
        protocol["analysis"]["inference_plan_sha256"] = compute_inference_plan_sha256(protocol)
        values = {
            "protocol": protocol, "eligible_manifest": eligible,
            "claim_input_manifest": claim_input, "claim_mapping_ledger": claim_parts[5],
            "claim_review_index": claim_parts[7], "item_rights_ledger": rights,
            "data_rights_index": evidence_index,
        }
        paths = {}
        for key, value in values.items():
            path = Path(root) / f"{key}.json"
            if key == "claim_input_manifest":
                path.write_bytes(claim_input_raw)
                self.assertEqual(path.read_bytes(), claim_input_raw)
            else:
                write_json(path, value)
            paths[key] = path
        return paths

    def test_all_structural_components_pass_but_preflight_remains_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fully_bound_synthetic_paths(tmp)
            report = preflight(paths)
            report_path = Path(tmp) / "cross-gate-report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "deliberately never grants execution"):
                require_execution_preflight(report_path)
        self.assertEqual(report["overall_status"], "blocked")
        self.assertEqual(report["structural_checks_status"], "passed")
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertIn("schema_v12_confirmatory_gate_unresolved", report["blocker_codes"])
        self.assertIn("protocol_stage_not_confirmatory", report["blocker_codes"])
        self.assertIn("sesoi_not_frozen", report["blocker_codes"])
        self.assertIn("execution_authorization_not_granted", report["blocker_codes"])

    @patch("preflight_claim_mask_gates.audit_data_rights", side_effect=RuntimeError("private detail"))
    @patch("preflight_claim_mask_gates.audit_item_rights")
    @patch("preflight_claim_mask_gates.audit_claim_review")
    def test_component_exception_is_generic_and_never_authorizes(self, claim, rights, _data):
        claim.return_value = {"schema_version": 1, "structural_status": "passed", "failure_codes": [], "execution_authorization": "not_granted"}
        rights.return_value = {"schema_version": 1, "structurally_valid": True, "failure_codes": [], "execution_authorization": "not_granted"}
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = self.paths_and_protocol(tmp)
            report = preflight(paths)
        self.assertEqual(report["components"]["data_rights_evidence"]["status"], "error")
        self.assertEqual(report["components"]["data_rights_evidence"]["error_code"], "component_audit_error")
        self.assertNotIn("private detail", json.dumps(report))
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertEqual(report["overall_status"], "blocked")

    @patch("preflight_claim_mask_gates.validate_protocol", side_effect=RuntimeError("private protocol detail"))
    def test_protocol_validator_exception_is_generic(self, _validate):
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = self.paths_and_protocol(tmp)
            report = preflight(paths)
        self.assertEqual(report["components"]["protocol"]["status"], "error")
        self.assertEqual(report["components"]["protocol"]["error_code"], "component_audit_error")
        self.assertNotIn("private protocol detail", json.dumps(report))
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_malformed_input_returns_generic_blocked_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = self.paths_and_protocol(tmp)
            paths["protocol"].write_text('{"duplicate": 1, "duplicate": 2}', encoding="utf-8")
            report = preflight(paths)
        self.assertEqual(report["structural_checks_status"], "failed")
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertEqual(report["blocker_codes"], ["private_input_invalid"])

    @patch("preflight_claim_mask_gates.audit_data_rights")
    @patch("preflight_claim_mask_gates.audit_item_rights")
    @patch("preflight_claim_mask_gates.audit_claim_review")
    def test_child_contract_and_truthy_status_cannot_pass(self, claim, rights, data_rights):
        claim.return_value = {"schema_version": 1, "structural_status": "passed",
                              "failure_codes": ["private-video-id"], "execution_authorization": "not_granted"}
        rights.return_value = {"schema_version": 1, "structurally_valid": "true",
                               "failure_codes": [], "execution_authorization": "not_granted"}
        data_rights.return_value = {"schema_version": True, "structural_status": "passed",
                                    "failure_codes": [], "execution_authorization": "granted"}
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = self.paths_and_protocol(tmp)
            report = preflight(paths)
        self.assertEqual(report["components"]["item_rights"]["status"], "failed")
        self.assertEqual(report["components"]["data_rights_evidence"]["status"], "failed")
        self.assertIn("component_contract_invalid", report["components"]["data_rights_evidence"]["failure_codes"])
        self.assertEqual(report["components"]["claim_review"]["status"], "failed")
        self.assertEqual(report["components"]["claim_review"]["failure_codes"],
                         ["component_contract_invalid", "component_validation_failed"])
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertNotIn("private-video-id", json.dumps(report))
        self.assertEqual(report["overall_status"], "blocked")

    def test_cli_reports_structural_pass_as_blocked_without_authorization(self):
        with patch("preflight_claim_mask_gates.audit_data_rights") as data_rights, \
             patch("preflight_claim_mask_gates.audit_item_rights") as rights, \
             patch("preflight_claim_mask_gates.audit_claim_review") as claim:
            data_rights.return_value = {"schema_version": 1, "structural_status": "passed", "failure_codes": [], "execution_authorization": "not_granted"}
            rights.return_value = {"schema_version": 1, "structurally_valid": True, "failure_codes": [], "execution_authorization": "not_granted"}
            claim.return_value = {"schema_version": 1, "structural_status": "passed", "failure_codes": [], "execution_authorization": "not_granted"}
            with tempfile.TemporaryDirectory() as tmp:
                paths, _ = self.paths_and_protocol(tmp)
                args = []
                for option, key in (
                    ("--protocol", "protocol"), ("--eligible-manifest", "eligible_manifest"),
                    ("--claim-input-manifest", "claim_input_manifest"),
                    ("--claim-mapping-ledger", "claim_mapping_ledger"),
                    ("--claim-review-index", "claim_review_index"),
                    ("--item-rights-ledger", "item_rights_ledger"),
                    ("--data-rights-index", "data_rights_index"),
                ):
                    args.extend((option, str(paths[key])))
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    status = main(args)
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 1)
        self.assertEqual(payload["overall_status"], "blocked")
        self.assertEqual(payload["execution_authorization"], "not_granted")
        self.assertNotIn("ready", output.getvalue())


if __name__ == "__main__":
    unittest.main()
