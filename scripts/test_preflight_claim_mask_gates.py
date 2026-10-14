import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from preflight_claim_mask_gates import main, preflight
from test_validate_claim_mask_protocol import fixture as protocol_fixture


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

    @patch("preflight_claim_mask_gates.audit_data_rights")
    @patch("preflight_claim_mask_gates.audit_item_rights")
    @patch("preflight_claim_mask_gates.audit_claim_review")
    def test_all_structural_components_pass_but_preflight_remains_blocked(
        self, claim, rights, data_rights
    ):
        claim.return_value = {"schema_version": 1, "structural_status": "passed", "failure_codes": [], "execution_authorization": "not_granted"}
        rights.return_value = {"schema_version": 1, "structurally_valid": True, "failure_codes": [], "execution_authorization": "not_granted"}
        data_rights.return_value = {"schema_version": 1, "structural_status": "passed", "failure_codes": [], "execution_authorization": "not_granted"}
        with tempfile.TemporaryDirectory() as tmp:
            paths, protocol = self.paths_and_protocol(tmp)
            report = preflight(paths)
        self.assertEqual(report["overall_status"], "blocked")
        self.assertEqual(report["structural_checks_status"], "passed")
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertIn("schema_v12_confirmatory_gate_unresolved", report["blocker_codes"])
        self.assertIn("protocol_stage_not_confirmatory", report["blocker_codes"])
        self.assertIn("sesoi_not_frozen", report["blocker_codes"])
        self.assertIn("execution_authorization_not_granted", report["blocker_codes"])
        claim.assert_called_once()
        rights.assert_called_once()
        data_rights.assert_called_once()

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
