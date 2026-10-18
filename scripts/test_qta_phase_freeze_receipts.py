from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from nextgqa_pre_review_integrity_receipt import (
    ROLES, _ROSTER_DOMAIN as _INTEGRITY_DOMAIN, _canonical, _public_der,
    _sha, _sign, ReceiptError,
)
from audit_nextgqa_review_release_order import RELEASE_ROLES, _ROSTER_DOMAIN as _RELEASE_DOMAIN
from qta_phase_freeze_receipts import (
    PHASE_ARTIFACT_ROLES, create_phase_freeze_receipt,
    verify_phase_freeze_receipt,
)


class QtaPhaseFreezeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.keys = {}
        for name in ("root", "owner", "capture", "claim", "frame", "prediction", "witness"):
            private = self.root / f"{name}.private.pem"
            public = self.root / f"{name}.public.pem"
            subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)],
                           check=True, capture_output=True)
            subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
                           check=True, capture_output=True)
            self.keys[name] = (private.read_bytes(), public.read_bytes())
        self.root_pin = _sha(_public_der(self.keys["root"][1]))
        integrity_entries = [self._entry(role, name, name) for role, name in (
            ("freeze_owner", "owner"), ("claim_reviewer", "claim"),
            ("frame_reviewer", "frame"))]
        self.integrity = self._signed_roster(
            "nextgqa_pre_review_trust_roster_v1", "integrity-01",
            integrity_entries, _INTEGRITY_DOMAIN)
        release_entries = [self._entry(role, name, name) for role, name in (
            ("freeze_owner", "owner"), ("custodian", "owner"),
            ("capture_operator", "capture"), ("claim_reviewer", "claim"),
            ("frame_reviewer", "frame"), ("prediction_link_reviewer", "prediction"),
            ("release_witness", "witness"))]
        self.release = self._signed_roster(
            "nextgqa_review_release_roster_v1", "release-01",
            release_entries, _RELEASE_DOMAIN)
        self.records = [{"record_token": f"{i + 1:064x}",
                         "processor_batch_sha256": f"{i + 1:064x}"}
                        for i in range(3)]
        self.p1_artifacts = {role: ("fixture:" + role).encode()
                             for role in PHASE_ARTIFACT_ROLES["pre_baseline"]}
        self.p1 = self._create("pre_baseline", self.records, self.p1_artifacts)
        self.p2_artifacts = {
            "phase1_receipt": self.p1,
            "baseline_pq_manifest": b"fixture:baseline-pq",
            "stage2_prediction_link_review": b"fixture:stage2-review",
            "nine_branch_plan_manifest": b"fixture:branch-plans",
            "condition_scorer_source_closure": b"fixture:scorer-closure",
        }
        self.p2 = self._create("post_baseline", self.records, self.p2_artifacts,
                               prior_phase1_receipt_raw=self.p1)

    def tearDown(self):
        self.temp.cleanup()

    def _entry(self, role, name, key_id):
        public = self.keys[name][1]
        return {"role": role, "key_id": key_id,
                "public_key_pem_base64": base64.b64encode(public).decode(),
                "public_key_der_sha256": _sha(_public_der(public))}

    def _signed_roster(self, purpose, roster_id, entries, domain):
        body = {"schema_version": 1, "purpose": purpose,
                "roster_id": roster_id, "keys": entries}
        signature = _sign(self.keys["root"][0], self.keys["root"][1],
                          domain + _canonical(body))
        return _canonical({**body, "signature_ed25519_base64":
                           base64.b64encode(signature).decode()}) + b"\n"

    def _create(self, phase, records, artifacts, *, prior_phase1_receipt_raw=None,
                owner_key="owner"):
        return create_phase_freeze_receipt(
            phase=phase, freeze_id="synthetic-freeze-01", record_rows=records,
            artifact_bytes=artifacts, integrity_roster_raw=self.integrity,
            release_roster_raw=self.release,
            root_public_key_pem=self.keys["root"][1],
            expected_root_public_der_sha256=self.root_pin,
            freeze_owner_private_key_pem=self.keys[owner_key][0],
            prior_phase1_receipt_raw=prior_phase1_receipt_raw)

    def _verify(self, receipt, phase, artifacts, **kwargs):
        return verify_phase_freeze_receipt(
            receipt_raw=receipt, phase=phase, artifact_bytes=artifacts,
            integrity_roster_raw=self.integrity, release_roster_raw=self.release,
            root_public_key_pem=self.keys["root"][1],
            expected_root_public_der_sha256=self.root_pin, **kwargs)

    def test_phase1_signs_exact_synthetic_artifacts_and_denies_execution(self):
        report = self._verify(self.p1, "pre_baseline", self.p1_artifacts)
        self.assertEqual(report["status"], "verified_signed_byte_bindings_non_authorizing")
        self.assertEqual(report["requested_rows"], 3)
        self.assertEqual(report["scheduled_branches"], 3)
        self.assertTrue(report["receipt_signature_valid"])
        self.assertTrue(report["artifact_bytes_match"])
        self.assertFalse(report["human_identity_verified"])
        self.assertFalse(report["semantic_review_authenticated"])
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertFalse(report["model_load_permitted"])
        self.assertFalse(report["scoring_authorized"])
        self.assertEqual(report["authorized_forwards"], 0)
        signed_body = json.loads(self.p1)
        self.assertEqual(signed_body["status"], "signed_non_authorizing_candidate")
        self.assertEqual(signed_body["execution_authorization"], "not_granted")
        self.assertFalse(signed_body["model_load_permitted"])
        self.assertFalse(signed_body["scoring_authorized"])
        self.assertFalse(signed_body["phase_authorized"])
        self.assertEqual(signed_body["authorized_forwards"], 0)
        self.assertEqual(signed_body["planned_forward_count"], 3)

    def test_phase2_recursively_verifies_and_binds_phase1(self):
        report = self._verify(
            self.p2, "post_baseline", self.p2_artifacts,
            prior_phase1_receipt_raw=self.p1,
            prior_phase1_artifact_bytes=self.p1_artifacts)
        self.assertEqual(report["scheduled_branches"], 24)
        self.assertTrue(report["prior_phase1_receipt_valid"])
        self.assertFalse(report["scoring_authorized"])
        self.assertEqual(report["authorized_forwards"], 0)
        signed_body = json.loads(self.p2)
        self.assertEqual(signed_body["status"], "signed_non_authorizing_candidate")
        self.assertEqual(signed_body["planned_forward_count"], 24)
        self.assertEqual(signed_body["execution_authorization"], "not_granted")
        self.assertFalse(signed_body["model_load_permitted"])
        self.assertFalse(signed_body["scoring_authorized"])
        self.assertFalse(signed_body["phase_authorized"])
        self.assertEqual(signed_body["authorized_forwards"], 0)

    def test_artifact_byte_mutation_fails_verification(self):
        changed = dict(self.p1_artifacts)
        changed["clock_frame_evidence"] += b"tampered"
        with self.assertRaisesRegex(ReceiptError, "phase_receipt_binding_mismatch"):
            self._verify(self.p1, "pre_baseline", changed)

    def test_phase2_cannot_replace_prior_receipt_or_its_review_artifacts(self):
        changed = dict(self.p2_artifacts)
        changed["phase1_receipt"] = b"other receipt"
        with self.assertRaisesRegex(ReceiptError, "phase1_receipt_required_or_mismatched"):
            self._verify(self.p2, "post_baseline", changed,
                         prior_phase1_receipt_raw=self.p1,
                         prior_phase1_artifact_bytes=self.p1_artifacts)
        prior_inputs = dict(self.p1_artifacts)
        prior_inputs["stage1_review_and_lineage"] += b"changed"
        with self.assertRaises(ReceiptError):
            self._verify(self.p2, "post_baseline", self.p2_artifacts,
                         prior_phase1_receipt_raw=self.p1,
                         prior_phase1_artifact_bytes=prior_inputs)

    def test_phase2_must_preserve_phase1_record_and_batch_bindings(self):
        changed_records = [dict(row) for row in self.records]
        changed_records[1]["processor_batch_sha256"] = "f" * 64
        changed = self._create("post_baseline", changed_records, self.p2_artifacts,
                               prior_phase1_receipt_raw=self.p1)
        with self.assertRaisesRegex(ReceiptError, "phase2_record_binding_differs_from_phase1"):
            self._verify(changed, "post_baseline", self.p2_artifacts,
                         prior_phase1_receipt_raw=self.p1,
                         prior_phase1_artifact_bytes=self.p1_artifacts)

    def test_freeze_owner_signature_key_must_match_root_roster(self):
        with self.assertRaisesRegex(ReceiptError, "signing_key_does_not_match_roster"):
            self._create("pre_baseline", self.records, self.p1_artifacts,
                         owner_key="capture")

    def test_receipt_signature_tampering_is_rejected(self):
        obj = json.loads(self.p1)
        obj["signature_ed25519_base64"] = base64.b64encode(b"x" * 64).decode()
        tampered = _canonical(obj) + b"\n"
        with self.assertRaisesRegex(ReceiptError, "phase_receipt_signature_invalid"):
            self._verify(tampered, "pre_baseline", self.p1_artifacts)

    def test_trust_root_pin_mismatch_fails_before_accepting_receipt(self):
        with self.assertRaisesRegex(ReceiptError, "external_trust_root_pin_mismatch"):
            verify_phase_freeze_receipt(
                receipt_raw=self.p1, phase="pre_baseline", artifact_bytes=self.p1_artifacts,
                integrity_roster_raw=self.integrity, release_roster_raw=self.release,
                root_public_key_pem=self.keys["root"][1],
                expected_root_public_der_sha256="0" * 64)

    def test_missing_artifact_role_is_rejected(self):
        incomplete = dict(self.p1_artifacts)
        incomplete.pop("atomic_cohort_preflight")
        with self.assertRaisesRegex(ReceiptError, "phase_artifact_role_set_invalid"):
            self._create("pre_baseline", self.records, incomplete)

    def test_record_tokens_must_be_opaque_fixed_length_hex(self):
        readable = [dict(row) for row in self.records]
        readable[0]["record_token"] = "video_123_question_5"
        with self.assertRaisesRegex(ReceiptError, "phase_record_token_invalid_or_duplicate"):
            self._create("pre_baseline", readable, self.p1_artifacts)

    def test_phase1_verifier_rejects_phase2_only_arguments(self):
        with self.assertRaisesRegex(ReceiptError, "phase1_receipt_not_allowed"):
            self._verify(self.p1, "pre_baseline", self.p1_artifacts,
                         prior_phase1_receipt_raw=self.p1,
                         prior_phase1_artifact_bytes=self.p1_artifacts)

    def test_signed_receipts_do_not_echo_protected_artifact_contents(self):
        self.assertNotIn(b"fixture:clock_frame_evidence", self.p1)
        self.assertNotIn(b"fixture:stage2-review", self.p2)
        self.assertNotIn(b"fixture:baseline-pq", self.p2)


if __name__ == "__main__":
    unittest.main()
