from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from audit_nextgqa_review_release_order import RELEASE_ROLES, _ROSTER_DOMAIN
from nextgqa_pre_review_integrity_receipt import _canonical, _public_der, _sha, _sign
from nextgqa_prediction_link_attestation import (
    PredictionLinkAttestationError,
    create_attestation,
    verify_attestation,
)
from nextgqa_prediction_link_review import displayed_fields_sha256
from qta_test_baseline_v3_fixture import build_synthetic_baseline_v3


class PredictionLinkAttestationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.keys = {}
        for name in ("root", "owner", "capture", "claim", "frame", "prediction", "witness"):
            private = self.root / f"{name}.pem"
            public = self.root / f"{name}.pub"
            subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519",
                            "-out", str(private)], check=True, capture_output=True)
            subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout",
                            "-out", str(public)], check=True, capture_output=True)
            private.chmod(0o600)
            public.chmod(0o600)
            self.keys[name] = (private.read_bytes(), public.read_bytes())
        self.root_pin = _sha(_public_der(self.keys["root"][1]))
        self.integrity_roster = self._signed_roster(
            "nextgqa_pre_review_trust_roster_v1",
            [("freeze_owner", "owner"), ("claim_reviewer", "claim"),
             ("frame_reviewer", "frame")], b"nextgqa_pre_review_roster_v1\0")
        role_to_key = {"freeze_owner": "owner", "custodian": "owner",
                       "capture_operator": "capture",
                       "claim_reviewer": "claim", "frame_reviewer": "frame",
                       "prediction_link_reviewer": "prediction",
                       "release_witness": "witness"}
        self.release_roster = self._signed_roster(
            "nextgqa_review_release_roster_v1",
            [(role, role_to_key[role]) for role in RELEASE_ROLES], _ROSTER_DOMAIN)
        self.video_review_hash = "c" * 64
        self.protocol_hash = "b" * 64
        (self.manifest_raw, self.rows,
         self.source_hashes) = build_synthetic_baseline_v3(self.video_review_hash)
        self.manifest_hash = _sha(self.manifest_raw)
        receipt = {
            "schema_version": 1, "purpose": "qta_prediction_link_adjudication",
            "status": "prediction_link_review_recorded_unvalidated",
            "baseline_manifest_sha256": self.manifest_hash,
            "protocol_sha256": self.protocol_hash,
            "prior_video_review_receipt_sha256": self.video_review_hash,
            "reviewer_role": "independent_prediction_link_reviewer",
            "reviewer_private_id": "opaque-reviewer",
            "independence_attestation": "independent_from_generation_scoring_custody_and_video_review",
            "gold_and_branch_outcomes_hidden": True,
            "reviewed_at_utc": "2026-10-18T10:00:00Z",
            "records": [{
                "review_token": row["review_token"],
                "displayed_fields_sha256": displayed_fields_sha256(
                    row, video_claim_review_receipt_sha256=self.video_review_hash),
                "judgment": "supports_p_over_q", "rationale": "Synthetic test only."}
                for row in self.rows],
            "execution_authorization": "not_granted",
        }
        self.receipt_raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":"))
                            .encode())
        self.expected = dict(
            receipt_raw=self.receipt_raw,
            baseline_manifest_raw=self.manifest_raw,
            expected_baseline_manifest_sha256=self.manifest_hash,
            expected_protocol_sha256=self.protocol_hash,
            expected_video_review_receipt_sha256=self.video_review_hash,
            expected_cohort_tokens=[row["review_token"] for row in self.rows],
            expected_source_field_sha256_by_token=self.source_hashes,
            integrity_roster_raw=self.integrity_roster,
            release_roster_raw=self.release_roster,
            root_public_key_pem=self.keys["root"][1],
            expected_root_public_der_sha256=self.root_pin,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _signed_roster(self, purpose, role_names, domain):
        entries = []
        key_ids = {"owner": "owner-key", "capture": "capture-key",
                   "claim": "claim-key", "frame": "frame-key",
                   "prediction": "prediction-key", "witness": "witness-key"}
        for role, name in role_names:
            public = self.keys[name][1]
            entries.append({"role": role, "key_id": key_ids[name],
                            "public_key_pem_base64": base64.b64encode(public).decode(),
                            "public_key_der_sha256": _sha(_public_der(public))})
        body = {"schema_version": 1, "purpose": purpose,
                "roster_id": "synthetic-roster", "keys": entries}
        signature = _sign(self.keys["root"][0], self.keys["root"][1],
                          domain + _canonical(body))
        return _canonical({**body, "signature_ed25519_base64":
                           base64.b64encode(signature).decode()}) + b"\n"

    def test_exact_bytes_are_attested_and_execution_remains_denied(self):
        envelope = create_attestation(
            **self.expected,
            reviewer_private_key_pem=self.keys["prediction"][0],
            reviewer_public_key_pem=self.keys["prediction"][1])
        result = verify_attestation(attestation_raw=envelope, **self.expected)
        self.assertTrue(result["reviewer_key_signature_valid"])
        self.assertTrue(result["all_rows_self_report_support_p_over_q"])
        self.assertFalse(result["human_identity_verified"])
        self.assertFalse(result["semantic_judgment_authenticated"])
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_changed_receipt_or_manifest_is_rejected(self):
        envelope = create_attestation(
            **self.expected,
            reviewer_private_key_pem=self.keys["prediction"][0],
            reviewer_public_key_pem=self.keys["prediction"][1])
        changed_receipt = self.receipt_raw + b" "
        with self.assertRaisesRegex(PredictionLinkAttestationError,
                                    "stage2_attestation_binding_mismatch"):
            verify_attestation(attestation_raw=envelope,
                               **{**self.expected, "receipt_raw": changed_receipt})
        with self.assertRaisesRegex(PredictionLinkAttestationError,
                                    "stage2_receipt_invalid:baseline_manifest_bytes_hash_mismatch"):
            verify_attestation(attestation_raw=envelope,
                               **{**self.expected,
                                  "baseline_manifest_raw": self.manifest_raw + b" "})

    def test_tampered_signature_and_oversized_receipt_are_rejected(self):
        envelope = create_attestation(
            **self.expected,
            reviewer_private_key_pem=self.keys["prediction"][0],
            reviewer_public_key_pem=self.keys["prediction"][1])
        parsed = json.loads(envelope)
        parsed["signature_ed25519_base64"] = base64.b64encode(b"x" * 64).decode()
        bad_envelope = json.dumps(parsed, sort_keys=True).encode()
        with self.assertRaisesRegex(PredictionLinkAttestationError,
                                    "stage2_signature_invalid"):
            verify_attestation(attestation_raw=bad_envelope, **self.expected)
        with self.assertRaisesRegex(PredictionLinkAttestationError,
                                    "stage2_receipt_bytes_invalid"):
            create_attestation(
                **{**self.expected, "receipt_raw": b" " * (2 * 1024 * 1024 + 1)},
                reviewer_private_key_pem=self.keys["prediction"][0],
                reviewer_public_key_pem=self.keys["prediction"][1])
        with self.assertRaisesRegex(PredictionLinkAttestationError,
                                    "stage2_attestation_bytes_invalid"):
            verify_attestation(attestation_raw=b" " * (2 * 1024 * 1024 + 1),
                               **self.expected)

    def test_duplicate_and_nonfinite_stage2_json_are_rejected(self):
        for invalid in (b'{"schema_version":1,"schema_version":1}',
                        self.receipt_raw.replace(b'"rationale":"Synthetic test only."',
                                                 b'"rationale":NaN')):
            with self.subTest(payload_kind="duplicate_or_nonfinite"), self.assertRaises(
                    PredictionLinkAttestationError):
                create_attestation(
                    **{**self.expected, "receipt_raw": invalid},
                    reviewer_private_key_pem=self.keys["prediction"][0],
                    reviewer_public_key_pem=self.keys["prediction"][1])

    def test_wrong_signer_and_non_support_cohort_are_rejected(self):
        with self.assertRaisesRegex(PredictionLinkAttestationError,
                                    "prediction_link_reviewer_key_does_not_match_roster"):
            create_attestation(
                **self.expected,
                reviewer_private_key_pem=self.keys["claim"][0],
                reviewer_public_key_pem=self.keys["claim"][1])
        parsed = json.loads(self.receipt_raw)
        parsed["records"][1]["judgment"] = "unrelated"
        bad_receipt = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaisesRegex(PredictionLinkAttestationError,
                                    "stage2_cohort_not_eligible"):
            create_attestation(
                **{**self.expected, "receipt_raw": bad_receipt},
                reviewer_private_key_pem=self.keys["prediction"][0],
                reviewer_public_key_pem=self.keys["prediction"][1])

    def test_private_file_cli_signs_verifies_and_refuses_overwrite(self):
        files = {}
        payloads = {
            "receipt": self.receipt_raw,
            "baseline": self.manifest_raw,
            "integrity": self.integrity_roster,
            "release": self.release_roster,
            "root_public": self.keys["root"][1],
            "reviewer_private": self.keys["prediction"][0],
            "reviewer_public": self.keys["prediction"][1],
            "expected": json.dumps({
                "cohort_tokens": [row["review_token"] for row in self.rows],
                "source_field_sha256_by_token": self.source_hashes,
            }, sort_keys=True).encode(),
        }
        for label, raw in payloads.items():
            path = self.root / f"{label}.json"
            path.write_bytes(raw)
            path.chmod(0o600)
            files[label] = path
        private_dir = self.root / "private-output"
        private_dir.mkdir(mode=0o700)
        private_dir.chmod(0o700)
        output = private_dir / "stage2-attestation.json"
        common = [
            "--receipt", str(files["receipt"]), "--baseline-manifest", str(files["baseline"]),
            "--expected-baseline-manifest-sha256", self.manifest_hash,
            "--expected-protocol-sha256", self.protocol_hash,
            "--expected-video-review-receipt-sha256", self.video_review_hash,
            "--expected-inputs-json", str(files["expected"]),
            "--integrity-roster", str(files["integrity"]),
            "--release-roster", str(files["release"]),
            "--root-public-key", str(files["root_public"]),
            "--root-public-der-sha256", self.root_pin,
        ]
        script = Path(__file__).with_name("nextgqa_prediction_link_attestation.py")
        env = {**os.environ, "PYTHONPATH": str(script.parent)}
        sign = [sys.executable, str(script), "sign", *common,
                "--reviewer-private-key", str(files["reviewer_private"]),
                "--reviewer-public-key", str(files["reviewer_public"]),
                "--output", str(output)]
        first = subprocess.run(sign, capture_output=True, text=True, env=env)
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        second = subprocess.run(sign, capture_output=True, text=True, env=env)
        self.assertEqual(second.returncode, 2)
        self.assertIn("output_already_exists", second.stdout)
        verify = subprocess.run(
            [sys.executable, str(script), "verify", *common,
             "--attestation", str(output)], capture_output=True, text=True, env=env)
        self.assertEqual(verify.returncode, 0, verify.stderr + verify.stdout)
        self.assertEqual(json.loads(verify.stdout)["execution_authorization"], "not_granted")


if __name__ == "__main__":
    unittest.main()
