from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from nextgqa_completed_review_attestation import (
    AttestationError,
    create_attestation,
    verify_attestation,
)
from nextgqa_pre_review_integrity_receipt import _public_der
from test_audit_nextgqa_review_handoff import fixture as review_fixture
import hashlib


class CompletedReviewAttestationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        private = root / "synthetic-reviewer-private.pem"
        public = root / "synthetic-reviewer-public.pem"
        subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519",
                        "-out", str(private)], check=True, capture_output=True)
        subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout",
                        "-out", str(public)], check=True, capture_output=True)
        self.private_pem, self.public_pem = private.read_bytes(), public.read_bytes()
        self.key_pin = hashlib.sha256(_public_der(self.public_pem)).hexdigest()
        self.receipt_sha = "a" * 64
        self.key_id = "synthetic-claim-reviewer"
        (self.worksheet, self.worksheet_raw, self.lineage, self.lineage_raw,
         self.review, self.review_raw, self.review_index,
         self.review_index_raw) = review_fixture()
        self.expected_lineage_sha256 = hashlib.sha256(self.lineage_raw).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def _create(self, **overrides):
        args = {
            "worksheet": self.worksheet,
            "worksheet_raw": self.worksheet_raw,
            "lineage": self.lineage,
            "lineage_raw": self.lineage_raw,
            "review": self.review,
            "review_raw": self.review_raw,
            "review_index": self.review_index,
            "review_index_raw": self.review_index_raw,
            "expected_lineage_sha256": self.expected_lineage_sha256,
            "pre_review_receipt_sha256": self.receipt_sha,
            "reviewer_key_id": self.key_id,
            "reviewer_private_key_pem": self.private_pem,
            "reviewer_public_key_pem": self.public_pem,
            "expected_reviewer_key_fingerprint_sha256": self.key_pin,
        }
        args.update(overrides)
        return create_attestation(**args)

    def _verify(self, attestation_raw, **overrides):
        args = {
            "attestation_raw": attestation_raw,
            "worksheet": self.worksheet,
            "worksheet_raw": self.worksheet_raw,
            "lineage": self.lineage,
            "lineage_raw": self.lineage_raw,
            "review": self.review,
            "review_raw": self.review_raw,
            "review_index": self.review_index,
            "review_index_raw": self.review_index_raw,
            "expected_lineage_sha256": self.expected_lineage_sha256,
            "pre_review_receipt_sha256": self.receipt_sha,
            "expected_reviewer_key_id": self.key_id,
            "reviewer_public_key_pem": self.public_pem,
            "expected_reviewer_key_fingerprint_sha256": self.key_pin,
        }
        args.update(overrides)
        return verify_attestation(**args)

    def test_signature_binds_exact_exports_and_stays_non_authorizing(self):
        report = self._verify(self._create())
        self.assertTrue(report["reviewer_key_signature_valid"])
        self.assertTrue(report["key_holder_adoption_attested"])
        self.assertFalse(report["human_identity_verified"])
        self.assertFalse(report["semantic_review_authenticated"])
        self.assertFalse(report["pre_review_receipt_verified"])
        self.assertFalse(report["reviewer_role_roster_verified"])
        self.assertFalse(report["source_media_custody_verified"])
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_changed_payload_or_index_is_rejected(self):
        signed = self._create()
        with self.assertRaisesRegex(AttestationError, "review_handoff_structure_invalid"):
            self._verify(signed, review_raw=self.review_raw + b" ")
        with self.assertRaisesRegex(AttestationError, "attestation_binding_or_statement_mismatch"):
            self._verify(signed, review_index_raw=self.review_index_raw + b" ")

    def test_external_fingerprint_pin_is_required(self):
        with self.assertRaisesRegex(AttestationError, "reviewer_public_key_pin_mismatch"):
            self._create(expected_reviewer_key_fingerprint_sha256="0" * 64)
        signed = self._create()
        with self.assertRaisesRegex(AttestationError, "reviewer_public_key_pin_mismatch"):
            self._verify(signed, expected_reviewer_key_fingerprint_sha256="0" * 64)

    def test_signing_private_key_must_match_pinned_public_key(self):
        other_private = Path(self.temp.name) / "other-private.pem"
        subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519",
                        "-out", str(other_private)], check=True, capture_output=True)
        with self.assertRaisesRegex(AttestationError, "reviewer_key_pair_mismatch"):
            self._create(reviewer_private_key_pem=other_private.read_bytes())

    def test_malformed_duplicate_key_json_fails_closed(self):
        duplicate = b'{"schema_version":3,"schema_version":3}'
        with self.assertRaisesRegex(AttestationError, "json_duplicate_key"):
            self._create(review_raw=duplicate)
        with self.assertRaisesRegex(AttestationError, "json_duplicate_key"):
            self._create(review_index_raw=b'{"schema_version":1,"schema_version":1}')

    def test_exponent_overflow_is_rejected_even_in_unknown_fields(self):
        malformed = self.review_raw[:-2] + b',"unused":1e9999}\n'
        with self.assertRaisesRegex(AttestationError, "json_nonfinite_value"):
            self._create(review_raw=malformed)

    def test_structurally_incomplete_export_cannot_be_attested(self):
        review = dict(self.review)
        review["records"] = []
        review_raw = json.dumps(review, sort_keys=True).encode() + b"\n"
        index = dict(self.review_index,
                     review_payload_sha256=hashlib.sha256(review_raw).hexdigest())
        index_raw = json.dumps(index, sort_keys=True).encode() + b"\n"
        with self.assertRaisesRegex(AttestationError, "review_handoff_structure_invalid"):
            self._create(review=review, review_raw=review_raw,
                         review_index=index, review_index_raw=index_raw)

    def test_lineage_pin_and_parsed_bytes_must_match(self):
        with self.assertRaisesRegex(AttestationError, "lineage_hash_pin_mismatch"):
            self._create(expected_lineage_sha256="0" * 64)
        with self.assertRaisesRegex(AttestationError, "review_bytes_mismatch"):
            self._create(review={**self.review, "review_completed": False})

    def test_receipt_or_reviewer_key_mismatch_is_rejected(self):
        signed = self._create()
        with self.assertRaisesRegex(AttestationError, "attestation_binding_or_statement_mismatch"):
            self._verify(signed, pre_review_receipt_sha256="b" * 64)
        with self.assertRaisesRegex(AttestationError, "attestation_binding_or_statement_mismatch"):
            self._verify(signed, expected_reviewer_key_id="other-reviewer")

    def test_unsigned_or_modified_envelope_is_rejected(self):
        signed = json.loads(self._create())
        signed["execution_authorization"] = "granted"
        tampered = json.dumps(signed).encode()
        with self.assertRaisesRegex(AttestationError, "attestation_binding_or_statement_mismatch"):
            self._verify(tampered)


if __name__ == "__main__":
    unittest.main()
