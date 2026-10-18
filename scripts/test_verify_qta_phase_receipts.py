from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import test_qta_phase_freeze_receipts as phase_receipt_fixtures
import verify_qta_phase_receipts as verifier
from verify_qta_phase_receipts import BundleError, main, verify_bundle


class VerifyQtaPhaseReceiptsCliTests(unittest.TestCase):
    def setUp(self):
        self.fixture = phase_receipt_fixtures.QtaPhaseFreezeReceiptTests(
            "test_phase2_rejects_stale_external_cohort_pin")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.bundle_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.bundle_temp.cleanup)
        self.root = Path(self.bundle_temp.name)
        self.root.chmod(0o700)
        p1_artifacts, p1, p2_artifacts, p2, _records = self.fixture._phase2_candidate_bundle()
        self.expected_pin = self.fixture.expected_external_cohort_pin_sha256
        self.expected_root = self.fixture.root_pin
        self._write_bundle(p1_artifacts, p1, p2_artifacts, p2)

    def _write_private(self, relative: str, raw: bytes) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.chmod(0o700)
        path.write_bytes(raw)
        path.chmod(0o600)
        return relative

    def _write_bundle(self, p1_artifacts, p1, p2_artifacts, p2):
        p1_map = {role: self._write_private(f"phase1/{role}.bin", raw)
                  for role, raw in p1_artifacts.items()}
        p2_map = {role: self._write_private(f"phase2/{role}.bin", raw)
                  for role, raw in p2_artifacts.items()}
        p1_receipt = self._write_private("phase1_receipt.json", p1)
        p2_receipt = self._write_private("phase2_receipt.json", p2)
        self.integrity = self._write_private("integrity_roster.json", self.fixture.integrity)
        self.release = self._write_private("release_roster.json", self.fixture.release)
        self.root_key = self._write_private("root_public_key.pem", self.fixture.keys["root"][1])
        sidecar_dir = self.root / "sidecars"
        sidecar_dir.mkdir(mode=0o700)
        for digest, raw in self.fixture.capture_sidecars.items():
            path = sidecar_dir / f"{digest}.bin"
            path.write_bytes(raw)
            path.chmod(0o600)
        self.manifest = {
            "schema_version": 1,
            "purpose": "qta_phase_receipt_bundle_non_authorizing_v1",
            "phase1_receipt": p1_receipt,
            "phase1_artifacts": p1_map,
            "phase2_receipt": p2_receipt,
            "phase2_artifacts": p2_map,
            "integrity_roster": self.integrity,
            "release_roster": self.release,
            "root_public_key": self.root_key,
            "attention_capture_sidecar_dir": "sidecars",
        }
        self._save_manifest()

    def _save_manifest(self):
        path = self.root / "bundle.json"
        path.write_text(json.dumps(self.manifest, sort_keys=True), encoding="utf-8")
        path.chmod(0o600)

    def _verify(self, *, pin=None, root=None):
        return verify_bundle(
            bundle_dir=self.root,
            expected_external_cohort_pin_sha256=(pin if pin is not None else self.expected_pin),
            expected_root_public_der_sha256=(root if root is not None else self.expected_root),
        )

    def test_valid_bundle_is_structurally_verified_but_never_authorized(self):
        report = self._verify()
        self.assertTrue(report["phase1_receipt_signature_valid"])
        self.assertTrue(report["phase2_receipt_signature_valid"])
        self.assertTrue(report["external_cohort_pin_matches_caller_pin"])
        self.assertFalse(report["external_cohort_pin_origin_authenticated"])
        self.assertFalse(report["trust_root_value_origin_authenticated"])
        self.assertFalse(report["forward_source_authenticated"])
        self.assertFalse(report["attention_capture_authenticated"])
        self.assertEqual(report["authorized_forwards"], 0)
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["scoring_authorization"], "not_granted")
        self.assertNotIn("record_tokens", report)

    def test_wrong_independent_pin_is_rejected_and_never_derived_from_bundle(self):
        with self.assertRaisesRegex(ValueError, "external_cohort_pin_external_hash_mismatch"):
            self._verify(pin="0" * 64)

    def test_wrong_root_pin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "external_trust_root_pin_mismatch"):
            self._verify(root="0" * 64)

    def test_manifest_path_traversal_and_symlink_are_rejected(self):
        self.manifest["phase1_receipt"] = "../phase1_receipt.json"
        self._save_manifest()
        with self.assertRaisesRegex(BundleError, "bundle_relative_path_invalid"):
            self._verify()
        self.manifest["phase1_receipt"] = "phase1_receipt.json"
        (self.root / "phase1_receipt.json").unlink()
        os.symlink(self.root / "phase2_receipt.json", self.root / "phase1_receipt.json")
        self._save_manifest()
        with self.assertRaisesRegex(BundleError, "bundle_symlink_rejected"):
            self._verify()

    def test_symlink_bundle_root_is_rejected(self):
        link = self.root.parent / (self.root.name + "-link")
        os.symlink(self.root, link, target_is_directory=True)
        with self.assertRaisesRegex(BundleError, "bundle_root_symlink_rejected"):
            verify_bundle(
                bundle_dir=link,
                expected_external_cohort_pin_sha256=self.expected_pin,
                expected_root_public_der_sha256=self.expected_root,
            )

    def test_duplicate_manifest_keys_and_permissive_artifact_mode_are_rejected(self):
        (self.root / "bundle.json").write_text(
            '{"schema_version":1,"schema_version":1}', encoding="utf-8")
        (self.root / "bundle.json").chmod(0o600)
        with self.assertRaisesRegex(BundleError, "bundle_manifest_duplicate_key"):
            self._verify()
        self._save_manifest()
        receipt = self.root / self.manifest["phase1_receipt"]
        receipt.chmod(0o644)
        with self.assertRaisesRegex(BundleError, "bundle_file_permissions_or_size_invalid"):
            self._verify()

    def test_sidecar_missing_extra_and_symlink_are_rejected(self):
        first_digest = next(iter(self.fixture.capture_sidecars))
        target = self.root / "sidecars" / f"{first_digest}.bin"
        payload = target.read_bytes()
        target.unlink()
        with self.assertRaisesRegex(BundleError, "bundle_sidecar_file_set_invalid"):
            self._verify()
        target.write_bytes(payload)
        target.chmod(0o600)
        extra = self.root / "sidecars" / ("0" * 64 + ".bin")
        extra.write_bytes(b"extra")
        extra.chmod(0o600)
        with self.assertRaisesRegex(BundleError, "bundle_sidecar_file_set_invalid"):
            self._verify()
        extra.unlink()
        target.unlink()
        outside = self.root / "sidecar-target.bin"
        outside.write_bytes(payload)
        outside.chmod(0o600)
        os.symlink(outside, target)
        with self.assertRaisesRegex(BundleError, "bundle_symlink_rejected"):
            self._verify()

    def test_oversized_manifest_is_rejected_before_parsing(self):
        original = verifier._MAX_MANIFEST_BYTES
        verifier._MAX_MANIFEST_BYTES = 8
        try:
            with self.assertRaisesRegex(BundleError,
                                        "bundle_file_permissions_or_size_invalid"):
                self._verify()
        finally:
            verifier._MAX_MANIFEST_BYTES = original

    def test_phase1_copy_must_match_phase1_receipt(self):
        changed = self.root / self.manifest["phase2_artifacts"]["phase1_receipt"]
        changed.write_bytes(b"substituted")
        changed.chmod(0o600)
        with self.assertRaisesRegex(BundleError, "bundle_phase1_receipt_copy_mismatch"):
            self._verify()

    def test_cli_requires_both_caller_supplied_trust_values(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            main(["--bundle-dir", str(self.root)])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("--expected-external-cohort-pin-sha256", stderr.getvalue())
        self.assertIn("--expected-trust-root-der-sha256", stderr.getvalue())

    def test_cli_outputs_only_non_authorizing_summary(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = main([
                "--bundle-dir", str(self.root),
                "--expected-external-cohort-pin-sha256", self.expected_pin,
                "--expected-trust-root-der-sha256", self.expected_root,
            ])
        self.assertEqual(result, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["authorized_forwards"], 0)
        self.assertFalse(report["model_load_permitted"])
        self.assertFalse(report["forward_source_authenticated"])
        self.assertFalse(report["attention_capture_authenticated"])
        self.assertNotIn("review_token", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
