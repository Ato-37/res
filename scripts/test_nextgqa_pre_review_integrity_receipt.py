from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from build_nextgqa_pre_review_byte_inventory import (
    CODE_FILES, PRIVATE_FILES, PUBLIC_OR_SOURCE_FILES,
)
from nextgqa_pre_review_integrity_receipt import (
    PACKAGE_ROLES,
    PRIVATE_ARTIFACT_ROLES,
    REQUIRED_ARTIFACT_ROLES,
    ReceiptError,
    _MAX_JSON,
    _canonical,
    _hash_source,
    _memfd,
    _public_der,
    _sha,
    _sign,
    _write_private_output,
    build_integrity_receipt,
    main,
    verify_integrity_receipt,
)


class SignedReviewReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.keys = {}
        for name in ("root", "owner", "claim", "frame"):
            private = self.root / f"{name}.private.pem"
            public = self.root / f"{name}.public.pem"
            subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519",
                            "-out", str(private)], check=True, capture_output=True)
            subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout",
                            "-out", str(public)], check=True, capture_output=True)
            private.chmod(0o600)
            public.chmod(0o600)
            self.keys[name] = (private.read_bytes(), public.read_bytes())
        self.artifacts = {role: ("synthetic:" + role).encode("ascii")
                          for role in REQUIRED_ARTIFACT_ROLES}
        rows = [{"role": role, "size_bytes": len(raw), "sha256": _sha(raw)}
                for role, raw in sorted(self.artifacts.items())]
        self.inventory_raw = _canonical({
            "schema_version": 1,
            "purpose": "pre_review_candidate_byte_inventory",
            "status": "candidate_only",
            "review_started": False,
            "review_completed": False,
            "joint_freeze": False,
            "score_ready": False,
            "execution_authorization": "not_granted",
            "rights_status": "unverified_local_only",
            "artifacts": rows,
            "missing_required_slots": [],
        }) + b"\n"
        self.roster_raw = self._make_roster()
        self.root_pin = _sha(_public_der(self.keys["root"][1]))

    def tearDown(self):
        self.temp.cleanup()

    def _make_roster(self):
        entries = []
        for role, name in (("freeze_owner", "owner"),
                           ("claim_reviewer", "claim"),
                           ("frame_reviewer", "frame")):
            public = self.keys[name][1]
            entries.append({
                "role": role,
                "key_id": name + "-synthetic",
                "public_key_pem_base64": base64.b64encode(public).decode("ascii"),
                "public_key_der_sha256": _sha(_public_der(public)),
            })
        body = {
            "schema_version": 1,
            "purpose": "nextgqa_pre_review_trust_roster_v1",
            "roster_id": "synthetic-roster-v1",
            "keys": entries,
        }
        signature = _sign(self.keys["root"][0], self.keys["root"][1],
                          b"nextgqa_pre_review_roster_v1\0" + _canonical(body))
        return _canonical({**body, "signature_ed25519_base64":
                           base64.b64encode(signature).decode("ascii")}) + b"\n"

    def _build(self, **overrides):
        args = {
            "roster_raw": self.roster_raw,
            "inventory_raw": self.inventory_raw,
            "artifact_sources": self.artifacts,
            "root_public_key_pem": self.keys["root"][1],
            "expected_root_public_der_sha256": self.root_pin,
            "owner_key_id": "owner-synthetic",
            "owner_private_key_pem": self.keys["owner"][0],
            "owner_public_key_pem": self.keys["owner"][1],
            "created_at_utc": "2026-10-17T12:00:00Z",
        }
        args.update(overrides)
        return build_integrity_receipt(**args)

    def test_positive_receipt_verifies_but_never_permits_opening(self):
        receipt = self._build()
        report = verify_integrity_receipt(
            receipt_raw=receipt,
            roster_raw=self.roster_raw,
            inventory_raw=self.inventory_raw,
            artifact_sources=self.artifacts,
            root_public_key_pem=self.keys["root"][1],
            expected_root_public_der_sha256=self.root_pin,
        )
        self.assertEqual(report["status"], "signature_valid_order_unverified")
        self.assertEqual(report["verified_artifact_count"], len(REQUIRED_ARTIFACT_ROLES))
        self.assertFalse(report["local_timestamp_authoritative"])
        self.assertFalse(report["chronology_verified"])
        self.assertFalse(report["package_opening_permitted"])
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_external_root_pin_is_required(self):
        with self.assertRaisesRegex(ReceiptError, "external_trust_root_pin_mismatch"):
            self._build(expected_root_public_der_sha256="0" * 64)

    def test_roster_signature_tampering_blocks(self):
        roster = json.loads(self.roster_raw)
        roster["roster_id"] = "changed-roster"
        with self.assertRaisesRegex(ReceiptError, "root_signed_roster_signature_invalid"):
            self._build(roster_raw=_canonical(roster))

    def test_receipt_signature_tampering_blocks(self):
        receipt = json.loads(self._build())
        signature = bytearray(base64.b64decode(receipt["signature_ed25519_base64"]))
        signature[0] ^= 1
        receipt["signature_ed25519_base64"] = base64.b64encode(signature).decode("ascii")
        with self.assertRaisesRegex(ReceiptError, "freeze_owner_receipt_signature_invalid"):
            verify_integrity_receipt(
                receipt_raw=_canonical(receipt), roster_raw=self.roster_raw,
                inventory_raw=self.inventory_raw, artifact_sources=self.artifacts,
                root_public_key_pem=self.keys["root"][1],
                expected_root_public_der_sha256=self.root_pin)

    def test_receipt_cannot_substitute_or_drop_artifact_bindings(self):
        for mutation in ("drop", "substitute"):
            receipt = json.loads(self._build())
            if mutation == "drop":
                receipt["artifact_bindings"].pop()
            else:
                receipt["artifact_bindings"][0]["role"] = "unlisted_role"
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    ReceiptError, "receipt_artifact_bindings_mismatch"):
                verify_integrity_receipt(
                    receipt_raw=_canonical(receipt), roster_raw=self.roster_raw,
                    inventory_raw=self.inventory_raw, artifact_sources=self.artifacts,
                    root_public_key_pem=self.keys["root"][1],
                    expected_root_public_der_sha256=self.root_pin)

    def test_root_signed_roster_rejects_reused_role_key(self):
        roster = json.loads(self.roster_raw)
        owner_key = roster["keys"][0]
        reviewer_key = roster["keys"][1]
        reviewer_key["public_key_pem_base64"] = owner_key["public_key_pem_base64"]
        reviewer_key["public_key_der_sha256"] = owner_key["public_key_der_sha256"]
        body = {key: value for key, value in roster.items()
                if key != "signature_ed25519_base64"}
        signature = _sign(self.keys["root"][0], self.keys["root"][1],
                          b"nextgqa_pre_review_roster_v1\0" + _canonical(body))
        signed_roster = _canonical({**body, "signature_ed25519_base64":
                                    base64.b64encode(signature).decode("ascii")})
        with self.assertRaisesRegex(ReceiptError, "roster_role_keys_must_be_distinct"):
            self._build(roster_raw=signed_roster)

    def test_artifact_bytes_must_match_inventory_exactly(self):
        altered = dict(self.artifacts)
        altered["review_policy_v6"] = b"wrong synthetic policy"
        with self.assertRaisesRegex(ReceiptError, "artifact_bytes_hash_mismatch"):
            self._build(artifact_sources=altered)

    def test_unknown_or_wrong_freeze_owner_key_is_rejected(self):
        with self.assertRaisesRegex(ReceiptError, "freeze_owner_key_id_not_rostered"):
            self._build(owner_key_id="unlisted-key")
        with self.assertRaisesRegex(ReceiptError, "signing_key_does_not_match_roster"):
            self._build(owner_private_key_pem=self.keys["claim"][0])

    def test_receipt_cannot_upgrade_local_time_to_trusted_chronology(self):
        receipt = json.loads(self._build())
        receipt["chronology_status"] = "trusted"
        with self.assertRaisesRegex(ReceiptError, "receipt_safety_status_invalid"):
            verify_integrity_receipt(
                receipt_raw=_canonical(receipt), roster_raw=self.roster_raw,
                inventory_raw=self.inventory_raw, artifact_sources=self.artifacts,
                root_public_key_pem=self.keys["root"][1],
                expected_root_public_der_sha256=self.root_pin)

    def test_receipt_cannot_enable_opening_scoring_or_model_loading(self):
        for field, value in (("package_opening_permitted", True),
                             ("score_ready", True),
                             ("model_load_permitted", True),
                             ("execution_authorization", "granted")):
            receipt = json.loads(self._build())
            receipt[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                    ReceiptError, "receipt_safety_status_invalid"):
                verify_integrity_receipt(
                    receipt_raw=_canonical(receipt), roster_raw=self.roster_raw,
                    inventory_raw=self.inventory_raw, artifact_sources=self.artifacts,
                    root_public_key_pem=self.keys["root"][1],
                    expected_root_public_der_sha256=self.root_pin)

    def test_missing_role_or_duplicate_json_key_fails_closed(self):
        incomplete = dict(self.artifacts)
        incomplete.pop(next(iter(PACKAGE_ROLES)))
        with self.assertRaisesRegex(ReceiptError, "artifact_source_role_set_mismatch"):
            self._build(artifact_sources=incomplete)
        with self.assertRaisesRegex(ReceiptError, "json_duplicate_key"):
            verify_integrity_receipt(
                receipt_raw=b'{"schema_version":1,"schema_version":1}',
                roster_raw=self.roster_raw, inventory_raw=self.inventory_raw,
                artifact_sources=self.artifacts, root_public_key_pem=self.keys["root"][1],
                expected_root_public_der_sha256=self.root_pin)

    def test_oversized_json_and_unsupported_artifact_source_fail_closed(self):
        with self.assertRaisesRegex(ReceiptError, "receipt_json_invalid"):
            verify_integrity_receipt(
                receipt_raw=b" " * (_MAX_JSON + 1), roster_raw=self.roster_raw,
                inventory_raw=self.inventory_raw, artifact_sources=self.artifacts,
                root_public_key_pem=self.keys["root"][1],
                expected_root_public_der_sha256=self.root_pin)
        altered = dict(self.artifacts)
        altered[next(iter(altered))] = object()
        with self.assertRaisesRegex(ReceiptError, "artifact_source_type_invalid"):
            self._build(artifact_sources=altered)

    def test_memfd_is_closed_when_write_fails(self):
        real_close = os.close
        with patch("nextgqa_pre_review_integrity_receipt.os.write",
                   side_effect=OSError("synthetic write failure")), \
                patch("nextgqa_pre_review_integrity_receipt.os.close",
                      wraps=real_close) as close:
            with self.assertRaisesRegex(ReceiptError, "openssl_input_fd_failed"):
                _memfd("synthetic-fd", b"payload")
            close.assert_called_once()

    def test_artifact_path_rejects_symlink_nonregular_and_wrong_private_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            source.write_bytes(b"synthetic")
            link = root / "link.bin"
            link.symlink_to(source)
            with self.assertRaisesRegex(ReceiptError, "artifact_open_failed"):
                _hash_source(link, "review_package_html")
            with self.assertRaisesRegex(ReceiptError, "artifact_file_invalid"):
                _hash_source(root, "review_package_html")
            source.chmod(0o644)
            with self.assertRaisesRegex(ReceiptError, "private_artifact_mode_invalid"):
                _hash_source(source, "review_package_html")
            with self.assertRaisesRegex(ReceiptError, "artifact_open_failed"):
                _hash_source("\x00", "review_package_html")

    def test_private_output_rejects_symlinked_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private = root / "private"
            private.mkdir(mode=0o700)
            link = root / "private-link"
            link.symlink_to(private, target_is_directory=True)
            with self.assertRaisesRegex(ReceiptError, "output_parent_invalid"):
                _write_private_output(link / "receipt.json", b"synthetic")

    def test_cli_build_verify_output_permissions_and_no_overwrite(self):
        private = self.root / "private"
        private.mkdir(mode=0o700)
        (private / "root.pem").write_bytes(self.keys["root"][1])
        (private / "root.pem").chmod(0o600)
        (private / "roster.json").write_bytes(self.roster_raw)
        (private / "roster.json").chmod(0o600)
        (private / "inventory.json").write_bytes(self.inventory_raw)
        (private / "inventory.json").chmod(0o600)
        (private / "owner.pem").write_bytes(self.keys["owner"][0])
        (private / "owner.pem").chmod(0o600)
        input_dir = private / "artifacts"
        input_dir.mkdir(mode=0o700)
        args = []
        for index, (role, raw) in enumerate(sorted(self.artifacts.items())):
            path = input_dir / f"artifact-{index:03d}"
            path.write_bytes(raw)
            path.chmod(0o600 if role in PRIVATE_ARTIFACT_ROLES else 0o644)
            args.extend(["--artifact", f"{role}={path}"])
        receipt_path = private / "integrity_receipt.json"
        common = ["--root-public-key", str(private / "root.pem"),
                  "--root-public-der-sha256", self.root_pin,
                  "--roster", str(private / "roster.json"),
                  "--inventory", str(private / "inventory.json"), *args]
        build_args = ["build", *common, "--owner-private-key", str(private / "owner.pem"),
                      "--created-at-utc", "2026-10-17T12:00:00Z",
                      "--output", str(receipt_path)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(build_args), 0)
        self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
        build_report = json.loads(output.getvalue())
        self.assertEqual(build_report["status"], "integrity_receipt_created_candidate_only")
        self.assertFalse(build_report["package_opening_permitted"])

        original = receipt_path.read_bytes()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(build_args), 2)
        self.assertEqual(receipt_path.read_bytes(), original)
        self.assertEqual(json.loads(output.getvalue())["failure_code"], "output_already_exists")

        verify_args = ["verify", *common, "--receipt", str(receipt_path)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(verify_args), 0)
        verify_report = json.loads(output.getvalue())
        self.assertEqual(verify_report["status"], "signature_valid_order_unverified")
        self.assertFalse(verify_report["package_opening_permitted"])

    def test_incomplete_inventory_never_signs(self):
        inventory = json.loads(self.inventory_raw)
        inventory["missing_required_slots"] = ["frame_review"]
        with self.assertRaisesRegex(ReceiptError, "inventory_not_complete_candidate"):
            self._build(inventory_raw=_canonical(inventory))


if __name__ == "__main__":
    unittest.main()
