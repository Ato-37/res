from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from audit_nextgqa_review_release_order import (
    _ATTESTATION_DOMAIN,
    _ROSTER_DOMAIN,
    _STATEMENT_ID,
    _attestation_body,
    verify_review_release_order,
)
from nextgqa_pre_review_integrity_receipt import (
    REQUIRED_ARTIFACT_ROLES,
    _canonical,
    _public_der,
    _sha,
    _sign,
    build_integrity_receipt,
    _validate_inventory,
)


class ReviewReleaseOrderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp.name)
        self.keys: dict[str, tuple[bytes, bytes]] = {}
        for name in ("root", "owner", "claim", "frame", "custodian", "witness"):
            private = self.root_dir / f"{name}.private.pem"
            public = self.root_dir / f"{name}.public.pem"
            subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519",
                            "-out", str(private)], check=True, capture_output=True)
            subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout",
                            "-out", str(public)], check=True, capture_output=True)
            private.chmod(0o600)
            public.chmod(0o600)
            self.keys[name] = (private.read_bytes(), public.read_bytes())
        self.root_pin = _sha(_public_der(self.keys["root"][1]))
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
        self.integrity_roster_raw = self._integrity_roster()
        self.release_roster_raw = self._release_roster()
        self.receipt_raw = build_integrity_receipt(
            roster_raw=self.integrity_roster_raw,
            inventory_raw=self.inventory_raw, artifact_sources=self.artifacts,
            root_public_key_pem=self.keys["root"][1],
            expected_root_public_der_sha256=self.root_pin,
            owner_key_id="owner-key", owner_private_key_pem=self.keys["owner"][0],
            owner_public_key_pem=self.keys["owner"][1],
            created_at_utc="2026-10-17T12:00:00Z")
        self.ceremony = "synthetic-ceremony-001"
        self.attestation_raw = self._attestation(self.ceremony)

    def tearDown(self):
        self.temp.cleanup()

    def _signed_roster(self, *, purpose, roster_id, entries, domain):
        body = {"schema_version": 1, "purpose": purpose,
                "roster_id": roster_id, "keys": entries}
        sig = _sign(self.keys["root"][0], self.keys["root"][1],
                    domain + _canonical(body))
        return _canonical({**body, "signature_ed25519_base64":
                           base64.b64encode(sig).decode("ascii")}) + b"\n"

    def _integrity_roster(self):
        entries = []
        for role, name, key_id in (
                ("freeze_owner", "owner", "owner-key"),
                ("claim_reviewer", "claim", "claim-key"),
                ("frame_reviewer", "frame", "frame-key")):
            pem = self.keys[name][1]
            entries.append({"role": role, "key_id": key_id,
                            "public_key_pem_base64": base64.b64encode(pem).decode(),
                            "public_key_der_sha256": _sha(_public_der(pem))})
        return self._signed_roster(
            purpose="nextgqa_pre_review_trust_roster_v1",
            roster_id="integrity-roster", entries=entries,
            domain=b"nextgqa_pre_review_roster_v1\0")

    def _release_roster(self, *, aliases=None, names=None):
        aliases = aliases or {}
        names = names or {"freeze_owner": "owner", "custodian": "custodian",
                          "claim_reviewer": "claim", "frame_reviewer": "frame",
                          "release_witness": "witness"}
        entries = []
        key_ids = {"freeze_owner": "owner-key", "custodian": "custodian-key",
                   "claim_reviewer": "claim-key", "frame_reviewer": "frame-key",
                   "release_witness": "witness-key"}
        for role, name in names.items():
            key_id = aliases.get(role, key_ids[role])
            pem = self.keys[name][1]
            entries.append({"role": role, "key_id": key_id,
                            "public_key_pem_base64": base64.b64encode(pem).decode(),
                            "public_key_der_sha256": _sha(_public_der(pem))})
        return self._signed_roster(
            purpose="nextgqa_review_release_roster_v1",
            roster_id="release-roster", entries=entries,
            domain=_ROSTER_DOMAIN)

    def _attestation(self, ceremony_id, *, receipt_raw=None, inventory_raw=None,
                     roster_raw=None, release_roster_raw=None,
                     artifacts=None, signer="witness"):
        receipt_raw = receipt_raw or self.receipt_raw
        inventory_raw = inventory_raw or self.inventory_raw
        roster_raw = roster_raw or self.integrity_roster_raw
        release_roster_raw = release_roster_raw or self.release_roster_raw
        receipt, bindings = _validate_inventory(inventory_raw)
        # Receipt is parsed separately because its artifact_bindings are the
        # exact objects signed by the owner, not rebuilt from synthetic labels.
        del receipt
        receipt_obj = json.loads(receipt_raw)
        release_roster = json.loads(release_roster_raw)
        body = _attestation_body(
            ceremony_id=ceremony_id, receipt_raw=receipt_raw,
            integrity_roster_raw=roster_raw,
            release_roster_raw=release_roster_raw,
            inventory_raw=inventory_raw,
            artifact_bindings=receipt_obj["artifact_bindings"],
            release_roster=release_roster)
        sig = _sign(self.keys[signer][0], self.keys[signer][1],
                    _ATTESTATION_DOMAIN + _canonical(body))
        return _canonical({**body, "witness_signature_ed25519_base64":
                           base64.b64encode(sig).decode()}) + b"\n"

    def _verify(self, **overrides):
        args = {"receipt_raw": self.receipt_raw,
                "integrity_roster_raw": self.integrity_roster_raw,
                "release_roster_raw": self.release_roster_raw,
                "inventory_raw": self.inventory_raw,
                "artifact_sources": self.artifacts,
                "root_public_key_pem": self.keys["root"][1],
                "expected_root_public_der_sha256": self.root_pin,
                "attestation_raw": self.attestation_raw,
                "expected_ceremony_id": self.ceremony}
        args.update(overrides)
        return verify_review_release_order(**args)

    def test_positive_is_procedural_opening_only(self):
        report = self._verify()
        self.assertEqual(report["status"], "review_release_order_witness_attestation_valid")
        self.assertEqual(report["scope"], "procedural_review_only_permission")
        self.assertTrue(report["package_opening_permitted"])
        self.assertFalse(report["human_identity_verified"])
        self.assertFalse(report["cryptographic_chronology_verified"])
        self.assertFalse(report["ceremony_id_replay_or_uniqueness_tracked"])
        self.assertFalse(report["model_load_permitted"])
        self.assertFalse(report["score_ready"])
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "json_duplicate_key"):
            self._verify(attestation_raw=b'{"schema_version":1,"schema_version":1}')

    def test_wrong_root_pin_and_wrong_role_roster_rejected(self):
        with self.assertRaisesRegex(ValueError, "external_trust_root_pin_mismatch"):
            self._verify(expected_root_public_der_sha256="0" * 64)
        swapped = self._release_roster(names={
            "freeze_owner": "claim", "custodian": "custodian",
            "claim_reviewer": "owner", "frame_reviewer": "frame",
            "release_witness": "witness"})
        with self.assertRaisesRegex(ValueError, "release_roster_original_role_key_mismatch"):
            self._verify(release_roster_raw=swapped)

    def test_unhashable_role_and_key_ids_fail_with_stable_error(self):
        roster = json.loads(self.release_roster_raw)
        for field, value in (("role", ["release_witness"]),
                             ("key_id", {"bad": "id"})):
            entries = json.loads(json.dumps(roster["keys"]))
            entries[0][field] = value
            malformed = self._signed_roster(
                purpose="nextgqa_review_release_roster_v1",
                roster_id="malformed-roster", entries=entries,
                domain=_ROSTER_DOMAIN)
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "release_roster_role_set_invalid"):
                self._verify(release_roster_raw=malformed)

    def test_witness_must_be_distinct_from_every_other_role(self):
        roster = self._release_roster(names={
            "freeze_owner": "owner", "custodian": "custodian",
            "claim_reviewer": "claim", "frame_reviewer": "frame",
            "release_witness": "owner"})
        with self.assertRaisesRegex(ValueError, "release_roster_role_keys_must_be_distinct|release_witness_must_be_independent"):
            self._verify(release_roster_raw=roster)

    def test_owner_custodian_alias_is_allowed_but_reviewers_are_separate(self):
        roster = self._release_roster(
            names={"freeze_owner": "owner", "custodian": "owner",
                   "claim_reviewer": "claim", "frame_reviewer": "frame",
                   "release_witness": "witness"},
            aliases={"custodian": "owner-key"})
        # The signed attestation names the exact role IDs and roster hash.
        attest = self._attestation(self.ceremony, release_roster_raw=roster)
        report = self._verify(release_roster_raw=roster, attestation_raw=attest)
        self.assertTrue(report["package_opening_permitted"])

    def test_custodian_cannot_be_reviewer(self):
        roster = self._release_roster(names={
            "freeze_owner": "owner", "custodian": "claim",
            "claim_reviewer": "claim", "frame_reviewer": "frame",
            "release_witness": "witness"}, aliases={"custodian": "claim-key"})
        with self.assertRaisesRegex(ValueError, "release_roster_role_set_invalid|release_roster_role_keys_must_be_distinct"):
            self._verify(release_roster_raw=roster)

    def test_inventory_receipt_and_artifact_bytes_are_bound(self):
        altered_inventory = self.inventory_raw.rstrip(b"\n") + b" \n"
        with self.assertRaisesRegex(ValueError, "receipt_inventory_hash_mismatch"):
            self._verify(inventory_raw=altered_inventory)
        changed = dict(self.artifacts)
        changed["review_policy_v6"] = b"changed"
        with self.assertRaisesRegex(ValueError, "artifact_bytes_hash_mismatch"):
            self._verify(artifact_sources=changed)

    def test_signature_tampering_and_missing_attestation_rejected(self):
        obj = json.loads(self.attestation_raw)
        signature = bytearray(base64.b64decode(obj["witness_signature_ed25519_base64"]))
        signature[-1] ^= 0x80
        obj["witness_signature_ed25519_base64"] = base64.b64encode(signature).decode()
        with self.assertRaisesRegex(ValueError, "release_witness_signature_invalid"):
            self._verify(attestation_raw=_canonical(obj))
        with self.assertRaisesRegex(ValueError, "release_attestation_json_invalid"):
            self._verify(attestation_raw=b"")

    def test_replay_or_mismatched_ceremony_and_receipt_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "ceremony_id_replay_or_mismatch"):
            self._verify(expected_ceremony_id="different-ceremony")
        other = self._attestation("another-ceremony")
        with self.assertRaisesRegex(ValueError, "ceremony_id_replay_or_mismatch"):
            self._verify(attestation_raw=other)

    def test_private_output_is_exclusive_and_mode_safe(self):
        from nextgqa_pre_review_integrity_receipt import _write_private_output
        out_dir = self.root_dir / "private"
        out_dir.mkdir(mode=0o700)
        out_dir.chmod(0o700)
        output = out_dir / "report.json"
        _write_private_output(output, b'{"safe":true}\n')
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "output_already_exists"):
            _write_private_output(output, b"overwrite")
        bad_dir = self.root_dir / "publicish"
        bad_dir.mkdir(mode=0o755)
        bad_dir.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "output_parent_must_be_private_directory"):
            _write_private_output(bad_dir / "out.json", b"no")

    def test_cli_rejects_insecure_input_mode_before_writing(self):
        # Verify the shared private reader's enforced mode at the CLI boundary.
        path = self.root_dir / "root.pem"
        path.write_bytes(self.keys["root"][1])
        path.chmod(0o644)
        from audit_nextgqa_review_release_order import main
        rc = main(["--root-public-key", str(path),
                   "--root-public-der-sha256", self.root_pin,
                   "--integrity-roster", str(path), "--release-roster", str(path),
                   "--inventory", str(path), "--receipt", str(path),
                   "--attestation", str(path), "--expected-ceremony-id", self.ceremony,
                   "--output", str(self.root_dir / "no" / "report.json")])
        self.assertEqual(rc, 2)
        self.assertFalse((self.root_dir / "no").exists())


if __name__ == "__main__":
    unittest.main()
