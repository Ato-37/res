from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from nextgqa_pre_review_integrity_receipt import (
    REQUIRED_ARTIFACT_ROLES, _canonical, _public_der, _sha,
    build_integrity_receipt,
)
from nextgqa_review_ceremony_signing import main
from audit_nextgqa_review_release_order import verify_review_release_order


class CeremonySigningCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.private_dir = self.root / "private"
        self.private_dir.mkdir(mode=0o700)
        self.private_dir.chmod(0o700)
        self.keys = {}
        for name in ("root", "owner", "claim", "frame", "witness"):
            private, public = self.root / f"{name}.pem", self.root / f"{name}.pub"
            subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)], check=True, capture_output=True)
            subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)], check=True, capture_output=True)
            private.chmod(0o600); public.chmod(0o600)
            self.keys[name] = (private, public)
        self.pin = _sha(_public_der(self.keys["root"][1].read_bytes()))
        self.base = ["--root-public-key", str(self.keys["root"][1]),
                     "--root-public-der-sha256", self.pin,
                     "--root-private-key", str(self.keys["root"][0])]
        self._run(["sign-integrity-roster", *self.base, "--roster-id", "int-roster",
                   *self._role_args(("freeze_owner", "owner"), ("claim_reviewer", "claim"),
                                    ("frame_reviewer", "frame")),
                   "--output", str(self.private_dir / "integrity.json")])
        self._run(["sign-release-roster", *self.base, "--roster-id", "rel-roster",
                   "--integrity-roster", str(self.private_dir / "integrity.json"),
                   *self._role_args(("freeze_owner", "owner"), ("custodian", "owner"),
                                    ("claim_reviewer", "claim"), ("frame_reviewer", "frame"),
                                    ("release_witness", "witness")),
                   "--output", str(self.private_dir / "release.json")])
        self.artifacts = {role: ("synthetic:" + role).encode() for role in REQUIRED_ARTIFACT_ROLES}
        for role, raw in self.artifacts.items():
            path = self.private_dir / (role + ".bin"); path.write_bytes(raw); path.chmod(0o600)
        inventory = {"schema_version": 1, "purpose": "pre_review_candidate_byte_inventory",
                     "status": "candidate_only", "review_started": False,
                     "review_completed": False, "joint_freeze": False,
                     "score_ready": False, "execution_authorization": "not_granted",
                     "rights_status": "unverified_local_only",
                     "artifacts": [{"role": r, "size_bytes": len(b), "sha256": _sha(b)}
                                   for r, b in sorted(self.artifacts.items())],
                     "missing_required_slots": []}
        inv = self.private_dir / "inventory.json"; inv.write_bytes(_canonical(inventory) + b"\n"); inv.chmod(0o600)
        self.inventory = inv
        self.receipt_raw = build_integrity_receipt(
            roster_raw=(self.private_dir / "integrity.json").read_bytes(),
            inventory_raw=inv.read_bytes(), artifact_sources=self.artifacts,
            root_public_key_pem=self.keys["root"][1].read_bytes(),
            expected_root_public_der_sha256=self.pin, owner_key_id="owner",
            owner_private_key_pem=self.keys["owner"][0].read_bytes(),
            owner_public_key_pem=self.keys["owner"][1].read_bytes(),
            created_at_utc="2026-10-17T00:00:00Z")
        rec = self.private_dir / "receipt.json"; rec.write_bytes(self.receipt_raw); rec.chmod(0o600)
        self.receipt = rec

    def tearDown(self):
        self.tmp.cleanup()

    def _role_args(self, *roles):
        args = []
        for role, key in roles:
            args.extend(["--role-key", f"{role}={key}={self.keys[key][1]}"])
        return args

    def _run(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)

    def _witness_args(self, output):
        args = ["sign-witness", "--root-public-key", str(self.keys["root"][1]),
                "--root-public-der-sha256", self.pin,
                "--integrity-roster", str(self.private_dir / "integrity.json"),
                "--release-roster", str(self.private_dir / "release.json"),
                "--inventory", str(self.inventory), "--receipt", str(self.receipt),
                "--witness-private-key", str(self.keys["witness"][0]),
                "--ceremony-id", "synthetic-fresh-01", "--output", str(output)]
        for role in sorted(self.artifacts):
            args.extend(["--artifact", f"{role}={self.private_dir / (role + '.bin')}"])
        return args

    def test_rosters_and_explicit_witness_signature_verify_exact_synthetic_bytes(self):
        output = self.private_dir / "attestation.json"
        self._run(self._witness_args(output) + ["--witness-confirms-unopened-preopen-handoff"])
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        report = verify_review_release_order(
            receipt_raw=self.receipt_raw,
            integrity_roster_raw=(self.private_dir / "integrity.json").read_bytes(),
            release_roster_raw=(self.private_dir / "release.json").read_bytes(),
            inventory_raw=self.inventory.read_bytes(), artifact_sources=self.artifacts,
            root_public_key_pem=self.keys["root"][1].read_bytes(),
            expected_root_public_der_sha256=self.pin, attestation_raw=output.read_bytes(),
            expected_ceremony_id="synthetic-fresh-01")
        self.assertFalse(report["human_identity_verified"])
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_confirmation_and_exact_artifact_bytes_are_required(self):
        output = self.private_dir / "no-confirmation.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self._witness_args(output)), 2)
        changed = self.private_dir / "changed.bin"; changed.write_bytes(b"changed"); changed.chmod(0o600)
        args = self._witness_args(output) + ["--witness-confirms-unopened-preopen-handoff"]
        i = args.index("--artifact")
        args[i + 1] = f"{sorted(self.artifacts)[0]}={changed}"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 2)
        self.assertFalse(output.exists())

    def test_outputs_never_overwrite(self):
        output = self.private_dir / "existing.json"; output.write_bytes(b"keep"); output.chmod(0o600)
        args = ["sign-integrity-roster", *self.base, "--roster-id", "another",
                *self._role_args(("freeze_owner", "owner"), ("claim_reviewer", "claim"),
                                 ("frame_reviewer", "frame")), "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 2)
        self.assertEqual(output.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
