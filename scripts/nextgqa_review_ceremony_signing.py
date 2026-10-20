#!/usr/bin/env python3
"""Locally sign review-only rosters and a witnessed package handoff.

This utility never creates keys, authenticates key holders, or grants model or
scoring permission. The release roster binds a distinct capture-operator key.
A witness signature records the rostered key's statement;
it cannot prove identity, chronology, or that an actual ceremony occurred.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

from audit_nextgqa_review_release_order import (
    RELEASE_ROLES, _ATTESTATION_DOMAIN, _ROSTER_DOMAIN, _attestation_body,
    _validate_release_roster, _receipt_bindings,
)
from nextgqa_pre_review_integrity_receipt import (
    ROLES, _ROSTER_DOMAIN as _INTEGRITY_DOMAIN, _MAX_JSON, _canonical,
    _parse_artifact_arguments, _public_der, _read_file, _sha, _sign,
    _validate_roster, _write_private_output, ReceiptError,
    verify_integrity_receipt,
)


def _signed_roster(*, purpose: str, roster_id: str, entries: list[dict[str, str]],
                   domain: bytes, root_private: bytes, root_public: bytes) -> bytes:
    body: dict[str, Any] = {"schema_version": 1, "purpose": purpose,
                            "roster_id": roster_id, "keys": entries}
    signature = _sign(root_private, root_public, domain + _canonical(body))
    return _canonical({**body, "signature_ed25519_base64":
                       base64.b64encode(signature).decode("ascii")}) + b"\n"


def _entries(args, roles: tuple[str, ...]) -> list[dict[str, str]]:
    if len(args.role_key) != len(roles):
        raise ReceiptError("roster_role_set_invalid")
    by_role: dict[str, tuple[str, Path]] = {}
    for value in args.role_key:
        role, sep, rest = value.partition("=")
        key_id, sep2, filename = rest.partition("=")
        if not sep or not sep2 or role not in roles or role in by_role:
            raise ReceiptError("roster_key_argument_invalid")
        by_role[role] = (key_id, Path(filename))
    if set(by_role) != set(roles):
        raise ReceiptError("roster_role_set_invalid")
    entries = []
    for role in roles:
        key_id, path = by_role[role]
        public = _read_file(path, private=True, max_bytes=64_000,
                            failure_code="role_public_key_file_invalid")
        entries.append({"role": role, "key_id": key_id,
                        "public_key_pem_base64": base64.b64encode(public).decode("ascii"),
                        "public_key_der_sha256": _sha(_public_der(public))})
    return entries


def _read_root(args) -> tuple[bytes, bytes, str]:
    public = _read_file(args.root_public_key, private=True, max_bytes=64_000,
                        failure_code="trust_root_key_file_invalid")
    private = _read_file(args.root_private_key, private=True, max_bytes=64_000,
                         failure_code="root_private_key_file_invalid")
    pin = args.root_public_der_sha256
    if _sha(_public_der(public)) != pin:
        raise ReceiptError("external_trust_root_pin_mismatch")
    return public, private, pin


def _roster_command(args, release: bool) -> int:
    root_public, root_private, pin = _read_root(args)
    if release:
        integrity = _read_file(args.integrity_roster, private=True, max_bytes=_MAX_JSON,
                               failure_code="roster_file_invalid")
        _validate_roster(integrity, root_public, pin)
        raw = _signed_roster(purpose="nextgqa_review_release_roster_v1",
                             roster_id=args.roster_id,
                             entries=_entries(args, RELEASE_ROLES),
                             domain=_ROSTER_DOMAIN, root_private=root_private,
                             root_public=root_public)
        _validate_release_roster(raw, root_public, pin, integrity)
    else:
        raw = _signed_roster(purpose="nextgqa_pre_review_trust_roster_v1",
                             roster_id=args.roster_id,
                             entries=_entries(args, ROLES), domain=_INTEGRITY_DOMAIN,
                             root_private=root_private, root_public=root_public)
        _validate_roster(raw, root_public, pin)
    _write_private_output(args.output, raw)
    print(json.dumps({"status": "root_signed_roster_created",
                      "roster_type": "release" if release else "integrity",
                      "key_generation_performed": False,
                      "identity_verified": False, "execution_authorization": "not_granted",
                      "scoring_permitted": False, "model_load_permitted": False,
                      "output_mode": "0600"}, sort_keys=True))
    return 0


def _witness(args) -> int:
    if not args.witness_confirms_unopened_preopen_handoff:
        raise ReceiptError("explicit_witness_confirmation_required")
    root_public = _read_file(args.root_public_key, private=True, max_bytes=64_000,
                             failure_code="trust_root_key_file_invalid")
    pin = args.root_public_der_sha256
    if _sha(_public_der(root_public)) != pin:
        raise ReceiptError("external_trust_root_pin_mismatch")
    integrity = _read_file(args.integrity_roster, private=True, max_bytes=_MAX_JSON,
                           failure_code="roster_file_invalid")
    release = _read_file(args.release_roster, private=True, max_bytes=_MAX_JSON,
                         failure_code="release_roster_file_invalid")
    inventory = _read_file(args.inventory, private=True, max_bytes=_MAX_JSON,
                           failure_code="inventory_file_invalid")
    receipt = _read_file(args.receipt, private=True, max_bytes=_MAX_JSON,
                         failure_code="receipt_file_invalid")
    artifacts = _parse_artifact_arguments(args.artifact)
    verify_integrity_receipt(receipt_raw=receipt, roster_raw=integrity,
                             inventory_raw=inventory, artifact_sources=artifacts,
                             root_public_key_pem=root_public,
                             expected_root_public_der_sha256=pin)
    release_obj, role_keys = _validate_release_roster(release, root_public, pin, integrity)
    witness_private = _read_file(args.witness_private_key, private=True,
                                  max_bytes=64_000,
                                  failure_code="witness_private_key_file_invalid")
    body = _attestation_body(ceremony_id=args.ceremony_id, receipt_raw=receipt,
                             integrity_roster_raw=integrity,
                             release_roster_raw=release, inventory_raw=inventory,
                             artifact_bindings=_receipt_bindings(receipt)[1],
                             release_roster=release_obj)
    signature = _sign(witness_private, role_keys["release_witness"],
                      _ATTESTATION_DOMAIN + _canonical(body))
    raw = _canonical({**body, "witness_signature_ed25519_base64":
                      base64.b64encode(signature).decode("ascii")}) + b"\n"
    _write_private_output(args.output, raw)
    print(json.dumps({"status": "witness_statement_signed",
                      "verified_artifact_count": len(artifacts),
                      "identity_verified": False, "chronology_verified": False,
                      "ceremony_uniqueness_tracked": False,
                      "execution_authorization": "not_granted",
                      "scoring_permitted": False, "model_load_permitted": False,
                      "output_mode": "0600"}, sort_keys=True))
    return 0


def _base(command, root_private=True):
    command.add_argument("--root-public-key", required=True, type=Path)
    command.add_argument("--root-public-der-sha256", required=True,
                         help="fingerprint obtained through an independent channel")
    if root_private:
        command.add_argument("--root-private-key", required=True, type=Path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sign-integrity-roster", "sign-release-roster"):
        cmd = sub.add_parser(name)
        _base(cmd)
        cmd.add_argument("--roster-id", required=True)
        cmd.add_argument("--role-key", action="append", required=True,
                         metavar="ROLE=KEY_ID=PUBLIC_KEY_PEM")
        cmd.add_argument("--integrity-roster", type=Path,
                         help="required for release roster")
        cmd.add_argument("--output", required=True, type=Path)
    cmd = sub.add_parser("sign-witness")
    _base(cmd, root_private=False)
    cmd.add_argument("--integrity-roster", required=True, type=Path)
    cmd.add_argument("--release-roster", required=True, type=Path)
    cmd.add_argument("--inventory", required=True, type=Path)
    cmd.add_argument("--receipt", required=True, type=Path)
    cmd.add_argument("--artifact", action="append", required=True, metavar="ROLE=PATH")
    cmd.add_argument("--witness-private-key", required=True, type=Path)
    cmd.add_argument("--ceremony-id", required=True,
                     help="fresh ID selected for this ceremony")
    cmd.add_argument("--witness-confirms-unopened-preopen-handoff", action="store_true",
                     help="explicitly affirm the witness statement being signed")
    cmd.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "sign-integrity-roster":
            return _roster_command(args, False)
        if args.command == "sign-release-roster":
            if args.integrity_roster is None:
                raise ReceiptError("integrity_roster_required")
            return _roster_command(args, True)
        return _witness(args)
    except ReceiptError as exc:
        print(json.dumps({"status": "blocked", "failure_code": str(exc),
                          "execution_authorization": "not_granted",
                          "scoring_permitted": False, "model_load_permitted": False},
                         sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "blocked", "failure_code": "internal_error",
                          "execution_authorization": "not_granted",
                          "scoring_permitted": False, "model_load_permitted": False},
                         sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
