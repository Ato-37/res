#!/usr/bin/env python3
"""Audit a witnessed, review-only package handoff without opening package data.

This tool combines the existing integrity-only receipt with a root-pinned
release roster and a detached witness signature over the exact receipt,
inventory, rosters, and artifact bindings. A positive report is only a
procedural review-only permission; it does not prevent out-of-band opening or
verify real-world identities/chronology. Local timestamps are not accepted as
release-order evidence. It never permits model loading or scoring.

The API accepts synthetic bytes for tests and local file paths via the CLI. It
does not create keys, sign attestations, open review pages, or access video
contents except to hash the inventory-bound files.
"""
from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from nextgqa_pre_review_integrity_receipt import (
    PRIVATE_ARTIFACT_ROLES,
    REQUIRED_ARTIFACT_ROLES,
    ReceiptError,
    _KEY_ID,
    _MAX_JSON,
    _canonical,
    _decode_signature,
    _parse,
    _public_der,
    _sha,
    _verify_signature,
    _validate_roster,
    _write_private_output,
    _read_file,
    _parse_artifact_arguments,
    verify_integrity_receipt,
)


RELEASE_ROLES = ("freeze_owner", "custodian", "claim_reviewer",
                 "frame_reviewer", "release_witness")
_ROSTER_DOMAIN = b"nextgqa_review_release_roster_v1\0"
_ATTESTATION_DOMAIN = b"nextgqa_review_release_order_attestation_v1\0"
_CEREMONY_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_STATEMENT_ID = "witnessed_pre_review_handoff_before_first_opening_v1"


def _fail(code: str) -> None:
    raise ReceiptError(code)


def _validate_release_roster(roster_raw: bytes, root_public_key_pem: bytes,
                             expected_root_public_der_sha256: str,
                             integrity_roster_raw: bytes) -> tuple[dict, dict[str, bytes]]:
    root_der = _public_der(root_public_key_pem)
    if (not isinstance(expected_root_public_der_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_root_public_der_sha256)
            or _sha(root_der) != expected_root_public_der_sha256):
        _fail("external_trust_root_pin_mismatch")
    roster = _parse(roster_raw, "release_roster_json_invalid")
    if (not isinstance(roster, dict)
            or set(roster) != {"schema_version", "purpose", "roster_id", "keys",
                               "signature_ed25519_base64"}
            or type(roster.get("schema_version")) is not int
            or roster.get("schema_version") != 1
            or roster.get("purpose") != "nextgqa_review_release_roster_v1"
            or not isinstance(roster.get("roster_id"), str)
            or not _KEY_ID.fullmatch(roster["roster_id"])):
        _fail("release_roster_schema_invalid")
    sig = _decode_signature(roster["signature_ed25519_base64"],
                            "release_roster_signature_invalid")
    body = {key: value for key, value in roster.items()
            if key != "signature_ed25519_base64"}
    if not _verify_signature(root_public_key_pem, sig,
                             _ROSTER_DOMAIN + _canonical(body)):
        _fail("root_signed_release_roster_signature_invalid")
    entries = roster.get("keys")
    if not isinstance(entries, list) or len(entries) != len(RELEASE_ROLES):
        _fail("release_roster_role_set_invalid")
    public_by_role: dict[str, bytes] = {}
    key_id_by_role: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    seen_ids: set[str] = set()
    for entry in entries:
        if (not isinstance(entry, dict)
                or set(entry) != {"role", "key_id", "public_key_pem_base64",
                                  "public_key_der_sha256"}):
            _fail("release_roster_key_entry_invalid")
        role, key_id = entry.get("role"), entry.get("key_id")
        if (not isinstance(role, str) or role not in RELEASE_ROLES
                or role in public_by_role):
            _fail("release_roster_role_set_invalid")
        if not isinstance(key_id, str) or not _KEY_ID.fullmatch(key_id):
            _fail("release_roster_role_set_invalid")
        role_id_reuse_ok = ({role, next((prior for prior, prior_id in key_id_by_role.items()
                                         if prior_id == key_id), "")} ==
                            {"freeze_owner", "custodian"})
        if key_id in seen_ids and not role_id_reuse_ok:
            _fail("release_roster_role_set_invalid")
        try:
            pem = base64.b64decode(entry["public_key_pem_base64"], validate=True)
        except (ValueError, TypeError):
            _fail("release_roster_public_key_encoding_invalid")
        der = _public_der(pem)
        digest = _sha(der)
        fingerprint_reuse_ok = ({role, next((prior for prior, prior_digest in fingerprints.items()
                                             if prior_digest == digest), "")} ==
                                {"freeze_owner", "custodian"})
        if entry.get("public_key_der_sha256") != digest:
            _fail("release_roster_public_key_hash_mismatch")
        if digest in fingerprints.values() and not fingerprint_reuse_ok:
            _fail("release_roster_role_keys_must_be_distinct")
        public_by_role[role] = pem
        key_id_by_role[role] = key_id
        fingerprints[role] = digest
        seen_ids.add(key_id)
    if set(public_by_role) != set(RELEASE_ROLES):
        _fail("release_roster_role_set_invalid")

    # The runbook permits a freeze owner to double as custodian. That alias
    # must identify the same key in both fields; otherwise a repeated key ID
    # or fingerprint could ambiguously name different role credentials.
    owner_id, custodian_id = (key_id_by_role["freeze_owner"],
                              key_id_by_role["custodian"])
    owner_fingerprint = fingerprints["freeze_owner"]
    custodian_fingerprint = fingerprints["custodian"]
    if ((owner_id == custodian_id) !=
            (owner_fingerprint == custodian_fingerprint)):
        _fail("release_roster_owner_custodian_alias_mismatch")

    # Reuse the original three-role, root-signed roster as the identity source
    # for its roles; the release roster adds explicit custody and witness.
    _, original_keys = _validate_roster(
        integrity_roster_raw, root_public_key_pem,
        expected_root_public_der_sha256)
    for role in ("freeze_owner", "claim_reviewer", "frame_reviewer"):
        if _public_der(public_by_role[role]) != _public_der(original_keys[role]):
            _fail("release_roster_original_role_key_mismatch")

    # Custody may be assigned to the freeze owner as the runbook permits, but
    # never to either reviewer. The witness must be an independent role/key.
    if key_id_by_role["custodian"] in {
            key_id_by_role["claim_reviewer"], key_id_by_role["frame_reviewer"]}:
        _fail("custodian_reviewer_role_collision")
    if fingerprints["custodian"] in {
            fingerprints["claim_reviewer"], fingerprints["frame_reviewer"]}:
        _fail("custodian_reviewer_key_collision")
    witness = "release_witness"
    if (key_id_by_role[witness] in {key_id_by_role[r] for r in RELEASE_ROLES if r != witness}
            or fingerprints[witness] in {fingerprints[r] for r in RELEASE_ROLES if r != witness}):
        _fail("release_witness_must_be_independent")
    return roster, public_by_role


def _receipt_bindings(receipt_raw: bytes) -> tuple[dict, list[dict[str, Any]]]:
    receipt = _parse(receipt_raw, "receipt_json_invalid")
    bindings = receipt.get("artifact_bindings") if isinstance(receipt, dict) else None
    if (not isinstance(bindings, list)
            or {row.get("role") for row in bindings if isinstance(row, dict)}
            != set(REQUIRED_ARTIFACT_ROLES)):
        _fail("receipt_artifact_bindings_invalid")
    return receipt, bindings


def _attestation_body(*, ceremony_id: str, receipt_raw: bytes,
                      integrity_roster_raw: bytes, release_roster_raw: bytes,
                      inventory_raw: bytes, artifact_bindings: list[dict[str, Any]],
                      release_roster: dict) -> dict[str, Any]:
    if not isinstance(ceremony_id, str) or not _CEREMONY_ID.fullmatch(ceremony_id):
        _fail("ceremony_id_invalid")
    key_ids = {entry["role"]: entry["key_id"] for entry in release_roster["keys"]}
    return {
        "schema_version": 1,
        "purpose": "nextgqa_review_release_order_witness_attestation_v1",
        "ceremony_id": ceremony_id,
        "integrity_receipt_sha256": _sha(receipt_raw),
        "integrity_roster_sha256": _sha(integrity_roster_raw),
        "release_roster_sha256": _sha(release_roster_raw),
        "inventory_sha256": _sha(inventory_raw),
        "artifact_bindings": artifact_bindings,
        "freeze_owner_key_id": key_ids["freeze_owner"],
        "custodian_key_id": key_ids["custodian"],
        "witness_key_id": key_ids["release_witness"],
        "statement_id": _STATEMENT_ID,
        "witness_attests_review_had_not_started": True,
        "witness_attests_exact_package_and_integrity_receipt_were_established": True,
        "witness_attests_handoff_preceded_first_opening": True,
        "timestamp_semantics": "no_clock_field_used_as_release_order_evidence",
    }


def verify_review_release_order(*, receipt_raw: bytes, integrity_roster_raw: bytes,
                                release_roster_raw: bytes, inventory_raw: bytes,
                                artifact_sources: Mapping[str, bytes | str | Path],
                                root_public_key_pem: bytes,
                                expected_root_public_der_sha256: str,
                                attestation_raw: bytes,
                                expected_ceremony_id: str) -> dict[str, Any]:
    """Verify exact candidate bytes and witnessed handoff; never authorize scoring."""
    # This confirms the freeze-owner signature, external root pin, unchanged
    # candidate inventory, and exact artifact bytes at release authorization.
    integrity_report = verify_integrity_receipt(
        receipt_raw=receipt_raw, roster_raw=integrity_roster_raw,
        inventory_raw=inventory_raw, artifact_sources=artifact_sources,
        root_public_key_pem=root_public_key_pem,
        expected_root_public_der_sha256=expected_root_public_der_sha256)
    release_roster, role_keys = _validate_release_roster(
        release_roster_raw, root_public_key_pem,
        expected_root_public_der_sha256, integrity_roster_raw)
    receipt, bindings = _receipt_bindings(receipt_raw)
    attestation = _parse(attestation_raw, "release_attestation_json_invalid")
    sig_key = "witness_signature_ed25519_base64"
    expected = _attestation_body(
        ceremony_id=(attestation.get("ceremony_id") if isinstance(attestation, dict) else None),
        receipt_raw=receipt_raw, integrity_roster_raw=integrity_roster_raw,
        release_roster_raw=release_roster_raw, inventory_raw=inventory_raw,
        artifact_bindings=bindings, release_roster=release_roster)
    if (not isinstance(expected_ceremony_id, str)
            or not _CEREMONY_ID.fullmatch(expected_ceremony_id)
            or expected["ceremony_id"] != expected_ceremony_id):
        _fail("ceremony_id_replay_or_mismatch")
    if not isinstance(attestation, dict) or set(attestation) != set(expected) | {sig_key}:
        _fail("release_attestation_schema_invalid")
    body = {key: value for key, value in attestation.items() if key != sig_key}
    if body != expected:
        _fail("release_attestation_binding_or_statement_mismatch")
    signature = _decode_signature(attestation.get(sig_key),
                                  "release_witness_signature_invalid")
    if not _verify_signature(role_keys["release_witness"], signature,
                             _ATTESTATION_DOMAIN + _canonical(body)):
        _fail("release_witness_signature_invalid")
    return {
        "status": "review_release_order_witness_attestation_valid",
        "scope": "procedural_review_only_permission",
        "ceremony_id": body["ceremony_id"],
        "verified_artifact_count": integrity_report["verified_artifact_count"],
        "trust_root_pin_matched": True,
        "root_signed_release_roster_valid": True,
        "freeze_owner_signature_valid": True,
        "release_witness_signature_valid": True,
        "witness_statement_signer_key_id": body["witness_key_id"],
        "ceremony_id_replay_or_uniqueness_tracked": False,
        "human_identity_verified": False,
        "cryptographic_chronology_verified": False,
        "local_timestamp_authoritative": False,
        "package_opening_permitted": True,
        "permission_semantics": (
            "procedural_only_cannot_prevent_out_of_band_opening"),
        "review_completed": False,
        "score_ready": False,
        "model_load_permitted": False,
        "execution_authorization": "not_granted",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-public-key", required=True, type=Path)
    parser.add_argument("--root-public-der-sha256", required=True)
    parser.add_argument("--integrity-roster", required=True, type=Path)
    parser.add_argument("--release-roster", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--expected-ceremony-id", required=True,
                        help="fresh ceremony ID independently selected for this release check")
    parser.add_argument("--artifact", action="append", default=[], metavar="ROLE=PATH")
    parser.add_argument("--output", required=True, type=Path,
                        help="new JSON report under an existing mode-0700 directory")
    args = parser.parse_args(argv)
    try:
        root = _read_file(args.root_public_key, private=True, max_bytes=64_000,
                          failure_code="trust_root_key_file_invalid")
        integrity_roster = _read_file(args.integrity_roster, private=True,
                                      max_bytes=_MAX_JSON, failure_code="roster_file_invalid")
        release_roster = _read_file(args.release_roster, private=True,
                                    max_bytes=_MAX_JSON, failure_code="release_roster_file_invalid")
        inventory = _read_file(args.inventory, private=True, max_bytes=_MAX_JSON,
                               failure_code="inventory_file_invalid")
        receipt = _read_file(args.receipt, private=True, max_bytes=_MAX_JSON,
                             failure_code="receipt_file_invalid")
        attestation = _read_file(args.attestation, private=True, max_bytes=_MAX_JSON,
                                 failure_code="release_attestation_file_invalid")
        artifacts = _parse_artifact_arguments(args.artifact)
        report = verify_review_release_order(
            receipt_raw=receipt, integrity_roster_raw=integrity_roster,
            release_roster_raw=release_roster, inventory_raw=inventory,
            artifact_sources=artifacts, root_public_key_pem=root,
            expected_root_public_der_sha256=args.root_public_der_sha256,
            attestation_raw=attestation,
            expected_ceremony_id=args.expected_ceremony_id)
        raw = (json.dumps(report, sort_keys=True, ensure_ascii=True, indent=2) + "\n").encode()
        _write_private_output(args.output, raw)
        print(json.dumps({key: report[key] for key in (
            "status", "scope", "verified_artifact_count", "package_opening_permitted",
            "model_load_permitted", "score_ready", "execution_authorization")}, sort_keys=True))
        return 0
    except ReceiptError as exc:
        print(json.dumps({"status": "blocked", "failure_code": str(exc),
                          "package_opening_permitted": False,
                          "model_load_permitted": False,
                          "execution_authorization": "not_granted"}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "blocked", "failure_code": "internal_error",
                          "package_opening_permitted": False,
                          "model_load_permitted": False,
                          "execution_authorization": "not_granted"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
