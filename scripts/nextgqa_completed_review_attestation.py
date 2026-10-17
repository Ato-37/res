#!/usr/bin/env python3
"""Create and verify a detached reviewer signature for completed v6 exports.

This additive envelope binds exact review JSON/index bytes to a reviewer key
whose DER fingerprint is pinned out of band. It does not alter the frozen v6
page, policy, review payload, or integrity index. A valid signature attests
only that the key holder signed/adopted these bytes; it does not authenticate
the human, prove viewing, validate semantics, verify media custody, or
authorize scoring. The caller must obtain the receipt digest and reviewer key
ID/fingerprint from the trusted workflow; this module does not verify a signed
role roster or the receipt itself.
"""
from __future__ import annotations

import base64
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from build_nextgqa_claim_mapping_packet import FROZEN_POLICY_SHA256
from audit_nextgqa_review_handoff import audit as audit_review_handoff
from nextgqa_pre_review_integrity_receipt import (
    ReceiptError,
    _canonical,
    _decode_signature,
    _public_der,
    _read_file,
    _sign,
    _verify_signature,
    _write_private_output,
)

MAX_JSON_BYTES = 2 * 1024 * 1024
DOMAIN = b"nextgqa_completed_review_attestation_v1\0"
POLICY_ID = "nextgqa_claim_review_v6"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_KEY_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class AttestationError(ValueError):
    """Stable failure code safe to show without echoing protected input."""


def _fail(code: str) -> None:
    raise AttestationError(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            _fail("json_duplicate_key")
        obj[key] = value
    return obj


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("json_nonfinite_value")
    if isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child)


def _parse(raw: bytes, code: str) -> Any:
    if not isinstance(raw, bytes) or len(raw) > MAX_JSON_BYTES:
        _fail(code)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=lambda _: _fail("json_nonfinite_value"))
        _reject_nonfinite(value)
        return value
    except AttestationError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        _fail(code)


def _validate_exports(*, worksheet: Any, worksheet_raw: bytes, lineage: Any,
                      lineage_raw: bytes, review: Any, review_raw: bytes,
                      review_index: Any, review_index_raw: bytes,
                      expected_lineage_sha256: str) -> None:
    if (not isinstance(expected_lineage_sha256, str)
            or not _SHA.fullmatch(expected_lineage_sha256)
            or expected_lineage_sha256 != _sha(lineage_raw)):
        _fail("lineage_hash_pin_mismatch")
    if _parse(worksheet_raw, "worksheet_json_invalid") != worksheet:
        _fail("worksheet_bytes_mismatch")
    if _parse(lineage_raw, "lineage_json_invalid") != lineage:
        _fail("lineage_bytes_mismatch")
    if _parse(review_raw, "review_json_invalid") != review:
        _fail("review_bytes_mismatch")
    if _parse(review_index_raw, "review_index_json_invalid") != review_index:
        _fail("review_index_bytes_mismatch")
    report = audit_review_handoff(
        worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw,
        review_index, review_index_raw, expected_lineage_sha256)
    if (report.get("status") != "unvalidated_handoff_structurally_valid"
            or report.get("execution_authorization") != "not_granted"
            or report.get("semantic_review_authenticated") is not False):
        _fail("review_handoff_structure_invalid")


def _validate_bindings(policy_sha256: str, pre_review_receipt_sha256: str,
                       reviewer_key_id: str) -> None:
    if policy_sha256 != FROZEN_POLICY_SHA256 or not _SHA.fullmatch(policy_sha256):
        _fail("policy_hash_invalid")
    if not isinstance(pre_review_receipt_sha256, str) or not _SHA.fullmatch(pre_review_receipt_sha256):
        _fail("pre_review_receipt_hash_invalid")
    if not isinstance(reviewer_key_id, str) or not _KEY_ID.fullmatch(reviewer_key_id):
        _fail("reviewer_key_id_invalid")


def _body(*, review_raw: bytes, review_index_raw: bytes,
          worksheet_raw: bytes, lineage_raw: bytes,
          pre_review_receipt_sha256: str, reviewer_key_id: str,
          reviewer_key_fingerprint_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "nextgqa_completed_review_bytes_adoption_attestation",
        "status": "reviewer_key_attested_unvalidated",
        "policy_id": POLICY_ID,
        "policy_sha256": FROZEN_POLICY_SHA256,
        "pre_review_receipt_sha256": pre_review_receipt_sha256,
        "review_payload_sha256": _sha(review_raw),
        "review_index_sha256": _sha(review_index_raw),
        "worksheet_sha256": _sha(worksheet_raw),
        "lineage_manifest_sha256": _sha(lineage_raw),
        "reviewer_role": "claim_reviewer",
        "reviewer_key_id": reviewer_key_id,
        "reviewer_key_fingerprint_sha256": reviewer_key_fingerprint_sha256,
        "statement_id": "key_holder_adopts_exact_review_export_bytes_v1",
        "semantic_review_authenticated": False,
        "execution_authorization": "not_granted",
    }


def create_attestation(*, worksheet: Any, worksheet_raw: bytes, lineage: Any,
                       lineage_raw: bytes, review: Any, review_raw: bytes,
                       review_index: Any, review_index_raw: bytes,
                       expected_lineage_sha256: str,
                       pre_review_receipt_sha256: str, reviewer_key_id: str,
                       reviewer_private_key_pem: bytes,
                       reviewer_public_key_pem: bytes,
                       expected_reviewer_key_fingerprint_sha256: str) -> bytes:
    """Return signed JSON bytes; key fingerprint pin must come from outside this call."""
    _validate_exports(worksheet=worksheet, worksheet_raw=worksheet_raw,
                      lineage=lineage, lineage_raw=lineage_raw,
                      review=review, review_raw=review_raw,
                      review_index=review_index, review_index_raw=review_index_raw,
                      expected_lineage_sha256=expected_lineage_sha256)
    _validate_bindings(FROZEN_POLICY_SHA256, pre_review_receipt_sha256, reviewer_key_id)
    if (not isinstance(expected_reviewer_key_fingerprint_sha256, str)
            or not _SHA.fullmatch(expected_reviewer_key_fingerprint_sha256)):
        _fail("reviewer_key_fingerprint_invalid")
    try:
        fingerprint = _sha(_public_der(reviewer_public_key_pem))
    except (AttestationError, ValueError, OSError):
        _fail("reviewer_public_key_invalid")
    if fingerprint != expected_reviewer_key_fingerprint_sha256:
        _fail("reviewer_public_key_pin_mismatch")
    body = _body(review_raw=review_raw, review_index_raw=review_index_raw,
                 worksheet_raw=worksheet_raw, lineage_raw=lineage_raw,
                 pre_review_receipt_sha256=pre_review_receipt_sha256,
                 reviewer_key_id=reviewer_key_id,
                 reviewer_key_fingerprint_sha256=fingerprint)
    try:
        signature = _sign(reviewer_private_key_pem, reviewer_public_key_pem,
                         DOMAIN + _canonical(body))
    except ReceiptError as exc:
        if str(exc) == "signing_key_does_not_match_roster":
            _fail("reviewer_key_pair_mismatch")
        _fail("reviewer_signing_key_invalid")
    except (ValueError, OSError):
        _fail("reviewer_signing_key_invalid")
    envelope = {**body, "signature_ed25519_base64": base64.b64encode(signature).decode("ascii")}
    return (json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def verify_attestation(*, attestation_raw: bytes, review_raw: bytes,
                       worksheet: Any, worksheet_raw: bytes, lineage: Any,
                       lineage_raw: bytes, review: Any, review_index: Any,
                       review_index_raw: bytes, expected_lineage_sha256: str,
                       pre_review_receipt_sha256: str,
                       expected_reviewer_key_id: str,
                       reviewer_public_key_pem: bytes,
                       expected_reviewer_key_fingerprint_sha256: str) -> dict[str, Any]:
    """Verify exact-byte adoption under an externally pinned reviewer key."""
    _validate_exports(worksheet=worksheet, worksheet_raw=worksheet_raw,
                      lineage=lineage, lineage_raw=lineage_raw,
                      review=review, review_raw=review_raw,
                      review_index=review_index, review_index_raw=review_index_raw,
                      expected_lineage_sha256=expected_lineage_sha256)
    _validate_bindings(FROZEN_POLICY_SHA256, pre_review_receipt_sha256,
                       expected_reviewer_key_id)
    if (not isinstance(expected_reviewer_key_fingerprint_sha256, str)
            or not _SHA.fullmatch(expected_reviewer_key_fingerprint_sha256)):
        _fail("reviewer_key_fingerprint_invalid")
    try:
        fingerprint = _sha(_public_der(reviewer_public_key_pem))
    except (ValueError, OSError):
        _fail("reviewer_public_key_invalid")
    if fingerprint != expected_reviewer_key_fingerprint_sha256:
        _fail("reviewer_public_key_pin_mismatch")
    attestation = _parse(attestation_raw, "attestation_json_invalid")
    expected = _body(review_raw=review_raw, review_index_raw=review_index_raw,
                     worksheet_raw=worksheet_raw, lineage_raw=lineage_raw,
                     pre_review_receipt_sha256=pre_review_receipt_sha256,
                     reviewer_key_id=expected_reviewer_key_id,
                     reviewer_key_fingerprint_sha256=fingerprint)
    signature_key = "signature_ed25519_base64"
    if not isinstance(attestation, dict) or set(attestation) != set(expected) | {signature_key}:
        _fail("attestation_schema_invalid")
    body = {key: value for key, value in attestation.items() if key != signature_key}
    if _canonical(body) != _canonical(expected):
        _fail("attestation_binding_or_statement_mismatch")
    signature = _decode_signature(attestation.get(signature_key), "attestation_signature_invalid")
    if not _verify_signature(reviewer_public_key_pem, signature, DOMAIN + _canonical(body)):
        _fail("attestation_signature_invalid")
    return {
        "status": "reviewer_key_attested_unvalidated",
        "reviewer_key_id": expected_reviewer_key_id,
        "reviewer_key_fingerprint_sha256": fingerprint,
        "review_payload_sha256": _sha(review_raw),
        "review_index_sha256": _sha(review_index_raw),
        "reviewer_key_signature_valid": True,
        "key_holder_adoption_attested": True,
        "pre_review_receipt_verified": False,
        "reviewer_role_roster_verified": False,
        "source_media_custody_verified": False,
        "human_identity_verified": False,
        "semantic_review_authenticated": False,
        "execution_authorization": "not_granted",
    }


def _read_json(path: Path, label: str) -> tuple[Any, bytes]:
    raw = _read_file(path, private=True, max_bytes=MAX_JSON_BYTES,
                     failure_code=f"{label}_file_invalid")
    return _parse(raw, f"{label}_json_invalid"), raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_artifacts(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--worksheet", required=True, type=Path)
        command_parser.add_argument("--lineage", required=True, type=Path)
        command_parser.add_argument("--review", required=True, type=Path)
        command_parser.add_argument("--review-index", required=True, type=Path)
        command_parser.add_argument("--expected-lineage-sha256", required=True)
        command_parser.add_argument("--pre-review-receipt-sha256", required=True)
        command_parser.add_argument("--reviewer-key-id", required=True)
        command_parser.add_argument("--reviewer-public-key", required=True, type=Path)
        command_parser.add_argument("--reviewer-key-fingerprint-sha256", required=True)
        command_parser.add_argument("--output", required=True, type=Path)

    sign_parser = subparsers.add_parser("sign", help="sign structurally valid exact review bytes")
    add_artifacts(sign_parser)
    sign_parser.add_argument("--reviewer-private-key", required=True, type=Path)

    verify_parser = subparsers.add_parser("verify", help="verify a detached review attestation")
    add_artifacts(verify_parser)
    verify_parser.add_argument("--attestation", required=True, type=Path)

    args = parser.parse_args(argv)
    try:
        worksheet, worksheet_raw = _read_json(args.worksheet, "worksheet")
        lineage, lineage_raw = _read_json(args.lineage, "lineage")
        review, review_raw = _read_json(args.review, "review")
        review_index, review_index_raw = _read_json(args.review_index, "review_index")
        public_key = _read_file(args.reviewer_public_key, private=True,
                                max_bytes=64_000, failure_code="reviewer_public_key_file_invalid")
        common = {
            "worksheet": worksheet, "worksheet_raw": worksheet_raw,
            "lineage": lineage, "lineage_raw": lineage_raw,
            "review": review, "review_raw": review_raw,
            "review_index": review_index, "review_index_raw": review_index_raw,
            "expected_lineage_sha256": args.expected_lineage_sha256,
            "pre_review_receipt_sha256": args.pre_review_receipt_sha256,
            "reviewer_key_id": args.reviewer_key_id,
            "expected_reviewer_key_fingerprint_sha256": args.reviewer_key_fingerprint_sha256,
        }
        if args.command == "sign":
            private_key = _read_file(args.reviewer_private_key, private=True,
                                     max_bytes=64_000, failure_code="reviewer_private_key_file_invalid")
            output = create_attestation(
                **common, reviewer_private_key_pem=private_key,
                reviewer_public_key_pem=public_key)
            _write_private_output(args.output, output)
            summary = {"status": "reviewer_key_attested_unvalidated",
                       "execution_authorization": "not_granted"}
        else:
            attestation_raw = _read_file(args.attestation, private=True,
                                         max_bytes=MAX_JSON_BYTES,
                                         failure_code="attestation_file_invalid")
            common["expected_reviewer_key_id"] = common.pop("reviewer_key_id")
            report = verify_attestation(
                **common, attestation_raw=attestation_raw,
                reviewer_public_key_pem=public_key)
            output = (json.dumps(report, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n").encode("utf-8")
            _write_private_output(args.output, output)
            summary = {"status": report["status"],
                       "reviewer_key_signature_valid": report["reviewer_key_signature_valid"],
                       "semantic_review_authenticated": False,
                       "execution_authorization": "not_granted"}
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (AttestationError, ReceiptError) as exc:
        print(json.dumps({"status": "blocked", "failure_code": str(exc),
                          "execution_authorization": "not_granted"}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "blocked", "failure_code": "internal_error",
                          "execution_authorization": "not_granted"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
