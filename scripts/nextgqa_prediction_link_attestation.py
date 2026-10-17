"""Sign and verify exact-byte adoption of a QTA stage-2 review receipt.

The envelope binds the stage-2 receipt and baseline manifest to the
root-signed release-roster prediction-link reviewer key. Structural validation
is rerun before every signature operation. This proves key adoption of exact
bytes only; it does not prove human identity, actual review, or semantic
correctness, and it never authorizes execution.
"""
from __future__ import annotations

import base64
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from audit_nextgqa_review_release_order import (
    _validate_release_roster,
)
from nextgqa_pre_review_integrity_receipt import (
    ReceiptError,
    _canonical,
    _decode_signature,
    _public_der,
    _read_file,
    _sha,
    _sign,
    _verify_signature,
    _write_private_output,
)
from nextgqa_prediction_link_review import (
    PredictionLinkReviewError,
    _unique_object,
    validate_prediction_link_receipt,
)


DOMAIN = b"qta_prediction_link_review_attestation_v1\0"
SIGNATURE_FIELD = "signature_ed25519_base64"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 8 * 1024 * 1024
_MAX_RECEIPT_BYTES = 2 * 1024 * 1024
_MAX_ATTESTATION_BYTES = 2 * 1024 * 1024


class PredictionLinkAttestationError(ValueError):
    """Stable, content-free stage-2 signature error."""


def _fail(code: str) -> None:
    raise PredictionLinkAttestationError(code)


def _validated_inputs(*, receipt_raw: bytes, baseline_manifest_raw: bytes,
                      expected_baseline_manifest_sha256: str,
                      expected_protocol_sha256: str,
                      expected_video_review_receipt_sha256: str,
                      expected_cohort_tokens: Sequence[str],
                      expected_source_field_sha256_by_token: dict[str, dict[str, str]],
                      integrity_roster_raw: bytes, release_roster_raw: bytes,
                      root_public_key_pem: bytes,
                      expected_root_public_der_sha256: str) -> tuple[dict, dict[str, bytes]]:
    if (not isinstance(receipt_raw, bytes)
            or len(receipt_raw) > _MAX_RECEIPT_BYTES):
        _fail("stage2_receipt_bytes_invalid")
    try:
        receipt = json.loads(receipt_raw.decode("utf-8"),
                             object_pairs_hook=_unique_object,
                             parse_constant=lambda _: _fail("json_nonfinite_value"))
        report = validate_prediction_link_receipt(
            receipt=receipt,
            baseline_manifest_raw=baseline_manifest_raw,
            expected_baseline_manifest_sha256=expected_baseline_manifest_sha256,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_video_review_receipt_sha256=expected_video_review_receipt_sha256,
            expected_cohort_tokens=expected_cohort_tokens,
            expected_source_field_sha256_by_token=expected_source_field_sha256_by_token,
        )
    except PredictionLinkReviewError as exc:
        _fail("stage2_receipt_invalid:" + str(exc))
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError, RecursionError):
        _fail("stage2_receipt_json_invalid")
    if not report["all_expected_rows_have_supporting_self_reported_judgment"]:
        _fail("stage2_cohort_not_eligible")
    try:
        _, public_by_role = _validate_release_roster(
            release_roster_raw, root_public_key_pem,
            expected_root_public_der_sha256,
            integrity_roster_raw)
    except ReceiptError as exc:
        _fail("release_roster_invalid:" + str(exc))
    return receipt, public_by_role


def _body(*, receipt_raw: bytes, baseline_manifest_raw: bytes,
          expected_protocol_sha256: str,
          expected_video_review_receipt_sha256: str,
          integrity_roster_raw: bytes, release_roster_raw: bytes,
          root_public_key_pem: bytes,
          expected_root_public_der_sha256: str) -> tuple[dict[str, Any], bytes]:
    try:
        _, role_keys = _validate_release_roster(
            release_roster_raw, root_public_key_pem,
            expected_root_public_der_sha256, integrity_roster_raw)
    except ReceiptError as exc:
        _fail("release_roster_invalid:" + str(exc))
    roster = json.loads(release_roster_raw.decode("utf-8"),
                        object_pairs_hook=_unique_object)
    entry = next((row for row in roster["keys"]
                  if row.get("role") == "prediction_link_reviewer"), None)
    if not isinstance(entry, dict):
        _fail("prediction_link_reviewer_role_missing")
    reviewer_public_key = role_keys["prediction_link_reviewer"]
    try:
        fingerprint = _sha(_public_der(reviewer_public_key))
    except (ReceiptError, ValueError, OSError):
        _fail("prediction_link_reviewer_key_invalid")
    if entry.get("public_key_der_sha256") != fingerprint:
        _fail("prediction_link_reviewer_roster_key_mismatch")
    if not isinstance(expected_root_public_der_sha256, str) or not _SHA256.fullmatch(
            expected_root_public_der_sha256):
        _fail("trust_root_fingerprint_invalid")
    for name, value in (("protocol_sha256", expected_protocol_sha256),
                        ("prior_video_review_receipt_sha256",
                         expected_video_review_receipt_sha256)):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            _fail(f"{name}_invalid")
    body = {
        "schema_version": 1,
        "purpose": "qta_prediction_link_review_bytes_adoption_v1",
        "status": "reviewer_key_attested_unvalidated",
        "baseline_manifest_sha256": _sha(baseline_manifest_raw),
        "stage2_receipt_sha256": _sha(receipt_raw),
        "protocol_sha256": expected_protocol_sha256,
        "prior_video_review_receipt_sha256": expected_video_review_receipt_sha256,
        "release_roster_sha256": _sha(release_roster_raw),
        "trust_root_der_sha256": expected_root_public_der_sha256,
        "reviewer_role": "prediction_link_reviewer",
        "reviewer_key_id": entry["key_id"],
        "reviewer_key_fingerprint_sha256": fingerprint,
        "statement_id": "key_holder_adopts_exact_stage2_receipt_bytes_v1",
        "human_identity_verified": False,
        "semantic_judgment_authenticated": False,
        "execution_authorization": "not_granted",
    }
    return body, reviewer_public_key


def create_attestation(*, receipt_raw: bytes, baseline_manifest_raw: bytes,
                       expected_baseline_manifest_sha256: str,
                       expected_protocol_sha256: str,
                       expected_video_review_receipt_sha256: str,
                       expected_cohort_tokens: Sequence[str],
                       expected_source_field_sha256_by_token: dict[str, dict[str, str]],
                       integrity_roster_raw: bytes, release_roster_raw: bytes,
                       root_public_key_pem: bytes,
                       expected_root_public_der_sha256: str,
                       reviewer_private_key_pem: bytes,
                       reviewer_public_key_pem: bytes) -> bytes:
    """Validate and sign exact stage-2 receipt bytes with the rostered key."""
    try:
        _, roles = _validated_inputs(
            receipt_raw=receipt_raw, baseline_manifest_raw=baseline_manifest_raw,
            expected_baseline_manifest_sha256=expected_baseline_manifest_sha256,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_video_review_receipt_sha256=expected_video_review_receipt_sha256,
            expected_cohort_tokens=expected_cohort_tokens,
            expected_source_field_sha256_by_token=expected_source_field_sha256_by_token,
            integrity_roster_raw=integrity_roster_raw,
            release_roster_raw=release_roster_raw, root_public_key_pem=root_public_key_pem,
            expected_root_public_der_sha256=expected_root_public_der_sha256)
        body, rostered_public = _body(
            receipt_raw=receipt_raw, baseline_manifest_raw=baseline_manifest_raw,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_video_review_receipt_sha256=expected_video_review_receipt_sha256,
            integrity_roster_raw=integrity_roster_raw,
            release_roster_raw=release_roster_raw, root_public_key_pem=root_public_key_pem,
            expected_root_public_der_sha256=expected_root_public_der_sha256)
        if _public_der(reviewer_public_key_pem) != _public_der(rostered_public):
            _fail("prediction_link_reviewer_key_does_not_match_roster")
        signature = _sign(reviewer_private_key_pem, reviewer_public_key_pem,
                          DOMAIN + _canonical(body))
        return (json.dumps({**body, SIGNATURE_FIELD: base64.b64encode(signature).decode("ascii")},
                           sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except ReceiptError as exc:
        _fail("signing_key_invalid:" + str(exc))


def verify_attestation(*, attestation_raw: bytes, receipt_raw: bytes,
                       baseline_manifest_raw: bytes,
                       expected_baseline_manifest_sha256: str,
                       expected_protocol_sha256: str,
                       expected_video_review_receipt_sha256: str,
                       expected_cohort_tokens: Sequence[str],
                       expected_source_field_sha256_by_token: dict[str, dict[str, str]],
                       integrity_roster_raw: bytes, release_roster_raw: bytes,
                       root_public_key_pem: bytes,
                       expected_root_public_der_sha256: str) -> dict[str, Any]:
    """Verify signature and rerun all stage-2 structural checks."""
    try:
        _, _ = _validated_inputs(
            receipt_raw=receipt_raw, baseline_manifest_raw=baseline_manifest_raw,
            expected_baseline_manifest_sha256=expected_baseline_manifest_sha256,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_video_review_receipt_sha256=expected_video_review_receipt_sha256,
            expected_cohort_tokens=expected_cohort_tokens,
            expected_source_field_sha256_by_token=expected_source_field_sha256_by_token,
            integrity_roster_raw=integrity_roster_raw,
            release_roster_raw=release_roster_raw, root_public_key_pem=root_public_key_pem,
            expected_root_public_der_sha256=expected_root_public_der_sha256)
        expected, reviewer_public = _body(
            receipt_raw=receipt_raw, baseline_manifest_raw=baseline_manifest_raw,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_video_review_receipt_sha256=expected_video_review_receipt_sha256,
            integrity_roster_raw=integrity_roster_raw,
            release_roster_raw=release_roster_raw, root_public_key_pem=root_public_key_pem,
            expected_root_public_der_sha256=expected_root_public_der_sha256)
        if (not isinstance(attestation_raw, bytes)
                or len(attestation_raw) > _MAX_ATTESTATION_BYTES):
            _fail("stage2_attestation_bytes_invalid")
        envelope = json.loads(attestation_raw.decode("utf-8"), object_pairs_hook=_unique_object,
                              parse_constant=lambda _: _fail("json_nonfinite_value"))
        if not isinstance(envelope, dict) or set(envelope) != set(expected) | {SIGNATURE_FIELD}:
            _fail("stage2_attestation_schema_invalid")
        body = {key: value for key, value in envelope.items() if key != SIGNATURE_FIELD}
        if _canonical(body) != _canonical(expected):
            _fail("stage2_attestation_binding_mismatch")
        signature = _decode_signature(envelope[SIGNATURE_FIELD], "stage2_signature_invalid")
        if not _verify_signature(reviewer_public, signature, DOMAIN + _canonical(body)):
            _fail("stage2_signature_invalid")
        return {
            "status": "reviewer_key_attested_unvalidated",
            "reviewer_role": "prediction_link_reviewer",
            "reviewer_key_id": expected["reviewer_key_id"],
            "stage2_receipt_sha256": expected["stage2_receipt_sha256"],
            "baseline_manifest_sha256": expected["baseline_manifest_sha256"],
            "reviewer_key_signature_valid": True,
            "all_rows_self_report_support_p_over_q": True,
            "human_identity_verified": False,
            "semantic_judgment_authenticated": False,
            "execution_authorization": "not_granted",
        }
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError, RecursionError) as exc:
        if isinstance(exc, PredictionLinkAttestationError):
            raise
        _fail("stage2_attestation_json_invalid")


def _read_json(path: Path, label: str) -> tuple[Any, bytes]:
    raw = _read_file(path, private=True, max_bytes=_MAX_BYTES,
                     failure_code=f"{label}_file_invalid")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError, RecursionError):
        _fail(f"{label}_json_invalid")
    return value, raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(command_parser):
        command_parser.add_argument("--receipt", required=True, type=Path)
        command_parser.add_argument("--baseline-manifest", required=True, type=Path)
        command_parser.add_argument("--expected-baseline-manifest-sha256", required=True)
        command_parser.add_argument("--expected-protocol-sha256", required=True)
        command_parser.add_argument("--expected-video-review-receipt-sha256", required=True)
        command_parser.add_argument("--expected-inputs-json", required=True, type=Path,
                                    help="Private JSON with cohort_tokens and source_field_sha256_by_token")
        command_parser.add_argument("--integrity-roster", required=True, type=Path)
        command_parser.add_argument("--release-roster", required=True, type=Path)
        command_parser.add_argument("--root-public-key", required=True, type=Path)
        command_parser.add_argument("--root-public-der-sha256", required=True)

    sign_parser = subparsers.add_parser("sign", help="sign exact, structurally valid stage-2 bytes")
    add_common(sign_parser)
    sign_parser.add_argument("--reviewer-private-key", required=True, type=Path)
    sign_parser.add_argument("--reviewer-public-key", required=True, type=Path)
    sign_parser.add_argument("--output", required=True, type=Path)

    verify_parser = subparsers.add_parser("verify", help="verify stage-2 exact-byte adoption")
    add_common(verify_parser)
    verify_parser.add_argument("--attestation", required=True, type=Path)

    args = parser.parse_args(argv)
    receipt_raw = _read_file(args.receipt, private=True, max_bytes=_MAX_BYTES,
                             failure_code="stage2_receipt_file_invalid")
    _, baseline_raw = _read_json(args.baseline_manifest, "baseline_manifest")
    inputs, _ = _read_json(args.expected_inputs_json, "expected_inputs")
    if (not isinstance(inputs, dict)
            or set(inputs) != {"cohort_tokens", "source_field_sha256_by_token"}):
        _fail("expected_inputs_schema_invalid")
    integrity_raw = _read_file(args.integrity_roster, private=True, max_bytes=_MAX_BYTES,
                                failure_code="integrity_roster_file_invalid")
    release_raw = _read_file(args.release_roster, private=True, max_bytes=_MAX_BYTES,
                              failure_code="release_roster_file_invalid")
    root_public = _read_file(args.root_public_key, private=True, max_bytes=64_000,
                             failure_code="root_public_key_file_invalid")
    common = dict(
        receipt_raw=receipt_raw, baseline_manifest_raw=baseline_raw,
        expected_baseline_manifest_sha256=args.expected_baseline_manifest_sha256,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_video_review_receipt_sha256=args.expected_video_review_receipt_sha256,
        expected_cohort_tokens=inputs["cohort_tokens"],
        expected_source_field_sha256_by_token=inputs["source_field_sha256_by_token"],
        integrity_roster_raw=integrity_raw, release_roster_raw=release_raw,
        root_public_key_pem=root_public,
        expected_root_public_der_sha256=args.root_public_der_sha256,
    )
    if args.command == "sign":
        private = _read_file(args.reviewer_private_key, private=True, max_bytes=64_000,
                             failure_code="reviewer_private_key_file_invalid")
        public = _read_file(args.reviewer_public_key, private=True, max_bytes=64_000,
                            failure_code="reviewer_public_key_file_invalid")
        output = create_attestation(
            **common, reviewer_private_key_pem=private,
            reviewer_public_key_pem=public)
        _write_private_output(args.output, output)
        print(json.dumps({"status": "reviewer_key_attested_unvalidated",
                          "output_mode": "0600", "human_identity_verified": False,
                          "semantic_judgment_authenticated": False,
                          "execution_authorization": "not_granted"}, sort_keys=True))
        return 0
    attestation = _read_file(args.attestation, private=True, max_bytes=_MAX_BYTES,
                             failure_code="stage2_attestation_file_invalid")
    report = verify_attestation(attestation_raw=attestation, **common)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PredictionLinkAttestationError, ReceiptError) as exc:
        print(json.dumps({"status": "blocked", "failure_code": str(exc),
                          "execution_authorization": "not_granted"}, sort_keys=True))
        raise SystemExit(2)
