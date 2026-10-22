"""Verify a signed pre-baseline receipt and bind its row pins to live batches.

This composition verifies exact receipt/artifact bytes under caller-supplied
rosters and trust-root pin, then compares the signed row digests against the
same three in-process CPU ``FrozenBatch`` objects. The helper is deliberately
non-authorizing: the root pin's out-of-band origin and the upstream human,
clock/frame, target, and baseline gates remain separate requirements.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from qta_phase_freeze_receipts import verify_phase_freeze_receipt
from qta_prebaseline_live_batch_binding import validate_live_processor_batches

ROW_COUNT = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")


def _fail(code: str) -> None:
    raise ValueError(code)


def _receipt_record_bindings(receipt_raw: bytes) -> list[dict[str, str]]:
    if not isinstance(receipt_raw, bytes) or not receipt_raw:
        _fail("phase1_receipt_bytes_invalid")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("phase1_receipt_duplicate_json_key")
            result[key] = value
        return result

    try:
        receipt = json.loads(receipt_raw.decode("utf-8"), object_pairs_hook=unique_object,
                             parse_constant=lambda _value: _fail(
                                 "phase1_receipt_nonfinite_json"))
    except ValueError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        _fail("phase1_receipt_json_invalid")
    rows = receipt.get("record_bindings") if isinstance(receipt, dict) else None
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        _fail("phase1_receipt_record_bindings_invalid")
    result = []
    seen = set()
    for row in rows:
        if (not isinstance(row, dict)
                or set(row) != {"record_token", "processor_batch_sha256"}):
            _fail("phase1_receipt_record_binding_schema_invalid")
        token, digest = row["record_token"], row["processor_batch_sha256"]
        if (not isinstance(token, str) or not _SHA256.fullmatch(token)
                or token in seen or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)):
            _fail("phase1_receipt_record_binding_invalid")
        seen.add(token)
        result.append({"record_token": token, "processor_batch_sha256": digest})
    return result


def verify_and_bind_signed_phase1_batches(
        *, receipt_raw: bytes, artifact_bytes: Mapping[str, bytes],
        integrity_roster_raw: bytes, release_roster_raw: bytes,
        root_public_key_pem: bytes, expected_root_public_der_sha256: str,
        expected_external_cohort_pin_sha256: str,
        live_rows: Any,
        expected_processor_batch_row_pins: Any = None,
        expected_processor_batch_protocol_sha256: str | None = None) -> dict[str, Any]:
    """Verify the signed phase-1 receipt then atomically bind its 3 row hashes.

    Only aggregate counts and fixed, content-free failure codes are returned.
    The helper does not return batches or trust a caller-created success report.
    """
    result: dict[str, Any] = {
        "status": "blocked", "requested_rows": ROW_COUNT,
        "validated_rows": 0, "phase_receipt_signature_valid": False,
        "artifact_bytes_match": False,
        "external_cohort_pin_origin_authenticated": False,
        "model_load_permitted": False, "scoring_authorized": False,
        "authorized_forwards": 0, "failure_code": "signed_batch_binding_blocked",
    }
    try:
        report = verify_phase_freeze_receipt(
            receipt_raw=receipt_raw, phase="pre_baseline", artifact_bytes=artifact_bytes,
            integrity_roster_raw=integrity_roster_raw,
            release_roster_raw=release_roster_raw,
            root_public_key_pem=root_public_key_pem,
            expected_root_public_der_sha256=expected_root_public_der_sha256,
            expected_external_cohort_pin_sha256=expected_external_cohort_pin_sha256,
            expected_processor_batch_row_pins=expected_processor_batch_row_pins,
            expected_processor_batch_protocol_sha256=expected_processor_batch_protocol_sha256,
        )
        if (report.get("status") != "verified_signed_byte_bindings_non_authorizing"
                or report.get("receipt_signature_valid") is not True
                or report.get("artifact_bytes_match") is not True
                or report.get("model_load_permitted") is not False
                or report.get("scoring_authorized") is not False
                or report.get("external_cohort_pin_origin_authenticated") is not False):
            _fail("phase1_receipt_verification_contract_invalid")
        records = _receipt_record_bindings(receipt_raw)
        stage1_raw = artifact_bytes.get("stage1_review_and_lineage")
        if not isinstance(stage1_raw, bytes):
            _fail("phase1_stage1_artifact_missing")
        try:
            stage1 = json.loads(
                stage1_raw.decode("utf-8"), object_pairs_hook=lambda pairs: _unique(pairs),
                parse_constant=lambda _value: _fail("phase1_stage1_nonfinite_json"))
        except ValueError:
            raise
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            _fail("phase1_stage1_artifact_invalid")
        tokens = stage1.get("expected_cohort_tokens") if isinstance(stage1, dict) else None
        source_rows = stage1.get("source_rows") if isinstance(stage1, dict) else None
        if (not isinstance(tokens, list) or len(tokens) != ROW_COUNT
                or not isinstance(source_rows, list) or len(source_rows) != ROW_COUNT):
            _fail("phase1_stage1_source_rows_invalid")
        source_crosswalk = []
        expected_rows = []
        for record, token, source in zip(records, tokens, source_rows, strict=True):
            if not isinstance(source, Mapping):
                _fail("phase1_source_crosswalk_row_invalid")
            review_token = token
            source_sha = source.get("source_video_sha256")
            if (not isinstance(review_token, str) or not _TOKEN.fullmatch(review_token)
                    or source.get("review_token") != review_token
                    or not isinstance(source_sha, str) or not _SHA256.fullmatch(source_sha)):
                _fail("phase1_source_crosswalk_row_invalid")
            token_digest = hashlib.sha256(review_token.encode("ascii")).hexdigest()
            if record["record_token"] != token_digest:
                _fail("phase1_receipt_source_order_mismatch")
            source_crosswalk.append({"review_token": review_token,
                                     "video_sha256": source_sha})
            expected_rows.append({
                "review_token": review_token,
                "source_video_sha256": source_sha,
                "batch_sha256": record["processor_batch_sha256"],
            })
        bound = validate_live_processor_batches(
            expected_rows=expected_rows, live_rows=live_rows,
            source_crosswalk=source_crosswalk,
        )
        result.update({
            "status": bound["status"],
            "validated_rows": bound["validated_rows"],
            "phase_receipt_signature_valid": True,
            "artifact_bytes_match": True,
            "failure_code": bound["failure_code"],
        })
    except Exception as exc:
        code = str(exc)
        result["failure_code"] = (code if re.fullmatch(r"[a-z][a-z0-9_]{0,119}", code)
                                  else "signed_batch_binding_failed")
    return result


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail("phase1_stage1_duplicate_json_key")
        value[key] = item
    return value
