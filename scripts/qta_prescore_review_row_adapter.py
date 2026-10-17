"""Adapt an audited first-stage review bundle to QTA mapping rows.

This small CPU-only adapter accepts the exact objects and raw bytes supplied
to ``audit_nextgqa_review_handoff.audit``. It checks the returned audit summary
and copies only the fields required by ``audit_qta_prescore_mapping_bundle``.
It reruns the handoff audit from the exact supplied source bytes instead of
trusting a caller-supplied success summary. It remains structural: it does not
authenticate a reviewer, verify custody/clock evidence, load a model, or
authorize scoring.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from audit_nextgqa_review_handoff import audit as audit_review_handoff


ROW_COUNT = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")
_DECISIONS = {
    "supports_claim", "partial", "contradicted", "unsupported",
    "ambiguous", "unlocalizable",
}


class ReviewRowAdapterError(ValueError):
    """Stable, content-free validation error."""


def _fail(code: str) -> None:
    raise ReviewRowAdapterError(code)


def _strict_json(raw: bytes, code: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate_json_key")
            result[key] = value
        return result

    if not isinstance(raw, bytes) or not raw:
        _fail(code)
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_object,
            parse_constant=lambda _value: _fail("nonfinite_json_value"),
        )
    except ReviewRowAdapterError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        _fail(code)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def adapt_review_bundle_to_qta_rows(
    *, worksheet_raw: bytes, lineage_raw: bytes,
    review_raw: bytes, review_index_raw: bytes,
    expected_lineage_sha256: str,
) -> list[dict[str, Any]]:
    """Audit exact handoff bytes and return rows expected by the QTA binder.

    The handoff validator is rerun here, so a caller-created success dict cannot
    stand in for source-byte checks. These rows retain worksheet order; the
    single audited review-payload digest is repeated for every row.
    """
    lineage = _strict_json(lineage_raw, "lineage_json_invalid")
    worksheet = _strict_json(worksheet_raw, "worksheet_json_invalid")
    review = _strict_json(review_raw, "review_payload_json_invalid")
    review_index = _strict_json(review_index_raw, "review_index_json_invalid")
    audit_report = audit_review_handoff(
        worksheet, worksheet_raw, lineage, lineage_raw, review, review_raw,
        review_index, review_index_raw, expected_lineage_sha256)
    if not isinstance(audit_report, dict):
        _fail("review_audit_not_successful")
    if (audit_report.get("status") != "unvalidated_handoff_structurally_valid"
            or audit_report.get("failure_codes") != []
            or type(audit_report.get("reviewed_records")) is not int
            or audit_report.get("reviewed_records") != ROW_COUNT
            or audit_report.get("execution_authorization") != "not_granted"
            or audit_report.get("semantic_review_authenticated") is not False):
        _fail("review_audit_not_successful")

    worksheet_sha = audit_report.get("worksheet_sha256")
    review_sha = audit_report.get("review_payload_sha256")
    if (not isinstance(worksheet_sha, str) or not _SHA256.fullmatch(worksheet_sha)
            or not isinstance(review_sha, str) or not _SHA256.fullmatch(review_sha)):
        _fail("review_audit_digest_invalid")
    if _sha256(worksheet_raw) != worksheet_sha:
        _fail("worksheet_raw_digest_mismatch")
    if _sha256(review_raw) != review_sha:
        _fail("review_payload_raw_digest_mismatch")

    parsed_worksheet = _strict_json(worksheet_raw, "worksheet_json_invalid")
    parsed_review = _strict_json(review_raw, "review_payload_json_invalid")
    if parsed_worksheet != worksheet or parsed_review != review:
        _fail("review_objects_do_not_match_raw_bytes")
    if not isinstance(worksheet, dict) or not isinstance(review, dict):
        _fail("review_bundle_schema_invalid")

    worksheet_rows = worksheet.get("records")
    review_rows = review.get("records")
    if (not isinstance(worksheet_rows, list) or len(worksheet_rows) != ROW_COUNT
            or not isinstance(review_rows, list) or len(review_rows) != ROW_COUNT):
        _fail("review_bundle_row_count_invalid")

    adapted: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    for ordinal, (source_row, reviewed_row) in enumerate(
            zip(worksheet_rows, review_rows, strict=True), start=1):
        if not isinstance(source_row, dict) or not isinstance(reviewed_row, dict):
            _fail(f"row_{ordinal}_schema_invalid")
        token = source_row.get("review_token")
        source_video_sha = source_row.get("video_sha256")
        if (not isinstance(token, str) or not _TOKEN.fullmatch(token)
                or token in seen_tokens):
            _fail(f"row_{ordinal}_worksheet_token_invalid")
        seen_tokens.add(token)
        if (not isinstance(source_video_sha, str)
                or not _SHA256.fullmatch(source_video_sha)):
            _fail(f"row_{ordinal}_source_video_digest_invalid")
        if (set(reviewed_row) != {"review_token", "review"}
                or reviewed_row.get("review_token") != token):
            _fail(f"row_{ordinal}_review_token_order_mismatch")
        item = reviewed_row.get("review")
        if not isinstance(item, dict):
            _fail(f"row_{ordinal}_review_payload_invalid")
        decision = item.get("support_decision")
        intervals = item.get("evidence_intervals_seconds")
        if not isinstance(decision, str) or decision not in _DECISIONS:
            _fail(f"row_{ordinal}_support_decision_invalid")
        if not isinstance(intervals, list):
            _fail(f"row_{ordinal}_evidence_intervals_invalid")

        adapted.append({
            "review_token": token,
            "source_video_sha256": source_video_sha,
            "review_payload_sha256": review_sha,
            "support_decision": decision,
            "evidence_intervals_seconds": deepcopy(intervals),
        })
    return adapted
