"""Validate a private prediction-link receipt against frozen input digests.

This checks structure and caller-pinned source bindings only. It does not
authenticate the reviewer, verify semantics, validate release/clock gates, or
authorize scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from nextgqa_pre_review_integrity_receipt import _read_file, _write_private_output
from nextgqa_prediction_link_review import (
    PredictionLinkReviewError,
    validate_prediction_link_receipt,
)


MAX_BINDING_BYTES = 1_000_000
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COHORT_KEYS = {
    "schema_version", "purpose", "status", "protocol_sha256",
    "baseline_manifest_sha256", "records", "execution_authorization",
}
COHORT_ROW_KEYS = {
    "review_token", "question_sha256", "options_sha256", "claim_sha256",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate_json_key")
        obj[key] = value
    return obj


def _private_root(repo: Path) -> Path:
    root = repo / "data" / "private_nextgqa"
    info = os.lstat(root)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private_root_invalid")
    return root


def _safe_name(name: str) -> str:
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or "/" in name or "\\" in name or "\x00" in name):
        raise ValueError("private_filename_invalid")
    return name


def validate_private_files(*, private_root: Path, baseline_name: str,
                           receipt_name: str, cohort_source_name: str,
                           output_name: str, expected_baseline_sha256: str,
                           expected_cohort_source_sha256: str,
                           expected_protocol_sha256: str) -> dict[str, Any]:
    """Read only private direct children and write a non-authorizing summary."""
    baseline_name, receipt_name, cohort_source_name, output_name = map(
        _safe_name, (baseline_name, receipt_name, cohort_source_name, output_name))
    for value in (expected_baseline_sha256, expected_cohort_source_sha256,
                  expected_protocol_sha256):
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            raise ValueError("expected_digest_invalid")
    baseline_raw = _read_file(private_root / baseline_name, private=True,
                              max_bytes=8 * 1024 * 1024,
                              failure_code="baseline_manifest_file_invalid")
    receipt_raw = _read_file(private_root / receipt_name, private=True,
                             max_bytes=2 * 1024 * 1024,
                             failure_code="prediction_link_receipt_file_invalid")
    cohort_raw = _read_file(private_root / cohort_source_name, private=True,
                            max_bytes=MAX_BINDING_BYTES,
                            failure_code="cohort_source_file_invalid")
    if hashlib.sha256(cohort_raw).hexdigest() != expected_cohort_source_sha256:
        raise ValueError("cohort_source_digest_mismatch")
    try:
        cohort = json.loads(cohort_raw.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                ValueError("nonfinite_json")))
        receipt = json.loads(receipt_raw.decode("utf-8"), object_pairs_hook=_unique_object,
                             parse_constant=lambda _: (_ for _ in ()).throw(
                                 ValueError("nonfinite_json")))
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise ValueError("cohort_or_receipt_json_invalid") from None
    if (not isinstance(cohort, dict) or set(cohort) != COHORT_KEYS
            or type(cohort.get("schema_version")) is not int
            or cohort.get("schema_version") != 1
            or cohort.get("purpose") != "qta_prediction_link_frozen_cohort_sources"
            or cohort.get("status") != "frozen_pre_score"
            or cohort.get("execution_authorization") != "not_granted"
            or cohort.get("protocol_sha256") != expected_protocol_sha256
            or cohort.get("baseline_manifest_sha256") != expected_baseline_sha256
            or not isinstance(cohort.get("records"), list)
            or not cohort["records"]):
        raise ValueError("cohort_source_schema_or_binding_invalid")
    tokens, sources = [], {}
    for row in cohort["records"]:
        if not isinstance(row, dict) or set(row) != COHORT_ROW_KEYS:
            raise ValueError("cohort_source_record_invalid")
        token = row.get("review_token")
        if not isinstance(token, str) or not token or token in sources:
            raise ValueError("cohort_source_token_invalid")
        tokens.append(token)
        sources[token] = {key: row[key] for key in (
            "question_sha256", "options_sha256", "claim_sha256")}
        if any(not isinstance(value, str) or not SHA256.fullmatch(value)
               for value in sources[token].values()):
            raise ValueError("cohort_source_field_digest_invalid")
    report = validate_prediction_link_receipt(
        receipt=receipt, baseline_manifest_raw=baseline_raw,
        expected_baseline_manifest_sha256=expected_baseline_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_cohort_tokens=tokens,
        expected_source_field_sha256_by_token=sources)
    summary = {
        "schema_version": 1,
        "purpose": "qta_prediction_link_private_structural_validation_summary",
        "status": report["status"],
        "baseline_manifest_sha256": expected_baseline_sha256,
        "cohort_source_sha256": expected_cohort_source_sha256,
        "protocol_sha256": expected_protocol_sha256,
        "requested_records": report["requested_records"],
        "support_judgment_count": report["support_judgment_count"],
        "all_expected_rows_have_supporting_self_reported_judgment": report[
            "all_expected_rows_have_supporting_self_reported_judgment"],
        "reviewer_identity_verified": False,
        "semantic_judgment_authenticated": False,
        "other_score_gates_verified": False,
        "execution_authorization": "not_granted",
        "limitations": report["limitations"],
    }
    _write_private_output(
        private_root / output_name,
        (json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--receipt-name", required=True)
    parser.add_argument("--cohort-source-name", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-cohort-source-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        report = validate_private_files(
            private_root=_private_root(Path(__file__).resolve().parents[1]),
            baseline_name=args.baseline_name, receipt_name=args.receipt_name,
            cohort_source_name=args.cohort_source_name, output_name=args.output_name,
            expected_baseline_sha256=args.expected_baseline_sha256,
            expected_cohort_source_sha256=args.expected_cohort_source_sha256,
            expected_protocol_sha256=args.expected_protocol_sha256)
    except (OSError, ValueError, PredictionLinkReviewError) as exc:
        parser.error(str(exc))
    print("Structural receipt validation complete; "
          f"status={report['status']}; records={report['requested_records']}; "
          f"support={report['support_judgment_count']}; "
          "other_gates=unverified; execution_authorization=not_granted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
