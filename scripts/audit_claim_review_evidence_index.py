#!/usr/bin/env python3
"""Bind schema-v12 claim-review hashes to the private input, ledger, and review index.

This CPU-only audit verifies byte/hash and row-reference consistency. It does
not determine whether claims are true, supported, independently reviewed, or
legally usable; it never authorizes inference and emits no IDs or locators.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from audit_claim_mapping_manifest import audit as audit_claim_ledger

SHA256 = re.compile(r"^[a-f0-9]{64}$")
MODEL_REVISION = re.compile(r"^.+@[a-f0-9]{40}$")
OCCURRENCE_POLICIES = {"one_sufficient_occurrence", "all_relevant_occurrences"}


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _read(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite_json")),
    )
    if _nonfinite(value):
        raise ValueError("non_finite_json")
    return value, raw


def _nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_nonfinite(k) or _nonfinite(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_nonfinite(x) for x in value)
    return False


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _raw_matches(value: Any, raw: Any) -> bool:
    if not isinstance(raw, bytes):
        return False
    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite_json")),
        )
        if _nonfinite(parsed) or _nonfinite(value):
            return False
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (AttributeError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        return False


def _ids(eligible: Any) -> set[str] | None:
    if not isinstance(eligible, list) or not eligible:
        return None
    values = [row.get("video_id") if isinstance(row, dict) else None for row in eligible]
    if any(not _text(value) for value in values):
        return None
    if len(set(values)) != len(values):
        return None
    return set(values)


def audit(protocol: Any, protocol_raw: Any, eligible: Any, eligible_raw: Any,
          claim_input: Any, claim_input_raw: Any, ledger: Any, ledger_raw: Any,
          review_index: Any, review_index_raw: Any) -> dict:
    failures: set[str] = set()
    for label, obj, raw in (
        ("protocol", protocol, protocol_raw), ("eligible_manifest", eligible, eligible_raw),
        ("claim_input_manifest", claim_input, claim_input_raw),
        ("claim_mapping_ledger", ledger, ledger_raw), ("review_evidence_index", review_index, review_index_raw),
    ):
        if not _raw_matches(obj, raw):
            failures.add(f"{label}_byte_binding_invalid")

    if (not isinstance(protocol, dict) or type(protocol.get("schema_version")) is not int
            or protocol.get("schema_version") != 12):
        failures.add("protocol_schema_version_invalid")
    claim_review = protocol.get("claim_review") if isinstance(protocol, dict) else None
    required_claim_review = {
        "claim_manifest_sha256", "mapping_sha256", "review_record_sha256",
        "review_status", "review_timing", "mapping_unit",
    }
    if (not isinstance(claim_review, dict) or set(claim_review) != required_claim_review
            or any(not _sha(claim_review.get(key)) for key in (
                "claim_manifest_sha256", "mapping_sha256", "review_record_sha256"))):
        failures.add("protocol_claim_review_refs_invalid")
        claim_review = {}
    elif (claim_review.get("review_status") != "completed"
          or claim_review.get("review_timing") != "before_scoring"
          or claim_review.get("mapping_unit") != "claim_to_evidence_frames"):
        failures.add("protocol_claim_review_declaration_invalid")

    eligible_ids = _ids(eligible)
    if eligible_ids is None:
        failures.add("eligible_manifest_invalid")
        eligible_ids = set()
    eligible_sha = hashlib.sha256(eligible_raw).hexdigest() if isinstance(eligible_raw, bytes) else ""
    if isinstance(protocol, dict):
        cohort = protocol.get("cohort")
        if (not isinstance(cohort, dict)
                or cohort.get("eligible_manifest_sha256") != eligible_sha
                or type(cohort.get("eligible_video_count")) is not int
                or cohort.get("eligible_video_count") != len(eligible_ids)):
            failures.add("protocol_eligible_manifest_mismatch")

    claim_input_sha = hashlib.sha256(claim_input_raw).hexdigest() if isinstance(claim_input_raw, bytes) else ""
    ledger_sha = hashlib.sha256(ledger_raw).hexdigest() if isinstance(ledger_raw, bytes) else ""
    review_index_sha = hashlib.sha256(review_index_raw).hexdigest() if isinstance(review_index_raw, bytes) else ""
    if claim_review.get("claim_manifest_sha256") != claim_input_sha:
        failures.add("protocol_claim_input_hash_mismatch")
    if claim_review.get("mapping_sha256") != ledger_sha:
        failures.add("protocol_claim_mapping_hash_mismatch")
    if claim_review.get("review_record_sha256") != review_index_sha:
        failures.add("protocol_review_index_hash_mismatch")

    input_keys = {
        "schema_version", "eligible_dataset_id", "eligible_manifest_sha256",
        "claim_policy_sha256", "occurrence_policy", "claims",
    }
    if (not isinstance(claim_input, dict) or set(claim_input) != input_keys
            or type(claim_input.get("schema_version")) is not int
            or claim_input.get("schema_version") != 1):
        failures.add("claim_input_manifest_schema_invalid")
        claim_input = {}
    if (claim_input.get("eligible_manifest_sha256") != eligible_sha
            or not _text(claim_input.get("eligible_dataset_id"))
            or not _sha(claim_input.get("claim_policy_sha256"))
            or not isinstance(claim_input.get("occurrence_policy"), str)
            or claim_input.get("occurrence_policy") not in OCCURRENCE_POLICIES):
        failures.add("claim_input_manifest_fields_invalid")

    if (not isinstance(ledger, dict)
            or type(ledger.get("schema_version")) is not int
            or ledger.get("schema_version") != 2):
        failures.add("claim_mapping_ledger_schema_invalid")
        ledger = {}
    if (ledger.get("eligible_manifest_sha256") != eligible_sha
            or ledger.get("eligible_dataset_id") != claim_input.get("eligible_dataset_id")
            or ledger.get("claim_policy_sha256") != claim_input.get("claim_policy_sha256")
            or ledger.get("occurrence_policy") != claim_input.get("occurrence_policy")):
        failures.add("claim_input_to_mapping_header_mismatch")

    input_rows = claim_input.get("claims")
    input_required = {"video_id", "claim_id", "claim_text", "claim_text_sha256", "model_revision", "generation_run_sha256"}
    input_by_video = {}
    if not isinstance(input_rows, list):
        failures.add("claim_input_rows_invalid")
        input_rows = []
    for row in input_rows:
        if (not isinstance(row, dict) or set(row) != input_required
                or not _text(row.get("video_id")) or not _text(row.get("claim_id"))
                or not isinstance(row.get("claim_text"), str) or not row["claim_text"].strip()
                or not _sha(row.get("claim_text_sha256"))
                or not isinstance(row.get("model_revision"), str)
                or MODEL_REVISION.fullmatch(row["model_revision"]) is None
                or not _sha(row.get("generation_run_sha256"))):
            failures.add("claim_input_row_invalid")
            continue
        try:
            claim_text_sha = hashlib.sha256(row["claim_text"].encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            failures.add("claim_input_text_invalid_unicode")
            claim_text_sha = None
        if claim_text_sha != row["claim_text_sha256"]:
            failures.add("claim_input_text_hash_mismatch")
        if row["video_id"] in input_by_video:
            failures.add("claim_input_duplicate_video")
        input_by_video[row["video_id"]] = row
        protocol_model = protocol.get("model") if isinstance(protocol, dict) else None
        if (not isinstance(protocol_model, dict)
                or not isinstance(protocol_model.get("revision"), str)
                or not row["model_revision"].endswith("@" + protocol_model["revision"])):
            failures.add("claim_input_model_revision_mismatch")
    if set(input_by_video) != eligible_ids:
        failures.add("claim_input_eligible_coverage_mismatch")

    ledger_rows = ledger.get("claims")
    if not isinstance(ledger_rows, list):
        failures.add("claim_ledger_rows_invalid")
        ledger_rows = []
    ledger_by_video = {}
    for row in ledger_rows:
        if not isinstance(row, dict) or not isinstance(row.get("video_id"), str):
            continue
        if row["video_id"] in ledger_by_video:
            failures.add("claim_ledger_duplicate_video")
        ledger_by_video[row["video_id"]] = row
        source = input_by_video.get(row["video_id"])
        if source is None:
            continue
        for field in ("claim_id", "claim_text_sha256", "model_revision", "generation_run_sha256"):
            if row.get(field) != source.get(field):
                failures.add("claim_input_to_mapping_row_mismatch")
                break
    if set(ledger_by_video) != eligible_ids:
        failures.add("claim_ledger_eligible_coverage_mismatch")
    if not isinstance(eligible_raw, bytes) or not isinstance(ledger_raw, bytes):
        failures.add("claim_ledger_audit_raw_bytes_invalid")
    else:
        ledger_result = audit_claim_ledger(eligible, eligible_raw, ledger, ledger_raw)
        if not ledger_result.get("structurally_valid"):
            failures.add("claim_mapping_ledger_audit_failed")

    review_keys = {"schema_version", "mapping_ledger_sha256", "records"}
    if (not isinstance(review_index, dict) or set(review_index) != review_keys
            or type(review_index.get("schema_version")) is not int
            or review_index.get("schema_version") != 1):
        failures.add("review_evidence_index_schema_invalid")
        review_index = {}
    if review_index.get("mapping_ledger_sha256") != ledger_sha:
        failures.add("review_index_mapping_hash_mismatch")
    records = review_index.get("records")
    if not isinstance(records, list):
        failures.add("review_index_records_invalid")
        records = []
    expected_reviews = {}
    for video_id, row in ledger_by_video.items():
        review = row.get("review")
        if isinstance(review, dict) and isinstance(review.get("review_record_sha256"), str):
            expected_reviews[video_id] = (row.get("claim_id"), review.get("review_record_sha256"))
    record_keys = {"video_id", "claim_id", "review_record_sha256", "private_locator"}
    observed_reviews = {}
    observed_review_hashes = set()
    for record in records:
        if (not isinstance(record, dict) or set(record) != record_keys
                or not _text(record.get("video_id")) or not _text(record.get("claim_id"))
                or not _sha(record.get("review_record_sha256"))
                or not _text(record.get("private_locator"))):
            failures.add("review_evidence_record_invalid")
            continue
        if record["video_id"] in observed_reviews:
            failures.add("review_evidence_duplicate_video")
        if record["review_record_sha256"] in observed_review_hashes:
            failures.add("review_evidence_duplicate_record_hash")
        observed_review_hashes.add(record["review_record_sha256"])
        observed_reviews[record["video_id"]] = (record["claim_id"], record["review_record_sha256"])
    if observed_reviews != expected_reviews or set(observed_reviews) != eligible_ids:
        failures.add("review_evidence_ledger_link_mismatch")

    return {
        "schema_version": 1,
        "purpose": "private_claim_review_hash_crosscheck",
        "structural_status": "passed" if not failures else "failed",
        "semantic_validation": "not_machine_verifiable",
        "execution_authorization": "not_granted",
        "requested_eligible_video_count": len(eligible_ids),
        "effective_claim_input_count": len(input_by_video),
        "effective_claim_ledger_count": len(ledger_by_video),
        "effective_review_record_count": len(observed_reviews),
        "failure_codes": sorted(failures),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--eligible-manifest", required=True, type=Path)
    parser.add_argument("--claim-input-manifest", required=True, type=Path)
    parser.add_argument("--claim-mapping-ledger", required=True, type=Path)
    parser.add_argument("--review-evidence-index", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        protocol, protocol_raw = _read(args.protocol)
        eligible, eligible_raw = _read(args.eligible_manifest)
        claim_input, claim_input_raw = _read(args.claim_input_manifest)
        ledger, ledger_raw = _read(args.claim_mapping_ledger)
        review_index, review_index_raw = _read(args.review_evidence_index)
        result = audit(protocol, protocol_raw, eligible, eligible_raw,
                       claim_input, claim_input_raw, ledger, ledger_raw,
                       review_index, review_index_raw)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, RecursionError):
        result = {
            "schema_version": 1, "purpose": "private_claim_review_hash_crosscheck",
            "structural_status": "failed", "semantic_validation": "not_machine_verifiable",
            "execution_authorization": "not_granted",
            "requested_eligible_video_count": 0, "effective_claim_input_count": 0,
            "effective_claim_ledger_count": 0, "effective_review_record_count": 0,
            "failure_codes": ["private_input_invalid"],
        }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["structural_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
