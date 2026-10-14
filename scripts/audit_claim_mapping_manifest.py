"""Privacy-safe structural check for private claim-to-evidence review records.

This CPU-only gate checks declarations needed before claim-mask scoring. It
cannot authenticate a model output or human review, determine whether an
interval semantically supports a claim, or authorize inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"^[a-f0-9]{64}$")
UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
MODEL_REVISION = re.compile(r"^.+@[a-f0-9]{40}$")
OCCURRENCE_POLICIES = {"one_sufficient_occurrence", "all_relevant_occurrences"}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
    )
    if _contains_nonfinite(value):
        raise ValueError("non-finite JSON number")
    return value, raw


def _matches_raw(value: Any, raw: bytes) -> bool:
    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
        )
        if _contains_nonfinite(parsed) or _contains_nonfinite(value):
            return False
        left = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        right = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return left == right
    except (AttributeError, UnicodeError, ValueError, TypeError, OverflowError):
        return False


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(key) or _contains_nonfinite(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _utc(value: Any) -> bool:
    if not isinstance(value, str) or not UTC.fullmatch(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False


def _intervals(value: Any) -> tuple[bool, tuple[tuple[float, float], ...]]:
    if not isinstance(value, list) or not value:
        return False, ()
    parsed = []
    for interval in value:
        if not isinstance(interval, list) or len(interval) != 2:
            return False, ()
        start, end = interval
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or start < 0 or end <= start):
            return False, ()
        try:
            float_start, float_end = float(start), float(end)
            if not math.isfinite(float_start) or not math.isfinite(float_end) or float_end <= float_start:
                return False, ()
        except (OverflowError, ValueError):
            return False, ()
        parsed.append((float(start), float(end)))
    if parsed != sorted(parsed) or len(parsed) != len(set(parsed)):
        return False, ()
    return True, tuple(parsed)


def _intervals_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _eligible_ids(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("eligible manifest invalid")
    ids = []
    for row in value:
        if (not isinstance(row, dict) or not isinstance(row.get("video_id"), str)
                or not row["video_id"].strip() or row["video_id"] != row["video_id"].strip()):
            raise ValueError("eligible manifest invalid")
        ids.append(row["video_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("eligible manifest invalid")
    return ids


def _review_row(row: Any, policy: str, scoring_started: Any, failures: set[str]) -> str | None:
    required = {
        "video_id", "claim_id", "claim_text_sha256", "claim_origin", "model_revision",
        "generation_run_sha256", "mapping_method", "qa_seed", "evidence_intervals", "review",
    }
    if not isinstance(row, dict) or set(row) != required:
        failures.add("claim_row_schema_invalid")
        return None
    video_id = row.get("video_id")
    if (not isinstance(video_id, str) or not video_id.strip()
            or video_id != video_id.strip()):
        failures.add("claim_row_identifier_invalid")
        video_id = None
    if (not isinstance(row.get("claim_id"), str) or not row["claim_id"].strip()
            or not _sha(row.get("claim_text_sha256"))
            or row.get("claim_origin") != "model_generated_frozen"
            or not isinstance(row.get("model_revision"), str)
            or MODEL_REVISION.fullmatch(row["model_revision"]) is None
            or not _sha(row.get("generation_run_sha256"))):
        failures.add("frozen_model_claim_missing")
    if row.get("mapping_method") != "independent_claim_review":
        failures.add("claim_specific_remapping_missing")

    seed = row.get("qa_seed")
    if not isinstance(seed, dict) or set(seed) != {"usage", "intervals"}:
        failures.add("qa_seed_declaration_invalid")
    elif seed.get("usage") == "not_used":
        if seed.get("intervals") != []:
            failures.add("qa_seed_declaration_invalid")
    elif seed.get("usage") == "seed_only":
        valid_seed, _ = _intervals(seed.get("intervals"))
        if not valid_seed:
            failures.add("qa_seed_declaration_invalid")
    else:
        failures.add("qa_span_copy_or_unknown_use")

    valid_intervals, intervals = _intervals(row.get("evidence_intervals"))
    interval_digest = _intervals_sha256(row.get("evidence_intervals")) if valid_intervals else None
    if not valid_intervals:
        failures.add("claim_evidence_intervals_invalid")
    elif policy == "one_sufficient_occurrence" and len(intervals) != 1:
        failures.add("occurrence_policy_mismatch")

    review = row.get("review")
    review_required = {
        "status", "reviewer_role", "review_record_sha256", "reviewed_at_utc",
        "claim_text_sha256_seen", "evidence_intervals_sha256_seen", "video_reviewed", "answer_and_gold_hidden",
        "qa_seed_hidden_until_mapping_frozen", "outcome_scores_seen", "claim_support_decision", "occurrence_policy_applied",
    }
    if not isinstance(review, dict) or set(review) != review_required:
        failures.add("independent_human_review_incomplete")
    else:
        if (review.get("status") != "completed" or review.get("reviewer_role") != "independent_human"
                or not _sha(review.get("review_record_sha256"))
                or not _utc(review.get("reviewed_at_utc"))
                or review.get("claim_text_sha256_seen") != row.get("claim_text_sha256")
                or not _sha(review.get("evidence_intervals_sha256_seen"))
                or review.get("evidence_intervals_sha256_seen") != interval_digest
                or review.get("video_reviewed") is not True
                or review.get("answer_and_gold_hidden") is not True
                or review.get("qa_seed_hidden_until_mapping_frozen") is not True
                or review.get("outcome_scores_seen") is not False
                or review.get("occurrence_policy_applied") != policy):
            failures.add("independent_human_review_incomplete")
        if scoring_started is not None and _utc(scoring_started) and _utc(review.get("reviewed_at_utc")):
            if review["reviewed_at_utc"] >= scoring_started:
                failures.add("review_not_before_scoring")
    if (isinstance(review, dict) and "claim_support_decision" in review
            and review["claim_support_decision"] != "supports_claim"):
        failures.add("claim_support_not_established")
    return video_id if isinstance(video_id, str) else None


def audit(eligible: Any, eligible_raw: bytes, ledger: Any, ledger_raw: bytes) -> dict:
    failures: set[str] = set()
    raw_bytes_valid = isinstance(eligible_raw, bytes) and isinstance(ledger_raw, bytes)
    if not raw_bytes_valid:
        failures.add("raw_input_bytes_invalid")
    if not _matches_raw(eligible, eligible_raw) or not _matches_raw(ledger, ledger_raw):
        failures.add("parsed_input_raw_bytes_mismatch")
    if _contains_nonfinite(eligible) or _contains_nonfinite(ledger):
        failures.add("nonfinite_json_number")
    try:
        ids = _eligible_ids(eligible)
    except ValueError:
        ids = []
        failures.add("eligible_manifest_invalid")
    required = {
        "schema_version", "eligible_dataset_id", "eligible_manifest_sha256",
        "claim_policy_sha256", "occurrence_policy", "scoring_started_at_utc", "claims",
    }
    if not isinstance(ledger, dict) or set(ledger) != required:
        failures.add("claim_ledger_schema_invalid")
        return _summary(failures, len(ids), 0)
    if type(ledger.get("schema_version")) is not int or ledger["schema_version"] != 2:
        failures.add("claim_ledger_version_invalid")
    if not isinstance(ledger.get("eligible_dataset_id"), str) or not ledger["eligible_dataset_id"].strip():
        failures.add("eligible_dataset_id_missing")
    if (not raw_bytes_valid or not _sha(ledger.get("eligible_manifest_sha256"))
            or ledger["eligible_manifest_sha256"] != hashlib.sha256(eligible_raw).hexdigest()):
        failures.add("eligible_manifest_hash_mismatch")
    if not _sha(ledger.get("claim_policy_sha256")):
        failures.add("claim_policy_hash_invalid")
    policy = ledger.get("occurrence_policy")
    if not isinstance(policy, str) or policy not in OCCURRENCE_POLICIES:
        failures.add("occurrence_policy_missing_or_invalid")
        policy = ""
    scoring_started = ledger.get("scoring_started_at_utc")
    if scoring_started is not None and not _utc(scoring_started):
        failures.add("scoring_timestamp_invalid")
    rows = ledger.get("claims")
    if not isinstance(rows, list):
        failures.add("claim_rows_not_array")
        rows = []
    row_ids = []
    seen_claims = set()
    for row in rows:
        row_id = _review_row(row, policy, scoring_started, failures)
        if row_id is not None:
            row_ids.append(row_id)
        if isinstance(row, dict):
            claim_id = row.get("claim_id")
            if isinstance(claim_id, str):
                key = (row_id, claim_id)
                if key in seen_claims:
                    failures.add("claim_ids_duplicate")
                seen_claims.add(key)
    if len(row_ids) != len(set(row_ids)):
        failures.add("duplicate_video_claim_rows")
    if set(row_ids) - set(ids):
        failures.add("claim_rows_outside_eligible_cohort")
    if set(ids) - set(row_ids):
        failures.add("claim_rows_missing")
    if len(rows) != len(ids):
        failures.add("claim_row_count_mismatch")
    return _summary(failures, len(ids), len(rows))


def _summary(failures: set[str], eligible_n: int, row_n: int) -> dict:
    return {
        "schema_version": 1,
        "structurally_valid": not failures,
        "execution_authorization": "not_granted",
        "semantic_validation": "not_machine_verifiable",
        "eligible_video_count": eligible_n,
        "claim_row_count": row_n,
        "failure_codes": sorted(failures),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible-manifest", required=True, type=Path)
    parser.add_argument("--claim-ledger", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        eligible, eligible_raw = _read_json(args.eligible_manifest)
        ledger, ledger_raw = _read_json(args.claim_ledger)
        result = audit(eligible, eligible_raw, ledger, ledger_raw)
    except Exception:
        result = _summary({"input_unreadable_or_malformed"}, 0, 0)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["structurally_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
