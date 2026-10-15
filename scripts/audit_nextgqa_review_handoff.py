#!/usr/bin/env python3
"""Structurally audit a private offline human-review handoff; never authorize scoring."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from build_nextgqa_claim_mapping_packet import FROZEN_POLICY_SHA256
from render_nextgqa_claim_review_html import render


INDEX_KEYS = {"schema_version", "purpose", "review_payload_sha256", "source_worksheet_sha256",
              "source_lineage_manifest_sha256",
              "claim_review_policy_sha256", "created_at_utc", "review_status",
              "execution_authorization"}
REVIEW_KEYS = {
    "source_identity_verified", "full_source_video_reviewed",
    "question_and_claim_only_context_shown", "gold_and_seed_hidden_until_mapping_frozen",
    "outcome_scores_not_accessed", "occurrence_policy_applied", "support_decision",
    "evidence_intervals_seconds", "interval_rationales", "reviewer_role",
    "review_record_reference", "reviewed_at_utc", "review_status",
    "claim_text_sha256_seen", "media_sha256_verified", "media_sha256_observed",
    "media_duration_seconds", "media_mime_type", "cohort_lineage_reviewed",
    "display_player", "claim_review_policy_sha256",
    "source_lineage_manifest_sha256", "source_lineage",
}
DECISIONS = {"supports_claim", "partial", "contradicted", "unsupported", "ambiguous", "unlocalizable"}
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _read(path: Path):
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    return value, raw


def _matches_raw(value: Any, raw: bytes) -> bool:
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        return parsed == value
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError, RecursionError):
        return False


def _utc(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
        return True
    except ValueError:
        return False


def audit(worksheet: Any, worksheet_raw: bytes, lineage: Any, lineage_raw: bytes,
          review: Any, review_raw: bytes, index: Any, index_raw: bytes,
          expected_lineage_sha256: str) -> dict:
    failures: set[str] = set()
    try:
        render(worksheet, worksheet_raw, lineage, lineage_raw, expected_lineage_sha256)
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError, RecursionError, OverflowError):
        failures.add("frozen_worksheet_invalid")
    if "frozen_worksheet_invalid" in failures:
        return {"status": "invalid", "failure_codes": sorted(failures),
                "reviewed_records": 0, "execution_authorization": "not_granted",
                "semantic_review_authenticated": False}
    if not _matches_raw(review, review_raw):
        failures.add("review_payload_bytes_mismatch")
    if not _matches_raw(index, index_raw):
        failures.add("handoff_index_bytes_mismatch")
    if (not isinstance(index, dict) or set(index) != INDEX_KEYS
            or index.get("schema_version") != 1
            or index.get("purpose") != "unvalidated_private_human_review_handoff"
            or index.get("review_status") != "unvalidated_requires_independent_audit"
            or index.get("execution_authorization") != "not_granted"):
        failures.add("handoff_index_invalid")
    if (not isinstance(review, dict) or not isinstance(worksheet, dict)
            or not isinstance(index, dict)):
        failures.add("handoff_payload_invalid")
        return {"status": "invalid", "failure_codes": sorted(failures),
                "reviewed_records": 0, "execution_authorization": "not_granted"}

    digest = hashlib.sha256(review_raw).hexdigest()
    if index.get("review_payload_sha256") != digest:
        failures.add("review_payload_hash_mismatch")
    if index.get("source_worksheet_sha256") != hashlib.sha256(worksheet_raw).hexdigest():
        failures.add("worksheet_hash_mismatch")
    lineage_hash = hashlib.sha256(lineage_raw).hexdigest()
    if (index.get("source_lineage_manifest_sha256") != lineage_hash
            or review.get("source_lineage_manifest_sha256") != lineage_hash):
        failures.add("lineage_manifest_hash_mismatch")
    policy_hash = worksheet.get("claim_review_policy_sha256")
    if (index.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
            or review.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
            or policy_hash != FROZEN_POLICY_SHA256):
        failures.add("policy_hash_mismatch")
    if not _utc(index.get("created_at_utc")):
        failures.add("review_timestamp_invalid")

    expected_review_keys = {k for k in worksheet if k != "records"} | {
        "source_lineage_manifest_sha256", "human_reviewer_role",
        "human_reviewer_private_id", "review_environment", "records"
    }
    if set(review) != expected_review_keys or review.get("status") != "human_review_recorded_unvalidated":
        failures.add("review_payload_schema_invalid")
    if (review.get("execution_authorization") != "not_granted"
            or review.get("review_completed") is not True
            or review.get("human_reviewer_role") != "independent_human"
            or not isinstance(review.get("human_reviewer_private_id"), str)
            or not review.get("human_reviewer_private_id", "").strip()):
        failures.add("review_attestation_incomplete")
    env = review.get("review_environment")
    if (not isinstance(env, dict)
            or set(env) != {"browser_user_agent", "offline_static_page", "network_fetch_available"}
            or not isinstance(env.get("browser_user_agent"), str)
            or not env.get("browser_user_agent")
            or env.get("offline_static_page") is not True
            or env.get("network_fetch_available") is not False):
        failures.add("review_environment_invalid")
    if (not isinstance(review.get("records"), list)
            or len(review["records"]) != 3
            or not isinstance(worksheet.get("records"), list)
            or len(worksheet["records"]) != 3):
        failures.add("review_record_count_invalid")
        return {"status": "invalid", "failure_codes": sorted(failures),
                "reviewed_records": 0, "execution_authorization": "not_granted"}

    row_keys = {k for k in worksheet["records"][0] if k != "video_path"}
    lineage_by_claim = {r["claim_id"]: r for r in lineage["records"]}
    for expected, row in zip(worksheet["records"], review["records"], strict=True):
        if not isinstance(row, dict) or set(row) != row_keys:
            failures.add("review_row_schema_invalid"); continue
        if any(row.get(k) != expected.get(k) for k in row_keys - {"review"}):
            failures.add("review_row_binding_mismatch")
        r = row.get("review")
        if not isinstance(r, dict) or set(r) != REVIEW_KEYS:
            failures.add("review_declaration_invalid"); continue
        if (r.get("review_status") != "completed"
                or r.get("reviewer_role") != "independent_human"
                or r.get("source_identity_verified") is not True
                or r.get("cohort_lineage_reviewed") is not True
                or r.get("media_sha256_verified") is not True
                or r.get("media_sha256_observed") != row.get("video_sha256")
                or r.get("full_source_video_reviewed") is not True
                or r.get("question_and_claim_only_context_shown") is not True
                or r.get("gold_and_seed_hidden_until_mapping_frozen") is not True
                or r.get("outcome_scores_not_accessed") is not True
                or r.get("occurrence_policy_applied") != "all_relevant_occurrences"
                or r.get("claim_text_sha256_seen") != row.get("claim_text_sha256")
                or r.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
                or not _utc(r.get("reviewed_at_utc"))
                or r.get("reviewed_at_utc") != index.get("created_at_utc")
                or r.get("review_record_reference") is not None):
            failures.add("review_declaration_incomplete")
        expected_lineage = lineage_by_claim.get(expected.get("claim_id"))
        if (not isinstance(expected_lineage, dict)
                or r.get("source_lineage_manifest_sha256") != lineage_hash
                or r.get("source_lineage") != {k: expected_lineage[k] for k in (
                    "dataset_question_id", "dataset_video_id", "video_key",
                    "relative_video_path", "question_sha256")}):
            failures.add("review_lineage_binding_mismatch")
        decision = r.get("support_decision")
        intervals = r.get("evidence_intervals_seconds")
        rationales = r.get("interval_rationales")
        duration = r.get("media_duration_seconds")
        if (decision not in DECISIONS or isinstance(duration, bool)
                or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
                or not isinstance(intervals, list) or not isinstance(rationales, list)
                or len(intervals) != len(rationales)
                or (decision == "supports_claim" and len(intervals) == 0)):
            failures.add("review_intervals_invalid"); continue
        previous_end = -1.0
        for interval, rationale in zip(intervals, rationales, strict=True):
            if (not isinstance(interval, list) or len(interval) != 2
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) for x in interval)
                    or interval[0] < 0 or interval[1] <= interval[0]
                    or interval[1] > duration or interval[0] <= previous_end
                    or any(abs(x * 1000 - round(x * 1000)) > 1e-6 for x in interval)
                    or not isinstance(rationale, str) or not rationale.strip()):
                failures.add("review_intervals_invalid"); break
            previous_end = interval[1]

    return {
        "status": "unvalidated_handoff_structurally_valid" if not failures else "invalid",
        "failure_codes": sorted(failures),
        "reviewed_records": 3 if not failures else 0,
        "execution_authorization": "not_granted",
        "semantic_review_authenticated": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worksheet", "lineage-manifest", "review-payload", "review-index"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output", required=True, help="summary JSON outside the repository")
    parser.add_argument("--expected-lineage-sha256", required=True,
                        help="independently recorded digest of the audited lineage manifest")
    args = parser.parse_args(argv)
    paths = [Path(args.worksheet).resolve(), Path(args.lineage_manifest).resolve(),
             Path(args.review_payload).resolve(), Path(args.review_index).resolve()]
    target = Path(args.output).resolve()
    if target.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("output_must_be_outside_repository")
    try:
        values, raws = zip(*(_read(p) for p in paths), strict=True)
        report = audit(values[0], raws[0], values[1], raws[1], values[2], raws[2],
                       values[3], raws[3], args.expected_lineage_sha256)
        if any((p.stat().st_mode & 0o777) != 0o600 for p in paths):
            report["failure_codes"] = sorted(set(report["failure_codes"]) | {"private_input_mode_invalid"})
            report["status"] = "invalid"
            report["reviewed_records"] = 0
        raw = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "handoff_audit_failed")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "unvalidated_handoff_structurally_valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
