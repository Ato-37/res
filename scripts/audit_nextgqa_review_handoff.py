#!/usr/bin/env python3
"""Audit a private anonymous review export and custodian media crosswalk.

This only verifies bytes, token coverage, and interval structure. It cannot
authenticate a human review or authorize faithfulness scoring.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from fractions import Fraction
import stat
from typing import Any

from build_nextgqa_claim_mapping_packet import FROZEN_POLICY_SHA256
from render_nextgqa_claim_review_html import render

TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")
DECISIONS = {"supports_claim", "partial", "contradicted", "unsupported", "ambiguous", "unlocalizable"}


def _json(raw: bytes) -> Any:
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate_json_key")
            out[k] = v
        return out
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def _utc(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)
    except ValueError:
        return False


def _utc_leq(left: Any, right: Any) -> bool:
    if not _utc(left) or not _utc(right):
        return False
    a = datetime.fromisoformat(left.replace("Z", "+00:00"))
    b = datetime.fromisoformat(right.replace("Z", "+00:00"))
    return a <= b


def audit(worksheet: Any, worksheet_raw: bytes, lineage: Any, lineage_raw: bytes,
          review: Any, review_raw: bytes, index: Any, index_raw: bytes,
          expected_lineage_sha256: str) -> dict:
    failures: set[str] = set()
    bindings = {
        "worksheet_sha256": hashlib.sha256(worksheet_raw).hexdigest(),
        "lineage_manifest_sha256": hashlib.sha256(lineage_raw).hexdigest(),
        "review_payload_sha256": hashlib.sha256(review_raw).hexdigest(),
        "review_index_sha256": hashlib.sha256(index_raw).hexdigest(),
    }
    try:
        # Validate the frozen worksheet through the production renderer without
        # granting it a media mapping; custody is audited separately below.
        rows = worksheet.get("records", []) if isinstance(worksheet, dict) else []
        validation_custody = {
            "schema_version": 1, "purpose": "private_custodian_token_to_verified_media_manifest",
            "created_at_utc": "2026-10-16T00:00:00Z",
            "worksheet_sha256": hashlib.sha256(worksheet_raw).hexdigest(),
            "source_lineage_manifest_sha256": hashlib.sha256(lineage_raw).hexdigest(),
            "claim_review_policy_sha256": worksheet.get("claim_review_policy_sha256"),
            "record_count": len(rows),
            "records": [{"review_token": row.get("review_token"), "row_ordinal": i,
                         "media_file": str(row.get("review_token")) + ".mp4",
                         "source_media_sha256": row.get("video_sha256"),
                         "staged_media_sha256": row.get("video_sha256"), "size_bytes": 1,
                         "media_check_salt": "00" * 32, "media_check_sha256": "00" * 32}
                        for i, row in enumerate(rows, 1) if isinstance(row, dict)]}
        validation_raw = json.dumps(validation_custody, sort_keys=True).encode()
        render(worksheet, worksheet_raw, lineage, lineage_raw, expected_lineage_sha256,
               validation_custody, validation_raw, hashlib.sha256(validation_raw).hexdigest())
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError, RecursionError, OverflowError, AttributeError):
        return {"status": "invalid", "failure_codes": ["frozen_worksheet_invalid"],
                "reviewed_records": 0, "execution_authorization": "not_granted",
                "semantic_review_authenticated": False, **bindings}
    try:
        parsed_review, parsed_index = _json(review_raw), _json(index_raw)
        if parsed_review != review or parsed_index != index:
            failures.add("payload_bytes_mismatch")
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        failures.add("payload_json_invalid")
    if not isinstance(review, dict) or not isinstance(index, dict):
        failures.add("handoff_payload_invalid")
        return {"status": "invalid", "failure_codes": sorted(failures),
                "reviewed_records": 0, "execution_authorization": "not_granted",
                "semantic_review_authenticated": False, **bindings}
    index_keys = {"schema_version", "purpose", "review_payload_sha256", "claim_review_policy_sha256",
                  "created_at_utc", "review_status", "execution_authorization"}
    if (set(index) != index_keys or index.get("schema_version") != 1
            or index.get("purpose") != "anonymous_unvalidated_human_review_handoff"
            or index.get("review_status") != "unvalidated_requires_custodian_audit"
            or index.get("execution_authorization") != "not_granted"):
        failures.add("handoff_index_invalid")
    if index.get("review_payload_sha256") != hashlib.sha256(review_raw).hexdigest():
        failures.add("review_payload_hash_mismatch")
    if (index.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
            or review.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
            or worksheet.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
            or review.get("claim_review_policy_id") != "nextgqa_claim_review_v6"):
        failures.add("policy_hash_mismatch")
    if not _utc(index.get("created_at_utc")):
        failures.add("review_timestamp_invalid")
    top_keys = {"schema_version", "purpose", "status", "review_completed", "execution_authorization",
                "claim_review_policy_id", "claim_review_policy_sha256", "human_reviewer_role",
                "human_reviewer_private_id", "review_environment", "independence_attestation",
                "export_disclosure_acknowledged", "records"}
    env = review.get("review_environment")
    if (set(review) != top_keys or review.get("schema_version") != 3
            or review.get("purpose") != "anonymous_local_claim_video_human_review"
            or review.get("status") != "human_review_recorded_unvalidated"
            or review.get("review_completed") is not True
            or review.get("execution_authorization") != "not_granted"
            or review.get("human_reviewer_role") != "independent_human"
            or review.get("export_disclosure_acknowledged") is not True
            or review.get("independence_attestation") != "reviewer_asserts_independence_from_generation_scoring_and_custody_and_did_not_view_custody_crosswalk"
            or not isinstance(review.get("human_reviewer_private_id"), str)
            or not review["human_reviewer_private_id"].strip()
            or not isinstance(env, dict)
            or set(env) != {"browser_user_agent", "offline_static_page", "network_fetch_available"}
            or not isinstance(env.get("browser_user_agent"), str) or not env.get("browser_user_agent")
            or env.get("offline_static_page") is not True or env.get("network_fetch_available") is not False):
        failures.add("review_payload_schema_invalid")
    expected = worksheet.get("records", [])
    actual = review.get("records")
    if not isinstance(actual, list) or len(actual) != 3 or len(expected) != 3:
        failures.add("review_record_count_invalid")
        actual = []
    for source, record in zip(expected, actual, strict=False):
        if not isinstance(record, dict) or set(record) != {"review_token", "review"}:
            failures.add("review_row_schema_invalid"); continue
        token, item = record.get("review_token"), record.get("review")
        if token != source.get("review_token") or not isinstance(token, str) or not TOKEN.fullmatch(token):
            failures.add("review_token_binding_invalid")
        keys = {"full_source_video_reviewed", "question_and_claim_only_context_shown",
                "gold_and_seed_hidden_until_mapping_frozen", "outcome_scores_not_accessed",
                "media_bytes_verified", "occurrence_policy_applied", "support_decision", "evidence_intervals_seconds",
                "interval_rationales", "reviewer_role", "first_played_at_utc",
                "reviewed_at_utc", "review_status",
                "display_player", "claim_review_policy_sha256", "media_duration_seconds"}
        if (not isinstance(item, dict) or set(item) != keys
                or item.get("review_status") != "completed"
                or item.get("reviewer_role") != "independent_human"
                or any(item.get(k) is not True for k in ("full_source_video_reviewed", "question_and_claim_only_context_shown",
                       "gold_and_seed_hidden_until_mapping_frozen", "outcome_scores_not_accessed"))
                or item.get("media_bytes_verified") is not True
                or item.get("occurrence_policy_applied") != "all_relevant_occurrences"
                or item.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
                or not _utc_leq(item.get("first_played_at_utc"), item.get("reviewed_at_utc"))
                or not _utc_leq(item.get("reviewed_at_utc"), index.get("created_at_utc"))
                or not isinstance(item.get("display_player"), str) or not item.get("display_player")):
            failures.add("review_declaration_incomplete"); continue
        intervals, rationales, decision = item.get("evidence_intervals_seconds"), item.get("interval_rationales"), item.get("support_decision")
        duration = item.get("media_duration_seconds")
        if (not isinstance(decision, str) or decision not in DECISIONS
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
                or not isinstance(intervals, list) or not isinstance(rationales, list)
                or len(intervals) != len(rationales) or (decision == "supports_claim" and not intervals)):
            failures.add("review_intervals_invalid"); continue
        previous_end = -1.0
        seen_intervals = set()
        for span, rationale in zip(intervals, rationales, strict=True):
            if (not isinstance(span, list) or len(span) != 2
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) for x in span)):
                failures.add("review_intervals_invalid"); break
            pair = tuple(span)
            duplicate = pair in seen_intervals
            if (span[0] < 0 or span[1] <= span[0] or span[1] > duration
                    or (not duplicate and span[0] <= previous_end)
                    or any((Fraction(str(x)) * 1000).denominator != 1 for x in span)
                    or not isinstance(rationale, str) or not rationale.strip()):
                failures.add("review_intervals_invalid"); break
            seen_intervals.add(pair)
            previous_end = max(previous_end, span[1])
    return {"status": "unvalidated_handoff_structurally_valid" if not failures else "invalid",
            "failure_codes": sorted(failures), "reviewed_records": 3 if not failures else 0,
            "execution_authorization": "not_granted", "semantic_review_authenticated": False,
            **bindings}


def audit_custody(worksheet: Any, worksheet_raw: bytes, lineage: Any, lineage_raw: bytes,
                  custody: Any, custody_raw: bytes, media_dir: Path,
                  expected_custody_sha256: str) -> list[str]:
    failures = set()
    if (not isinstance(worksheet, dict) or not isinstance(worksheet.get("records"), list)
            or not isinstance(lineage, dict) or not isinstance(lineage.get("records"), list)):
        return ["custody_source_documents_invalid"]
    if hashlib.sha256(custody_raw).hexdigest() != expected_custody_sha256:
        failures.add("custody_manifest_hash_mismatch")
    keys = {"schema_version", "purpose", "created_at_utc", "worksheet_sha256",
            "source_lineage_manifest_sha256", "claim_review_policy_sha256", "record_count", "records"}
    if (not isinstance(custody, dict) or set(custody) != keys or custody.get("schema_version") != 1
            or custody.get("purpose") != "private_custodian_token_to_verified_media_manifest"
            or custody.get("record_count") != 3 or not _utc(custody.get("created_at_utc"))
            or custody.get("worksheet_sha256") != hashlib.sha256(worksheet_raw).hexdigest()
            or custody.get("source_lineage_manifest_sha256") != hashlib.sha256(lineage_raw).hexdigest()
            or custody.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256):
        failures.add("custody_manifest_invalid")
        return sorted(failures)
    rows = custody.get("records")
    expected = worksheet.get("records", [])
    if not isinstance(rows, list) or len(rows) != 3:
        return ["custody_record_count_invalid"]
    if (media_dir.is_symlink() or not media_dir.is_dir() or stat.S_IMODE(media_dir.stat().st_mode) != 0o700
            or media_dir.stat().st_uid != os.geteuid()):
        failures.add("custody_media_directory_invalid")
    expected_tokens = [r.get("review_token") for r in expected]
    if any(not isinstance(token, str) or not TOKEN.fullmatch(token) for token in expected_tokens):
        return sorted(failures | {"custody_worksheet_token_invalid"})
    observed_tokens = [r.get("review_token") for r in rows if isinstance(r, dict)]
    if observed_tokens != expected_tokens:
        failures.add("custody_token_coverage_invalid")
    expected_files = {token + ".mp4" for token in expected_tokens}
    try:
        if {p.name for p in media_dir.iterdir()} != expected_files:
            failures.add("custody_media_directory_contents_invalid")
    except OSError:
        failures.add("custody_media_directory_unreadable")
    for ordinal, (source, row) in enumerate(zip(expected, rows, strict=True), start=1):
        row_keys = {"review_token", "row_ordinal", "media_file", "source_media_sha256", "staged_media_sha256", "size_bytes",
                    "media_check_salt", "media_check_sha256"}
        if not isinstance(row, dict) or set(row) != row_keys:
            failures.add("custody_row_invalid"); continue
        filename = source.get("review_token") + ".mp4"
        if (row.get("review_token") != source.get("review_token") or row.get("row_ordinal") != ordinal
                or row.get("media_file") != filename or row.get("source_media_sha256") != source.get("video_sha256")
                or row.get("staged_media_sha256") != source.get("video_sha256")):
            failures.add("custody_row_binding_invalid"); continue
        if (not TOKEN.fullmatch(row.get("review_token", "")) or Path(filename).name != filename
                or not re.fullmatch(r"[a-f0-9]{64}", str(row.get("media_check_salt", "")))
                or not re.fullmatch(r"[a-f0-9]{64}", str(row.get("media_check_sha256", "")))):
            failures.add("custody_media_token_invalid"); continue
        path = media_dir / filename
        try:
            info = path.lstat()
            if (path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or path.resolve().parent != media_dir.resolve()):
                failures.add("custody_media_file_invalid"); continue
            raw = path.read_bytes()
            if len(raw) != row.get("size_bytes") or hashlib.sha256(raw).hexdigest() != source.get("video_sha256"):
                failures.add("custody_media_hash_mismatch")
            salt = bytes.fromhex(row["media_check_salt"])
            if hashlib.sha256(salt + raw).hexdigest() != row["media_check_sha256"]:
                failures.add("custody_media_check_digest_mismatch")
        except OSError:
            failures.add("custody_media_file_missing")
    return sorted(failures)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("worksheet", "lineage-manifest", "review-payload", "review-index", "custody-manifest", "media-dir"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--expected-lineage-sha256", required=True)
    p.add_argument("--expected-custody-sha256", required=True)
    p.add_argument("--output", required=True, help="summary JSON outside the repository")
    a = p.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    target = Path(a.output).resolve()
    if target.is_relative_to(root):
        p.error("output_must_be_outside_repository")
    paths = [Path(a.worksheet), Path(a.lineage_manifest), Path(a.review_payload), Path(a.review_index), Path(a.custody_manifest)]
    try:
        if any(stat.S_IMODE(x.stat().st_mode) != 0o600 or x.stat().st_uid != os.geteuid()
               for x in paths):
            raise ValueError("private_input_mode_invalid")
        raw = [x.read_bytes() for x in paths]
        values = [_json(x) for x in raw]
        result = audit(values[0], raw[0], values[1], raw[1], values[2], raw[2],
                       values[3], raw[3], a.expected_lineage_sha256)
        custody_failures = (audit_custody(values[0], raw[0], values[1], raw[1], values[4], raw[4],
                                          Path(a.media_dir), a.expected_custody_sha256)
                            if "frozen_worksheet_invalid" not in result["failure_codes"] else [])
        result["failure_codes"] = sorted(set(result["failure_codes"]) | set(custody_failures))
        if custody_failures:
            result.update(status="invalid", reviewed_records=0)
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        p.error(str(exc).split(":", 1)[0] or "handoff_audit_failed")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "unvalidated_handoff_structurally_valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
