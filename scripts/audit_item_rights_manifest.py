"""Privacy-safe structural audit for a private item-level rights ledger.

This tool compares identifiers locally and reports counts, hashes,
structural-status fields, and failure codes. It cannot determine whether
license evidence or a reviewer decision is legally or factually correct, and
never authorizes inference. Dataset-level access or metadata agreements are
outside the v2 ledger and are explicitly reported as not machine verified.
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
ALLOWED_DECISIONS = {"permitted", "not_permitted", "unresolved"}
ALLOWED_USES = {"local_inference", "local_retention", "aggregate_publication", "redistribution"}


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
    )
    if _contains_nonfinite(value):
        raise ValueError("non-finite JSON number")
    return value, raw


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(key) or _contains_nonfinite(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _is_json_tree(value: Any) -> bool:
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_is_json_tree(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _is_json_tree(item) for key, item in value.items())
    return False


def _matches_raw(value: Any, raw: bytes) -> bool:
    if not isinstance(raw, bytes) or not _is_json_tree(value):
        return False
    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
        )
        if not _is_json_tree(parsed):
            return False
        left = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        right = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return left == right
    except (AttributeError, UnicodeError, ValueError, TypeError, OverflowError):
        return False


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _is_utc(value: Any) -> bool:
    if not isinstance(value, str) or not UTC.fullmatch(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _identifier(value: Any) -> bool:
    return _nonempty_text(value) and value == value.strip()


def _eligible_ids(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("eligible manifest must be a non-empty JSON array")
    ids = []
    for row in value:
        if not isinstance(row, dict) or not _identifier(row.get("video_id")):
            raise ValueError("eligible manifest row schema invalid")
        ids.append(row["video_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("eligible manifest video IDs are not unique")
    return ids


def _audit_annotation_rights(value: Any, intended_uses: list[str], failures: set[str]) -> bool:
    required = {"dataset_id", "license_id", "evidence_locator", "evidence_sha256", "review_record_sha256",
                "reviewer_role", "review_status", "reviewed_at_utc", "decisions"}
    if not isinstance(value, dict) or set(value) != required:
        failures.add("annotation_rights_schema_invalid")
        return False
    valid = True
    for field in ("dataset_id", "license_id", "evidence_locator", "reviewer_role"):
        if not _nonempty_text(value.get(field)):
            failures.add("annotation_rights_evidence_invalid")
            valid = False
    if (not _is_sha(value.get("evidence_sha256"))
            or not _is_sha(value.get("review_record_sha256"))
            or value.get("review_status") != "completed"
            or not _is_utc(value.get("reviewed_at_utc"))):
        failures.add("annotation_rights_evidence_invalid")
        valid = False
    if not isinstance(value.get("decisions"), dict) or set(value["decisions"]) != set(intended_uses):
        failures.add("annotation_use_decisions_incomplete")
        return False
    for decision in value["decisions"].values():
        if not isinstance(decision, str) or decision not in ALLOWED_DECISIONS:
            failures.add("annotation_use_decision_invalid")
            valid = False
        elif decision == "not_permitted":
            failures.add("annotation_use_not_permitted")
            valid = False
        elif decision == "unresolved":
            failures.add("annotation_use_unresolved")
            valid = False
    return valid


def _audit_source_lineage(row: dict, eligible_dataset_id: Any, failures: set[str]) -> bool:
    steps = row.get("source_lineage")
    required = {"from_dataset_id", "from_record_id", "to_dataset_id", "to_record_id",
                "mapping_evidence_locator", "mapping_evidence_sha256"}
    if not isinstance(steps, list) or not steps:
        failures.add("item_source_lineage_missing")
        return False
    valid = True
    normalized = []
    for step in steps:
        if not isinstance(step, dict) or set(step) != required:
            failures.add("item_source_lineage_schema_invalid")
            valid = False
            continue
        if (not all(_identifier(step.get(key)) for key in (
                "from_dataset_id", "from_record_id", "to_dataset_id", "to_record_id",
                "mapping_evidence_locator"))
                or not _is_sha(step.get("mapping_evidence_sha256"))):
            failures.add("item_source_lineage_evidence_invalid")
            valid = False
        normalized.append(step)
    for previous, current in zip(normalized, normalized[1:]):
        if (previous.get("to_dataset_id") != current.get("from_dataset_id")
                or previous.get("to_record_id") != current.get("from_record_id")):
            failures.add("item_source_lineage_disconnected")
            valid = False
    if normalized:
        if (normalized[0].get("from_dataset_id") != eligible_dataset_id
                or normalized[0].get("from_record_id") != row.get("video_id")
                or normalized[-1].get("to_dataset_id") != row.get("source_dataset_id")
                or normalized[-1].get("to_record_id") != row.get("source_video_id")):
            failures.add("item_source_lineage_endpoint_mismatch")
            valid = False
    return valid


def _audit_item(row: Any, eligible_dataset_id: Any, intended_uses: list[str], failures: set[str]) -> tuple[str | None, bool]:
    required = {"video_id", "source_dataset_id", "source_video_id", "media_license_id", "media_evidence_locator",
                "media_evidence_sha256", "media_review_record_sha256", "media_reviewer_role",
                "media_review_status", "media_reviewed_at_utc", "conditions_record_sha256",
                "conditions_status", "source_lineage", "decisions"}
    if not isinstance(row, dict) or set(row) != required:
        failures.add("item_rights_schema_invalid")
        return None, False
    video_id = row.get("video_id")
    if not _identifier(video_id):
        failures.add("item_rights_video_id_invalid")
        video_id = None
    if not all(_identifier(row.get(k)) for k in (
            "source_dataset_id", "source_video_id", "media_license_id",
            "media_evidence_locator", "media_reviewer_role")):
        failures.add("item_rights_source_evidence_invalid")
    if not _audit_source_lineage(row, eligible_dataset_id, failures):
        failures.add("item_rights_source_lineage_invalid")
    if (not _is_sha(row.get("media_evidence_sha256"))
            or not _is_sha(row.get("media_review_record_sha256"))
            or row.get("media_review_status") != "completed"
            or not _is_utc(row.get("media_reviewed_at_utc"))):
        failures.add("item_rights_source_evidence_invalid")
    if (not _is_sha(row.get("conditions_record_sha256"))
            or row.get("conditions_status") != "reviewed_resolved"):
        failures.add("item_rights_conditions_unresolved")
    decisions = row.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) != set(intended_uses):
        failures.add("item_use_decisions_incomplete")
    else:
        for decision in decisions.values():
            if not isinstance(decision, str) or decision not in ALLOWED_DECISIONS:
                failures.add("item_use_decision_invalid")
            elif decision == "not_permitted":
                failures.add("item_use_not_permitted")
            elif decision == "unresolved":
                failures.add("item_use_unresolved")
    covered = not failures
    return video_id if isinstance(video_id, str) else None, covered


def audit(eligible: Any, eligible_raw: bytes, ledger: Any, ledger_raw: bytes) -> dict:
    failures: set[str] = set()
    eligible_raw_bytes = eligible_raw if isinstance(eligible_raw, bytes) else b""
    ledger_raw_bytes = ledger_raw if isinstance(ledger_raw, bytes) else b""
    if (not _matches_raw(eligible, eligible_raw) or not _matches_raw(ledger, ledger_raw)):
        failures.add("parsed_input_raw_bytes_mismatch")
    try:
        ids = _eligible_ids(eligible)
    except ValueError:
        ids = []
        failures.add("eligible_manifest_invalid")

    if not isinstance(ledger, dict) or set(ledger) != {
        "schema_version", "eligible_dataset_id", "eligible_manifest_sha256", "intended_use_scope", "intended_uses",
        "annotation_rights", "items",
    }:
        failures.add("rights_ledger_schema_invalid")
        return _summary(failures, eligible_raw_bytes, ledger_raw_bytes, len(ids), 0, 0, 0)
    if type(ledger.get("schema_version")) is not int or ledger["schema_version"] != 2:
        failures.add("rights_ledger_version_invalid")
    if not _identifier(ledger.get("eligible_dataset_id")):
        failures.add("eligible_dataset_id_missing")
    if not _is_sha(ledger.get("eligible_manifest_sha256")) or ledger.get("eligible_manifest_sha256") != hashlib.sha256(eligible_raw_bytes).hexdigest():
        failures.add("eligible_manifest_hash_mismatch")
    if not _nonempty_text(ledger.get("intended_use_scope")):
        failures.add("intended_use_scope_missing")
    uses = ledger.get("intended_uses")
    if (not isinstance(uses, list) or not uses
            or any(not isinstance(use, str) or use not in ALLOWED_USES for use in uses)
            or (all(isinstance(use, str) for use in uses) and len(uses) != len(set(uses)))
            or "local_inference" not in uses):
        failures.add("intended_uses_invalid")
        uses = []

    annotation_ok = _audit_annotation_rights(ledger.get("annotation_rights"), uses, failures)
    rows = ledger.get("items")
    if not isinstance(rows, list):
        failures.add("rights_items_not_array")
        rows = []

    row_ids: list[str] = []
    item_covered = 0
    for row in rows:
        row_failures: set[str] = set()
        row_id, covered = _audit_item(row, ledger.get("eligible_dataset_id"), uses, row_failures)
        failures.update(row_failures)
        if row_id is not None:
            row_ids.append(row_id)
        if covered:
            item_covered += 1

    if len(row_ids) != len(set(row_ids)):
        failures.add("rights_video_ids_duplicate")
    expected, supplied = set(ids), set(row_ids)
    if expected - supplied:
        failures.add("rights_rows_missing")
    if supplied - expected:
        failures.add("rights_rows_outside_eligible_cohort")
    if len(rows) != len(ids):
        failures.add("rights_row_count_mismatch")

    complete = not failures and annotation_ok and item_covered == len(ids)
    return _summary(failures, eligible_raw_bytes, ledger_raw_bytes, len(ids), len(rows), item_covered,
                    item_covered if complete else 0, complete)


def _summary(failures: set[str], eligible_raw: bytes, ledger_raw: bytes, eligible_n: int,
             row_n: int, covered_n: int, complete_declaration_n: int,
             complete: bool = False) -> dict:
    return {
        "schema_version": 1,
        "structurally_valid": not failures,
        "declared_item_and_annotation_use_gates_structurally_complete": bool(complete and not failures),
        "dataset_access_agreement_gate": "not_machine_verified",
        "execution_authorization": "not_granted",
        "eligible_video_count": eligible_n,
        "rights_row_count": row_n,
        "item_rows_with_complete_permitted_decisions": covered_n,
        "eligible_rows_with_complete_permitted_declarations": complete_declaration_n,
        "eligible_manifest_sha256": hashlib.sha256(eligible_raw).hexdigest(),
        "rights_ledger_sha256": hashlib.sha256(ledger_raw).hexdigest(),
        "failure_codes": sorted(failures),
        "semantic_validation": "not_machine_verifiable",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible-manifest", required=True, type=Path)
    parser.add_argument("--rights-ledger", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        eligible, eligible_raw = _read_json(args.eligible_manifest)
        ledger, ledger_raw = _read_json(args.rights_ledger)
        result = audit(eligible, eligible_raw, ledger, ledger_raw)
    except Exception:
        result = _summary({"input_unreadable_or_malformed"}, b"", b"", 0, 0, 0, 0)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["structurally_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
