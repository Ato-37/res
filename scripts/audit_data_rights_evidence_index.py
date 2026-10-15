#!/usr/bin/env python3
"""Cross-check a private rights evidence index without judging legal truth.

This CPU-only checker binds exact bytes and declared cohort/use scope across a
schema-v13 protocol, eligible manifest, v2 item-rights ledger, and private
evidence-index JSON. It reports no identifiers or private locators and never
authorizes inference. Human review remains required for evidence authenticity,
authority, applicability, permissions, and condition fulfillment.
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

SHA256 = re.compile(r"^[a-f0-9]{64}$")
UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
USES = {"local_inference", "local_retention", "aggregate_publication", "redistribution"}
DECISIONS = {"permitted", "not_permitted", "unresolved"}


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
    if _has_nonfinite(value):
        raise ValueError("non_finite_json")
    return value, raw


def _has_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_nonfinite(k) or _has_nonfinite(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_nonfinite(x) for x in value)
    return False


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _utc(value: Any) -> bool:
    if not isinstance(value, str) or UTC.fullmatch(value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _exact_keys(value: Any, keys: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == keys


def _raw_matches(value: Any, raw: Any) -> bool:
    if not isinstance(raw, bytes):
        return False
    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite_json")),
        )
        if _has_nonfinite(parsed) or _has_nonfinite(value):
            return False
        left = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        right = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return left == right
    except (AttributeError, UnicodeError, ValueError, TypeError, OverflowError):
        return False


def audit(protocol: Any, protocol_raw: Any, eligible: Any, eligible_raw: Any,
          ledger: Any, ledger_raw: bytes, index: Any, index_raw: bytes) -> dict:
    failures: set[str] = set()
    for label, value, raw in (
        ("protocol", protocol, protocol_raw), ("eligible_manifest", eligible, eligible_raw),
        ("item_rights_ledger", ledger, ledger_raw), ("evidence_index", index, index_raw),
    ):
        if not _raw_matches(value, raw):
            failures.add(f"{label}_byte_binding_invalid")
    rights = protocol.get("data_rights") if isinstance(protocol, dict) else None
    if (not isinstance(protocol, dict)
            or type(protocol.get("schema_version")) is not int
            or protocol.get("schema_version") != 13):
        failures.add("protocol_schema_version_invalid")
    if not isinstance(rights, dict):
        failures.add("protocol_rights_missing")
    elif rights.get("status") != "verified":
        failures.add("protocol_rights_not_declared_verified")

    protocol_index_sha = rights.get("evidence_sha256") if isinstance(rights, dict) else None
    if (not isinstance(index_raw, bytes) or not _sha(protocol_index_sha)
            or hashlib.sha256(index_raw).hexdigest() != protocol_index_sha):
        failures.add("evidence_index_hash_mismatch")
    ledger_sha = hashlib.sha256(ledger_raw).hexdigest() if isinstance(ledger_raw, bytes) else ""
    if (not isinstance(rights, dict) or rights.get("item_rights_manifest_sha256") != ledger_sha):
        failures.add("protocol_ledger_hash_mismatch")
    eligible_sha = hashlib.sha256(eligible_raw).hexdigest() if isinstance(eligible_raw, bytes) else ""
    cohort = protocol.get("cohort") if isinstance(protocol, dict) else None
    if not isinstance(eligible, list):
        eligible_input_count = -1
    else:
        eligible_input_count = len(eligible)
    if (not isinstance(cohort, dict)
            or cohort.get("eligible_manifest_sha256") != eligible_sha
            or type(cohort.get("eligible_video_count")) is not int
            or cohort.get("eligible_video_count") != eligible_input_count):
        failures.add("protocol_eligible_manifest_mismatch")
    if not isinstance(eligible, list) or not eligible:
        failures.add("eligible_manifest_invalid")
        eligible_count = 0
        eligible_ids: set[str] = set()
    else:
        ids = [x.get("video_id") if isinstance(x, dict) else None for x in eligible]
        if any(not _text(x) for x in ids) or len(ids) != len(set(ids)):
            failures.add("eligible_manifest_identity_invalid")
        eligible_ids = {x for x in ids if isinstance(x, str)}
        eligible_count = len(eligible)

    if (not isinstance(ledger, dict) or type(ledger.get("schema_version")) is not int
            or ledger.get("schema_version") != 2):
        failures.add("item_rights_ledger_invalid")
        ledger = {}
    if ledger.get("eligible_manifest_sha256") != eligible_sha:
        failures.add("ledger_eligible_hash_mismatch")
    ledger_items = ledger.get("items")
    if not isinstance(ledger_items, list):
        failures.add("ledger_items_invalid")
        ledger_items = []
    ledger_ids = [x.get("video_id") if isinstance(x, dict) else None for x in ledger_items]
    if (len(ledger_items) != eligible_count or any(not isinstance(x, str) for x in ledger_ids)
            or len(set(ledger_ids)) != len(ledger_ids) or set(ledger_ids) != eligible_ids):
        failures.add("ledger_eligible_coverage_mismatch")
    intended_uses = ledger.get("intended_uses")
    if (not isinstance(intended_uses, list) or not intended_uses
            or any(not isinstance(x, str) or x not in USES for x in intended_uses)
            or (all(isinstance(x, str) for x in intended_uses)
                and len(set(intended_uses)) != len(intended_uses))
            or "local_inference" not in intended_uses):
        failures.add("ledger_intended_uses_invalid")
        intended_uses = []
    ledger_scope = ledger.get("intended_use_scope")
    if (not _text(ledger_scope) or not isinstance(rights, dict)
            or rights.get("scope") != ledger_scope):
        failures.add("planned_use_scope_mismatch")
    if isinstance(rights, dict):
        covered = rights.get("covered_eligible_video_count")
        unresolved = rights.get("unresolved_eligible_video_count")
        if (type(covered) is not int or type(unresolved) is not int
                or covered != eligible_count or unresolved != 0):
            failures.add("protocol_rights_coverage_incomplete")

    required_index = {
        "schema_version", "eligible_manifest", "item_rights_manifest",
        "annotation_terms", "item_media_conditions", "dataset_access_agreement",
    }
    if (not _exact_keys(index, required_index)
            or type(index.get("schema_version")) is not int
            or index.get("schema_version") != 1):
        failures.add("evidence_index_schema_invalid")
        index = {}
    expected_links = {
        "eligible_manifest": {"sha256", "private_locator", "eligible_video_count"},
        "item_rights_manifest": {"sha256", "private_locator"},
        "annotation_terms": {"sha256", "private_locator"},
        "item_media_conditions": {"sha256", "private_locator", "covered_video_count"},
    }
    for name, keys in expected_links.items():
        entry = index.get(name)
        if (not _exact_keys(entry, keys) or not _sha(entry.get("sha256"))
                or not _text(entry.get("private_locator"))):
            failures.add(f"{name}_entry_invalid")
    eligible_entry = index.get("eligible_manifest", {})
    eligible_entry = eligible_entry if isinstance(eligible_entry, dict) else {}
    ledger_entry = index.get("item_rights_manifest", {})
    ledger_entry = ledger_entry if isinstance(ledger_entry, dict) else {}
    media_entry = index.get("item_media_conditions", {})
    media_entry = media_entry if isinstance(media_entry, dict) else {}
    if eligible_entry.get("sha256") != eligible_sha:
        failures.add("index_eligible_hash_mismatch")
    if (type(eligible_entry.get("eligible_video_count")) is not int
            or eligible_entry.get("eligible_video_count") != eligible_count):
        failures.add("index_eligible_count_mismatch")
    if ledger_entry.get("sha256") != ledger_sha:
        failures.add("index_ledger_hash_mismatch")
    if (type(media_entry.get("covered_video_count")) is not int
            or media_entry.get("covered_video_count") != eligible_count):
        failures.add("index_media_coverage_incomplete")
    annotation = index.get("annotation_terms", {})
    annotation = annotation if isinstance(annotation, dict) else {}
    ledger_annotation = ledger.get("annotation_rights", {})
    ledger_annotation = ledger_annotation if isinstance(ledger_annotation, dict) else {}
    if annotation.get("sha256") != ledger_annotation.get("evidence_sha256"):
        failures.add("index_annotation_hash_mismatch")
    if (not isinstance(ledger_annotation, dict)
            or ledger_annotation.get("review_status") != "completed"
            or not isinstance(ledger_annotation.get("decisions"), dict)
            or set(ledger_annotation.get("decisions", {})) != set(intended_uses)
            or any(v != "permitted" for v in ledger_annotation.get("decisions", {}).values())):
        failures.add("ledger_annotation_use_not_cleared")
    for item in ledger_items:
        if (not isinstance(item, dict) or item.get("media_review_status") != "completed"
                or item.get("conditions_status") != "reviewed_resolved"
                or not isinstance(item.get("decisions"), dict)
                or set(item.get("decisions", {})) != set(intended_uses)
                or any(v != "permitted" for v in item.get("decisions", {}).values())):
            failures.add("ledger_item_use_not_cleared")
            break

    agreement_keys = {
        "applicability", "sha256", "private_locator", "review_record_sha256",
        "reviewer_role", "reviewed_at_utc", "planned_use_scope",
        "decisions", "non_applicability_basis_sha256",
    }
    agreement = index.get("dataset_access_agreement")
    if (not _exact_keys(agreement, agreement_keys)
            or not isinstance(agreement.get("applicability"), str)
            or agreement.get("applicability") not in {"applicable", "not_applicable", "unresolved"}
            or not _sha(agreement.get("sha256"))
            or not _text(agreement.get("private_locator"))
            or not _sha(agreement.get("review_record_sha256"))
            or not _text(agreement.get("reviewer_role"))
            or not _utc(agreement.get("reviewed_at_utc"))
            or agreement.get("planned_use_scope") != (ledger.get("intended_use_scope"))):
        failures.add("dataset_access_agreement_entry_invalid")
    elif agreement["applicability"] == "unresolved":
        failures.add("dataset_access_agreement_unresolved")
    elif agreement["applicability"] == "not_applicable":
        if (not _sha(agreement.get("non_applicability_basis_sha256"))
                or agreement.get("non_applicability_basis_sha256") != agreement.get("sha256")
                or agreement.get("decisions") is not None):
            failures.add("dataset_access_non_applicability_unsubstantiated")
    else:
        decisions = agreement.get("decisions")
        if (not isinstance(decisions, dict) or set(decisions) != set(intended_uses)
            or any(not isinstance(value, str) or value not in DECISIONS
                   for value in decisions.values())):
            failures.add("dataset_access_use_decisions_invalid")
        elif any(value != "permitted" for value in decisions.values()):
            failures.add("dataset_access_use_not_cleared")
        if agreement.get("non_applicability_basis_sha256") is not None:
            failures.add("dataset_access_applicability_basis_unexpected")

    return {
        "schema_version": 1,
        "purpose": "private_data_rights_evidence_index_crosscheck",
        "structural_status": "passed" if not failures else "failed",
        "execution_authorization": "not_granted",
        "requested_eligible_video_count": eligible_count,
        "effective_eligible_video_count": len(ledger_ids),
        "requested_intended_use_count": len(intended_uses),
        "effective_intended_use_count": len(intended_uses) if not failures else 0,
        "failure_codes": sorted(failures),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--eligible-manifest", required=True, type=Path)
    parser.add_argument("--item-rights-ledger", required=True, type=Path)
    parser.add_argument("--evidence-index", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        protocol, protocol_raw = _read(args.protocol)
        eligible, eligible_raw = _read(args.eligible_manifest)
        ledger, ledger_raw = _read(args.item_rights_ledger)
        index, index_raw = _read(args.evidence_index)
        report = audit(protocol, protocol_raw, eligible, eligible_raw,
                       ledger, ledger_raw, index, index_raw)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        print("data-rights evidence audit: failed; failure_codes=private_input_invalid; execution_authorization=not_granted")
        return 2
    print("data-rights evidence audit: "
          f"status={report['structural_status']}; "
          f"eligible={report['effective_eligible_video_count']}/{report['requested_eligible_video_count']}; "
          f"uses={report['effective_intended_use_count']}/{report['requested_intended_use_count']}; "
          f"failures={len(report['failure_codes'])}; execution_authorization=not_granted")
    return 0 if report["structural_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
