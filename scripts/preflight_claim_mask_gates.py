#!/usr/bin/env python3
"""CPU-only structural cross-gate report for claim-mask artifacts.

This command never initializes a model or video processor and never grants
execution authorization. Schema v12 is exploratory-only and SESOI remains
unresolved, so even a synthetic structural pass is reported as blocked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from audit_claim_review_evidence_index import audit as audit_claim_review
from audit_data_rights_evidence_index import audit as audit_data_rights
from audit_item_rights_manifest import audit as audit_item_rights
from validate_claim_mask_protocol import validate as validate_protocol


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate_json_key")
        out[key] = value
    return out


def _nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_nonfinite(k) or _nonfinite(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_nonfinite(item) for item in value)
    return False


def _read(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite_json")),
    )
    if _nonfinite(value):
        raise ValueError("non_finite_json")
    return value, raw


def _component(status: str, report: dict | None = None, error: str | None = None,
               expected_schema_version: int = 1) -> dict:
    child_failures = report.get("failure_codes", []) if isinstance(report, dict) else []
    failures = ["component_validation_failed"] if isinstance(child_failures, list) and child_failures else []
    invalid_failure_codes = (not isinstance(child_failures, list)
                             or any(not isinstance(code, str) for code in child_failures))
    inconsistent_status = ((status == "passed" and bool(child_failures))
                           or (status == "failed" and not child_failures))
    if report is not None and (not isinstance(report, dict) or type(report.get("schema_version")) is not int
            or report.get("schema_version") != expected_schema_version
            or report.get("execution_authorization") != "not_granted"
            or invalid_failure_codes or inconsistent_status):
        status = "failed"
        failures.append("component_contract_invalid")
    if status == "failed" and not failures:
        failures = ["component_status_invalid"]
    return {"status": status, "failure_codes": sorted(set(failures)),
            **({"error_code": error} if error else {})}


def _strict_pass(report: Any, field: str, expected: Any) -> bool:
    if not isinstance(report, dict):
        return False
    actual = report.get(field)
    if isinstance(expected, bool):
        return type(actual) is bool and actual is expected
    return type(actual) is type(expected) and actual == expected


def _blocked_result(components: dict, blockers: list[str]) -> dict:
    passed = bool(components) and all(item["status"] == "passed" for item in components.values())
    return {
        "schema_version": 1,
        "purpose": "claim_mask_structural_cross_gate_preflight",
        "overall_status": "blocked",
        "structural_checks_status": "passed" if passed else "failed",
        "components": components,
        "blocker_codes": sorted(set(blockers)),
        "execution_authorization": "not_granted",
    }


def preflight(paths: dict[str, Path]) -> dict:
    components: dict[str, dict] = {}
    blockers: list[str] = []
    required = {"protocol", "eligible_manifest", "claim_input_manifest", "claim_mapping_ledger",
                "claim_review_index", "item_rights_ledger", "data_rights_index"}
    if set(paths) != required or any(not isinstance(path, Path) for path in paths.values()):
        return _blocked_result({"inputs": _component("error", error="input_paths_invalid")}, ["input_paths_invalid"])
    try:
        loaded = {key: _read(path) for key, path in paths.items()}
    except (OSError, UnicodeError, ValueError, RecursionError):
        return _blocked_result({"inputs": _component("error", error="private_input_invalid")}, ["private_input_invalid"])

    values = {key: pair[0] for key, pair in loaded.items()}
    raws = {key: pair[1] for key, pair in loaded.items()}
    protocol, protocol_raw = values["protocol"], raws["protocol"]
    eligible, eligible_raw = values["eligible_manifest"], raws["eligible_manifest"]
    try:
        protocol_report = validate_protocol(protocol, hashlib.sha256(protocol_raw).hexdigest())
        components["protocol"] = _component(
            "passed" if _strict_pass(protocol_report, "structurally_valid", True) else "failed",
            protocol_report, expected_schema_version=12)
    except Exception:
        components["protocol"] = _component("error", error="component_audit_error")
    if components["protocol"]["status"] != "passed":
        blockers.append("protocol_validation_failed")

    try:
        claim_report = audit_claim_review(
            protocol, protocol_raw, eligible, eligible_raw,
            values["claim_input_manifest"], raws["claim_input_manifest"],
            values["claim_mapping_ledger"], raws["claim_mapping_ledger"],
            values["claim_review_index"], raws["claim_review_index"],
        )
        components["claim_review"] = _component(
            "passed" if _strict_pass(claim_report, "structural_status", "passed") else "failed", claim_report)
    except Exception:
        components["claim_review"] = _component("error", error="component_audit_error")
    if components["claim_review"]["status"] != "passed":
        blockers.append("claim_review_structure_failed")

    try:
        item_report = audit_item_rights(
            eligible, eligible_raw, values["item_rights_ledger"], raws["item_rights_ledger"])
        components["item_rights"] = _component(
            "passed" if _strict_pass(item_report, "structurally_valid", True) else "failed", item_report)
    except Exception:
        components["item_rights"] = _component("error", error="component_audit_error")
    if components["item_rights"]["status"] != "passed":
        blockers.append("item_rights_structure_failed")

    try:
        rights_report = audit_data_rights(
            protocol, protocol_raw, eligible, eligible_raw,
            values["item_rights_ledger"], raws["item_rights_ledger"],
            values["data_rights_index"], raws["data_rights_index"],
        )
        components["data_rights_evidence"] = _component(
            "passed" if _strict_pass(rights_report, "structural_status", "passed") else "failed", rights_report)
    except Exception:
        components["data_rights_evidence"] = _component("error", error="component_audit_error")
    if components["data_rights_evidence"]["status"] != "passed":
        blockers.append("data_rights_evidence_structure_failed")

    # Never treat structural validation as scientific readiness. v12 refuses
    # confirmatory labels by design; no CLI flag can override this stop.
    blockers.append("schema_v12_confirmatory_gate_unresolved")
    if not isinstance(protocol, dict) or protocol.get("protocol_stage") != "confirmatory":
        blockers.append("protocol_stage_not_confirmatory")
    analysis = protocol.get("analysis") if isinstance(protocol, dict) else None
    sesoi = analysis.get("sesoi") if isinstance(analysis, dict) else None
    if not isinstance(sesoi, dict) or sesoi.get("status") != "preoutcome_fixed":
        blockers.append("sesoi_not_frozen")
    blockers.append("execution_authorization_not_granted")
    return _blocked_result(components, blockers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "eligible-manifest", "claim-input-manifest", "claim-mapping-ledger",
                 "claim-review-index", "item-rights-ledger", "data-rights-index"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    paths = {
        "protocol": args.protocol, "eligible_manifest": args.eligible_manifest,
        "claim_input_manifest": args.claim_input_manifest,
        "claim_mapping_ledger": args.claim_mapping_ledger,
        "claim_review_index": args.claim_review_index,
        "item_rights_ledger": args.item_rights_ledger,
        "data_rights_index": args.data_rights_index,
    }
    report = preflight(paths)
    print(json.dumps(report, sort_keys=True))
    # This preflight is intentionally never a successful launch prerequisite:
    # every report is blocked until a future protocol version clears gates.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
