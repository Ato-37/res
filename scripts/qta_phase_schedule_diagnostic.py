"""Non-authorizing CPU check of the frozen two-phase QTA forward schedule.

This checks only the *shape* of a prospective schedule. It does not verify
signatures, human judgments, cohort provenance, artifact bytes, model inputs,
or execution authorization, and must never be used as a production score gate.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from qta_nine_branch_plan import BRANCH_ORDER

ROW_COUNT = 3
# The nine-branch plan is the canonical condition vocabulary and order.
# Phase 1 runs baseline_full; phase 2 runs the remaining eight branches.
BASELINE = BRANCH_ORDER[:1]
POST_BASELINE = BRANCH_ORDER[1:]
DENIAL = {
    "model_load_permitted": False,
    "scoring_authorized": False,
    "phase_authorized": False,
    "authorized_forwards": 0,
}
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _valid_rows(rows: Any, expected_branches: tuple[str, ...]) -> tuple[list[dict], str | None]:
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        return [], "row_count_invalid"
    clean: list[dict] = []
    tokens: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"record_token", "processor_batch_sha256", "branches"}:
            return [], "row_schema_invalid"
        token = row.get("record_token")
        digest = row.get("processor_batch_sha256")
        branches = row.get("branches")
        if not isinstance(token, str) or not _TOKEN.fullmatch(token) or token in tokens:
            return [], "record_token_invalid_or_duplicate"
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            return [], "processor_batch_digest_invalid"
        if (not isinstance(branches, list) or len(branches) != len(expected_branches)
                or any(not isinstance(name, str) for name in branches)
                or branches != list(expected_branches)):
            return [], "branch_set_invalid"
        tokens.add(token)
        clean.append({"record_token": token, "processor_batch_sha256": digest,
                      "branches": list(branches)})
    return clean, None


def diagnose_qta_phase_schedule(*, phase1_rows: Any, phase2_rows: Any,
                                phase2_bindings: Any) -> dict[str, Any]:
    """Check exact 3+24 branch shape and digest-field syntax, always deny use.

    The nine names are one baseline-full forward plus eight post-baseline
    branches per row. Phase 2 must bind syntactically valid hashes for phase-1
    receipt, the complete baseline manifest, and the independent stage-2
    review receipt. This function does not load or verify any of those bytes.
    """
    gates: dict[str, dict[str, Any]] = {}
    p1, error = _valid_rows(phase1_rows, BASELINE)
    gates["phase1_baseline_schedule"] = {
        "status": "blocked" if error else "passed_shape_only",
        "requested_rows": ROW_COUNT,
        "schedule_branches": 0 if error else ROW_COUNT,
        "failure_code": error,
    }
    p2, error2 = _valid_rows(phase2_rows, POST_BASELINE)
    gates["phase2_branch_schedule"] = {
        "status": "blocked" if error2 else "passed_shape_only",
        "requested_rows": ROW_COUNT,
        "schedule_branches": 0 if error2 else ROW_COUNT * len(POST_BASELINE),
        "failure_code": error2,
    }

    if error or error2:
        gates["cross_phase_batch_binding"] = {
            "status": "blocked", "failure_code": "upstream_schedule_invalid"}
    else:
        ok = all(a["record_token"] == b["record_token"]
                 and a["processor_batch_sha256"] == b["processor_batch_sha256"]
                 for a, b in zip(p1, p2, strict=True))
        gates["cross_phase_batch_binding"] = {
            "status": "passed_shape_only" if ok else "blocked",
            "failure_code": None if ok else "phase_row_or_batch_binding_mismatch",
        }

    required_bindings = {"phase1_receipt_sha256", "baseline_manifest_sha256",
                         "stage2_review_receipt_sha256"}
    bindings_ok = (isinstance(phase2_bindings, Mapping)
                   and set(phase2_bindings) == required_bindings
                   and all(isinstance(phase2_bindings.get(key), str)
                           and _SHA256.fullmatch(phase2_bindings[key])
                           for key in required_bindings))
    gates["phase2_artifact_binding_shape"] = {
        "status": "passed_shape_only" if bindings_ok else "blocked",
        "failure_code": None if bindings_ok else "phase2_artifact_binding_invalid",
    }

    valid = all(gate["status"] == "passed_shape_only" for gate in gates.values())
    return {
        "schema_version": 1,
        "purpose": "qta_two_phase_schedule_diagnostic_only",
        "status": "shape_valid_non_authorizing" if valid else "diagnostic_blocked",
        "requested_rows": ROW_COUNT,
        "phase1_schedule_branches": ROW_COUNT if not error else 0,
        "phase2_schedule_branches": ROW_COUNT * len(POST_BASELINE) if not error2 else 0,
        "total_schedule_branches": (ROW_COUNT * 9 if valid else 0),
        "counts_are_schedule_shape_only": True,
        "execution_performed": False,
        "gates": gates,
        "artifact_bytes_verified": False,
        "signatures_verified": False,
        "human_identity_verified": False,
        "semantic_review_authenticated": False,
        "faithfulness_scored": False,
        **DENIAL,
    }
