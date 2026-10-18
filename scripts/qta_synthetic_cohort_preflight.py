"""Structural integration wrapper for the existing QTA row binder.

This function connects the review, clock, processor, and baseline row artifacts
through ``qta_prescore_mapping_binding``. It checks structural agreement only.
Input provenance is not verified: although tests use synthetic fixtures, the
wrapper accepts arbitrary row records. It does not authenticate inputs,
establish a jointly frozen clock, authenticate prediction-link claims, load a
model, score, or authorize execution. It is not a production preflight.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import re
from typing import Any

from qta_prescore_mapping_binding import audit_qta_prescore_mapping_bundle


_ROW_COLLECTIONS = ("review_rows", "clock_rows", "processor_rows", "baseline_rows")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FAILURE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,119}$")


def preflight_synthetic_qta_cohort(**inputs: Any) -> dict[str, Any]:
    """Run the existing QTA structural join, retaining a fixed denominator of 3.

    Inputs are the keyword arguments accepted by
    ``audit_qta_prescore_mapping_bundle``. Success remains unvalidated and
    non-authorizing. Malformed or incomplete inputs fail closed without
    shrinking the requested cohort.
    """
    failure = None
    if not isinstance(inputs, dict):  # defensive; Python ** normally guarantees this
        failure = "integration_inputs_invalid"
    else:
        for name in _ROW_COLLECTIONS:
            rows = inputs.get(name)
            if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
                    or len(rows) != 3):
                failure = "cohort_row_count_mismatch"
                break

    if failure is None:
        try:
            report = audit_qta_prescore_mapping_bundle(**inputs)
        except (TypeError, ValueError, KeyError, OverflowError, ZeroDivisionError):
            report = None
        if not isinstance(report, dict):
            failure = "integration_input_validation_failed"
        elif report.get("requested_records") != 3:
            failure = "cohort_row_count_mismatch"

    if failure is not None:
        return _result(
            status="blocked_structural_row_binding_integration",
            failure_code=failure,
            verified_rows=0,
            row_digests=[],
            bundle_digest=None,
        )

    assert isinstance(report, dict)
    succeeded = report.get("status") == "structurally_bound_pending_independent_score_freeze"
    failure_code = report.get("failure_code", "structural_join_failed")
    verified_rows = report.get("verified_structural_rows", 0)
    row_digests = report.get("row_binding_sha256")
    bundle_digest = report.get("bundle_binding_sha256")
    if succeeded:
        if (type(verified_rows) is not int or verified_rows != 3
                or not isinstance(row_digests, list) or len(row_digests) != 3
                or any(not isinstance(digest, str) or not _SHA256.fullmatch(digest)
                       for digest in row_digests)
                or not isinstance(bundle_digest, str) or not _SHA256.fullmatch(bundle_digest)):
            succeeded = False
            failure_code = "structural_join_success_summary_invalid"
        else:
            expected_bundle = hashlib.sha256(json.dumps(
                row_digests, separators=(",", ":"), ensure_ascii=True,
                allow_nan=False).encode("ascii")).hexdigest()
            if bundle_digest != expected_bundle:
                succeeded = False
                failure_code = "structural_join_bundle_digest_mismatch"
    if type(verified_rows) is not int or not 0 <= verified_rows <= 3:
        verified_rows = 0
    if not isinstance(failure_code, str) or not _FAILURE_CODE.fullmatch(failure_code):
        failure_code = "structural_join_failed"
    # This adapter exposes only an atomic three-row cohort result. The lower
    # level binder may report how many rows it inspected before a mismatch,
    # but that partial-completion count is not accepted evidence here.
    if not succeeded:
        verified_rows = 0
    return _result(
        status=("structural_row_binding_passed_unvalidated" if succeeded
                else "blocked_structural_row_binding_integration"),
        failure_code=None if succeeded else failure_code,
        verified_rows=verified_rows,
        row_digests=row_digests if succeeded else [],
        bundle_digest=bundle_digest if succeeded else None,
    )


def _result(*, status: str, failure_code: str | None, verified_rows: int,
            row_digests: list, bundle_digest: str | None) -> dict[str, Any]:
    """Emit a fixed, explicitly non-authorizing result schema."""
    return {
        "status": status,
        "scope": "structural_row_binding_integration_only",
        "data_provenance_status": "not_verified_by_this_preflight",
        "requested_records": 3,
        "verified_structural_rows": verified_rows,
        "failure_code": failure_code,
        "row_binding_sha256": row_digests,
        "bundle_binding_sha256": bundle_digest,
        "clock_joint_freeze_authenticated": False,
        "prediction_link_authentication": "self_asserted_unauthenticated",
        "model_load_permitted": False,
        "branch_forwards": 0,
        "scoring_authorization": "not_granted",
        "execution_authorization": "not_granted",
    }
