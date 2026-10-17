"""Compose QTA structural row binding and three-plan aggregation.

This CPU-only wrapper is a synthetic/engineering preflight, not an execution
gate. It treats all supplied artifacts and expected hashes as unauthenticated,
performs no model load or scoring, and fails closed on the fixed three-row
cohort.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from qta_synthetic_cohort_preflight import preflight_synthetic_qta_cohort
from qta_three_row_plan_aggregate import aggregate_qta_nine_branch_plans


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def preflight_synthetic_qta_plan_bundle(*, mapping_inputs: Mapping[str, Any],
                                        plans: Sequence[Mapping[str, Any]],
                                        expected_score_binding_sha256_by_token: Mapping[str, str],
                                        expected_selection_spec_sha256_by_token: Mapping[str, str]
                                        ) -> dict[str, Any]:
    """Bind a fixed cohort's upstream rows and plans without authorizing work.

    The mapping audit establishes structural consistency only. Expected score
    and selection hashes remain caller supplied pins; this wrapper cannot
    authenticate their source or the provenance of any input.
    """
    blocked = _blocked("plan_bundle_input_invalid")
    try:
        if not isinstance(mapping_inputs, Mapping):
            return blocked
        tokens = mapping_inputs.get("expected_cohort_tokens")
        if (not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes))
                or len(tokens) != 3 or any(not isinstance(token, str) or not token
                                           for token in tokens)
                or len(set(tokens)) != 3):
            return _blocked("cohort_token_inventory_invalid")
        tokens = tuple(tokens)
        for supplied in (expected_score_binding_sha256_by_token,
                         expected_selection_spec_sha256_by_token):
            if not isinstance(supplied, Mapping) or set(supplied) != set(tokens):
                return _blocked("expected_plan_pin_inventory_invalid")
            if any(not isinstance(value, str) or not _SHA256.fullmatch(value)
                   for value in supplied.values()):
                return _blocked("expected_plan_pin_invalid")
        if (not isinstance(plans, Sequence) or isinstance(plans, (str, bytes))
                or len(plans) != 3):
            return _blocked("cohort_plan_count_mismatch")

        mapping_report = preflight_synthetic_qta_cohort(**dict(mapping_inputs))
        if mapping_report.get("status") != "structural_row_binding_passed_unvalidated":
            return _blocked("upstream_structural_join_blocked",
                            upstream_failure=mapping_report.get("failure_code"))
        processors = mapping_inputs["processor_rows"]
        groups_by_token = mapping_inputs["expected_processor_groups_by_token"]
        batch_by_token = mapping_inputs["expected_processor_batch_sha256_by_token"]
        expected_rows, rows = [], []
        for offset, (token, processor, plan) in enumerate(
                zip(tokens, processors, plans, strict=True), start=1):
            if not isinstance(processor, Mapping) or not isinstance(plan, Mapping):
                return _blocked("plan_row_schema_invalid")
            groups = groups_by_token[token]
            row_hash = mapping_report["row_binding_sha256"][offset - 1]
            expected_rows.append({
                "row_binding_sha256": row_hash,
                "typed_batch_sha256": batch_by_token[token],
                "score_binding_sha256": expected_score_binding_sha256_by_token[token],
                "video_grid_thw": groups["video_grid_thw"],
                "evidence_group_indices": groups["evidence_group_indices"],
                "control_group_indices": groups["control_group_indices"],
                "temporal_group_token_positions": groups["temporal_group_token_positions"],
                "selection_spec_sha256": expected_selection_spec_sha256_by_token[token],
            })
            rows.append({"record_position": offset,
                         "row_binding_sha256": row_hash, "plan": plan})
        aggregate = aggregate_qta_nine_branch_plans(rows=rows,
                                                    expected_rows=expected_rows)
        if aggregate.get("status") != "three_plans_bound_unvalidated":
            return _blocked("qta_plan_aggregation_blocked",
                            upstream_failure=aggregate.get("failure_code"))
        return {
            "status": "three_row_structural_and_plan_binding_passed_unvalidated",
            "scope": "synthetic_non_authorizing_qta_cohort_plan_preflight",
            "requested_records": 3,
            "verified_structural_rows": 3,
            "validated_plans": 3,
            "requested_forwards": 27,
            "planned_forwards": 27,
            "completed_forwards": 0,
            "row_binding_sha256": mapping_report["row_binding_sha256"],
            "mapping_bundle_binding_sha256": mapping_report["bundle_binding_sha256"],
            "plan_bundle_binding_sha256": aggregate["bundle_binding_sha256"],
            "data_provenance_status": "not_verified_by_this_preflight",
            "human_review_authenticated": False,
            "prediction_link_authenticated": False,
            "model_load_permitted": False,
            "scoring_authorization": "not_granted",
            "execution_authorization": "not_granted",
            "failure_code": None,
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return blocked


def _blocked(code: str, *, upstream_failure: Any = None) -> dict[str, Any]:
    safe_code = code if isinstance(code, str) and re.fullmatch(
        r"[a-z][a-z0-9_]{0,119}", code) else "plan_bundle_input_invalid"
    safe_upstream = (upstream_failure if isinstance(upstream_failure, str)
                     and re.fullmatch(r"[a-z][a-z0-9_]{0,119}", upstream_failure)
                     else None)
    return {
        "status": "blocked_synthetic_qta_cohort_plan_preflight",
        "scope": "synthetic_non_authorizing_qta_cohort_plan_preflight",
        "requested_records": 3,
        "verified_structural_rows": 0,
        "validated_plans": 0,
        "requested_forwards": 27,
        "planned_forwards": 0,
        "completed_forwards": 0,
        "row_binding_sha256": [],
        "mapping_bundle_binding_sha256": None,
        "plan_bundle_binding_sha256": None,
        "data_provenance_status": "not_verified_by_this_preflight",
        "human_review_authenticated": False,
        "prediction_link_authenticated": False,
        "model_load_permitted": False,
        "scoring_authorization": "not_granted",
        "execution_authorization": "not_granted",
        "failure_code": safe_code,
        "upstream_failure_code": safe_upstream,
    }
