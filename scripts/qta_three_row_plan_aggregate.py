"""All-or-none structural aggregation of three QTA branch plans.

This binds plan summaries to externally supplied per-row bindings and expected
typed-batch/scorer hashes. It does not authenticate those inputs, perform the
upstream human/trust/timing preflight, load a model, or authorize scoring.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence

from qta_nine_branch_plan import validate_qta_nine_branch_plan


ROW_COUNT = 3
BRANCHES_PER_ROW = 9
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()


def _valid_hash(value):
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def aggregate_qta_nine_branch_plans(*, rows, expected_rows):
    """Bind all three ordered row plans or fail without partial acceptance.

    Each `rows` item is exactly `{record_position, row_binding_sha256, plan}`.
    The matching `expected_rows` item binds row, batch, scorer, temporal grid,
    evidence/control groups, and the frozen selection-spec digest from the
    upstream join. These values are
    caller supplied, not authenticated by this function. Record-specific batch
    and scorer digests may differ, while intervention layers and bias must be
    identical across the three plans.
    """
    try:
        if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
                or len(rows) != ROW_COUNT):
            return _blocked("cohort_row_count_mismatch")
        if (not isinstance(expected_rows, Sequence) or isinstance(expected_rows, (str, bytes))
                or len(expected_rows) != ROW_COUNT):
            return _blocked("expected_row_count_mismatch")
        verified = []
        seen_row_bindings = set()
        common_treatment = None
        for offset, (row, expected) in enumerate(zip(rows, expected_rows, strict=True), start=1):
            if (not isinstance(row, Mapping)
                    or set(row) != {"record_position", "row_binding_sha256", "plan"}
                    or not isinstance(expected, Mapping)
                    or set(expected) != {"row_binding_sha256", "typed_batch_sha256",
                                         "score_binding_sha256", "video_grid_thw",
                                         "evidence_group_indices", "control_group_indices",
                                         "selection_spec_sha256"}):
                return _blocked("row_schema_invalid")
            row_hash = row["row_binding_sha256"]
            if (type(row["record_position"]) is not int
                    or row["record_position"] != offset or not _valid_hash(row_hash)
                    or row_hash != expected["row_binding_sha256"]
                    or not isinstance(row["plan"], Mapping)
                    or row["plan"].get("row_binding_sha256") != row_hash
                    or row_hash in seen_row_bindings):
                return _blocked("row_order_or_binding_mismatch")
            seen_row_bindings.add(row_hash)
            if (not _valid_hash(expected["typed_batch_sha256"])
                    or not _valid_hash(expected["score_binding_sha256"])
                    or not _valid_hash(expected["selection_spec_sha256"])):
                return _blocked("expected_plan_hash_invalid")
            grid = expected["video_grid_thw"]
            if (not isinstance(grid, list) or len(grid) != 3
                    or any(type(value) is not int or value <= 0 for value in grid)):
                return _blocked("expected_processor_grid_invalid")
            for field in ("evidence_group_indices", "control_group_indices"):
                values = expected[field]
                if (not isinstance(values, list) or not values
                        or any(type(value) is not int or value < 0 for value in values)
                        or values != sorted(set(values))):
                    return _blocked("expected_processor_groups_invalid")
            plan = row["plan"]
            try:
                validate_qta_nine_branch_plan(plan)
            except (TypeError, ValueError, KeyError, OverflowError):
                return _blocked("row_plan_validation_failed")
            if (plan["typed_batch_sha256"] != expected["typed_batch_sha256"]
                    or plan["score_binding_sha256"] != expected["score_binding_sha256"]
                    or plan["selection_spec_sha256"] != expected["selection_spec_sha256"]):
                return _blocked("row_plan_expected_hash_mismatch")
            if (plan["temporal_group_count"] != grid[0]
                    or plan["selected_groups"]["evidence"]
                    != expected["evidence_group_indices"]
                    or plan["selected_groups"]["control"]
                    != expected["control_group_indices"]
                    or any(index >= grid[0] for field in
                           ("evidence_group_indices", "control_group_indices")
                           for index in expected[field])):
                return _blocked("row_plan_processor_group_mismatch")
            treatment = (tuple(plan["intervention_layers_1_based"]),
                         float(plan["logit_bias"]), plan["selection_spec_sha256"])
            if common_treatment is None:
                common_treatment = treatment
            elif treatment != common_treatment:
                return _blocked("cohort_treatment_settings_mismatch")
            verified.append({
                "record_position": offset,
                "row_binding_sha256": row_hash,
                "plan_sha256": plan["plan_sha256"],
            })
        bundle = {
            "schema_version": 1,
            "scope": "non_authorizing_qta_plan_aggregation_only",
            "rows": verified,
            "requested_records": ROW_COUNT,
            "planned_forwards": ROW_COUNT * BRANCHES_PER_ROW,
            "intervention_layers_1_based": list(common_treatment[0]),
            "logit_bias": common_treatment[1],
            "selection_spec_sha256": common_treatment[2],
        }
        return {
            "schema_version": 1,
            "status": "three_plans_bound_unvalidated",
            "scope": "non_authorizing_qta_plan_aggregation_only",
            "requested_records": ROW_COUNT,
            "validated_plans": ROW_COUNT,
            "requested_forwards": ROW_COUNT * BRANCHES_PER_ROW,
            "completed_forwards": 0,
            "planned_forwards": ROW_COUNT * BRANCHES_PER_ROW,
            "record_plan_sha256": verified,
            "bundle_binding_sha256": _hash(bundle),
            "data_provenance_status": "not_verified_by_aggregator",
            "model_load_permitted": False,
            "scoring_authorization": "not_granted",
            "execution_authorization": "not_granted",
            "failure_code": None,
        }
    except (TypeError, ValueError, KeyError, OverflowError):
        return _blocked("plan_bundle_input_invalid")


def _blocked(code):
    return {
        "schema_version": 1,
        "status": "blocked_qta_plan_bundle",
        "scope": "non_authorizing_qta_plan_aggregation_only",
        "requested_records": ROW_COUNT,
        "validated_plans": 0,
        "requested_forwards": ROW_COUNT * BRANCHES_PER_ROW,
        "completed_forwards": 0,
        "planned_forwards": 0,
        "record_plan_sha256": [],
        "bundle_binding_sha256": None,
        "data_provenance_status": "not_verified_by_aggregator",
        "model_load_permitted": False,
        "scoring_authorization": "not_granted",
        "execution_authorization": "not_granted",
        "failure_code": code,
    }
