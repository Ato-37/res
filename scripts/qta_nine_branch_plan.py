"""Non-authorizing, hash-bound plan for baseline/QTA/sham x three conditions.

This composes only intervention metadata around one already-processed model
input. It does not open media, authenticate supplied hashes or score a model.
The same fixed six-condition score binding is deliberately shared by QTA and
sham; hashes bind the immutable plan descriptor, not data provenance.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence


BRANCH_ORDER = (
    "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
    "target_full", "target_evidence_ablation", "target_control_ablation",
    "sham_full", "sham_evidence_ablation", "sham_control_ablation",
)
CONDITION_SUFFIXES = ("full", "evidence_ablation", "control_ablation")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SELECTOR_ID = "question_token_attention_mean_layers"
SHAM_SELECTOR_ID = "sha256_utf8_qta-sham-v1_seed_opaque-token_group-index"
TOP_K_BUDGET_RULE = "max(1,ceil(temporal_group_count/10))"


def _canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _indices(values, *, name, upper=None, allow_empty=False):
    if (not isinstance(values, Sequence) or isinstance(values, (str, bytes))
            or (not allow_empty and not values)
            or any(type(value) is not int or value < 0 for value in values)
            or tuple(sorted(set(values))) != tuple(values)
            or (upper is not None and any(value >= upper for value in values))):
        raise ValueError(f"{name}_invalid")
    return tuple(values)


def build_qta_nine_branch_plan(*, row_binding_sha256, temporal_groups,
                               target_group_indices, sham_group_indices, evidence_group_indices,
                               control_group_indices, typed_batch_sha256,
                               score_binding_sha256, intervention_layers_1_based,
                               logit_bias, sham_seed):
    """Build and self-check an ordered nine-branch intervention plan.

    The model input is shared unchanged across all nine forwards. Evidence and
    control are equal-size, disjoint key sets. Target and sham may overlap
    either set; that overlap is retained as plan metadata and is not filtered.
    """
    if (not isinstance(temporal_groups, Sequence)
            or isinstance(temporal_groups, (str, bytes)) or not temporal_groups):
        raise ValueError("temporal_groups_invalid")
    groups = tuple(_indices(group, name="temporal_group") for group in temporal_groups)
    flattened = tuple(position for group in groups for position in group)
    if len(set(flattened)) != len(flattened):
        raise ValueError("temporal_group_positions_overlap")
    group_count = len(groups)
    if not isinstance(row_binding_sha256, str) or not _SHA256.fullmatch(row_binding_sha256):
        raise ValueError("row_binding_sha256_invalid")
    target = _indices(target_group_indices, name="target_group", upper=group_count)
    sham = _indices(sham_group_indices, name="sham_group", upper=group_count)
    evidence = _indices(evidence_group_indices, name="evidence_group", upper=group_count)
    control = _indices(control_group_indices, name="control_group", upper=group_count)
    if len(target) != len(sham):
        raise ValueError("target_sham_group_count_mismatch")
    expected_top_k = max(1, (group_count + 9) // 10)
    if len(target) != expected_top_k:
        raise ValueError("selector_group_budget_mismatch")
    if len(evidence) != len(control) or set(evidence) & set(control):
        raise ValueError("evidence_control_group_contract_invalid")

    for name, digest in (("typed_batch_sha256", typed_batch_sha256),
                         ("score_binding_sha256", score_binding_sha256)):
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"{name}_invalid")
    layers = _indices(intervention_layers_1_based, name="intervention_layers")
    if any(layer < 1 for layer in layers):
        raise ValueError("intervention_layers_invalid")
    if (isinstance(logit_bias, bool) or not isinstance(logit_bias, (int, float))
            or not math.isfinite(float(logit_bias))):
        raise ValueError("logit_bias_invalid")
    if not isinstance(sham_seed, str) or not sham_seed or len(sham_seed) > 128:
        raise ValueError("sham_seed_invalid")
    bias = float(logit_bias)
    selection_spec = {
        "selector_id": SELECTOR_ID,
        "selector_layers_1_based": list(layers),
        "top_k_budget_rule": TOP_K_BUDGET_RULE,
        "tie_break": "earliest_temporal_group_index",
        "sham_selector_id": SHAM_SELECTOR_ID,
        "sham_seed": sham_seed,
    }
    selection_spec_sha256 = _canonical_hash(selection_spec)

    positions = {
        name: tuple(sorted(position for index in selected for position in groups[index]))
        for name, selected in (("target", target), ("sham", sham),
                               ("evidence", evidence), ("control", control))
    }
    if any(not values for values in positions.values()):
        raise ValueError("branch_key_positions_empty")
    branches = []
    for name in BRANCH_ORDER:
        treatment, suffix = name.split("_", 1)
        is_treated = treatment in {"target", "sham"}
        bias_positions = positions[treatment] if is_treated else ()
        ablation_positions = (
            positions["evidence"] if suffix == "evidence_ablation" else
            positions["control"] if suffix == "control_ablation" else ())
        descriptor = {
            "name": name,
            "treatment": treatment,
            "condition": suffix,
            "row_binding_sha256": row_binding_sha256,
            "typed_batch_sha256": typed_batch_sha256,
            "score_binding_sha256": score_binding_sha256,
            "biased_visual_key_positions": list(bias_positions),
            "ablated_visual_key_positions": list(ablation_positions),
            "intervention_layers_1_based": list(layers) if is_treated else [],
            "logit_bias": bias if is_treated else 0.0,
        }
        branches.append({**descriptor, "branch_sha256": _canonical_hash(descriptor)})
    plan = {
        "schema_version": 1,
        "status": "constructed_unvalidated",
        "scope": "synthetic_or_preflighted_input_plan_only",
        "typed_batch_sha256": typed_batch_sha256,
        "score_binding_sha256": score_binding_sha256,
        "row_binding_sha256": row_binding_sha256,
        "temporal_group_count": group_count,
        "temporal_groups": [list(group) for group in groups],
        "selected_groups": {
            "target": list(target), "sham": list(sham),
            "evidence": list(evidence), "control": list(control),
        },
        "intervention_layers_1_based": list(layers),
        "logit_bias": bias,
        "selection_spec": selection_spec,
        "selection_spec_sha256": selection_spec_sha256,
        "branch_order": list(BRANCH_ORDER),
        "branches": branches,
        "requested_forwards": len(BRANCH_ORDER),
        "plan_sha256": None,
        "execution_authorization": "not_granted",
    }
    plan["plan_sha256"] = _plan_digest(plan)
    validate_qta_nine_branch_plan(plan)
    return plan


def _plan_digest(plan):
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return _canonical_hash(payload)


def validate_qta_nine_branch_plan(plan):
    """Reject branch omission, mutation, changed hashes, or authorization claims."""
    if (not isinstance(plan, Mapping)
            or set(plan) != {"schema_version", "status", "scope", "typed_batch_sha256",
                             "score_binding_sha256", "row_binding_sha256",
                             "temporal_group_count", "temporal_groups",
                             "selected_groups", "intervention_layers_1_based", "logit_bias",
                             "selection_spec", "selection_spec_sha256", "branch_order",
                             "branches", "requested_forwards", "plan_sha256",
                             "execution_authorization"}):
        raise ValueError("qta_plan_schema_invalid")
    if (type(plan["schema_version"]) is not int or plan["schema_version"] != 1
            or plan["status"] != "constructed_unvalidated"
            or plan["scope"] != "synthetic_or_preflighted_input_plan_only"
            or plan["execution_authorization"] != "not_granted"
            or type(plan["requested_forwards"]) is not int
            or plan["requested_forwards"] != 9
            or plan["branch_order"] != list(BRANCH_ORDER)
            or not isinstance(plan["branches"], list)
            or len(plan["branches"]) != 9):
        raise ValueError("qta_plan_contract_invalid")
    if (any(not isinstance(plan[field], str) or not _SHA256.fullmatch(plan[field])
            for field in ("typed_batch_sha256", "score_binding_sha256",
                          "row_binding_sha256"))):
        raise ValueError("qta_plan_binding_hash_invalid")
    raw_groups = plan["temporal_groups"]
    if (not isinstance(raw_groups, list) or not raw_groups
            or type(plan["temporal_group_count"]) is not int
            or plan["temporal_group_count"] != len(raw_groups)):
        raise ValueError("qta_plan_temporal_groups_invalid")
    groups = tuple(_indices(group, name="temporal_group") for group in raw_groups)
    if len({position for group in groups for position in group}) != sum(map(len, groups)):
        raise ValueError("qta_plan_temporal_groups_invalid")
    selected = plan["selected_groups"]
    if (not isinstance(selected, Mapping)
            or set(selected) != {"target", "sham", "evidence", "control"}):
        raise ValueError("qta_plan_selected_groups_invalid")
    normalized = {name: _indices(selected[name], name=f"{name}_group",
                                 upper=len(groups))
                  for name in ("target", "sham", "evidence", "control")}
    if len(normalized["target"]) != max(1, (len(groups) + 9) // 10):
        raise ValueError("qta_plan_selector_budget_invalid")
    if (len(normalized["target"]) != len(normalized["sham"])
            or len(normalized["evidence"]) != len(normalized["control"])
            or set(normalized["evidence"]) & set(normalized["control"])):
        raise ValueError("qta_plan_group_assignment_invalid")
    layers = _indices(plan["intervention_layers_1_based"], name="intervention_layers")
    if any(layer < 1 for layer in layers):
        raise ValueError("qta_plan_intervention_layers_invalid")
    bias = plan["logit_bias"]
    if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not math.isfinite(bias):
        raise ValueError("qta_plan_logit_bias_invalid")
    spec = plan["selection_spec"]
    if (not isinstance(spec, Mapping)
            or set(spec) != {"selector_id", "selector_layers_1_based", "top_k_budget_rule",
                             "tie_break", "sham_selector_id", "sham_seed"}
            or spec["selector_id"] != SELECTOR_ID
            or spec["selector_layers_1_based"] != list(layers)
            or spec["top_k_budget_rule"] != TOP_K_BUDGET_RULE
            or spec["tie_break"] != "earliest_temporal_group_index"
            or spec["sham_selector_id"] != SHAM_SELECTOR_ID
            or not isinstance(spec["sham_seed"], str) or not spec["sham_seed"]
            or len(spec["sham_seed"]) > 128
            or plan["selection_spec_sha256"] != _canonical_hash(spec)):
        raise ValueError("qta_plan_selection_spec_invalid")
    positions = {
        name: [position for index in selected_groups for position in groups[index]]
        for name, selected_groups in normalized.items()
    }
    for name, branch in zip(BRANCH_ORDER, plan["branches"], strict=True):
        if not isinstance(branch, Mapping) or branch.get("name") != name:
            raise ValueError("qta_plan_branch_order_invalid")
        treatment, suffix = name.split("_", 1)
        treated = treatment in {"target", "sham"}
        expected_bias = positions[treatment] if treated else []
        expected_ablation = (positions["evidence"] if suffix == "evidence_ablation" else
                             positions["control"] if suffix == "control_ablation" else [])
        descriptor = {key: value for key, value in branch.items() if key != "branch_sha256"}
        if (set(branch) != {"name", "treatment", "condition", "row_binding_sha256",
                            "typed_batch_sha256",
                            "score_binding_sha256", "biased_visual_key_positions",
                            "ablated_visual_key_positions", "intervention_layers_1_based",
                            "logit_bias", "branch_sha256"}
                or branch.get("treatment") != treatment
                or branch.get("condition") != suffix
                or branch["row_binding_sha256"] != plan["row_binding_sha256"]
                or branch["typed_batch_sha256"] != plan["typed_batch_sha256"]
                or branch["score_binding_sha256"] != plan["score_binding_sha256"]
                or branch["biased_visual_key_positions"] != expected_bias
                or branch["ablated_visual_key_positions"] != expected_ablation
                or branch["intervention_layers_1_based"] != (list(layers) if treated else [])
                or branch["logit_bias"] != (bias if treated else 0.0)
                or not isinstance(branch["branch_sha256"], str)
                or branch["branch_sha256"] != _canonical_hash(descriptor)):
            raise ValueError("qta_plan_branch_semantics_invalid")
    if plan["plan_sha256"] != _plan_digest(plan):
        raise ValueError("qta_plan_digest_mismatch")
    return True
