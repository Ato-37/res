"""Reproduce QTA target/sham selection from declared per-layer group masses.

This is an arithmetic-only, non-authorizing contract. It does not prove that
the supplied group masses came from a Qwen baseline forward, or that question
positions and temporal groups are semantically correct.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = 1
PURPOSE = "qta_selection_arithmetic_provenance_non_authorizing"
AGGREGATION_ID = "ordered_python_sum_of_layer_group_mass_f64_div_layer_count_v1"
FROZEN_SELECTOR_LAYERS_1_BASED = list(range(5, 17))
FROZEN_SELECTOR_SOURCE_SHA256 = (
    "d1a369636cb5cfc1ff6d0ffb0f86e94fa3643fa7bc878c9a06014871284e12f2"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")
_PROVENANCE_KEYS = {
    "schema_version", "purpose", "status", "record_token",
    "opaque_review_token",
    "row_binding_sha256", "typed_batch_sha256", "score_binding_sha256",
    "baseline_pq_row_sha256", "prompt_sha256", "input_ids_sha256",
    "input_length", "score_position",
    "question_positions", "question_positions_sha256", "temporal_groups",
    "temporal_group_count", "selector_source_sha256",
    "selector_layers_1_based", "aggregation_id", "top_k",
    "layer_group_mass_f32_be_hex", "group_scores_f64_be_hex",
    "target_ranked_group_indices", "target_group_indices", "sham_seed",
    "sham_ranked_group_indices", "sham_group_indices",
    "forward_source_authenticated", "attention_capture_authenticated",
    "selection_arithmetic_reproduced", "execution_authorization",
    "model_load_permitted", "scoring_authorized", "authorized_forwards",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError):
        raise ValueError("qta_selection_json_invalid") from None


def _hex_f32(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("qta_selection_group_mass_invalid")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("qta_selection_group_mass_invalid")
    try:
        raw = struct.pack(">f", value)
    except (OverflowError, struct.error):
        raise ValueError("qta_selection_group_mass_invalid") from None
    rounded = struct.unpack(">f", raw)[0]
    if not math.isfinite(rounded) or rounded < 0:
        raise ValueError("qta_selection_group_mass_invalid")
    return raw.hex()


def _decode_f32(value: Any) -> float:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{8}", value):
        raise ValueError("qta_selection_f32_encoding_invalid")
    decoded = struct.unpack(">f", bytes.fromhex(value))[0]
    if not math.isfinite(decoded) or decoded < 0 or _hex_f32(decoded) != value:
        raise ValueError("qta_selection_f32_value_invalid")
    return float(decoded)


def _hex_f64(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("qta_selection_group_score_invalid")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("qta_selection_group_score_invalid")
    raw = struct.pack(">d", value)
    return raw.hex()


def _decode_f64(value: Any) -> float:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value):
        raise ValueError("qta_selection_f64_encoding_invalid")
    decoded = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(decoded) or decoded < 0 or _hex_f64(decoded) != value:
        raise ValueError("qta_selection_f64_value_invalid")
    return decoded


def _group_geometry(value: Any) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("qta_selection_temporal_groups_invalid")
    groups = []
    for group in value:
        if (not isinstance(group, Sequence) or isinstance(group, (str, bytes))
                or not group or any(type(position) is not int or position < 0
                                    for position in group)):
            raise ValueError("qta_selection_temporal_groups_invalid")
        positions = list(group)
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise ValueError("qta_selection_temporal_groups_invalid")
        groups.append(positions)
    flattened = [position for group in groups for position in group]
    if (len(set(flattened)) != len(flattened)
            or flattened != list(range(flattened[0], flattened[0] + len(flattened)))
            or len({len(group) for group in groups}) != 1):
        raise ValueError("qta_selection_temporal_groups_invalid")
    return groups


def _is_exact_indices(value: Any, expected: list[int]) -> bool:
    return (isinstance(value, list) and len(value) == len(expected)
            and all(type(index) is int for index in value)
            and value == expected)


def select_top_temporal_groups(scores: Sequence[float], *, top_k: int) -> list[int]:
    if (not isinstance(scores, Sequence) or isinstance(scores, (str, bytes))
            or not scores or type(top_k) is not int or not 1 <= top_k <= len(scores)):
        raise ValueError("qta_selection_top_k_invalid")
    if any(not math.isfinite(score) or score < 0 for score in scores):
        raise ValueError("qta_selection_group_score_invalid")
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]


def select_hash_sham_groups(*, group_count: int, top_k: int, seed: str,
                            opaque_review_token: str) -> tuple[list[int], list[int]]:
    if (type(group_count) is not int or group_count < 1
            or type(top_k) is not int or not 1 <= top_k <= group_count
            or not isinstance(seed, str) or not seed or "\x00" in seed
            or not isinstance(opaque_review_token, str)
            or not _REVIEW_TOKEN.fullmatch(opaque_review_token)):
        raise ValueError("qta_selection_sham_input_invalid")
    prefix = f"qta-sham-v1\0{seed}\0{opaque_review_token}\0"
    ranked = sorted(
        range(group_count),
        key=lambda index: (hashlib.sha256(f"{prefix}{index}".encode("utf-8")).digest(),
                           index),
    )
    return ranked, sorted(ranked[:top_k])


def build_qta_selection_provenance(*, record_token: str,
                                   opaque_review_token: str,
                                   row_binding_sha256: str,
                                   typed_batch_sha256: str,
                                   score_binding_sha256: str,
                                   baseline_pq_row_sha256: str,
                                   prompt_sha256: str, input_ids_sha256: str,
                                   input_length: int, score_position: int,
                                   question_positions: Sequence[int],
                                   temporal_groups: Sequence[Sequence[int]],
                                   selector_source_sha256: str,
                                   selector_layers_1_based: Sequence[int],
                                   layer_group_masses: Sequence[Sequence[float]],
                                   sham_seed: str) -> dict[str, Any]:
    hashes = (record_token, row_binding_sha256, typed_batch_sha256,
              score_binding_sha256, baseline_pq_row_sha256, prompt_sha256,
              input_ids_sha256, selector_source_sha256)
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in hashes):
        raise ValueError("qta_selection_binding_digest_invalid")
    if (not isinstance(opaque_review_token, str)
            or not _REVIEW_TOKEN.fullmatch(opaque_review_token)):
        raise ValueError("qta_selection_sham_input_invalid")
    if (type(input_length) is not int or input_length < 1
            or type(score_position) is not int or score_position != input_length - 1):
        raise ValueError("qta_selection_score_position_invalid")
    if selector_source_sha256 != FROZEN_SELECTOR_SOURCE_SHA256:
        raise ValueError("qta_selection_source_not_frozen")
    groups = _group_geometry(temporal_groups)
    if (not isinstance(question_positions, Sequence) or isinstance(question_positions, (str, bytes))
            or not question_positions
            or any(type(position) is not int or position < 0 for position in question_positions)
            or list(question_positions) != list(range(question_positions[0],
                                                       question_positions[-1] + 1))
            or groups[-1][-1] >= question_positions[0]
            or question_positions[-1] >= score_position):
        raise ValueError("qta_selection_question_positions_invalid")
    layers = list(selector_layers_1_based)
    if (layers != FROZEN_SELECTOR_LAYERS_1_BASED
            or any(type(layer) is not int or layer < 1 for layer in layers)):
        raise ValueError("qta_selection_layers_invalid")
    if (not isinstance(layer_group_masses, Sequence)
            or len(layer_group_masses) != len(layers)
            or any(not isinstance(row, Sequence) or isinstance(row, (str, bytes))
                   or len(row) != len(groups) for row in layer_group_masses)):
        raise ValueError("qta_selection_layer_mass_shape_invalid")
    masses = [[_hex_f32(value) for value in row] for row in layer_group_masses]
    decoded = [[_decode_f32(value) for value in row] for row in masses]
    group_scores = [sum(row[index] for row in decoded) / len(decoded)
                    for index in range(len(groups))]
    group_score_hex = [_hex_f64(score) for score in group_scores]
    top_k = max(1, (len(groups) + 9) // 10)
    target_ranked = select_top_temporal_groups(group_scores, top_k=top_k)
    sham_ranked, sham_selected = select_hash_sham_groups(
        group_count=len(groups), top_k=top_k, seed=sham_seed,
        opaque_review_token=opaque_review_token)
    question_positions = list(question_positions)
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "status": "arithmetic_only_non_authorizing",
        "record_token": record_token,
        "opaque_review_token": opaque_review_token,
        "row_binding_sha256": row_binding_sha256,
        "typed_batch_sha256": typed_batch_sha256,
        "score_binding_sha256": score_binding_sha256,
        "baseline_pq_row_sha256": baseline_pq_row_sha256,
        "prompt_sha256": prompt_sha256,
        "input_ids_sha256": input_ids_sha256,
        "input_length": input_length,
        "score_position": score_position,
        "question_positions": question_positions,
        "question_positions_sha256": _sha(_canonical(question_positions)),
        "temporal_groups": groups,
        "temporal_group_count": len(groups),
        "selector_source_sha256": selector_source_sha256,
        "selector_layers_1_based": layers,
        "aggregation_id": AGGREGATION_ID,
        "top_k": top_k,
        "layer_group_mass_f32_be_hex": masses,
        "group_scores_f64_be_hex": group_score_hex,
        "target_ranked_group_indices": target_ranked,
        "target_group_indices": sorted(target_ranked),
        "sham_seed": sham_seed,
        "sham_ranked_group_indices": sham_ranked,
        "sham_group_indices": sham_selected,
        "forward_source_authenticated": False,
        "attention_capture_authenticated": False,
        "selection_arithmetic_reproduced": True,
        "execution_authorization": "not_granted",
        "model_load_permitted": False,
        "scoring_authorized": False,
        "authorized_forwards": 0,
    }


def validate_qta_selection_provenance(value: Any, *, expected_bindings: Mapping[str, Any],
                                      plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or set(value) != _PROVENANCE_KEYS
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != SCHEMA_VERSION
            or value["purpose"] != PURPOSE
            or value["status"] != "arithmetic_only_non_authorizing"
            or value["forward_source_authenticated"] is not False
            or value["attention_capture_authenticated"] is not False
            or value["selection_arithmetic_reproduced"] is not True
            or value["execution_authorization"] != "not_granted"
            or value["model_load_permitted"] is not False
            or value["scoring_authorized"] is not False
            or type(value["authorized_forwards"]) is not int
            or value["authorized_forwards"] != 0):
        raise ValueError("qta_selection_provenance_schema_or_authority_invalid")
    if (not isinstance(expected_bindings, Mapping)
            or set(expected_bindings) != {
                "record_token", "row_binding_sha256", "typed_batch_sha256",
                "opaque_review_token",
                "score_binding_sha256", "baseline_pq_row_sha256", "prompt_sha256",
                "input_ids_sha256", "input_length", "score_position",
                "question_positions", "temporal_groups",
                "selector_source_sha256", "selector_layers_1_based", "sham_seed"}):
        raise ValueError("qta_selection_expected_binding_schema_invalid")
    for field in ("record_token", "row_binding_sha256", "typed_batch_sha256",
                  "score_binding_sha256", "baseline_pq_row_sha256", "prompt_sha256",
                  "input_ids_sha256", "selector_source_sha256"):
        expected = expected_bindings[field]
        if (not isinstance(expected, str) or not _SHA256.fullmatch(expected)
                or value[field] != expected):
            raise ValueError("qta_selection_binding_mismatch")
    if (type(expected_bindings["input_length"]) is not int
            or expected_bindings["input_length"] < 1
            or type(expected_bindings["score_position"]) is not int
            or expected_bindings["score_position"] != expected_bindings["input_length"] - 1
            or type(value["input_length"]) is not int
            or type(value["score_position"]) is not int
            or value["input_length"] != expected_bindings["input_length"]
            or value["score_position"] != expected_bindings["score_position"]):
        raise ValueError("qta_selection_score_position_mismatch")
    if expected_bindings["selector_source_sha256"] != FROZEN_SELECTOR_SOURCE_SHA256:
        raise ValueError("qta_selection_source_not_frozen")
    if (not isinstance(expected_bindings["opaque_review_token"], str)
            or not _REVIEW_TOKEN.fullmatch(expected_bindings["opaque_review_token"])
            or value["opaque_review_token"] != expected_bindings["opaque_review_token"]):
        raise ValueError("qta_selection_sham_token_mismatch")
    groups = _group_geometry(expected_bindings["temporal_groups"])
    positions = expected_bindings["question_positions"]
    value_groups = _group_geometry(value["temporal_groups"])
    if (value_groups != groups or type(value["temporal_group_count"]) is not int
            or value["temporal_group_count"] != len(groups)
            or not isinstance(value["question_positions"], list)
            or any(type(position) is not int for position in value["question_positions"])
            or value["question_positions"] != positions
            or positions[-1] >= value["score_position"]
            or groups[-1][-1] >= positions[0]
            or value["question_positions_sha256"] != _sha(_canonical(positions))
            or value["selector_layers_1_based"] != expected_bindings["selector_layers_1_based"]
            or value["sham_seed"] != expected_bindings["sham_seed"]):
        raise ValueError("qta_selection_geometry_or_protocol_mismatch")
    layers = value["selector_layers_1_based"]
    masses = value["layer_group_mass_f32_be_hex"]
    if (not isinstance(layers, list) or layers != FROZEN_SELECTOR_LAYERS_1_BASED
            or any(type(layer) is not int or layer < 1 for layer in layers)
            or not isinstance(masses, list) or len(masses) != len(layers)
            or any(not isinstance(row, list) or len(row) != len(groups) for row in masses)):
        raise ValueError("qta_selection_layer_mass_shape_invalid")
    decoded = [[_decode_f32(item) for item in row] for row in masses]
    scores = [sum(row[index] for row in decoded) / len(decoded)
              for index in range(len(groups))]
    score_hex = [_hex_f64(score) for score in scores]
    top_k = max(1, (len(groups) + 9) // 10)
    target_ranked = select_top_temporal_groups(scores, top_k=top_k)
    sham_ranked, sham_selected = select_hash_sham_groups(
        group_count=len(groups), top_k=top_k, seed=value["sham_seed"],
        opaque_review_token=value["opaque_review_token"])
    if (value["aggregation_id"] != AGGREGATION_ID
            or type(value["top_k"]) is not int or value["top_k"] != top_k
            or value["group_scores_f64_be_hex"] != score_hex
            or not _is_exact_indices(value["target_ranked_group_indices"], target_ranked)
            or not _is_exact_indices(value["target_group_indices"], sorted(target_ranked))
            or not _is_exact_indices(value["sham_ranked_group_indices"], sham_ranked)
            or not _is_exact_indices(value["sham_group_indices"], sham_selected)):
        raise ValueError("qta_selection_recomputation_mismatch")
    if plan is not None:
        from qta_nine_branch_plan import (
            SELECTOR_ID, SHAM_SELECTOR_ID, TOP_K_BUDGET_RULE,
            validate_qta_nine_branch_plan,
        )
        try:
            validate_qta_nine_branch_plan(plan)
        except (TypeError, ValueError, KeyError, OverflowError):
            raise ValueError("qta_selection_plan_invalid") from None
        if (plan.get("row_binding_sha256") != expected_bindings["row_binding_sha256"]
                or plan.get("typed_batch_sha256") != expected_bindings["typed_batch_sha256"]
                or plan.get("score_binding_sha256")
                != expected_bindings["score_binding_sha256"]):
            raise ValueError("qta_selection_plan_binding_mismatch")
        spec = plan["selection_spec"]
        if (spec["selector_id"] != SELECTOR_ID
                or spec["selector_layers_1_based"] != layers
                or spec["top_k_budget_rule"] != TOP_K_BUDGET_RULE
                or spec["tie_break"] != "earliest_temporal_group_index"
                or spec["sham_selector_id"] != SHAM_SELECTOR_ID
                or spec["sham_seed"] != value["sham_seed"]):
            raise ValueError("qta_selection_plan_protocol_mismatch")
        if (plan.get("temporal_groups") != groups
                or plan.get("temporal_group_count") != len(groups)
                or plan.get("selected_groups", {}).get("target") != sorted(target_ranked)
                or plan.get("selected_groups", {}).get("sham") != sham_selected):
            raise ValueError("qta_selection_plan_group_mismatch")
        positions_by_kind = {
            name: [position for index in plan["selected_groups"][name]
                   for position in groups[index]]
            for name in ("target", "sham", "evidence", "control")
        }
        for branch in plan.get("branches", []):
            treatment = branch.get("treatment")
            expected_positions = (positions_by_kind[treatment]
                                  if treatment in ("target", "sham") else [])
            if branch.get("biased_visual_key_positions") != expected_positions:
                raise ValueError("qta_selection_plan_key_position_mismatch")
    return {
        "status": "selection_arithmetic_reproduced_non_authorizing",
        "selection_arithmetic_reproduced": True,
        "forward_source_authenticated": False,
        "attention_capture_authenticated": False,
        "validated_group_count": len(groups),
        "target_group_indices": sorted(target_ranked),
        "sham_group_indices": sham_selected,
        "execution_authorization": "not_granted",
        "model_load_permitted": False,
        "scoring_authorized": False,
        "authorized_forwards": 0,
    }


def selection_provenance_sha256(value: Mapping[str, Any]) -> str:
    return _sha(_canonical(value))
