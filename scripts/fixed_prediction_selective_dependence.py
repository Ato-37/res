"""Draft CPU helpers for fixed-prediction selective-dependence scoring.

These helpers freeze the baseline model-predicted option and foil once, then
reuse those labels in every condition. They do not load a model, read data,
authorize scoring, or define the unresolved internal-ablation operation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from numbers import Real


CONDITION_NAMES = frozenset({"full", "evidence_ablation", "control_ablation"})
ANSWER_COUNT = 5
INTERVENTION_CONDITIONS = frozenset({
    "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
    "intervention_full", "intervention_evidence_ablation", "intervention_control_ablation",
})
ANSWER_LABEL_ORDER = ("A", "B", "C", "D", "E")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _finite_five(values: Sequence[float]) -> tuple[float, ...]:
    if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
            or len(values) != ANSWER_COUNT):
        raise ValueError("five_answer_log_probabilities_required")
    result = tuple(_finite_real(value, "answer_log_probability_invalid")
                   for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError("answer_log_probability_nonfinite")
    if any(value > 0.0 for value in result):
        raise ValueError("answer_log_probability_positive")
    return result


def _finite_real(value: object, error_code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(error_code)
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(error_code) from None
    if not math.isfinite(result):
        raise ValueError("answer_log_probability_nonfinite"
                         if error_code == "answer_log_probability_invalid"
                         else "condition_margin_nonfinite")
    return result


def freeze_baseline_answer_and_foil(
    baseline_full_log_probabilities: Sequence[float],
) -> tuple[int, int]:
    """Freeze the highest-scoring baseline option and its best competitor.

    Indices follow A–E order. Python's stable numeric ordering gives the
    predeclared alphabetical tie break for both the prediction and foil.
    """
    values = _finite_five(baseline_full_log_probabilities)
    ranked = sorted(range(ANSWER_COUNT), key=lambda index: (-values[index], index))
    return ranked[0], ranked[1]


def fixed_answer_foil_margin(
    log_probabilities: Sequence[float], *, answer_index: int, foil_index: int,
) -> float:
    """Return one condition's margin using externally frozen label indices."""
    values = _finite_five(log_probabilities)
    if (type(answer_index) is not int or type(foil_index) is not int
            or not 0 <= answer_index < ANSWER_COUNT
            or not 0 <= foil_index < ANSWER_COUNT
            or answer_index == foil_index):
        raise ValueError("frozen_answer_foil_indices_invalid")
    return values[answer_index] - values[foil_index]


def selective_dependence_contrast(
    *, baseline_margins: Mapping[str, float],
    intervention_margins: Mapping[str, float],
) -> dict[str, float]:
    """Compute F and ΔF from margins already scored with fixed answer/foil.

    ``F = (full - evidence) - (full - control) = control - evidence``.
    The explicit branch drops are returned as diagnostics; this helper makes
    no population or causal claim and performs no statistical inference.
    """
    if (not isinstance(baseline_margins, Mapping)
            or not isinstance(intervention_margins, Mapping)
            or set(baseline_margins) != CONDITION_NAMES
            or set(intervention_margins) != CONDITION_NAMES):
        raise ValueError("exact_three_condition_margins_required")
    base = {key: _finite_real(baseline_margins[key], "condition_margin_invalid")
            for key in CONDITION_NAMES}
    treated = {key: _finite_real(intervention_margins[key], "condition_margin_invalid")
               for key in CONDITION_NAMES}

    base_evidence_drop = base["full"] - base["evidence_ablation"]
    base_control_drop = base["full"] - base["control_ablation"]
    treatment_evidence_drop = treated["full"] - treated["evidence_ablation"]
    treatment_control_drop = treated["full"] - treated["control_ablation"]
    base_f = base_evidence_drop - base_control_drop
    treatment_f = treatment_evidence_drop - treatment_control_drop
    derived = (base_evidence_drop, base_control_drop, treatment_evidence_drop,
               treatment_control_drop, base_f, treatment_f, treatment_f - base_f)
    if any(not math.isfinite(value) for value in derived):
        raise ValueError("condition_margin_arithmetic_nonfinite")
    return {
        "baseline_evidence_drop": base_evidence_drop,
        "baseline_control_drop": base_control_drop,
        "baseline_selective_dependence": base_f,
        "intervention_evidence_drop": treatment_evidence_drop,
        "intervention_control_drop": treatment_control_drop,
        "intervention_selective_dependence": treatment_f,
        "delta_f": treatment_f - base_f,
    }


def fixed_prediction_intervention_contrast(
    condition_log_probabilities: Mapping[str, Sequence[float]],
) -> dict:
    """Freeze answer/foil from baseline-full once, then score all six branches.

    This is the pure arithmetic contract for the proposed internal-intervention
    estimand. It expects already-bound A–E token log probabilities from the
    same single next-token position in every branch. It does not tokenize,
    inspect inputs, run a model, or authorize a real-data score.
    """
    if (not isinstance(condition_log_probabilities, Mapping)
            or set(condition_log_probabilities) != INTERVENTION_CONDITIONS):
        raise ValueError("exact_six_intervention_conditions_required")
    values = {condition: _finite_five(condition_log_probabilities[condition])
              for condition in INTERVENTION_CONDITIONS}
    answer_index, foil_index = freeze_baseline_answer_and_foil(values["baseline_full"])
    margins = {
        condition: fixed_answer_foil_margin(
            values[condition], answer_index=answer_index, foil_index=foil_index)
        for condition in INTERVENTION_CONDITIONS
    }
    result = selective_dependence_contrast(
        baseline_margins={
            "full": margins["baseline_full"],
            "evidence_ablation": margins["baseline_evidence_ablation"],
            "control_ablation": margins["baseline_control_ablation"],
        },
        intervention_margins={
            "full": margins["intervention_full"],
            "evidence_ablation": margins["intervention_evidence_ablation"],
            "control_ablation": margins["intervention_control_ablation"],
        },
    )
    return {
        "status": "synthetic_contract_only",
        "answer_index_frozen_from": "baseline_full",
        "answer_index": answer_index,
        "foil_index_frozen_from": "baseline_full",
        "foil_index": foil_index,
        "margin_by_condition": margins,
        **result,
        "interpretation_limit": (
            "six-condition fixed-label arithmetic only; no tokenizer/prefix/position "
            "binding, model inference, human review, or faithfulness efficacy"
        ),
    }


def validate_six_condition_input_binding(branch_bindings: Mapping[str, Mapping]) -> dict:
    """Require identical tokenizer, prompt-token inputs, labels, and score row.

    The caller must compute each digest from the actual tokenizer/model input
    and bind the same hook-free text prefix before calling this validator. The
    returned digest summarizes declarations; this pure helper cannot
    authenticate their source or run a model.
    """
    if (not isinstance(branch_bindings, Mapping)
            or set(branch_bindings) != INTERVENTION_CONDITIONS):
        raise ValueError("exact_six_intervention_bindings_required")
    required = {
        "answer_label_order", "answer_token_ids", "input_ids_sha256",
        "attention_mask_sha256", "tokenizer_sha256", "model_revision",
        "prompt_sha256", "input_length", "score_position",
    }
    canonical = None
    for condition in INTERVENTION_CONDITIONS:
        row = branch_bindings[condition]
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("condition_binding_schema_invalid")
        if (not isinstance(row["answer_label_order"], (list, tuple))
                or tuple(row["answer_label_order"]) != ANSWER_LABEL_ORDER
                or not isinstance(row["answer_token_ids"], (list, tuple))
                or len(row["answer_token_ids"]) != len(ANSWER_LABEL_ORDER)
                or any(type(token_id) is not int or token_id < 0
                       for token_id in row["answer_token_ids"])
                or len(set(row["answer_token_ids"])) != len(ANSWER_LABEL_ORDER)):
            raise ValueError("answer_label_token_binding_invalid")
        for key in ("input_ids_sha256", "attention_mask_sha256",
                    "tokenizer_sha256", "prompt_sha256"):
            if not isinstance(row[key], str) or SHA256_PATTERN.fullmatch(row[key]) is None:
                raise ValueError("condition_binding_hash_invalid")
        if not isinstance(row["model_revision"], str) or not row["model_revision"].strip():
            raise ValueError("condition_binding_model_invalid")
        if (type(row["input_length"]) is not int or row["input_length"] <= 0
                or type(row["score_position"]) is not int
                or row["score_position"] != row["input_length"] - 1):
            raise ValueError("condition_binding_score_position_invalid")
        current = {
            "answer_label_order": list(ANSWER_LABEL_ORDER),
            "answer_token_ids": list(row["answer_token_ids"]),
            "input_ids_sha256": row["input_ids_sha256"],
            "attention_mask_sha256": row["attention_mask_sha256"],
            "tokenizer_sha256": row["tokenizer_sha256"],
            "model_revision": row["model_revision"],
            "prompt_sha256": row["prompt_sha256"],
            "input_length": row["input_length"],
            "score_position": row["score_position"],
        }
        if canonical is None:
            canonical = current
        elif current != canonical:
            raise ValueError("six_condition_scoring_binding_mismatch")
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return {
        "status": "synthetic_binding_contract_only",
        "bound_conditions": len(INTERVENTION_CONDITIONS),
        "input_length": canonical["input_length"],
        "score_position": canonical["score_position"],
        "answer_token_ids_sha256": hashlib.sha256(
            json.dumps(canonical["answer_token_ids"], separators=(",", ":")).encode()
        ).hexdigest(),
        "binding_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "execution_authorization": "not_granted",
        "interpretation_limit": (
            "caller-supplied binding declarations only; does not authenticate tokenizer, "
            "prompt, model input, human review, or faithfulness score"
        ),
    }
