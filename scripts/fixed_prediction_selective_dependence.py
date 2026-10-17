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
    # Compute the algebraically equivalent reduced form directly. Subtracting
    # two branch drops would reintroduce a large shared full margin and lose
    # precision through cancellation, even though `full` cancels by definition.
    base_f = base["control_ablation"] - base["evidence_ablation"]
    treatment_f = treated["control_ablation"] - treated["evidence_ablation"]
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
def score_six_condition_logits(
    *, condition_outputs: Mapping[str, object], answer_token_ids: Sequence[int],
    input_lengths: Mapping[str, int], torch_module: object,
) -> dict:
    """Extract fixed A–E next-token log-probabilities from six model outputs.

    Accepts either full-sequence logits or a `logits_to_keep=1` output, always
    taking the last returned row and binding it to the caller's recorded final
    prompt position. This validates tensor shape and arithmetic; it cannot
    authenticate the model inputs or establish a treatment/faithfulness result.
    """
    if (not isinstance(condition_outputs, Mapping)
            or set(condition_outputs) != INTERVENTION_CONDITIONS
            or not isinstance(input_lengths, Mapping)
            or set(input_lengths) != INTERVENTION_CONDITIONS):
        raise ValueError("exact_six_intervention_outputs_required")
    if (not isinstance(answer_token_ids, Sequence)
            or isinstance(answer_token_ids, (str, bytes))
            or len(answer_token_ids) != ANSWER_COUNT
            or any(type(token_id) is not int or token_id < 0
                   for token_id in answer_token_ids)
            or len(set(answer_token_ids)) != ANSWER_COUNT):
        raise ValueError("answer_token_ids_invalid")
    if torch_module is None or not callable(getattr(torch_module, "log_softmax", None)):
        raise ValueError("torch_module_invalid")
    if (any(type(value) is not int or value <= 0 for value in input_lengths.values())
            or len(set(input_lengths.values())) != 1):
        raise ValueError("six_condition_input_length_mismatch")

    scores = {}
    common_vocab_size = None
    bound_positions = {}
    for condition in sorted(INTERVENTION_CONDITIONS):
        input_length = input_lengths[condition]
        if type(input_length) is not int or input_length <= 0:
            raise ValueError("input_length_invalid")
        output = condition_outputs[condition]
        logits = (output.get("logits") if isinstance(output, Mapping)
                  else getattr(output, "logits", None))
        shape = getattr(logits, "shape", None)
        if (logits is None or shape is None or len(shape) != 3
                or int(shape[0]) != 1 or int(shape[1]) not in (1, input_length)
                or int(shape[2]) <= 0):
            raise ValueError("next_token_logits_shape_invalid")
        vocab_size = int(shape[2])
        if common_vocab_size is None:
            common_vocab_size = vocab_size
        elif vocab_size != common_vocab_size:
            raise ValueError("condition_vocab_size_mismatch")
        if any(token_id >= vocab_size for token_id in answer_token_ids):
            raise ValueError("answer_token_id_out_of_range")
        try:
            last_logits = logits[0, -1].float()
            if not bool(torch_module.isfinite(last_logits).all()):
                raise ValueError("next_token_logits_nonfinite")
            log_probs = torch_module.log_softmax(last_logits, dim=-1)
            selected = log_probs[list(answer_token_ids)].detach().cpu().tolist()
        except ValueError:
            raise
        except Exception:
            raise ValueError("next_token_logits_scoring_failed") from None
        values = _finite_five(selected)
        scores[condition] = list(values)
        bound_positions[condition] = input_length - 1

    contrast = fixed_prediction_intervention_contrast(scores)
    return {
        "status": "logits_adapter_contract_only",
        "answer_token_ids": list(answer_token_ids),
        "score_position_by_condition": bound_positions,
        "choice_log_probabilities_by_condition": scores,
        "contrast": contrast,
        "interpretation_limit": (
            "extracts six caller-supplied model output tensors only; input, tokenizer, "
            "treatment, video provenance, human review, and faithfulness are unauthenticated"
        ),
    }


def score_bound_six_condition_logits(
    *, branch_bindings: Mapping[str, Mapping],
    condition_outputs: Mapping[str, object], torch_module: object,
) -> dict:
    """Validate six branch declarations and derive all scorer arguments from them.

    This closes the local API seam where independently validated bindings
    could be paired with different caller-supplied token IDs or input lengths.
    The binding declarations and output tensors remain caller-supplied; this
    helper does not authenticate their provenance or prove they came from the
    declared model inputs.
    """
    binding = validate_six_condition_input_binding(branch_bindings)
    first = branch_bindings["baseline_full"]
    answer_token_ids = first["answer_token_ids"]
    input_length = first["input_length"]
    score_position = first["score_position"]
    result = score_six_condition_logits(
        condition_outputs=condition_outputs,
        answer_token_ids=answer_token_ids,
        input_lengths={condition: input_length for condition in INTERVENTION_CONDITIONS},
        torch_module=torch_module,
    )
    if (result.get("score_position_by_condition") != {
            condition: score_position for condition in INTERVENTION_CONDITIONS}):
        raise ValueError("bound_score_position_mismatch")
    return {
        **result,
        "status": "bound_logits_adapter_contract_only",
        "input_binding_sha256": binding["binding_sha256"],
        "interpretation_limit": (
            "token IDs and score position are derived from equal caller declarations; "
            "input hashes, processor pixels, model outputs, video provenance, human review, "
            "and faithfulness efficacy are not authenticated"
        ),
    }
