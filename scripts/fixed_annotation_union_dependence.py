"""Four-condition fixed-prediction arithmetic for one-sided annotation masking.

This contract is distinct from the six-condition matched-control DeltaF metric.
It uses the baseline-full predicted answer/foil in all four branches.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from fixed_prediction_selective_dependence import (
    fixed_answer_foil_margin, freeze_baseline_answer_and_foil, _finite_five,
)

CONDITIONS = frozenset({"baseline_full", "baseline_union_mask",
                        "intervention_full", "intervention_union_mask"})


def contrast_from_choice_log_probabilities(values: Mapping[str, Sequence[float]]) -> dict:
    if not isinstance(values, Mapping) or set(values) != CONDITIONS:
        raise ValueError("exact_four_union_conditions_required")
    scores = {name: list(_finite_five(values[name])) for name in CONDITIONS}
    answer, foil = freeze_baseline_answer_and_foil(scores["baseline_full"])
    margins = {name: fixed_answer_foil_margin(row, answer_index=answer, foil_index=foil)
               for name, row in scores.items()}
    drops = {treatment: margins[f"{treatment}_full"] - margins[f"{treatment}_union_mask"]
             for treatment in ("baseline", "intervention")}
    delta = drops["intervention"] - drops["baseline"]
    if not math.isfinite(delta):
        raise ValueError("union_dependence_contrast_nonfinite")
    return {"metric": "delta_union_dependence",
        "answer_index": answer, "foil_index": foil,
        "answer_foil_source": "baseline_full_frozen_for_all_four_conditions",
        "margin_by_condition": margins,
        "baseline_union_dependence": drops["baseline"],
        "intervention_union_dependence": drops["intervention"],
        "delta_union_dependence": delta,
        "sign_interpretation": "positive means QTA increases the fixed answer/foil margin drop after masking the full annotation-covered union",
        "inferential_status": "finite-cohort exploratory; no p-value or population confidence interval"}


def score_four_condition_logits(*, condition_outputs: Mapping[str, object],
                                 answer_token_ids: Sequence[int],
                                 input_lengths: Mapping[str, int], torch_module: object) -> dict:
    """Extract A-E log-probs from exactly four same-prompt forward outputs."""
    if (not isinstance(condition_outputs, Mapping) or set(condition_outputs) != CONDITIONS
            or not isinstance(input_lengths, Mapping) or set(input_lengths) != CONDITIONS):
        raise ValueError("exact_four_union_conditions_required")
    if (not isinstance(answer_token_ids, Sequence) or isinstance(answer_token_ids, (str, bytes))
            or len(answer_token_ids) != 5
            or any(type(token_id) is not int or token_id < 0 for token_id in answer_token_ids)
            or len(set(answer_token_ids)) != 5):
        raise ValueError("answer_token_ids_invalid")
    if (torch_module is None or not callable(getattr(torch_module, "log_softmax", None))
            or any(type(n) is not int or n <= 0 for n in input_lengths.values())
            or len(set(input_lengths.values())) != 1):
        raise ValueError("four_condition_input_contract_invalid")
    values, vocab_size = {}, None
    positions = {}
    for name in sorted(CONDITIONS):
        output = condition_outputs[name]
        logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
        shape = getattr(logits, "shape", None)
        if (shape is None or len(shape) != 3 or int(shape[0]) != 1
                or int(shape[1]) not in (1, input_lengths[name]) or int(shape[2]) <= 0):
            raise ValueError("next_token_logits_shape_invalid")
        if vocab_size is None:
            vocab_size = int(shape[2])
        elif vocab_size != int(shape[2]):
            raise ValueError("condition_vocab_size_mismatch")
        if any(token >= vocab_size for token in answer_token_ids):
            raise ValueError("answer_token_id_out_of_range")
        try:
            row = logits[0, -1].float()
            if not bool(torch_module.isfinite(row).all()):
                raise ValueError("next_token_logits_nonfinite")
            log_probs = torch_module.log_softmax(row, dim=-1)
            values[name] = log_probs[list(answer_token_ids)].detach().cpu().tolist()
        except ValueError:
            raise
        except Exception:
            raise ValueError("next_token_logits_scoring_failed") from None
        positions[name] = input_lengths[name] - 1
    result = contrast_from_choice_log_probabilities(values)
    return {"status": "four_condition_logits_contract_only",
        "answer_token_ids": list(answer_token_ids),
        "score_position_by_condition": positions,
        "choice_log_probabilities_by_condition": values,
        "contrast": result,
        "interpretation_limit": "validates fixed A-E arithmetic only; caller must establish shared inputs, treatment, video provenance, and model execution"}
