"""Draft CPU helpers for fixed-prediction selective-dependence scoring.

These helpers freeze the baseline model-predicted option and foil once, then
reuse those labels in every condition. They do not load a model, read data,
authorize scoring, or define the unresolved internal-ablation operation.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real


CONDITION_NAMES = frozenset({"full", "evidence_ablation", "control_ablation"})
ANSWER_COUNT = 5


def _finite_five(values: Sequence[float]) -> tuple[float, ...]:
    if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
            or len(values) != ANSWER_COUNT):
        raise ValueError("five_answer_log_probabilities_required")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("answer_log_probability_invalid")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError("answer_log_probability_nonfinite")
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
    raw_values = tuple(baseline_margins.values()) + tuple(intervention_margins.values())
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_values):
        raise ValueError("condition_margin_invalid") from None
    base = {key: float(baseline_margins[key]) for key in CONDITION_NAMES}
    treated = {key: float(intervention_margins[key]) for key in CONDITION_NAMES}
    if any(not math.isfinite(value) for value in (*base.values(), *treated.values())):
        raise ValueError("condition_margin_nonfinite")

    base_evidence_drop = base["full"] - base["evidence_ablation"]
    base_control_drop = base["full"] - base["control_ablation"]
    treatment_evidence_drop = treated["full"] - treated["evidence_ablation"]
    treatment_control_drop = treated["full"] - treated["control_ablation"]
    base_f = base_evidence_drop - base_control_drop
    treatment_f = treatment_evidence_drop - treatment_control_drop
    return {
        "baseline_evidence_drop": base_evidence_drop,
        "baseline_control_drop": base_control_drop,
        "baseline_selective_dependence": base_f,
        "intervention_evidence_drop": treatment_evidence_drop,
        "intervention_control_drop": treatment_control_drop,
        "intervention_selective_dependence": treatment_f,
        "delta_f": treatment_f - base_f,
    }
