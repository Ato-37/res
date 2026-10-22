#!/usr/bin/env python3
"""Outcome-free property audit of the frozen QTA v4 four-condition arithmetic.

Uses deterministic synthetic normalized five-choice log probabilities and an
independent scalar reference formula. It reads no dataset or private artifact,
loads no model, and cannot establish annotation validity or faithfulness.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from typing import Sequence

from fixed_annotation_union_dependence import (
    contrast_from_choice_log_probabilities,
)

SEED = 20261018
DEFAULT_DRAWS = 10_000
NAMES = ("baseline_full", "baseline_union_mask",
         "intervention_full", "intervention_union_mask")


def _log_softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    return [value - normalizer for value in values]


def _reference(values: dict[str, list[float]]) -> tuple[int, int, float, float, float]:
    baseline = values["baseline_full"]
    ranking = sorted(range(5), key=lambda i: (-baseline[i], i))
    answer, foil = ranking[:2]
    base_margin_full = baseline[answer] - baseline[foil]
    base_mask = values["baseline_union_mask"][answer] - values["baseline_union_mask"][foil]
    treated_full = values["intervention_full"][answer] - values["intervention_full"][foil]
    treated_mask = values["intervention_union_mask"][answer] - values["intervention_union_mask"][foil]
    base_drop = base_margin_full - base_mask
    treated_drop = treated_full - treated_mask
    return answer, foil, base_drop, treated_drop, treated_drop - base_drop


def _draw(rng: random.Random) -> dict[str, list[float]]:
    return {name: _log_softmax([rng.uniform(-12.0, 12.0) for _ in range(5)])
            for name in NAMES}


def audit(draws: int = DEFAULT_DRAWS, seed: int = SEED) -> dict:
    if type(draws) is not int or draws < 1:
        raise ValueError("draws_must_be_positive_integer")
    rng = random.Random(seed)
    maximum_error = 0.0
    treatment_rank_changes = 0
    for _ in range(draws):
        values = _draw(rng)
        answer, foil, base_drop, treated_drop, delta = _reference(values)
        result = contrast_from_choice_log_probabilities(values)
        maximum_error = max(maximum_error,
            abs(result["baseline_union_dependence"] - base_drop),
            abs(result["intervention_union_dependence"] - treated_drop),
            abs(result["delta_union_dependence"] - delta))
        if (result["answer_index"], result["foil_index"]) != (answer, foil):
            raise AssertionError("baseline_answer_foil_disagreement")

        # Metamorphic check: a common per-condition shift cannot change a
        # within-condition answer/foil margin or the contrast.
        shifted = {name: [value - 2.0 for value in row]
                   for name, row in values.items()}
        shifted_result = contrast_from_choice_log_probabilities(shifted)
        if not math.isclose(shifted_result["delta_union_dependence"],
                            result["delta_union_dependence"],
                            rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("common_shift_changed_margin_contrast")

        # A shared change to the frozen answer score in both masked branches
        # must cancel from the treatment-minus-baseline difference-in-drops.
        shared_mask_shift = {name: list(row) for name, row in values.items()}
        shared_mask_shift["baseline_union_mask"][answer] -= 0.25
        shared_mask_shift["intervention_union_mask"][answer] -= 0.25
        shifted_delta = contrast_from_choice_log_probabilities(
            shared_mask_shift)["delta_union_dependence"]
        if not math.isclose(shifted_delta, result["delta_union_dependence"],
                            rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("shared_mask_effect_did_not_cancel")

        # Count whether treatment condition rankings would differ if someone
        # incorrectly recomputed labels after intervention.
        treated_rank = sorted(range(5), key=lambda i: (-values["intervention_full"][i], i))
        if tuple(treated_rank[:2]) != (answer, foil):
            treatment_rank_changes += 1

    if maximum_error > 1e-12:
        raise AssertionError("reference_arithmetic_error_exceeded_tolerance")
    tie_values = {name: [-math.log(5.0)] * 5 for name in NAMES}
    tie_result = contrast_from_choice_log_probabilities(tie_values)
    if (tie_result["answer_index"], tie_result["foil_index"],
            tie_result["delta_union_dependence"]) != (0, 1, 0.0):
        raise AssertionError("alphabetical_tie_break_or_zero_case_failed")
    return {
        "status": "synthetic_arithmetic_properties_passed",
        "seed": seed,
        "synthetic_draws": draws,
        "max_abs_difference_from_independent_reference_nat": maximum_error,
        "draws_where_treatment_top_two_would_change_if_recomputed": treatment_rank_changes,
        "tie_break_and_zero_delta_case": "passed",
        "properties": ["fixed baseline answer/foil", "common-shift margin invariance",
                       "shared mask-answer effect cancellation", "independent scalar formula"],
        "dataset_rows": 0,
        "model_forwards": 0,
        "faithfulness_result": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(audit(args.draws, args.seed), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
