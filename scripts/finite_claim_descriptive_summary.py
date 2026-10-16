"""Aggregate a complete three-video claim-linked score as finite description.

This is a post-score summary helper, not a scorer, preflight, or authorization.
It deliberately reports no inferential precision or partial-cohort mean.
"""
from __future__ import annotations

import math
import re


def _margin(scores, gold_index):
    if (not isinstance(scores, list) or len(scores) != 5
            or type(gold_index) is not int or not 0 <= gold_index < 5
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in scores)):
        raise ValueError("choice_score_vector_invalid")
    return float(scores[gold_index]) - max(float(x) for i, x in enumerate(scores)
                                            if i != gold_index)


def summarize_finite_three(records):
    """Return a finite-cohort D3 only when all frozen triplets are complete.

    Input rows have an anonymous fixed ordinal 1..3, a completion state,
    completed forward count, safe failure code (when failed), gold index, and
    the five A-E next-token label log-probabilities for full/claim/control.
    No row-level scores or identifiers are returned.
    """
    base = {
        "purpose": "finite_cohort_claim_linked_answer_margin_description",
        "scope": "exact_frozen_three_video_cohort_only",
        "status": "blocked",
        "n_requested": 3,
        "n_complete_triplets": 0,
        "model_forwards_requested": 9,
        "model_forwards_completed": 0,
        "D3_nat": None,
        "positive_effects": None,
        "negative_effects": None,
        "zero_effects": None,
        "inference": {
            "performed": False,
            "ci95": None,
            "p_value": None,
            "sesoi": None,
            "reason": "descriptive_finite_cohort_without_generalization_inference",
        },
        "execution_authorization": "not_granted",
    }
    if not isinstance(records, list) or len(records) != 3:
        base["failure_codes"] = ["cohort_record_count_invalid"]
        return base
    if any(not isinstance(row, dict) or type(row.get("cohort_ordinal")) is not int
           for row in records):
        base["failure_codes"] = ["cohort_row_invalid"]
        return base
    ordered = sorted(records, key=lambda row: row["cohort_ordinal"])
    if [row["cohort_ordinal"] for row in ordered] != [1, 2, 3]:
        base["failure_codes"] = ["cohort_ordinal_coverage_invalid"]
        return base

    failures = []
    completed = 0
    effects = []
    forwards = 0
    for row in ordered:
        count = row.get("completed_model_forwards")
        if type(count) is not int or not 0 <= count <= 3:
            failures.append("forward_count_invalid")
            continue
        forwards += count
        if row.get("status") != "complete" or count != 3:
            code = row.get("failure_code")
            failures.append(code if isinstance(code, str) and re.fullmatch(r"[a-z0-9_]+", code)
                            else "triplet_incomplete")
            continue
        try:
            gold = row.get("gold_index")
            _margin(row.get("full_scores"), gold)
            claim_margin = _margin(row.get("claim_mask_scores"), gold)
            control_margin = _margin(row.get("control_mask_scores"), gold)
        except (TypeError, ValueError, OverflowError):
            failures.append("choice_scores_invalid")
            continue
        effects.append(control_margin - claim_margin)
        completed += 1

    base["n_complete_triplets"] = completed
    base["model_forwards_completed"] = forwards
    if failures or completed != 3:
        base["status"] = "partial_no_cohort_estimate"
        base["failure_codes"] = sorted(set(failures or ["triplet_incomplete"]))
        return base

    mean = sum(effects) / 3
    base.update(
        status="finite_description_complete",
        D3_nat=mean,
        positive_effects=sum(value > 0 for value in effects),
        negative_effects=sum(value < 0 for value in effects),
        zero_effects=sum(value == 0 for value in effects),
        failure_codes=[],
    )
    return base
