#!/usr/bin/env python3
"""Validate and summarize a local annotation-key-bias receipt without row IDs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import stat
import statistics
from typing import Any

from scipy import stats

PURPOSE = "exploratory_nextgqa_qa_annotation_temporal_key_bias_sensitivity"
CONDITIONS = ("baseline", "annotation_interval_key_bias", "hash_sham_key_bias")
LAYERS = tuple(range(5, 17))


def _read_owner_json(path: Path, private_root: Path) -> dict:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(private_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("receipt_must_be_inside_private_root") from exc
    info = resolved.stat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError("receipt_must_be_owner_only_regular_file")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("receipt_shape_invalid")
    return payload


def _finite_vector(value: Any) -> list[float]:
    if (not isinstance(value, list) or len(value) != 5
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(float(x)) for x in value)):
        raise ValueError("choice_logprob_vector_invalid")
    return [float(x) for x in value]


def summarize_receipt(payload: dict) -> dict:
    if (payload.get("purpose") != PURPOSE or payload.get("mode") != "full"
            or payload.get("status") != "completed"
            or payload.get("dataset_split") not in {"val", "test"}):
        raise ValueError("completed_full_receipt_required")
    requested = payload.get("requested_videos")
    rows = payload.get("records")
    if (type(requested) is not int or requested < 3 or not isinstance(rows, list)
            or len(rows) != requested or payload.get("effective_videos") != requested
            or payload.get("requested_forwards") != requested * 3
            or payload.get("attempted_forwards") != requested * 3
            or payload.get("completed_forwards") != requested * 3
            or len({row.get("row_token_sha256") for row in rows
                    if isinstance(row, dict)}) != requested):
        raise ValueError("receipt_completion_or_unique_row_arithmetic_invalid")

    deltas = {"annotation_minus_sham": [], "annotation_minus_baseline": [],
              "sham_minus_baseline": []}
    correct = {name: 0 for name in CONDITIONS}
    for row in rows:
        if not isinstance(row, dict) or set(row.get("conditions", {})) != set(CONDITIONS):
            raise ValueError("row_or_condition_set_invalid")
        gold, foil = row.get("gold_choice_index"), row.get("baseline_best_foil_index")
        if (type(gold) is not int or not 0 <= gold < 5 or type(foil) is not int
                or not 0 <= foil < 5 or gold == foil):
            raise ValueError("gold_foil_binding_invalid")
        baseline = _finite_vector(row["conditions"]["baseline"].get("choice_logprobs_nats"))
        best_foil = max((i for i in range(5) if i != gold), key=lambda i: baseline[i])
        if best_foil != foil:
            raise ValueError("baseline_best_foil_mismatch")
        margins = {}
        for condition in CONDITIONS:
            branch = row["conditions"][condition]
            scores = _finite_vector(branch.get("choice_logprobs_nats"))
            margin = scores[gold] - scores[foil]
            for field, expected in (
                    ("fixed_gold_minus_baseline_foil_margin_nats", margin),
                    ("gold_choice_logprob_nats", scores[gold]),
                    ("baseline_best_foil_logprob_nats", scores[foil])):
                value = branch.get(field)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(float(value)) or abs(float(value) - expected) > 1e-7):
                    raise ValueError("stored_score_arithmetic_mismatch")
            margins[condition] = margin
            is_correct = max(range(5), key=lambda i: (scores[i], -i)) == gold
            if row.get("gold_prediction_correct_by_condition", {}).get(condition) != is_correct:
                raise ValueError("stored_prediction_correctness_mismatch")
            correct[condition] += int(is_correct)
        expected_deltas = {
            "primary_annotation_minus_sham_nats": margins["annotation_interval_key_bias"] - margins["hash_sham_key_bias"],
            "annotation_minus_baseline_nats": margins["annotation_interval_key_bias"] - margins["baseline"],
            "sham_minus_baseline_nats": margins["hash_sham_key_bias"] - margins["baseline"],
        }
        for field, expected in expected_deltas.items():
            value = row.get(field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or abs(float(value) - expected) > 1e-7):
                raise ValueError("stored_contrast_arithmetic_mismatch")
        deltas["annotation_minus_sham"].append(expected_deltas["primary_annotation_minus_sham_nats"])
        deltas["annotation_minus_baseline"].append(expected_deltas["annotation_minus_baseline_nats"])
        deltas["sham_minus_baseline"].append(expected_deltas["sham_minus_baseline_nats"])
        for field in ("annotation_mask_audit", "sham_mask_audit"):
            audit = row.get(field, {})
            if (audit.get("all_and_only_intended_cells_changed") is not True
                    or audit.get("observed_changed_cells") != audit.get("intended_changed_cells")
                    or type(audit.get("biased_key_count")) is not int
                    or audit["biased_key_count"] <= 0
                    or audit.get("hook_calls_by_layer") != {str(i): 1 for i in LAYERS}):
                raise ValueError("attention_mask_audit_invalid")

    summaries = {}
    for name, values in deltas.items():
        mean = statistics.mean(values)
        half_width = float(stats.t.ppf(0.975, len(values) - 1)
                           * stats.sem(values))
        summaries[name] = {"n": len(values), "mean_nats": statistics.mean(values),
            "median_nats": statistics.median(values), "min_nats": min(values),
            "max_nats": max(values), "positive": sum(value > 1e-7 for value in values),
            "approximately_zero": sum(abs(value) <= 1e-7 for value in values),
            "negative": sum(value < -1e-7 for value in values),
            "descriptive_t_interval_95pct_nats": [mean - half_width, mean + half_width]}
    return {"status": "validated_descriptive_summary", "dataset_split": payload["dataset_split"],
        "requested_and_effective_videos": requested, "completed_forwards": requested * 3,
        "gold_accuracy_count_by_condition": correct, "margin_deltas": summaries,
        "statistical_test": "none",
        "interval_note": "unadjusted one-sample t interval is descriptive only; hash-selected availability cohort",
        "population_inference": "not warranted; selected exploratory cohort"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    private_root = repo / "data/private_nextgqa"
    try:
        print(json.dumps(summarize_receipt(_read_owner_json(args.receipt, private_root)),
                         sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc).split(":", 1)[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
