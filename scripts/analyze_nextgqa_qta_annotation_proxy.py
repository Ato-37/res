#!/usr/bin/env python3
"""Recompute aggregate checks for private QTA annotation-proxy receipts.

Only aggregate values are printed. The input receipt contains protected
record-level benchmark outputs and must remain local.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import stat
import sys


EXPECTED_STATUS = "completed_exploratory_single_anchor_finite_cohort"
EXPECTED_SCOPE = "NExT-GQA annotation-interval dependence proxy; not explanation faithfulness"
BRANCHES = (
    "baseline_control_ablation", "baseline_evidence_ablation", "baseline_full",
    "intervention_control_ablation", "intervention_evidence_ablation",
    "intervention_full",
)
MARGIN_KEYS = {
    "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
    "intervention_full", "intervention_evidence_ablation",
    "intervention_control_ablation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_receipt(receipt: dict, repo: Path) -> dict:
    """Fail closed on receipt identity, denominator, geometry, or source drift."""
    if (not isinstance(receipt, dict)
            or receipt.get("status") != EXPECTED_STATUS
            or receipt.get("scope") != EXPECTED_SCOPE
            or receipt.get("evidence_selection_policy") != "single_overlap_anchor"
            or receipt.get("inference") != "descriptive_only_no_p_value_or_population_CI"):
        raise ValueError("receipt_identity_or_scope_invalid")
    n = receipt.get("requested_n")
    if (type(n) is not int or n <= 1
            or receipt.get("effective_n") != n
            or receipt.get("failures") != 0
            or receipt.get("preflight_requested") != n
            or receipt.get("preflight_attempted") != n
            or receipt.get("preflight_completed") != n
            or receipt.get("requested_forwards") != 6 * n
            or receipt.get("attempted_forwards") != 6 * n
            or receipt.get("completed_forwards") != 6 * n):
        raise ValueError("all_or_none_denominator_or_forward_count_invalid")
    branches = receipt.get("branch_counts")
    if (not isinstance(branches, dict) or set(branches) != set(BRANCHES)
            or any(not isinstance(branches[name], dict)
                   or any(branches[name].get(key) != n
                          for key in ("requested", "attempted", "completed", "effective"))
                   for name in BRANCHES)):
        raise ValueError("branch_counts_invalid")
    records = receipt.get("records")
    if not isinstance(records, list) or len(records) != n:
        raise ValueError("record_denominator_invalid")
    deltas, correct, overlaps, positions, covered = [], 0, [], [], []
    for ordinal, row in enumerate(records, start=1):
        if not isinstance(row, dict) or row.get("ordinal") != ordinal:
            raise ValueError("record_order_or_schema_invalid")
        if any(key in row for key in ("question", "answer", "video", "logits", "raw_text")):
            raise ValueError("unexpected_sensitive_record_field")
        total = row.get("temporal_group_count")
        covered_indices = row.get("annotation_covered_group_indices")
        evidence = row.get("evidence_group_indices")
        control = row.get("control_group_indices")
        anchor = row.get("representative_annotation_group_index")
        if (type(total) is not int or total < 2
                or not isinstance(covered_indices, list) or not covered_indices
                or row.get("annotation_covered_group_count") != len(covered_indices)
                or not isinstance(evidence, list) or len(evidence) != 1
                or not isinstance(control, list) or len(control) != 1
                or type(anchor) is not int or evidence != [anchor]
                or anchor not in covered_indices or control[0] in covered_indices
                or any(type(index) is not int or not 0 <= index < total
                       for index in covered_indices + evidence + control)
                or row.get("evidence_group_count") != 1
                or row.get("control_group_count") != 1):
            raise ValueError("annotation_anchor_geometry_invalid")
        delta = row.get("delta_f")
        if type(delta) not in (int, float) or not math.isfinite(delta):
            raise ValueError("nonfinite_delta_f")
        answer_index, foil_index = row.get("answer_index"), row.get("foil_index")
        margins = row.get("margin_by_condition")
        if (type(answer_index) is not int or type(foil_index) is not int
                or not 0 <= answer_index < 5 or not 0 <= foil_index < 5
                or answer_index == foil_index or not isinstance(margins, dict)
                or set(margins) != MARGIN_KEYS
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in margins.values())):
            raise ValueError("fixed_prediction_margin_contract_invalid")
        baseline_f = (margins["baseline_control_ablation"]
                      - margins["baseline_evidence_ablation"])
        intervention_f = (margins["intervention_control_ablation"]
                          - margins["intervention_evidence_ablation"])
        if not math.isclose(delta, intervention_f - baseline_f, rel_tol=0, abs_tol=1e-12):
            raise ValueError("delta_f_margin_recomputation_mismatch")
        if type(row.get("baseline_full_correct")) is not int or row["baseline_full_correct"] not in (0, 1):
            raise ValueError("baseline_correctness_invalid")
        overlap = row.get("representative_annotation_overlap_fraction")
        position = row.get("representative_group_position_fraction")
        try:
            overlap_value = float(Fraction(overlap))
        except (TypeError, ValueError, ZeroDivisionError):
            raise ValueError("anchor_diagnostic_invalid") from None
        if (not math.isfinite(overlap_value)
                or not 0 <= overlap_value <= 1 or type(position) not in (int, float)
                or not math.isfinite(position) or not 0 <= position <= 1):
            raise ValueError("anchor_diagnostic_invalid")
        deltas.append(float(delta))
        correct += row["baseline_full_correct"]
        overlaps.append(overlap_value)
        positions.append(float(position))
        covered.append(len(covered_indices))

    sources = receipt.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source_manifest_missing")
    verified, post_run_document_updates = 0, 0
    for relative, expected in sources.items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo.resolve()) or not path.is_file():
            raise ValueError("source_manifest_path_invalid")
        if sha256_file(path) == expected:
            verified += 1
        elif relative.startswith("experiments/"):
            # Experiment logs are append-only and are updated after the run;
            # their original run-time hash remains in the private receipt.
            post_run_document_updates += 1
        else:
            raise ValueError("source_manifest_hash_mismatch")

    mean = statistics.fmean(deltas)
    sample_sd = statistics.stdev(deltas)
    summary = {
        "status": EXPECTED_STATUS,
        "scope": EXPECTED_SCOPE,
        "requested_n": n,
        "effective_n": n,
        "failures": 0,
        "forwards_completed": 6 * n,
        "baseline_full_accuracy": {"correct": correct, "n": n, "accuracy": correct / n},
        "delta_f_descriptive_nat": {
            "mean": mean, "sample_sd": sample_sd,
            "min": min(deltas), "max": max(deltas),
            "positive": sum(value > 0 for value in deltas),
            "negative": sum(value < 0 for value in deltas),
            "zero": sum(value == 0 for value in deltas),
            "inference": "finite_cohort_descriptive_only_no_p_value_or_population_CI",
        },
        "anchor_diagnostics": {
            "mean_overlap_fraction": statistics.fmean(overlaps),
            "mean_temporal_position_fraction": statistics.fmean(positions),
            "mean_covered_group_count": statistics.fmean(covered),
        },
        "source_files_verified_current": verified,
        "post_run_experiment_documents_updated": post_run_document_updates,
    }
    stored = receipt.get("finite_cohort_delta_f", {})
    if (stored.get("n") != n
            or not math.isclose(stored.get("mean", math.nan), mean, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(stored.get("sample_sd", math.nan), sample_sd, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(stored.get("min", math.nan), min(deltas), rel_tol=0, abs_tol=1e-12)
            or not math.isclose(stored.get("max", math.nan), max(deltas), rel_tol=0, abs_tol=1e-12)):
        raise ValueError("stored_delta_f_summary_mismatch")
    stored_accuracy = receipt.get("finite_cohort_baseline_full_restricted_choice_accuracy", {})
    if (stored_accuracy.get("n") != n or stored_accuracy.get("correct") != correct
            or not math.isclose(stored_accuracy.get("accuracy", math.nan), correct / n,
                                rel_tol=0, abs_tol=1e-12)):
        raise ValueError("stored_baseline_accuracy_mismatch")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="private local receipt JSON")
    args = parser.parse_args(argv)
    try:
        mode = stat.S_IMODE(args.receipt.stat().st_mode)
        if mode & 0o077:
            raise ValueError("receipt_permissions_must_be_private")
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        summary = analyze_receipt(receipt, Path(__file__).resolve().parents[1])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"analysis_failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
