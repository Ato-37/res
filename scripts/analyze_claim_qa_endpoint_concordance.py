#!/usr/bin/env python3
"""Compare QA-margin and saved-claim likelihood intervention endpoints.

Exploratory paired analysis only. Both receipts and the manifest are local,
protected research artifacts; output contains derived values and hashes only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from private_output_paths import validate_private_output
from run_machine_localized_claim_mask_pilot import _read_selected_window_records
from run_saved_claim_frame_permutation_attention_smoke import _sha256_file


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("correlation_input_size_invalid")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        raise ValueError("correlation_constant_input")
    return sum(x * y for x, y in zip(dx, dy, strict=True)) / denom


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for position in order[start:end]:
            ranks[position] = average_rank
        start = end
    return ranks


def _within_cohort_pearson(cohorts: list[dict]) -> float:
    """Correlate row-level deviations after removing each cohort's mean."""
    xs, ys = [], []
    for cohort in cohorts:
        rows = cohort["rows"]
        qa = [row["qa_margin_D_nats"] for row in rows]
        claim = [row["claim_text_D_nats_per_token"] for row in rows]
        qa_mean, claim_mean = statistics.mean(qa), statistics.mean(claim)
        xs.extend(value - qa_mean for value in qa)
        ys.extend(value - claim_mean for value in claim)
    return _pearson(xs, ys)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hash_mismatches(repo: Path, receipt: dict) -> list[str]:
    mismatches = []
    for name, recorded in receipt.get("source_sha256", {}).items():
        candidates = (repo / name, repo / "scripts" / name)
        source = next((path for path in candidates if path.is_file()), None)
        if source is None or _sha256(source) != recorded:
            mismatches.append(name)
    return sorted(mismatches)


def _validate_unique_cohort_labels(cohorts: list) -> None:
    """Reject duplicate labels so summaries cannot silently overwrite cohorts."""
    labels = []
    for spec in cohorts:
        label = spec.get("label") if isinstance(spec, dict) else None
        if not isinstance(label, str) or not label:
            raise ValueError("cohort_label_invalid")
        labels.append(label)
    if len(labels) != len(set(labels)):
        raise ValueError("cohort_labels_not_unique")


def _fixed_full_prediction_dependence(choice_logprobs: dict) -> dict:
    """Score selected-vs-control margin for the frozen full-video top-2 labels."""
    evidence_name = "3b_likelihood_selected_window_black"
    control_name = "temporal_control_black"
    if not isinstance(choice_logprobs, dict) or set(choice_logprobs) != {
            "full", evidence_name, control_name}:
        raise ValueError("predicted_margin_conditions_invalid")
    scores = {}
    for condition, values in choice_logprobs.items():
        if (not isinstance(values, list) or len(values) != 5
                or any(isinstance(value, bool) for value in values)):
            raise ValueError("predicted_margin_choice_vector_invalid")
        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError):
            raise ValueError("predicted_margin_choice_vector_invalid") from None
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("predicted_margin_choice_vector_nonfinite")
        scores[condition] = vector
    # Python's first-maximum rule implements frozen A–E order for ties.
    predicted = max(range(5), key=lambda index: scores["full"][index])
    foil = max((index for index in range(5) if index != predicted),
               key=lambda index: scores["full"][index])
    evidence_margin = scores[evidence_name][predicted] - scores[evidence_name][foil]
    control_margin = scores[control_name][predicted] - scores[control_name][foil]
    predicted_logprob_shift = (scores[control_name][predicted]
                               - scores[evidence_name][predicted])
    foil_logprob_shift = (scores[control_name][foil]
                          - scores[evidence_name][foil])
    return {
        "frozen_full_prediction_index": predicted,
        "frozen_full_foil_index": foil,
        "evidence_mask_margin_nats": evidence_margin,
        "control_mask_margin_nats": control_margin,
        "predicted_label_control_minus_evidence_logprob_nats": predicted_logprob_shift,
        "foil_label_control_minus_evidence_logprob_nats": foil_logprob_shift,
        "control_minus_evidence_predicted_margin_nats": predicted_logprob_shift - foil_logprob_shift,
    }


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("json_object_required")
    return value


def analyze(repo: Path, manifest_path: Path) -> dict:
    manifest = _load_json(manifest_path)
    cohorts = manifest.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        raise ValueError("cohort_manifest_invalid")
    _validate_unique_cohort_labels(cohorts)
    all_qa: list[float] = []
    all_claim: list[float] = []
    cohort_results = []
    seen_tokens: set[str] = set()
    for spec in cohorts:
        label = spec.get("label") if isinstance(spec, dict) else None
        try:
            if not isinstance(label, str) or not label:
                raise ValueError("cohort_label_invalid")
            relpaths = [spec[key] for key in ("selection", "lock", "pairs", "direct", "qa")]
            packets = spec.get("claim_packets")
            if (not all(isinstance(x, str) for x in relpaths)
                    or not isinstance(packets, list) or not packets
                    or not all(isinstance(x, str) for x in packets)):
                raise ValueError("cohort_paths_invalid")
            selection, lock, pairs, direct_path, qa_path = [repo / x for x in relpaths]
            packet_paths = [repo / x for x in packets]
            rows, actual_hashes = _read_selected_window_records(
                repo, spec["selection"], packets, spec["lock"], spec["pairs"])
            direct, qa = _load_json(direct_path), _load_json(qa_path)
            if (direct.get("status") != "completed_exploratory_claim_text_likelihood"
                    or direct.get("effective_records") != 3
                    or direct.get("completed_forwards") != 9
                    or qa.get("status") != "completed"
                    or qa.get("effective_records") != 3
                    or qa.get("completed_forwards") != 9):
                raise ValueError("cohort_receipt_incomplete")
            if direct.get("inputs_sha256") != actual_hashes:
                raise ValueError("direct_input_hash_map_mismatch")
            selector_hash = actual_hashes[spec["selection"]]
            if qa.get("inputs_sha256", {}).get(spec["selection"]) != selector_hash:
                raise ValueError("qa_selector_hash_mismatch")
            for packet in packets:
                if qa.get("inputs_sha256", {}).get(packet) != actual_hashes[packet]:
                    raise ValueError("qa_claim_packet_hash_mismatch")
            direct_rows, qa_rows = direct.get("records"), qa.get("records")
            if (not isinstance(direct_rows, list) or not isinstance(qa_rows, list)
                    or len(direct_rows) != 3 or len(qa_rows) != 3):
                raise ValueError("cohort_rows_incomplete")
            if len(rows) != 3:
                raise ValueError("selected_claim_rows_incomplete")
            qa_values, claim_values, prediction_values, row_diagnostics = [], [], [], []
            for position, (source, direct_row, qa_row) in enumerate(
                    zip(rows, direct_rows, qa_rows, strict=True), 1):
                token = hashlib.sha256((source["video_sha256"]
                    + str(source["dataset_question_id"])).encode()).hexdigest()
                if (direct_row.get("record_position") != position
                        or direct_row.get("row_token_sha256") != token
                        or qa_row.get("record_position") != position
                        or token in seen_tokens):
                    raise ValueError("paired_row_binding_mismatch")
                seen_tokens.add(token)
                qa_d = float(qa_row["d_control_minus_3b_likelihood_window_nats"])
                claim_d = float(direct_row["control_minus_selected_mask_nats_per_token"])
                if not math.isfinite(qa_d) or not math.isfinite(claim_d):
                    raise ValueError("paired_endpoint_nonfinite")
                prediction_diagnostic = _fixed_full_prediction_dependence(
                    qa_row.get("choice_log_probabilities_nats"))
                prediction_d = prediction_diagnostic[
                    "control_minus_evidence_predicted_margin_nats"]
                qa_values.append(qa_d)
                claim_values.append(claim_d)
                prediction_values.append(prediction_d)
                row_diagnostics.append({
                    "record_position": position,
                    "row_token_sha256": token,
                    "qa_margin_D_nats": qa_d,
                    "claim_text_D_nats_per_token": claim_d,
                    **prediction_diagnostic,
                    "signs_agree": (qa_d > 0) == (claim_d > 0),
                })
            all_qa.extend(qa_values)
            all_claim.extend(claim_values)
            cohort_results.append({
                "label": label,
                "direct_receipt_sha256": _sha256(direct_path),
                "qa_receipt_sha256": _sha256(qa_path),
                "direct_source_sha256": direct.get("source_sha256"),
                "qa_source_sha256": qa.get("source_sha256"),
                "direct_source_hash_mismatches": _source_hash_mismatches(repo, direct),
                "qa_source_hash_mismatches": _source_hash_mismatches(repo, qa),
                "selection_sha256": selector_hash,
                "n": 3,
                "mean_qa_margin_D_nats": statistics.mean(qa_values),
                "mean_claim_text_D_nats_per_token": statistics.mean(claim_values),
                "mean_frozen_prediction_margin_D_nats": statistics.mean(prediction_values),
                "rows": row_diagnostics,
            })
        except Exception as exc:
            raise ValueError(f"cohort_{label or 'unknown'}_invalid:{exc}") from exc
    if len(seen_tokens) != len(all_qa):
        raise ValueError("duplicate_paired_rows_across_cohorts")
    qa_ranks, claim_ranks = _ranks(all_qa), _ranks(all_claim)
    leave_one_cohort_out = []
    for omitted in cohort_results:
        retained = [row for cohort in cohort_results if cohort["label"] != omitted["label"]
                    for row in cohort["rows"]]
        retained_cohorts = [cohort for cohort in cohort_results
                            if cohort["label"] != omitted["label"]]
        leave_one_cohort_out.append({
            "omitted_cohort": omitted["label"],
            "pearson_r": _pearson(
                [row["qa_margin_D_nats"] for row in retained],
                [row["claim_text_D_nats_per_token"] for row in retained]),
            "within_cohort_centered_pearson_r": _within_cohort_pearson(retained_cohorts),
        })
    return {
        "schema_version": 1,
        "status": "completed_exploratory_endpoint_concordance",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "describe agreement of QA endpoints and post-hoc frozen-full-prediction dependence under the same machine-selected-window vs temporal-control masks",
        "direct_faithfulness": False,
        "human_review": {"review_actually_performed": False},
        "rights_status": "unverified; local-only",
        "cohort_count": len(cohort_results),
        "row_count": len(all_qa),
        "cohorts": cohort_results,
        "row_level_summary": {
            "sign_agreement_count": sum((x > 0) == (y > 0)
                for x, y in zip(all_qa, all_claim, strict=True)),
            "pearson_r": _pearson(all_qa, all_claim),
            "spearman_rho": _pearson(qa_ranks, claim_ranks),
            "within_cohort_centered_pearson_r": _within_cohort_pearson(cohort_results),
            "leave_one_cohort_out_sensitivity": leave_one_cohort_out,
            "qa_margin_D_range_nats": [min(all_qa), max(all_qa)],
            "claim_text_D_range_nats_per_token": [min(all_claim), max(all_claim)],
            "frozen_prediction_margin_D_mean_nats": statistics.mean(
                row["control_minus_evidence_predicted_margin_nats"]
                for cohort in cohort_results for row in cohort["rows"]),
            "frozen_prediction_margin_D_median_nats": statistics.median(
                row["control_minus_evidence_predicted_margin_nats"]
                for cohort in cohort_results for row in cohort["rows"]),
            "frozen_prediction_margin_D_range_nats": [
                min(row["control_minus_evidence_predicted_margin_nats"]
                    for cohort in cohort_results for row in cohort["rows"]),
                max(row["control_minus_evidence_predicted_margin_nats"]
                    for cohort in cohort_results for row in cohort["rows"])],
            "frozen_prediction_margin_D_positive_count": sum(
                row["control_minus_evidence_predicted_margin_nats"] > 0
                for cohort in cohort_results for row in cohort["rows"]),
            "predicted_label_control_minus_evidence_mean_nats": statistics.mean(
                row["predicted_label_control_minus_evidence_logprob_nats"]
                for cohort in cohort_results for row in cohort["rows"]),
            "foil_label_control_minus_evidence_mean_nats": statistics.mean(
                row["foil_label_control_minus_evidence_logprob_nats"]
                for cohort in cohort_results for row in cohort["rows"]),
        },
        "inferential_status": "post-result sequential exploratory test-split rows; descriptive correlations only; no p-value, CI, or population inference",
        "predicted_answer_endpoint_status": "post-hoc secondary diagnostic; prediction and foil are fixed by full-video A-E log probabilities within each row; no intervention benefit or semantic evidence validation claim",
        "limitations": [
            "six small sequential cohorts were selected/extended after earlier outcomes and are not independent confirmation",
            "QA answer-margin and claim-token likelihood are distinct endpoints with different scales",
            "claims are generated and unreviewed; this analysis does not establish semantic evidence support or truth",
            "Qwen-family localization/scoring and four-window maximum selection may induce dependence and winner's-curse bias",
            "black masking may create out-of-distribution inputs",
            "the frozen-prediction endpoint was derived post hoc from saved five-choice logits; the selected temporal window is machine-localized and not human-validated",
            "baseline predicted labels and best foils may be incorrect; margin sensitivity is not calibrated correctness or proof of faithful reasoning",
        ],
        "manifest_sha256": _sha256(manifest_path),
        "source_sha256": _sha256(Path(__file__).resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = validate_private_output(args.output, repo)
    if output.parent != repo / "data" / "private_nextgqa" or output.suffix != ".json":
        raise ValueError("private_output_must_be_json_child_of_private_data_root")
    result = analyze(repo, repo / args.manifest)
    result["source_sha256"] = _sha256(Path(__file__).resolve())
    raw = (json.dumps(result, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
        getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    summary = result["row_level_summary"]
    print(json.dumps({"status": result["status"], "cohorts": result["cohort_count"],
        "rows": result["row_count"], **summary, "output_name": output.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
