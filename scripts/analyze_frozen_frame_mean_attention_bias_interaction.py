#!/usr/bin/env python3
"""Validate and summarize the local, frozen frame-mean interaction cohort.

Output is hash/count/effect metadata only. Protected packet contents and media
remain local and are never printed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_frozen_frame_mean_attention_bias_interaction import (
    DIRECT_FAITHFULNESS, LOCK_PURPOSE, PACKET_PURPOSE, PROMPT_VERSION,
    compute_interaction, read_private_json, sha256_file,
    validate_frozen_membership, validate_packet_filter_binding,
    validate_packet_payload,
)


def summarize(rows: list[dict], requested: int) -> dict:
    if type(requested) is not int or requested < 1 or len(rows) != requested:
        raise ValueError("cohort_row_count_invalid")
    ordinals = [row.get("row_ordinal") for row in rows]
    if sorted(ordinals) != list(range(requested)):
        raise ValueError("cohort_ordinals_invalid")
    scored = [row for row in rows if row.get("score_status") == "completed"]
    failed = [row for row in rows if row.get("score_status") == "failed"]
    if len(scored) + len(failed) != requested:
        raise ValueError("cohort_status_count_invalid")
    # Deliberately do not calculate a mean over survivors when any locked row failed.
    return {
        "requested_rows": requested,
        "effective_score_rows": len(scored),
        "failed_rows": len(failed),
        "full_cohort_estimate_available": len(failed) == 0,
        "direct_faithfulness": False,
        "rows": sorted(rows, key=lambda row: row["row_ordinal"]),
    }


def analyze(repo: Path, args) -> dict:
    lock, lock_sha, _ = read_private_json(repo, args.lock)
    full_filter, full_filter_sha, _ = read_private_json(repo, args.pair_filter)
    if (lock.get("purpose") != LOCK_PURPOSE
            or lock.get("status") != "frozen_before_claim_generation_and_scoring"
            or lock.get("dataset_split") != "test"):
        raise ValueError("lock_protocol_invalid")
    for name, expected in lock.get("source_sha256", {}).items():
        if sha256_file(repo / name) != expected:
            raise ValueError("lock_source_hash_mismatch")

    packet_specs = [
        (0, args.smoke_packet, args.smoke_pair_filter, 0),
        (1, args.remaining_packet, args.remaining_pair_filter, 0),
        (2, args.remaining_packet, args.remaining_pair_filter, 1),
    ]
    receipt_paths = {0: args.row0_receipt, 1: args.row1_receipt, 2: args.row2_receipt}
    rows = []
    for ordinal, packet_rel, packet_filter_rel, record_index in packet_specs:
        packet, packet_sha, _ = read_private_json(repo, packet_rel)
        packet_filter, packet_filter_sha, _ = read_private_json(repo, packet_filter_rel)
        row = validate_packet_payload(packet, record_index)
        filter_binding = validate_packet_filter_binding(packet, packet_filter,
                                                        packet_filter_sha, row)
        binding = validate_frozen_membership(lock, lock_sha, full_filter,
            full_filter_sha, row, ordinal, record_index)
        if (packet.get("purpose") != PACKET_PURPOSE
                or packet.get("generation_prompt_version") != PROMPT_VERSION
                or packet.get("dataset_split") != "test"):
            raise ValueError("packet_protocol_invalid")
        claim_ok = (isinstance(row.get("generated_parsed"), dict)
                    and row["generated_parsed"].get("valid") is True
                    and isinstance(row["generated_parsed"].get("claim"), str)
                    and bool(row["generated_parsed"]["claim"].strip()))
        receipt, receipt_sha, _ = read_private_json(repo, receipt_paths[ordinal])
        common = {
            "row_ordinal": ordinal,
            "score_receipt_sha256": receipt_sha,
            "packet_sha256": packet_sha,
            "packet_pair_filter_sha256": packet_filter_sha,
            "claim_valid": claim_ok,
        }
        if (receipt.get("row_ordinal") != ordinal
                or receipt.get("source_video_sha256") != binding["source_video_sha256"]
                or receipt.get("candidate_pair_sha256") != binding["candidate_pair_sha256"]
                or receipt.get("dataset_question_id_sha256") != binding["dataset_question_id_sha256"]
                or receipt.get("cohort_lock_sha256") != lock_sha
                or receipt.get("selected_pair_filter_sha256") != full_filter_sha
                or receipt.get("packet_pair_filter_sha256") != packet_filter_sha
                or receipt.get("private_input_packet_sha256") != packet_sha
                or receipt.get("direct_faithfulness") is not DIRECT_FAITHFULNESS):
            raise ValueError("receipt_row_binding_invalid")
        for name, expected in receipt.get("source_sha256", {}).items():
            if sha256_file(repo / name) != expected:
                raise ValueError("receipt_source_hash_mismatch")
        if not claim_ok:
            if (receipt.get("status") != "failed_frozen_frame_mean_attention_bias_interaction"
                    or receipt.get("stage") != "private_input_validation"
                    or receipt.get("failure_code") != "selected_claim_invalid"
                    or receipt.get("requested_forwards") != 4
                    or receipt.get("attempted_forwards") != 0
                    or receipt.get("completed_forwards") != 0
                    or receipt.get("review_index") != record_index):
                raise ValueError("invalid_claim_failure_receipt_mismatch")
            rows.append({**common, "score_status": "failed",
                          "completed_forwards": 0,
                          "failure_code": "selected_claim_invalid"})
            continue

        if (receipt.get("status") != "completed_one_row_frozen_frame_mean_attention_bias_interaction"
                or receipt.get("requested_forwards") != 4
                or receipt.get("attempted_forwards") != 4
                or receipt.get("completed_forwards") != 4
                or receipt.get("claim_valid") is not True
                or receipt.get("packet_record_index_zero_based") != record_index
                or receipt.get("prompt_version") != PROMPT_VERSION):
            raise ValueError("completed_receipt_contract_invalid")
        scores = receipt.get("scores")
        try:
            for condition in ("original_video", "per_frame_rgb_mean_video"):
                for branch, count in (("baseline", 0), ("biased", 1)):
                    score = scores[branch][condition]
                    value = score["target_logprob_mean"]
                    if (not isinstance(value, (int, float)) or isinstance(value, bool)
                            or not math.isfinite(float(value))
                            or score["target_token_count"] != receipt["target_token_count"]
                            or set(score["intervention_hook_invocations_by_layer_1_based"].values()) != {count}):
                        raise ValueError("score_branch_value_or_hook_invalid")
            estimate = compute_interaction(
                scores["baseline"]["original_video"]["target_logprob_mean"],
                scores["baseline"]["per_frame_rgb_mean_video"]["target_logprob_mean"],
                scores["biased"]["original_video"]["target_logprob_mean"],
                scores["biased"]["per_frame_rgb_mean_video"]["target_logprob_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("score_branch_validation_failed") from exc
        recorded = receipt.get("primary_interaction")
        for key, value in estimate.items():
            if (not isinstance(recorded, dict) or key not in recorded
                    or not math.isclose(float(recorded[key]), value, rel_tol=0, abs_tol=1e-12)):
                raise ValueError("interaction_arithmetic_mismatch")
        rows.append({**common, "score_status": "completed",
            "completed_forwards": 4,
            "interaction_nats_per_token": estimate["bias_by_frame_mean_interaction_nats_per_token"]})
    result = summarize(rows, lock["effective_videos"])
    result.update({"cohort_lock_sha256": lock_sha,
        "selected_pair_filter_sha256": full_filter_sha,
        "cohort_selection_status": "post_result_exploratory_local_history_overlap_checked_not_confirmatory",
        "confidence_interval": None, "p_value": None,
        "multiplicity": "single prespecified descriptive contrast; inferential testing not applicable to this incomplete exploratory cohort",
        "rights_status": "unverified; all protected data and receipts remain local"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--pair-filter", type=Path, required=True)
    parser.add_argument("--smoke-packet", type=Path, required=True)
    parser.add_argument("--smoke-pair-filter", type=Path, required=True)
    parser.add_argument("--remaining-packet", type=Path, required=True)
    parser.add_argument("--remaining-pair-filter", type=Path, required=True)
    parser.add_argument("--row0-receipt", type=Path, required=True)
    parser.add_argument("--row1-receipt", type=Path, required=True)
    parser.add_argument("--row2-receipt", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result = analyze(repo, args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
