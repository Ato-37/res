#!/usr/bin/env python3
"""Verify locked temporal-order interaction rows without pooling failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

LAYER_KEYS = {str(i): i for i in range(5, 17)}
BRANCHES = {"original_baseline", "permuted_baseline",
            "original_visual_bias", "permuted_visual_bias"}


def _read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _source_hash(relative_path: str) -> str:
    source = Path(__file__).resolve().parents[1] / relative_path
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _finite(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _expected_interaction(branches: dict) -> float:
    if not isinstance(branches, dict) or set(branches) != BRANCHES:
        raise ValueError("receipt_branch_set_invalid")
    values = {name: branch.get("target_logprob_mean")
              for name, branch in branches.items()}
    if not all(_finite(value) for value in values.values()):
        raise ValueError("receipt_branch_score_nonfinite")
    return ((float(values["original_visual_bias"])
             - float(values["permuted_visual_bias"]))
            - (float(values["original_baseline"])
               - float(values["permuted_baseline"])))


def _expected_packet_status(valid_rows: int, requested_rows: int) -> str:
    if not 0 <= valid_rows <= requested_rows:
        raise ValueError("claim_packet_valid_count_out_of_range")
    return "passed" if valid_rows == requested_rows else "partial"


def _finite_cohort_descriptives(values: list[float], *, complete: bool) -> dict:
    if not complete:
        return {"mean": None, "median": None, "range": None}
    if len(values) != 3 or not all(_finite(value) for value in values):
        raise ValueError("complete_cohort_descriptives_require_three_finite_rows")
    return {"mean": sum(values) / 3, "median": float(median(values)),
            "range": [min(values), max(values)]}


def analyze(*, lock_path: Path, smoke_packet_path: Path,
            remaining_packet_path: Path, smoke_pair_filter_path: Path,
            remaining_pair_filter_path: Path,
            receipt_paths: tuple[Path, ...]) -> dict:
    lock, _lock_hash = _read(lock_path)
    packets = []
    packet_hashes = {}
    for name, path in (("smoke", smoke_packet_path), ("remaining", remaining_packet_path)):
        payload, digest = _read(path)
        packets.append((name, payload, path))
        packet_hashes[path.name] = digest
    assignments = lock.get("assignments")
    expected_split = lock.get("dataset_split")
    if (lock.get("status") != "frozen_before_claim_generation_and_scoring"
            or lock.get("effective_videos") != 3
            or expected_split not in ("val", "test")
            or not isinstance(assignments, list) or len(assignments) != 3):
        raise ValueError("cohort_lock_invalid")
    by_media = {row.get("media_sha256"): (i, row.get("qid_sha256"))
                for i, row in enumerate(assignments)}
    if len(by_media) != 3:
        raise ValueError("cohort_lock_duplicate_media")
    valid_rows: dict[int, tuple[str, dict, Path]] = {}
    invalid_rows = []
    seen_rows = set()
    generation_runner = "scripts/generate_nextgqa_claim_review_pilot.py"
    pilot_runner = "scripts/run_nextgqa_gold_interval_margin_pilot.py"
    generation_runner_actual = _source_hash(generation_runner)
    pilot_runner_actual = _source_hash(pilot_runner)
    for label, packet, path, filter_path in (
            ("smoke", packets[0][1], packets[0][2], smoke_pair_filter_path),
            ("remaining", packets[1][1], packets[1][2], remaining_pair_filter_path)):
        expected_count = 1 if label == "smoke" else 2
        pair_filter, filter_hash = _read(filter_path)
        pair_values = pair_filter.get("pair_sha256")
        if (pair_filter.get("schema_version") != 1
                or not isinstance(pair_values, list)
                or len(pair_values) != expected_count
                or len(set(pair_values)) != expected_count
                or packet.get("candidate_pair_filter_sha256") != filter_hash):
            raise ValueError("candidate_pair_filter_binding_invalid")
        if (packet.get("runner_sha256") != generation_runner_actual
                or packet.get("pilot_runner_sha256") != pilot_runner_actual):
            raise ValueError("claim_generation_source_hash_mismatch")
        pair_filter_digest = hashlib.sha256(json.dumps(sorted(pair_values),
            ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        # The generator's helper hashes a canonical JSON list identically.
        if packet.get("candidate_pair_filter_digest") != pair_filter_digest:
            raise ValueError("candidate_pair_filter_digest_mismatch")
        records = packet.get("records")
        if (packet.get("dataset_split") != expected_split
                or packet.get("status") not in ("passed", "partial")
                or packet.get("requested_videos") != expected_count
                or packet.get("effective_videos") != expected_count
                or not isinstance(records, list) or len(records) != expected_count):
            raise ValueError("claim_generation_packet_contract_invalid")
        packet_valid = 0
        for record in records:
            media = record.get("video_sha256")
            if media not in by_media:
                raise ValueError("claim_row_not_in_frozen_lock")
            row_index, qid_hash = by_media[media]
            if row_index in seen_rows:
                raise ValueError("duplicate_locked_packet_row")
            seen_rows.add(row_index)
            question_id = record.get("dataset_question_id")
            if (not isinstance(question_id, (str, int))
                    or hashlib.sha256(str(question_id).encode("utf-8")).hexdigest() != qid_hash):
                raise ValueError("claim_qid_not_in_frozen_lock")
            parsed = record.get("generated_parsed")
            if not isinstance(parsed, dict):
                raise ValueError("claim_parse_record_missing")
            candidate_pair = record.get("candidate_pair_sha256")
            if candidate_pair not in pair_values:
                raise ValueError("claim_pair_not_in_locked_pair_filter")
            if parsed.get("valid") is True and isinstance(parsed.get("claim"), str) and parsed["claim"].strip():
                packet_valid += 1
                if row_index in valid_rows:
                    raise ValueError("duplicate_valid_claim_row")
                valid_rows[row_index] = (path.name, record, path)
            else:
                invalid_rows.append({"locked_row": row_index,
                                     "failure_code": parsed.get("claim_text_quality_failure")})
        if packet.get("fully_structured_claim_records") != packet_valid:
            raise ValueError("generation_packet_valid_count_mismatch")
        if packet.get("status") != _expected_packet_status(packet_valid, expected_count):
            raise ValueError("generation_packet_status_count_mismatch")
    if seen_rows != {0, 1, 2}:
        raise ValueError("claim_packets_do_not_cover_frozen_lock")
    receipts_by_row = {}
    row_outputs = []
    for receipt_path in receipt_paths:
        receipt, receipt_hash = _read(receipt_path)
        media = receipt.get("source_video_sha256")
        if media not in by_media:
            raise ValueError("receipt_not_in_frozen_lock")
        row_index = by_media[media][0]
        if row_index in receipts_by_row:
            raise ValueError("duplicate_receipt_for_locked_row")
        if row_index not in valid_rows:
            raise ValueError("receipt_for_invalid_claim_row")
        packet_name, record, packet_path = valid_rows[row_index]
        packet, packet_hash = _read(packet_path)
        if (receipt.get("status") != "completed_one_row_exploratory_sensitivity_smoke"
                or (receipt.get("requested_forwards"), receipt.get("attempted_forwards"),
                    receipt.get("completed_forwards")) != (4, 4, 4)
                or receipt.get("direct_faithfulness") is not False
                or receipt.get("faithfulness_result") is not False
                or receipt.get("claim_truth_reviewed") is not False
                or receipt.get("human_review", {}).get("review_actually_performed") is not False
                or receipt.get("private_input_packet_name") != packet_path.name
                or receipt.get("private_input_packet_sha256") != packet_hash
                or receipt.get("partial_input_packet_allowed", False) != (packet.get("status") == "partial")):
            raise ValueError("scoring_receipt_provenance_invalid")
        # A receipt is only attributable to this implementation if the scorer
        # source recorded at run time is still available byte-for-byte. Do not
        # silently credit results from an unarchived runner revision.
        source_map = receipt.get("source_sha256")
        runner_rel = "scripts/run_saved_claim_frame_permutation_attention_smoke.py"
        runner_path = Path(__file__).resolve().parents[1] / runner_rel
        runner_actual = hashlib.sha256(runner_path.read_bytes()).hexdigest()
        runner_recorded = source_map.get(runner_rel) if isinstance(source_map, dict) else None
        source_verified = runner_recorded == runner_actual
        expected = _expected_interaction(receipt.get("branch_target_scores"))
        observed = receipt.get("primary_descriptive_interaction", {}).get(
            "bias_by_frame_order_interaction_nats_per_token")
        if not _finite(observed) or not math.isclose(expected, float(observed), rel_tol=0, abs_tol=1e-12):
            raise ValueError("stored_interaction_arithmetic_mismatch")
        hooks_ok = True
        for branch, values in receipt["branch_target_scores"].items():
            counts = values.get("intervention_hook_invocations_by_layer_1_based")
            expected_count = 0 if branch.endswith("baseline") else 1
            if not isinstance(counts, dict) or set(counts) != set(LAYER_KEYS):
                hooks_ok = False
            elif any(counts[layer] != expected_count for layer in LAYER_KEYS):
                hooks_ok = False
        if not hooks_ok:
            raise ValueError("attention_hook_count_mismatch")
        if source_verified:
            receipts_by_row[row_index] = receipt_hash
        row_outputs.append({"locked_row": row_index,
                            "interaction_nats_per_token": float(observed),
                            "completed_forwards": 4,
                            "runner_source_verified": source_verified,
                            "runner_source_sha256_recorded": runner_recorded,
                            "runner_source_sha256_available": runner_actual,
                            "receipt_sha256": receipt_hash})
    if set(receipts_by_row) != set(valid_rows):
        # Keep the evidence visible but do not count provenance-unverified rows
        # as analyzed observations.
        provenance_failed_rows = sorted(set(valid_rows) - set(receipts_by_row))
    else:
        provenance_failed_rows = []
    scored = len(receipts_by_row)
    verified_rows = [row for row in row_outputs if row["runner_source_verified"]]
    unverified_rows = [row for row in row_outputs if not row["runner_source_verified"]]
    complete = scored == 3 and not invalid_rows and not provenance_failed_rows
    descriptives = _finite_cohort_descriptives(
        [row["interaction_nats_per_token"] for row in verified_rows], complete=complete)
    return {
        "status": "incomplete_cohort_no_effect_estimate" if scored != 3 or invalid_rows or provenance_failed_rows else "complete_exploratory_finite_cohort",
        "requested_locked_rows": 3,
        "claim_generation_valid_rows": len(valid_rows),
        "claim_generation_failed_rows": len(invalid_rows),
        "scored_rows": scored,
        "completed_forwards": 4 * scored,
        "requested_forwards_if_complete": 12,
        "failed_claim_rows": invalid_rows,
        "provenance_unverified_rows": provenance_failed_rows,
        "verified_scored_rows_individual_descriptives_only": sorted(verified_rows, key=lambda row: row["locked_row"]),
        "provenance_unverified_receipt_diagnostics": sorted(unverified_rows, key=lambda row: row["locked_row"]),
        "cohort_mean": descriptives["mean"],
        "cohort_median": descriptives["median"],
        "cohort_range": descriptives["range"],
        "confidence_interval": None,
        "hypothesis_test": None,
        "direct_faithfulness": False,
        "human_review_performed": False,
        "private_generation_packet_sha256": packet_hashes,
        "generation_runner_sha256_verified": generation_runner_actual,
        "pilot_runner_sha256_verified": pilot_runner_actual,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--smoke-packet", type=Path, required=True)
    parser.add_argument("--remaining-packet", type=Path, required=True)
    parser.add_argument("--smoke-pair-filter", type=Path, required=True)
    parser.add_argument("--remaining-pair-filter", type=Path, required=True)
    parser.add_argument("receipts", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(lock_path=args.lock, smoke_packet_path=args.smoke_packet,
        remaining_packet_path=args.remaining_packet,
        smoke_pair_filter_path=args.smoke_pair_filter,
        remaining_pair_filter_path=args.remaining_pair_filter,
        receipt_paths=tuple(args.receipts)), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
