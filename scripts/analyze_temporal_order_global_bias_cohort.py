#!/usr/bin/env python3
"""Validate and summarize the frozen three-row temporal-order GPU cohort.

Prints derived statistics and provenance checks only; never prints questions,
claims, media paths, IDs, or per-item raw branch logits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import stat

import run_nextgqa_gold_interval_margin_pilot as pilot
import run_saved_claim_frame_permutation_attention_smoke as scorer

DEFAULT_PACKET = Path("data/private_nextgqa/claim_generation_temporal_order_test_exploratory_20261021_05.json")
DEFAULT_LOCK = Path("data/private_nextgqa/temporal_order_claim_cohort_20261021_05.json")
DEFAULT_FILTER = Path("data/private_nextgqa/temporal_order_claim_pair_filter_20261021_05.json")
DEFAULT_RECEIPT_STEM = "data/private_nextgqa/temporal_order_global_bias_cohort_20261021_05_row"
LOCK_SOURCES = {
    "scripts/freeze_temporal_order_claim_cohort.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/build_claim_exposure_union_manifest.py",
}
GENERATOR_SOURCES = {
    "scripts/generate_nextgqa_claim_review_pilot.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
}
RECEIPT_SOURCES = {
    "scripts/run_saved_claim_frame_permutation_attention_smoke.py",
    "scripts/qta_multiquery_video_key_intervention.py",
    "scripts/generate_nextgqa_claim_review_pilot.py",
    "scripts/claim_mask_video_decoder.py",
}


def _private_bytes(root: Path, relative: Path) -> bytes:
    if (relative.is_absolute() or ".." in relative.parts
            or not relative.is_relative_to(Path("data/private_nextgqa"))):
        raise ValueError("private_input_path_invalid")
    root = root.resolve(strict=True)
    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    opened_dirs: list[int] = []
    try:
        # Walk by directory descriptor so a concurrent path replacement cannot
        # redirect a later open outside the private tree.
        data_fd = os.open(root / "data", dir_flags)
        opened_dirs.append(data_fd)
        data_info = os.fstat(data_fd)
        if not stat.S_ISDIR(data_info.st_mode) or data_info.st_uid != os.getuid():
            raise ValueError("private_data_root_invalid")
        private_fd = os.open("private_nextgqa", dir_flags, dir_fd=data_fd)
        opened_dirs.append(private_fd)
        private_info = os.fstat(private_fd)
        if (not stat.S_ISDIR(private_info.st_mode) or private_info.st_uid != os.getuid()
                or stat.S_IMODE(private_info.st_mode) != 0o700):
            raise ValueError("private_root_permissions_invalid")
        parts = relative.parts[2:]
        parent_fd = private_fd
        for part in parts[:-1]:
            child_fd = os.open(part, dir_flags, dir_fd=parent_fd)
            opened_dirs.append(child_fd)
            child_info = os.fstat(child_fd)
            if (not stat.S_ISDIR(child_info.st_mode) or child_info.st_uid != os.getuid()
                    or stat.S_IMODE(child_info.st_mode) != 0o700):
                raise ValueError("private_parent_permissions_invalid")
            parent_fd = child_fd
        fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        opened_dirs.append(fd)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_input_permissions_invalid")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except ValueError:
        raise
    except OSError as exc:
        reason = "private_input_path_invalid" if exc.errno in (20, 40) else "private_input_open_failed"
        raise ValueError(reason) from exc
    finally:
        for fd in reversed(opened_dirs):
            os.close(fd)


def _json(raw: bytes) -> dict:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("private_input_duplicate_json_key")
            value[key] = item
        return value

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                           parse_constant=lambda _value: (_ for _ in ()).throw(
                               ValueError("private_input_nonfinite_json_number")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private_input_json_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("private_input_json_object_required")
    return value


def _source_sha256(root: Path, relative_name: str) -> str:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source_path_invalid")
    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    opened_dirs: list[int] = []
    try:
        parent_fd = os.open(root, dir_flags)
        opened_dirs.append(parent_fd)
        for part in relative.parts[:-1]:
            child_fd = os.open(part, dir_flags, dir_fd=parent_fd)
            opened_dirs.append(child_fd)
            parent_fd = child_fd
        fd = os.open(relative.parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                     dir_fd=parent_fd)
        opened_dirs.append(fd)
    except OSError as exc:
        for opened in reversed(opened_dirs):
            os.close(opened)
        raise ValueError("source_open_failed") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("source_path_invalid")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        for opened in reversed(opened_dirs):
            os.close(opened)


def analyze(root: Path, packet_rel: Path = DEFAULT_PACKET,
            lock_rel: Path = DEFAULT_LOCK, filter_rel: Path = DEFAULT_FILTER,
            receipt_stem: str = DEFAULT_RECEIPT_STEM) -> dict:
    packet_raw = _private_bytes(root, packet_rel)
    lock_raw = _private_bytes(root, lock_rel)
    filter_raw = _private_bytes(root, filter_rel)
    packet, lock, pair_filter = map(_json, (packet_raw, lock_raw, filter_raw))
    records = packet.get("records")
    assignments = lock.get("assignments")
    pairs = pair_filter.get("pair_sha256")
    if (packet.get("status") != "passed" or packet.get("dataset_split") != "test"
            or packet.get("model_revision") != scorer.MODEL_REVISION
            or packet.get("generation_prompt_version") != "v1"
            or packet.get("effective_videos") != 3
            or packet.get("fully_structured_claim_records") != 3
            or not isinstance(records, list) or len(records) != 3
            or lock.get("effective_videos") != 3 or lock.get("dataset_split") != "test"
            or not isinstance(assignments, list) or len(assignments) != 3
            or pair_filter.get("schema_version") != 1 or not isinstance(pairs, list)
            or len(pairs) != 3 or len(set(pairs)) != 3):
        raise ValueError("frozen_cohort_contract_invalid")
    if (any(not isinstance(record, dict) for record in records)
            or any(not isinstance(assignment, dict) for assignment in assignments)
            or any(not isinstance(pair, str) or len(pair) != 64
                   or any(ch not in "0123456789abcdef" for ch in pair) for pair in pairs)):
        raise ValueError("frozen_cohort_row_schema_invalid")
    packet_sha = hashlib.sha256(packet_raw).hexdigest()
    filter_sha = hashlib.sha256(filter_raw).hexdigest()
    lock_pair_sha = hashlib.sha256((json.dumps(pair_filter, sort_keys=True,
        separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()
    if (lock.get("selected_pair_filter_sha256") != lock_pair_sha
            or packet.get("candidate_pair_filter_sha256") != filter_sha
            or packet.get("candidate_pair_filter_digest") != pilot._hash(sorted(pairs))
            or packet.get("cohort_sha256") != lock.get("cohort_sha256")):
        raise ValueError("cohort_filter_or_packet_binding_mismatch")
    lock_sources = lock.get("source_sha256")
    if (not isinstance(lock_sources, dict) or set(lock_sources) != LOCK_SOURCES
            or any(_source_sha256(root, name) != lock_sources[name]
                   for name in LOCK_SOURCES)):
        raise ValueError("lock_source_hash_mismatch")
    if (packet.get("runner_sha256") != _source_sha256(
            root, "scripts/generate_nextgqa_claim_review_pilot.py")
            or packet.get("pilot_runner_sha256") != _source_sha256(
                root, "scripts/run_nextgqa_gold_interval_margin_pilot.py")):
        raise ValueError("generation_source_hash_mismatch")

    for i, (record, assignment, pair_hash) in enumerate(
            zip(records, assignments, pairs, strict=True)):
        parsed = record.get("generated_parsed")
        qid_hash = hashlib.sha256(str(record.get("dataset_question_id", "")).encode()).hexdigest()
        if (not isinstance(parsed, dict) or parsed.get("valid") is not True
                or not isinstance(parsed.get("claim"), str) or not parsed["claim"].strip()
                or record.get("generation_prompt_version") != "v1"
                or assignment.get("media_sha256") != record.get("video_sha256")
                or assignment.get("qid_sha256") != qid_hash
                or record.get("candidate_pair_sha256") != pair_hash):
            raise ValueError("cohort_row_assignment_mismatch")

    values = []
    receipt_hashes = []
    forwards = 0
    source_bindings_ok = True
    for i, record in enumerate(records):
        receipt_rel = Path(f"{receipt_stem}{i}.json")
        raw = _private_bytes(root, receipt_rel)
        receipt = _json(raw)
        branches = receipt.get("branch_target_scores")
        if not isinstance(branches, dict):
            raise ValueError("receipt_branches_missing")
        if (not isinstance(receipt.get("human_review"), dict)
                or not isinstance(receipt.get("primary_descriptive_interaction"), dict)):
            raise ValueError("receipt_nested_schema_invalid")
        if (receipt.get("status") != "completed_one_row_exploratory_sensitivity_smoke"
                or receipt.get("packet_record_index_zero_based") != i
                or receipt.get("private_input_packet_sha256") != packet_sha
                or receipt.get("source_video_sha256") != record.get("video_sha256")
                or receipt.get("saved_claim_text_sha256") != hashlib.sha256(
                    record["generated_parsed"]["claim"].encode()).hexdigest()
                or receipt.get("row_token_sha256") != hashlib.sha256((
                    record["video_sha256"] + str(record["dataset_question_id"])
                ).encode()).hexdigest()
                or receipt.get("completed_forwards") != 4
                or receipt.get("requested_forwards") != 4
                or receipt.get("direct_faithfulness") is not False
                or receipt.get("faithfulness_result") is not False
                or receipt.get("human_review", {}).get("review_actually_performed") is not False
                or receipt.get("model_revision") != scorer.MODEL_REVISION
                or receipt.get("precision") != "4-bit weights; BF16 compute; eager attention"):
            raise ValueError("receipt_cohort_binding_or_scope_invalid")
        required = {"original_baseline", "permuted_baseline",
                    "original_visual_bias", "permuted_visual_bias"}
        if set(branches) != required:
            raise ValueError("receipt_condition_set_invalid")
        if any(not isinstance(branches[name], dict) for name in required):
            raise ValueError("receipt_nested_schema_invalid")
        interaction = scorer.compute_bias_frame_order_interaction(
            original_baseline=branches["original_baseline"]["target_logprob_mean"],
            permuted_baseline=branches["permuted_baseline"]["target_logprob_mean"],
            original_biased=branches["original_visual_bias"]["target_logprob_mean"],
            permuted_biased=branches["permuted_visual_bias"]["target_logprob_mean"])
        observed = receipt.get("primary_descriptive_interaction", {}).get(
            "bias_by_frame_order_interaction_nats_per_token")
        calculated = interaction["bias_by_frame_order_interaction_nats_per_token"]
        try:
            observed_float = float(observed)
        except (TypeError, ValueError, OverflowError):
            observed_float = math.nan
        if (isinstance(observed, bool) or not isinstance(observed, (int, float))
                or not math.isfinite(observed_float)
                or abs(observed_float - calculated) > 1e-12):
            raise ValueError("receipt_interaction_arithmetic_mismatch")
        source_hashes = receipt.get("source_sha256")
        if (not isinstance(source_hashes, dict)
                or set(source_hashes) != RECEIPT_SOURCES
                or any(_source_sha256(root, name) != source_hashes[name]
                       for name in RECEIPT_SOURCES)):
            source_bindings_ok = False
        values.append(calculated)
        forwards += receipt["completed_forwards"]
        receipt_hashes.append(hashlib.sha256(raw).hexdigest())
    if not source_bindings_ok:
        raise ValueError("receipt_source_hash_mismatch")
    return {
        "status": "complete_exploratory_cohort",
        "requested_videos": 3,
        "effective_videos": 3,
        "structured_claims": 3,
        "completed_forwards": forwards,
        "failed_or_excluded_rows": 0,
        "row_interactions_nats_per_token": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "range": [min(values), max(values)],
        "positive_rows": sum(value > 0 for value in values),
        "source_hashes_verified": True,
        "row_assignment_and_pair_bindings_verified": True,
        "direct_faithfulness_measured": False,
        "receipt_sha256": receipt_hashes,
        "inferential_status": "finite-cohort descriptive only; no CI, test, or population inference",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--pair-filter", type=Path, default=DEFAULT_FILTER)
    parser.add_argument("--receipt-stem", default=DEFAULT_RECEIPT_STEM)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        result = analyze(root, args.packet, args.lock, args.pair_filter,
                         args.receipt_stem)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError,
            OverflowError, RecursionError) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "input_schema_or_io_rejected"
        print(json.dumps({"status": "rejected", "reason": reason}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
