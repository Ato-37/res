#!/usr/bin/env python3
"""Recompute the fixed v2_short frame-order interaction from private receipts.

This is a finite-cohort audit of an outcome-exposed three-row exploratory
rerun. It is not a population estimate or direct faithfulness score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

PRIVATE = Path("data/private_nextgqa")
PROMPT_SHA256 = "853589028867f20e0c1be6f98a26ec7c81d90cc0bf4e29f749865891f602976e"
EXPECTED_FORWARDS = 4


def read_receipt(path: Path, repo: Path) -> tuple[dict, str]:
    if path.is_absolute() or not path.is_relative_to(PRIVATE):
        raise ValueError("receipt_path_outside_private_root")
    full = repo / path
    if full.is_symlink():
        raise ValueError("receipt_symlink_rejected")
    resolved = full.resolve(strict=True)
    private_root = (repo / PRIVATE).resolve(strict=True)
    if not resolved.is_relative_to(private_root):
        raise ValueError("receipt_path_outside_private_root")
    fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not os.path.isfile(resolved) or info.st_uid != os.getuid() or info.st_mode & 0o777 != 0o600:
            raise ValueError("receipt_permissions_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("receipt_object_required")
    return payload, hashlib.sha256(raw).hexdigest()


def audit(new_paths: list[Path], old_paths: list[Path], repo: Path) -> dict:
    if len(new_paths) < 1 or len(new_paths) != len(old_paths):
        raise ValueError("matched_nonempty_new_and_old_receipts_required")
    new = [read_receipt(path, repo)[0] for path in new_paths]
    old = [read_receipt(path, repo)[0] for path in old_paths]
    old_by_token = {row.get("row_token_sha256"): row for row in old}
    if len(old_by_token) != len(old) or None in old_by_token:
        raise ValueError("old_receipt_row_tokens_invalid")
    packet_hashes = {row.get("private_input_packet_sha256") for row in new}
    if not packet_hashes or not all(isinstance(value, str) and len(value) == 64 for value in packet_hashes):
        raise ValueError("new_receipt_packet_binding_invalid")
    rows = []
    for receipt in new:
        if (receipt.get("status") != "completed_one_row_exploratory_sensitivity_smoke"
                or receipt.get("generation_prompt_version") != "v2_short"
                or receipt.get("prompt_suffix_sha256") != PROMPT_SHA256
                or receipt.get("requested_forwards") != EXPECTED_FORWARDS
                or receipt.get("completed_forwards") != EXPECTED_FORWARDS
                or receipt.get("direct_faithfulness") is not False
                or receipt.get("faithfulness_result") is not False):
            raise ValueError("new_receipt_contract_invalid")
        token = receipt.get("row_token_sha256")
        prior = old_by_token.get(token)
        if (prior is None or prior.get("source_video_sha256") != receipt.get("source_video_sha256")
                or prior.get("saved_claim_text_sha256") != receipt.get("saved_claim_text_sha256")
                or prior.get("private_input_packet_sha256") != receipt.get("private_input_packet_sha256")):
            raise ValueError("old_new_row_binding_mismatch")
        for name, digest in receipt.get("source_sha256", {}).items():
            source = (repo / name).resolve(strict=True)
            if not source.is_relative_to(repo.resolve()) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError("source_hash_mismatch")
        score = receipt.get("branch_target_scores", {})
        try:
            baseline_gap = float(score["original_baseline"]["target_logprob_mean"]) - float(score["permuted_baseline"]["target_logprob_mean"])
            biased_gap = float(score["original_visual_bias"]["target_logprob_mean"]) - float(score["permuted_visual_bias"]["target_logprob_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("branch_scores_invalid") from exc
        interaction = biased_gap - baseline_gap
        stored = receipt.get("primary_descriptive_interaction", {}).get("bias_by_frame_order_interaction_nats_per_token")
        if not all(math.isfinite(x) for x in (baseline_gap, biased_gap, interaction)) or not math.isclose(interaction, float(stored), rel_tol=0, abs_tol=1e-8):
            raise ValueError("interaction_recomputation_mismatch")
        old_interaction = float(prior["primary_descriptive_interaction"]["bias_by_frame_order_interaction_nats_per_token"])
        rows.append((token, baseline_gap, biased_gap, interaction, old_interaction))
    if len({x[0] for x in rows}) != len(rows):
        raise ValueError("duplicate_new_receipt_rows")
    vals = [x[3] for x in rows]
    old_vals = [x[4] for x in rows]
    return {
        "cohort_rows": len(rows),
        "completed_forwards": sum(EXPECTED_FORWARDS for _ in rows),
        "packet_sha256s": sorted(packet_hashes),
        "prompt_version": "v2_short",
        "same_rows_as_prior_receipts": True,
        "all_new_source_hashes_recomputed": True,
        "baseline_order_gap_mean": statistics.mean(x[1] for x in rows),
        "biased_order_gap_mean": statistics.mean(x[2] for x in rows),
        "corrected_interaction_rows": [x[3] for x in rows],
        "corrected_interaction_mean": statistics.mean(vals),
        "corrected_interaction_median": statistics.median(vals),
        "corrected_interaction_range": [min(vals), max(vals)],
        "corrected_positive_rows": sum(x > 0 for x in vals),
        "old_v1_prompt_interaction_rows": old_vals,
        "old_v1_prompt_interaction_mean": statistics.mean(old_vals),
        "old_v1_prompt_negative_rows": sum(x < 0 for x in old_vals),
        "inferential_status": f"outcome-exposed fixed n={len(rows)} descriptive rerun; no CI or test; not direct faithfulness",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", nargs="+", type=Path, required=True)
    parser.add_argument("--old", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    print(json.dumps(audit(args.new, args.old, repo), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
