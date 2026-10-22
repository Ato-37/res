#!/usr/bin/env python3
"""Recompute the frozen three-row test-split internal-bias interaction."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat

ROOT = Path("data/private_nextgqa")
RUNS = (
    ("test_split_internal_bias_frame_mean_interaction_smoke_20261021_01.json",
     "saved_claim_frame_mean_test_smoke_20261021_02_retry01.json"),
    ("test_split_internal_bias_frame_mean_interaction_row0_20261021_01.json",
     "saved_claim_frame_mean_test_remaining_20261021_02_row0.json"),
    ("test_split_internal_bias_frame_mean_interaction_row1_20261021_01.json",
     "saved_claim_frame_mean_test_remaining_20261021_02_row1.json"),
)
PACKETS = (
    ("claim_generation_test_frame_mean_smoke_20261021_02.json", 0),
    ("claim_generation_test_frame_mean_remaining_20261021_02.json", 0),
    ("claim_generation_test_frame_mean_remaining_20261021_02.json", 1),
)
LOCK_NAME = "saved_claim_frame_mean_test_lock_20261021_02.json"
FILTER_NAME = "saved_claim_frame_mean_test_pairs_20261021_02.json"


def read_private(path: Path) -> tuple[dict, str]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("private_receipt_permissions_invalid")
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()



def _source_matches(repo: Path, source_name: str, source_digest: str) -> bool:
    """Current source must match the recorded digest, or an exact frozen copy of the executed version must.

    A later edit to a tracked script must not silently invalidate or relax provenance: the hash is still
    enforced, against `scripts/frozen_sources/<stem>__sha256_<first8>.py` when the live file has moved on.
    """
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    source_path = repo / source_name
    if source_path.is_file() and digest(source_path) == source_digest:
        return True
    frozen = repo / "scripts" / "frozen_sources" / f"{Path(source_name).stem}__sha256_{source_digest[:8]}.py"
    return frozen.is_file() and not frozen.is_symlink() and digest(frozen) == source_digest

def summarize(repo: Path) -> dict:
    lock, lock_sha = read_private(repo / ROOT / LOCK_NAME)
    pair_filter, pair_filter_sha = read_private(repo / ROOT / FILTER_NAME)
    if (lock.get("dataset_split") != "test" or lock.get("status") != "frozen_before_claim_generation_and_scoring"
            or lock.get("effective_videos") != 3 or lock.get("selected_pair_filter_sha256") != pair_filter_sha
            or not isinstance(pair_filter.get("pair_sha256"), list)):
        raise ValueError("frozen_cohort_lock_or_filter_invalid")
    selected_pairs = set(pair_filter["pair_sha256"])
    deltas = []
    requested = completed = failures = 0
    details = []
    for row_index, ((run_name, baseline_name), (packet_name, packet_index)) in enumerate(zip(RUNS, PACKETS, strict=True)):
        run, run_sha = read_private(repo / ROOT / run_name)
        base, base_sha = read_private(repo / ROOT / baseline_name)
        packet_path = repo / ROOT / packet_name
        packet = json.loads(packet_path.read_text())
        row = packet["records"][packet_index]
        requested += int(run.get("requested_forwards", 0))
        completed += int(run.get("completed_forwards", 0))
        if run.get("status") != "completed_one_row_test_split_internal_bias_frame_mean_interaction":
            failures += 1
            raise ValueError("run_not_completed:" + run_name)
        if (run.get("dataset_split") != "test" or run.get("direct_faithfulness") is not False
                or run.get("completed_forwards") != 2
                or run.get("baseline_receipt_sha256") != base_sha
                or base.get("status") != "completed_one_row_posthoc_frame_mean_sensitivity"
                or base.get("dataset_split") != "test"):
            raise ValueError("run_or_baseline_binding_invalid:" + run_name)
        qid = row.get("dataset_question_id")
        media = row.get("video_sha256")
        assignment = lock["assignments"][row_index]
        if (assignment.get("media_sha256") != media
                or assignment.get("qid_sha256") != hashlib.sha256(str(qid).encode()).hexdigest()
                or row.get("candidate_pair_sha256") not in selected_pairs
                or base.get("private_input_packet_sha256") != hashlib.sha256(packet_path.read_bytes()).hexdigest()
                or run.get("cohort_lock_sha256") != lock_sha
                or run.get("selected_pair_filter_sha256") != pair_filter_sha):
            raise ValueError("frozen_packet_assignment_or_filter_membership_invalid:" + run_name)
        if (run.get("condition_hashes", {}).get("original_processor_pixels_sha256")
                != base.get("condition", {}).get("original_processor_pixels_sha256")
                or run.get("condition_hashes", {}).get("frame_mean_processor_pixels_sha256")
                != base.get("condition", {}).get("frame_mean_processor_pixels_sha256")):
            raise ValueError("processor_condition_hash_join_invalid:" + run_name)
        for source_name, source_digest in run.get("source_sha256", {}).items():
            if not _source_matches(repo, source_name, source_digest):
                raise ValueError("run_source_hash_mismatch:" + source_name)
        expected_layers = {str(layer): 1 for layer in range(5, 17)}
        for branch in ("original_video", "per_frame_rgb_mean_video"):
            score = run.get("biased_scores", {}).get(branch, {})
            if (score.get("intervention_hook_invocations_by_layer_1_based") != expected_layers
                    or score.get("score_semantics") != "mean conditional logprob of frozen target token IDs"
                    or score.get("target_token_count") != run.get("target_token_count")
                    or score.get("visual_key_position_count", 0) < 1):
                raise ValueError("biased_branch_audit_invalid:" + run_name + ":" + branch)
        base_scores = base["branch_target_scores"]
        run_scores = run["biased_scores"]
        baseline_contrast = (base_scores["original_video"]["target_logprob_mean"]
                             - base_scores["per_frame_rgb_mean_video"]["target_logprob_mean"])
        biased_contrast = (run_scores["original_video"]["target_logprob_mean"]
                           - run_scores["per_frame_rgb_mean_video"]["target_logprob_mean"])
        delta = biased_contrast - baseline_contrast
        stored = run["primary_interaction"]["bias_by_frame_mean_interaction_nats_per_token"]
        if not math.isclose(delta, stored, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("interaction_arithmetic_mismatch:" + run_name)
        deltas.append(delta)
        details.append({"row_ordinal": len(details), "baseline_contrast": baseline_contrast,
                        "biased_contrast": biased_contrast, "interaction": delta,
                        "receipt_sha256": run_sha})
    ordered = sorted(deltas)
    return {"status": "verified_finite_cohort_summary", "requested_rows": 3,
        "effective_rows": len(deltas), "requested_new_forwards": requested,
        "completed_new_forwards": completed, "failed_rows": failures,
        "mean_interaction": sum(deltas) / len(deltas), "median_interaction": ordered[1],
        "range_interaction": [ordered[0], ordered[-1]],
        "positive_rows": sum(value > 0 for value in deltas), "rows": details,
        "uncertainty": "No population CI or hypothesis test; post-result selected n=3."}


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    print(json.dumps(summarize(repo), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
