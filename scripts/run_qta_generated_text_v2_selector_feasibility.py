#!/usr/bin/env python3
"""Local-only selector feasibility gate for the prospective QTA proxy v2.

The fixed v2 pool is hash-lock indices 4..15. This command does not generate
text or score target tokens. Its smoke mode checks wiring on already-used index
0, excluded from v2 eligibility and outcomes. Its pool mode freezes the first
four rows eligible under the v2 disjoint-set rule, or fails without expansion.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import subprocess
import time
from datetime import datetime, timezone

from freeze_qta_generated_text_candidate_cohort import sha256_file
from preflight_qta_generated_text_annotation_dependence import (
    DATA_FILES, SOURCE_FILES as V1_SOURCE_FILES, MODEL_REVISION,
    prepare_candidate_geometry,
)
from qta_disjoint_control_design import select_qta_disjoint_rank_matched_control_groups
from qta_multiquery_video_key_intervention import validate_evidence_control_key_sets
from run_qta_generated_text_annotation_dependence import (
    DATASET_REL, PRIVATE_REL, _capture_selector, _move_batch_to_cuda,
    _read_private_json, load_dataset_indexes,
)

LOCK_NAME = "qta_generated_text_candidate_cohort_local_20261019_01.json"
PREFLIGHT_NAME = "qta_generated_text_geometry_preflight_local_20261019_11.json"
MODEL_REL = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
             / "snapshots" / MODEL_REVISION)
FRAMES = 16
LAYERS_1_BASED = tuple(range(5, 17))
SEED = 20261020
MIN_FREE_VRAM = 11 * 1024**3
MIN_FREE_HOST = 8 * 1024**3
POOL_START = 4
POOL_STOP_EXCLUSIVE = 16
POOL_REQUIRED = 4


def _write_private(path: Path, private_root: Path, payload: dict) -> None:
    root = private_root.resolve(strict=True)
    if (stat.S_IMODE(root.stat().st_mode) != 0o700
            or path.resolve().parent != root or os.path.lexists(path)):
        raise ValueError("private_output_path_invalid_or_exists")
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _source_inventory(repo: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    names = tuple(dict.fromkeys((*V1_SOURCE_FILES,
        "scripts/qta_disjoint_control_design.py",
        "scripts/run_qta_generated_text_v2_selector_feasibility.py")))
    return {name: sha256_file(repo / name) for name in names}, names


def validate_candidate_lock(lock: dict, data_hashes: dict[str, str]) -> None:
    """Reject stale candidate order or dataset bindings before model load."""
    if (not isinstance(lock, dict) or type(lock.get("schema_version")) is not int
            or lock.get("schema_version") != 1
            or lock.get("status") != "candidate_order_frozen_geometry_not_preflighted_not_authorizing"
            or not isinstance(lock.get("candidates"), list)
            or len(lock["candidates"]) < POOL_STOP_EXCLUSIVE
            or not isinstance(data_hashes, dict)
            or lock.get("dataset_sha256") != data_hashes):
        raise ValueError("candidate_lock_or_dataset_binding_invalid")


def candidate_indices(mode: str) -> tuple[list[int], int]:
    if mode == "smoke":
        return [0], 1
    if mode == "pool":
        return list(range(POOL_START, POOL_STOP_EXCLUSIVE)), POOL_REQUIRED
    raise ValueError("selector_mode_invalid")


def validate_successful_smoke_receipt(receipt: dict, *, lock_sha: str,
        preflight_sha: str, source_hashes: dict[str, str],
        data_hashes: dict[str, str]) -> None:
    """Require a complete, hash-bound selector-only smoke before pool work."""
    if not isinstance(receipt, dict):
        raise ValueError("successful_hash_bound_selector_smoke_receipt_required")
    attempts = receipt.get("attempts")
    valid = (
        type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 1
        and receipt.get("status") == "selector_engineering_smoke_completed_no_scoring"
        and receipt.get("mode") == "smoke"
        and receipt.get("candidate_lock_sha256") == lock_sha
        and receipt.get("smoke_preflight_sha256") == preflight_sha
        and receipt.get("source_sha256") == source_hashes
        and receipt.get("dataset_sha256") == data_hashes
        and receipt.get("candidate_indices_attempted") == [0]
        and receipt.get("cpu_geometry_attempted") == 1
        and receipt.get("cpu_geometry_passed") == 1
        and receipt.get("selector_forwards_attempted") == 1
        and receipt.get("selector_forwards_completed") == 1
        and receipt.get("eligible_count") == 1
        and receipt.get("generation_performed") is False
        and receipt.get("target_scoring_performed") is False
        and receipt.get("scoring_authorized") is False
        and receipt.get("direct_faithfulness") is False
        and receipt.get("faithfulness_result") is False
        and isinstance(attempts, list) and len(attempts) == 1
        and attempts[0].get("lock_index") == 0
        and attempts[0].get("status") == "engineering_smoke_eligible"
    )
    if not valid:
        raise ValueError("successful_hash_bound_selector_smoke_receipt_required")


def pool_attempts_complete(*, attempts: list[dict], eligible_count: int,
        required: int) -> bool:
    """Check that every attempted pool row has a terminal recorded outcome."""
    allowed = {"geometry_ineligible", "selector_failure", "selector_ineligible", "eligible"}
    if not isinstance(attempts, list) or not attempts:
        return False
    indices = [row.get("lock_index") for row in attempts]
    if indices != list(range(POOL_START, POOL_START + len(indices))) or any(
            row.get("status") not in allowed for row in attempts):
        return False
    observed_eligible = sum(row.get("status") == "eligible" for row in attempts)
    return observed_eligible == eligible_count == required


def selector_failure_requires_stop(exc: Exception, reason: str) -> bool:
    """Stop after an exhausted-memory failure, but consume row-local errors."""
    return ("free_vram_guard_failed" in reason
        or "out of memory" in str(exc).lower()
        or type(exc).__name__ == "OutOfMemoryError")


def _eligible_sets(*, mapped: dict, group_positions: tuple[tuple[int, ...], ...],
                   group_scores: list[float]) -> dict:
    from qwen_question_temporal_attention_selector import select_top_temporal_groups

    total = len(group_positions)
    qta_indices = tuple(select_top_temporal_groups(
        group_scores, top_k=max(1, (total + 9) // 10)))
    evidence_indices = tuple(mapped["evidence_group_indices"])
    control_indices = tuple(select_qta_disjoint_rank_matched_control_groups(
        evidence_group_indices=evidence_indices,
        qta_group_indices=qta_indices,
        temporal_group_count=total))
    evidence = tuple(mapped["evidence_token_positions"])
    control = tuple(sorted(position for i in control_indices
                           for position in group_positions[i]))
    qta = tuple(sorted(position for i in qta_indices
                       for position in group_positions[i]))
    budget = validate_evidence_control_key_sets(
        evidence_positions=evidence, control_positions=control)
    if set(qta).intersection(evidence):
        raise ValueError("qta_evidence_visual_key_overlap")
    if set(qta).intersection(control):
        raise ValueError("qta_control_visual_key_overlap")
    if not (len(evidence) == len(control)
            and len(set(evidence) | set(control) | set(qta))
            == len(evidence) + len(control) + len(qta)):
        raise ValueError("three_way_visual_key_partition_invalid")
    rank_cost = sum(abs(a - b) for a, b in
                    zip(sorted(evidence_indices), sorted(control_indices)))
    return {"evidence_group_indices": list(evidence_indices),
        "qta_group_indices": list(qta_indices),
        "control_group_indices": list(control_indices),
        "evidence_key_count": len(evidence), "qta_key_count": len(qta),
        "control_key_count": len(control), "rank_distance_cost": rank_cost,
        "pairwise_disjoint_equal_key_sets": True}


def run(mode: str, output_name: str, repo: Path,
        smoke_receipt_name: str | None = None) -> dict:
    private = repo / PRIVATE_REL
    dataset = repo / DATASET_REL
    output = private / output_name
    if (Path(output_name).name != output_name
            or not output_name.startswith("qta_generated_text_v2_selector_")):
        raise ValueError("output_name_must_be_private_basename")
    if os.path.lexists(output):
        raise ValueError("refusing_to_overwrite_private_output")
    lock, lock_sha = _read_private_json(private / LOCK_NAME, private)
    source_hashes, source_names = _source_inventory(repo)
    data_hashes = {name: sha256_file(dataset / name) for name in DATA_FILES}
    validate_candidate_lock(lock, data_hashes)
    smoke_preflight, smoke_preflight_sha = _read_private_json(
        private / PREFLIGHT_NAME, private)
    if (smoke_preflight.get("status") != "geometry_preflight_passed_not_scoring_authorization"
            or smoke_preflight.get("candidate_order_sha256") != lock_sha
            or smoke_preflight.get("source_sha256") != {
                name: sha256_file(repo / name) for name in V1_SOURCE_FILES}
            or smoke_preflight.get("dataset_sha256") != data_hashes
            or len(smoke_preflight.get("selected_candidates", [])) != 4):
        raise ValueError("engineering_smoke_preflight_binding_failed")
    smoke_receipt_sha = None
    if mode == "pool":
        if (not isinstance(smoke_receipt_name, str)
                or Path(smoke_receipt_name).name != smoke_receipt_name
                or not smoke_receipt_name.startswith("qta_generated_text_v2_selector_smoke_")
                or not smoke_receipt_name.endswith(".json")):
            raise ValueError("pool_requires_private_smoke_receipt_name")
        smoke_receipt, smoke_receipt_sha = _read_private_json(
            private / smoke_receipt_name, private)
        validate_successful_smoke_receipt(smoke_receipt,
            lock_sha=lock_sha, preflight_sha=smoke_preflight_sha,
            source_hashes=source_hashes, data_hashes=data_hashes)
    start = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA_BF16_required_no_CPU_fallback")
    free, total_vram = torch.cuda.mem_get_info()
    host_free = next((int(line.split()[1]) * 1024 for line in
        Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")), 0)
    if free < MIN_FREE_VRAM or host_free < MIN_FREE_HOST:
        raise RuntimeError("pre_model_gpu_or_host_memory_guard_failed")

    from transformers import (AutoProcessor, BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration)
    processor = AutoProcessor.from_pretrained(str(MODEL_REL), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    import run_nextgqa_validation_qta_annotation_proxy as kernel
    checkpoint_audit = kernel.audit_pinned_checkpoint(repo, MODEL_REL)
    processor_pins = kernel.processor_provenance(processor, MODEL_REL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL_REL),
        local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if (int(model.config.video_token_id) != int(processor.video_token_id)
            or len(model.model.language_model.layers) < max(LAYERS_1_BASED)):
        raise ValueError("pinned_model_processor_contract_invalid")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()

    rows, gsub = load_dataset_indexes(dataset)
    video_root = Path.home() / "nextgqa_sample/NExTVideo"
    indices, required = candidate_indices(mode)
    if mode == "smoke":
        expected_geometry = smoke_preflight["selected_candidates"][0]["geometry_sha256"]
    else:
        indices = list(range(POOL_START, POOL_STOP_EXCLUSIVE))
        required = POOL_REQUIRED
        expected_geometry = None

    cpu_geometry_passed = 0
    selector_attempted = 0
    selector_completed = 0
    eligible = []
    exclusions: dict[str, int] = {}
    attempts = []
    for lock_index in indices:
        candidate = lock["candidates"][lock_index]
        attempt = {"lock_index": lock_index,
            "video_sha256": candidate["video_sha256"],
            "qid_sha256": hashlib.sha256(candidate["qid"].encode()).hexdigest()}
        try:
            prepared = prepare_candidate_geometry(dataset=dataset,
                video_root=video_root, processor=processor,
                candidate=candidate, rows=rows, gsub=gsub)
        except Exception as exc:
            code = str(exc).splitlines()[0][:120]
            reason = "geometry:" + type(exc).__name__ + ":" + code
            exclusions[reason] = exclusions.get(reason, 0) + 1
            attempt.update({"status": "geometry_ineligible", "reason": reason})
            attempts.append(attempt)
            if mode == "smoke":
                break
            continue
        cpu_geometry_passed += 1
        if mode == "smoke" and prepared["geometry_sha256"] != expected_geometry:
            raise ValueError("engineering_smoke_geometry_digest_mismatch")
        group_positions = tuple(tuple(group["token_positions"])
                                for group in prepared["mapped"]["groups"])
        selector_attempted += 1
        batch = None
        try:
            free_now, _ = torch.cuda.mem_get_info()
            if free_now < 2 * 1024**3:
                raise RuntimeError("per_selector_free_vram_guard_failed")
            batch = _move_batch_to_cuda(prepared["batch"], torch)
            question_positions = list(prepared["question_positions"])
            group_scores, capture_receipt = _capture_selector(
                model, batch, torch=torch, question_positions=question_positions,
                groups=list(group_positions))
            selector_completed += 1
        except Exception as exc:
            reason = "selector:" + type(exc).__name__ + ":" + str(exc).splitlines()[0][:120]
            exclusions[reason] = exclusions.get(reason, 0) + 1
            attempt.update({"status": "selector_failure", "reason": reason,
                "geometry_sha256": prepared["geometry_sha256"]})
            attempts.append(attempt)
            if batch is not None:
                del batch
            del prepared
            gc.collect(); torch.cuda.empty_cache()
            # A row-local capture error consumes this frozen candidate. Continue
            # the prespecified pool; stop only if the per-forward memory guard
            # itself failed, because the remaining rows cannot safely run now.
            if selector_failure_requires_stop(exc, reason):
                break
            if mode == "smoke":
                break
            continue
        try:
            sets = _eligible_sets(mapped=prepared["mapped"],
                group_positions=group_positions, group_scores=group_scores)
        except ValueError as exc:
            reason_code = str(exc).splitlines()[0][:120]
            allowed = {"qta_evidence_groups_overlap",
                "qta_disjoint_equal_count_control_unavailable",
                "evidence_control_visual_key_budget_mismatch",
                "evidence_control_visual_key_overlap",
                "qta_evidence_visual_key_overlap", "qta_control_visual_key_overlap",
                "three_way_visual_key_partition_invalid"}
            if reason_code not in allowed:
                raise
            reason = "eligibility:" + reason_code
            exclusions[reason] = exclusions.get(reason, 0) + 1
            attempt.update({"status": "selector_ineligible", "reason": reason,
                "geometry_sha256": prepared["geometry_sha256"],
                "group_attention_scores": group_scores,
                "temporal_group_count": len(group_positions)})
            attempts.append(attempt)
            del batch, prepared
            gc.collect(); torch.cuda.empty_cache()
            if mode == "smoke":
                break
            continue
        attempt.update({"geometry_sha256": prepared["geometry_sha256"],
            "temporal_group_count": len(group_positions),
            "selector_capture": capture_receipt,
            "group_attention_scores": group_scores,
            **sets})
        if mode == "smoke":
            attempt["status"] = "engineering_smoke_eligible"
            eligible.append(attempt)
        else:
            eligible.append(attempt)
            attempt["status"] = "eligible"
        attempts.append(attempt)
        del batch, prepared
        gc.collect(); torch.cuda.empty_cache()
        if len(eligible) >= required:
            break

    if mode == "smoke":
        complete = (len(eligible) == 1 and selector_completed == 1
                    and attempts and attempts[0].get("status") == "engineering_smoke_eligible")
        status = ("selector_engineering_smoke_completed_no_scoring"
                  if complete else "selector_engineering_smoke_failed_no_scoring")
    else:
        complete = pool_attempts_complete(attempts=attempts,
            eligible_count=len(eligible), required=required)
        status = ("v2_selector_feasibility_passed_no_generation_or_scoring"
                  if complete else "v2_selector_feasibility_insufficient_no_scoring")

    payload = {"schema_version": 1, "status": status,
        "scope": "local exploratory selector/control eligibility only; no generation or target score",
        "mode": mode, "rights_status": "unverified_local_only",
        "direct_faithfulness": False, "faithfulness_result": False,
        "scoring_authorized": False, "generation_performed": False,
        "target_scoring_performed": False,
        "candidate_lock_name": LOCK_NAME, "candidate_lock_sha256": lock_sha,
        "candidate_index_policy": ({"smoke_index": 0, "excluded_from_v2_pool": True}
            if mode == "smoke" else {"start_index": POOL_START,
                "stop_index_exclusive": POOL_STOP_EXCLUSIVE,
                "max_candidates": POOL_STOP_EXCLUSIVE - POOL_START,
                "required_eligible": POOL_REQUIRED,
                "stop_after_required_eligible": True,
                "pool_expandable": False}),
        "smoke_preflight_sha256": smoke_preflight_sha,
        "smoke_receipt_name": smoke_receipt_name,
        "smoke_receipt_sha256": smoke_receipt_sha,
        "candidate_indices_attempted": [a["lock_index"] for a in attempts],
        "cpu_geometry_attempted": len(attempts),
        "cpu_geometry_passed": cpu_geometry_passed,
        "selector_forwards_attempted": selector_attempted,
        "selector_forwards_completed": selector_completed,
        "eligibility_required": required,
        "eligible_count": len(eligible), "exclusions_by_reason": exclusions,
        "attempts": attempts, "eligible_rows": eligible,
        "unattempted_pool_indices": [i for i in indices
            if i not in {a["lock_index"] for a in attempts}],
        "selection_rule": "first eligible rows in frozen lock order; QTA selector and E/Q/C constraints fixed before generation/scoring",
        "qta_layers_1_based": list(LAYERS_1_BASED),
        "qta_logit_bias_for_future_scoring": 0.6931471805599453,
        "selector_top_fraction_ceiling": "max(1,ceil(0.10*T))",
        "control_rule": "equal evidence group count; exclude E and Q; minimize paired sorted integer group-rank L1 distance; lexicographic tie-break",
        "frame_count": FRAMES, "seed": SEED,
        "model_revision": MODEL_REVISION,
        "precision": "4bit_bf16_compute_bf16_eager",
        "environment": {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")},
        "hardware": {"device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda, "free_vram_before_model_load_gib": free / 1024**3,
            "total_vram_gib": total_vram / 1024**3,
            "host_available_before_model_load_gib": host_free / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
        "checkpoint_byte_audit": checkpoint_audit,
        "processor_provenance": processor_pins,
        "source_commit": source_commit, "source_sha256": source_hashes,
        "dataset_sha256": data_hashes, "data_scope": DATA_FILES,
        "elapsed_seconds": time.monotonic() - start,
        "started_utc": started_utc,
        "limitations": [
            "selector eligibility conditions the v2 cohort on model attention and annotation geometry",
            "candidate pool is not certified fresh; prior exposure history is incomplete",
            "gsub is QA timing, not independently verified semantic support for a generated claim",
            "this receipt contains no generation, target likelihood, direct faithfulness, hallucination, or efficacy result"]}
    if mode == "smoke":
        payload["status"] = status
    _write_private(output, private, payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pool"), required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--smoke-receipt-name",
        help="pool mode only: previously completed private selector-smoke JSON basename")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    try:
        payload = run(args.mode, args.output_name,
                      Path(__file__).resolve().parents[1],
                      smoke_receipt_name=args.smoke_receipt_name)
    except Exception as exc:
        parser.error(f"selector_feasibility_failed:{type(exc).__name__}:{str(exc).split(':',1)[0]}")
    print(json.dumps({"status": payload["status"], "mode": args.mode,
        "attempted": payload["cpu_geometry_attempted"],
        "selector_completed": payload["selector_forwards_completed"],
        "eligible": payload["eligible_count"], "required": payload["eligibility_required"],
        "exclusions_by_reason": payload["exclusions_by_reason"],
        "generation_performed": False, "target_scoring_performed": False,
        "private_output": args.output_name,
        "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))
    return 0 if "passed" in payload["status"] or "smoke_completed" in payload["status"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
