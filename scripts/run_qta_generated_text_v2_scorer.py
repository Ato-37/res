#!/usr/bin/env python3
"""Score the frozen four-row v2 QTA annotation-dependence feasibility cohort.

The selector receipt is the source of QTA/evidence/control groups. This local
exploratory endpoint is not direct explanation faithfulness or a hallucination
judgment. Smoke row 7 must pass before rows 8-10 can be scored.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import time
from datetime import datetime, timezone

from freeze_qta_generated_text_candidate_cohort import sha256_file
from preflight_qta_generated_text_annotation_dependence import (
    DATA_FILES, MODEL_REVISION, SOURCE_FILES as V1_SOURCE_FILES,
    prepare_candidate_geometry,
)
from qta_disjoint_control_design import select_qta_disjoint_rank_matched_control_groups
from qta_multiquery_video_key_intervention import (
    append_target_to_unpadded_batch,
    compute_text_annotation_selective_dependence,
    freeze_generated_continuation,
    report_qta_key_overlap,
    validate_evidence_control_key_sets,
)
from qwen_question_temporal_attention_selector import select_top_temporal_groups
from run_qta_generated_text_annotation_dependence import (
    DATASET_REL, PRIVATE_REL, _forward, _move_batch_to_cuda,
    _read_private_json, _write_private_json, load_dataset_indexes,
)
from run_qta_generated_text_v2_selector_feasibility import (
    LOCK_NAME, PREFLIGHT_NAME, POOL_REQUIRED, V1_SOURCE_FILES,
    _source_inventory as selector_source_inventory,
    validate_successful_smoke_receipt,
)

POOL_RECEIPT_NAME = "qta_generated_text_v2_selector_pool_local_20261019_01.json"
SELECTOR_SMOKE_NAME = "qta_generated_text_v2_selector_smoke_local_20261019_01.json"
MODEL_REL = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
             / "snapshots" / MODEL_REVISION)
SEED = 20261020
FRAMES = 16
LAYERS_1_BASED = tuple(range(5, 17))
LOGIT_BIAS = 0.6931471805599453
MIN_FREE_VRAM = 11 * 1024**3
MIN_FREE_HOST = 8 * 1024**3
SCORING_BRANCHES = (
    "baseline_evidence_ablation", "baseline_control_ablation",
    "qta_evidence_ablation", "qta_control_ablation",
)


def scorer_source_inventory(repo: Path) -> dict[str, str]:
    names = tuple(dict.fromkeys((*V1_SOURCE_FILES,
        "scripts/run_qta_generated_text_annotation_dependence.py",
        "scripts/qta_multiquery_video_key_intervention.py",
        "scripts/qta_disjoint_control_design.py",
        "scripts/run_qta_generated_text_v2_selector_feasibility.py",
        "scripts/run_qta_generated_text_v2_scorer.py")))
    return {name: sha256_file(repo / name) for name in names}


def validate_selector_pool(pool: dict, *, pool_sha: str, lock: dict,
        lock_sha: str, preflight: dict, preflight_sha: str,
        selector_smoke: dict, selector_smoke_sha: str,
        source_hashes: dict[str, str], data_hashes: dict[str, str]) -> list[dict]:
    """Validate the immutable selector feasibility receipt and its hash chain."""
    if (not isinstance(pool, dict) or not isinstance(pool_sha, str)
            or len(pool_sha) != 64 or any(c not in "0123456789abcdef" for c in pool_sha)):
        raise ValueError("selector_pool_receipt_invalid")
    if (pool.get("status") != "v2_selector_feasibility_passed_no_generation_or_scoring"
            or pool.get("mode") != "pool"
            or pool.get("candidate_lock_sha256") != lock_sha
            or pool.get("smoke_preflight_sha256") != preflight_sha
            or pool.get("smoke_receipt_name") != SELECTOR_SMOKE_NAME
            or pool.get("smoke_receipt_sha256") != selector_smoke_sha
            or pool.get("source_sha256") != source_hashes
            or pool.get("dataset_sha256") != data_hashes
            or pool.get("direct_faithfulness") is not False
            or pool.get("faithfulness_result") is not False
            or pool.get("generation_performed") is not False
            or pool.get("target_scoring_performed") is not False
            or pool.get("scoring_authorized") is not False):
        raise ValueError("selector_pool_receipt_binding_invalid")
    attempts = pool.get("attempts")
    if (not isinstance(attempts, list)
            or [row.get("lock_index") for row in attempts] != list(range(4, 11))
            or sum(row.get("status") == "eligible" for row in attempts) != POOL_REQUIRED
            or [row for row in attempts if row.get("status") == "eligible"]
                != pool.get("eligible_rows")
            or pool.get("candidate_indices_attempted") != list(range(4, 11))
            or pool.get("eligible_count") != POOL_REQUIRED
            or pool.get("cpu_geometry_attempted") != 7
            or pool.get("cpu_geometry_passed") != 7
            or pool.get("selector_forwards_attempted") != 7
            or pool.get("selector_forwards_completed") != 7
            or pool.get("exclusions_by_reason") != {
                "eligibility:qta_disjoint_equal_count_control_unavailable": 2,
                "eligibility:qta_evidence_groups_overlap": 1}
            or any(row.get("status") != "selector_ineligible" for row in attempts[:3])
            or any(row.get("status") != "eligible" for row in attempts[3:])
            or pool.get("unattempted_pool_indices") != list(range(11, 16))
            or len(pool.get("eligible_rows", [])) != POOL_REQUIRED):
        raise ValueError("selector_pool_partition_or_counts_invalid")
    if (lock.get("status") != "candidate_order_frozen_geometry_not_preflighted_not_authorizing"
            or len(lock.get("candidates", [])) < 16
            or lock.get("dataset_sha256") != data_hashes
            or preflight.get("status") != "geometry_preflight_passed_not_scoring_authorization"
            or preflight.get("candidate_order_sha256") != lock_sha
            or preflight.get("dataset_sha256") != data_hashes
            or selector_smoke.get("candidate_lock_sha256") != lock_sha
            or selector_smoke.get("smoke_preflight_sha256") != preflight_sha
            or selector_smoke.get("source_sha256") != source_hashes
            or selector_smoke.get("dataset_sha256") != data_hashes):
        raise ValueError("selector_upstream_artifact_binding_invalid")
    validate_successful_smoke_receipt(selector_smoke,
        lock_sha=lock_sha, preflight_sha=preflight_sha,
        source_hashes=source_hashes, data_hashes=data_hashes)
    eligible = pool["eligible_rows"]
    if [row.get("lock_index") for row in eligible] != [7, 8, 9, 10]:
        raise ValueError("selector_eligible_order_invalid")
    return eligible


def validate_scoring_smoke_receipt(receipt: dict, *, pool_sha: str,
        source_hashes: dict[str, str], data_hashes: dict[str, str]) -> None:
    if not isinstance(receipt, dict):
        raise ValueError("successful_v2_scoring_smoke_required")
    counts = receipt.get("branch_counts")
    records = receipt.get("records")
    valid_record = False
    if isinstance(records, list) and len(records) == 1 and isinstance(records[0], dict):
        record = records[0]
        target_ids = record.get("target_token_ids")
        digest = (hashlib.sha256(json.dumps(target_ids, separators=(",", ":"))
            .encode("ascii")).hexdigest() if isinstance(target_ids, list)
            and target_ids and all(type(x) is int and x >= 0 for x in target_ids) else None)
        scores = record.get("target_score_by_condition")
        contrast = record.get("primary_contrast")
        valid_record = (record.get("lock_index") == 7
            and record.get("target_token_count") == len(target_ids or [])
            and record.get("target_token_ids_sha256") == digest
            and isinstance(scores, dict) and set(scores) == set(SCORING_BRANCHES)
            and all(isinstance(scores[name], dict)
                and scores[name].get("target_token_count") == len(target_ids or [])
                and isinstance(scores[name].get("target_logprob_mean"), (int, float))
                and math.isfinite(float(scores[name]["target_logprob_mean"]))
                and isinstance(scores[name].get("target_token_logprobs"), list)
                and len(scores[name]["target_token_logprobs"]) == len(target_ids or [])
                and all(isinstance(x, (int, float)) and math.isfinite(float(x))
                    for x in scores[name]["target_token_logprobs"])
                for name in SCORING_BRANCHES)
            and isinstance(contrast, dict)
            and all(isinstance(contrast.get(key), (int, float))
                and math.isfinite(float(contrast[key])) for key in (
                    "baseline_annotation_selective_text_dependence",
                    "qta_annotation_selective_text_dependence",
                    "delta_annotation_selective_text_dependence")))
    if (receipt.get("status") != "completed_v2_exploratory_smoke_no_faithfulness_claim"
            or receipt.get("mode") != "smoke"
            or receipt.get("selector_pool_sha256") != pool_sha
            or receipt.get("source_sha256") != source_hashes
            or receipt.get("dataset_sha256") != data_hashes
            or receipt.get("candidate_lock_indices") != [7]
            or receipt.get("effective_n") != 1
            or receipt.get("requested_n") != 1
            or receipt.get("failures") != []
            or receipt.get("generation_performed") is not True
            or receipt.get("target_scoring_performed") is not True
            or receipt.get("direct_faithfulness") is not False
            or receipt.get("faithfulness_result") is not False
            or receipt.get("claim_truth_reviewed") is not False
            or not valid_record
            or not isinstance(counts, dict)
            or set(counts) != set(SCORING_BRANCHES)
            or any(value != {"requested": 1, "attempted": 1, "completed": 1}
                   for value in counts.values())
            or receipt.get("generation_calls") != {"requested": 1, "attempted": 1, "completed": 1}):
        raise ValueError("successful_hash_bound_v2_scoring_smoke_required")


def candidate_interventions(*, mapped: dict, group_positions: tuple[tuple[int, ...], ...],
        frozen_selector_row: dict) -> dict:
    """Rebuild and validate the exact group/key sets frozen by selector pool."""
    evidence_groups = tuple(mapped["evidence_group_indices"])
    qta_groups = tuple(frozen_selector_row["qta_group_indices"])
    control_groups = tuple(frozen_selector_row["control_group_indices"])
    group_count = len(group_positions)
    if (len(set(qta_groups)) != len(qta_groups)
            or tuple(sorted(set(evidence_groups))) != evidence_groups
            or tuple(sorted(set(control_groups))) != control_groups
            or any(i < 0 or i >= group_count
                   for seq in (evidence_groups, qta_groups, control_groups) for i in seq)):
        raise ValueError("frozen_selector_group_indices_invalid")
    expected_qta = tuple(select_top_temporal_groups(
        frozen_selector_row["group_attention_scores"],
        top_k=max(1, (group_count + 9) // 10)))
    if expected_qta != qta_groups or evidence_groups != tuple(
            frozen_selector_row["evidence_group_indices"]):
        raise ValueError("frozen_selector_or_evidence_groups_changed")
    expected_control = tuple(select_qta_disjoint_rank_matched_control_groups(
        evidence_group_indices=evidence_groups, qta_group_indices=tuple(sorted(qta_groups)),
        temporal_group_count=group_count))
    if expected_control != control_groups:
        raise ValueError("frozen_rank_matched_control_groups_changed")
    evidence = tuple(mapped["evidence_token_positions"])
    qta = tuple(sorted(pos for i in qta_groups for pos in group_positions[i]))
    control = tuple(sorted(pos for i in control_groups for pos in group_positions[i]))
    budget = validate_evidence_control_key_sets(
        evidence_positions=evidence, control_positions=control)
    if (len(evidence) != len(control)
            or set(evidence) & set(qta) or set(evidence) & set(control)
            or set(qta) & set(control)
            or len(set(evidence) | set(qta) | set(control)) != len(evidence)+len(qta)+len(control)
            or len(evidence) != frozen_selector_row["evidence_key_count"]
            or len(control) != frozen_selector_row["control_key_count"]
            or len(qta) != frozen_selector_row["qta_key_count"]):
        raise ValueError("frozen_expanded_visual_key_sets_invalid")
    return {"evidence": evidence, "qta": qta, "control": control,
        "evidence_groups": evidence_groups, "qta_groups": qta_groups,
        "control_groups": control_groups, "budget": budget,
        "overlap": report_qta_key_overlap(qta_positions=qta,
            evidence_positions=evidence, control_positions=control)}


def score_candidate(*, repo: Path, candidate: dict, selector_row: dict,
        processor, model, torch, rows: dict, gsub: dict,
        branch_counts: dict, generation_calls: dict, stage: dict) -> dict:
    stage["value"] = "prepare_candidate_geometry"
    prepared = prepare_candidate_geometry(dataset=repo / DATASET_REL,
        video_root=Path.home() / "nextgqa_sample/NExTVideo", processor=processor,
        candidate=candidate, rows=rows, gsub=gsub)
    if prepared["geometry_sha256"] != selector_row["geometry_sha256"]:
        raise ValueError("geometry_changed_after_selector_receipt")
    groups = tuple(tuple(group["token_positions"]) for group in prepared["mapped"]["groups"])
    intervention = candidate_interventions(mapped=prepared["mapped"],
        group_positions=groups, frozen_selector_row=selector_row)
    prompt_batch = _move_batch_to_cuda(prepared["batch"], torch)
    prompt_length = int(prompt_batch["input_ids"].shape[1])
    stage["value"] = "generation_call"
    generation_calls["attempted"] += 1
    with torch.inference_mode():
        generated = model.generate(**prompt_batch, do_sample=False, use_cache=True,
            max_new_tokens=128, return_dict_in_generate=True)
    generation_calls["completed"] += 1
    sequence = generated.sequences[0].detach().to("cpu").tolist()
    prompt_ids = prepared["batch"]["input_ids"][0].detach().cpu().tolist()
    frozen = freeze_generated_continuation(prompt_token_ids=prompt_ids,
        generated_sequence_ids=sequence,
        eos_token_id=processor.tokenizer.eos_token_id)
    target_ids = tuple(frozen["target_token_ids"])
    if not target_ids or len(target_ids) > 128:
        raise ValueError("generated_target_token_count_invalid")
    del generated, sequence
    torch.cuda.empty_cache()
    extended_batch, target_queries = append_target_to_unpadded_batch(
        batch=prompt_batch, target_token_ids=target_ids, torch_module=torch)
    all_video_keys = tuple(pos for group in groups for pos in group)
    if max(all_video_keys) >= min(target_queries):
        raise ValueError("visual_keys_not_causal_for_all_target_queries")

    conditions = {
        "baseline_evidence_ablation": ((), intervention["evidence"]),
        "baseline_control_ablation": ((), intervention["control"]),
        "qta_evidence_ablation": (intervention["qta"], intervention["evidence"]),
        "qta_control_ablation": (intervention["qta"], intervention["control"]),
    }
    scores = {}
    for name, (biased, ablated) in conditions.items():
        stage["value"] = "score_" + name
        counter = branch_counts[name]
        counter["attempted"] += 1
        scores[name] = _forward(model, extended_batch, torch=torch,
            target_ids=target_ids, prompt_length=prompt_length,
            query_positions=target_queries, biased_positions=biased,
            ablated_positions=ablated)
        counter["completed"] += 1
    contrast = compute_text_annotation_selective_dependence(
        baseline_evidence_logprob=scores["baseline_evidence_ablation"]["target_logprob_mean"],
        baseline_control_logprob=scores["baseline_control_ablation"]["target_logprob_mean"],
        qta_evidence_logprob=scores["qta_evidence_ablation"]["target_logprob_mean"],
        qta_control_logprob=scores["qta_control_ablation"]["target_logprob_mean"])
    result = {"lock_index": selector_row["lock_index"],
        "video_sha256": candidate["video_sha256"],
        "qid_sha256": hashlib.sha256(candidate["qid"].encode()).hexdigest(),
        "geometry_sha256": selector_row["geometry_sha256"],
        "prompt_render_sha256": hashlib.sha256(prepared["rendered"].encode()).hexdigest(),
        "prompt_token_count": prompt_length,
        "target_token_ids": list(target_ids),
        "target_token_ids_sha256": frozen["target_token_ids_sha256"],
        "target_token_count": frozen["target_token_count"],
        "eos_token_id": frozen["eos_token_id"], "terminal_seen": frozen["terminal_seen"],
        "generation_config": {"do_sample": False, "use_cache": True,
            "max_new_tokens": 128, "logits_processors": "none"},
        "target_score_by_condition": scores, "primary_contrast": contrast,
        "qta_group_indices": list(intervention["qta_groups"]),
        "evidence_group_indices": list(intervention["evidence_groups"]),
        "control_group_indices": list(intervention["control_groups"]),
        "evidence_control_key_budget": intervention["budget"],
        "qta_overlap": intervention["overlap"],
        "claim_truth_reviewed": False, "direct_faithfulness": False,
        "faithfulness_result": False}
    del extended_batch, prompt_batch, prepared
    gc.collect(); torch.cuda.empty_cache()
    return result


def run(mode: str, output_name: str, repo: Path,
        scoring_smoke_name: str | None = None) -> dict:
    private = repo / PRIVATE_REL
    dataset = repo / DATASET_REL
    output = private / output_name
    prefix = "qta_generated_text_v2_scorer_"
    if (Path(output_name).name != output_name or not output_name.startswith(prefix)
            or os.path.lexists(output)):
        raise ValueError("private_output_basename_invalid_or_exists")
    pool, pool_sha = _read_private_json(private / POOL_RECEIPT_NAME, private)
    selector_smoke, selector_smoke_sha = _read_private_json(
        private / SELECTOR_SMOKE_NAME, private)
    lock, lock_sha = _read_private_json(private / LOCK_NAME, private)
    preflight, preflight_sha = _read_private_json(private / PREFLIGHT_NAME, private)
    selector_sources, _ = selector_source_inventory(repo)
    current_data = {name: sha256_file(dataset / name) for name in DATA_FILES}
    if (pool.get("source_sha256") != selector_sources
            or pool.get("dataset_sha256") != current_data
            or preflight.get("source_sha256") != {
                name: sha256_file(repo / name) for name in V1_SOURCE_FILES}):
        raise ValueError("selector_source_or_dataset_changed")
    eligible = validate_selector_pool(pool, pool_sha=pool_sha,
        lock=lock, lock_sha=lock_sha, preflight=preflight,
        preflight_sha=preflight_sha, selector_smoke=selector_smoke,
        selector_smoke_sha=selector_smoke_sha,
        source_hashes=selector_sources, data_hashes=current_data)
    current_sources = scorer_source_inventory(repo)
    if mode == "smoke":
        if scoring_smoke_name is not None:
            raise ValueError("smoke_mode_must_not_consume_scoring_receipt")
        selected = eligible[:1]
        requested = 1
        run_start_index = 7
    elif mode == "pool":
        if (not isinstance(scoring_smoke_name, str)
                or Path(scoring_smoke_name).name != scoring_smoke_name
                or not scoring_smoke_name.startswith("qta_generated_text_v2_scorer_smoke_")
                or not scoring_smoke_name.endswith(".json")):
            raise ValueError("pool_requires_private_scoring_smoke_name")
        scoring_smoke, scoring_smoke_sha = _read_private_json(
            private / scoring_smoke_name, private)
        validate_scoring_smoke_receipt(scoring_smoke, pool_sha=pool_sha,
            source_hashes=current_sources, data_hashes=current_data)
        selected = eligible[1:]
        requested = 3
        run_start_index = 8
    else:
        raise ValueError("scorer_mode_invalid")
    if [row["lock_index"] for row in selected] != list(
            range(run_start_index, run_start_index + requested)):
        raise ValueError("frozen_scoring_partition_invalid")

    started = time.monotonic()
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
    branch_counts = {name: {"requested": requested, "attempted": 0, "completed": 0}
                     for name in SCORING_BRANCHES}
    generation_calls = {"requested": requested, "attempted": 0, "completed": 0}
    candidate_map = {candidate["lock_index"]: lock["candidates"][candidate["lock_index"]]
                     for candidate in selected}
    records, failures = [], []
    for ordinal, selector_row in enumerate(selected, 1):
        stage = {"value": "prepare_candidate"}
        try:
            records.append(score_candidate(repo=repo,
                candidate=candidate_map[selector_row["lock_index"]],
                selector_row=selector_row, processor=processor, model=model,
                torch=torch, rows=rows, gsub=gsub, branch_counts=branch_counts,
                generation_calls=generation_calls, stage=stage))
        except Exception as exc:
            failures.append({"candidate_ordinal": ordinal,
                "lock_index": selector_row["lock_index"],
                "failure_type": type(exc).__name__,
                "failure_code": str(exc).splitlines()[0][:160],
                "stage": stage["value"]})
            break
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        failures.append({"candidate_ordinal": None, "lock_index": None,
            "failure_type": type(exc).__name__,
            "failure_code": str(exc).splitlines()[0][:160],
            "stage": "cuda_synchronize"})
    complete = (len(records) == requested and not failures
        and generation_calls == {"requested": requested, "attempted": requested,
                                 "completed": requested}
        and all(value == {"requested": requested, "attempted": requested,
                          "completed": requested} for value in branch_counts.values()))
    summary = None
    if complete:
        deltas = [r["primary_contrast"]["delta_annotation_selective_text_dependence"]
                  for r in records]
        summary = {"n": len(deltas), "mean": sum(deltas) / len(deltas),
            "median": sorted(deltas)[len(deltas)//2] if len(deltas) % 2 else
                (sorted(deltas)[len(deltas)//2-1] + sorted(deltas)[len(deltas)//2]) / 2,
            "range": [min(deltas), max(deltas)],
            "positive": sum(x > 0 for x in deltas),
            "negative": sum(x < 0 for x in deltas),
            "zero": sum(x == 0 for x in deltas),
            "inference": "finite_cohort_descriptive_only_no_p_value_or_population_interval"}
    payload = {"schema_version": 1,
        "status": ("completed_v2_exploratory_smoke_no_faithfulness_claim" if mode == "smoke" and complete
            else "completed_v2_exploratory_finite_cohort_no_faithfulness_claim" if complete
            else "failed_incomplete_no_cohort_estimate"),
        "scope": "QTA conditional generated-text sensitivity to gsub timing groups vs rank-matched disjoint controls; not direct faithfulness",
        "mode": mode, "rights_status": "unverified_local_only",
        "selector_pool_name": POOL_RECEIPT_NAME,
        "selector_pool_sha256": pool_sha,
        "selector_smoke_sha256": selector_smoke_sha,
        "candidate_lock_sha256": lock_sha, "preflight_sha256": preflight_sha,
        "scoring_smoke_name": scoring_smoke_name,
        "scoring_smoke_sha256": (scoring_smoke_sha if mode == "pool" else None),
        "candidate_lock_indices": [r["lock_index"] for r in selected],
        "requested_n": requested, "effective_n": len(records),
        "failures": failures,
        "generation_performed": generation_calls["completed"] > 0,
        "target_scoring_performed": sum(x["completed"] for x in branch_counts.values()) > 0,
        "generation_calls": generation_calls, "branch_counts": branch_counts,
        "requested_target_forwards": len(SCORING_BRANCHES) * requested,
        "completed_target_forwards": sum(x["completed"] for x in branch_counts.values()),
        "primary": "(L(control)-L(evidence))_QTA - (L(control)-L(evidence))_baseline on one frozen generated target",
        "records": records, "finite_cohort_primary_summary": summary,
        "direct_faithfulness": False, "faithfulness_result": False,
        "claim_truth_reviewed": False,
        "model_revision": MODEL_REVISION, "precision": "4bit_bf16_compute_bf16_eager",
        "seed": SEED, "frame_count": FRAMES,
        "qta_layers_1_based": list(LAYERS_1_BASED), "qta_logit_bias": LOGIT_BIAS,
        "checkpoint_byte_audit": checkpoint_audit,
        "processor_provenance": processor_pins,
        "hardware": {"device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "free_vram_before_model_load_gib": free / 1024**3,
            "total_vram_gib": total_vram / 1024**3,
            "host_available_before_model_load_gib": host_free / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
        "environment": {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")},
        "source_commit": source_commit, "source_sha256": current_sources,
        "dataset_sha256": current_data,
        "elapsed_seconds": time.monotonic() - started,
        "started_utc": started_utc,
        "limitations": [
            "selector eligibility conditions the cohort on model attention and annotation geometry",
            "candidate history is not certified fresh because prior exposure records are incomplete",
            "gsub is QA timing, not independently verified semantic support for a generated claim",
            "conditional likelihood changes under internal attention bias and visual-key ablations may reflect interactions or masking distribution shift",
            "finite feasibility cohort only; no inferential test or population interval",
            "this endpoint is not sentence truth, hallucination status, direct faithfulness, or an efficacy claim"]}
    _write_private_json(output, private, payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pool"), required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--scoring-smoke-name",
        help="pool mode only: completed local scoring-smoke JSON basename")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    try:
        payload = run(args.mode, args.output_name,
            Path(__file__).resolve().parents[1],
            scoring_smoke_name=args.scoring_smoke_name)
    except Exception as exc:
        parser.error(f"v2_scoring_failed:{type(exc).__name__}:{str(exc).split(':',1)[0]}")
    print(json.dumps({"status": payload["status"], "mode": args.mode,
        "requested_n": payload["requested_n"], "effective_n": payload["effective_n"],
        "candidate_lock_indices": payload["candidate_lock_indices"],
        "failures": payload["failures"],
        "completed_target_forwards": payload["completed_target_forwards"],
        "finite_cohort_primary_summary": payload["finite_cohort_primary_summary"],
        "direct_faithfulness": False,
        "private_output": args.output_name,
        "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))
    return 0 if payload["effective_n"] == payload["requested_n"] and not payload["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
