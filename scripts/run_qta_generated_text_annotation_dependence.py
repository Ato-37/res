#!/usr/bin/env python3
"""Run a local-only QTA conditional generated-text / gsub-dependence pilot.

This exploratory benchmark-annotation proxy is not a direct faithfulness score.
It generates once from baseline, freezes raw token IDs, and compares the target
mean log-likelihood under annotation/control visual-key ablations with and
without a fixed question-attention QTA bias.
"""
from __future__ import annotations

import argparse
import csv
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
    DATA_FILES, MODEL_REVISION, SOURCE_FILES, prepare_candidate_geometry,
)
from qta_multiquery_video_key_intervention import (
    aggregate_captured_question_group_attention,
    append_target_to_unpadded_batch,
    compute_text_annotation_selective_dependence,
    freeze_generated_continuation,
    install_multiquery_video_key_intervention_hooks,
    remove_hooks,
    report_qta_key_overlap,
    score_target_logits_window,
    validate_evidence_control_key_sets,
)

PRIVATE_REL = Path("data/private_nextgqa")
DATASET_REL = Path("NExT-GQA/datasets/nextgqa")
LAYERS_1_BASED = tuple(range(5, 17))
LOGIT_BIAS = 0.6931471805599453
SEED = 20261019
MIN_FREE_VRAM = 11 * 1024**3
MIN_FREE_HOST = 8 * 1024**3


def _read_private_json(path: Path, private_root: Path) -> tuple[dict, str]:
    root = private_root.resolve(strict=True)
    if stat.S_IMODE(root.stat().st_mode) != 0o700 or path.resolve(strict=True).parent != root:
        raise ValueError("private_input_root_or_path_invalid")
    info = path.stat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError("private_input_permissions_invalid")
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def load_dataset_indexes(dataset: Path) -> tuple[dict, dict]:
    """Load the question and temporal annotation indexes used by the frozen rows."""
    with (dataset / "test.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = {(str(row["video_id"]), str(row["qid"])): row
                for row in csv.DictReader(stream)}
    gsub = json.loads((dataset / "gsub_test.json").read_text(encoding="utf-8"))
    if not rows or not isinstance(gsub, dict):
        raise ValueError("dataset_question_or_annotation_index_invalid")
    return rows, gsub


def _write_private_json(path: Path, private_root: Path, payload: dict) -> None:
    root = private_root.resolve(strict=True)
    if stat.S_IMODE(root.stat().st_mode) != 0o700 or path.resolve().parent != root or os.path.lexists(path):
        raise ValueError("private_output_path_invalid_or_exists")
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _move_batch_to_cuda(batch: dict, torch) -> dict:
    return {key: value.to("cuda") if torch.is_tensor(value) else value
            for key, value in batch.items()}


def validate_scoring_partition(*, start_index: int, n: int,
        smoke_receipt: dict | None = None, preflight_sha256: str | None = None,
        source_sha256: dict | None = None) -> None:
    """Enforce disjoint smoke/descriptive partitions and a passing smoke receipt."""
    if (start_index, n) == (0, 1):
        if smoke_receipt is not None:
            raise ValueError("smoke_run_must_not_consume_prior_smoke_receipt")
        return
    if (start_index, n) != (1, 3):
        raise ValueError("scoring_partition_invalid")
    if (not isinstance(smoke_receipt, dict)
            or smoke_receipt.get("status") != "completed_exploratory_finite_cohort"
            or smoke_receipt.get("requested_n") != 1
            or smoke_receipt.get("effective_n") != 1
            or smoke_receipt.get("start_index") != 0
            or smoke_receipt.get("failures") != []
            or smoke_receipt.get("preflight_sha256") != preflight_sha256
            or smoke_receipt.get("source_sha256") != source_sha256
            or smoke_receipt.get("direct_faithfulness") is not False
            or smoke_receipt.get("faithfulness_result") is not False):
        raise ValueError("successful_same_preflight_smoke_receipt_required")


def execute_candidate_loop(*, candidates: list, expected_rows: list,
        score_one, synchronize) -> tuple[list[dict], list[dict]]:
    """Preserve candidate and final CUDA synchronization failures in the receipt."""
    records: list[dict] = []
    failures: list[dict] = []
    for ordinal, (candidate, expected) in enumerate(
            zip(candidates, expected_rows, strict=True), 1):
        stage = {"value": "score_candidate"}
        try:
            records.append(score_one(candidate, expected, stage))
        except Exception as exc:
            failures.append({"candidate_ordinal": ordinal,
                "failure_type": type(exc).__name__, "failure_code":
                    str(exc).splitlines()[0][:160], "stage": stage["value"]})
            break
    try:
        synchronize()
    except Exception as exc:
        failures.append({"candidate_ordinal": None,
            "failure_type": type(exc).__name__, "failure_code":
                str(exc).splitlines()[0][:160], "stage": "cuda_synchronize"})
    return records, failures


def scoring_cohort_complete(*, requested_n: int, records: list,
        failures: list, branch_counts: dict, auxiliary_counts: dict) -> bool:
    return (len(records) == requested_n and not failures
        and all(row["attempted"] == requested_n and row["completed"] == requested_n
                for row in branch_counts.values())
        and all(row["attempted"] == requested_n and row["completed"] == requested_n
                for row in auxiliary_counts.values()))


def tracked_call(counter: dict, function):
    """Increment attempted before a call and completed only after a return."""
    counter["attempted"] += 1
    result = function()
    counter["completed"] += 1
    return result


def _forward(model, batch: dict, *, torch, target_ids: tuple[int, ...],
             prompt_length: int, query_positions: tuple[int, ...],
             biased_positions: tuple[int, ...] = (),
             ablated_positions: tuple[int, ...] = ()) -> dict:
    handles = []
    try:
        if biased_positions or ablated_positions:
            handles = install_multiquery_video_key_intervention_hooks(
                model=model, layer_indices=tuple(layer - 1 for layer in LAYERS_1_BASED),
                biased_positions=biased_positions, ablated_positions=ablated_positions,
                query_positions=query_positions, logit_bias=LOGIT_BIAS,
                torch_module=torch)
        free, _ = torch.cuda.mem_get_info()
        if free < 2 * 1024**3:
            raise RuntimeError("per_forward_free_vram_guard_failed")
        with torch.inference_mode():
            output = model(**batch, use_cache=False, logits_to_keep=len(target_ids) + 1)
        logits = output.logits
        if (logits.ndim != 3 or tuple(logits.shape[:2]) != (1, len(target_ids) + 1)
                or not bool(torch.isfinite(logits).all())):
            raise ValueError("target_logits_window_invalid")
        scored = score_target_logits_window(logits=logits,
            target_token_ids=target_ids, prompt_length=prompt_length,
            first_query_position=prompt_length - 1, torch_module=torch)
        del output
        torch.cuda.empty_cache()
        return scored
    finally:
        remove_hooks(handles)


def selector_forward(model, batch: dict):
    """Use the tested QTA path; eager attention hooks still see per-layer weights."""
    return model(**batch, use_cache=False, logits_to_keep=1)


def _capture_selector(model, batch: dict, *, torch, question_positions: list[int],
                      groups: list[tuple[int, ...]]) -> tuple[list[float], dict]:
    from qta_attention_forward_capture import capture_qta_attention_forward_once
    from qta_attention_capture_contract import LAYERS
    input_length = int(batch["input_ids"].shape[1])
    output, rows_by_layer, capture_receipt = capture_qta_attention_forward_once(
        model=model,
        forward=lambda: selector_forward(model, batch),
        question_positions=question_positions, input_length=input_length,
        torch_module=torch)
    del output
    scores = aggregate_captured_question_group_attention(
        attention_by_layer=rows_by_layer, layer_indices=tuple(LAYERS),
        group_positions=groups, question_token_count=len(question_positions),
        torch_module=torch)
    return scores, capture_receipt


def _score_candidate(*, repo: Path, candidate: dict, expected: dict,
                     processor, model, torch, rows: dict, gsub: dict,
                     branch_counts: dict, auxiliary_counts: dict,
                     stage: dict | None = None) -> dict:
    if stage is not None:
        stage["value"] = "prepare_candidate"
    prepared = prepare_candidate_geometry(dataset=repo / DATASET_REL,
        video_root=Path.home() / "nextgqa_sample/NExTVideo", processor=processor,
        candidate=candidate, rows=rows, gsub=gsub)
    if prepared["geometry_sha256"] != expected["geometry_sha256"]:
        raise ValueError("geometry_changed_after_preflight")
    prompt_batch = _move_batch_to_cuda(prepared["batch"], torch)
    prompt_length = int(prompt_batch["input_ids"].shape[1])
    question_positions = tuple(prepared["question_positions"])
    groups = [tuple(group["token_positions"]) for group in prepared["mapped"]["groups"]]
    if stage is not None:
        stage["value"] = "selector_forward"
    group_scores, selector_receipt = tracked_call(
        auxiliary_counts["selector_forward"], lambda: _capture_selector(
            model, prompt_batch, torch=torch,
            question_positions=list(question_positions), groups=groups))
    from qwen_question_temporal_attention_selector import select_top_temporal_groups
    selected_group_indices = select_top_temporal_groups(
        group_scores, top_k=max(1, (len(groups) + 9) // 10))
    qta_positions = tuple(sorted(position for index in selected_group_indices
                                 for position in groups[index]))

    if stage is not None:
        stage["value"] = "generation_call"
    def generate_once():
        with torch.inference_mode():
            return model.generate(**prompt_batch, do_sample=False, use_cache=True,
                max_new_tokens=128, return_dict_in_generate=True)
    generated = tracked_call(auxiliary_counts["generation_call"], generate_once)
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
    evidence = tuple(prepared["mapped"]["evidence_token_positions"])
    control = tuple(prepared["mapped"]["control_token_positions"])
    budget = validate_evidence_control_key_sets(evidence_positions=evidence,
                                                control_positions=control)
    overlap = report_qta_key_overlap(qta_positions=qta_positions,
        evidence_positions=evidence, control_positions=control)
    all_video_keys = tuple(position for group in groups for position in group)
    if max(all_video_keys) >= min(target_queries):
        raise ValueError("visual_keys_not_causal_for_all_target_queries")

    conditions = {
        "baseline_full": ((), ()),
        "baseline_annotation_ablation": ((), evidence),
        "baseline_control_ablation": ((), control),
        "qta_full": (qta_positions, ()),
        "qta_annotation_ablation": (qta_positions, evidence),
        "qta_control_ablation": (qta_positions, control),
    }
    scores = {}
    for name, (biased, ablated) in conditions.items():
        if stage is not None:
            stage["value"] = "score_" + name
        scores[name] = tracked_call(branch_counts[name], lambda: _forward(
            model, extended_batch, torch=torch, target_ids=target_ids,
            prompt_length=prompt_length, query_positions=target_queries,
            biased_positions=biased, ablated_positions=ablated))
    contrast = compute_text_annotation_selective_dependence(
        baseline_evidence_logprob=scores["baseline_annotation_ablation"]["target_logprob_mean"],
        baseline_control_logprob=scores["baseline_control_ablation"]["target_logprob_mean"],
        qta_evidence_logprob=scores["qta_annotation_ablation"]["target_logprob_mean"],
        qta_control_logprob=scores["qta_control_ablation"]["target_logprob_mean"])
    result = {"video_id": candidate["video_id"], "qid": candidate["qid"],
        "video_sha256": candidate["video_sha256"],
        "geometry_sha256": expected["geometry_sha256"],
        "prompt_render_sha256": hashlib.sha256(prepared["rendered"].encode()).hexdigest(),
        "prompt_token_count": prompt_length,
        "target_token_ids": list(target_ids),
        "target_token_ids_sha256": frozen["target_token_ids_sha256"],
        "target_token_count": frozen["target_token_count"],
        "eos_token_id": frozen["eos_token_id"], "terminal_seen": frozen["terminal_seen"],
        "generation_config": {"do_sample": False, "use_cache": True,
                              "max_new_tokens": 128, "logits_processors": "none"},
        "target_score_by_condition": scores,
        "primary_contrast": contrast,
        "qta_temporal_group_scores": group_scores,
        "qta_selected_group_indices": selected_group_indices,
        "qta_selected_video_key_count": len(qta_positions),
        "evidence_group_indices": prepared["mapped"]["evidence_group_indices"],
        "control_group_indices": prepared["mapped"]["control_group_indices"],
        "evidence_control_key_budget": budget, "qta_overlap": overlap,
        "selector_capture": selector_receipt,
        "target_query_positions": list(target_queries),
        "claim_truth_reviewed": False, "direct_faithfulness": False,
        "faithfulness_result": False}
    del extended_batch, prompt_batch, prepared
    gc.collect(); torch.cuda.empty_cache()
    return result


def _summary(payload: dict, output_name: str) -> dict:
    rows = payload.get("records", [])
    values = [r["primary_contrast"]["delta_annotation_selective_text_dependence"]
              for r in rows if "primary_contrast" in r]
    return {"status": payload.get("status"), "requested_n": payload.get("requested_n"),
        "effective_n": len(rows), "failures": payload.get("failures"),
        "completed_forwards": payload.get("completed_forwards"),
        "mean_delta_annotation_selective_text_dependence": (
            sum(values) / len(values) if values else None),
        "direct_faithfulness": False, "private_output": output_name,
        "elapsed_seconds": payload.get("elapsed_seconds")}


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-name", required=True)
    parser.add_argument("--start-index", type=int, default=0,
                        help="zero-based index in the already frozen geometry receipt")
    parser.add_argument("--n", type=int, choices=(1, 3), required=True)
    parser.add_argument("--smoke-receipt-name")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    if type(args.start_index) is not int or args.start_index < 0:
        parser.error("start_index_invalid")
    if (Path(args.output_name).name != args.output_name
            or not args.output_name.startswith("qta_generated_text_annotation_dependence_local_")):
        parser.error("private_output_basename_invalid")
    repo = Path(__file__).resolve().parents[1]
    private = repo / PRIVATE_REL
    output = private / args.output_name
    if os.path.lexists(output):
        parser.error("refusing_to_overwrite_private_output")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    preflight, preflight_sha = _read_private_json(private / args.preflight_name, private)
    lock, lock_sha = _read_private_json(private / "qta_generated_text_candidate_cohort_local_20261019_01.json", private)
    dataset = repo / DATASET_REL
    if (preflight.get("status") != "geometry_preflight_passed_not_scoring_authorization"
            or preflight.get("scoring_authorized") is not False
            or preflight.get("direct_faithfulness") is not False
            or preflight.get("candidate_order_sha256") != lock_sha
            or preflight.get("source_sha256") != {name: sha256_file(repo / name) for name in SOURCE_FILES}
            or preflight.get("dataset_sha256") != {name: sha256_file(dataset / name) for name in DATA_FILES}
            or args.start_index + args.n > len(preflight.get("selected_candidates", []))):
        parser.error("preflight_receipt_or_current_source_binding_failed")
    smoke_receipt = None
    if args.n == 3:
        if not args.smoke_receipt_name:
            parser.error("successful_smoke_receipt_required_for_descriptive_run")
        smoke_receipt, _ = _read_private_json(private / args.smoke_receipt_name, private)
    elif args.smoke_receipt_name:
        parser.error("smoke_receipt_not_allowed_for_engineering_smoke")
    try:
        validate_scoring_partition(start_index=args.start_index, n=args.n,
            smoke_receipt=smoke_receipt, preflight_sha256=preflight_sha,
            source_sha256={name: sha256_file(repo / name) for name in SOURCE_FILES})
    except ValueError as exc:
        parser.error(str(exc))
    frozen_rows = preflight["selected_candidates"][args.start_index:args.start_index + args.n]
    candidate_map = {(row["video_id"], row["qid"]): row for row in lock["candidates"]}
    candidates = [candidate_map.get((row["video_id"], row["qid"])) for row in frozen_rows]
    if any(row is None for row in candidates):
        parser.error("preflight_candidate_missing_from_lock")

    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REVISION)
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA_BF16_required_no_CPU_fallback")
    free, total = torch.cuda.mem_get_info()
    host_free = next((int(line.split()[1]) * 1024 for line in
                      Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemAvailable:")), 0)
    if free < MIN_FREE_VRAM or host_free < MIN_FREE_HOST:
        parser.error("pre_model_gpu_or_host_memory_guard_failed")
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    import run_nextgqa_validation_qta_annotation_proxy as kernel
    checkpoint_audit = kernel.audit_pinned_checkpoint(repo, model_path)
    processor_pins = kernel.processor_provenance(processor, model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map="cuda", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if (int(model.config.video_token_id) != int(processor.video_token_id)
            or len(model.model.language_model.layers) < max(LAYERS_1_BASED)):
        parser.error("pinned_model_processor_contract_invalid")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()
    rows, gsub = load_dataset_indexes(dataset)
    branch_counts = {name: {"requested": args.n, "attempted": 0, "completed": 0}
        for name in ("baseline_full", "baseline_annotation_ablation", "baseline_control_ablation",
                     "qta_full", "qta_annotation_ablation", "qta_control_ablation")}
    auxiliary_counts = {name: {"requested": args.n, "attempted": 0, "completed": 0}
        for name in ("selector_forward", "generation_call")}
    def score_one(candidate, expected, stage):
        return _score_candidate(repo=repo, candidate=candidate,
                expected=expected, processor=processor, model=model, torch=torch,
                rows=rows, gsub=gsub, branch_counts=branch_counts,
                auxiliary_counts=auxiliary_counts, stage=stage)
    records, failures = execute_candidate_loop(candidates=candidates,
        expected_rows=frozen_rows, score_one=score_one,
        synchronize=torch.cuda.synchronize)
    completed_forwards = sum(row["completed"] for row in branch_counts.values())
    all_complete = scoring_cohort_complete(requested_n=args.n, records=records,
        failures=failures, branch_counts=branch_counts,
        auxiliary_counts=auxiliary_counts)
    payload = {"schema_version": 1,
        "status": "completed_exploratory_finite_cohort" if all_complete else "failed_incomplete_no_cohort_estimate",
        "scope": "QTA conditional generated-text sensitivity to NExT-GQA gsub annotation keys vs matched controls; not faithfulness",
        "prior_exposure_status": "not found in current local private JSON inventory; historical exposure unknown",
        "rights_status": "unverified_local_only",
        "direct_faithfulness": False, "faithfulness_result": False,
        "requested_n": args.n, "effective_n": len(records), "failures": failures,
        "start_index": args.start_index, "preflight_name": args.preflight_name,
        "preflight_sha256": preflight_sha, "candidate_lock_sha256": lock_sha,
        "cohort_partition": "frozen deterministic hash-order candidates; outcome-blind geometry passed",
        "requested_forward_conditions": 6 * args.n,
        "attempted_forwards": sum(row["attempted"] for row in branch_counts.values()),
        "completed_forwards": completed_forwards, "branch_counts": branch_counts,
        "auxiliary_model_calls": auxiliary_counts,
        "primary": "(mean_logp_control_ablation - mean_logp_annotation_ablation)_QTA - same_baseline",
        "records": records,
        "finite_cohort_primary_summary": None if not all_complete else {
            "n": len(records), "mean": sum(r["primary_contrast"]["delta_annotation_selective_text_dependence"] for r in records) / len(records),
            "sign_counts": {"positive": sum(r["primary_contrast"]["delta_annotation_selective_text_dependence"] > 0 for r in records),
                            "negative": sum(r["primary_contrast"]["delta_annotation_selective_text_dependence"] < 0 for r in records),
                            "zero": sum(r["primary_contrast"]["delta_annotation_selective_text_dependence"] == 0 for r in records)},
            "inference": "finite_cohort_descriptive_only_no_p_value_or_population_interval"},
        "model_revision": MODEL_REVISION, "precision": "4bit_bf16_compute_bf16_eager",
        "processor_provenance": processor_pins, "checkpoint_byte_audit": checkpoint_audit,
        "seed": SEED, "frame_count": 16, "qta_layers_1_based": list(LAYERS_1_BASED),
        "qta_logit_bias": LOGIT_BIAS,
        "hardware": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "free_vram_before_model_load_gib": free / 1024**3,
            "total_vram_gib": total / 1024**3, "host_available_before_model_load_gib": host_free / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
        "environment": {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")},
        "source_commit": source_commit,
        "source_sha256": {name: sha256_file(repo / name) for name in SOURCE_FILES},
        "data_sha256": {name: sha256_file(dataset / name) for name in DATA_FILES},
        "started_utc": started_utc,
        "elapsed_seconds": time.monotonic() - started,
        "limitations": ["gsub is a QA interval annotation, not independently verified support for each generated sentence",
            "historical prior exposure is unknown even though matching local hashes were excluded",
            "this measures conditional text likelihood sensitivity, not sentence truth, hallucination, or direct faithfulness",
            "n is a feasibility-only finite cohort with no population interval or inferential test",
            "attention key ablation and QTA logit bias are internal interventions with possible interaction; overlapping QTA and gsub keys are reported and ablation takes precedence" ]}
    try:
        _write_private_json(output, private, payload)
    except Exception as exc:
        parser.error(f"private_result_write_failed:{type(exc).__name__}")
    print(json.dumps(_summary(payload, args.output_name), sort_keys=True))
    return 0 if all_complete else 2


if __name__ == "__main__":
    raise SystemExit(run())
