#!/usr/bin/env python3
"""Local exploratory hash-sham control for the fixed-prediction QTA ablation.

Adds only sham-bias evidence/control branches to the frozen three-row cohort.
The sham groups match the QTA groups' overlap counts with evidence/control
windows, exclude the QTA group set itself, then are selected deterministically
by SHA-256 from the eligible sets.
This is a post-hoc falsification diagnostic, not confirmatory evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import stat
import time

from private_output_paths import validate_private_output
import run_qta_fixed_prediction_key_ablation as qta
from qta_multiquery_video_key_intervention import (
    install_multiquery_video_key_intervention_hooks, remove_hooks,
)
from qwen_question_temporal_attention_selector import select_hash_sham_temporal_groups

ANCHOR = "data/private_nextgqa/qta_fixed_prediction_key_ablation_cohort12_20261021_03.json"
SHAM_SEED = "qta-overlap-matched-sham-v1-20261021"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> tuple[dict, str]:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_input_permissions_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("private_input_object_required")
    return value, hashlib.sha256(raw).hexdigest()


def _write_new_json(path: Path, value: dict) -> None:
    """Create a mode-0600 receipt without following or overwriting a path."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def choose_overlap_matched_sham(*, temporal_group_count: int,
                                evidence_groups: tuple[int, ...],
                                control_groups: tuple[int, ...],
                                qta_groups: tuple[int, ...],
                                opaque_row_token: str) -> tuple[int, ...]:
    """Hash-rank combinations matching QTA/evidence/control overlap counts."""
    if (type(temporal_group_count) is not int or temporal_group_count < 1
            or not opaque_row_token):
        raise ValueError("sham_selection_inputs_invalid")
    size = len(qta_groups)
    qta_set, evidence, control = set(qta_groups), set(evidence_groups), set(control_groups)
    if (size < 1 or len(qta_set) != size
            or any(type(x) is not int or not 0 <= x < temporal_group_count
                   for x in qta_set | evidence | control)):
        raise ValueError("sham_selection_groups_invalid")
    target_e = len(qta_set & evidence)
    target_c = len(qta_set & control)
    candidates = [c for c in itertools.combinations(range(temporal_group_count), size)
                  if len(set(c) & evidence) == target_e
                  and len(set(c) & control) == target_c
                  and set(c) != qta_set]
    if not candidates:
        raise ValueError("overlap_matched_sham_unavailable")
    prefix = f"{SHAM_SEED}\0{opaque_row_token}\0"
    return min(candidates, key=lambda c: (
        hashlib.sha256((prefix + ",".join(map(str, c))).encode()).digest(), c))


def _score_vector(output, answer_ids, torch):
    logits = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
    values = logits[torch.tensor(answer_ids, device=logits.device)].detach().cpu().tolist()
    if len(values) != 5 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("sham_choice_logprob_invalid")
    return [float(x) for x in values]


def _score_sham_row(row, anchor_row, model, torch, counters, sham_groups):
    if torch.cuda.mem_get_info()[0] < 2 * 1024**3 or qta._host_available_bytes() < 3 * 1024**3:
        raise RuntimeError("sham_pretransfer_resource_guard_failed")
    batch = row["batch"].to("cuda")
    if (qta.batch_sha256(row["batch"]) != row["batch_sha256"]
            or qta.batch_sha256(batch) != row["batch_sha256"]):
        raise ValueError("sham_frozen_batch_mismatch")
    seq_len = int(batch["input_ids"].shape[1])
    groups = row["mapping"]["temporal_groups"]
    sham_keys = tuple(sorted(p for idx in sham_groups for p in groups[idx]))
    evidence = tuple(row["mapping"]["evidence_token_positions"])
    control = tuple(row["mapping"]["control_token_positions"])
    labels = (anchor_row["primary"]["fixed_prediction_index"],
              anchor_row["primary"]["fixed_foil_index"])
    if labels[0] == labels[1]:
        raise ValueError("anchor_fixed_prediction_foil_invalid")
    outputs = {}
    details = {}
    for name, ablation in (("sham_evidence_ablation", evidence),
                           ("sham_control_ablation", control)):
        if torch.cuda.mem_get_info()[0] < 2 * 1024**3 or qta._host_available_bytes() < 3 * 1024**3:
            raise RuntimeError("sham_per_forward_resource_guard_failed")
        if qta.batch_sha256(batch) != row["batch_sha256"]:
            raise ValueError("sham_batch_changed_before_forward")
        handles = install_multiquery_video_key_intervention_hooks(
            model=model, layer_indices=tuple(i - 1 for i in qta.LAYERS),
            biased_positions=sham_keys, ablated_positions=ablation,
            query_positions=(seq_len - 1,), logit_bias=qta.LOGIT_BIAS,
            torch_module=torch)
        observed = {i: 0 for i in qta.LAYERS}
        observers = []
        try:
            layers = model.model.language_model.layers
            for layer_i in qta.LAYERS:
                module = layers[layer_i - 1].self_attn
                def observe(_module, args, kwargs, *, layer=layer_i):
                    mask = kwargs.get("attention_mask")
                    if not torch.is_tensor(mask) or mask.ndim != 4:
                        raise ValueError("sham_observer_mask_invalid")
                    target = mask[0, 0, seq_len - 1]
                    if any(float(target[p].item()) != float("-inf") for p in ablation):
                        raise ValueError("sham_ablation_mask_value_mismatch")
                    representable = qta._mask_representable_bias(torch, mask.dtype, qta.LOGIT_BIAS)
                    for p in sham_keys:
                        if p not in ablation and not math.isclose(float(target[p].item()),
                                representable, rel_tol=0, abs_tol=1e-6):
                            raise ValueError("sham_bias_mask_value_mismatch")
                    observed[layer] += 1
                    return args, kwargs
                observers.append(module.register_forward_pre_hook(observe, with_kwargs=True))
            counters["attempted"] += 1
            with torch.inference_mode():
                output = model(**batch, use_cache=False, logits_to_keep=1)
            counters["completed"] += 1
            outputs[name] = _score_vector(output, row["answer_ids"], torch)
            if observed != {i: 1 for i in qta.LAYERS}:
                raise ValueError("sham_observer_layer_count_mismatch")
            details[name] = {"hook_calls_by_layer": {str(k): v for k, v in observed.items()},
                             "bias_key_count": len(sham_keys),
                             "ablation_key_count": len(ablation)}
            del output
        finally:
            for handle in reversed(observers):
                handle.remove()
            remove_hooks(handles)
        torch.cuda.empty_cache()
    pred, foil = labels
    margin = lambda vector: float(vector[pred]) - float(vector[foil])
    f_sham = margin(outputs["sham_control_ablation"]) - margin(outputs["sham_evidence_ablation"])
    base_branches = anchor_row["branch_choice_logprobs_nats"]
    f_base = (margin(base_branches["baseline_control_ablation"])
              - margin(base_branches["baseline_evidence_ablation"]))
    f_qta = anchor_row["primary"]["f_qta_nats"]
    return {"row_token_sha256": anchor_row["row_token_sha256"],
            "source_video_sha256": anchor_row["source_video_sha256"],
            "input_batch_sha256": row["batch_sha256"],
            "qta_groups": anchor_row["question_attention_group_indices"],
            "sham_groups": list(sham_groups),
            "qta_overlap_counts": {"evidence": len(set(anchor_row["question_attention_group_indices"])
                                                        & set(anchor_row["evidence_group_indices"])),
                                   "control": len(set(anchor_row["question_attention_group_indices"])
                                                       & set(anchor_row["control_group_indices"]))},
            "sham_branch_logprobs_nats": outputs, "fixed_prediction_index": pred,
            "fixed_foil_index": foil, "f_baseline_nats": f_base,
            "f_qta_nats": f_qta, "f_sham_nats": f_sham,
            "delta_f_qta_nats": f_qta - f_base,
            "delta_f_sham_nats": f_sham - f_base,
            "qta_minus_sham_specificity_nats": f_qta - f_sham,
            "forward_count": 2, "calls": details}


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--anchor-name", default=ANCHOR)
    parser.add_argument("--selection-receipt", default=qta.SELECTION)
    parser.add_argument("--cohort-lock", default=qta.LOCK)
    parser.add_argument("--cohort-pairs", default=qta.PAIRS)
    parser.add_argument("--claim-packet", action="append")
    parser.add_argument("--smoke-first-record", action="store_true",
        help="score one row (2 forwards) for path validation only; no cohort estimate")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    repo = Path(__file__).resolve().parents[1]
    output_path = validate_private_output(args.output_name, repo)
    private = repo / "data/private_nextgqa"
    if output_path.parent != private or output_path.suffix != ".json":
        raise ValueError("output_path_must_be_private_json")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    attempted = completed = 0
    counters = {"attempted": 0, "completed": 0}
    stage = "preflight"
    try:
        anchor_relative = qta._private_input_path(repo, args.anchor_name)
        selection_path = qta._private_input_path(repo, args.selection_receipt)
        lock_path = qta._private_input_path(repo, args.cohort_lock)
        pairs_path = qta._private_input_path(repo, args.cohort_pairs)
        packet_paths = [qta._private_input_path(repo, item) for item in
            ([qta.PACKETS[0]] if args.claim_packet is None else args.claim_packet)]
        anchor, anchor_hash = _read_json(repo / anchor_relative)
        if (anchor.get("status") != "completed_exploratory_internal_key_ablation"
                or anchor.get("effective_records") != 3
                or anchor.get("completed_forwards") != 15
                or len(anchor.get("records", [])) != 3
                or anchor.get("direct_faithfulness") is not False
                or anchor.get("model_revision") != qta.MODEL_REVISION
                or anchor.get("model") != "Qwen/Qwen2.5-VL-7B-Instruct"
                or anchor.get("precision") != "4-bit weights; BF16 compute; eager attention"):
            raise ValueError("sham_anchor_incomplete_or_invalid")
        for rel, expected in anchor.get("source_sha256", {}).items():
            if _sha(repo / rel) != expected:
                raise ValueError("sham_anchor_source_hash_mismatch")
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("sham_cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info()
        host_free = qta._host_available_bytes()
        if free < qta.MIN_FREE_VRAM or host_free < qta.MIN_HOST_AVAILABLE:
            raise RuntimeError("sham_pre_model_resource_guard_failed")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  Qwen2_5_VLForConditionalGeneration)
        from qwen_vl_utils import process_vision_info
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / qta.MODEL_REVISION)
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        rows, input_hashes = qta._prepare_rows(repo, processor, torch, process_vision_info,
            selection_path, lock_path, pairs_path, packet_paths)
        if len(rows) != 3:
            raise ValueError("sham_preflight_record_count_invalid")
        for relative in ("NExT-GQA/datasets/nextgqa/test.csv",
                         "NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"):
            input_hashes[relative] = _sha(repo / relative)
        if any(_sha(repo / rel) != expected
               for rel, expected in anchor.get("input_sha256", {}).items()
               if rel != anchor_relative):
            raise ValueError("sham_anchor_input_hash_mismatch")
        sham_sets = []
        for i, (row, anchor_row) in enumerate(zip(rows, anchor["records"], strict=True)):
            if (anchor_row.get("row_token_sha256") != row["binding"]["row_token_sha256"]
                    or anchor_row.get("source_video_sha256") != row["binding"]["source_video_sha256"]
                    or anchor_row.get("input_batch_sha256") != row["batch_sha256"]
                    or anchor_row.get("choice_token_binding_sha256")
                        != qta._canonical_hash(row["choice_binding"])
                    or anchor_row.get("selected_window_frame_indices") != row["mapping"]["evidence_frame_window"]
                    or anchor_row.get("evidence_group_indices") != row["mapping"]["evidence_group_indices"]
                    or anchor_row.get("control_group_indices") != row["mapping"]["control_group_indices"]):
                raise ValueError("sham_anchor_row_binding_mismatch")
            sham = choose_overlap_matched_sham(
                temporal_group_count=len(row["mapping"]["temporal_groups"]),
                evidence_groups=tuple(anchor_row["evidence_group_indices"]),
                control_groups=tuple(anchor_row["control_group_indices"]),
                qta_groups=tuple(anchor_row["question_attention_group_indices"]),
                opaque_row_token=anchor_row["row_token_sha256"])
            sham_sets.append(sham)
        stage = "load_model"
        free, total = torch.cuda.mem_get_info(); host_free = qta._host_available_bytes()
        if free < qta.MIN_FREE_VRAM or host_free < qta.MIN_HOST_AVAILABLE:
            raise RuntimeError("sham_pre_model_resource_guard_failed")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("sham_model_processor_video_token_mismatch")
        torch.manual_seed(20261021); torch.cuda.manual_seed_all(20261021)
        torch.cuda.reset_peak_memory_stats()
        stage = "score_sham_branches"
        records = []
        scored_rows = zip(rows, anchor["records"], sham_sets, strict=True)
        if args.smoke_first_record:
            scored_rows = zip(rows[:1], anchor["records"][:1], sham_sets[:1], strict=True)
        for row, anchor_row, sham in scored_rows:
            before = dict(counters)
            records.append(_score_sham_row(row, anchor_row, model, torch, counters, sham))
            records[-1]["forward_count"] = {"attempted": counters["attempted"]-before["attempted"],
                                               "completed": counters["completed"]-before["completed"]}
        expected_rows = 1 if args.smoke_first_record else 3
        expected_forwards = 2 * expected_rows
        if (len(records) != expected_rows or counters != {
                "attempted": expected_forwards, "completed": expected_forwards}):
            raise RuntimeError("sham_cohort_incomplete")
        del model
        import gc
        gc.collect(); torch.cuda.empty_cache()
        contrasts = [r["qta_minus_sham_specificity_nats"] for r in records]
        source_files = [Path(__file__).resolve(), Path(qta.__file__).resolve(),
                        repo / "scripts/private_output_paths.py",
                        repo / "scripts/qta_attention_forward_capture.py",
                        repo / "scripts/qta_choice_token_binding.py",
                        repo / "scripts/qta_multiquery_video_key_intervention.py",
                        repo / "scripts/qta_processor_choice_suffix.py",
                        repo / "scripts/qwen_question_temporal_attention_selector.py",
                        repo / "scripts/qwen_frozen_frame_window_groups.py",
                        repo / "scripts/run_machine_localized_claim_mask_pilot.py",
                        repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py",
                        repo / "scripts/run_nextgqa_real_video_gpu_smoke.py",
                        repo / "scripts/claim_mask_video_decoder.py",
                        repo / "scripts/nextgqa_atomic_batch_freeze.py"]
        source_hashes = {str(p.relative_to(repo)): _sha(p) for p in source_files}
        payload = {"schema_version": 1,
            "status": ("partial_overlap_matched_sham_smoke" if args.smoke_first_record
                else "completed_exploratory_overlap_matched_sham_ablation"),
            "started_utc": started_utc,
            "anchor": anchor_relative, "anchor_sha256": anchor_hash,
            "cohort": anchor["cohort"], "cohort_reuse": anchor.get("cohort_reuse"),
            "requested_records": expected_rows, "effective_records": len(records),
            "requested_forwards": expected_forwards, "attempted_forwards": expected_forwards,
            "completed_forwards": expected_forwards,
            "human_review": {"collection_waived_by_user": True,
                              "review_actually_performed": False},
            "rights_status": "unverified; local-only",
            "sham_protocol": {"seed": SHAM_SEED,
                "selection": "SHA-256-ranked combinations matching QTA group's counts of overlap with evidence and control windows",
                "selection_scope": "post-hoc diagnostic on the same three sequentially analyzed rows",
                "bias": qta.LOGIT_BIAS, "layers_1_based": list(qta.LAYERS)},
            "input_sha256": {**anchor.get("input_sha256", {}), anchor_relative: anchor_hash},
            "source_sha256": source_hashes, "records": records,
            "finite_cohort_summary": (None if args.smoke_first_record else {"n": len(contrasts),
                "mean_qta_minus_sham_specificity_nats": sum(contrasts)/len(contrasts),
                "median_qta_minus_sham_specificity_nats": sorted(contrasts)[1],
                "range_qta_minus_sham_specificity_nats": [min(contrasts), max(contrasts)],
                "positive_rows": sum(x > 0 for x in contrasts),
                "interpretation": "finite-cohort exploratory contrast only; no population inference"}),
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": qta.MODEL_REVISION, "precision": "4-bit weights; BF16 compute; eager attention",
            "library_versions": {name: __import__("importlib.metadata", fromlist=["version"]).version(name)
                for name in ("torch", "transformers", "bitsandbytes", "qwen-vl-utils")},
            "hardware": {"device": torch.cuda.get_device_name(0),
                "free_vram_before_load_gib": free/1024**3, "total_vram_gib": total/1024**3,
                "host_available_before_load_gib": host_free/1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3},
            "elapsed_seconds": time.monotonic()-started,
            "direct_faithfulness": False,
            "limitations": ["three post-hoc sequentially analyzed rows",
                "sham overlap constraints use post-hoc QTA-window geometry",
                "global visual-key bias; no claim-specific or human-reviewed evidence",
                "no CI, significance test, or population inference",
                "human review waived for this local experiment but not performed"]}
        _write_new_json(output_path, payload)
        print(json.dumps({"status": payload["status"], "records": len(records),
            "forwards": expected_forwards,
            "mean_qta_minus_sham_specificity_nats": None if args.smoke_first_record else sum(contrasts)/len(contrasts),
            "output_name": output_path.name}))
        return 0
    except Exception as exc:
        failure = {"schema_version": 1, "status": "failed_exploratory_overlap_matched_sham_ablation",
            "stage": stage, "failure_type": type(exc).__name__, "failure_code": str(exc),
            "requested_records": 1 if args.smoke_first_record else 3, "effective_records": 0,
            "requested_forwards": 2 if args.smoke_first_record else 6,
            "attempted_forwards": counters["attempted"], "completed_forwards": counters["completed"],
            "direct_faithfulness": False, "human_review_performed": False,
            "rights_status": "unverified; local-only"}
        _write_new_json(output_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(run())
