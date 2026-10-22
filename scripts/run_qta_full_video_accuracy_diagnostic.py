#!/usr/bin/env python3
"""Exploratory full-video answer-choice diagnostic for QTA and matched sham.

Reuses nine already-frozen, three-row local cohorts. For each video, one
unmodified forward captures question attention and freezes the baseline
prediction; one full-video forward applies the fixed top-10% temporal-group
QTA bias, and one applies an equal-budget hash-selected sham bias. This is a
post-hoc QA-accuracy diagnostic, not a direct explanation-faithfulness test.
All source videos, prompts, answers, and row-level scores remain local.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import unicodedata
import stat
import time

from private_output_paths import validate_private_output
import run_qta_fixed_prediction_key_ablation as qta
from qta_attention_forward_capture import (
    aggregate_captured_qta_rows, capture_qta_attention_forward_once,
)
from qta_multiquery_video_key_intervention import (
    install_multiquery_video_key_intervention_hooks, remove_hooks,
)
from qwen_question_temporal_attention_selector import (
    aggregate_temporal_group_attention, select_top_temporal_groups,
)
from run_qta_hash_matched_sham_ablation import choose_overlap_matched_sham

LAYERS = tuple(range(5, 17))
BIAS = math.log(2.0)
ANCHORS = [
    f"data/private_nextgqa/qta_fixed_prediction_key_ablation_cohort{n}.json"
    for n in (16, 17, 18, 20, 21, 22, 23, 24, 26)
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_private(repo: Path, relative: str) -> tuple[dict, str]:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("private_input_path_invalid")
    path = repo / candidate
    private_root = (repo / "data/private_nextgqa").resolve(strict=True)
    try:
        path.resolve(strict=True).relative_to(private_root)
    except (OSError, ValueError) as exc:
        raise ValueError("private_input_path_invalid") from exc
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_input_permissions_invalid")
        blocks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
    finally:
        os.close(fd)
    raw = b"".join(blocks)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("private_json_object_required")
    return value, hashlib.sha256(raw).hexdigest()


def _write_private_exclusive(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _gold_index(row: dict) -> int:
    norm = lambda value: " ".join(unicodedata.normalize(
        "NFKC", str(value)).strip().casefold().split())
    gold = norm(row["answer"])
    matches = [i for i in range(5) if norm(row[f"a{i}"]) == gold]
    if len(matches) != 1:
        raise ValueError("gold_choice_join_invalid")
    return matches[0]


def _argmax(values: list[float]) -> int:
    if len(values) != 5 or any(not math.isfinite(float(x)) for x in values):
        raise ValueError("choice_vector_invalid")
    return max(range(5), key=lambda i: (float(values[i]), -i))


def summarize_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError("records_required")
    n = len(records)
    result = {}
    for condition in ("baseline", "qta", "sham"):
        correct = sum(int(row["conditions"][condition]["gold_choice_correct"])
                      for row in records)
        result[condition] = {"correct": correct, "n": n, "accuracy": correct / n}
    result["qta_minus_baseline_correct"] = (
        result["qta"]["correct"] - result["baseline"]["correct"])
    result["qta_minus_sham_correct"] = result["qta"]["correct"] - result["sham"]["correct"]
    result["baseline_to_qta_flips"] = {
        "wrong_to_correct": sum(not r["conditions"]["baseline"]["gold_choice_correct"]
                                and r["conditions"]["qta"]["gold_choice_correct"] for r in records),
        "correct_to_wrong": sum(r["conditions"]["baseline"]["gold_choice_correct"]
                                and not r["conditions"]["qta"]["gold_choice_correct"] for r in records),
    }
    result["qta_to_sham_flips"] = {
        "sham_wrong_qta_correct": sum(not r["conditions"]["sham"]["gold_choice_correct"]
                                      and r["conditions"]["qta"]["gold_choice_correct"] for r in records),
        "sham_correct_qta_wrong": sum(r["conditions"]["sham"]["gold_choice_correct"]
                                      and not r["conditions"]["qta"]["gold_choice_correct"] for r in records),
    }
    result["interpretation"] = (
        "post-hoc finite-cohort QA-accuracy diagnostic only; no population or "
        "direct-faithfulness inference")
    return result


def _choice_logprobs(logits, answer_ids, torch) -> list[float]:
    vector = torch.log_softmax(logits[0, -1].float(), dim=-1)
    ids = torch.tensor(answer_ids, device=vector.device)
    values = vector[ids].detach().cpu().tolist()
    if len(values) != 5 or any(not math.isfinite(float(x)) for x in values):
        raise ValueError("choice_logprobs_invalid")
    return [float(x) for x in values]


def _intervened_forward(*, model, batch, answer_ids, positions, sequence_len,
                        torch, counters, expected_batch_hash: str,
                        cpu_batch, name: str) -> tuple[list[float], dict]:
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 2 * 1024**3 or qta._host_available_bytes() < 3 * 1024**3:
        raise RuntimeError("per_forward_resource_guard_failed")
    if (qta.batch_sha256(cpu_batch) != expected_batch_hash
            or qta.batch_sha256(batch) != expected_batch_hash):
        raise ValueError("frozen_batch_changed_before_intervention_forward")
    handles = install_multiquery_video_key_intervention_hooks(
        model=model, layer_indices=tuple(i - 1 for i in LAYERS),
        biased_positions=positions, ablated_positions=(),
        query_positions=(sequence_len - 1,), logit_bias=BIAS,
        torch_module=torch)
    calls = {layer: 0 for layer in LAYERS}
    observers = []
    try:
        layers = model.model.language_model.layers
        for layer_i in LAYERS:
            module = layers[layer_i - 1].self_attn
            def observe(_module, args, kwargs, *, layer=layer_i):
                mask = kwargs.get("attention_mask")
                if not torch.is_tensor(mask) or mask.ndim != 4:
                    raise ValueError("intervention_observer_mask_invalid")
                expected = float(torch.tensor(BIAS, dtype=mask.dtype).item())
                row = mask[0, 0, sequence_len - 1]
                if any(not math.isclose(float(row[p].item()), expected,
                                        rel_tol=0, abs_tol=1e-6) for p in positions):
                    raise ValueError("intervention_mask_value_mismatch")
                calls[layer] += 1
                return args, kwargs
            observers.append(module.register_forward_pre_hook(observe, with_kwargs=True))
        counters["attempted"] += 1
        with torch.inference_mode():
            output = model(**batch, use_cache=False, logits_to_keep=1)
        counters["completed"] += 1
        if calls != {layer: 1 for layer in LAYERS}:
            raise ValueError("intervention_layer_call_count_mismatch")
        scores = _choice_logprobs(output.logits, answer_ids, torch)
        del output
        return scores, {"hook_calls_by_layer": {str(k): v for k, v in calls.items()},
                        "biased_key_count": len(positions),
                        "effective_mask_bias": float(torch.tensor(BIAS, dtype=torch.bfloat16).item())}
    finally:
        for handle in reversed(observers):
            handle.remove()
        remove_hooks(handles)


def _score_row(prepared: dict, anchor_record: dict, model, torch,
               counters: dict, cohort: str, row_index: int) -> dict:
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 2 * 1024**3 or qta._host_available_bytes() < 3 * 1024**3:
        raise RuntimeError("per_forward_resource_guard_failed")
    batch = prepared["batch"].to("cuda")
    sequence_len = int(batch["input_ids"].shape[1])
    if (qta.batch_sha256(prepared["batch"]) != prepared["batch_sha256"]
            or qta.batch_sha256(batch) != prepared["batch_sha256"]):
        raise ValueError("frozen_batch_mismatch_before_baseline")
    groups = [tuple(group) for group in prepared["mapping"]["temporal_groups"]]
    counters["attempted"] += 1
    def baseline_forward():
        with torch.inference_mode():
            output = model(**batch, use_cache=False, logits_to_keep=1)
            counters["completed"] += 1
            return output
    output, attention, capture = capture_qta_attention_forward_once(
        model=model, forward=baseline_forward, question_positions=prepared["qpos"],
        input_length=sequence_len, torch_module=torch)
    answer_ids = prepared["answer_ids"]
    baseline_scores = _choice_logprobs(output.logits, answer_ids, torch)
    prior_full = anchor_record["full_choice_logprobs_nats"]
    if max(abs(float(a) - float(b)) for a, b in
           zip(baseline_scores, prior_full, strict=True)) > 1e-4:
        raise ValueError("current_full_video_baseline_drifted_from_anchor")
    group_scores = aggregate_captured_qta_rows(rows=attention,
        group_positions=groups, torch_module=torch)
    k = max(1, math.ceil(len(groups) * .10))
    qta_groups = tuple(sorted(select_top_temporal_groups(group_scores, top_k=k)))
    if list(qta_groups) != anchor_record["question_attention_group_indices"]:
        raise ValueError("recomputed_qta_selector_differs_from_anchor")
    row_token = prepared["binding"]["row_token_sha256"]
    evidence = tuple(prepared["mapping"]["evidence_group_indices"])
    control = tuple(prepared["mapping"]["control_group_indices"])
    sham_groups = choose_overlap_matched_sham(temporal_group_count=len(groups),
        evidence_groups=evidence, control_groups=control, qta_groups=qta_groups,
        opaque_row_token=row_token)
    qta_keys = tuple(sorted(p for idx in qta_groups for p in groups[idx]))
    sham_keys = tuple(sorted(p for idx in sham_groups for p in groups[idx]))
    del output, attention
    torch.cuda.empty_cache()
    qta_scores, qta_audit = _intervened_forward(model=model, batch=batch,
        answer_ids=answer_ids,
        positions=qta_keys, sequence_len=sequence_len, torch=torch,
        counters=counters, expected_batch_hash=prepared["batch_sha256"],
        cpu_batch=prepared["batch"], name="qta")
    sham_scores, sham_audit = _intervened_forward(model=model, batch=batch,
        answer_ids=answer_ids,
        positions=sham_keys, sequence_len=sequence_len, torch=torch,
        counters=counters, expected_batch_hash=prepared["batch_sha256"],
        cpu_batch=prepared["batch"], name="sham")
    gold = _gold_index(prepared["row"])
    conditions = {}
    for name, scores in (("baseline", baseline_scores), ("qta", qta_scores),
                         ("sham", sham_scores)):
        prediction = _argmax(scores)
        conditions[name] = {"prediction_index": prediction,
            "gold_choice_correct": prediction == gold,
            "gold_logprob_nats": scores[gold],
            "gold_minus_best_foil_margin_nats": scores[gold] - max(
                score for i, score in enumerate(scores) if i != gold),
            "choice_logprobs_nats": scores}
    return {"cohort": cohort, "row_position": row_index,
        "row_token_sha256": row_token,
        "source_video_sha256": prepared["binding"]["source_video_sha256"],
        "input_batch_sha256": prepared["batch_sha256"],
        "gold_choice_index": gold,
        "qta_group_indices": list(qta_groups),
        "sham_group_indices": list(sham_groups),
        "qta_group_attention_scores": [group_scores[i] for i in qta_groups],
        "conditions": {"baseline": conditions["baseline"],
            "qta": {**conditions["qta"], "mask_audit": qta_audit},
            "sham": {**conditions["sham"], "mask_audit": sham_audit}},
        "baseline_attention_capture": capture,
        "forwards": 3,
        "direct_faithfulness": False}


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--anchor", action="append", default=None,
        help="owner-only completed QTA receipt; repeat for all nine frozen cohorts")
    parser.add_argument("--smoke-first-row", action="store_true")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    repo = Path(__file__).resolve().parents[1]
    output = validate_private_output(args.output_name, repo)
    if output.parent != repo / "data/private_nextgqa" or output.suffix != ".json":
        raise ValueError("output_path_must_be_private_json")
    anchors = args.anchor or ANCHORS
    if len(anchors) != 9 and not args.smoke_first_row:
        raise ValueError("exactly_nine_frozen_anchors_required")
    if len(anchors) == 1 and not args.smoke_first_row:
        raise ValueError("single_anchor_requires_smoke_mode")
    if len(set(anchors)) != len(anchors):
        raise ValueError("duplicate_anchor_receipt")
    started = time.monotonic()
    anchor_values = []
    for path in anchors:
        value, digest = _read_private(repo, path)
        if (value.get("status") != "completed_exploratory_internal_key_ablation"
                or value.get("effective_records") != 3
                or value.get("completed_forwards") != 15
                or len(value.get("records", [])) != 3):
            raise ValueError("anchor_receipt_incomplete")
        for item_path, expected in value.get("input_sha256", {}).items():
            candidate = Path(item_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("anchor_input_path_invalid")
            resolved = (repo / candidate).resolve(strict=True)
            try:
                resolved.relative_to(repo.resolve())
            except ValueError as exc:
                raise ValueError("anchor_input_path_invalid") from exc
            if _sha256(resolved) != expected:
                raise ValueError("anchor_input_hash_mismatch")
        for source_path, expected in value.get("source_sha256", {}).items():
            candidate = Path(source_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("anchor_source_path_invalid")
            source = (repo / candidate).resolve(strict=True)
            try:
                source.relative_to(repo.resolve())
            except ValueError as exc:
                raise ValueError("anchor_source_path_invalid") from exc
            if not source.is_file() or _sha256(source) != expected:
                raise ValueError("anchor_source_hash_mismatch")
        anchor_values.append((path, value, digest))
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("cuda_bf16_unavailable")
    free, total_vram = torch.cuda.mem_get_info()
    if free < qta.MIN_FREE_VRAM or qta._host_available_bytes() < qta.MIN_HOST_AVAILABLE:
        raise RuntimeError("pre_model_resource_guard_failed")
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
        / "snapshots" / qta.MODEL_REVISION)
    if not model_path.is_dir():
        raise ValueError("pinned_model_checkpoint_missing")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if int(model.config.video_token_id) != int(processor.video_token_id):
        raise ValueError("model_processor_video_token_mismatch")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "gpu_name": torch.cuda.get_device_name(0),
        "initial_free_vram_bytes": int(free),
        "total_vram_bytes": int(total_vram),
        "host_available_bytes_before_model_load": qta._host_available_bytes(),
    }
    torch.manual_seed(20261021); torch.cuda.manual_seed_all(20261021)
    torch.cuda.reset_peak_memory_stats()
    counters = {"attempted": 0, "completed": 0}
    records = []
    source_files = [Path(__file__).resolve(), Path(qta.__file__).resolve(),
        repo / "scripts/qta_multiquery_video_key_intervention.py",
        repo / "scripts/qta_attention_forward_capture.py",
        repo / "scripts/qta_attention_capture_contract.py",
        repo / "scripts/qta_attention_capture_sidecars.py",
        repo / "scripts/qwen_question_temporal_attention_selector.py",
        repo / "scripts/run_qta_hash_matched_sham_ablation.py",
        repo / "scripts/claim_mask_video_decoder.py"]
    source_hashes = {p.relative_to(repo).as_posix(): _sha256(p) for p in source_files}
    for anchor_path, anchor, anchor_hash in anchor_values:
        inputs = anchor["input_sha256"]
        private_paths = [p for p in inputs if p.startswith("data/private_nextgqa/")]
        selection = next(p for p in private_paths if "window_selection" in p)
        lock = next(p for p in private_paths if "window_test_lock" in p)
        pairs = next(p for p in private_paths if "window_test_pairs" in p)
        packet = next(p for p in private_paths if "claims_remaining" in p)
        prior = next(p for p in private_paths if "selected_window_qa_margin" in p)
        rows, hashes = qta._prepare_rows(repo, processor, torch, process_vision_info,
            selection, lock, pairs, [packet])
        if len(rows) != 3:
            raise ValueError("anchored_row_preflight_invalid")
        row_positions = [0] if args.smoke_first_row and not records else range(3)
        for row_pos in row_positions:
            prepared = rows[row_pos]
            anchor_record = anchor["records"][row_pos]
            if (anchor_record["source_video_sha256"] != prepared["binding"]["source_video_sha256"]
                    or anchor_record["input_batch_sha256"] != prepared["batch_sha256"]
                    or anchor_record["row_token_sha256"] != prepared["binding"]["row_token_sha256"]):
                raise ValueError("anchor_row_binding_mismatch")
        # Preserve the historical QA packet as a bound input and verify the full-video baseline.
        prior_obj, prior_hash = _read_private(repo, prior)
        prior_rows = prior_obj.get("records")
        if not isinstance(prior_rows, list) or len(prior_rows) != 3:
            raise ValueError("anchor_prior_qa_rows_invalid")
        if any(max(abs(float(a)-float(b)) for a,b in zip(
                prior_row["choice_log_probabilities_nats"]["full"],
                anchor_row["full_choice_logprobs_nats"], strict=True)) > 1e-4
               for prior_row, anchor_row in zip(prior_rows, anchor["records"], strict=True)):
            raise ValueError("anchor_baseline_prior_mismatch")
        for row_pos in row_positions:
            try:
                record = _score_row(rows[row_pos], anchor["records"][row_pos],
                    model, torch, counters, Path(anchor_path).stem, row_pos + 1)
            except Exception as exc:
                failed_path = output.with_name(output.stem + "_failure.json")
                failure = {"schema_version": 1,
                    "status": "failed_exploratory_full_video_answer_accuracy",
                    "purpose": "posthoc_full_video_restricted_choice_accuracy_under_qta_and_hash_matched_sham",
                    "requested_records": 27,
                    "attempted_forwards": counters["attempted"],
                    "completed_forwards": counters["completed"],
                    "failure_stage": "score_frozen_row",
                    "failure_code": str(exc)[:200],
                    "failed_cohort": Path(anchor_path).stem,
                    "failed_row_position": row_pos + 1,
                    "completed_row_records": len(records),
                    "anchor_receipts": [{"path": p, "sha256": d}
                        for p, _, d in anchor_values],
                    "source_sha256": source_hashes,
                    "created_at_utc": datetime.now(timezone.utc).isoformat()}
                _write_private_exclusive(failed_path, failure)
                raise
            record["anchor_receipt_sha256"] = anchor_hash
            record["anchor_prior_qa_sha256"] = prior_hash
            record["input_sha256"] = hashes
            records.append(record)
            if args.smoke_first_row:
                break
        del rows
        gc.collect(); torch.cuda.empty_cache()
        if args.smoke_first_row:
            break
    del model; gc.collect(); torch.cuda.empty_cache()
    expected = 1 if args.smoke_first_row else 27
    if len(records) != expected or counters["completed"] != expected * 3:
        raise RuntimeError("incomplete_diagnostic_run")
    if (len({record["row_token_sha256"] for record in records}) != expected
            or len({record["source_video_sha256"] for record in records}) != expected):
        raise ValueError("cohort_rows_not_unique")
    payload = {"schema_version": 1,
        "status": "smoke_only" if args.smoke_first_row else "completed_exploratory_full_video_answer_accuracy",
        "purpose": "posthoc_full_video_restricted_choice_accuracy_under_qta_and_hash_matched_sham",
        "interpretation_scope": "NExT-GQA restricted five-choice QA; not generated-explanation faithfulness",
        "human_review": {"actual_review_performed": False},
        "rights_status": "unverified; protected data and outputs kept local",
        "requested_records": expected, "effective_records": len(records),
        "attempted_forwards": counters["attempted"], "completed_forwards": counters["completed"],
        "anchor_receipts": [{"path": p, "sha256": d} for p, _, d in anchor_values],
        "source_sha256": source_hashes,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": qta.MODEL_REVISION,
        "environment": environment,
        "precision": "4-bit weights; BF16 compute; eager attention",
        "layers_1_based": list(LAYERS), "bias_requested": BIAS,
        "seed": 20261021, "records": records,
        "finite_cohort_summary": None if args.smoke_first_row else summarize_records(records),
        "runtime_seconds": time.monotonic() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "requested_work_estimate": "27 records × 3 forwards = 81 forwards plus one model load"}
    _write_private_exclusive(output, payload)
    print(json.dumps({"status": payload["status"], "records": expected,
        "forwards": counters["completed"], "accuracy": payload["finite_cohort_summary"],
        "runtime_seconds": payload["runtime_seconds"],
        "peak_vram_allocated_bytes": payload["peak_vram_allocated_bytes"],
        "output_sha256": _sha256(output), "output_mode": oct(stat.S_IMODE(output.stat().st_mode))},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
