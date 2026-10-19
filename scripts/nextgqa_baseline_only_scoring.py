"""Non-authorizing three-row full-video baseline scorer primitive.

This inference primitive consumes already-built CPU ``FrozenBatch`` objects.
It does not inspect or create treatment branches, load weights, authenticate
upstream reviews, or grant execution authorization. A separately validated
pre-load gate and a caller-owned model instance are required.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import torch

from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256
from nextgqa_model_call_boundary import ModelCallBoundary
from qta_attention_forward_capture import capture_qta_attention_forward_once
from qta_attention_capture_contract import LAYERS as QTA_CAPTURE_LAYERS
from qwen_question_temporal_attention_selector import (
    select_hash_sham_temporal_groups, select_top_temporal_groups,
)


ROW_COUNT = 3
LABELS = ("A", "B", "C", "D", "E")
RUN_DIGEST_FIELDS = frozenset({
    "model_sha256", "processor_sha256", "tokenizer_sha256",
    "runtime_sha256", "protocol_sha256", "scorer_sha256",
})
_SHA256_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_BATCH_KEYS = frozenset({
    "branch", "condition", "intervention", "mask", "mask_type", "treatment",
})
_REQUIRED_VIDEO_KEYS = frozenset({
    "input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw",
    "second_per_grid_ts",
})
SHAM_SEED = "qta-sham-seed-20261018-v1"


def _fail(code: str) -> None:
    raise ValueError(code)


def _digest(value: Any, code: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in _SHA256_HEX for char in value)):
        _fail(code)
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        _fail("baseline_value_not_canonicalizable")


def _walk_batch(value: Any, path: str = "batch") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str or not key:
                _fail("baseline_batch_key_invalid")
            if key.lower() in _FORBIDDEN_BATCH_KEYS:
                _fail("baseline_treatment_input_rejected")
            _walk_batch(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_batch(child, f"{path}.{index}")
    elif isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            _fail("baseline_batch_must_be_cpu")
        if value.layout != torch.strided or value.is_quantized or value.is_complex():
            _fail("baseline_batch_tensor_unsupported")
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            _fail("baseline_batch_nonfinite_tensor")
    elif value is None or type(value) in (str, int, float, bool):
        if isinstance(value, float) and not math.isfinite(value):
            _fail("baseline_batch_nonfinite_value")
    else:
        _fail("baseline_batch_value_unsupported")


def _input_ids_and_hash(batch: Mapping[str, Any]) -> tuple[list[int], str]:
    ids = batch.get("input_ids")
    if (not isinstance(ids, torch.Tensor) or ids.layout != torch.strided
            or ids.device.type != "cpu" or ids.ndim != 2 or ids.shape[0] != 1
            or ids.shape[1] < 1 or ids.is_floating_point() or ids.is_complex()
            or ids.dtype is torch.bool):
        _fail("baseline_input_ids_schema_invalid")
    try:
        values = ids[0].detach().cpu().tolist()
    except Exception:
        _fail("baseline_input_ids_unavailable")
    if any(type(value) is not int or value < 0 for value in values):
        _fail("baseline_input_ids_value_invalid")
    return values, hashlib.sha256(_canonical(values)).hexdigest()


def _position_contract(question_positions: Any, temporal_groups: Any,
                       input_length: int) -> tuple[list[int], list[list[int]]]:
    if (not isinstance(question_positions, (list, tuple)) or not question_positions
            or any(type(value) is not int or value < 0 or value >= input_length
                   for value in question_positions)
            or list(question_positions) != list(range(
                question_positions[0], question_positions[-1] + 1))):
        _fail("baseline_question_position_contract_invalid")
    if (not isinstance(temporal_groups, (list, tuple)) or not temporal_groups
            or any(not isinstance(group, (list, tuple)) or not group
                   or any(type(value) is not int or value < 0 or value >= input_length
                          for value in group)
                   or list(group) != list(range(group[0], group[-1] + 1))
                   for group in temporal_groups)):
        _fail("baseline_temporal_group_contract_invalid")
    groups = [list(group) for group in temporal_groups]
    flattened = [position for group in groups for position in group]
    if (groups != sorted(groups, key=lambda group: group[0])
            or len(set(flattened)) != len(flattened)
            or len({len(group) for group in groups}) != 1
            or max(flattened) >= question_positions[0]):
        _fail("baseline_temporal_group_contract_invalid")
    return list(question_positions), groups


def _validate_inputs(*, model: Any, rows: Any,
                     expected_record_bindings: Any,
                     run_digests: Any) -> tuple[list[dict], int]:
    if model is None or not callable(getattr(model, "forward", None)):
        _fail("baseline_model_instance_required")
    if getattr(model, "training", None) is not False:
        _fail("baseline_model_must_be_in_eval_mode")
    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if type(vocab_size) is not int or vocab_size < 5:
        _fail("baseline_model_vocab_size_invalid")
    if getattr(config, "_attn_implementation", None) != "eager":
        _fail("baseline_eager_attention_required")
    try:
        hooked_attention_modules = [
            model.model.language_model.layers[layer - 1].self_attn
            for layer in QTA_CAPTURE_LAYERS
        ]
    except (AttributeError, IndexError, TypeError):
        _fail("baseline_attention_module_inventory_invalid")
    if (len(hooked_attention_modules) != len(QTA_CAPTURE_LAYERS)
            or any(getattr(module, "training", None) is not False
                   for module in hooked_attention_modules)):
        _fail("baseline_attention_modules_must_be_in_eval_mode")
    if not isinstance(run_digests, Mapping) or set(run_digests) != RUN_DIGEST_FIELDS:
        _fail("baseline_run_digest_inventory_invalid")
    for field in RUN_DIGEST_FIELDS:
        _digest(run_digests[field], "baseline_run_digest_invalid")
    if (not isinstance(rows, (list, tuple)) or len(rows) != ROW_COUNT
            or not isinstance(expected_record_bindings, (list, tuple))
            or len(expected_record_bindings) != ROW_COUNT):
        _fail("baseline_three_row_inventory_required")

    normalized = []
    seen_ids = set()
    seen_sources = set()
    for position, (row, expected) in enumerate(
            zip(rows, expected_record_bindings, strict=True), start=1):
        if not isinstance(row, Mapping) or set(row) != {
                "record_id", "source_video_sha256", "condition", "prompt_sha256",
                "input_ids_sha256", "batch_sha256", "answer_token_ids",
                "question_positions", "temporal_groups", "opaque_record_token",
                "batch"}:
            _fail(f"baseline_row_{position}_schema_invalid")
        if not isinstance(expected, Mapping) or set(expected) != {
                "record_id", "source_video_sha256", "prompt_sha256",
                "input_ids_sha256", "batch_sha256", "question_positions",
                "temporal_groups", "opaque_record_token", "answer_token_ids"}:
            _fail(f"baseline_expected_row_{position}_schema_invalid")
        record_id = row["record_id"]
        if (not isinstance(record_id, str) or not record_id or record_id in seen_ids
                or record_id != expected["record_id"]):
            _fail("baseline_record_order_or_identity_mismatch")
        seen_ids.add(record_id)
        if row["condition"] != "full":
            _fail("baseline_only_full_condition_required")
        for field in ("source_video_sha256", "prompt_sha256",
                      "input_ids_sha256", "batch_sha256"):
            actual = _digest(row[field], f"baseline_{field}_invalid")
            pinned = _digest(expected[field], f"baseline_expected_{field}_invalid")
            if actual != pinned:
                _fail(f"baseline_{field}_binding_mismatch")
        if row["source_video_sha256"] in seen_sources:
            _fail("baseline_duplicate_source_video_rejected")
        seen_sources.add(row["source_video_sha256"])
        if type(row["batch"]) is not FrozenBatch or not row["batch"]:
            _fail("baseline_frozen_batch_required")
        if ("pixel_values" in row["batch"]
                or not _REQUIRED_VIDEO_KEYS.issubset(row["batch"])):
            _fail("baseline_full_video_batch_required")
        pixels = row["batch"].get("pixel_values_videos")
        if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 2
                or pixels.numel() < 1 or not pixels.dtype.is_floating_point):
            _fail("baseline_video_pixel_payload_invalid")
        grid = row["batch"].get("video_grid_thw")
        if (not isinstance(grid, torch.Tensor) or grid.ndim != 2
                or grid.shape[0] != 1 or grid.shape[1] != 3
                or grid.dtype not in (torch.int32, torch.int64)
                or bool((grid <= 0).any().item())):
            _fail("baseline_video_grid_invalid")
        ids_tensor = row["batch"]["input_ids"]
        mask = row["batch"]["attention_mask"]
        time_steps, grid_h, grid_w = map(int, grid[0].tolist())
        if (not isinstance(mask, torch.Tensor) or mask.shape != ids_tensor.shape
                or not bool((mask == 1).all().item())):
            _fail("baseline_unpadded_attention_mask_required")
        if int(pixels.shape[0]) != time_steps * grid_h * grid_w:
            _fail("baseline_video_pixel_grid_count_mismatch")
        seconds = row["batch"].get("second_per_grid_ts")
        if (not isinstance(seconds, torch.Tensor) or seconds.ndim != 1
                or tuple(seconds.shape) != (1,) or not seconds.dtype.is_floating_point
                or not bool(torch.isfinite(seconds).all().item())
                or not bool((seconds > 0).all().item())):
            _fail("baseline_video_timing_invalid")
        _walk_batch(row["batch"])
        values, observed_ids_sha = _input_ids_and_hash(row["batch"])
        if observed_ids_sha != row["input_ids_sha256"]:
            _fail("baseline_input_ids_digest_mismatch")
        try:
            observed_batch_sha = batch_sha256(row["batch"])
        except Exception:
            _fail("baseline_batch_digest_unavailable")
        if observed_batch_sha != row["batch_sha256"]:
            _fail("baseline_batch_digest_mismatch")
        question_positions, temporal_groups = _position_contract(
            row["question_positions"], row["temporal_groups"], len(values))
        if (question_positions != expected["question_positions"]
                or temporal_groups != expected["temporal_groups"]
                or row["opaque_record_token"] != expected["opaque_record_token"]):
            _fail("baseline_selector_position_binding_mismatch")
        if len(temporal_groups) != time_steps:
            _fail("baseline_temporal_group_video_grid_count_mismatch")
        token = row["opaque_record_token"]
        if (not isinstance(token, str) or len(token) != 24
                or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
                       for char in token)):
            _fail("baseline_opaque_record_token_invalid")
        answer_ids = row["answer_token_ids"]
        if (not isinstance(answer_ids, (list, tuple)) or len(answer_ids) != 5
                or any(type(token) is not int or token < 0 or token >= vocab_size
                       for token in answer_ids)
                or len(set(answer_ids)) != 5):
            _fail("baseline_answer_token_contract_invalid")
        if list(answer_ids) != expected["answer_token_ids"]:
            _fail("baseline_answer_token_binding_mismatch")
        normalized.append({
            "record_id": record_id,
            "source_video_sha256": row["source_video_sha256"],
            "prompt_sha256": row["prompt_sha256"],
            "input_ids_sha256": row["input_ids_sha256"],
            "batch_sha256": row["batch_sha256"],
            "input_ids": values,
            "answer_token_ids": list(answer_ids),
            "question_positions": question_positions,
            "temporal_groups": temporal_groups,
            "opaque_record_token": token,
            "batch": row["batch"],
        })
    return normalized, vocab_size


def score_full_video_baseline_three_rows(*, model: Any, rows: Any,
                                         expected_record_bindings: Any,
                                         run_digests: Any,
                                         torch_module: Any = torch,
                                         device: str = "cuda") -> dict[str, Any]:
    """Score exactly three prebuilt full-video batches with one forward each.

    All rows and token IDs are validated before the first forward. The batch
    itself is passed unchanged apart from device transfer; ``use_cache=False``
    and ``logits_to_keep=1`` are inference controls verified at root-forward
    entry. The caller must independently validate every pre-load gate and load
    the model; this function has no loader or authorization interface.
    """
    normalized, vocab_size = _validate_inputs(
        model=model, rows=rows,
        expected_record_bindings=expected_record_bindings,
        run_digests=run_digests)
    if (torch_module is not torch or not isinstance(device, str) or not device):
        _fail("baseline_scoring_runtime_invalid")

    boundary = ModelCallBoundary()
    attempted = completed = 0
    scores: list[dict[str, Any]] = []
    failure_code = None
    for position, row in enumerate(normalized, start=1):
        try:
            # Rehash immediately before transfer and again after transfer so a
            # mutated CPU snapshot or changed device copy cannot reach forward.
            if batch_sha256(row["batch"]) != row["batch_sha256"]:
                raise ValueError("baseline_batch_changed_before_forward")
            batch = row["batch"].to(device)
            if (not isinstance(batch, Mapping)
                    or batch_sha256(batch) != row["batch_sha256"]):
                raise ValueError("baseline_batch_changed_after_transfer")
            call_kwargs = dict(batch)
            call_kwargs.update(use_cache=False, logits_to_keep=1)
            attempted += 1
            with torch_module.inference_mode():
                output, captured_rows, capture_report = capture_qta_attention_forward_once(
                    model=model,
                    forward=lambda: boundary(
                        model=model,
                        call_kwargs=call_kwargs,
                        expected_batch_keys=tuple(batch),
                        expected_batch_sha256=row["batch_sha256"],
                        row_position=position,
                        condition_position=1,
                        condition="full",
                    ),
                    question_positions=row["question_positions"],
                    input_length=len(row["input_ids"]),
                    torch_module=torch_module,
                )
            completed += 1
            logits = getattr(output, "logits", None)
            if (not isinstance(logits, torch.Tensor) or logits.ndim != 3
                    or logits.shape[0] != 1 or logits.shape[1] < 1
                    or logits.shape[2] != vocab_size):
                raise ValueError("baseline_logits_schema_invalid")
            last_logits = logits[0, -1].float()
            if not bool(torch.isfinite(last_logits).all().item()):
                raise ValueError("baseline_nonfinite_logits")
            full_vocab_log_probs = torch.log_softmax(last_logits, dim=-1)
            if not bool(torch.isfinite(full_vocab_log_probs).all().item()):
                raise ValueError("baseline_nonfinite_log_probabilities")
            answer_scores = [float(full_vocab_log_probs[token].item())
                             for token in row["answer_token_ids"]]
            if any(not math.isfinite(value) for value in answer_scores):
                raise ValueError("baseline_answer_log_probability_invalid")
            order = sorted(range(5), key=lambda index: (-answer_scores[index], index))
            p_index, q_index = order[:2]
            layer_group_masses = []
            for layer in QTA_CAPTURE_LAYERS:
                attention = captured_rows[layer]
                averaged = attention.to(device="cpu", dtype=torch.float32).mean(dim=(0, 1))
                layer_group_masses.append([
                    float(averaged[group].sum().item())
                    for group in row["temporal_groups"]
                ])
            if (capture_report.get("status") !=
                    "one_forward_attention_rows_captured_non_authorizing"
                    or capture_report.get("forward_calls") != 1
                    or capture_report.get("layer_indices_1_based") != QTA_CAPTURE_LAYERS
                    or capture_report.get("attention_rows_satisfy_source_dtype_normalization_tolerance") is not True
                    or capture_report.get("forward_source_authenticated") is not False
                    or any(count != 1 for count in capture_report.get("layer_calls", {}).values())
                    or len(layer_group_masses) != len(QTA_CAPTURE_LAYERS)):
                raise ValueError("baseline_selector_attention_capture_invalid")
            group_scores = [
                sum(layer[index] for layer in layer_group_masses)
                / len(layer_group_masses)
                for index in range(len(row["temporal_groups"]))
            ]
            top_k = max(1, math.ceil(len(group_scores) * 0.10))
            target_ranked_groups = select_top_temporal_groups(group_scores, top_k=top_k)
            target_groups = sorted(target_ranked_groups)
            sham_groups = select_hash_sham_temporal_groups(
                temporal_group_count=len(group_scores), top_k=top_k,
                protocol_seed=SHAM_SEED,
                opaque_record_token=row["opaque_record_token"],
            )
            scores.append({
                "record_id": row["record_id"],
                "source_video_sha256": row["source_video_sha256"],
                "prompt_sha256": row["prompt_sha256"],
                "input_ids_sha256": row["input_ids_sha256"],
                "input_length": len(row["input_ids"]),
                "score_position": len(row["input_ids"]) - 1,
                "score_input_token_id": row["input_ids"][-1],
                "processor_batch_sha256": row["batch_sha256"],
                "question_positions": row["question_positions"],
                "temporal_groups": row["temporal_groups"],
                "selector_attention_layers_1_based": QTA_CAPTURE_LAYERS,
                "selector_attention_layer_group_masses": layer_group_masses,
                "selector_group_scores": group_scores,
                "selector_top_k": top_k,
                "target_group_indices": target_groups,
                "target_ranked_group_indices": target_ranked_groups,
                "target_token_positions": [
                    token_position for group_index in target_groups
                    for token_position in row["temporal_groups"][group_index]
                ],
                "sham_seed": SHAM_SEED,
                "sham_group_indices": sham_groups,
                "sham_token_positions": [
                    token_position for group_index in sham_groups
                    for token_position in row["temporal_groups"][group_index]
                ],
                "selector_attention_capture_status": capture_report["status"],
                "selector_attention_source_dtype": capture_report[
                    "attention_source_dtype"],
                "answer_token_ids": row["answer_token_ids"],
                "answer_token_ids_sha256": hashlib.sha256(
                    _canonical(row["answer_token_ids"])).hexdigest(),
                "answer_label_log_probabilities_nats": dict(zip(
                    LABELS, answer_scores, strict=True)),
                "p_label": LABELS[p_index],
                "p_index": p_index,
                "p_token_id": row["answer_token_ids"][p_index],
                "q_label": LABELS[q_index],
                "q_index": q_index,
                "q_token_id": row["answer_token_ids"][q_index],
            })
        except Exception as exc:
            # Keep reports free of exception text, which can contain local data.
            failure_code = (str(exc) if isinstance(exc, ValueError)
                            and str(exc).startswith("baseline_")
                            else "baseline_forward_or_score_failed_" + type(exc).__name__)
            break

    if failure_code is None and (
            attempted != ROW_COUNT or completed != ROW_COUNT
            or len(scores) != ROW_COUNT or boundary.root_call_entries != ROW_COUNT
            or boundary.verified_root_calls != ROW_COUNT
            or boundary.completed_calls != ROW_COUNT):
        failure_code = "baseline_exact_three_forward_count_mismatch"
    success = failure_code is None
    report = {
        "schema_version": 1,
        "purpose": "qta_three_row_full_video_baseline_score_diagnostic_non_authorizing",
        "status": "completed_non_authorizing" if success else "failed_non_authorizing",
        "scope": "three_full_video_baseline_rows_only",
        "score_definition": (
            "log_softmax over full next-token vocabulary, indexed at each row's five A-E token IDs"),
        "requested_records": ROW_COUNT,
        "requested_forwards": ROW_COUNT,
        "attempted_forwards": attempted,
        # Count root forwards whose wrapped model call returned. Attention
        # post-hook validation can still fail after that point.
        "completed_forwards": boundary.completed_calls,
        "root_call_entries": boundary.root_call_entries,
        "verified_root_calls": boundary.verified_root_calls,
        "completed_boundary_calls": boundary.completed_calls,
        "failure_code": failure_code,
        "records": scores if success else [],
        "run_digests": dict(run_digests) if success else None,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
        "faithfulness_result": False,
    }
    return report
