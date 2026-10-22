#!/usr/bin/env python3
"""Exploratory paired QA-interval generation with a frozen temporal key bias.

This local-only runner compares greedy baseline, question-attention-selected
temporal-key bias, and equal-count hash-sham bias on the frozen three rows in
``claim_likelihood_claims_remaining_20261021_30.json``. The endpoint is
temporal IoU against each row's NExT-GQA QA interval. It is a QA temporal
localization proxy, not explanation-claim support or direct faithfulness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any

PACKET_REL = Path("data/private_nextgqa/claim_likelihood_claims_remaining_20261021_30.json")
PACKET_NAME = PACKET_REL.name
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
FRAME_COUNT = 16
LAYERS_1_BASED = tuple(range(5, 17))
LOGIT_BIAS = math.log(2.0)
SEED = 20261021
PROTOCOL_SEED = "nextgqa-qa-interval-generation-temporal-bias-v1"
BRANCHES = ("baseline", "selected_temporal_bias", "hash_sham_bias")
PROMPT_PROTOCOL = (
    "The video is represented by 16 frames sampled in chronological order, "
    "indexed 00 through 15. Identify the shortest inclusive frame-index interval "
    "that contains visual evidence useful for answering the question. Do not "
    "answer the question. Return only START,END as two zero-padded decimal "
    "frame indices, for example 03,07. No words, spaces, or punctuation beyond "
    "the comma."
)
SOURCE_REL = (
    "scripts/run_qta_intervened_interval_generation.py",
    "scripts/qwen_visual_attention_boost.py",
    "scripts/qwen_question_temporal_attention_selector.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/presentation_time_utils.py",
    "scripts/necessity_choice_margin_utils.py",
    "scripts/private_output_paths.py",
)
MIN_FREE_VRAM = 12 * 1024**3
MIN_FREE_HOST = 8 * 1024**3


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_private_json(path: Path, private_root: Path) -> tuple[dict, bytes]:
    root = private_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("private_input_path_invalid")
    fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
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
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    return json.loads(raw.decode("utf-8")), raw


def validate_packet(payload: dict) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("frozen_three_record_packet_contract_invalid")
    records = payload.get("records")
    if (payload.get("status") != "passed" or payload.get("dataset_split") != "test"
            or payload.get("model") != MODEL_NAME
            or payload.get("model_revision") != MODEL_REVISION
            or payload.get("claim_review_completed") is not False
            or payload.get("requested_videos") != 3
            or payload.get("effective_videos") != 3
            or payload.get("fully_structured_claim_records") != 3
            or not isinstance(records, list) or len(records) != 3):
        raise ValueError("frozen_three_record_packet_contract_invalid")
    media = []
    for row in records:
        parsed = row.get("generated_parsed") if isinstance(row, dict) else None
        intervals = row.get("gold_interval_seconds") if isinstance(row, dict) else None
        if (not isinstance(row.get("question"), str) or not row["question"].strip()
                or not isinstance(row.get("choices"), list) or len(row["choices"]) != 5
                or not isinstance(row.get("local_video_path"), str)
                or not _is_sha256(row.get("video_sha256"))
                or not isinstance(parsed, dict) or parsed.get("valid") is not True
                or not isinstance(intervals, list) or not intervals
                or not all(_valid_gold_interval(span) for span in intervals)):
            raise ValueError("frozen_record_fields_invalid")
        media.append(row["video_sha256"])
    if len(set(media)) != 3:
        raise ValueError("frozen_records_not_unique_media")
    return records


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _valid_gold_interval(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(float(x)) for x in value)
            and 0 <= float(value[0]) < float(value[1]))


def selected_group_indices(scores, group_count: int) -> tuple[int, ...]:
    from qwen_question_temporal_attention_selector import select_top_temporal_groups
    if type(group_count) is not int or group_count < 1:
        raise ValueError("temporal_group_count_invalid")
    top_k = max(1, math.ceil(group_count * 0.10))
    selected = tuple(select_top_temporal_groups(scores, top_k=top_k))
    if len(scores) != group_count or len(selected) != top_k:
        raise ValueError("temporal_selector_count_mismatch")
    return selected


def hash_sham_group_indices(*, group_count: int, top_k: int,
                            selected: tuple[int, ...], record_token: str,
                            protocol_seed: str = PROTOCOL_SEED) -> tuple[int, ...]:
    """Return a deterministic hash-ranked equal-size sham, disjoint from treatment."""
    if (type(group_count) is not int or group_count < 1 or type(top_k) is not int
            or not 1 <= top_k <= group_count or not record_token or not protocol_seed
            or tuple(sorted(set(selected))) != selected
            or any(type(index) is not int or index < 0 or index >= group_count
                   for index in selected)
            or group_count - len(selected) < top_k):
        raise ValueError("hash_sham_selection_inputs_invalid")
    prefix = f"interval-generation-sham-v1\0{protocol_seed}\0{record_token}\0"
    candidates = (index for index in range(group_count) if index not in set(selected))
    ranked = sorted(candidates, key=lambda index: (
        hashlib.sha256(f"{prefix}{index}".encode("utf-8")).digest(), index))
    result = tuple(sorted(ranked[:top_k]))
    if len(result) != top_k or set(result) & set(selected):
        raise ValueError("hash_sham_group_contract_invalid")
    return result


def interval_iou(prediction: Any, gold_intervals: list, duration: float) -> dict:
    """QA interval max tIoU; invalid generations return zero and remain counted."""
    valid_duration = (isinstance(duration, (int, float)) and not isinstance(duration, bool)
                      and math.isfinite(float(duration)) and float(duration) > 0)
    if not valid_duration or not isinstance(gold_intervals, list) or not gold_intervals:
        raise ValueError("qa_interval_scoring_inputs_invalid")
    gold = []
    for span in gold_intervals:
        if not _valid_gold_interval(span) or float(span[1]) > float(duration):
            raise ValueError("qa_gold_interval_invalid")
        gold.append((float(span[0]), float(span[1])))
    if (not isinstance(prediction, (list, tuple)) or len(prediction) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(float(x)) for x in prediction)):
        return {"temporal_iou": 0.0, "status": "invalid_prediction"}
    start, end = map(float, prediction)
    if not 0 <= start < end <= float(duration):
        return {"temporal_iou": 0.0, "status": "invalid_prediction"}
    values = []
    for gold_start, gold_end in gold:
        intersection = max(0.0, min(end, gold_end) - max(start, gold_start))
        union = (end - start) + (gold_end - gold_start) - intersection
        values.append(intersection / union if union else 0.0)
    return {"temporal_iou": max(values), "status": "scored"}


def interval_code_token_sequences(tokenizer, frame_count: int = FRAME_COUNT, *,
                                  rendered_prompt: str | None = None) -> dict[tuple[int, ...], str]:
    """Enumerate intervals; bind text boundary without confusing multimodal IDs.

    Qwen's processor expands rendered video markers into many input IDs, so its
    actual multimodal prompt IDs cannot be compared with tokenizer encoding of
    the text-only chat template. We instead prove every code is an exact stable
    textual continuation of that template and constrain only generated suffix IDs.
    """
    if type(frame_count) is not int or not 1 <= frame_count <= 100:
        raise ValueError("interval_code_frame_count_invalid")
    text_prefix = (tokenizer.encode(rendered_prompt, add_special_tokens=True)
                   if rendered_prompt is not None else None)
    sequences = {}
    for start in range(frame_count):
        for end in range(start, frame_count):
            code = f"{start:02d},{end:02d}"
            if rendered_prompt is None:
                ids = tuple(tokenizer.encode(code, add_special_tokens=False))
            else:
                full_ids = tokenizer.encode(rendered_prompt + code, add_special_tokens=True)
                if full_ids[:len(text_prefix)] != text_prefix:
                    raise ValueError("interval_code_prompt_token_boundary_mismatch")
                ids = tuple(full_ids[len(text_prefix):])
            if (not ids or any(type(token) is not int or token < 0 for token in ids)
                    or tuple(tokenizer.encode(tokenizer.decode(ids, skip_special_tokens=True),
                                               add_special_tokens=False)) != ids
                    or tokenizer.decode(ids, skip_special_tokens=True) != code
                    or ids in sequences):
                raise ValueError("interval_code_tokenization_not_injective")
            sequences[ids] = code
    return sequences


def allowed_interval_next_tokens(sequences: dict[tuple[int, ...], str],
                                 generated: tuple[int, ...], eos_token_id: int) -> list[int]:
    """Return only tokens that keep the continuation on a complete valid code."""
    if (not isinstance(sequences, dict) or not sequences
            or type(eos_token_id) is not int or eos_token_id < 0
            or not isinstance(generated, tuple)
            or any(type(token) is not int or token < 0 for token in generated)):
        raise ValueError("interval_code_prefix_inputs_invalid")
    matching = [sequence for sequence in sequences
                if len(generated) <= len(sequence) and sequence[:len(generated)] == generated]
    if not matching:
        raise ValueError("interval_code_prefix_not_in_grammar")
    allowed = {sequence[len(generated)] for sequence in matching
               if len(sequence) > len(generated)}
    if any(len(sequence) == len(generated) for sequence in matching):
        allowed.add(eos_token_id)
    return sorted(allowed)


def generation_eos_ids(value: Any) -> tuple[int, ...]:
    """Normalize Transformers' version-dependent scalar/list EOS setting."""
    values = value if isinstance(value, (list, tuple, set)) else [value]
    if (not values or any(type(token) is not int or token < 0 for token in values)
            or len(set(values)) != len(values)):
        raise ValueError("interval_code_eos_config_invalid")
    return tuple(values)


def interval_prefix_allowed_tokens(batch_id: int, input_ids: Any, *,
                                   prompt_length: int,
                                   sequences: dict[tuple[int, ...], str],
                                   eos_token_id: int) -> list[int]:
    """Transformers prefix callback receives a 1-D sequence for each row."""
    if type(batch_id) is not int or batch_id != 0 or type(prompt_length) is not int:
        raise ValueError("interval_code_generation_prefix_invalid")
    if hasattr(input_ids, "detach"):
        if getattr(input_ids, "ndim", None) != 1:
            raise ValueError("interval_code_generation_prefix_invalid")
        values = input_ids.detach().cpu().tolist()
    else:
        values = input_ids
    if not isinstance(values, (list, tuple)) or len(values) < prompt_length:
        raise ValueError("interval_code_generation_prefix_invalid")
    suffix = tuple(int(value) for value in values[prompt_length:])
    return allowed_interval_next_tokens(sequences, suffix, eos_token_id)


def frame_cell_edges(frame_pts: list[float], duration: float) -> list[float]:
    """Map sampled frame indices to contiguous time cells using actual sampled PTS."""
    if (not isinstance(frame_pts, list) or len(frame_pts) != FRAME_COUNT
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(float(x)) for x in frame_pts)
            or isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration)) or duration <= 0):
        raise ValueError("sampled_frame_time_geometry_invalid")
    pts = [float(x) for x in frame_pts]
    if (any(x < 0 or x > float(duration) for x in pts)
            or any(b <= a for a, b in zip(pts, pts[1:]))):
        raise ValueError("sampled_frame_pts_invalid")
    edges = [0.0]
    edges.extend((left + right) / 2.0 for left, right in zip(pts, pts[1:]))
    edges.append(float(duration))
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("sampled_frame_cell_edges_invalid")
    return edges


def validate_processor_video_metadata(metadata: Any, target_fps: float) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("processor_video_metadata_mismatch")
    frame_total = metadata.get("total_num_frames")
    fps = metadata.get("fps")
    if (metadata.get("frames_indices") != list(range(FRAME_COUNT))
            or isinstance(frame_total, bool)
            or not isinstance(frame_total, (int, float))
            or not math.isfinite(float(frame_total))
            or not math.isclose(float(frame_total), FRAME_COUNT,
                                rel_tol=1e-7, abs_tol=1e-7)
            or isinstance(fps, bool) or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or not math.isclose(float(fps), target_fps,
                                rel_tol=1e-7, abs_tol=1e-7)):
        raise ValueError("processor_video_metadata_mismatch")


def summarize_generation_logits(logits: Any, generated_token_ids: list[int], torch) -> list[dict]:
    """Compact raw-logit diagnostics without retaining/publishing full vocab arrays."""
    if not isinstance(logits, (tuple, list)) or len(logits) != len(generated_token_ids):
        raise ValueError("generation_raw_logit_step_count_mismatch")
    result = []
    for step, (raw, chosen_id) in enumerate(zip(logits, generated_token_ids, strict=True)):
        if (not torch.is_tensor(raw) or raw.ndim != 2 or int(raw.shape[0]) != 1
                or type(chosen_id) is not int or not 0 <= chosen_id < int(raw.shape[-1])):
            raise ValueError("generation_raw_logit_shape_invalid")
        scores = raw[0].float()
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("generation_raw_logits_nonfinite")
        log_probs = torch.log_softmax(scores, dim=-1)
        top_values, top_ids = torch.topk(scores, k=2)
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum()
        result.append({"step": step, "generated_token_id": chosen_id,
            "greedy_top1_token_id": int(top_ids[0].detach().cpu()),
            "greedy_top1_matches_generated": int(top_ids[0].detach().cpu()) == chosen_id,
            "top1_minus_top2_logit": float((top_values[0] - top_values[1]).detach().cpu()),
            "generated_token_log_probability": float(log_probs[chosen_id].detach().cpu()),
            "entropy_nats": float(entropy.detach().cpu())})
    return result


def compare_generation_logit_diagnostics(audits: dict[str, dict]) -> dict:
    """Exploratory same-prefix score contrast; never a faithfulness endpoint."""
    baseline = audits["baseline"].get("raw_logit_diagnostics")
    treatment = audits["selected_temporal_bias"].get("raw_logit_diagnostics")
    sham = audits["hash_sham_bias"].get("raw_logit_diagnostics")
    if not all(isinstance(value, list) for value in (baseline, treatment, sham)):
        raise ValueError("generation_raw_logit_diagnostics_missing")
    common = 0
    for index, items in enumerate(zip(baseline, treatment, sham)):
        if len({item["generated_token_id"] for item in items}) != 1:
            break
        common = index + 1
    return {"scope": "post hoc mechanistic diagnostic; not an efficacy or faithfulness endpoint",
        "common_generated_prefix_tokens": common,
        "compared_log_probability_deltas": {
            "selected_minus_baseline": [
                treatment[i]["generated_token_log_probability"]
                - baseline[i]["generated_token_log_probability"] for i in range(common)],
            "selected_minus_sham": [
                treatment[i]["generated_token_log_probability"]
                - sham[i]["generated_token_log_probability"] for i in range(common)]},
        "argmax_changed_steps_vs_baseline": sum(
            treatment[i]["greedy_top1_token_id"] != baseline[i]["greedy_top1_token_id"]
            for i in range(common)),
        "entropy_deltas_selected_minus_baseline": [
            treatment[i]["entropy_nats"] - baseline[i]["entropy_nats"]
            for i in range(common)]}
def parse_interval_generation(text: str, duration: float,
                              frame_pts: list[float], frame_count: int = FRAME_COUNT) -> dict:
    if type(frame_count) is not int or not 1 <= frame_count <= FRAME_COUNT:
        raise ValueError("interval_code_frame_count_invalid")
    if not isinstance(text, str) or re.fullmatch(r"\d{2},\d{2}", text) is None:
        return {"prediction": None, "status": "invalid_interval_code"}
    start, end = (int(value) for value in text.split(","))
    if not 0 <= start <= end < frame_count:
        return {"prediction": None, "status": "interval_code_out_of_range"}
    edges = frame_cell_edges(frame_pts, duration)
    if len(edges) != frame_count + 1:
        raise ValueError("sampled_frame_time_geometry_count_mismatch")
    return {"prediction": [edges[start], edges[end + 1]],
            "status": "valid_constrained_interval_code",
            "start_frame_index": start, "end_frame_index": end}


def paired_summary(records: list[dict], requested_n: int) -> dict:
    if type(requested_n) is not int or requested_n not in (1, 3):
        raise ValueError("requested_count_invalid")
    complete = (len(records) == requested_n and all(
        isinstance(row, dict) and isinstance(row.get("condition_scores"), dict)
        and set(row["condition_scores"]) == set(BRANCHES)
        and all(isinstance(row["condition_scores"][name].get("temporal_iou"), (int, float))
                and math.isfinite(float(row["condition_scores"][name]["temporal_iou"]))
                for name in BRANCHES) for row in records))
    if not complete:
        return {"effective_n": len(records), "paired_delta_iou": None,
                "mean_iou_by_condition": None,
                "inferential_status": "none_incomplete_or_not_exact_cohort"}
    deltas = [float(row["condition_scores"]["selected_temporal_bias"]["temporal_iou"])
              - float(row["condition_scores"]["baseline"]["temporal_iou"])
              for row in records]
    means = {name: sum(float(row["condition_scores"][name]["temporal_iou"])
                       for row in records) / requested_n for name in BRANCHES}
    return {"effective_n": requested_n,
            "mean_iou_by_condition": means,
            "paired_delta_iou": deltas,
            "mean_paired_delta_iou": sum(deltas) / requested_n,
            "mean_selected_minus_sham_iou": sum(
                float(row["condition_scores"]["selected_temporal_bias"]["temporal_iou"])
                - float(row["condition_scores"]["hash_sham_bias"]["temporal_iou"])
                for row in records) / requested_n,
            "inferential_status": "finite_frozen_cohort_descriptive_only_no_p_value_or_population_ci"}


def validate_receipt(payload: dict, *, requested_n: int, packet_sha256: str) -> None:
    if (not isinstance(payload, dict) or payload.get("requested_n") != requested_n
            or payload.get("packet_sha256") != packet_sha256
            or payload.get("direct_faithfulness") is not False
            or payload.get("explanation_claim_support_established") is not False
            or payload.get("endpoint") != "paired generated QA interval temporal IoU vs NExT-GQA QA interval"):
        raise ValueError("interval_generation_receipt_binding_invalid")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("interval_generation_receipt_records_invalid")
    summary = paired_summary(records, requested_n)
    if payload.get("summary") != summary:
        raise ValueError("interval_generation_receipt_summary_mismatch")
    branches = payload.get("branch_counts")
    if not isinstance(branches, dict) or set(branches) != set(BRANCHES):
        raise ValueError("interval_generation_branch_counts_invalid")
    if payload.get("status", "").startswith("completed"):
        if len(records) != requested_n or any(branches[name] != {
                "requested": requested_n, "attempted": requested_n,
                "completed": requested_n} for name in BRANCHES):
            raise ValueError("interval_generation_completed_denominator_invalid")
        for row in records:
            scores = row.get("condition_scores")
            audits = row.get("generation_audit")
            if (not isinstance(scores, dict) or set(scores) != set(BRANCHES)
                    or not isinstance(audits, dict) or set(audits) != set(BRANCHES)):
                raise ValueError("interval_generation_completed_row_incomplete")
            for branch in BRANCHES:
                score = scores[branch]
                audit = audits[branch]
                value = score.get("temporal_iou") if isinstance(score, dict) else None
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                        or score.get("parse_status") != "valid_constrained_interval_code"
                        or score.get("score_status") != "scored"
                        or score.get("prediction") is None):
                    raise ValueError("interval_generation_completed_score_invalid")
                if (not isinstance(audit, dict)
                        or audit.get("constraint_code_count") != 136
                        or audit.get("constraint_eos_terminated") is not True
                        or audit.get("generated_token_count") != audit.get("constraint_code_token_count") + 1
                        or audit.get("forward_calls") != len(audit.get("shape_trace", []))
                        or audit.get("forward_calls") != audit.get("generated_token_count")):
                    raise ValueError("interval_generation_completed_audit_invalid")
                diagnostics = audit.get("raw_logit_diagnostics")
                if (not isinstance(diagnostics, list)
                        or len(diagnostics) != audit.get("generated_token_count")
                        or any(not isinstance(step, dict)
                            or not math.isfinite(float(step.get("generated_token_log_probability", float("nan"))))
                            or not math.isfinite(float(step.get("entropy_nats", float("nan"))))
                            or not math.isfinite(float(step.get("top1_minus_top2_logit", float("nan"))))
                            for step in diagnostics)):
                    raise ValueError("interval_generation_raw_logit_diagnostics_invalid")
                mask_audit = audit.get("intervention_mask_audit")
                should_apply = branch != "baseline"
                if (not isinstance(mask_audit, dict)
                        or mask_audit.get("applied") is not should_apply
                        or mask_audit.get("all_selected_cells_match_bias") is not True
                        or mask_audit.get("logit_bias") != (LOGIT_BIAS if should_apply else 0.0)
                        or (should_apply and (not isinstance(mask_audit.get("effective_mask_bias"), (int, float))
                            or not 0 < float(mask_audit["effective_mask_bias"]) <= LOGIT_BIAS))
                        or (should_apply and mask_audit.get("verified_cells", 0) < 1)
                        or (not should_apply and mask_audit.get("hook_calls") != 0)):
                    raise ValueError("interval_generation_completed_intervention_unverified")
            comparison = row.get("mechanistic_logits_comparison")
            common = comparison.get("common_generated_prefix_tokens") if isinstance(comparison, dict) else None
            if (not isinstance(comparison, dict)
                    or comparison.get("scope") != "post hoc mechanistic diagnostic; not an efficacy or faithfulness endpoint"
                    or type(common) is not int or common < 1
                    or not isinstance(comparison.get("compared_log_probability_deltas"), dict)
                    or any(not isinstance(comparison["compared_log_probability_deltas"].get(key), list)
                        or len(comparison["compared_log_probability_deltas"][key]) != common
                        or any(not math.isfinite(float(value))
                               for value in comparison["compared_log_probability_deltas"][key])
                        for key in ("selected_minus_baseline", "selected_minus_sham"))
                    or not isinstance(comparison.get("entropy_deltas_selected_minus_baseline"), list)
                    or len(comparison["entropy_deltas_selected_minus_baseline"]) != common
                    or any(not math.isfinite(float(value))
                           for value in comparison["entropy_deltas_selected_minus_baseline"])
                    or type(comparison.get("argmax_changed_steps_vs_baseline")) is not int):
                raise ValueError("interval_generation_mechanistic_comparison_invalid")


def _sha_path(path: Path) -> str:
    return sha256_file(path)


def _source_hashes(repo: Path) -> dict[str, str]:
    result = {}
    for name in SOURCE_REL:
        path = repo / name
        if not path.is_file():
            raise ValueError("source_dependency_missing")
        result[name] = _sha_path(path)
    return result


def _pinned_model_hashes(model_path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    json_files = sorted(path for path in model_path.glob("*.json") if path.is_file())
    weight_files = sorted(path for path in model_path.glob("*.safetensors") if path.is_file())
    if not json_files or not weight_files:
        raise ValueError("pinned_checkpoint_manifest_or_weights_missing")
    metadata = {path.name: sha256_file(path) for path in json_files}
    weights = {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
               for path in weight_files}
    if any(item["bytes"] <= 0 for item in weights.values()):
        raise ValueError("pinned_checkpoint_weight_shard_empty")
    return metadata, weights


def _read_packet(repo: Path) -> tuple[dict, bytes, Path]:
    private = repo / "data" / "private_nextgqa"
    packet, raw = read_private_json(repo / PACKET_REL, private)
    validate_packet(packet)
    return packet, raw, private


def _host_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("host_memory_probe_unavailable")


def _prompt(processor, record: dict, frames: list, fps: float,
            process_vision_info, torch):
    choices = record["choices"]
    question = record["question"].strip()
    prompt_text = (question + "\nAnswer choices:\n" +
                   "\n".join(f"{letter}. {choice}" for letter, choice in
                             zip("ABCDE", choices, strict=True)) +
                   "\n\n" + PROMPT_PROTOCOL)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames, "sample_fps": fps},
        {"type": "text", "text": prompt_text},
    ]}]
    rendered = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or not isinstance(videos, list) or len(videos) != 1:
        raise ValueError("video_preprocess_utility_contract_invalid")
    pixels, metadata = videos[0]
    if not torch.is_tensor(pixels) or int(pixels.shape[0]) != FRAME_COUNT:
        raise ValueError("processor_video_metadata_mismatch")
    validate_processor_video_metadata(metadata, fps)
    batch = processor(text=[rendered], images=images, videos=[pixels],
        video_metadata=[metadata], do_sample_frames=False,
        cap_pixels_per_frame=True, padding=True, return_tensors="pt")
    return rendered, prompt_text, batch


def _video_groups(batch: dict, processor, model) -> list[tuple[int, ...]]:
    input_ids = batch.get("input_ids")
    grid = batch.get("video_grid_thw")
    if input_ids is None or grid is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("processor_video_geometry_missing")
    grid_values = [int(x) for x in grid[0].detach().cpu().tolist()]
    if len(grid_values) != 3 or any(x <= 0 for x in grid_values):
        raise ValueError("processor_video_grid_invalid")
    temporal_steps, height, width = grid_values
    merge = int(processor.image_processor.merge_size)
    temporal_patch = int(processor.video_processor.temporal_patch_size)
    if merge < 1 or temporal_patch < 1 or height % merge or width % merge:
        raise ValueError("processor_video_patch_geometry_invalid")
    ids = input_ids[0].detach().cpu().tolist()
    positions = [index for index, token in enumerate(ids)
                 if token == int(processor.video_token_id)]
    per_group = (height // merge) * (width // merge)
    if (len(ids) != int(input_ids.shape[1])
            or len(positions) != temporal_steps * per_group
            or positions != list(range(positions[0], positions[0] + len(positions)))
            or FRAME_COUNT % temporal_patch != 0
            or FRAME_COUNT // temporal_patch != temporal_steps):
        raise ValueError("processor_video_group_geometry_mismatch")
    groups = [tuple(positions[index * per_group:(index + 1) * per_group])
              for index in range(temporal_steps)]
    return groups


def _attention_selector(model, processor, rendered: str, question: str,
                        batch: dict, groups: list[tuple[int, ...]], torch) -> tuple[list[float], dict]:
    from qwen_question_temporal_attention_selector import (
        aggregate_temporal_group_attention, map_question_token_positions)

    question_positions = map_question_token_positions(
        tokenizer=processor.tokenizer, rendered_text=rendered,
        question_text=question, processed_input_ids=batch["input_ids"],
        video_token_id=int(processor.video_token_id))
    layer_scores: dict[int, list[float]] = {}
    handles = []
    target_layers = tuple(layer - 1 for layer in LAYERS_1_BASED)
    input_length = int(batch["input_ids"].shape[1])
    for layer_index in target_layers:
        attention = model.model.language_model.layers[layer_index].self_attn
        if not callable(getattr(attention, "register_forward_hook", None)):
            raise ValueError("selector_attention_hook_unavailable")

        def capture(_module, _inputs, output, *, index=layer_index):
            if (not isinstance(output, tuple) or len(output) < 2
                    or output[1] is None or output[1].ndim != 4
                    or int(output[1].shape[-2]) != input_length
                    or int(output[1].shape[-1]) != input_length):
                raise ValueError("selector_attention_capture_invalid")
            layer_scores[index] = aggregate_temporal_group_attention(
                attention_by_layer={index: output[1]}, layer_indices=(index,),
                question_positions=question_positions, group_positions=groups,
                torch_module=torch)

        handles.append(attention.register_forward_hook(capture))
    try:
        moved = {key: value.to("cuda") if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        with torch.inference_mode():
            # The pinned eager Qwen attention module returns per-layer weights
            # to its local hook even when output_attentions=False. Keep the
            # tensors transient per layer; asking the model output to retain
            # every full sequence attention map can exhaust VRAM.
            model(**moved, use_cache=False, output_attentions=False, logits_to_keep=1)
    finally:
        for handle in reversed(handles):
            handle.remove()
    if tuple(sorted(layer_scores)) != target_layers:
        raise ValueError("selector_attention_layer_capture_incomplete")
    means = [sum(layer_scores[layer][index] for layer in target_layers) / len(target_layers)
             for index in range(len(groups))]
    if any(not math.isfinite(value) or value < 0 for value in means):
        raise ValueError("selector_group_score_invalid")
    return means, {"query_token_count": len(question_positions),
                   "layer_indices_zero_based": list(target_layers),
                   "layer_capture_count": len(layer_scores),
                   "full_prefill_sequence_length": input_length}


def _observe_generation_shapes(model, layer_indices: tuple[int, ...]):
    observations = {index: [] for index in layer_indices}
    handles = []
    for index in layer_indices:
        attention = model.model.language_model.layers[index].self_attn

        def observe(_module, _args, kwargs, *, layer_index=index):
            mask = kwargs.get("attention_mask")
            if (mask is None or not hasattr(mask, "shape") or len(mask.shape) != 4
                    or tuple(mask.shape[:2]) != (1, 1)):
                raise ValueError("generation_shape_observation_invalid")
            observations[layer_index].append([int(mask.shape[-2]), int(mask.shape[-1])])

        handles.append(attention.register_forward_pre_hook(observe, with_kwargs=True))
    return observations, handles


def validate_dynamic_shape_trace(observations: dict, layer_indices: tuple[int, ...],
                                 prompt_length: int) -> list[list[int]]:
    traces = [observations.get(index) for index in layer_indices]
    if not traces or any(not isinstance(trace, list) or not trace for trace in traces):
        raise ValueError("dynamic_cache_shape_trace_empty")
    first = traces[0]
    if any(trace != first for trace in traces[1:]):
        raise ValueError("dynamic_cache_layer_shape_trace_disagreement")
    if first[0] != [prompt_length, prompt_length]:
        raise ValueError("dynamic_cache_prefill_shape_mismatch")
    previous_k = prompt_length
    for qlen, klen in first[1:]:
        if qlen != 1 or klen != previous_k + 1:
            raise ValueError("dynamic_cache_decode_shape_not_incremental")
        previous_k = klen
    return first


def validate_generation_mask_audit(observations: list[dict], trace: list[list[int]],
                                   layer_indices: tuple[int, ...], *,
                                   expected_applied: bool) -> dict:
    if not expected_applied:
        if observations:
            raise ValueError("baseline_intervention_mask_observed")
        return {"applied": False, "hook_calls": 0, "verified_cells": 0,
                "all_selected_cells_match_bias": True, "logit_bias": 0.0}
    expected_calls = len(trace) * len(layer_indices)
    if len(observations) != expected_calls or not observations:
        raise ValueError("intervention_mask_audit_call_count_mismatch")
    per_layer = {str(index): 0 for index in layer_indices}
    cells = 0
    effective_biases = set()
    for item in observations:
        layer = item.get("layer_index")
        if (type(layer) is not int or layer not in layer_indices
                or item.get("verified") is not True
                or type(item.get("modified_cells")) is not int
                or item["modified_cells"] < 1
                or not math.isfinite(float(item.get("effective_bias", float("nan"))))
                or not 0 < float(item["effective_bias"]) <= LOGIT_BIAS
                or not math.isfinite(float(item.get("max_abs_delta_error", float("nan"))))
                or float(item["max_abs_delta_error"]) > 1e-6):
            raise ValueError("intervention_mask_audit_value_mismatch")
        effective_biases.add(float(item["effective_bias"]))
        per_layer[str(layer)] += 1
        cells += item["modified_cells"]
    if any(value != len(trace) for value in per_layer.values()):
        raise ValueError("intervention_mask_audit_layer_count_mismatch")
    if len(effective_biases) != 1:
        raise ValueError("intervention_mask_audit_effective_bias_mismatch")
    return {"applied": True, "hook_calls": len(observations),
            "calls_per_layer": per_layer, "verified_cells": cells,
            "all_selected_cells_match_bias": True, "logit_bias": LOGIT_BIAS,
            "effective_mask_bias": next(iter(effective_biases))}


def _generate_one(*, model, processor, batch: dict, code_sequences: dict,
                  torch,
                  selected_positions: tuple[int, ...] | None,
                  prompt_length: int, layer_indices: tuple[int, ...]) -> tuple[str, dict]:
    from qwen_visual_attention_boost import (
        install_generation_video_key_bias_hooks, remove_hooks)

    observations, observer_handles = _observe_generation_shapes(model, layer_indices)
    mask_observations = []
    intervention_handles = []
    try:
        if selected_positions is not None:
            intervention_handles = install_generation_video_key_bias_hooks(
                model=model, layer_indices=layer_indices,
                selected_positions=selected_positions,
                first_query_position=prompt_length - 1,
                logit_bias=LOGIT_BIAS, torch_module=torch,
                observations=mask_observations)
        moved = {key: value.to("cuda") if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        free_vram, _total_vram = torch.cuda.mem_get_info()
        if free_vram < 2 * 1024**3 or _host_available_bytes() < 1024**3:
            raise RuntimeError("per_generation_resource_guard_failed")
        eos_token_id = processor.tokenizer.eos_token_id
        if (type(eos_token_id) is not int or eos_token_id < 0
                or eos_token_id not in generation_eos_ids(
                    model.generation_config.eos_token_id)):
            raise ValueError("interval_code_eos_token_invalid")
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            return interval_prefix_allowed_tokens(batch_id, input_ids,
                prompt_length=prompt_length, sequences=code_sequences,
                eos_token_id=eos_token_id)
        max_code_tokens = max(len(sequence) for sequence in code_sequences)
        with torch.inference_mode():
            generated = model.generate(
                **moved, do_sample=False, use_cache=True,
                cache_implementation="dynamic", max_new_tokens=max_code_tokens + 1,
                eos_token_id=eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                return_dict_in_generate=True,
                output_logits=True,
                pad_token_id=model.generation_config.pad_token_id)
        sequence = generated.sequences[0].detach().cpu().tolist()
        prompt_ids = batch["input_ids"][0].detach().cpu().tolist()
        if sequence[:len(prompt_ids)] != prompt_ids or len(sequence) <= len(prompt_ids):
            raise ValueError("generation_prefix_or_tail_invalid")
        tail = sequence[len(prompt_ids):]
        if not tail or tail[-1] != eos_token_id:
            raise ValueError("interval_code_generation_eos_missing")
        code_ids = tuple(tail[:-1])
        text = code_sequences.get(code_ids)
        if text is None or processor.tokenizer.decode(code_ids, skip_special_tokens=True) != text:
            raise ValueError("interval_code_generation_not_one_complete_code")
        trace = validate_dynamic_shape_trace(observations, layer_indices, prompt_length)
        mask_audit = validate_generation_mask_audit(
            mask_observations, trace, layer_indices,
            expected_applied=selected_positions is not None)
        logit_diagnostics = summarize_generation_logits(generated.logits, tail, torch)
        return text, {"shape_trace": trace, "per_layer_call_counts": {
            str(index): len(observations[index]) for index in layer_indices},
            "forward_calls": len(trace), "generated_token_count": len(tail),
            "cache_implementation": "dynamic",
            "constraint_code_count": len(code_sequences),
            "constraint_code_token_count": len(code_ids),
            "constraint_eos_terminated": True,
            "intervention_mask_audit": mask_audit,
            "raw_logit_diagnostics": logit_diagnostics}
    finally:
        remove_hooks(intervention_handles)
        for handle in reversed(observer_handles):
            handle.remove()


def _score_branch(*, text: str, duration: float, gold_intervals: list,
                  frame_pts: list[float]) -> dict:
    parsed = parse_interval_generation(text, duration, frame_pts)
    score = interval_iou(parsed["prediction"], gold_intervals, duration)
    return {"prediction": parsed["prediction"], "parse_status": parsed["status"],
            "temporal_iou": score["temporal_iou"], "score_status": score["status"],
            "generated_text_sha256": sha256_bytes(text.encode("utf-8"))}


def _run_record(*, record: dict, frames, decode: dict, processor, process_vision_info,
                model, torch, layer_indices: tuple[int, ...], branch_counts: dict) -> dict:
    from qwen_question_temporal_attention_selector import select_top_temporal_groups
    from claim_mask_video_decoder import verify_sampled_frame_pixels
    verify_sampled_frame_pixels(frames, decode)
    fps_ratio = decode["effective_sample_fps"]
    fps = fps_ratio[0] / fps_ratio[1]
    rendered, prompt_text, batch = _prompt(
        processor, record, frames, fps, process_vision_info, torch)
    groups = _video_groups(batch, processor, model)
    score_vector, selector_info = _attention_selector(
        model, processor, rendered, record["question"].strip(), batch, groups, torch)
    k = max(1, math.ceil(len(groups) * 0.10))
    selected_groups = tuple(select_top_temporal_groups(score_vector, top_k=k))
    token = sha256_bytes((record["video_sha256"] + str(record["dataset_question_id"])).encode())
    sham_groups = hash_sham_group_indices(group_count=len(groups), top_k=k,
                                          selected=selected_groups, record_token=token)
    selected_positions = tuple(sorted(position for i in selected_groups for position in groups[i]))
    sham_positions = tuple(sorted(position for i in sham_groups for position in groups[i]))
    if (not selected_positions or not sham_positions
            or len(selected_positions) != len(sham_positions)
            or set(selected_positions) & set(sham_positions)):
        raise ValueError("intervention_key_budget_or_overlap_invalid")
    prompt_length = int(batch["input_ids"].shape[1])
    prompt_ids = [int(value) for value in batch["input_ids"][0].detach().cpu().tolist()]
    code_sequences = interval_code_token_sequences(
        processor.tokenizer, FRAME_COUNT, rendered_prompt=rendered)
    if len(code_sequences) != FRAME_COUNT * (FRAME_COUNT + 1) // 2:
        raise ValueError("interval_code_grammar_count_mismatch")
    duration = _probe_duration_seconds(record["local_video_path"])
    frame_pts = [float(Fraction(*item["relative_presentation_time"]))
                 for item in decode["sampled_frames"]]
    gold = record["gold_interval_seconds"]
    if any(float(span[1]) > duration for span in gold):
        raise ValueError("gold_interval_exceeds_decoded_media_duration")
    outputs = {}
    traces = {}
    for branch, positions in (("baseline", None),
                              ("selected_temporal_bias", selected_positions),
                              ("hash_sham_bias", sham_positions)):
        branch_counts[branch]["attempted"] += 1
        text, audit = _generate_one(model=model, processor=processor, batch=batch,
            code_sequences=code_sequences, torch=torch,
            selected_positions=positions, prompt_length=prompt_length,
            layer_indices=layer_indices)
        branch_counts[branch]["completed"] += 1
        outputs[branch] = _score_branch(text=text, duration=duration,
            gold_intervals=gold, frame_pts=frame_pts)
        traces[branch] = audit
        del text
        torch.cuda.empty_cache()
    return {"video_sha256": record["video_sha256"],
        "question_sha256": sha256_bytes(record["question"].encode("utf-8")),
        "pair_sha256": record["candidate_pair_sha256"],
        "gold_qa_intervals_seconds": gold,
        "gold_qa_interval_count": len(gold),
        "prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
        "prompt_token_count": prompt_length,
        "sampled_frame_count": len(frames),
        "video_grid_thw": [int(x) for x in batch["video_grid_thw"][0].tolist()],
        "temporal_group_count": len(groups),
        "selector": {**selector_info, "scores": score_vector,
            "selected_group_indices": list(selected_groups),
            "hash_sham_group_indices": list(sham_groups),
            "selected_visual_key_count": len(selected_positions),
            "sham_visual_key_count": len(sham_positions),
            "selected_key_positions_sha256": sha256_bytes(json.dumps(
                selected_positions, separators=(",", ":")).encode()),
            "sham_key_positions_sha256": sha256_bytes(json.dumps(
                sham_positions, separators=(",", ":")).encode())},
        "condition_scores": outputs, "generation_audit": traces,
        "mechanistic_logits_comparison": compare_generation_logit_diagnostics(traces),
        "duration_seconds": duration}


def _probe_duration_seconds(path: str) -> float:
    import av
    with av.open(path, mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError("video_stream_count_invalid")
        stream = streams[0]
        if (stream.duration is not None and stream.time_base is not None
                and int(stream.duration) > 0):
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        else:
            raise ValueError("video_duration_unavailable")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video_duration_invalid")
    return duration


def _write_private_json(path: Path, payload: dict) -> None:
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _load_smoke_receipt(path: Path, private_root: Path, *, packet_sha256: str,
                        sources: dict[str, str]) -> dict:
    payload, raw = read_private_json(path, private_root)
    if (payload.get("mode") != "smoke" or payload.get("status") != "completed_exploratory"
            or payload.get("effective_n") != 1 or payload.get("requested_n") != 1
            or payload.get("packet_sha256") != packet_sha256
            or payload.get("source_sha256") != sources
            or payload.get("failures") != [] or payload.get("records") is None
            or len(payload["records"]) != 1):
        raise ValueError("successful_matching_smoke_receipt_required")
    validate_receipt(payload, requested_n=1, packet_sha256=packet_sha256)
    return {"sha256": sha256_bytes(raw), "name": path.name}


def run(mode: str, output_arg: str, *, smoke_receipt_arg: str | None = None) -> int:
    from private_output_paths import validate_private_output
    repo = Path(__file__).resolve().parents[1]
    output = validate_private_output(output_arg, repo)
    private = repo / "data" / "private_nextgqa"
    if output.parent != private or output.suffix != ".json":
        raise ValueError("output_must_be_json_child_of_private_data_root")
    packet, packet_raw, _ = _read_packet(repo)
    packet_sha = sha256_bytes(packet_raw)
    records = validate_packet(packet)
    n = 1 if mode == "smoke" else 3
    if mode not in ("smoke", "full"):
        raise ValueError("mode_invalid")
    if mode == "smoke" and smoke_receipt_arg is not None:
        raise ValueError("smoke_must_not_consume_smoke_receipt")
    source_hashes = _source_hashes(repo)
    smoke_binding = None
    if mode == "full":
        if not smoke_receipt_arg:
            raise ValueError("full_mode_requires_successful_smoke")
        smoke_path = Path(smoke_receipt_arg)
        smoke_binding = _load_smoke_receipt(smoke_path, private,
            packet_sha256=packet_sha, sources=source_hashes)
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    failures = []
    results = []
    model_metadata_hashes = {}
    model_weight_hashes = {}
    branch_counts = {name: {"requested": n, "attempted": 0, "completed": 0}
                      for name in BRANCHES}
    preflight = {"attempted": 0, "completed": 0}
    hardware = {}
    stage = "resource_preflight"
    try:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info()
        host_free = _host_available_bytes()
        hardware = {"device": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "free_vram_before_load_gib": free / 1024**3,
            "total_vram_gib": total / 1024**3,
            "host_available_before_load_gib": host_free / 1024**3}
        if free < MIN_FREE_VRAM or host_free < MIN_FREE_HOST:
            raise RuntimeError("resource_guard_before_model_load_failed")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
            / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise RuntimeError("pinned_local_model_checkpoint_missing")
        model_metadata_hashes, model_weight_hashes = _pinned_model_hashes(model_path)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from claim_mask_video_decoder import decode_sampled_video

        stage = "load_pinned_model"
        preflight["attempted"] += 1
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if (int(processor.video_token_id) != int(model.config.video_token_id)
                or getattr(model.config, "_attn_implementation", None) != "eager"):
            raise ValueError("model_processor_or_attention_backend_mismatch")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        layer_indices = tuple(layer - 1 for layer in LAYERS_1_BASED)
        preflight["completed"] += 1
        for index, record in enumerate(records[:n]):
            stage = "decode_video_and_verify_hash"
            video_path = Path(record["local_video_path"]).resolve(strict=True)
            if sha256_file(video_path) != record["video_sha256"]:
                raise ValueError("frozen_packet_video_hash_mismatch")
            frames, decode = decode_sampled_video(video_path, num_frames=FRAME_COUNT,
                uniform_tolerance_seconds=Fraction(1, 20))
            if decode.get("source_sha256") != record["video_sha256"]:
                raise ValueError("decoded_video_hash_mismatch")
            stage = "paired_interval_generation"
            row = _run_record(record=record, frames=frames, decode=decode,
                processor=processor, process_vision_info=process_vision_info,
                model=model, torch=torch, layer_indices=layer_indices,
                branch_counts=branch_counts)
            results.append(row)
            del frames, decode
            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        hardware["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    except Exception as exc:
        failures.append({"record_ordinal": len(results) + 1 if len(results) < n else None,
            "stage": stage, "failure_type": type(exc).__name__,
            "failure_code": str(exc).splitlines()[0][:180]})
    status = "completed_exploratory" if not failures and len(results) == n else "failed_incomplete_no_cohort_estimate"
    payload = {"schema_version": 1, "status": status, "mode": mode,
        "started_utc": started, "elapsed_seconds": time.monotonic() - start,
        "purpose": "paired greedy generation of shortest QA evidence interval under fixed temporal-key attention bias",
        "endpoint": "paired generated QA interval temporal IoU vs NExT-GQA QA interval",
        "endpoint_scope": "QA temporal localization only; not explanation-claim support",
        "time_origin_assumption": "generated timestamps and NExT-GQA intervals share the beginning-of-video origin; independent player-to-decoder clock calibration was not performed",
        "direct_faithfulness": False,
        "explanation_claim_support_established": False,
        "human_review": {"review_actually_performed": False,
            "collection_waived_for_local_exploratory_run": True},
        "rights_status": "unverified; local-only under user authorization",
        "authorization": "user-authorized-local-only",
        "packet_name": PACKET_NAME, "packet_sha256": packet_sha,
        "cohort_sha256": packet.get("cohort_sha256"),
        "source_sha256": source_hashes,
        "model_metadata_sha256": model_metadata_hashes,
        "model_weight_sha256": model_weight_hashes,
        "model": MODEL_NAME, "model_revision": MODEL_REVISION,
        "precision": "4-bit weights; BF16 compute; eager attention",
        "cache": {"implementation_requested": "dynamic",
            "shape_validation": "square prefill then Q=1, K increments by one at every cached decode step across layers 5-16"},
        "sampling": {"frames_per_video": FRAME_COUNT,
            "grid": "endpoint-inclusive uniform decoded-frame-index sample",
            "uniform_tolerance_seconds": [1, 20]},
        "selector": {"source": "untreated question-token attention aggregated over complete Qwen temporal groups, layers 5-16",
            "top_fraction": 0.10, "ties": "earliest temporal group",
            "treatment": {"layers_1_based": list(LAYERS_1_BASED), "logit_bias": LOGIT_BIAS},
            "sham": "equal-count deterministic SHA-256-ranked nonselected groups"},
        "generation": {"do_sample": False, "use_cache": True,
            "cache_implementation": "dynamic", "max_new_tokens": 6,
            "format": "prefix-constrained inclusive sampled-frame code SS,EE",
            "constraint_codes": FRAME_COUNT * (FRAME_COUNT + 1) // 2,
            "prompt_sha256_only": True, "raw_generation_text_saved": False,
            "prompt_protocol": PROMPT_PROTOCOL},
        "requested_n": n, "effective_n": len(results), "records": results,
        "generation_calls_requested": 3 * n,
        "generation_calls_attempted": sum(x["attempted"] for x in branch_counts.values()),
        "generation_calls_completed": sum(x["completed"] for x in branch_counts.values()),
        "branch_counts": branch_counts, "preflight": preflight,
        "failures": failures,
        "smoke_receipt_binding": smoke_binding,
        "summary": paired_summary(results, n),
        "versions": {name: _pkg_version(name) for name in
            ("torch", "transformers", "bitsandbytes", "av", "qwen-vl-utils", "Pillow")},
        "hardware": hardware,
        "inference": "finite frozen cohort descriptive only; no significance test, population CI, or generalization",
        "limitations": [
            "NExT-GQA intervals are QA annotations, not claim-specific explanation evidence.",
            "The timestamp clock is assumed to share the decoded-media beginning; no independent player-to-decoder calibration is performed.",
            "Question-attention temporal selection and positive key bias do not establish semantic evidence use.",
            "Greedy output and parsing are prompt/model dependent; invalid intervals score zero and remain in requested denominator.",
            "Three records are a fixed exploratory cohort; no confirmatory inference.",
            "Dataset rights status was not independently verified; artifacts remain local."]}
    validate_receipt(payload, requested_n=n, packet_sha256=packet_sha)
    _write_private_json(output, payload)
    return 0 if status == "completed_exploratory" else 2


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--output", required=True,
        help="new mode-0600 JSON directly under data/private_nextgqa")
    parser.add_argument("--smoke-receipt", default=None,
        help="required passing smoke receipt for --mode full")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit --user-authorized-local-only authorization required")
    try:
        return run(args.mode, args.output, smoke_receipt_arg=args.smoke_receipt)
    except FileExistsError:
        parser.error("refusing_to_overwrite_existing_output")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc).splitlines()[0][:180])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
