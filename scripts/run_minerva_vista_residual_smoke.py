#!/usr/bin/env python3
"""One-row VISTA-inspired residual-stream feasibility smoke for Minerva-Ego.

The CPU mode binds one frozen row, the official media/time-map preflight, the
exact 64-frame prompt batches, and anchor/sham adjacent frame pairs. The GPU
mode is a technical exploratory smoke only: it generates one baseline sentence
once, then compares teacher-forced object-name-token likelihood under baseline,
anchor steering, and temporal-sham steering. It does not establish faithfulness.
All receipts remain local under data/private_minerva.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import time

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"
COHORT_DEFAULT = PRIVATE / "minerva_metadata_proxy_anchor_cohort_v2_20261022_03.json"
COHORT_PREFLIGHT_DEFAULT = PRIVATE / "minerva_visible_object_v2_cohort_cpu_preflight_20261022_02.json"
BASELINE_RECEIPT_DEFAULT = PRIVATE / "minerva_visible_object_v2_generation_bias_row0_gpu_20261022_01.json"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
FRAME_COUNT = 64
BLOCK_INDEX_ZERO_BASED = 13  # block 14, frozen in docs/image_to_video_intervention_options_20261022.md
DOSE_RMS_FRACTION = 0.10
SOURCE_FILES = (
    "scripts/run_minerva_vista_residual_smoke.py",
    "scripts/run_minerva_generated_object_pixel_pilot.py",
    "scripts/minerva_object_pixel_utils.py",
    "scripts/minerva_ego_anchor_utils.py",
    "scripts/preflight_minerva_grounding_cohort.py",
    "scripts/run_minerva_ego_anchor_gpu_pilot.py",
    "scripts/run_claim_support_qta_16frame_exploratory.py",
    "scripts/qta_multiquery_video_key_intervention.py",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_root() -> None:
    if not PRIVATE.is_dir() or PRIVATE.is_symlink() or PRIVATE.stat().st_mode & 0o077:
        raise ValueError("private_root_contract_invalid")


def _private_path(path: Path, *, must_exist: bool) -> None:
    _private_root()
    if path.is_symlink():
        raise ValueError("private_path_symlink_rejected")
    if must_exist and not path.is_file():
        raise ValueError("private_input_missing")
    resolved = path.resolve(strict=must_exist)
    if not resolved.is_relative_to(PRIVATE.resolve()):
        raise ValueError("private_path_outside_root")


def _private_bytes(path: Path, max_bytes: int = 50_000_000) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > max_bytes):
            raise ValueError("private_input_contract_invalid")
        chunks, total = [], 0
        while True:
            block = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > max_bytes:
                raise ValueError("private_input_too_large")
        if total != info.st_size:
            raise ValueError("private_input_changed_while_reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _write_private(path: Path, payload: dict) -> None:
    _private_path(path, must_exist=False)
    if path.parent.resolve() != PRIVATE.resolve():
        raise ValueError("private_output_must_be_direct_child")
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _pair_start(rank: int, *, frame_count: int = FRAME_COUNT) -> int:
    if type(rank) is not int or type(frame_count) is not int or frame_count != 64 or not 0 <= rank < frame_count:
        raise ValueError("pair_rank_or_frame_count_invalid")
    return (rank // 2) * 2


def _choose_sham_group(anchor_group: int, *, group_count: int = FRAME_COUNT // 2) -> int:
    if (type(anchor_group) is not int or type(group_count) is not int
            or group_count < 2 or not 0 <= anchor_group < group_count):
        raise ValueError("anchor_or_group_count_invalid")
    return min((group for group in range(group_count) if group != anchor_group),
               key=lambda group: (-abs(group - anchor_group), group))


def _dose_vector(full_vector, filled_vector, *, torch, fraction: float = DOSE_RMS_FRACTION):
    """Return full-minus-filled vector scaled to a fixed fraction of full RMS."""
    if (not torch.is_tensor(full_vector) or not torch.is_tensor(filled_vector)
            or tuple(full_vector.shape) != tuple(filled_vector.shape)
            or full_vector.ndim != 1 or not full_vector.numel()
            or not math.isfinite(float(fraction)) or fraction <= 0):
        raise ValueError("residual_vector_contract_invalid")
    full = full_vector.detach().to(dtype=torch.float32)
    filled = filled_vector.detach().to(dtype=torch.float32)
    raw = full - filled
    full_rms = torch.sqrt(torch.mean(full.square()))
    raw_rms = torch.sqrt(torch.mean(raw.square()))
    if (not bool(torch.isfinite(full).all()) or not bool(torch.isfinite(filled).all())
            or not bool(torch.isfinite(full_rms)) or not bool(torch.isfinite(raw_rms))
            or float(full_rms) <= 0 or float(raw_rms) <= 0):
        raise ValueError("residual_vector_zero_or_nonfinite")
    target_rms = full_rms * float(fraction)
    vector = raw * (target_rms / raw_rms)
    measured = torch.sqrt(torch.mean(vector.square()))
    if not bool(torch.isfinite(vector).all()) or not math.isclose(
            float(measured), float(target_rms), rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError("residual_vector_dose_validation_failed")
    return vector, {
        "full_residual_rms_fp32": float(full_rms),
        "raw_full_minus_filled_rms_fp32": float(raw_rms),
        "target_delta_rms_fp32": float(target_rms),
        "normalized_delta_rms_fp32": float(measured),
        "dose_fraction_of_full_residual_rms": float(fraction),
    }


def _cosine(a, b, *, torch) -> float:
    a, b = a.detach().to(dtype=torch.float32), b.detach().to(dtype=torch.float32)
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if (not bool(torch.isfinite(denominator)) or float(denominator) <= 0
            or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all())):
        raise ValueError("vector_cosine_zero_or_nonfinite")
    value = torch.dot(a, b) / denominator
    if not bool(torch.isfinite(value)):
        raise ValueError("vector_cosine_nonfinite")
    return float(value)


def _target_name_token_positions(text: str, target_ids: list[int], object_name: str,
                                 tokenizer) -> tuple[int, ...]:
    """Map the alias span onto the exact generated IDs using prefix decodes.

    Re-tokenizing a decoded continuation can change BPE boundaries. Instead,
    each raw generated prefix must decode monotonically into the final string;
    ambiguous/non-monotonic mappings fail closed.
    """
    if not isinstance(text, str) or not text.strip() or not isinstance(object_name, str) or not object_name:
        raise ValueError("generated_target_or_name_invalid")
    if not isinstance(target_ids, list) or not target_ids:
        raise ValueError("generated_target_token_ids_invalid")
    intervals = []
    previous = ""
    for index in range(len(target_ids)):
        prefix = tokenizer.decode(target_ids[:index + 1], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
        if not isinstance(prefix, str) or not prefix.startswith(previous):
            raise ValueError("generated_prefix_decode_nonmonotonic")
        intervals.append((len(previous), len(prefix)))
        previous = prefix
    if previous != text:
        raise ValueError("generated_prefix_decode_final_text_mismatch")
    matches = list(re.finditer(re.escape(object_name), text, flags=re.IGNORECASE))
    if len(matches) != 1:
        raise ValueError("generated_object_name_occurrence_invalid")
    start, end = matches[0].span()
    positions = tuple(i for i, (left, right) in enumerate(intervals)
                      if right > left and left < end and right > start)
    if not positions or tuple(range(positions[0], positions[-1] + 1)) != positions:
        raise ValueError("object_name_token_span_invalid")
    return positions


def _score_name_tokens(*, logits, target_ids: list[int], prompt_len: int,
                       target_positions: tuple[int, ...], torch):
    expected_queries = tuple(prompt_len + i - 1 for i in range(len(target_ids)))
    if (not target_ids or not target_positions or prompt_len < 1
            or any(type(i) is not int or not 0 <= i < len(target_ids) for i in target_positions)
            or logits.ndim != 3 or logits.shape[0] != 1
            or logits.shape[1] != len(target_ids) + 1):
        raise ValueError("teacher_forcing_query_shape_invalid")
    selected = logits[0, list(target_positions), :].to(dtype=torch.float32)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("target_logits_nonfinite")
    labels = torch.tensor([target_ids[i] for i in target_positions], device=selected.device,
                          dtype=torch.long)
    logprobs = torch.log_softmax(selected, dim=-1).gather(1, labels[:, None]).squeeze(1)
    if not bool(torch.isfinite(logprobs).all()):
        raise ValueError("target_logprob_nonfinite")
    return {"object_name_token_count": len(target_positions),
            "target_token_indices": list(target_positions),
            "absolute_query_positions": [expected_queries[i] for i in target_positions],
            "mean_object_name_logprob_nats_per_token": float(logprobs.mean().item()),
            "object_name_token_logprobs": [float(x) for x in logprobs.detach().cpu().tolist()]}


def _prepare_batches(prepared: dict) -> dict:
    """Create baseline and two full-frame-pair mean-fill batches on CPU."""
    import numpy as np
    import torch
    from qwen_vl_utils import process_vision_info
    frames = prepared["frames"]
    if len(frames) != FRAME_COUNT:
        raise ValueError("frozen_sample_must_have_64_frames")
    anchor_start = _pair_start(prepared["anchor_rank"])
    anchor_group = anchor_start // 2
    sham_group = _choose_sham_group(anchor_group)
    sham_start = sham_group * 2
    if sham_start == anchor_start or sham_start + 1 >= FRAME_COUNT:
        raise ValueError("temporal_sham_pair_not_disjoint")

    def pair_fill(source_frames, start):
        result = [frame.copy() for frame in source_frames]
        for rank in (start, start + 1):
            frame = result[rank]
            from minerva_object_pixel_utils import mean_fill_box
            result[rank] = mean_fill_box(frame, (0, 0, frame.width, frame.height))
        return result

    videos_by_condition = {"baseline": frames,
        "anchor_pair_mean_fill": pair_fill(frames, anchor_start),
        "temporal_sham_pair_mean_fill": pair_fill(frames, sham_start)}
    # Derive text from the same frozen row/protocol used by the existing pilot.
    from run_minerva_generated_object_pixel_pilot import _prompt
    prompt = _prompt(prepared["row"], prepared["object_name"], prepared["anchor_clip_seconds"])
    processor = prepared["processor"]
    rendered = None
    batches = {}
    for label, condition_frames in videos_by_condition.items():
        messages = [{"role": "user", "content": [
            {"type": "video", "video": condition_frames,
             "sample_fps": prepared["record"]["clip_sampling"]["sample_fps"]},
            {"type": "text", "text": prompt}]}]
        candidate_rendered = processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True)
        if rendered is None:
            rendered = candidate_rendered
        elif candidate_rendered != rendered:
            raise ValueError("condition_prompt_changed")
        if prepared.get("rendered_prompt") != rendered:
            raise ValueError("prompt_differs_from_frozen_cpu_prepare")
        images, videos = process_vision_info(messages, return_video_metadata=True)
        if images is not None or not isinstance(videos, list) or len(videos) != 1:
            raise ValueError("qwen_video_utility_contract_invalid")
        pixels, metadata = videos[0]
        if metadata.get("frames_indices") != list(range(FRAME_COUNT)):
            raise ValueError("processor_frame_indices_changed")
        batch = processor(text=[rendered], images=images, videos=[pixels],
            video_metadata=[metadata], do_sample_frames=False,
            cap_pixels_per_frame=True, padding=True, return_tensors="pt")
        batches[label] = batch
    base = batches["baseline"]
    if (base["input_ids"].shape[0] != 1 or not bool((base["attention_mask"] == 1).all())
            or not torch.equal(base["input_ids"], prepared["batches"]["base"]["input_ids"])
            or base["input_ids"].shape[1] != prepared["batches"]["base"]["input_ids"].shape[1]):
        raise ValueError("single_unpadded_same_prompt_required")
    grid = [int(x) for x in base["video_grid_thw"][0].tolist()]
    for label, batch in batches.items():
        if [int(x) for x in batch["video_grid_thw"][0].tolist()] != grid:
            raise ValueError("condition_video_grid_changed")
        if tuple(batch["pixel_values_videos"].shape) != tuple(base["pixel_values_videos"].shape):
            raise ValueError("condition_video_tensor_shape_changed")
        for key, base_value in base.items():
            if key == "pixel_values_videos":
                continue
            value = batch.get(key)
            if torch.is_tensor(base_value) and (not torch.is_tensor(value)
                    or not torch.equal(base_value, value)):
                raise ValueError("condition_nonvisual_input_changed")
    pixels_hashes = {label: _sha(batch["pixel_values_videos"].detach().cpu().contiguous()
        .view(torch.uint8).numpy().tobytes()) for label, batch in batches.items()}
    if len(set(pixels_hashes.values())) != 3:
        raise ValueError("condition_pixel_inputs_not_distinct")
    return {"batches": batches, "rendered_prompt_sha256": _sha(rendered.encode()),
            "input_ids_sha256": _sha(json.dumps(base["input_ids"][0].tolist(),
                separators=(",", ":")).encode("ascii")),
            "pixel_values_sha256": pixels_hashes, "video_grid_thw": grid,
            "anchor_pair_sample_ranks": [anchor_start, anchor_start + 1],
            "sham_pair_sample_ranks": [sham_start, sham_start + 1],
            "anchor_pair_source_indices": [prepared["sampled"][i]["decoded_frame_index"]
                                            for i in (anchor_start, anchor_start + 1)],
            "sham_pair_source_indices": [prepared["sampled"][i]["decoded_frame_index"]
                                          for i in (sham_start, sham_start + 1)],
            "anchor_pair_pts_seconds": [prepared["sampled"][i]["time_seconds"]
                                         for i in (anchor_start, anchor_start + 1)],
            "sham_pair_pts_seconds": [prepared["sampled"][i]["time_seconds"]
                                       for i in (sham_start, sham_start + 1)],
            "frame_count": FRAME_COUNT, "prompt_token_count": int(base["input_ids"].shape[1]),
            "video_tensor_shape": list(base["pixel_values_videos"].shape),
            "control_group_index": sham_group,
            "control_selection_rule": "farthest non-anchor adjacent pair by absolute group-index distance; ties select lower group index; no outcome-dependent choice"}


def _source_hashes() -> dict:
    return {Path(path).name: _sha_file(ROOT / path) for path in SOURCE_FILES}


def _receipt(prepared: dict, tensors: dict, row_index: int,
             baseline_receipt_raw: bytes, baseline_receipt: dict) -> dict:
    return {"schema_version": 1, "status": "completed_vista_residual_cpu_preflight",
        "experiment_scope": "single_row_technical_smoke_only",
        "rights_status": "unverified; local-only per user authorization",
        "human_review_performed": False, "direct_generated_explanation_faithfulness": False,
        "cohort_sha256": _sha(prepared["cohort_raw"]),
        "cohort_cpu_preflight_sha256": _sha(prepared["cohort_preflight_raw"]),
        "generation_mode": "prior_receipt_bound" if baseline_receipt is not None else "fresh_in_run_greedy",
        "prior_baseline_receipt_sha256": _sha(baseline_receipt_raw) if baseline_receipt is not None else None,
        "prior_baseline_generation_token_ids_sha256": (baseline_receipt["outcomes"]["baseline"]["generated_token_ids_sha256"]
            if baseline_receipt is not None else None),
        "prior_baseline_generation_text_sha256": (baseline_receipt["outcomes"]["baseline"]["generated_text_sha256"]
            if baseline_receipt is not None else None),
        "row_index": row_index, "row_key_sha256": _sha(prepared["row_key"].encode()),
        "video_sha256": prepared["video_sha256"], "time_map_sha256": prepared["time_map_sha256"],
        "annotation_source_sha256": prepared["annotation_source_sha256"],
        "model_revision": MODEL_REVISION,
        "source_sha256": _source_hashes(), "runner_sha256": _sha_file(Path(__file__)),
        "frame_count": FRAME_COUNT, "prompt_token_count": tensors["prompt_token_count"],
        "input_ids_sha256": tensors["input_ids_sha256"],
        "rendered_prompt_sha256": tensors["rendered_prompt_sha256"],
        "video_grid_thw": tensors["video_grid_thw"],
        "video_tensor_shape": tensors["video_tensor_shape"],
        "pixel_values_sha256": tensors["pixel_values_sha256"],
        "anchor_sample_rank": prepared["anchor_rank"],
        "anchor_pair_sample_ranks": tensors["anchor_pair_sample_ranks"],
        "anchor_pair_source_indices": tensors["anchor_pair_source_indices"],
        "anchor_pair_pts_seconds": tensors["anchor_pair_pts_seconds"],
        "temporal_sham_group_index": tensors["control_group_index"],
        "temporal_sham_selection_rule": tensors["control_selection_rule"],
        "sham_pair_sample_ranks": tensors["sham_pair_sample_ranks"],
        "sham_pair_source_indices": tensors["sham_pair_source_indices"],
        "sham_pair_pts_seconds": tensors["sham_pair_pts_seconds"],
        "fixed_block_1_based": BLOCK_INDEX_ZERO_BASED + 1,
        "fixed_block_index_zero_based": BLOCK_INDEX_ZERO_BASED,
        "dose_rms_fraction": DOSE_RMS_FRACTION,
        "sampling": {key: prepared["record"]["clip_sampling"][key] for key in
            ("sample_frame_indices", "sample_fps", "clip_start_source_seconds",
             "clip_end_source_seconds", "anchor_sample_rank", "maximum_grid_time_error_seconds")},
        "limitations": ["one-row engineering smoke", "does not establish efficacy or faithfulness",
            "global mean-fill is out-of-distribution",
            "farthest-endpoint temporal sham maximizes temporal separation and is not representative or exchangeable"]}


def _validate_receipt(receipt: dict, raw: bytes, expected: dict) -> None:
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"cpu_receipt_binding_mismatch:{key}")
    if receipt.get("status") != "completed_vista_residual_cpu_preflight" or not raw:
        raise ValueError("cpu_receipt_status_or_hash_invalid")


def _batch_to_cuda(batch, torch):
    return {key: value.to("cuda") if torch.is_tensor(value) else value
            for key, value in batch.items()}


def _forward_capture(model, batch_cpu, *, target_ids: list[int], torch,
                     prompt_len: int, vector=None):
    """One inference forward; hook captures or steers exactly prompt position P-1."""
    batch = _batch_to_cuda(batch_cpu, torch)
    layer = model.model.language_model.layers[BLOCK_INDEX_ZERO_BASED]
    hook_count_before = len(layer._forward_hooks)
    calls = {"count": 0, "capture": None, "dose": None}
    input_sequence_length = int(batch["input_ids"].shape[1])
    target_len = len(target_ids)
    if input_sequence_length != prompt_len:
        raise ValueError("unextended_prompt_length_mismatch")
    sequence_length = prompt_len + target_len

    from run_claim_support_qta_16frame_exploratory import _read_host_available_bytes
    free_vram, _ = torch.cuda.mem_get_info()
    if free_vram < 2 * 1024**3 or _read_host_available_bytes() < 3 * 1024**3:
        raise RuntimeError("per_forward_resource_guard_failed")

    def hook(_module, _inputs, output):
        calls["count"] += 1
        if calls["count"] != 1 or not torch.is_tensor(output):
            raise ValueError("residual_hook_call_or_output_invalid")
        if tuple(output.shape) != (1, sequence_length, 3584):
            raise ValueError("residual_hook_shape_invalid")
        if not bool(torch.isfinite(output[0, prompt_len - 1]).all()):
            raise ValueError("captured_residual_nonfinite")
        calls["capture"] = output[0, prompt_len - 1].detach().to(dtype=torch.float32).cpu()
        if vector is None:
            return output
        injected = output.clone()
        delta = vector.to(device=output.device, dtype=output.dtype)
        before = output[0, prompt_len - 1].detach().to(dtype=torch.float32)
        injected[0, prompt_len - 1, :] = output[0, prompt_len - 1, :] + delta
        after = injected[0, prompt_len - 1, :].detach().to(dtype=torch.float32)
        actual_delta = after - before
        before_rms = torch.sqrt(torch.mean(before.square()))
        actual_rms = torch.sqrt(torch.mean(actual_delta.square()))
        if (not bool(torch.isfinite(injected[0, prompt_len - 1]).all())
                or not bool(torch.isfinite(actual_rms)) or float(actual_rms) <= 0
                or not bool(torch.isfinite(before_rms)) or float(before_rms) <= 0):
            raise ValueError("injected_residual_nonfinite_or_zero")
        calls["dose"] = {"pre_update_rms_fp32": float(before_rms),
            "post_update_rms_fp32": float(torch.sqrt(torch.mean(after.square()))),
            "effective_displacement_rms_fp32": float(actual_rms),
            "effective_displacement_fraction_of_pre_rms": float(actual_rms / before_rms)}
        return injected

    handle = layer.register_forward_hook(hook)
    try:
        from qta_multiquery_video_key_intervention import append_target_to_unpadded_batch
        extended, queries = append_target_to_unpadded_batch(
            batch=batch, target_token_ids=target_ids, torch_module=torch)
        if queries != tuple(range(prompt_len - 1, prompt_len + target_len - 1)):
            raise ValueError("teacher_forcing_query_offsets_invalid")
        with torch.inference_mode():
            output = model(**extended, use_cache=False, logits_to_keep=target_len + 1)
        if calls["count"] != 1 or calls["capture"] is None:
            raise ValueError("residual_hook_count_not_one")
    finally:
        handle.remove()
        calls["hook_registry_before"] = hook_count_before
        calls["hook_registry_after_removal"] = len(layer._forward_hooks)
    if calls["hook_registry_after_removal"] != hook_count_before:
        raise ValueError("residual_hook_removal_failed")
    return output, calls, queries


def _gpu_smoke(prepared: dict, tensors: dict, *, torch, cpu_receipt_raw: bytes,
               baseline_receipt: dict, baseline_receipt_raw: bytes,
               progress: dict) -> dict:
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
        / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        raise ValueError("pinned_checkpoint_missing")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("cuda_bf16_unavailable")
    free, total = torch.cuda.mem_get_info()
    from run_claim_support_qta_16frame_exploratory import _read_host_available_bytes
    host_available = _read_host_available_bytes()
    if free < 12 * 1024**3 or host_available < 8 * 1024**3:
        raise ValueError("initial_gpu_or_host_memory_guard_failed")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    progress["model_loaded"] = True
    layer_count = len(model.model.language_model.layers)
    if layer_count != 28 or not 0 <= BLOCK_INDEX_ZERO_BASED < layer_count:
        raise ValueError("decoder_block_count_or_index_invalid")
    base_cpu = tensors["batches"]["baseline"]
    prompt_len = int(base_cpu["input_ids"].shape[1])
    device_base = _batch_to_cuda(base_cpu, torch)
    torch.manual_seed(20261022)
    torch.cuda.manual_seed_all(20261022)
    torch.cuda.reset_peak_memory_stats()
    if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
        raise RuntimeError("per_forward_free_vram_guard_failed")
    started = time.monotonic()
    progress["baseline_generation_attempted"] = True
    with torch.inference_mode():
        generation = model.generate(**device_base, do_sample=False, num_beams=1,
            use_cache=True, cache_implementation="dynamic", max_new_tokens=64,
            return_dict_in_generate=True,
            output_logits=False)
    progress["baseline_generation_completed"] = True
    sequences = generation.sequences[0].detach().cpu().tolist()
    prompt_ids = base_cpu["input_ids"][0].tolist()
    if sequences[:len(prompt_ids)] != prompt_ids:
        raise ValueError("generation_prompt_prefix_invalid")
    target_ids = sequences[len(prompt_ids):]
    decoded_text = prepared["processor"].tokenizer.decode(target_ids,
        skip_special_tokens=True, clean_up_tokenization_spaces=False)
    text = decoded_text.strip()
    saved_baseline = (baseline_receipt or {}).get("outcomes", {}).get("baseline", {})
    target_ids_sha256 = _sha(json.dumps(target_ids, separators=(",", ":")).encode("ascii"))
    target_text_sha256 = _sha(text.encode("utf-8"))
    if baseline_receipt is not None and (
            saved_baseline.get("mechanical_diagnostic", {}).get("status") != "mechanically_valid"
            or saved_baseline.get("generated_token_ids_sha256") != target_ids_sha256
            or saved_baseline.get("generated_text_sha256") != target_text_sha256):
        raise ValueError("greedy_generation_differs_from_frozen_baseline_receipt")
    from run_minerva_generated_object_pixel_pilot import _generated_target
    _generated_target(text, target_ids, prepared["object_name"])
    target_alias = prepared["object_name"].split()[-1]
    target_positions = _target_name_token_positions(decoded_text, target_ids,
        target_alias, prepared["processor"].tokenizer)
    progress["frozen_target_validated"] = True
    del generation, device_base
    torch.cuda.empty_cache()

    # A no-op baseline hook both captures full-input residual and supplies baseline logits.
    baseline_output, baseline_hook, queries = _forward_capture(model,
        tensors["batches"]["baseline"], target_ids=target_ids, torch=torch,
        prompt_len=prompt_len)
    progress["baseline_capture_score_completed"] = True
    if tuple(queries) != tuple(range(prompt_len - 1, prompt_len + len(target_ids) - 1)):
        raise ValueError("teacher_forcing_query_offsets_invalid")
    base_vector = baseline_hook["capture"]
    scores = {"baseline": _score_name_tokens(logits=baseline_output.logits,
        target_ids=target_ids, prompt_len=prompt_len, target_positions=target_positions,
        torch=torch)}
    del baseline_output
    torch.cuda.empty_cache()

    captured = {}
    capture_audit = {"baseline": {"hook_call_count": baseline_hook["count"],
        "hook_registry_before": baseline_hook["hook_registry_before"],
        "hook_registry_after_removal": baseline_hook["hook_registry_after_removal"]}}
    for label in ("anchor_pair_mean_fill", "temporal_sham_pair_mean_fill"):
        output, calls, masked_queries = _forward_capture(model, tensors["batches"][label],
            target_ids=target_ids, torch=torch, prompt_len=prompt_len)
        if masked_queries != queries:
            raise ValueError("masked_capture_query_offsets_changed")
        captured[label] = calls["capture"]
        progress[label + "_capture_completed"] = True
        capture_audit[label] = {"hook_call_count": calls["count"], "forward_only": True,
            "hook_registry_before": calls["hook_registry_before"],
            "hook_registry_after_removal": calls["hook_registry_after_removal"]}
        del output
        torch.cuda.empty_cache()

    vectors, vector_audit = {}, {}
    for label, key in (("anchor_steering", "anchor_pair_mean_fill"),
                       ("temporal_sham_steering", "temporal_sham_pair_mean_fill")):
        vector, audit = _dose_vector(base_vector, captured[key], torch=torch)
        vectors[label] = vector
        vector_audit[label] = audit
    vector_audit["anchor_sham_cosine_raw"] = _cosine(
        base_vector - captured["anchor_pair_mean_fill"],
        base_vector - captured["temporal_sham_pair_mean_fill"], torch=torch)
    score_hook_audit = {}
    for label, vector_name in (("anchor_steering", "anchor_steering"),
                               ("temporal_sham_steering", "temporal_sham_steering")):
        output, calls, steered_queries = _forward_capture(model, tensors["batches"]["baseline"],
            target_ids=target_ids, torch=torch, prompt_len=prompt_len,
            vector=vectors[vector_name])
        if steered_queries != queries or calls["count"] != 1:
            raise ValueError("steered_forward_query_or_hook_count_invalid")
        scores[label] = _score_name_tokens(logits=output.logits,
            target_ids=target_ids, prompt_len=prompt_len,
            target_positions=target_positions, torch=torch)
        progress[label + "_score_completed"] = True
        score_hook_audit[label] = calls["dose"] | {"hook_call_count": calls["count"],
            "hook_registry_before": calls["hook_registry_before"],
            "hook_registry_after_removal": calls["hook_registry_after_removal"]}
        del output
        torch.cuda.empty_cache()
    interaction = ((scores["anchor_steering"]["mean_object_name_logprob_nats_per_token"]
                    - scores["baseline"]["mean_object_name_logprob_nats_per_token"])
                   - (scores["temporal_sham_steering"]["mean_object_name_logprob_nats_per_token"]
                      - scores["baseline"]["mean_object_name_logprob_nats_per_token"]))
    torch.cuda.synchronize()
    result = {"schema_version": 1, "status": "completed_vista_residual_one_row_gpu_technical_smoke",
        "rights_status": "unverified; local-only per user authorization",
        "human_review_performed": False, "direct_generated_explanation_faithfulness": False,
        "cohort_sha256": _sha(prepared["cohort_raw"]),
        "cohort_cpu_preflight_sha256": _sha(prepared["cohort_preflight_raw"]),
        "cpu_receipt_sha256": _sha(cpu_receipt_raw),
        "generation_mode": "prior_receipt_bound" if baseline_receipt is not None else "fresh_in_run_greedy",
        "prior_baseline_receipt_sha256": _sha(baseline_receipt_raw) if baseline_receipt is not None else None,
        "row_index": prepared["row_index"], "row_key_sha256": _sha(prepared["row_key"].encode()),
        "video_sha256": prepared["video_sha256"], "time_map_sha256": prepared["time_map_sha256"],
        "model_revision": MODEL_REVISION, "runtime_versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "transformers": importlib.metadata.version("transformers"),
            "qwen-vl-utils": importlib.metadata.version("qwen-vl-utils"),
            "gpu_name": torch.cuda.get_device_name(0)},
        "source_sha256": _source_hashes(), "runner_sha256": _sha_file(Path(__file__)),
        "frame_count": FRAME_COUNT, "video_grid_thw": tensors["video_grid_thw"],
        "prompt_token_count": prompt_len, "target_token_count": len(target_ids),
        "target_token_ids_sha256": _sha(json.dumps(target_ids, separators=(",", ":")).encode("ascii")),
        "target_text_sha256": _sha(text.encode("utf-8")),
        "object_name_alias_sha256": _sha(target_alias.casefold().encode("utf-8")),
        "object_name_token_count": len(target_positions),
        "object_name_token_indices": list(target_positions),
        "fixed_block_index_zero_based": BLOCK_INDEX_ZERO_BASED,
        "fixed_block_1_based": BLOCK_INDEX_ZERO_BASED + 1,
        "dose_rms_fraction": DOSE_RMS_FRACTION,
        "model_call_counts": {"generate_api_invocations": 1,
            "explicit_teacher_forced_model_forward_calls": 5,
            "top_level_model_invocations": 6,
            "generation_internal_decoder_forward_count": "not instrumented"},
        "teacher_forced_forward_counts": {"baseline_capture_and_score": 1,
            "anchor_mask_capture": 1, "sham_mask_capture": 1,
            "anchor_steered_score": 1, "sham_steered_score": 1,
            "total": 5},
        "capture_hook_audit": capture_audit, "score_hook_audit": score_hook_audit,
        "vector_dose_audit": vector_audit,
        "teacher_forcing_query_positions": list(queries),
        "object_name_target_query_positions": [prompt_len + index - 1 for index in target_positions],
        "mean_object_name_logprob_nats_per_token": {key:
            value["mean_object_name_logprob_nats_per_token"] for key, value in scores.items()},
        "delta_anchor_minus_baseline_nats_per_token":
            scores["anchor_steering"]["mean_object_name_logprob_nats_per_token"] -
            scores["baseline"]["mean_object_name_logprob_nats_per_token"],
        "delta_sham_minus_baseline_nats_per_token":
            scores["temporal_sham_steering"]["mean_object_name_logprob_nats_per_token"] -
            scores["baseline"]["mean_object_name_logprob_nats_per_token"],
        "anchor_minus_sham_change_nats_per_token": interaction,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "initial_free_vram_bytes": free, "visible_total_vram_bytes": total,
        "limitations": ["one-row technical smoke, not an efficacy test",
            "predicted object-name token likelihood is not semantic support or faithfulness",
            "whole-pair global mean-fill can be out of distribution",
            "temporal sham is matched only by frozen temporal-group selection"]}
    del model
    torch.cuda.empty_cache()
    return result


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu-preflight", "gpu-smoke"), required=True)
    parser.add_argument("--cohort", type=Path, default=COHORT_DEFAULT)
    parser.add_argument("--cohort-preflight", type=Path, default=COHORT_PREFLIGHT_DEFAULT)
    parser.add_argument("--baseline-receipt", type=Path, default=None,
        help="prior mechanically-valid row0 greedy-generation receipt; reproduced target hashes must match "
             "(default for row 0 unless --fresh-generation)")
    parser.add_argument("--fresh-generation", action="store_true",
        help="fresh-cohort mode: no prior receipt; greedy target is generated in-run and must pass the "
             "mechanical object-name validation, otherwise the failure is retained without replacement")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--time-map", type=Path, required=True)
    parser.add_argument("--minerva-json", type=Path, default=Path("/tmp/minerva_ego_20260205.json"))
    parser.add_argument("--assoc-info", type=Path, default=Path("/tmp/hd_epic_assoc_info.json"))
    parser.add_argument("--mask-info", type=Path, default=Path("/tmp/hd_epic_mask_info.json"))
    parser.add_argument("--manifest", type=Path, default=Path("/tmp/hd-epic-downloader/data/md5.txt"))
    parser.add_argument("--cpu-receipt", type=Path,
        help="mode-0600 receipt from --mode cpu-preflight; required for GPU smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.fresh_generation:
        if args.baseline_receipt is not None or args.row_index < 0:
            raise ValueError("fresh_generation_excludes_prior_receipt")
        if args.cohort == COHORT_DEFAULT or args.cohort_preflight == COHORT_PREFLIGHT_DEFAULT:
            raise ValueError("fresh_generation_requires_explicit_new_cohort")
    else:
        if args.row_index != 0:
            raise ValueError("technical_smoke_is_frozen_to_row0")
        if args.baseline_receipt is None:
            args.baseline_receipt = BASELINE_RECEIPT_DEFAULT
    _private_path(args.cohort, must_exist=True)
    _private_path(args.cohort_preflight, must_exist=True)
    if args.baseline_receipt is not None:
        _private_path(args.baseline_receipt, must_exist=True)
    _private_path(args.video, must_exist=True)
    _private_path(args.time_map, must_exist=True)
    _private_path(args.output, must_exist=False)
    if args.output.exists() or (args.cpu_receipt is not None and args.output.resolve() == args.cpu_receipt.resolve()):
        raise ValueError("output_must_be_new_and_distinct")
    if args.mode == "cpu-preflight" and args.cpu_receipt is not None:
        raise ValueError("cpu_preflight_cannot_consume_receipt")
    if args.mode == "gpu-smoke":
        if args.cpu_receipt is None:
            raise ValueError("gpu_mode_requires_cpu_receipt")
        _private_path(args.cpu_receipt, must_exist=True)
        if args.output.resolve() == args.cpu_receipt.resolve():
            raise ValueError("gpu_output_must_differ_from_cpu_receipt")
    for path in (args.minerva_json, args.assoc_info, args.mask_info, args.manifest):
        if path.is_symlink() or not path.is_file():
            raise ValueError("local_annotation_or_manifest_missing")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from run_minerva_generated_object_pixel_pilot import _prepare
    prepared = _prepare(cohort_path=args.cohort, cohort_preflight_path=args.cohort_preflight,
        row_index=args.row_index, video_path=args.video, time_map_path=args.time_map,
        minerva_json_path=args.minerva_json, assoc_info_path=args.assoc_info,
        mask_info_path=args.mask_info, manifest_path=args.manifest)
    tensors = _prepare_batches(prepared)
    import torch
    baseline_pixel_hash = prepared["pixel_tensor_sha256"].get("base")
    if tensors["pixel_values_sha256"].get("baseline") != baseline_pixel_hash:
        raise ValueError("baseline_video_tensor_differs_from_frozen_preflight")
    if args.fresh_generation:
        baseline_receipt_raw, baseline_receipt = b"", None
    else:
        baseline_receipt_raw = _private_bytes(args.baseline_receipt)
        baseline_receipt = json.loads(baseline_receipt_raw)
    baseline = (baseline_receipt or {}).get("outcomes", {}).get("baseline", {})
    if baseline_receipt is not None and (baseline_receipt.get("status") != "completed_minerva_visible_object_generation_bias_v1"
            or baseline_receipt.get("cohort_sha256") != _sha(prepared["cohort_raw"])
            or baseline_receipt.get("row_key_sha256") != _sha(prepared["row_key"].encode())
            or baseline_receipt.get("row_index") != args.row_index
            or baseline_receipt.get("prompt_sha256") != tensors["rendered_prompt_sha256"]
            or baseline.get("mechanical_diagnostic", {}).get("status") != "mechanically_valid"):
        raise ValueError("prior_baseline_receipt_binding_invalid")
    expected = _receipt(prepared, tensors, args.row_index,
        baseline_receipt_raw, baseline_receipt)
    if args.mode == "cpu-preflight":
        _write_private(args.output, expected)
        print(json.dumps({"status": expected["status"], "row_index": args.row_index,
            "frame_count": FRAME_COUNT, "video_grid_thw": tensors["video_grid_thw"],
            "anchor_pair_sample_ranks": tensors["anchor_pair_sample_ranks"],
            "sham_pair_sample_ranks": tensors["sham_pair_sample_ranks"],
            "receipt": str(args.output)}))
        return 0
    raw_receipt = _private_bytes(args.cpu_receipt)
    _validate_receipt(json.loads(raw_receipt), raw_receipt, expected)
    progress = {"model_loaded": False, "baseline_generation_attempted": False,
        "baseline_generation_completed": False, "frozen_target_validated": False,
        "baseline_capture_score_completed": False,
        "anchor_pair_mean_fill_capture_completed": False,
        "temporal_sham_pair_mean_fill_capture_completed": False,
        "anchor_steering_score_completed": False,
        "temporal_sham_steering_score_completed": False}
    gpu_started = time.monotonic()
    try:
        result = _gpu_smoke(prepared, tensors, torch=torch, cpu_receipt_raw=raw_receipt,
            baseline_receipt=baseline_receipt, baseline_receipt_raw=baseline_receipt_raw,
            progress=progress)
    except Exception as exc:
        failure_code = (str(exc) if isinstance(exc, ValueError)
                        and re.fullmatch(r"[a-z0-9_:]{1,200}", str(exc))
                        else type(exc).__name__)
        try:
            peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else None
        except Exception:
            peak = None
        failure = {"schema_version": 1, "status": "failed_vista_residual_one_row_gpu_technical_smoke",
            "rights_status": "unverified; local-only per user authorization",
            "human_review_performed": False, "direct_generated_explanation_faithfulness": False,
            "cohort_sha256": _sha(prepared["cohort_raw"]),
            "cohort_cpu_preflight_sha256": _sha(prepared["cohort_preflight_raw"]),
            "cpu_receipt_sha256": _sha(raw_receipt),
            "prior_baseline_receipt_sha256": _sha(baseline_receipt_raw) if baseline_receipt is not None else None,
            "row_index": args.row_index, "row_key_sha256": _sha(prepared["row_key"].encode()),
            "source_sha256": _source_hashes(), "runner_sha256": _sha_file(Path(__file__)),
            "failure_type": type(exc).__name__, "failure_code": failure_code,
            "completed_stages": progress, "elapsed_seconds": time.monotonic() - gpu_started,
            "peak_vram_bytes": peak}
        _write_private(args.output, failure)
        raise
    _write_private(args.output, result)
    print(json.dumps({"status": result["status"], "row_index": args.row_index,
        "object_name_token_count": result["object_name_token_count"],
        "anchor_minus_sham_change_nats_per_token": result["anchor_minus_sham_change_nats_per_token"],
        "peak_vram_bytes": result["peak_vram_bytes"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
