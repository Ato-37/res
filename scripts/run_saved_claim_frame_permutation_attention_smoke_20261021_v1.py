#!/usr/bin/env python3
"""Run one private real-video exploratory claim-token sensitivity smoke.

The first row of the fixed 2026-10-19 claim-review smoke packet is used. Its
saved parsed claim string is deterministically re-tokenized with the pinned
tokenizer; these IDs are not the original IDs emitted during generation. Four
teacher-forced scores compare natural vs temporal-block-reversed video with
baseline attention and a fixed +ln(2) bias on all visual keys in Qwen layers
5--16. Human review is neither required nor claimed to have occurred. This is
a conditional language-model sensitivity diagnostic, not a faithfulness,
claim-truth, or hallucination result.

Example (GPU execution, local-only):
  .venv/bin/python scripts/run_saved_claim_frame_permutation_attention_smoke.py \\
    --output data/private_nextgqa/saved_claim_frame_permutation_attention_smoke_20261021_01.json \\
    --user-authorized-local-only
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import time

from private_output_paths import validate_private_output

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
PACKET_REL = Path("data/private_nextgqa/claim_review_val_exploratory_gpu_smoke_local_20261019_01.json")
FRAME_COUNT = 16
LAYERS_1_BASED = tuple(range(5, 17))
LOGIT_BIAS = math.log(2.0)
MIN_FREE_VRAM = 12 * 1024**3
MIN_HOST_AVAILABLE = 8 * 1024**3
MIN_FORWARD_FREE_VRAM = 2 * 1024**3
LABELS = "ABCDE"


def deterministic_reverse_block_indices(frame_count: int,
                                         temporal_patch_size: int) -> tuple[int, ...]:
    """Reverse temporal-patch-sized frame blocks; preserve order within blocks."""
    if (type(frame_count) is not int or frame_count < 2
            or type(temporal_patch_size) is not int or temporal_patch_size < 1
            or frame_count % temporal_patch_size != 0
            or frame_count // temporal_patch_size < 2):
        raise ValueError("frame_block_permutation_geometry_invalid")
    blocks = [tuple(range(start, start + temporal_patch_size))
              for start in range(0, frame_count, temporal_patch_size)]
    result = tuple(index for block in reversed(blocks) for index in block)
    if result == tuple(range(frame_count)) or sorted(result) != list(range(frame_count)):
        raise ValueError("frame_block_permutation_not_a_derangement")
    return result


def reorder_frames(frames, indices: tuple[int, ...]):
    """Copy frames into a validated permutation, without mutating the source."""
    if (not isinstance(frames, (list, tuple)) or not frames
            or not isinstance(indices, tuple) or len(indices) != len(frames)
            or any(type(index) is not int for index in indices)
            or sorted(indices) != list(range(len(frames)))):
        raise ValueError("frame_permutation_indices_invalid")
    output = []
    for index in indices:
        frame = frames[index]
        copier = getattr(frame, "copy", None)
        if not callable(copier):
            raise ValueError("frame_copy_unavailable")
        output.append(copier())
    return output


def compute_bias_frame_order_interaction(*, original_baseline: float,
        permuted_baseline: float, original_biased: float,
        permuted_biased: float) -> dict:
    """Difference-in-differences on mean target-token log probability."""
    values = (original_baseline, permuted_baseline,
              original_biased, permuted_biased)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in values):
        raise ValueError("branch_logprob_must_be_finite")
    baseline_order_effect = float(original_baseline) - float(permuted_baseline)
    biased_order_effect = float(original_biased) - float(permuted_biased)
    return {
        "original_minus_permuted_baseline_nats_per_token": baseline_order_effect,
        "original_minus_permuted_biased_nats_per_token": biased_order_effect,
        "bias_by_frame_order_interaction_nats_per_token": (
            biased_order_effect - baseline_order_effect),
        "interpretation": (
            "single-row sensitivity interaction for re-tokenized saved claim text; not faithfulness"),
    }


def validate_biased_mask_rows(attention_mask, *, key_positions: tuple[int, ...],
                              query_positions: tuple[int, ...], bias: float,
                              torch) -> bool:
    """Check that every instrumented visual-key/query cell carries the bias."""
    if (not torch.is_tensor(attention_mask) or attention_mask.ndim != 4
            or tuple(attention_mask.shape[:2]) != (1, 1)
            or not key_positions or not query_positions
            or not math.isfinite(float(bias)) or bias <= 0):
        raise ValueError("biased_mask_audit_shape_or_parameters_invalid")
    rows = torch.tensor(query_positions, dtype=torch.long,
                        device=attention_mask.device)
    cols = torch.tensor(key_positions, dtype=torch.long,
                        device=attention_mask.device)
    observed = attention_mask[0, 0][rows[:, None], cols[None, :]]
    expected = torch.full_like(observed, float(bias))
    if not torch.allclose(observed, expected, rtol=0.0, atol=1e-6):
        raise ValueError("biased_mask_audit_value_mismatch")
    return True


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value, torch) -> str:
    contiguous = value.detach().to(device="cpu").contiguous()
    raw = contiguous.view(torch.uint8).numpy().tobytes()
    description = json.dumps({"dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape)}, sort_keys=True,
        separators=(",", ":")).encode("ascii")
    return _sha256_bytes(description + b"\0" + raw)


def _host_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("host_memory_probe_unavailable")


def _read_packet(repo: Path) -> tuple[dict, str]:
    path = repo / PACKET_REL
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("claim_packet_permissions_invalid")
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
    if (payload.get("status") != "passed" or payload.get("dataset_split") != "val"
            or payload.get("model_revision") != MODEL_REVISION
            or payload.get("claim_review_completed") is not False
            or not isinstance(payload.get("records"), list)
            or not payload["records"]):
        raise ValueError("claim_packet_contract_invalid")
    return payload["records"][0], _sha256_bytes(raw)


def _build_prompt(processor, record: dict, frames: list, sample_fps: float,
                  prompt_suffix: str) -> tuple[str, list]:
    choices = record.get("choices")
    question = record.get("question")
    if (not isinstance(question, str) or not question.strip()
            or not isinstance(choices, list) or len(choices) != 5
            or any(not isinstance(choice, str) or not choice.strip() for choice in choices)):
        raise ValueError("claim_generation_prompt_source_invalid")
    prompt = (question + "\nOptions:\n"
        + "\n".join(f"{label}. {choice}" for label, choice in zip(LABELS, choices, strict=True))
        + prompt_suffix)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames, "sample_fps": sample_fps},
        {"type": "text", "text": prompt},
    ]}]
    rendered = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True)
    return rendered, messages


def _total_frame_count_matches(value, expected: int = FRAME_COUNT) -> bool:
    try:
        return math.isclose(float(value), float(expected), rel_tol=1e-7, abs_tol=1e-7)
    except (TypeError, ValueError):
        return False


def _prepare_processor_batch(processor, process_vision_info, record: dict,
        frames: list, sample_fps: float, prompt_suffix: str, torch):
    rendered, messages = _build_prompt(processor, record, frames,
                                       sample_fps, prompt_suffix)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or not isinstance(videos, list) or len(videos) != 1:
        raise ValueError("qwen_video_utility_contract_invalid")
    video_pixels, video_metadata = videos[0]
    total_frames_matches = _total_frame_count_matches(
        video_metadata.get("total_num_frames"))
    if (not torch.is_tensor(video_pixels)
            or int(video_pixels.shape[0]) != FRAME_COUNT
            or video_metadata.get("frames_indices") != list(range(FRAME_COUNT))
            or not total_frames_matches
            or not math.isclose(float(video_metadata.get("fps", 0)), sample_fps,
                                rel_tol=1e-7, abs_tol=1e-7)):
        raise ValueError("qwen_video_metadata_mismatch")
    batch = processor(text=[rendered], images=images, videos=[video_pixels],
        video_metadata=[video_metadata], do_sample_frames=False,
        cap_pixels_per_frame=True, padding=True, return_tensors="pt")
    return batch, video_metadata


def _move_tensors(batch: dict, torch) -> dict:
    return {key: value.to("cuda") if torch.is_tensor(value) else value
            for key, value in batch.items()}


def _score_branch(*, model, batch_cpu: dict, target_ids: tuple[int, ...],
                  video_token_id: int, biased: bool, torch) -> dict:
    from qta_multiquery_video_key_intervention import (
        append_target_to_unpadded_batch,
        install_multiquery_video_key_intervention_hooks,
        remove_hooks,
        score_target_logits_window,
    )
    batch = _move_tensors(batch_cpu, torch)
    prompt_length = int(batch["input_ids"].shape[1])
    extended, query_positions = append_target_to_unpadded_batch(
        batch=batch, target_token_ids=target_ids, torch_module=torch)
    visual_positions_tensor = (batch["input_ids"][0] == video_token_id).nonzero(
        as_tuple=True)[0]
    visual_positions = tuple(int(x) for x in visual_positions_tensor.detach().cpu().tolist())
    if (not visual_positions or not query_positions
            or max(visual_positions) >= min(query_positions)):
        raise ValueError("visual_keys_not_causal_for_claim_tokens")
    hooks = []
    audit_hooks = []
    hook_counts = {layer: 0 for layer in LAYERS_1_BASED}
    try:
        if biased:
            hooks = install_multiquery_video_key_intervention_hooks(
                model=model,
                layer_indices=tuple(layer - 1 for layer in LAYERS_1_BASED),
                biased_positions=visual_positions,
                ablated_positions=(), query_positions=query_positions,
                logit_bias=LOGIT_BIAS, torch_module=torch)
            layers = model.model.language_model.layers
            for layer_1_based in LAYERS_1_BASED:
                layer_index = layer_1_based - 1
                def audit_hook(_module, args, kwargs, *, _layer=layer_1_based):
                    del args
                    if "attention_mask" not in kwargs:
                        raise ValueError("biased_mask_audit_attention_mask_missing")
                    validate_biased_mask_rows(kwargs["attention_mask"],
                        key_positions=visual_positions,
                        query_positions=query_positions, bias=LOGIT_BIAS,
                        torch=torch)
                    hook_counts[_layer] += 1
                audit_hooks.append(layers[layer_index].self_attn.register_forward_pre_hook(
                    audit_hook, with_kwargs=True))
        free, _total = torch.cuda.mem_get_info()
        if free < MIN_FORWARD_FREE_VRAM or _host_available_bytes() < MIN_HOST_AVAILABLE:
            raise RuntimeError("per_forward_resource_guard_failed")
        with torch.inference_mode():
            output = model(**extended, use_cache=False,
                           logits_to_keep=len(target_ids) + 1)
        logits = output.logits
        if (logits.ndim != 3
                or tuple(logits.shape[:2]) != (1, len(target_ids) + 1)
                or not bool(torch.isfinite(logits).all())):
            raise ValueError("target_logits_window_invalid")
        result = score_target_logits_window(logits=logits,
            target_token_ids=target_ids, prompt_length=prompt_length,
            first_query_position=prompt_length - 1, torch_module=torch)
        if biased and any(count != 1 for count in hook_counts.values()):
            raise ValueError("biased_hook_invocation_count_mismatch")
        result["intervention_hook_invocations_by_layer_1_based"] = hook_counts
        result["visual_key_position_count"] = len(visual_positions)
        result["visual_key_positions_sha256"] = _sha256_bytes(json.dumps(
            visual_positions, separators=(",", ":")).encode("ascii"))
        del output, logits
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return result
    finally:
        for handle in reversed(audit_hooks):
            handle.remove()
        remove_hooks(hooks)
        del extended, batch
        torch.cuda.empty_cache()


def _write_receipt_exclusive(output: Path, payload: dict) -> None:
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def run(output_arg: str) -> int:
    repo = Path(__file__).resolve().parents[1]
    output = validate_private_output(output_arg, repo)
    if output.parent != repo / "data" / "private_nextgqa" or output.suffix != ".json":
        raise ValueError("private_output_must_be_json_child_of_private_data_root")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    stage = "private_input_validation"
    row_token = None
    packet_hash = None
    video_hash = None
    attempted = completed = 0
    try:
        record, packet_hash = _read_packet(repo)
        review = record.get("human_review")
        claim = record.get("generated_parsed", {}).get("claim")
        video_path = record.get("local_video_path")
        video_hash = record.get("video_sha256")
        if (not isinstance(claim, str) or not claim.strip()
                or not isinstance(review, dict)
                or review.get("claim_visually_supported_by_reported_frames") is not None
                or not isinstance(video_path, str)
                or not isinstance(video_hash, str) or len(video_hash) != 64):
            raise ValueError("first_claim_packet_row_invalid")
        row_token = _sha256_bytes((video_hash + str(record.get("dataset_question_id", "")))
                                  .encode("utf-8"))

        stage = "cuda_resource_preflight"
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info()
        host_free = _host_available_bytes()
        if free < MIN_FREE_VRAM or host_free < MIN_HOST_AVAILABLE:
            raise ValueError("pre_model_resource_guard_failed")

        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
            / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise ValueError("pinned_local_checkpoint_missing")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  Qwen2_5_VLForConditionalGeneration)
        from qwen_vl_utils import process_vision_info
        from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
        from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX

        stage = "load_processor_and_decode_video"
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        if int(processor.video_token_id) < 0:
            raise ValueError("processor_video_token_id_invalid")
        frames, decoded = decode_sampled_video(video_path, num_frames=FRAME_COUNT,
            uniform_tolerance_seconds=Fraction(1, 20))
        verify_sampled_frame_pixels(frames, decoded)
        if decoded["source_sha256"] != video_hash:
            raise ValueError("claim_video_digest_mismatch")
        packet_times = record.get("sampled_frame_relative_pts_seconds")
        decoded_times = [float(Fraction(*row["relative_presentation_time"]))
                         for row in decoded["sampled_frames"]]
        if (not isinstance(packet_times, list) or len(packet_times) != FRAME_COUNT
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(float(value)) for value in packet_times)
                or len(decoded_times) != FRAME_COUNT
                or any(abs(float(saved) - actual) > 1e-6
                       for saved, actual in zip(packet_times, decoded_times, strict=True))):
            raise ValueError("saved_claim_sampling_grid_mismatch")
        sample_fps = Fraction(*decoded["effective_sample_fps"])
        if len(frames) != FRAME_COUNT:
            raise ValueError("decoded_frame_count_invalid")
        temporal_patch_size = int(processor.video_processor.temporal_patch_size)
        permutation = deterministic_reverse_block_indices(FRAME_COUNT, temporal_patch_size)
        permuted_frames = reorder_frames(frames, permutation)

        stage = "prepare_original_and_permuted_processor_batches"
        original_cpu, original_meta = _prepare_processor_batch(
            processor, process_vision_info, record, frames,
            float(sample_fps), PROMPT_SUFFIX, torch)
        permuted_cpu, permuted_meta = _prepare_processor_batch(
            processor, process_vision_info, record, permuted_frames,
            float(sample_fps), PROMPT_SUFFIX, torch)
        if (not torch.equal(original_cpu["input_ids"], permuted_cpu["input_ids"])
                or not torch.equal(original_cpu["video_grid_thw"], permuted_cpu["video_grid_thw"])
                or not torch.equal(original_cpu["second_per_grid_ts"], permuted_cpu["second_per_grid_ts"])
                or original_meta != permuted_meta):
            raise ValueError("permuted_processor_geometry_or_timing_changed")
        if set(original_cpu) != set(permuted_cpu):
            raise ValueError("permuted_batch_key_set_changed")
        for key in original_cpu:
            a, b = original_cpu[key], permuted_cpu[key]
            if torch.is_tensor(a) and torch.is_tensor(b):
                if tuple(a.shape) != tuple(b.shape):
                    raise ValueError("permuted_batch_tensor_shape_changed")
                if key != "pixel_values_videos" and not torch.equal(a, b):
                    raise ValueError("permuted_nonvisual_tensor_changed")
            elif a != b:
                raise ValueError("permuted_non_tensor_batch_field_changed")
        temporal_grid = int(original_cpu["video_grid_thw"][0][0])
        if temporal_grid != FRAME_COUNT // temporal_patch_size:
            raise ValueError("processor_temporal_grid_mismatch")
        original_pixels_hash = _tensor_sha256(original_cpu["pixel_values_videos"], torch)
        permuted_pixels_hash = _tensor_sha256(permuted_cpu["pixel_values_videos"], torch)
        if original_pixels_hash == permuted_pixels_hash:
            raise ValueError("frame_permutation_did_not_change_processor_pixels")

        target_encoding = processor.tokenizer(claim, add_special_tokens=False)
        target_ids = tuple(target_encoding["input_ids"])
        if (not target_ids or any(type(token) is not int or token < 0 for token in target_ids)
                or len(target_ids) > 512):
            raise ValueError("saved_claim_retokenization_invalid")
        target_ids_hash = _sha256_bytes(json.dumps(target_ids,
            separators=(",", ":")).encode("ascii"))
        claim_hash = _sha256_bytes(claim.encode("utf-8"))

        stage = "load_pinned_scorer"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("model_processor_video_token_mismatch")
        if len(model.model.language_model.layers) < max(LAYERS_1_BASED):
            raise ValueError("pinned_model_layer_count_invalid")
        torch.manual_seed(20261021)
        torch.cuda.manual_seed_all(20261021)
        torch.cuda.reset_peak_memory_stats()

        branch_scores = {}
        branch_order = (("original_baseline", original_cpu, False),
            ("permuted_baseline", permuted_cpu, False),
            ("original_visual_bias", original_cpu, True),
            ("permuted_visual_bias", permuted_cpu, True))
        stage = "four_teacher_forced_forwards"
        for name, batch, biased in branch_order:
            attempted += 1
            branch_scores[name] = _score_branch(model=model, batch_cpu=batch,
                target_ids=target_ids, video_token_id=int(processor.video_token_id),
                biased=biased, torch=torch)
            completed += 1
        torch.cuda.synchronize()
        interaction = compute_bias_frame_order_interaction(
            original_baseline=branch_scores["original_baseline"]["target_logprob_mean"],
            permuted_baseline=branch_scores["permuted_baseline"]["target_logprob_mean"],
            original_biased=branch_scores["original_visual_bias"]["target_logprob_mean"],
            permuted_biased=branch_scores["permuted_visual_bias"]["target_logprob_mean"])
        source_files = [Path(__file__).resolve(),
            repo / "scripts" / "qta_multiquery_video_key_intervention.py",
            repo / "scripts" / "generate_nextgqa_claim_review_pilot.py",
            repo / "scripts" / "claim_mask_video_decoder.py"]
        checkpoint_hashes = {name: _sha256_file(model_path / name)
            for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json")
            if (model_path / name).is_file()}
        payload = {
            "schema_version": 1,
            "status": "completed_one_row_exploratory_sensitivity_smoke",
            "started_utc": started_utc,
            "scope": "re-tokenized saved-claim conditional logprob sensitivity to temporal-block reversal and fixed visual-key bias",
            "direct_faithfulness": False,
            "faithfulness_result": False,
            "claim_truth_reviewed": False,
            "human_review": {"review_actually_performed": False,
                "collection_waived_by_user_for_this_exploratory_run": True},
            "rights_status": "unverified; local-only per user authorization",
            "requested_records": 1, "effective_records": 1,
            "requested_forwards": 4, "attempted_forwards": attempted,
            "completed_forwards": completed,
            "row_token_sha256": row_token,
            "private_input_packet_sha256": packet_hash,
            "source_video_sha256": video_hash,
            "saved_claim_text_sha256": claim_hash,
            "reencoded_claim_token_ids_sha256": target_ids_hash,
            "target_token_count": len(target_ids),
            "target_id_semantics": "deterministic_tokenizer_encoding_of_saved_parsed_claim_string; not original generation emission IDs; no special tokens added",
            "prompt_template": "question + A-E choices + pinned PROMPT_SUFFIX; video-first; add_generation_prompt=True",
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": MODEL_REVISION,
            "checkpoint_config_sha256": checkpoint_hashes,
            "precision": "4-bit weights; BF16 compute; eager attention",
            "seed": 20261021,
            "frames": {"requested": FRAME_COUNT, "effective": len(frames),
                "sampling_fps_numerator": sample_fps.numerator,
                "sampling_fps_denominator": sample_fps.denominator,
                "temporal_patch_size": temporal_patch_size,
                "sampled_frame_source_index_permutation": list(permutation),
                "permutation_rule": "reverse complete temporal_patch_size blocks, preserve within-block frame order",
                "decoded_sample_manifest_sha256": _sha256_bytes(json.dumps(
                    decoded["sampled_frames"], sort_keys=True,
                    separators=(",", ":")).encode("utf-8")),
                "original_processor_pixels_sha256": original_pixels_hash,
                "permuted_processor_pixels_sha256": permuted_pixels_hash,
                "video_grid_thw": original_cpu["video_grid_thw"][0].tolist(),
                "second_per_grid_ts": [float(x) for x in original_cpu["second_per_grid_ts"].tolist()],
                "metadata_frames_indices": original_meta.get("frames_indices"),
                "metadata_fps": float(original_meta["fps"]),
                "metadata_total_num_frames": int(original_meta["total_num_frames"])},
            "attention_intervention": {"layers_1_based": list(LAYERS_1_BASED),
                "logit_bias": LOGIT_BIAS,
                "key_set": "all processor-expanded video-token positions, fixed identically across all branches",
                "query_set": "all appended re-encoded saved-claim target token prediction rows",
                "baseline_bias": 0.0},
            "branch_target_scores": branch_scores,
            "primary_descriptive_interaction": interaction,
            "hardware": {"device": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
                "free_vram_before_load_gib": free / 1024**3,
                "total_vram_gib": total / 1024**3,
                "host_available_before_load_gib": host_free / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
            "versions": {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")},
            "source_sha256": {str(path.relative_to(repo)): _sha256_file(path)
                for path in source_files},
            "elapsed_seconds": time.monotonic() - started,
            "inferential_status": "single row; descriptive only; no population inference",
            "limitations": [
                "saved parsed sentence is re-tokenized; original emitted token IDs were not retained",
                "frame-order reversal changes chronology and may be out-of-distribution",
                "uniform bias on all visual keys is a coarse internal intervention",
                "score is conditional text likelihood, not claim truth, visual support, or complete explanation faithfulness",
                "one fixed row cannot establish a population effect",
                "human review was waived for this exploratory run but was not performed",
                "rights status remains unverified; inputs and output remain local-only"]}
        _write_receipt_exclusive(output, payload)
        print(json.dumps({"status": payload["status"], "requested_records": 1,
            "effective_records": 1, "completed_forwards": completed,
            "target_token_count": len(target_ids),
            "bias_by_frame_order_interaction_nats_per_token": interaction[
                "bias_by_frame_order_interaction_nats_per_token"],
            "output_name": output.name}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema_version": 1,
            "status": "failed_incomplete_no_effect_estimate",
            "scope": "re-tokenized saved-claim frame-permutation attention smoke",
            "failure_stage": stage,
            "failure_type": type(exc).__name__,
            "failure_code": ("run_failed_ValueError" if isinstance(exc, ValueError)
                else "run_failed_" + type(exc).__name__),
            "requested_records": 1, "effective_records": 0,
            "requested_forwards": 4, "attempted_forwards": attempted,
            "completed_forwards": completed,
            "direct_faithfulness": False, "faithfulness_result": False,
            "claim_truth_reviewed": False,
            "human_review": {"review_actually_performed": False,
                "collection_waived_by_user_for_this_exploratory_run": True},
            "rights_status": "unverified; local-only per user authorization",
            "private_input_packet_sha256": packet_hash,
            "source_video_sha256": video_hash,
            "row_token_sha256": row_token,
            "elapsed_seconds": time.monotonic() - started,
            "started_utc": started_utc}
        try:
            _write_receipt_exclusive(output, failure)
        except Exception:
            pass
        print(json.dumps({"status": failure["status"],
            "failure_stage": stage, "failure_type": type(exc).__name__,
            "output_name": output.name}, sort_keys=True))
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
        help="new local mode-0600 receipt directly under data/private_nextgqa/")
    parser.add_argument("--user-authorized-local-only", action="store_true",
        help="confirm local-only use; rights status remains unverified")
    args = parser.parse_args()
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    try:
        return run(args.output)
    except Exception as exc:
        parser.error(str(exc) if type(exc) is ValueError else type(exc).__name__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
