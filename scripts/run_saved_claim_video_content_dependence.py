#!/usr/bin/env python3
"""Exploratory likelihood sensitivity to real versus constant-gray video.

This local-only diagnostic uses fixed saved-claim rows and exactly two
teacher-forced forwards per invocation. It does not score claim truth or
faithfulness. See experiments/2026-10-21_saved_claim_video_content_dependence_plan.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from private_output_paths import validate_private_output
from run_saved_claim_frame_permutation_attention_smoke import (
    FRAME_COUNT, MODEL_REVISION, MIN_FREE_VRAM, MIN_HOST_AVAILABLE,
    PACKET_REL, REMAINING_PACKET_REL, _host_available_bytes, _read_packet,
    _prepare_processor_batch, _score_branch, _sha256_bytes, _sha256_file,
    _tensor_sha256,
)


def make_constant_gray_frames(frames, gray: int = 127):
    """Return same-size RGB frames with all source content removed."""
    if type(gray) is not int or not 0 <= gray <= 255:
        raise ValueError("gray_value_invalid")
    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError("decoded_frames_invalid")
    output = []
    for frame in frames:
        if not hasattr(frame, "size") or not hasattr(frame, "mode"):
            raise ValueError("decoded_frame_shape_unavailable")
        from PIL import Image
        width, height = frame.size
        if width < 1 or height < 1:
            raise ValueError("decoded_frame_dimensions_invalid")
        output.append(Image.new("RGB", (width, height), (gray, gray, gray)))
    return output


def content_logprob_contrast(original: float, gray: float) -> float:
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(float(x)) for x in (original, gray)):
        raise ValueError("branch_logprob_must_be_finite")
    return float(original) - float(gray)


def resolve_claim_prompt_suffix(record: dict, suffixes: dict[str, str]) -> tuple[str, str]:
    """Resolve the exact versioned generation prompt recorded on a claim row."""
    version = record.get("generation_prompt_version", "v1")
    if not isinstance(version, str) or version not in suffixes:
        raise ValueError("claim_generation_prompt_version_invalid")
    suffix = suffixes[version]
    if not isinstance(suffix, str) or not suffix:
        raise ValueError("claim_generation_prompt_suffix_invalid")
    return version, suffix


def _hash_tensor(value, torch) -> str:
    return _sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def run(output_arg: str, packet_rel: Path, record_index: int) -> int:
    repo = Path(__file__).resolve().parents[1]
    output = validate_private_output(output_arg, repo)
    if output.parent != repo / "data" / "private_nextgqa" or output.suffix != ".json":
        raise ValueError("private_output_must_be_json_child_of_private_data_root")
    start = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    attempted = completed = 0
    stage = "private_input_validation"
    try:
        record, packet_hash = _read_packet(repo, packet_rel, record_index,
                                           allow_partial=True)
        claim = record.get("generated_parsed", {}).get("claim")
        video_path = record.get("local_video_path")
        video_hash = record.get("video_sha256")
        review = record.get("human_review")
        if (not isinstance(claim, str) or not claim.strip()
                or not isinstance(video_path, str) or not isinstance(video_hash, str)
                or len(video_hash) != 64 or not isinstance(review, dict)
                or review.get("claim_visually_supported_by_reported_frames") is not None):
            raise ValueError("claim_packet_row_invalid")
        row_token = _sha256_bytes((video_hash + str(record.get("dataset_question_id", "")))
                                  .encode("utf-8"))

        stage = "cuda_resource_preflight"
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info()
        host_free = _host_available_bytes()
        if free < MIN_FREE_VRAM or host_free < MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_model_resource_guard_failed")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
            / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise ValueError("pinned_local_checkpoint_missing")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
        from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_BY_VERSION

        stage = "decode_and_bind_video"
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        frames, decoded = decode_sampled_video(video_path, num_frames=FRAME_COUNT,
            uniform_tolerance_seconds=Fraction(1, 20))
        verify_sampled_frame_pixels(frames, decoded)
        if decoded["source_sha256"] != video_hash:
            raise ValueError("claim_video_digest_mismatch")
        saved_pts = record.get("sampled_frame_relative_pts_seconds")
        decoded_pts = [float(Fraction(*x["relative_presentation_time"]))
                             for x in decoded["sampled_frames"]]
        if (len(frames) != FRAME_COUNT or not isinstance(saved_pts, list)
                or len(saved_pts) != FRAME_COUNT or len(decoded_pts) != FRAME_COUNT
                or any(abs(float(a) - b) > 1e-6 for a, b in zip(saved_pts, decoded_pts, strict=True))):
            raise ValueError("saved_claim_sampling_grid_mismatch")
        fps = float(Fraction(*decoded["effective_sample_fps"]))
        gray_frames = make_constant_gray_frames(frames)
        prompt_version, prompt_suffix = resolve_claim_prompt_suffix(
            record, PROMPT_SUFFIX_BY_VERSION)

        stage = "prepare_matched_processor_batches"
        original, original_meta = _prepare_processor_batch(
            processor, process_vision_info, record, frames, fps, prompt_suffix, torch)
        gray, gray_meta = _prepare_processor_batch(
            processor, process_vision_info, record, gray_frames, fps, prompt_suffix, torch)
        if original_meta != gray_meta or set(original) != set(gray):
            raise ValueError("gray_condition_metadata_or_keys_changed")
        for key in original:
            a, b = original[key], gray[key]
            if torch.is_tensor(a) and torch.is_tensor(b):
                if tuple(a.shape) != tuple(b.shape):
                    raise ValueError("gray_condition_tensor_shape_changed")
                if key != "pixel_values_videos" and not torch.equal(a, b):
                    raise ValueError("gray_condition_nonvisual_tensor_changed")
            elif a != b:
                raise ValueError("gray_condition_non_tensor_field_changed")
        original_pixel_hash = _tensor_sha256(original["pixel_values_videos"], torch)
        gray_pixel_hash = _tensor_sha256(gray["pixel_values_videos"], torch)
        if original_pixel_hash == gray_pixel_hash:
            raise ValueError("gray_condition_pixels_not_distinct")
        target_ids = tuple(processor.tokenizer(claim, add_special_tokens=False)["input_ids"])
        if not target_ids or len(target_ids) > 512 or any(type(t) is not int or t < 0 for t in target_ids):
            raise ValueError("saved_claim_retokenization_invalid")

        stage = "load_pinned_scorer"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("model_processor_video_token_mismatch")
        torch.manual_seed(20261021); torch.cuda.manual_seed_all(20261021)
        torch.cuda.reset_peak_memory_stats()
        scores = {}
        for name, batch in (("original_video", original), ("constant_gray_video", gray)):
            attempted += 1
            scores[name] = _score_branch(model=model, batch_cpu=batch,
                target_ids=target_ids, video_token_id=int(processor.video_token_id),
                biased=False, torch=torch)
            completed += 1
        contrast = content_logprob_contrast(
            scores["original_video"]["target_logprob_mean"],
            scores["constant_gray_video"]["target_logprob_mean"])
        source_files = [Path(__file__).resolve(), repo / "scripts" / "run_saved_claim_frame_permutation_attention_smoke.py",
            repo / "scripts" / "qta_multiquery_video_key_intervention.py",
            repo / "scripts" / "claim_mask_video_decoder.py",
            repo / "scripts" / "generate_nextgqa_claim_review_pilot.py"]
        payload = {
            "schema_version": 1, "status": "completed_one_row_exploratory_video_content_sensitivity",
            "started_utc": started_utc,
            "scope": "conditional likelihood sensitivity of a re-tokenized saved claim to original versus constant-gray sampled video",
            "direct_faithfulness": False, "faithfulness_result": False,
            "claim_truth_reviewed": False,
            "human_review": {"review_actually_performed": False,
                "collection_waived_by_user_for_this_exploratory_run": True},
            "rights_status": "unverified; local-only per user authorization",
            "requested_records": 1, "effective_records": 1,
            "requested_forwards": 2, "attempted_forwards": attempted,
            "completed_forwards": completed, "row_token_sha256": row_token,
            "private_input_packet_sha256": packet_hash, "private_input_packet_name": packet_rel.name,
            "packet_record_index_zero_based": record_index, "source_video_sha256": video_hash,
            "saved_claim_text_sha256": _sha256_bytes(claim.encode("utf-8")),
            "retokenized_claim_ids_sha256": _sha256_bytes(json.dumps(target_ids,separators=(",", ":")).encode("ascii")),
            "target_token_count": len(target_ids),
            "target_id_semantics": "deterministic tokenizer encoding of saved parsed sentence; not original emission IDs",
            "prompt_template": f"question + A-E choices + pinned PROMPT_SUFFIX[{prompt_version}]; video-first",
            "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": MODEL_REVISION,
            "precision": "4-bit weights; BF16 compute; eager attention", "seed": 20261021,
            "video_conditions": {"original_processor_pixels_sha256": original_pixel_hash,
                "constant_gray_processor_pixels_sha256": gray_pixel_hash,
                "gray_rgb": [127,127,127], "frame_count": len(frames),
                "sampling_fps": fps, "timestamps_equal": True,
                "metadata_frames_indices": original_meta["frames_indices"],
                "metadata_total_num_frames": float(original_meta["total_num_frames"])},
            "branch_target_scores": scores,
            "primary_descriptive_contrast_original_minus_gray_nats_per_token": contrast,
            "hardware": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "free_vram_before_load_gib": free / 1024**3, "total_vram_gib": total / 1024**3,
                "host_available_before_load_gib": host_free / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
            "source_sha256": {str(path.relative_to(repo)): _sha256_file(path) for path in source_files},
            "elapsed_seconds": time.monotonic() - start,
            "inferential_status": "single-row diagnostic; no population inference",
            "limitations": ["constant-gray video is a severe out-of-distribution perturbation",
                "conditional likelihood sensitivity does not establish claim truth or semantic evidence support",
                "saved claim token IDs were re-tokenized, not original emission IDs",
                "human review was waived as collection but was not performed",
                "rights unverified; all inputs and results remain local"]}
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os,"O_NOFOLLOW",0), 0o600)
        with os.fdopen(fd,"wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({"status":payload["status"], "completed_forwards":completed,
            "contrast_original_minus_gray_nats_per_token":contrast, "output_name":output.name}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema_version":1,"status":"failed_exploratory_video_content_sensitivity",
            "stage":stage,"exception_type":type(exc).__name__,"failure_code":str(exc)[:160],
            "started_utc":started_utc,"requested_forwards":2,"attempted_forwards":attempted,
            "completed_forwards":completed,"direct_faithfulness":False,"human_review_performed":False,
            "rights_status":"unverified; local-only","row_token_sha256":locals().get("row_token"),
            "source_video_sha256":locals().get("video_hash"),"private_input_packet_sha256":locals().get("packet_hash")}
        try:
            raw=(json.dumps(failure,sort_keys=True,separators=(",",":"))+"\n").encode()
            fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
            with os.fdopen(fd,"wb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        except FileExistsError:
            pass
        print(json.dumps({"status":failure["status"],"stage":stage,
            "completed_forwards":completed,"failure_code":failure["failure_code"],"output_name":output.name},sort_keys=True))
        return 2


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--packet",choices=("smoke","remaining"),default="smoke")
    parser.add_argument("--record-index",type=int,default=0)
    parser.add_argument("--user-authorized-local-only",action="store_true",required=True)
    args=parser.parse_args()
    packet=PACKET_REL if args.packet=="smoke" else REMAINING_PACKET_REL
    return run(args.output,packet,args.record_index)


if __name__ == "__main__":
    raise SystemExit(main())
