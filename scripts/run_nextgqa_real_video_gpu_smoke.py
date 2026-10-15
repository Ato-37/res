#!/usr/bin/env python3
"""Run a privacy-preserving Qwen GPU path smoke on one local NExT-GQA video.

This checks real benchmark-media decoding, PTS-derived frame sampling, Qwen
video preprocessing, and one CUDA model forward. It does not use QA text,
score an answer, evaluate a grounding interval, or estimate faithfulness.
The output is an aggregate local diagnostic and contains no sample identity.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import glob
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
from fractions import Fraction

from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels

FRAMES = 4
SEED = 20261015
PROMPT = "Describe the visible scene briefly. Use only visual information."


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def video_key(value: str) -> str:
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    if parts:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return "/".join(parts[-2:])


def choose_videos(repo: Path, dataset: Path) -> list[Path]:
    """List valid, local test videos absent from prior manifests."""
    video_root = Path.home() / "nextgqa_sample" / "NExTVideo"
    grouped = {}
    for path in video_root.rglob("*.mp4"):
        grouped.setdefault(video_key(path.relative_to(video_root).as_posix()), []).append(path)
    by_key = {key: paths[0] for key, paths in grouped.items() if len(paths) == 1}
    excluded = set()
    for manifest in glob.glob(str(repo / "data" / "nextgqa_samples_*.json")):
        try:
            records = json.loads(Path(manifest).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(records, list):
            excluded.update(video_key(r["video"]) for r in records
                            if isinstance(r, dict) and isinstance(r.get("video"), str))
    video_map = json.loads((dataset / "map_vid_vidorID.json").read_text(encoding="utf-8"))
    gsub = json.loads((dataset / "gsub_test.json").read_text(encoding="utf-8"))
    csv_path = dataset / "test.csv"
    candidates = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            video_id, qid = str(row["video_id"]), str(row["qid"])
            info = gsub.get(video_id)
            raw = (info.get("location", {}).get(qid)
                   if isinstance(info, dict) and isinstance(info.get("location"), dict)
                   else None)
            if video_id not in video_map or not isinstance(info, dict) or not isinstance(raw, list) or not raw:
                continue
            try:
                duration = float(info["duration"])
                spans = [(float(s[0]), float(s[1])) for s in raw]
            except (KeyError, TypeError, ValueError, IndexError, OverflowError):
                continue
            if not all(0 <= a <= b <= duration for a, b in spans):
                continue
            key = video_key(video_map[video_id])
            if key not in by_key or key in excluded:
                continue
            candidates.append(by_key[key])
    candidates = sorted(set(candidates), key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())
    if not candidates:
        raise RuntimeError("no_valid_nonoverlapping_local_test_video")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new local JSON summary; overwrite refused")
    parser.add_argument("--frames", type=int, choices=(4, 16, 32, 64), default=FRAMES)
    parser.add_argument("--uniform-tolerance-ms", type=int, choices=(1, 50), default=1)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("output exists; choose a unique path")
    repo = Path(__file__).resolve().parents[1]
    dataset = repo / "NExT-GQA" / "datasets" / "nextgqa"
    candidates = choose_videos(repo, dataset)
    frame_count = args.frames
    uniform_tolerance = Fraction(args.uniform_tolerance_ms, 1000)
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    if not model_path.is_dir():
        parser.error("pinned local Qwen checkpoint missing; downloads disabled")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import numpy as np
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA and BF16 required; CPU fallback disabled")
    cuda_device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
    if free_bytes < 12 * 1024**3:
        parser.error("at least 12 GiB free VRAM required before model loading")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    video = None
    frames = meta = None
    timing_rejections = 0
    attempted = 0
    # Search only for a technical pipeline-smoke candidate; no model output is
    # observed during selection. Both frame count and timing tolerance are logged.
    for candidate in candidates[:25]:
        attempted += 1
        try:
            frames, meta = decode_sampled_video(
                candidate, num_frames=frame_count,
                uniform_tolerance_seconds=uniform_tolerance)
            video = candidate
            break
        except ValueError:
            timing_rejections += 1
    if video is None:
        raise RuntimeError("no_uniform_pts_video_in_first_25_candidates")
    verify_sampled_frame_pixels(frames, meta)
    sample_fps = meta["effective_sample_fps"]
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames,
         "sample_fps": sample_fps[0] / sample_fps[1]},
        {"type": "text", "text": PROMPT},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or len(videos) != 1:
        raise RuntimeError("qwen_video_utility_contract_failed")
    pixels, metadata = videos[0]
    if not isinstance(pixels, torch.Tensor) or pixels.shape[0] != frame_count:
        raise RuntimeError("processor_frame_count_mismatch")
    if metadata.get("frames_indices") != list(range(frame_count)):
        raise RuntimeError("processor_frame_order_mismatch")
    if abs(float(metadata.get("fps", 0)) - sample_fps[0] / sample_fps[1]) > 1e-7:
        raise RuntimeError("processor_fps_mismatch")
    signatures = torch.stack([
        torch.from_numpy(np.asarray(frame).copy()).float().mean(dim=(0, 1)) for frame in frames
    ])
    tensor_signatures = pixels.float().mean(dim=(-1, -2))
    if tensor_signatures.shape != signatures.shape or not torch.allclose(
            tensor_signatures, signatures, rtol=0, atol=2.0):
        raise RuntimeError("processor_pixel_order_or_identity_mismatch")
    verify_sampled_frame_pixels(frames, meta)
    batch = processor(text=[prompt], images=images, videos=[pixels],
                      video_metadata=[metadata], do_sample_frames=False,
                      cap_pixels_per_frame=True, padding=True,
                      return_tensors="pt").to("cuda")
    grid = batch["video_grid_thw"][0].tolist()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if int(model.config.video_token_id) != int(processor.video_token_id):
        raise RuntimeError("model_processor_video_token_mismatch")
    torch.cuda.reset_peak_memory_stats(cuda_device)
    forward_start = time.monotonic()
    with torch.inference_mode():
        out = model(**batch, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(out.logits).all())
    if not finite:
        raise RuntimeError("nonfinite_model_logits")
    post_free_bytes, post_total_bytes = torch.cuda.mem_get_info(cuda_device)
    memory_stats = torch.cuda.memory_stats(cuda_device)
    memory_segments = [segment for segment in torch.cuda.memory_snapshot(include_traces=False)
                       if segment.get("device") == cuda_device]
    snapshot_totals = {
        key: sum(int(segment.get(key, 0)) for segment in memory_segments)
        for key in ("total_size", "allocated_size", "active_size")
    }
    payload = {
        "schema_version": 1,
        "purpose": "local_nextgqa_real_video_qwen_gpu_path_smoke_not_faithfulness",
        "status": "passed",
        "started_utc": started,
        "runner_sha256": sha256(Path(__file__)),
        "dataset_split": "test",
        "candidate_selection": "valid annotated local video, absent from existing sample manifests; deterministic hash order",
        "candidate_videos_inspected_for_timing": attempted,
        "candidate_timing_rejections": timing_rejections,
        "uniform_grid_tolerance_seconds": [uniform_tolerance.numerator,
                                            uniform_tolerance.denominator],
        "requested_videos": 1,
        "effective_videos": 1,
        "requested_frames": frame_count,
        "effective_frames": len(frames),
        "decoded_frame_count": meta["decoded_frame_count"],
        "sampled_pts_sha256": meta["decoded_pts_sequence_sha256"],
        "sampled_pixels_sha256": meta["pass1_decoded_rgb_sequence_sha256"],
        "qwen_grid_thw": grid,
        "input_tokens": int(batch["input_ids"].shape[1]),
        "finite_logits": finite,
        "forward_seconds": time.monotonic() - forward_start,
        "elapsed_seconds": time.monotonic() - timer,
        # Keep allocator accounting distinct from physical device occupancy.
        # This WSL run produced tensor-allocation peaks above reported VRAM.
        "peak_allocated_gib": torch.cuda.max_memory_allocated(cuda_device) / 1024**3,
        "pytorch_peak_reserved_gib": torch.cuda.max_memory_reserved(cuda_device) / 1024**3,
        "pytorch_current_allocated_gib": memory_stats["allocated_bytes.all.current"] / 1024**3,
        "pytorch_current_reserved_gib": memory_stats["reserved_bytes.all.current"] / 1024**3,
        "pytorch_peak_allocated_stat_gib": memory_stats["allocated_bytes.all.peak"] / 1024**3,
        "pytorch_peak_reserved_stat_gib": memory_stats["reserved_bytes.all.peak"] / 1024**3,
        "allocator_snapshot_segment_count": len(memory_segments),
        "allocator_snapshot_total_size_gib": snapshot_totals["total_size"] / 1024**3,
        "allocator_snapshot_allocated_size_gib": snapshot_totals["allocated_size"] / 1024**3,
        "allocator_snapshot_active_size_gib": snapshot_totals["active_size"] / 1024**3,
        "cuda_driver_free_after_gib": post_free_bytes / 1024**3,
        "cuda_driver_total_after_gib": post_total_bytes / 1024**3,
        "cuda_current_device_index": cuda_device,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_allocator_backend": torch.cuda.get_allocator_backend(),
        "cuda_allocator_config": {key: os.environ.get(key) for key in
                                  ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")},
        "gpu": torch.cuda.get_device_name(cuda_device),
        "cuda_device_total_property_gib": (
            torch.cuda.get_device_properties(cuda_device).total_memory / 1024**3),
        "process_id": os.getpid(),
        "memory_sampled_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_free_before_gib": free_bytes / 1024**3,
        "gpu_total_gib": total_bytes / 1024**3,
        "cuda_version": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "bitsandbytes", "qwen-vl-utils"]},
        "qa_text_used": False,
        "interval_used_for_frame_selection": False,
        "faithfulness_effect_estimate": None,
        "annotation_clock_calibrated": False,
        "privacy": "No sample/video identity, local path, QA text, generated text, or raw media in output.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"NExT-GQA real-video Qwen CUDA smoke passed: {frame_count}/{frame_count} frames, "
          f"grid={grid}, finite_logits={finite}, elapsed={payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
