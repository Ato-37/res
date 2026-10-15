"""Measure real-video Qwen input/resource scaling at fixed spatial settings.

This feasibility sweep tests 4/8/16 endpoint-inclusive samples from the same
NOAA clip. It uses the PyAV PTS adapter, identical 224-equivalent processor
settings, one BF16/4-bit model, and one forward per condition. It uses no QA,
annotation, generated text, masking treatment, or faithfulness score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time

from claim_mask_video_decoder import decode_sampled_video_sets, verify_sampled_frame_pixels
from run_claim_mask_real_video_gpu_smoke import (
    EXPECTED_MEDIA_SHA256, MODEL_REVISION, PROMPT, sha256,
)
from run_qwen_video_cache_gpu_smoke import GpuTelemetrySampler


RUN_STATE = {"stage": "not_started", "requested_frame_counts": [], "conditions": []}
SCHEMA_VERSION = 2
ALLOWED_FRAME_COUNTS = (4, 8, 16)
# Half a source-frame period at the fixed 30000/1001 fps CFR source. This is a
# capacity-diagnostic tolerance for index quantization, not a claim-mask rule.
UNIFORM_TOLERANCE = Fraction(1001, 60000)
MIN_INITIAL_FREE_VRAM_BYTES = 12 * 1024**3
MIN_PER_FORWARD_FREE_VRAM_BYTES = 5 * 1024**3
MIN_HOST_AVAILABLE_BYTES = 8 * 1024**3


def parse_frame_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(part) for part in value.split(","))
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("frame counts must be comma-separated integers") from exc
    if (not counts or any(count not in ALLOWED_FRAME_COUNTS for count in counts)
            or tuple(sorted(set(counts))) != counts or counts[0] != 4
            or (16 in counts and 8 not in counts)):
        raise argparse.ArgumentTypeError("use 4, then optionally 8 and 16 in order")
    return counts


def expected_video_tokens(grid: list[int], merge_size: int) -> int:
    if (not isinstance(grid, list) or len(grid) != 3
            or any(type(value) is not int or value <= 0 for value in grid)
            or type(merge_size) is not int or merge_size <= 0
            or grid[1] % merge_size or grid[2] % merge_size):
        raise ValueError("invalid_video_grid")
    return grid[0] * (grid[1] // merge_size) * (grid[2] // merge_size)


def validate_fixed_spatial_grid(grids: list[list[int]]) -> list[int]:
    if not grids or any(not isinstance(grid, list) or len(grid) != 3
                        or any(type(value) is not int or value <= 0 for value in grid)
                        for grid in grids):
        raise ValueError("invalid_video_grid_sequence")
    spatial = grids[0][1:]
    if any(grid[1:] != spatial for grid in grids[1:]):
        raise ValueError("spatial video grid changed across temporal conditions")
    return spatial


def host_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    raise RuntimeError("MemAvailable counter missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-counts", type=parse_frame_counts, default=(4, 8, 16))
    parser.add_argument("--output", required=True,
                        help="new privacy-safe summary path inside results/")
    parser.add_argument("--processor-only", action="store_true",
                        help="decode and processor-check actual video on CPU; skip CUDA/model")
    args = parser.parse_args()
    RUN_STATE.update(stage="preflight", requested_frame_counts=list(args.frame_counts), conditions=[])
    repo = Path(__file__).resolve().parents[1]
    output = (repo / args.output).resolve()
    if not output.is_relative_to(repo / "results") or output.exists():
        parser.error("Use a new output inside results/; overwrite refused")
    video = repo / "datasets/noaa_qa_pilot_20261011/media/source.mp4"
    if not video.is_file() or sha256(video) != EXPECTED_MEDIA_SHA256:
        parser.error("Fixed real video missing or source hash mismatch")
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        parser.error("Pinned local model checkpoint missing; downloads disabled")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if args.processor_only:
        free_bytes = total_bytes = None
    else:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            parser.error("CUDA and BF16 required; CPU fallback disabled")
        if host_available_bytes() < MIN_HOST_AVAILABLE_BYTES:
            parser.error("At least 8 GiB host MemAvailable required")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < MIN_INITIAL_FREE_VRAM_BYTES:
            parser.error("At least 12 GiB free VRAM required before model loading")

    seed = 20261015
    random.seed(seed)
    torch.manual_seed(seed)
    if not args.processor_only:
        torch.cuda.manual_seed_all(seed)
    start_utc = datetime.now(timezone.utc).isoformat()
    total_timer = time.monotonic()

    # Decode and processor-check every condition before model loading or GPU use.
    RUN_STATE["stage"] = "cpu_decode_and_processor_preflight"
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True, min_pixels=256 * 28 * 28,
        max_pixels=256 * 28 * 28)
    shared_decode_start = time.monotonic()
    decoded_sets, decode_metadata = decode_sampled_video_sets(
        video, frame_counts=args.frame_counts,
        uniform_tolerance_seconds=UNIFORM_TOLERANCE)
    shared_video_decode_seconds = time.monotonic() - shared_decode_start
    prepared = []
    for frame_count in args.frame_counts:
        prepare_start = time.monotonic()
        frames = decoded_sets[frame_count]
        decode_meta = decode_metadata[frame_count]
        if decode_meta["source_sha256"] != EXPECTED_MEDIA_SHA256:
            raise RuntimeError("PyAV source hash differs from pinned media")
        verify_sampled_frame_pixels(frames, decode_meta)
        sample_fps = Fraction(*decode_meta["effective_sample_fps"])
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames, "sample_fps": float(sample_fps)},
            {"type": "text", "text": PROMPT},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        images, videos = process_vision_info(messages, return_video_metadata=True)
        if images is not None or len(videos) != 1:
            raise RuntimeError("Qwen utility did not return exactly one video")
        tensor, metadata = videos[0]
        if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != frame_count:
            raise RuntimeError("Qwen utility changed video frame count")
        if metadata.get("frames_indices") != list(range(frame_count)):
            raise RuntimeError("Qwen utility changed video frame order")
        if float(metadata.get("fps", 0)) != float(sample_fps):
            raise RuntimeError("Qwen utility FPS differs from PTS-derived average")
        if int(metadata.get("total_num_frames", 0)) != frame_count:
            raise RuntimeError("Qwen utility total frame count differs")
        source_signatures = torch.stack([
            torch.from_numpy(np.asarray(frame).copy()).float().mean(dim=(0, 1))
            for frame in frames
        ])
        tensor_signatures = tensor.float().mean(dim=(-1, -2))
        if tensor_signatures.shape != source_signatures.shape or not torch.allclose(
                tensor_signatures, source_signatures, rtol=0, atol=2.0):
            raise RuntimeError("Qwen utility changed frame signatures/order")
        verify_sampled_frame_pixels(frames, decode_meta)
        batch = processor(
            text=[prompt], images=images, videos=[tensor], video_metadata=[metadata],
            do_sample_frames=False, cap_pixels_per_frame=True, padding=True,
            return_tensors="pt")
        grid = batch["video_grid_thw"][0].tolist()
        temporal_patch = int(processor.video_processor.temporal_patch_size)
        merge_size = int(processor.video_processor.merge_size)
        tokens = expected_video_tokens(grid, merge_size)
        token_positions = (batch["input_ids"][0] == int(processor.video_token_id)).nonzero(
            as_tuple=True)[0]
        seconds_per_grid = float(batch["second_per_grid_ts"][0])
        expected_temporal_grid = frame_count // temporal_patch
        if grid[0] != expected_temporal_grid or len(token_positions) != tokens:
            raise RuntimeError("Qwen token/temporal grid mismatch")
        if abs(seconds_per_grid - temporal_patch / float(sample_fps)) > 1e-5:
            raise RuntimeError("Qwen seconds-per-grid differs from supplied FPS")
        prepared.append({
            "frame_count": frame_count,
            "frames": frames,
            "decode_meta": decode_meta,
            "sample_fps": sample_fps,
            "batch": batch,
            "grid": grid,
            "video_tokens": tokens,
            "seconds_per_grid": seconds_per_grid,
            "processor_prepare_seconds": time.monotonic() - prepare_start,
        })

    validate_fixed_spatial_grid([row["grid"] for row in prepared])

    if args.processor_only:
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "purpose": "real_video_pyav_qwen_temporal_input_capacity_cpu_preflight",
            "status": "passed",
            "source_sha256": {
                "runner": sha256(Path(__file__)),
                "decoder": sha256(repo / "scripts/claim_mask_video_decoder.py"),
            },
            "media_sha256": EXPECTED_MEDIA_SHA256,
            "requested_conditions": len(prepared),
            "effective_conditions": len(prepared),
            "conditions": [{
                "requested_frames": row["frame_count"],
                "effective_frames": len(row["frames"]),
                "decoded_frame_count": row["decode_meta"]["decoded_frame_count"],
                "sample_fps": [row["sample_fps"].numerator, row["sample_fps"].denominator],
                "max_sampled_grid_deviation_seconds": row["decode_meta"][
                    "max_sampled_grid_deviation_seconds"],
                "grid_thw": row["grid"],
                "video_tokens": row["video_tokens"],
                "seconds_per_grid": row["seconds_per_grid"],
            "processor_prepare_seconds": row["processor_prepare_seconds"],
            } for row in prepared],
        "shared_video_decode_seconds": shared_video_decode_seconds,
            "faithfulness_effect_estimate": None,
            "privacy": "No raw media, QA/answer text, generated text, or absolute local path.",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        RUN_STATE["stage"] = "complete"
        print("Real-video CPU capacity preflight passed: " + "; ".join(
            f"{r['effective_frames']}f {r['video_tokens']} video tokens"
            for r in artifact["conditions"]))
        print(f"Summary: {args.output}")
        return

    RUN_STATE["stage"] = "model_load"
    model_load_start = time.monotonic()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    model_load_seconds = time.monotonic() - model_load_start
    if int(model.config.video_token_id) != int(processor.video_token_id):
        raise RuntimeError("processor/model video token IDs differ")

    condition_rows = RUN_STATE["conditions"]
    RUN_STATE["stage"] = "gpu_forwards"
    for condition in prepared:
        free_before, total_current = torch.cuda.mem_get_info()
        host_before = host_available_bytes()
        if host_before < MIN_HOST_AVAILABLE_BYTES:
            raise RuntimeError(f"less than 8 GiB host MemAvailable before {condition['frame_count']}-frame forward")
        if free_before < MIN_PER_FORWARD_FREE_VRAM_BYTES:
            raise RuntimeError(f"insufficient free VRAM before {condition['frame_count']}-frame forward")
        if condition["frame_count"] == 16 and free_before < 6 * 1024**3:
            raise RuntimeError("less than 6 GiB headroom before 16-frame forward")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device_batch = condition["batch"].to("cuda")
        telemetry = GpuTelemetrySampler(interval_seconds=0.5)
        telemetry.start()
        forward_start = time.monotonic()
        try:
            with torch.inference_mode():
                result = model(**device_batch, output_attentions=False,
                               use_cache=False, logits_to_keep=1)
            torch.cuda.synchronize()
        finally:
            telemetry_summary = telemetry.stop()
        forward_seconds = time.monotonic() - forward_start
        finite_logits = bool(torch.isfinite(result.logits).all())
        if not finite_logits:
            raise RuntimeError(f"nonfinite logits at {condition['frame_count']} frames")
        condition_rows.append({
            "requested_frames": condition["frame_count"],
            "effective_frames": len(condition["frames"]),
            "decoded_frame_count": condition["decode_meta"]["decoded_frame_count"],
            "sampled_frame_indices": [row["decoded_frame_index"]
                                      for row in condition["decode_meta"]["sampled_frames"]],
            "sampled_presentation_times_seconds": [
                [row["relative_presentation_time"][0], row["relative_presentation_time"][1]]
                for row in condition["decode_meta"]["sampled_frames"]],
            "sample_fps": [condition["sample_fps"].numerator,
                           condition["sample_fps"].denominator],
            "max_sampled_grid_deviation_seconds": condition["decode_meta"][
                "max_sampled_grid_deviation_seconds"],
            "grid_thw": condition["grid"],
            "video_tokens": condition["video_tokens"],
            "input_tokens": int(device_batch["input_ids"].shape[1]),
            "seconds_per_grid": condition["seconds_per_grid"],
            "processor_prepare_seconds": condition["processor_prepare_seconds"],
            "forward_seconds": forward_seconds,
            "finite_logits": finite_logits,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "free_vram_before_forward_gib": free_before / 1024**3,
            "host_available_before_forward_gib": host_before / 1024**3,
            "telemetry": telemetry_summary,
        })
        del result, device_batch
        torch.cuda.empty_cache()
        gc.collect()

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "real_video_pyav_qwen_temporal_input_capacity_sweep",
        "status": "passed",
        "started_utc": start_utc,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_sha256": {
            "runner": sha256(Path(__file__)),
            "decoder": sha256(repo / "scripts/claim_mask_video_decoder.py"),
            "four_frame_runner": sha256(repo / "scripts/run_claim_mask_real_video_gpu_smoke.py"),
            "telemetry_helper": sha256(repo / "scripts/run_qwen_video_cache_gpu_smoke.py"),
        },
        "media_sha256": EXPECTED_MEDIA_SHA256,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": MODEL_REVISION,
        "model_config_sha256": sha256(model_path / "config.json"),
        "seed": seed,
        "quantization": "4bit",
        "compute_dtype": "bfloat16",
        "attention_implementation": "eager",
        "output_attentions": False,
        "use_cache": False,
        "logits_to_keep": 1,
        "frame_max_dimension": 224,
        "max_pixels_per_frame": 256 * 28 * 28,
        "uniform_grid_tolerance_seconds": [UNIFORM_TOLERANCE.numerator,
                                           UNIFORM_TOLERANCE.denominator],
        "tolerance_basis": "half a 1001/30000 second source frame period; capacity sweep only",
        "requested_conditions": len(args.frame_counts),
        "effective_conditions": len(condition_rows),
        "failures": 0,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "gpu_free_before_load_gib": free_bytes / 1024**3,
        "gpu_total_gib": total_bytes / 1024**3,
        "host_available_before_run_gib": host_available_bytes() / 1024**3,
        "model_load_seconds": model_load_seconds,
        "shared_video_decode_seconds": shared_video_decode_seconds,
        "conditions": condition_rows,
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av"]},
        "pyav_version": __import__("av").__version__,
        "ffmpeg_libraries": {
            name: list(version) for name, version in sorted(__import__("av").library_versions.items())
        },
        "elapsed_seconds": time.monotonic() - total_timer,
        "faithfulness_effect_estimate": None,
        "limitations": [
            "Resource/processor feasibility only; no QA, annotations, treatment/control, or answer scoring.",
            "Frame-count differences do not establish evidence coverage, quality, or optimal sampling.",
            "One video, one forward per condition; utilization telemetry is sparse and descriptive.",
            "The 16.683 ms tolerance addresses source-frame index quantization only and is not an inferential protocol threshold.",
        ],
        "privacy": "No raw media, pixels, QA/answer text, generated text, logits, or absolute local path.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    RUN_STATE["stage"] = "complete"
    print("Real-video capacity sweep passed: " + "; ".join(
        f"{r['effective_frames']}f {r['video_tokens']} video tokens "
        f"{r['peak_allocated_gib']:.2f} GiB {r['forward_seconds']:.2f}s "
        f"GPU-util-mean={r['telemetry']['gpu_util_percent']['mean']}%"
        for r in condition_rows))
    print(f"Summary: {args.output}; elapsed {artifact['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        arguments = __import__("sys").argv[1:]
        if "--output" in arguments:
            index = arguments.index("--output")
            if index + 1 < len(arguments):
                repo = Path(__file__).resolve().parents[1]
                output = (repo / arguments[index + 1]).resolve()
                if output.is_relative_to(repo / "results") and not output.exists():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("x", encoding="utf-8") as stream:
                        json.dump({
                            "schema_version": SCHEMA_VERSION,
                            "purpose": "real_video_pyav_qwen_temporal_input_capacity_sweep",
                            "status": "failed",
                            "stage": RUN_STATE["stage"],
                            "requested_frame_counts": RUN_STATE["requested_frame_counts"],
                            "completed_conditions": RUN_STATE["conditions"],
                            "effective_conditions": len(RUN_STATE["conditions"]),
                            "failures": 1,
                            "failure_type": type(error).__name__,
                            "created_utc": datetime.now(timezone.utc).isoformat(),
                        }, stream, indent=2, sort_keys=True)
                        stream.write("\n")
        print(f"Real-video capacity sweep failed at {RUN_STATE['stage']}: "
              f"{type(error).__name__}; no faithfulness result.", file=__import__("sys").stderr)
        raise
