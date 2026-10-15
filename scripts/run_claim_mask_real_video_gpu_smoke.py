"""Run one label-free real-video Qwen forward through the PyAV PTS adapter.

This is a local exploratory execution check, not a faithfulness or QA test.
It uses one fixed NOAA source video, four endpoint-inclusive frames, and one
Qwen2.5-VL-7B forward. Raw media and model outputs are never written to results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels


EXPECTED_MEDIA_SHA256 = "13ab1d0e12d9c6b9e776890d123f363c12c572d52d1f58464b57a2e98667f126"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
NUM_FRAMES = 4
UNIFORM_TOLERANCE = Fraction(1, 1000)
PROMPT = "Describe the visible scene briefly. Use only visual information."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
                        help="new privacy-safe summary path inside results/")
    parser.add_argument("--processor-only", action="store_true",
                        help="run real-video CPU decode+Qwen processor preflight without CUDA/model")
    args = parser.parse_args()
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
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if args.processor_only:
        free_bytes, total_bytes = None, None
    else:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            parser.error("CUDA and BF16 required; CPU fallback disabled")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < 12 * 1024**3:
            parser.error("At least 12 GiB free VRAM required")

    seed = 20261015
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started_utc = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()

    # The CPU smoke and metadata gates run before loading the 7B model.
    frames, decode_meta = decode_sampled_video(
        video, num_frames=NUM_FRAMES, uniform_tolerance_seconds=UNIFORM_TOLERANCE)
    if decode_meta["source_sha256"] != EXPECTED_MEDIA_SHA256:
        raise RuntimeError("PyAV decoder source hash differs from the pinned input")
    verify_sampled_frame_pixels(frames, decode_meta)
    sample_fps = Fraction(*decode_meta["effective_sample_fps"])

    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True, min_pixels=256 * 28 * 28,
        max_pixels=256 * 28 * 28)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames, "sample_fps": float(sample_fps)},
        {"type": "text", "text": PROMPT},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or len(videos) != 1:
        raise RuntimeError("Qwen utility did not return exactly one video")
    video_tensor, video_metadata = videos[0]
    import numpy as np
    if not isinstance(video_tensor, torch.Tensor) or video_tensor.shape[0] != len(frames):
        raise RuntimeError("Qwen utility did not preserve the decoded frame count")
    if video_metadata.get("frames_indices") != list(range(NUM_FRAMES)):
        raise RuntimeError("Qwen metadata does not preserve decoded frame order")
    if float(video_metadata.get("fps", 0)) != float(sample_fps):
        raise RuntimeError("Qwen metadata FPS differs from the PTS-derived rate")
    if int(video_metadata.get("total_num_frames", 0)) != len(frames):
        raise RuntimeError("Qwen metadata frame count differs from decoded frames")
    input_signatures = torch.stack([
        torch.from_numpy(np.asarray(frame).copy()).float().mean(dim=(0, 1))
        for frame in frames
    ])
    output_signatures = video_tensor.float().mean(dim=(-1, -2))
    if output_signatures.shape != input_signatures.shape or not torch.allclose(
            output_signatures, input_signatures, rtol=0, atol=2.0):
        raise RuntimeError("Qwen utility changed or reordered frame color signatures")
    verify_sampled_frame_pixels(frames, decode_meta)
    batch = processor(
        text=[prompt], images=images, videos=[video_tensor],
        video_metadata=[video_metadata], do_sample_frames=False,
        cap_pixels_per_frame=True, padding=True, return_tensors="pt").to(
            "cpu" if args.processor_only else "cuda")

    grid = batch["video_grid_thw"][0].tolist()
    patch = int(processor.video_processor.temporal_patch_size)
    merge = int(processor.video_processor.merge_size)
    video_token_id = int(processor.video_token_id)
    video_token_positions = (batch["input_ids"][0] == video_token_id).nonzero(
        as_tuple=True)[0]
    expected_video_tokens = int(batch["video_grid_thw"][0].prod()) // (merge ** 2)
    seconds_per_grid = float(batch["second_per_grid_ts"][0])
    if len(video_token_positions) != expected_video_tokens:
        raise RuntimeError("Qwen video-token/grid count mismatch")
    if grid[0] != NUM_FRAMES // patch:
        raise RuntimeError("Qwen temporal grid differs from the expected frame/patch count")
    if abs(seconds_per_grid - patch / float(sample_fps)) > 1e-5:
        raise RuntimeError("Qwen temporal-grid/FPS metadata mismatch")

    if args.processor_only:
        payload = {
            "schema_version": 1,
            "purpose": "real_video_pyav_to_qwen_processor_cpu_preflight",
            "status": "passed",
            "source_sha256": {
                "runner": sha256(Path(__file__)),
                "decoder": sha256(repo / "scripts/claim_mask_video_decoder.py"),
            },
            "media_sha256": EXPECTED_MEDIA_SHA256,
            "requested_videos": 1,
            "effective_videos": 1,
            "requested_frames": NUM_FRAMES,
            "effective_frames": len(frames),
            "decode_frame_count": decode_meta["decoded_frame_count"],
            "sample_fps": [sample_fps.numerator, sample_fps.denominator],
            "grid_thw": grid,
            "seconds_per_grid": seconds_per_grid,
            "video_tokens": len(video_token_positions),
            "annotation_clock_calibrated": False,
            "annotation_or_qa_used": False,
            "faithfulness_effect_estimate": None,
            "privacy": "No raw media, QA/answer text, generated text, absolute local path, or personal identifier.",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"Passed real-video CPU PyAV→Qwen processor preflight: 1/1 video, "
              f"{len(frames)}/4 frames, grid {grid}. Summary: {args.output}")
        return

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if int(model.config.video_token_id) != video_token_id:
        raise RuntimeError("processor and checkpoint video token IDs differ")

    torch.cuda.reset_peak_memory_stats()
    forward_started = time.monotonic()
    with torch.inference_mode():
        result = model(**batch, output_attentions=True, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize()
    finite_logits = bool(torch.isfinite(result.logits).all())
    finite_attentions = all(bool(torch.isfinite(value).all()) for value in result.attentions)
    if not finite_logits or not finite_attentions:
        raise RuntimeError("Qwen produced nonfinite logits or attention")

    payload = {
        "schema_version": 1,
        "purpose": "real_video_pyav_to_qwen_gpu_path_smoke",
        "status": "passed",
        "started_utc": started_utc,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_sha256": {
            "runner": sha256(Path(__file__)),
            "decoder": sha256(repo / "scripts/claim_mask_video_decoder.py"),
        },
        "media_sha256": EXPECTED_MEDIA_SHA256,
        "media_description": "fixed NOAA EX2503 dive03 public-domain source; 300-second clip",
        "credit": "NOAA Ocean Exploration",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": MODEL_REVISION,
        "seed": seed,
        "quantization": "4bit",
        "compute_dtype": "bfloat16",
        "attention_implementation": "eager",
        "requested_videos": 1,
        "effective_videos": 1,
        "requested_frames": NUM_FRAMES,
        "effective_frames": len(frames),
        "decode_frame_count": decode_meta["decoded_frame_count"],
        "decoded_pts_sequence_sha256": decode_meta["decoded_pts_sequence_sha256"],
        "pass1_decoded_rgb_sequence_sha256": decode_meta["pass1_decoded_rgb_sequence_sha256"],
        "sampled_frames": decode_meta["sampled_frames"],
        "uniform_tolerance_seconds": [UNIFORM_TOLERANCE.numerator,
                                       UNIFORM_TOLERANCE.denominator],
        "sample_fps": [sample_fps.numerator, sample_fps.denominator],
        "annotation_clock_calibrated": False,
        "annotation_or_qa_used": False,
        "grid_thw": grid,
        "seconds_per_grid": seconds_per_grid,
        "video_tokens": len(video_token_positions),
        "input_tokens": int(batch["input_ids"].shape[1]),
        "attention_layers": len(result.attentions),
        "finite_logits": finite_logits,
        "finite_attentions": finite_attentions,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "forward_seconds": time.monotonic() - forward_started,
        "elapsed_seconds": time.monotonic() - timer,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_free_before_gib": free_bytes / 1024**3,
        "gpu_total_gib": total_bytes / 1024**3,
        "cuda_version": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "bitsandbytes", "qwen-vl-utils"]},
        "faithfulness_effect_estimate": None,
        "limitations": [
            "One real video, four sparse frames, one forward, no treatment/control or answer labels.",
            "No claim-level effect, confidence interval, significance test, or faithfulness conclusion.",
            "PTS is relative to the first decoded frame; annotation clock is uncalibrated.",
            "The sampled uniform-grid tolerance is frozen for this one-video feasibility check only.",
        ],
        "privacy": "No raw media, QA/answer text, generated text, absolute local path, or personal identifier.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Passed real-video PyAV→Qwen GPU smoke: 1/1 video, {len(frames)}/4 frames, "
          f"{len(result.attentions)} attention layers, {payload['peak_allocated_gib']:.3f} GiB peak; "
          f"{payload['elapsed_seconds']:.1f}s elapsed. Summary: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        parser_args = sys.argv[1:]
        if "--output" in parser_args:
            index = parser_args.index("--output")
            if index + 1 < len(parser_args):
                repo = Path(__file__).resolve().parents[1]
                output = (repo / parser_args[index + 1]).resolve()
                if output.is_relative_to(repo / "results") and not output.exists():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("x", encoding="utf-8") as stream:
                        json.dump({
                            "schema_version": 1, "purpose": "real_video_pyav_to_qwen_gpu_path_smoke",
                            "status": "failed", "requested_videos": 1,
                            "effective_videos": 0, "failures": 1,
                            "failure_type": type(error).__name__,
                            "created_utc": datetime.now(timezone.utc).isoformat(),
                        }, stream, indent=2, sort_keys=True)
                        stream.write("\n")
        print(f"Real-video GPU smoke failed: {type(error).__name__}; no faithfulness result.",
              file=sys.stderr)
        raise
