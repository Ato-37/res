"""Run one Qwen attention forward on an in-memory generated synthetic MP4.

Execution/VRAM wiring smoke only. It uses no project media, labels, QA, or
grounding score, and writes an aggregate-only result beneath ``results/``.
The pinned checkpoint must already be cached; network downloads are disabled.

Usage::

    .venv/bin/python scripts/smoke_qwen_attention.py --output results/UNIQUE.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import cv2
import numpy as np


REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
VIDEO_FRAMES = 24
VIDEO_FPS = 8.0
MAX_PIXELS = 256 * 28 * 28
SEED = 20261015


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_synthetic_video(path: Path) -> str:
    """Write deterministic color frames and return a hash of the file bytes."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (224, 224)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic MP4")
    try:
        for index in range(VIDEO_FRAMES):
            color = ((index * 13) % 256, (index * 29) % 256, (index * 47) % 256)
            writer.write(np.full((224, 224, 3), color, dtype=np.uint8))
    finally:
        writer.release()
    return sha256(path.read_bytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output = (repo / args.output).resolve()
    results = (repo / "results").resolve()
    if not output.is_relative_to(results) or output == results or output.exists():
        parser.error("Use a new output under results/; overwrite refused")
    model_path = (
        Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"
        / REVISION
    )
    if not model_path.is_dir():
        parser.error("Pinned local checkpoint missing; downloads are disabled")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from grounding_check import extract_frames

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA and BF16 required; CPU fallback disabled")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < 12 * 1024**3:
        parser.error("At least 12 GiB free VRAM required")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with tempfile.TemporaryDirectory(prefix="qwen_attention_synthetic_") as temp:
        root = Path(temp)
        video = root / "synthetic.mp4"
        media_hash = make_synthetic_video(video)
        frame_paths, frame_dir, timestamps, sampled_fps = extract_frames(
            str(video), 4, 224
        )
        try:
            started = datetime.now(timezone.utc).isoformat()
            timer = time.monotonic()
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                attn_implementation="eager", device_map="cuda",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                ),
            ).eval()
            processor = AutoProcessor.from_pretrained(
                str(model_path), local_files_only=True,
                min_pixels=256 * 28 * 28, max_pixels=MAX_PIXELS,
            )
            messages = [{"role": "user", "content": [
                {"type": "video", "video": frame_paths, "sample_fps": sampled_fps},
                {"type": "text", "text": "Describe the synthetic color changes."},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images, videos = process_vision_info(messages, return_video_metadata=True)
            if images is not None or len(videos) != 1:
                raise RuntimeError("Expected exactly one synthetic video tensor")
            video_tensor, metadata = videos[0]
            inputs = processor(
                text=[text], images=images, videos=[video_tensor],
                video_metadata=[metadata], do_sample_frames=False,
                cap_pixels_per_frame=True, padding=True, return_tensors="pt",
            ).to("cuda")
            grid = [int(value) for value in inputs["video_grid_thw"][0].tolist()]
            video_token_id = int(model.config.video_token_id)
            video_tokens = int((inputs["input_ids"][0] == video_token_id).sum().item())
            expected_tokens = int(inputs["video_grid_thw"][0].prod().item()) // int(
                processor.video_processor.merge_size
            ) ** 2
            if len(timestamps) != 4 or video_tokens != expected_tokens:
                raise RuntimeError("Synthetic frame/token geometry mismatch")

            torch.cuda.reset_peak_memory_stats(0)
            forward_started = time.monotonic()
            with torch.inference_mode():
                output_state = model(
                    **inputs, output_attentions=True, use_cache=False, logits_to_keep=1
                )
            torch.cuda.synchronize(0)
            elapsed_forward = time.monotonic() - forward_started
            finite_logits = bool(torch.isfinite(output_state.logits).all())
            finite_attentions = all(
                bool(torch.isfinite(layer).all()) for layer in output_state.attentions
            )
            if not finite_logits or not finite_attentions:
                raise RuntimeError("Non-finite synthetic forward output")
            result = {
                "schema_version": 1,
                "purpose": "synthetic_video_attention_wiring_smoke",
                "status": "complete",
                "started_utc": started,
                "runner_sha256": sha256(Path(__file__).read_bytes()),
                "synthetic_video_sha256": media_hash,
                "model_revision": REVISION,
                "model_config_sha256": sha256((model_path / "config.json").read_bytes()),
                "seed": SEED,
                "requested_frames": 4,
                "effective_frames": len(timestamps),
                "video_grid_thw": grid,
                "video_token_count": video_tokens,
                "input_token_count": int(inputs["input_ids"].shape[1]),
                "attention_layer_count": len(output_state.attentions),
                "finite_logits": finite_logits,
                "finite_attentions": finite_attentions,
                "forward_seconds": elapsed_forward,
                "elapsed_seconds": time.monotonic() - timer,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_free_before_bytes": int(free_bytes),
                "gpu_total_bytes": int(total_bytes),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "versions": {name: importlib.metadata.version(name) for name in (
                    "torch", "transformers", "bitsandbytes", "qwen-vl-utils",
                    "opencv-python-headless",
                )},
                "limitations": [
                    "Synthetic media only; no project dataset or annotations.",
                    "Checks one attention forward and resource wiring only.",
                    "No QA, grounding, faithfulness score, treatment/control, or effect estimate.",
                ],
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
            os.chmod(output, 0o600)
            print(
                f"Synthetic attention smoke passed: frames=4/4, layers="
                f"{len(output_state.attentions)}, forward={elapsed_forward:.2f}s, "
                f"peak={result['peak_allocated_bytes'] / 1024**3:.2f} GiB, output={args.output}"
            )
            return 0
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
