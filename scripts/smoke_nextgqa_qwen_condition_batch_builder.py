#!/usr/bin/env python3
"""CPU-only real-Qwen-processor smoke using synthetic frames and dummy QA.

Constructs the fixed 3 x 3 condition-input bundle in memory. It does not read
dataset files, load model weights, use CUDA, score logits, or write artifacts.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from nextgqa_condition_batch_builder import build_exact_triplet_batches


MODEL_REV = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MIN_MAX_PIXELS = 256 * 28 * 28


def synthetic_records() -> list[dict]:
    records = []
    y, x = np.mgrid[0:224, 0:224]
    for row in range(3):
        frames = []
        for frame_index in range(32):
            rgb = np.stack((
                (x * 3 + frame_index * 7 + row * 17) % 256,
                (y * 5 + frame_index * 11 + row * 29) % 256,
                (x + y * 2 + frame_index * 13 + row * 43) % 256,
            ), axis=-1).astype(np.uint8)
            frames.append(Image.fromarray(rgb, mode="RGB"))
        records.append({
            "frames": frames,
            "presentation_times": [Fraction(i, 1) for i in range(32)],
            "sampled_indices": list(range(32)),
            "intervals": [[Fraction(5), Fraction(8)]],
            "qa_item": {
                "question": f"Synthetic question {row + 1}?",
                "choices": [f"synthetic choice {i}" for i in range(5)],
                "gold": row,
            },
            "sample_fps": Fraction(1, 1),
        })
    return records


def main() -> None:
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REV)
    if not model_path.is_dir():
        raise SystemExit("pinned_processor_not_cached_downloads_disabled")
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True,
        min_pixels=MIN_MAX_PIXELS, max_pixels=MIN_MAX_PIXELS)
    if torch.cuda.is_initialized():
        raise SystemExit("unexpected_cuda_initialization")
    result = build_exact_triplet_batches(records=synthetic_records(), processor=processor)
    hashes = result["digest_manifest"]["records"]
    report = {
        "status": result["status"],
        "scope": "synthetic_cpu_processor_path_only",
        "model_revision": MODEL_REV,
        "processor_version": __import__("importlib.metadata", fromlist=["version"]).version("transformers"),
        "requested_records": result["requested_records"],
        "requested_batches": result["requested_batches"],
        "effective_batches": len(hashes) * 3,
        "record_batch_manifest_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "frame_pixel_identity_verified": result["frame_pixel_identity_verified"],
        "source_authentication": result["source_authentication"],
        "review_authentication": result["review_authentication"],
        "clock_authentication": result["clock_authentication"],
        "protocol_freeze": result["protocol_freeze"],
        "model_loaded": result["model_loaded"],
        "model_forwards": result["model_forwards"],
        "cuda_initialized": torch.cuda.is_initialized(),
        "execution_authorization": result["execution_authorization"],
    }
    print(json.dumps(report, sort_keys=True))
    if (report["status"] != "condition_batches_constructed_unvalidated"
            or report["effective_batches"] != 9
            or report["model_forwards"] != 0
            or report["cuda_initialized"]):
        raise SystemExit("synthetic_cpu_processor_smoke_failed")


if __name__ == "__main__":
    main()
