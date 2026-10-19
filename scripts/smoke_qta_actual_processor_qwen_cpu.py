#!/usr/bin/env python3
"""CPU-only synthetic smoke for the actual cached Qwen2.5-VL processor.

No checkpoint weights or protected media are loaded. The JSON summary contains
only aggregate shapes and non-authorizing status fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import time

from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from qta_actual_processor_row_candidate import (
    build_actual_processor_row_candidate,
    verify_actual_processor_row_candidate,
)
from qta_prescore_mapping_binding import interval_list_sha256


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    default_snapshot = (
        Path.home() / ".cache" / "huggingface" / "hub"
        / "models--Qwen--Qwen2.5-VL-7B-Instruct" / "snapshots"
        / "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshot", type=Path, default=default_snapshot,
                        help="Local cached Qwen2.5-VL snapshot directory")
    args = parser.parse_args()
    snapshot = args.model_snapshot.resolve(strict=True)
    started = time.monotonic()

    processor = AutoProcessor.from_pretrained(
        str(snapshot), local_files_only=True,
        min_pixels=56 * 56, max_pixels=4 * 56 * 56,
    )
    tokenizer = processor.tokenizer
    tokenizer_sha = digest_file(snapshot / "tokenizer.json")
    processor_files = ("preprocessor_config.json", "tokenizer.json",
                       "tokenizer_config.json", "chat_template.json",
                       "merges.txt", "vocab.json")
    processor_sha = hashlib.sha256(canonical([
        [name, digest_file(snapshot / name)] for name in processor_files
    ])).hexdigest()
    revision = snapshot.name
    video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    temporal_patch_size = int(processor.image_processor.temporal_patch_size)
    spatial_merge_size = int(processor.image_processor.merge_size)

    rows, expected, sampler_rows = [], [], []
    for i in range(3):
        token = f"synthetic-row-{i + 1}"
        source_sha = hashlib.sha256(f"synthetic-video-{i + 1}".encode()).hexdigest()
        question = f"Which color appears in synthetic clip {i + 1}?"
        choices = [f"color {letter}" for letter in "ABCDE"]
        claim = f"Synthetic clip {i + 1} contains changing colors."
        frames = [Image.new("RGB", (56, 56),
                            (j * 7 % 256, i * 31 % 256, (j + i) * 13 % 256))
                  for j in range(32)]
        sampled_pts = [[j, 2] for j in range(32)]
        sampler = {
            "source_sha256": source_sha,
            "decoded_frame_count": 32,
            "sampled_frame_count": 32,
            "sampled_indices": list(range(32)),
            "sampled_pts": sampled_pts,
            "effective_sample_fps": [2, 1],
            "sampled_rgb_sha256": [
                hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames
            ],
        }
        sampler_rows.append(sampler)
        rows.append({
            "review_token": token,
            "source_video_sha256": source_sha,
            "question": question,
            "choices": choices,
            "claim": claim,
            "frames": frames,
            "video_duration_seconds": 16,
            "reviewed_intervals_seconds": [[2, 6]],
            "interval_boundary_guard_seconds": 0,
            "processor_revision": revision,
            "processor_fingerprint_sha256": processor_sha,
            "video_token_id": video_token_id,
            "spatial_merge_size": spatial_merge_size,
            "temporal_patch_size": temporal_patch_size,
        })
        expected.append({
            "review_token": token,
            "source_video_sha256": source_sha,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "options_sha256": hashlib.sha256(canonical(choices)).hexdigest(),
            "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
            "reviewed_intervals_sha256": interval_list_sha256([["2", "6"]]),
            "tokenizer_sha256": tokenizer_sha,
            "processor_revision": revision,
            "processor_fingerprint_sha256": processor_sha,
            "sampler_row_sha256": hashlib.sha256(canonical(sampler)).hexdigest(),
            "video_duration_seconds": "16",
            "interval_boundary_guard_seconds": "0",
        })

    candidate = build_actual_processor_row_candidate(
        rows=rows, expected_rows=expected, sampler_rows=sampler_rows,
        processor=processor, tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha,
        vision_info_fn=process_vision_info, torch_module=torch,
    )
    verify_actual_processor_row_candidate(candidate, expected_rows=expected)
    output = {
        "status": "synthetic_actual_processor_candidate_smoke_passed",
        "processor_revision": revision,
        "rows_requested": 3,
        "rows_built": len(candidate["candidate_rows"]),
        "video_grid_thw": [row["video_grid_thw"] for row in candidate["candidate_rows"]],
        "answer_id_count_per_row": [len(row["answer_token_ids"])
                                    for row in candidate["candidate_rows"]],
        "model_load_permitted": candidate["model_load_permitted"],
        "scoring_authorized": candidate["scoring_authorized"],
        "faithfulness_scored": candidate["faithfulness_scored"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "protected_media_used": False,
        "model_weights_loaded": False,
    }
    print(json.dumps(output, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
