#!/usr/bin/env python3
"""Check A-E continuation IDs through the pinned local Qwen processor on CPU.

Uses synthetic question/options and two synthetic video frames. No model weights
are loaded and no real dataset, protected review material, or GPU is used.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from qta_processor_choice_suffix import verify_processor_choice_suffix
from run_nextgqa_validation_qta_annotation_proxy import (
    MODEL_REVISION, _choice_token_ids, processor_provenance,
)
from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt


def main() -> int:
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REVISION).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError("pinned_local_processor_missing")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    provenance = processor_provenance(processor, model_path)
    frames = [Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8)),
              Image.fromarray(np.full((448, 448, 3), 127, dtype=np.uint8))]
    item = {"question": "Which object is visible?",
            "choices": ["a cup", "a book", "a ball", "a chair", "a clock"]}
    rendered, messages = build_qa_prompt(processor, item, frames, (1, 1))
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or not isinstance(videos, list) or len(videos) != 1:
        raise ValueError("synthetic_video_utility_contract_invalid")
    video_pixels, metadata = videos[0]

    def process(text: str):
        batch = processor(text=[text], images=images, videos=[video_pixels],
            video_metadata=[metadata], do_sample_frames=False,
            cap_pixels_per_frame=True, padding=True, return_tensors="pt")
        return batch["input_ids"][0].tolist()

    result = verify_processor_choice_suffix(
        tokenizer=processor.tokenizer, rendered_prompt=rendered,
        process_text=process)
    if result["answer_token_ids"] != _choice_token_ids(processor.tokenizer, rendered):
        raise ValueError("kernel_choice_token_binding_mismatch")
    print("status=passed scope=synthetic_multimodal_processor_suffix_only "
          f"labels={result['processed_suffix_matches']} "
          f"snapshot_directory={MODEL_REVISION} "
          f"processor_fingerprint={provenance['processor_fingerprint_sha256']} "
          "weights_loaded=false provenance_authenticated=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
