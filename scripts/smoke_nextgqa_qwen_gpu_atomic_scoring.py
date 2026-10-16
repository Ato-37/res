#!/usr/bin/env python3
"""Run the exact nine-forward scorer on synthetic data only (GPU required).

This is a capacity/integration test, not a NExT-GQA result. It uses a locally
cached Qwen2.5-VL checkpoint, deterministic synthetic RGB frames and dummy QA;
it prints only aggregate runtime/resource counters, never item scores.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from nextgqa_atomic_batch_freeze import CONDITIONS, batch_sha256, freeze_all_batches
from nextgqa_claim_mask_scoring import score_preflighted_triplets
from nextgqa_condition_batch_builder import build_exact_triplet_batches


MODEL_REV = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MIN_MAX_PIXELS = 256 * 28 * 28
MIN_FREE_BEFORE_LOAD = 12 * 1024**3
MIN_FREE_BEFORE_FORWARD = 512 * 1024**2


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


def main() -> int:
    started_utc = datetime.now(timezone.utc).isoformat()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA_and_BF16_required")
    device_index = torch.cuda.current_device()

    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REV)
    if not model_path.is_dir():
        raise SystemExit("pinned_checkpoint_not_cached_downloads_disabled")
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True,
        min_pixels=MIN_MAX_PIXELS, max_pixels=MIN_MAX_PIXELS)

    built = build_exact_triplet_batches(records=synthetic_records(), processor=processor)
    if (built["status"] != "condition_batches_constructed_unvalidated"
            or built["requested_records"] != 3 or built["requested_batches"] != 9
            or built["model_loaded"] is not False or built["model_forwards"] != 0):
        raise SystemExit("synthetic_condition_batch_build_failed")
    label_ids = []
    for label in "ABCDE":
        ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise SystemExit("answer_label_token_contract_invalid")
        label_ids.append(int(ids[0]))
    if len(set(label_ids)) != 5:
        raise SystemExit("answer_label_token_ids_not_distinct")

    record_ids = [f"synthetic-{i}" for i in range(1, 4)]
    triplets = []
    hashes = {}
    for record_id, row in zip(record_ids, built["triplets"], strict=True):
        if set(row["batches"]) != set(CONDITIONS):
            raise SystemExit("synthetic_condition_set_invalid")
        triplets.append({"record_id": record_id, "gold_index": row["gold_index"],
                         "batches": row["batches"]})
        hashes[record_id] = {condition: batch_sha256(row["batches"][condition])
                             for condition in CONDITIONS}
    # True is permitted only for this script's fully synthetic fixture; the
    # resulting object still carries execution_authorization=not_granted.
    frozen = freeze_all_batches(
        triplets=triplets, token_ids=label_ids, expected_record_ids=record_ids,
        expected_batch_hashes=hashes, upstream_precheck=lambda: True,
        authenticated_gate_check=lambda: True)
    if (frozen["status"] != "all_nine_batches_frozen"
            or frozen["requested_forwards"] != 9
            or frozen["execution_authorization"] != "not_granted"):
        raise SystemExit("synthetic_nine_batch_freeze_failed")

    # CUDA context initialization may occur during this read-only memory query;
    # all nine CPU batches are already built and frozen, and weights are not yet loaded.
    free_before, total = torch.cuda.mem_get_info(device_index)
    if free_before < MIN_FREE_BEFORE_LOAD:
        raise SystemExit("insufficient_free_vram_before_model_load")
    model = None
    scoring = None
    forward_guard = None
    scoring_started = None
    stats_reset = False
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise RuntimeError("model_processor_video_token_mismatch")
        torch.cuda.reset_peak_memory_stats(device_index)
        stats_reset = True

        class ForwardMemoryGuard:
            """Check free VRAM after batch transfer, directly before model call."""
            def __init__(self):
                self.calls = 0
                self.minimum_free_gib = None

            def __call__(self, **batch):
                free, _total = torch.cuda.mem_get_info(device_index)
                free_gib = free / 1024**3
                self.minimum_free_gib = (free_gib if self.minimum_free_gib is None
                                         else min(self.minimum_free_gib, free_gib))
                if free < MIN_FREE_BEFORE_FORWARD:
                    raise RuntimeError("synthetic_gpu_resource_floor")
                self.calls += 1
                return model(**batch)

        forward_guard = ForwardMemoryGuard()

        scoring_started = time.monotonic()
        scoring = score_preflighted_triplets(
            model=forward_guard, triplets=frozen["triplets"],
            token_ids=frozen["token_ids"], record_ids=frozen["record_ids"],
            expected_batch_hashes=frozen["batch_sha256"], torch_module=torch,
            device="cuda")
        torch.cuda.synchronize()
        elapsed = time.monotonic() - scoring_started
        free_after, _ = torch.cuda.mem_get_info(device_index)
        report = {
            "status": "passed" if scoring["status"] == "completed" else "incomplete",
            "scope": "synthetic_only_nine_forward_capacity_and_scoring_path",
            "started_utc": started_utc,
            "model_revision": MODEL_REV,
            "precision": "bf16_4bit_bf16_compute",
            "gpu": torch.cuda.get_device_name(device_index),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "model_weights_loaded": True,
            "requested_records": scoring["requested_records"],
            "requested_forwards": scoring["requested_forwards"],
            "scorer_attempts_including_guard_rejections": scoring["attempted_forwards"],
            "qwen_model_forward_calls": forward_guard.calls,
            "completed_forwards": scoring["completed_forwards"],
            "completed_triplets": scoring["completed_triplets"],
            "failure_code": scoring["failure_code"],
            "scoring_elapsed_seconds": elapsed,
            "scoring_phase_peak_allocated_gib": torch.cuda.max_memory_allocated(device_index) / 1024**3,
            "scoring_phase_peak_reserved_gib": torch.cuda.max_memory_reserved(device_index) / 1024**3,
            "gpu_free_before_load_gib": free_before / 1024**3,
            "gpu_total_gib": total / 1024**3,
            "minimum_free_before_model_call_gib": forward_guard.minimum_free_gib,
            "gpu_free_at_scoring_exit_gib": free_after / 1024**3,
            "faithfulness_result": False,
            "source_review_clock_authentication": "not_performed",
            "execution_authorization": "not_granted",
        }
        # Never emit item log-probabilities or D3 from the synthetic fixture.
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "passed" else 2
    except Exception as exc:
        # Do not expose local cache paths or model/processor error details.
        try:
            free_after_error = torch.cuda.mem_get_info(device_index)[0] / 1024**3
        except Exception:
            free_after_error = None
        report = {
            "status": "incomplete",
            "scope": "synthetic_only_nine_forward_capacity_and_scoring_path",
            "started_utc": started_utc,
            "model_revision": MODEL_REV,
            "model_weights_loaded": model is not None,
            "requested_records": 3,
            "requested_forwards": 9,
            "scorer_attempts_including_guard_rejections": (
                scoring["attempted_forwards"] if scoring is not None else 0),
            "qwen_model_forward_calls": (forward_guard.calls if forward_guard else 0),
            "completed_forwards": (scoring["completed_forwards"] if scoring is not None else 0),
            "completed_triplets": (scoring["completed_triplets"] if scoring is not None else 0),
            "failure_code": "gpu_smoke_failed_" + type(exc).__name__,
            "scoring_elapsed_seconds": (time.monotonic() - scoring_started
                                         if scoring_started is not None else None),
            "scoring_phase_peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device_index) / 1024**3
                if stats_reset else None),
            "scoring_phase_peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device_index) / 1024**3
                if stats_reset else None),
            "minimum_free_before_model_call_gib": (
                forward_guard.minimum_free_gib if forward_guard else None),
            "gpu_free_at_scoring_exit_gib": free_after_error,
            "faithfulness_result": False,
            "source_review_clock_authentication": "not_performed",
            "execution_authorization": "not_granted",
        }
        print(json.dumps(report, sort_keys=True))
        return 2
    finally:
        scoring = None
        model = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
