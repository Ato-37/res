"""Build one complete unvalidated 3 x 3 Qwen input bundle in memory.

The caller supplies decoded sampled RGB frames, exact full-video PTS times,
sample indices, intervals, QA items, and nominal sample FPS. This module only
constructs conditions and processor outputs, then hashes them. It does not read
source/review data, authenticate inputs, load weights, score, write files, or
grant execution authorization.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from nextgqa_claim_mask_conditions import CONDITIONS, FRAME_COUNT, build_claim_mask_conditions
from nextgqa_batch_digest_freeze import build_batch_digest_manifest
from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video


_RECORD_KEYS = {"frames", "presentation_times", "sampled_indices", "intervals",
                "qa_item", "sample_fps"}
_QA_KEYS = {"question", "choices", "gold"}
_REQUIRED_BATCH_KEYS = {"input_ids", "attention_mask", "pixel_values_videos",
                        "video_grid_thw", "second_per_grid_ts"}
_VARYING_BATCH_KEY = "pixel_values_videos"


def _validate_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, (list, tuple)) or len(records) != 3:
        raise ValueError("requested_record_count_mismatch")
    prepared = []
    # Build every condition assignment before invoking the processor. A bad
    # third row therefore cannot leave a returned/processor-partial cohort.
    for row_index, record in enumerate(records, 1):
        if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
            raise ValueError(f"record_{row_index}_schema_invalid")
        qa_item = record["qa_item"]
        if (not isinstance(qa_item, Mapping) or set(qa_item) != _QA_KEYS
                or not isinstance(qa_item.get("question"), str)
                or not qa_item["question"].strip()
                or not isinstance(qa_item.get("choices"), (list, tuple))
                or len(qa_item["choices"]) != 5
                or any(not isinstance(choice, str) or not choice.strip()
                       for choice in qa_item["choices"])
                or type(qa_item.get("gold")) is not int
                or not 0 <= qa_item["gold"] < 5):
            raise ValueError(f"record_{row_index}_qa_item_invalid")
        sample_fps = record["sample_fps"]
        if not isinstance(sample_fps, Fraction) or sample_fps <= 0:
            raise ValueError(f"record_{row_index}_sample_fps_invalid")
        condition_payload = build_claim_mask_conditions(
            frames=record["frames"], sampled_indices=record["sampled_indices"],
            presentation_times=record["presentation_times"], intervals=record["intervals"],
            include_metadata=True)
        if (not isinstance(condition_payload, Mapping)
                or set(condition_payload) != {"conditions", "metadata"}
                or set(condition_payload["conditions"]) != set(CONDITIONS)
                or condition_payload["metadata"].get("status") != "assignment_constructed_unvalidated"):
            raise ValueError(f"record_{row_index}_condition_assignment_invalid")
        prepared.append({"qa_item": dict(qa_item), "sample_fps": sample_fps,
                         "conditions": condition_payload["conditions"],
                         "assignment_metadata": condition_payload["metadata"]})
    return prepared


def _nonmask_descriptors(manifest_row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    result = {}
    for condition in CONDITIONS:
        batch_descriptor = manifest_row["conditions"][condition]
        if (not isinstance(batch_descriptor, Mapping)
                or batch_descriptor.get("type") != "mapping"
                or not isinstance(batch_descriptor.get("items"), Mapping)):
            raise ValueError(f"record_{row_index}_{condition}_batch_digest_invalid")
        items = dict(batch_descriptor["items"])
        if _VARYING_BATCH_KEY not in items:
            raise ValueError(f"record_{row_index}_{condition}_pixel_tensor_missing")
        items.pop(_VARYING_BATCH_KEY)
        result[condition] = items
    return result


def _expected_qwen_video_pixels(frames, output_shape, *, torch_module, numpy_module):
    """Reconstruct Qwen's exact resize path from the caller's ordered RGB frames."""
    try:
        from qwen_vl_utils import vision_process
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize
    except ImportError:
        raise ValueError("qwen_video_pixel_audit_dependencies_unavailable") from None
    # qwen_vl_utils.fetch_video() calls fetch_image() on each supplied frame,
    # then stacks in order and applies this same batched resize. `sample_fps`
    # metadata is ignored by fetch_image and only used for timestamps later.
    image_factor = 14 * vision_process.SPATIAL_MERGE_SIZE
    processed = [vision_process.fetch_image({"image": frame}, image_factor)
                 for frame in frames]
    raw = torch_module.stack([
        torch_module.from_numpy(numpy_module.array(frame).transpose(2, 0, 1))
        for frame in processed
    ])
    if len(frames) % vision_process.FRAME_FACTOR:
        raise ValueError("qwen_video_frame_factor_mismatch")
    return resize(raw, list(output_shape), interpolation=InterpolationMode.BICUBIC,
                  antialias=True).float()


def build_exact_triplet_batches(*, records: Any, processor: Any,
                                vision_info_fn=None) -> dict[str, Any]:
    """Construct all 9 processor batches and a digest manifest, without scoring.

    `vision_info_fn` must follow `qwen_vl_utils.process_vision_info` and return
    `(images, videos)` when called with `return_video_metadata=True`. The
    existing `process_qa_video` call is used unchanged, so the pinned
    Qwen2.5-VL processor's configured `return_metadata` default remains in
    force; this wrapper does not silently override processor kwargs.
    """
    if processor is None:
        raise ValueError("processor_invalid")
    prepared = _validate_records(records)
    if vision_info_fn is None:
        from qwen_vl_utils import process_vision_info
        vision_info_fn = process_vision_info
    if not callable(vision_info_fn):
        raise ValueError("vision_info_function_invalid")

    triplets = []
    condition_metadata = []
    for row_index, row in enumerate(prepared, 1):
        batches = {}
        rendered_texts = []
        for condition in CONDITIONS:
            sample_fps = row["sample_fps"]
            fps_pair = (sample_fps.numerator, sample_fps.denominator)
            text, messages = build_qa_prompt(
                processor, row["qa_item"], row["conditions"][condition], fps_pair)
            if not isinstance(text, str) or not text:
                raise ValueError(f"record_{row_index}_{condition}_rendered_prompt_invalid")
            rendered_texts.append(text)
            try:
                vision_result = vision_info_fn(messages, return_video_metadata=True)
            except Exception:
                raise ValueError(f"record_{row_index}_{condition}_vision_preprocess_failed") from None
            if (not isinstance(vision_result, (list, tuple)) or len(vision_result) != 2):
                raise ValueError(f"record_{row_index}_{condition}_vision_utility_contract_invalid")
            images, videos = vision_result
            if images is not None or not isinstance(videos, list) or len(videos) != 1:
                raise ValueError(f"record_{row_index}_{condition}_video_utility_contract_invalid")
            video_payload = videos[0]
            if not isinstance(video_payload, (list, tuple)) or len(video_payload) != 2:
                raise ValueError(f"record_{row_index}_{condition}_video_metadata_pair_invalid")
            pixels, video_metadata = video_payload
            try:
                import torch
                import numpy as np
                expected_fps = float(sample_fps)
                if (not isinstance(pixels, torch.Tensor) or pixels.ndim < 1
                        or pixels.shape[0] != FRAME_COUNT or not isinstance(video_metadata, Mapping)
                        or video_metadata.get("frames_indices") != list(range(FRAME_COUNT))
                        or isinstance(video_metadata.get("fps"), bool)
                        or not isinstance(video_metadata.get("fps"), (int, float))
                        or not math.isfinite(float(video_metadata["fps"]))
                        or abs(float(video_metadata["fps"]) - expected_fps) > 1e-9):
                    raise ValueError(f"record_{row_index}_{condition}_video_geometry_invalid")
                expected_pixels = _expected_qwen_video_pixels(
                    row["conditions"][condition], pixels.shape[-2:],
                    torch_module=torch, numpy_module=np)
                if expected_pixels.shape != pixels.shape or not torch.equal(expected_pixels, pixels):
                    raise ValueError(f"record_{row_index}_{condition}_frame_pixel_identity_mismatch")
            except ImportError:
                raise ValueError("torch_unavailable") from None
            try:
                batch = process_qa_video(processor, text, images, pixels, video_metadata)
            except Exception:
                raise ValueError(f"record_{row_index}_{condition}_processor_failed") from None
            if (not isinstance(batch, Mapping) or not _REQUIRED_BATCH_KEYS.issubset(batch)
                    or _VARYING_BATCH_KEY not in batch):
                raise ValueError(f"record_{row_index}_{condition}_processor_output_invalid")
            batches[condition] = batch

        if len(set(rendered_texts)) != 1:
            raise ValueError(f"record_{row_index}_condition_prompt_mismatch")
        triplets.append({"gold_index": row["qa_item"]["gold"], "batches": batches})
        condition_metadata.append({
            "record_position": row_index,
            "rendered_prompt_sha256": hashlib.sha256(rendered_texts[0].encode("utf-8")).hexdigest(),
            "assignment": row["assignment_metadata"],
        })

    manifest = build_batch_digest_manifest(triplets)
    # Only the image tensor may vary between conditions. This compares exact
    # typed tensor/scalar descriptors, including tokens and video metadata.
    for row_index, manifest_row in enumerate(manifest["records"], 1):
        nonmask = _nonmask_descriptors(manifest_row, row_index)
        if len({json.dumps(value, sort_keys=True, separators=(",", ":"))
                for value in nonmask.values()}) != 1:
            raise ValueError(f"record_{row_index}_nonmask_processor_input_changed")

    return {
        "status": "condition_batches_constructed_unvalidated",
        "scope": "local_in_memory_exact_3x3_batch_construction_only",
        "requested_records": 3,
        "requested_batches": 9,
        "triplets": triplets,
        "condition_metadata": condition_metadata,
        "digest_manifest": manifest,
        "frame_content_check": "reconstructed_qwen_fetch_image_and_video_resize_exact_tensor_match",
        "frame_pixel_identity_verified": True,
        "source_authentication": "not_performed",
        "review_authentication": "not_performed",
        "clock_authentication": "not_performed",
        "protocol_freeze": "not_performed",
        "model_loaded": False,
        "model_forwards": 0,
        "execution_authorization": "not_granted",
    }
