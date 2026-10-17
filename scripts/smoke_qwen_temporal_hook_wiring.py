#!/usr/bin/env python3
"""Synthetic-only Qwen temporal-token hook wiring smoke (GPU required).

This checks one fixed hidden-state hook on deterministic synthetic video input.
It is wiring-only: the chosen token group and gain are not a treatment or an
efficacy candidate, and no faithfulness outcome is computed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from build_nextgqa_diagnostic_target_candidate import map_temporal_video_token_groups


MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
FRAME_COUNT = 32
SYNTHETIC_FPS = 1.0
LAYER_INDEX = 8  # Python zero-based index; selected only as a hook wiring target.
TEMPORAL_GROUP_INDEX = 0  # Fixed synthetic-only position, not data-selected.
NONZERO_STRENGTH = 0.1
MIN_PIXELS = 256 * 28 * 28
MIN_FREE_BEFORE_LOAD = 12 * 1024**3
MIN_FREE_BEFORE_FORWARD = 512 * 1024**2


def temporal_group_hook(positions, strength, *, torch_module=None, touched=None,
                       changed=None):
    """Return a decoder-layer hook that scales exactly the requested sequence rows."""
    positions = tuple(positions)
    if (not positions or any(type(position) is not int or position < 0
                             for position in positions)
            or tuple(sorted(set(positions))) != positions):
        raise ValueError("temporal_group_positions_invalid")
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError("hook_strength_invalid")
    strength = float(strength)
    if not math.isfinite(strength):
        raise ValueError("hook_strength_invalid")
    multiplier = 1.0 + strength

    def hook(_module, _inputs, output):
        is_tensor_output = (torch_module is not None and
                            torch_module.is_tensor(output))
        if not is_tensor_output and (not isinstance(output, tuple) or not output):
            raise ValueError("decoder_layer_output_contract_invalid")
        hidden = output if is_tensor_output else output[0]
        if not hasattr(hidden, "shape") or len(hidden.shape) != 3:
            raise ValueError("decoder_hidden_shape_invalid")
        sequence_length = int(hidden.shape[1])
        if positions[-1] >= sequence_length:
            raise ValueError("temporal_group_out_of_bounds")
        if torch_module is not None and not torch_module.is_tensor(hidden):
            raise ValueError("decoder_hidden_tensor_invalid")
        if touched is not None:
            touched.append(positions)
        if strength == 0.0:
            return output
        modified = hidden.clone()
        modified[:, list(positions), :] *= multiplier
        if changed is not None:
            changed.append(not torch_module.equal(
                hidden[:, list(positions), :], modified[:, list(positions), :]))
        return modified if is_tensor_output else (modified, *output[1:])

    return hook


def _synthetic_frames(np, image_type):
    y, x = np.mgrid[0:224, 0:224]
    frames = []
    for frame_index in range(FRAME_COUNT):
        rgb = np.stack((
            (x * 3 + frame_index * 7 + 17) % 256,
            (y * 5 + frame_index * 11 + 29) % 256,
            (x + y * 2 + frame_index * 13 + 43) % 256,
        ), axis=-1).astype(np.uint8)
        frames.append(image_type.fromarray(rgb, mode="RGB"))
    return frames


def _build_batch(processor, torch, np, image_type):
    from qwen_vl_utils import process_vision_info

    frames = _synthetic_frames(np, image_type)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames, "sample_fps": SYNTHETIC_FPS},
        {"type": "text", "text": (
            "Synthetic wiring smoke. Question: which color channel changes "
            "most between the first and last synthetic frame? Answer briefly."
        )},
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or len(videos) != 1:
        raise RuntimeError("synthetic_video_utility_contract_failed")
    pixels, metadata = videos[0]
    if not isinstance(pixels, torch.Tensor) or int(pixels.shape[0]) != FRAME_COUNT:
        raise RuntimeError("synthetic_processor_frame_count_invalid")
    if metadata.get("frames_indices") != list(range(FRAME_COUNT)):
        raise RuntimeError("synthetic_processor_frame_order_invalid")
    batch = processor(
        text=[prompt], images=images, videos=[pixels], video_metadata=[metadata],
        do_sample_frames=False, cap_pixels_per_frame=True, padding=True,
        return_tensors="pt")
    grid = [int(value) for value in batch["video_grid_thw"][0].tolist()]
    video_processor = processor.video_processor
    temporal_patch_size = int(video_processor.temporal_patch_size)
    merge_size = int(video_processor.merge_size)
    mapping = map_temporal_video_token_groups(
        input_ids=[int(value) for value in batch["input_ids"][0].tolist()],
        video_token_id=int(processor.video_token_id), grid_thw=grid,
        merge_size=merge_size, temporal_patch_size=temporal_patch_size,
        frame_count=FRAME_COUNT,
        sampled_pts=[str(index) for index in range(FRAME_COUNT)],
    )
    if not 0 <= TEMPORAL_GROUP_INDEX < len(mapping):
        raise RuntimeError("synthetic_temporal_group_missing")
    group = mapping[TEMPORAL_GROUP_INDEX]
    positions = tuple(range(
        int(group["first_video_token_position"]),
        int(group["last_video_token_position_exclusive"]),
    ))
    if (len(positions) != int(group["tokens_in_grid"])
            or any(int(batch["input_ids"][0, position]) != int(processor.video_token_id)
                   for position in positions)):
        raise RuntimeError("synthetic_temporal_group_positions_invalid")
    return batch, grid, mapping, positions


def main() -> int:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)

    started = datetime.now(timezone.utc).isoformat()
    started_timer = time.monotonic()
    stage = "initialize"
    model = None
    forwards = 0
    peak_allocated_gib = None
    peak_reserved_gib = None
    minimum_free_before_forward_gib = None
    free_before_load_gib = None
    free_after_gib = None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda_bf16_unavailable")
        device = torch.cuda.current_device()
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise RuntimeError("pinned_checkpoint_not_cached")
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True,
            min_pixels=MIN_PIXELS, max_pixels=MIN_PIXELS)
        stage = "build_synthetic_batch"
        batch_cpu, grid, groups, positions = _build_batch(processor, torch, np, Image)
        if len(groups) != grid[0]:
            raise RuntimeError("temporal_grid_group_count_mismatch")

        free_before, total = torch.cuda.mem_get_info(device)
        free_before_load_gib = free_before / 1024**3
        if free_before < MIN_FREE_BEFORE_LOAD:
            raise RuntimeError("insufficient_free_vram_before_model_load")

        stage = "load_model"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        ).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise RuntimeError("model_processor_video_token_mismatch")
        language_model = model.model.language_model
        if len(language_model.layers) <= LAYER_INDEX:
            raise RuntimeError("hook_layer_index_unavailable")
        batch = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                 for key, value in batch_cpu.items()}
        torch.cuda.reset_peak_memory_stats(device)

        logits = {}
        activations_changed = False
        exact_touches = False
        for name, strength in (("baseline", None), ("zero_strength", 0.0),
                               ("fixed_nonzero_strength", NONZERO_STRENGTH)):
            stage = f"{name}_forward"
            free, _ = torch.cuda.mem_get_info(device)
            free_gib = free / 1024**3
            minimum_free_before_forward_gib = (
                free_gib if minimum_free_before_forward_gib is None
                else min(minimum_free_before_forward_gib, free_gib))
            if free < MIN_FREE_BEFORE_FORWARD:
                raise RuntimeError("insufficient_free_vram_before_forward")
            touched = []
            changed = []
            handle = None
            if strength is not None:
                handle = language_model.layers[LAYER_INDEX].register_forward_hook(
                    temporal_group_hook(positions, strength,
                                        torch_module=torch, touched=touched,
                                        changed=changed))
            try:
                with torch.inference_mode():
                    output = model(**batch, use_cache=False, logits_to_keep=1)
                torch.cuda.synchronize(device)
            finally:
                if handle is not None:
                    handle.remove()
            forwards += 1
            if output.logits.ndim != 3 or not bool(torch.isfinite(output.logits).all()):
                raise RuntimeError("logits_shape_or_finiteness_invalid")
            stage = f"{name}_record_logits"
            logits[name] = output.logits.detach().float().cpu()
            if name == "zero_strength":
                exact_touches = touched == [positions]
            elif name == "fixed_nonzero_strength":
                exact_touches = exact_touches and touched == [positions]
                activations_changed = changed == [True]

        stage = "compare_zero_strength"
        if not torch.equal(logits["baseline"], logits["zero_strength"]):
            raise RuntimeError("zero_strength_hook_not_equivalent")
        if not exact_touches or not activations_changed:
            raise RuntimeError("temporal_group_hook_touch_invalid")
        free_after, _ = torch.cuda.mem_get_info(device)
        free_after_gib = free_after / 1024**3
        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
        report = {
            "status": "passed", "scope": "synthetic_temporal_hook_wiring_only",
            "started_utc": started, "model_revision": MODEL_REVISION,
            "precision": "bf16_4bit_bf16_compute", "gpu": torch.cuda.get_device_name(device),
            "cuda_version": torch.version.cuda, "torch_version": torch.__version__,
            "requested_frames": FRAME_COUNT, "effective_frames": FRAME_COUNT,
            "temporal_grid_groups": len(groups), "selected_group_index": TEMPORAL_GROUP_INDEX,
            "selected_group_token_count": len(positions), "hook_layer_index_zero_based": LAYER_INDEX,
            "forward_count": forwards, "zero_strength_logits_equal_baseline": True,
            "nonzero_hook_touched_exact_group": True,
            "nonzero_hook_activation_changed": True,
            "logits_shape_and_finiteness_passed": True,
            "free_before_load_gib": free_before_load_gib,
            "gpu_total_gib": total / 1024**3,
            "minimum_free_before_forward_gib": minimum_free_before_forward_gib,
            "free_after_gib": free_after_gib,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "elapsed_seconds": time.monotonic() - started_timer,
            "faithfulness_result": False, "treatment_frozen": False,
            "execution_authorization": "not_applicable_synthetic_only",
        }
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        try:
            if torch.cuda.is_available():
                free_after_gib = torch.cuda.mem_get_info()[0] / 1024**3
                if model is not None:
                    device = torch.cuda.current_device()
                    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
                    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
        except Exception:
            pass
        print(json.dumps({
            "status": "incomplete", "scope": "synthetic_temporal_hook_wiring_only",
            "started_utc": started, "model_revision": MODEL_REVISION,
            "forward_count": forwards,
            "failure_code": f"wiring_smoke_failed_{stage}_{type(exc).__name__}",
            "free_before_load_gib": free_before_load_gib,
            "minimum_free_before_forward_gib": minimum_free_before_forward_gib,
            "free_after_gib": free_after_gib,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "elapsed_seconds": time.monotonic() - started_timer,
            "faithfulness_result": False, "execution_authorization": "not_applicable_synthetic_only",
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
