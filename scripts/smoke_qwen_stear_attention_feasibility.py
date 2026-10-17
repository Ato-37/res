#!/usr/bin/env python3
"""Synthetic-only feasibility smoke for STEAR-style Qwen attention routing.

This checks whether the pinned Qwen2.5-VL/Transformers stack exposes finite
middle-layer text-query-to-video-token attention and supports a top-10% visual
token selector. It does not implement STEAR, validate its omitted details, read
real data, or estimate faithfulness/efficacy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from smoke_qwen_temporal_hook_wiring import (  # noqa: E402
    MIN_FREE_BEFORE_LOAD,
    MODEL_REVISION,
    _build_batch,
)

LAYER_FIRST_PAPER = 5  # Paper's reported range is inclusive and 1-based.
LAYER_LAST_PAPER = 16
VISUAL_RATIO = 0.10
SMOKE_FRAME_COUNT = 8
MIN_FREE_BEFORE_FORWARD = 768 * 1024**2


def select_top_visual_positions(attentions, visual_positions, *, layer_indices,
                                ratio=VISUAL_RATIO):
    """Rank visual keys by mean head/layer attention from the final prompt row."""
    import math
    from numbers import Real

    if not isinstance(attentions, (tuple, list)) or not attentions:
        raise ValueError("attention_layers_missing")
    if (not isinstance(visual_positions, (tuple, list)) or not visual_positions
            or any(type(pos) is not int or pos < 0 for pos in visual_positions)
            or tuple(sorted(set(visual_positions))) != tuple(visual_positions)):
        raise ValueError("visual_positions_invalid")
    if (not isinstance(layer_indices, (tuple, list)) or not layer_indices
            or any(type(i) is not int or i < 0 or i >= len(attentions)
                   for i in layer_indices)
            or len(set(layer_indices)) != len(layer_indices)):
        raise ValueError("attention_layer_indices_invalid")
    if isinstance(ratio, bool) or not isinstance(ratio, Real):
        raise ValueError("visual_ratio_invalid")
    ratio = float(ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("visual_ratio_invalid")

    first = attentions[layer_indices[0]]
    if first is None or not hasattr(first, "shape") or len(first.shape) != 4:
        raise ValueError("attention_tensor_shape_invalid")
    sequence_length = int(first.shape[-1])
    if (int(first.shape[0]) != 1
            or int(first.shape[-2]) not in (1, sequence_length)
            or visual_positions[-1] >= sequence_length):
        raise ValueError("attention_sequence_shape_invalid")

    aggregate = None
    for layer_index in layer_indices:
        attention = attentions[layer_index]
        if (attention is None or tuple(attention.shape) != tuple(first.shape)
                or not bool(attention.isfinite().all())):
            raise ValueError("attention_tensor_invalid")
        # Batch 1; average heads, and use the last prefill query as the next-token state.
        values = attention[0, :, -1, list(visual_positions)].float().mean(dim=0)
        aggregate = values if aggregate is None else aggregate + values
    scores = aggregate / len(layer_indices)
    count = max(1, math.ceil(len(visual_positions) * ratio))
    ranked_offsets = sorted(
        range(len(visual_positions)),
        key=lambda offset: (-float(scores[offset].item()), visual_positions[offset]),
    )[:count]
    chosen = tuple(sorted(visual_positions[offset] for offset in ranked_offsets))
    if not bool(scores.isfinite().all()):
        raise ValueError("aggregated_attention_nonfinite")
    return chosen, tuple(float(value.item()) for value in scores)


def main(argv=None) -> int:
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/synthetic_qwen_stear_attention_feasibility_8frames_final_20261017.json"),
        help="unique aggregate-only JSON path; existing outputs are never overwritten",
    )
    args = parser.parse_args(argv)
    output_path = args.output
    if output_path.exists():
        raise RuntimeError("output_already_exists")
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    stage = "initialize"
    model = None
    hooks = []
    free_before = free_forward = None
    result = {"status": "failed", "purpose": "synthetic_attention_wiring_only"}
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
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        batch_cpu, grid, groups, _ = _build_batch(
            processor, torch, np, Image, frame_count=SMOKE_FRAME_COUNT)
        free_before, total = torch.cuda.mem_get_info(device)
        if free_before < MIN_FREE_BEFORE_LOAD:
            raise RuntimeError("insufficient_free_vram_before_model_load")

        stage = "load_model"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        ).eval()
        if len(model.model.language_model.layers) < LAYER_LAST_PAPER:
            raise RuntimeError("paper_layer_range_unavailable")
        batch = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                 for key, value in batch_cpu.items()}
        visual_positions = tuple(
            index for index, token in enumerate(batch["input_ids"][0].tolist())
            if int(token) == int(processor.video_token_id))
        if not visual_positions or len(groups) != grid[0]:
            raise RuntimeError("visual_token_mapping_unavailable")

        free_forward, _ = torch.cuda.mem_get_info(device)
        if free_forward < MIN_FREE_BEFORE_FORWARD:
            raise RuntimeError("insufficient_free_vram_before_forward")
        torch.cuda.reset_peak_memory_stats(device)
        stage = "register_attention_capture"
        captured = {}
        for paper_layer in range(LAYER_FIRST_PAPER, LAYER_LAST_PAPER + 1):
            layer_index = paper_layer - 1

            def capture(_module, _inputs, output, *, capture_index=layer_index):
                if (not isinstance(output, tuple) or len(output) < 2
                        or output[1] is None or len(output[1].shape) != 4):
                    raise RuntimeError("self_attention_weights_unavailable")
                weights = output[1]
                captured[capture_index] = (
                    weights[0, :, -1, :].float().mean(dim=0).detach().cpu())

            hooks.append(model.model.language_model.layers[layer_index]
                         .self_attn.register_forward_hook(capture))

        stage = "attention_forward"
        with torch.inference_mode():
            output = model(**batch, use_cache=False, return_dict=True)
        for hook in hooks:
            hook.remove()
        hooks.clear()
        if set(captured) != set(range(LAYER_FIRST_PAPER - 1, LAYER_LAST_PAPER)):
            raise RuntimeError("middle_layer_attention_capture_incomplete")
        # Paper layers 5--16 inclusive -> zero-based indices 4--15.
        layer_indices = tuple(range(LAYER_FIRST_PAPER - 1, LAYER_LAST_PAPER))
        sequence_length = int(batch["input_ids"].shape[-1])
        compact_attentions = []
        for index in range(LAYER_LAST_PAPER):
            row = captured.get(index)
            if row is None:
                compact_attentions.append(None)
            else:
                compact_attentions.append(row.view(1, 1, 1, sequence_length))
        selected, scores = select_top_visual_positions(
            compact_attentions, visual_positions, layer_indices=layer_indices)
        if len(selected) != max(1, int(np.ceil(len(visual_positions) * VISUAL_RATIO))):
            raise RuntimeError("top_ratio_selection_count_mismatch")
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        free_after, _ = torch.cuda.mem_get_info(device)
        result = {
            "status": "passed",
            "purpose": "synthetic_attention_wiring_only",
            "started_at_utc": started_at,
            "source_code_sha256": __import__("hashlib").sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "model_revision": MODEL_REVISION,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "synthetic_frames": SMOKE_FRAME_COUNT,
            "video_grid_thw": grid,
            "temporal_groups": len(groups),
            "visual_token_count": len(visual_positions),
            "paper_layer_range_1_based": [LAYER_FIRST_PAPER, LAYER_LAST_PAPER],
            "attention_layers_observed": len(captured),
            "attention_shape": [1, "num_heads", sequence_length, sequence_length],
            "visual_ratio": VISUAL_RATIO,
            "selected_visual_token_count": len(selected),
            "selected_token_positions": list(selected),
            "selected_scores_finite": all(np.isfinite(scores)),
            "forward_calls": 1,
            "free_vram_before_load_gib": free_before / 1024**3,
            "total_vram_gib": total / 1024**3,
            "free_vram_before_forward_gib": free_forward / 1024**3,
            "free_vram_after_gib": free_after / 1024**3,
            "peak_allocated_gib_after_model_load": peak_allocated,
            "peak_reserved_gib_after_model_load": peak_reserved,
            "elapsed_seconds": time.monotonic() - start,
            "limitations": [
                "synthetic video only; no claim review, answer scoring, treatment, or efficacy",
                "mean attention extraction is a feasibility convention, not a full STEAR reproduction",
                "does not validate the paper's trigger neighborhood or reinjection details",
            ],
        }
    except Exception as exc:
        for hook in hooks:
            hook.remove()
        if torch.cuda.is_available():
            try:
                result["peak_allocated_gib_after_model_load"] = (
                    torch.cuda.max_memory_allocated() / 1024**3)
            except Exception:
                pass
        if free_before is not None:
            result["free_vram_before_load_gib"] = free_before / 1024**3
        if free_forward is not None:
            result["free_vram_before_forward_gib"] = free_forward / 1024**3
        result.update({"failure_type": type(exc).__name__, "stage": stage,
                       "elapsed_seconds": time.monotonic() - start})
        known_code = {"insufficient_free_vram_before_model_load": "gpu_memory_gate_blocked",
                      "insufficient_free_vram_before_forward": "gpu_memory_gate_blocked"}.get(str(exc))
        message = str(exc)
        safe_message = bool(re.fullmatch(r"[a-z0-9_]+", message)) and message.startswith((
            "attention_", "visual_", "top_ratio_", "middle_layer_", "paper_",
            "cuda_", "pinned_", "synthetic_", "insufficient_"))
        result["failure_code"] = (
            known_code or (message if safe_message else
                           f"smoke_failed_{stage}_{type(exc).__name__}"))
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
