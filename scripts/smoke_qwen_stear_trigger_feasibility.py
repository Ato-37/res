#!/usr/bin/env python3
"""Synthetic-only Qwen feasibility smoke for STEAR trigger/KES wiring.

At the final prefill query, this captures decoder layers 5--16, applies the
model's shared final RMSNorm and LM head to each hidden row, evaluates
normalized entropy, and conditionally selects top-attention video-token
positions using a fixed radius-one fixture. It does not reproduce STEAR's
complete KES, reinjection, counterfactual branch, or efficacy evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from smoke_qwen_temporal_hook_wiring import (  # noqa: E402
    MIN_FREE_BEFORE_LOAD,
    MODEL_REVISION,
    _build_batch,
)
from smoke_qwen_stear_attention_feasibility import (  # noqa: E402
    select_top_visual_positions,
)
from stear_uncertainty_trigger import select_uncertain_layer  # noqa: E402

LAYER_FIRST_PAPER = 5
LAYER_LAST_PAPER = 16
TRIGGER_THRESHOLD = 0.85
VISUAL_RATIO = 0.10
SMOKE_NEIGHBORHOOD_RADIUS = 1
SMOKE_FRAME_COUNT = 8
MIN_FREE_BEFORE_FORWARD = 768 * 1024**2


def _capture_projected_logits(*, output, layer_index, expected_indices,
                              final_layer_index, language_model, model,
                              logits_by_layer, final_projection_check,
                              torch_module):
    """Capture one batch's final-position logits using the shared model head."""
    if (not torch_module.is_tensor(output) or output.ndim != 3
            or int(output.shape[0]) != 1):
        raise RuntimeError("decoder_hidden_output_contract_invalid")
    last_hidden = output[:, -1, :]
    normalized = language_model.norm(last_hidden)
    layer_logits = model.lm_head(normalized)[0]
    if layer_logits.ndim != 1:
        raise RuntimeError("intermediate_vocab_logits_shape_invalid")
    frozen_logits = layer_logits.detach().float().cpu()
    if layer_index in expected_indices:
        logits_by_layer[layer_index] = frozen_logits
    if layer_index == final_layer_index:
        final_projection_check["logits"] = frozen_logits


def _capture_attention_row(*, output, layer_index, attention_rows, torch_module):
    """Capture the final-query attention row under the fixed batch-one contract."""
    if (not isinstance(output, tuple) or len(output) < 2
            or not torch_module.is_tensor(output[1]) or output[1].ndim != 4
            or int(output[1].shape[0]) != 1):
        raise RuntimeError("self_attention_weights_unavailable")
    weights = output[1]
    attention_rows[layer_index] = (
        weights[0, :, -1, :].float().mean(dim=0).detach().cpu())


def _select_engineering_kes(*, attention_rows, visual_positions, selected_layer,
                            layer_range, radius, sequence_length, ratio=VISUAL_RATIO):
    """Apply a radius-limited mean-attention top-token engineering selector."""
    if (not isinstance(layer_range, tuple) or len(layer_range) != 2
            or any(type(index) is not int or index < 0 for index in layer_range)
            or layer_range[0] > layer_range[1]
            or type(selected_layer) is not int
            or not layer_range[0] <= selected_layer <= layer_range[1]
            or type(radius) is not int or radius < 0
            or type(sequence_length) is not int or sequence_length < 1):
        raise ValueError("engineering_kes_contract_invalid")
    neighborhood = tuple(range(max(layer_range[0], selected_layer - radius),
                               min(layer_range[1], selected_layer + radius) + 1))
    compact_rows = [None] * (layer_range[1] + 1)
    for index, row in attention_rows.items():
        if (type(index) is not int
                or not layer_range[0] <= index <= layer_range[1]):
            raise ValueError("engineering_attention_layer_invalid")
        compact_rows[index] = row.view(1, 1, 1, sequence_length)
    positions, _ = select_top_visual_positions(
        compact_rows, visual_positions, layer_indices=neighborhood, ratio=ratio)
    return neighborhood, positions


def main(argv=None) -> int:
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="unique aggregate-only JSON path; existing outputs are never overwritten",
    )
    parser.add_argument(
        "--engineering-trigger-threshold", type=float, default=TRIGGER_THRESHOLD,
        help="synthetic engineering override; the study default remains the paper's 0.85",
    )
    args = parser.parse_args(argv)
    if (not math.isfinite(args.engineering_trigger_threshold)
            or not 0.0 <= args.engineering_trigger_threshold <= 1.0):
        parser.error("--engineering-trigger-threshold must be finite and in [0, 1]")
    output_path = args.output
    if output_path.exists():
        raise RuntimeError("output_already_exists")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    stage = "initialize"
    model = None
    hooks = []
    free_before = free_forward = None
    result = {"status": "failed", "purpose": "synthetic_entropy_trigger_wiring_only"}
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
        language_model = model.model.language_model
        if len(language_model.layers) < LAYER_LAST_PAPER:
            raise RuntimeError("paper_layer_range_unavailable")
        batch = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                 for key, value in batch_cpu.items()}

        free_forward, _ = torch.cuda.mem_get_info(device)
        if free_forward < MIN_FREE_BEFORE_FORWARD:
            raise RuntimeError("insufficient_free_vram_before_forward")
        torch.cuda.reset_peak_memory_stats(device)
        logits_by_layer = {}
        final_projection_check = {}
        attention_rows = {}
        stage = "register_layer_logits_capture"
        expected_indices = tuple(range(LAYER_FIRST_PAPER - 1, LAYER_LAST_PAPER))
        final_layer_index = len(language_model.layers) - 1
        capture_indices = tuple(sorted(set(expected_indices) | {final_layer_index}))
        for layer_index in capture_indices:

            def capture(_module, _inputs, output, *, capture_index=layer_index):
                _capture_projected_logits(
                    output=output, layer_index=capture_index,
                    expected_indices=expected_indices,
                    final_layer_index=final_layer_index,
                    language_model=language_model, model=model,
                    logits_by_layer=logits_by_layer,
                    final_projection_check=final_projection_check,
                    torch_module=torch)

            hooks.append(language_model.layers[layer_index].register_forward_hook(capture))

        stage = "register_attention_capture"
        for layer_index in expected_indices:
            def capture_attention(_module, _inputs, output, *, capture_index=layer_index):
                _capture_attention_row(
                    output=output, layer_index=capture_index,
                    attention_rows=attention_rows, torch_module=torch)

            hooks.append(language_model.layers[layer_index].self_attn
                         .register_forward_hook(capture_attention))

        stage = "synthetic_entropy_forward"
        with torch.inference_mode():
            output = model(**batch, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)
        for hook in hooks:
            hook.remove()
        hooks.clear()
        if tuple(sorted(logits_by_layer)) != expected_indices:
            raise RuntimeError("intermediate_layer_capture_incomplete")
        if tuple(sorted(attention_rows)) != expected_indices:
            raise RuntimeError("middle_layer_attention_capture_incomplete")
        if "logits" not in final_projection_check:
            raise RuntimeError("final_layer_projection_capture_missing")
        projection_max_abs_error = float((
            final_projection_check["logits"] - output.logits[0, -1].float().cpu()
        ).abs().max().item())
        if not torch.allclose(final_projection_check["logits"],
                              output.logits[0, -1].float().cpu(),
                              rtol=1e-5, atol=1e-5):
            raise RuntimeError("shared_norm_lm_head_projection_mismatch")
        triggered, selected_layer, maximum_entropy = select_uncertain_layer(
            logits_by_layer, layer_indices=expected_indices,
            threshold=args.engineering_trigger_threshold)
        if (output.logits.ndim != 3 or not bool(torch.isfinite(output.logits).all())
                or not np.isfinite(maximum_entropy)):
            raise RuntimeError("final_logits_or_entropy_invalid")

        visual_positions = tuple(
            index for index, token in enumerate(batch["input_ids"][0].tolist())
            if int(token) == int(processor.video_token_id))
        if not visual_positions:
            raise RuntimeError("synthetic_visual_positions_missing")
        selected_visual_positions = ()
        neighborhood_indices = ()
        if triggered:
            sequence_length = int(batch["input_ids"].shape[-1])
            neighborhood_indices, selected_visual_positions = _select_engineering_kes(
                attention_rows=attention_rows, visual_positions=visual_positions,
                selected_layer=selected_layer,
                layer_range=(LAYER_FIRST_PAPER - 1, LAYER_LAST_PAPER - 1),
                radius=SMOKE_NEIGHBORHOOD_RADIUS,
                sequence_length=sequence_length, ratio=VISUAL_RATIO)

        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        free_after, _ = torch.cuda.mem_get_info(device)
        result = {
            "status": "passed",
            "purpose": "synthetic_entropy_trigger_wiring_only",
            "started_at_utc": started_at,
            "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "trigger_helper_sha256": hashlib.sha256(
                Path(__file__).with_name("stear_uncertainty_trigger.py").read_bytes()
            ).hexdigest(),
            "attention_selector_sha256": hashlib.sha256(
                Path(__file__).with_name("smoke_qwen_stear_attention_feasibility.py")
                .read_bytes()
            ).hexdigest(),
            "model_revision": MODEL_REVISION,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "synthetic_frames": SMOKE_FRAME_COUNT,
            "video_grid_thw": grid,
            "temporal_groups": len(groups),
            "paper_layer_range_1_based": [LAYER_FIRST_PAPER, LAYER_LAST_PAPER],
            "captured_layer_count": len(logits_by_layer),
            "final_layer_projection_check_1_based": final_layer_index + 1,
            "final_projection_max_abs_error": projection_max_abs_error,
            "vocabulary_size": int(next(iter(logits_by_layer.values())).numel()),
            "paper_reported_trigger_threshold": TRIGGER_THRESHOLD,
            "trigger_threshold": args.engineering_trigger_threshold,
            "trigger_threshold_mode": (
                "paper_reported_default" if args.engineering_trigger_threshold == TRIGGER_THRESHOLD
                else "synthetic_engineering_override"),
            "triggered": triggered,
            "selected_layer_1_based": selected_layer + 1,
            "maximum_normalized_entropy": maximum_entropy,
            "per_layer_normalized_entropy": {
                str(index + 1): score for index, score in _layer_entropies(
                    logits_by_layer, expected_indices).items()
            },
            "attention_layers_observed": len(attention_rows),
            "visual_token_count": len(visual_positions),
            "kes_layer_neighborhood_1_based": [index + 1 for index in neighborhood_indices],
            "kes_visual_ratio": VISUAL_RATIO,
            "selected_visual_token_count": len(selected_visual_positions),
            "forward_calls": 1,
            "free_vram_before_load_gib": free_before / 1024**3,
            "total_vram_gib": total / 1024**3,
            "free_vram_before_forward_gib": free_forward / 1024**3,
            "free_vram_after_gib": free_after / 1024**3,
            "peak_allocated_gib_after_model_load": peak_allocated,
            "peak_reserved_gib_after_model_load": peak_reserved,
            "elapsed_seconds": time.monotonic() - started,
            "limitations": [
                "synthetic video only; first next-token position only",
                "uses shared final norm and LM head for intermediate layer rows",
                "when triggered, uses simple mean-attention top-token selection on synthetic input",
                "radius-one neighborhood and token-level selection are engineering fixtures, not full STEAR KES",
                "does not implement evidence reinjection or counterfactual decoding",
                "not a STEAR reproduction and no faithfulness outcome was measured",
            ],
        }
    except Exception as exc:
        for hook in hooks:
            hook.remove()
        if free_before is not None:
            result["free_vram_before_load_gib"] = free_before / 1024**3
        if free_forward is not None:
            result["free_vram_before_forward_gib"] = free_forward / 1024**3
        message = str(exc)
        safe_code = message if message.isascii() and message.replace("_", "").isalnum() else ""
        result.update({
            "failure_type": type(exc).__name__,
            "failure_code": safe_code or f"smoke_failed_{stage}_{type(exc).__name__}",
            "stage": stage,
            "elapsed_seconds": time.monotonic() - started,
        })
    finally:
        for hook in hooks:
            hook.remove()
        hooks.clear()
        # Remove every strong reference to model/device tensors before cache
        # cleanup; deleting only `model` leaves the language_model alias alive.
        try:
            del capture
        except UnboundLocalError:
            pass
        try:
            del output
        except UnboundLocalError:
            pass
        try:
            del batch
        except UnboundLocalError:
            pass
        try:
            del batch_cpu
        except UnboundLocalError:
            pass
        try:
            del language_model
        except UnboundLocalError:
            pass
        try:
            del processor
        except UnboundLocalError:
            pass
        try:
            del logits_by_layer
        except UnboundLocalError:
            pass
        try:
            del final_projection_check
        except UnboundLocalError:
            pass
        if model is not None:
            del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


def _layer_entropies(logits_by_layer, layer_indices):
    """Return a tiny audit summary without retaining a second logits copy."""
    import torch

    result = {}
    for index in layer_indices:
        logits = logits_by_layer[index].double()
        log_probabilities = torch.log_softmax(logits, dim=0)
        entropy = -(log_probabilities.exp() * log_probabilities).sum()
        vocab_size = int(logits.numel())
        result[index] = min(1.0, max(0.0, float((entropy / math.log(vocab_size)).item())))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
