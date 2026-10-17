#!/usr/bin/env python3
"""Synthetic Qwen GPU smoke for the fixed visual-key attention-bias treatment.

Runs one deterministic synthetic video twice: baseline with middle-layer
attention capture, then with a frozen top-10% video-key selector and a
positive ln(2) additive bias at the final prompt query in layers 5--16. This
checks hook execution and nonzero logit-path response only. It is not STEAR,
does not read NExT-GQA, and is not a faithfulness or efficacy experiment.
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
from qwen_visual_attention_boost import (  # noqa: E402
    install_video_key_bias_hooks,
    remove_hooks,
)


PAPER_LAYER_FIRST = 5
PAPER_LAYER_LAST = 16
LAYER_INDICES = tuple(range(PAPER_LAYER_FIRST - 1, PAPER_LAYER_LAST))
VISUAL_RATIO = 0.10
LOGIT_BIAS = math.log(2.0)
FRAME_COUNT = 8
MIN_FREE_BEFORE_FORWARD = 768 * 1024**2


def _capture_attention_row(output, layer_index, rows, torch_module):
    if (not isinstance(output, tuple) or len(output) < 2
            or not torch_module.is_tensor(output[1]) or output[1].ndim != 4
            or int(output[1].shape[0]) != 1):
        raise RuntimeError("attention_matrix_unavailable")
    weights = output[1]
    rows[layer_index] = weights[0, :, -1, :].float().mean(dim=0).detach().cpu()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new aggregate-only JSON path; existing outputs are refused")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("output_already_exists")

    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)

    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    stage = "initialize"
    model = None
    hooks = []
    report = {"status": "failed", "scope": "synthetic_hook_smoke_only"}
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
            processor, torch, np, Image, frame_count=FRAME_COUNT)
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
        if len(language_model.layers) < PAPER_LAYER_LAST:
            raise RuntimeError("paper_layer_range_unavailable")
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise RuntimeError("model_processor_video_token_mismatch")
        batch = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                 for key, value in batch_cpu.items()}
        input_ids = batch.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError("synthetic_input_ids_invalid")
        visual_positions = tuple(
            index for index, token in enumerate(input_ids[0].tolist())
            if int(token) == int(processor.video_token_id)
        )
        if not visual_positions or len(groups) != grid[0]:
            raise RuntimeError("synthetic_video_token_mapping_invalid")
        query_position = int(input_ids.shape[1]) - 1
        if visual_positions[-1] >= query_position:
            raise RuntimeError("video_keys_not_causal_predecessors")

        free_before_forward, _ = torch.cuda.mem_get_info(device)
        if free_before_forward < MIN_FREE_BEFORE_FORWARD:
            raise RuntimeError("insufficient_free_vram_before_forward")
        torch.cuda.reset_peak_memory_stats(device)

        attention_rows = {}
        stage = "baseline_attention_capture"
        for layer_index in LAYER_INDICES:
            def capture(_module, _inputs, output, *, selected_layer=layer_index):
                _capture_attention_row(output, selected_layer, attention_rows, torch)
            hooks.append(language_model.layers[layer_index].self_attn
                         .register_forward_hook(capture))
        with torch.inference_mode():
            baseline_output = model(**batch, use_cache=False, logits_to_keep=1)
        for handle in hooks:
            handle.remove()
        hooks.clear()
        if tuple(sorted(attention_rows)) != LAYER_INDICES:
            raise RuntimeError("baseline_attention_capture_incomplete")
        compact_rows = [None] * PAPER_LAYER_LAST
        for index, row in attention_rows.items():
            compact_rows[index] = row.view(1, 1, 1, int(input_ids.shape[1]))
        selected_positions, selection_scores = select_top_visual_positions(
            compact_rows, visual_positions, layer_indices=LAYER_INDICES,
            ratio=VISUAL_RATIO)
        if len(selected_positions) != max(1, math.ceil(len(visual_positions) * VISUAL_RATIO)):
            raise RuntimeError("frozen_selector_count_invalid")
        if not all(math.isfinite(float(value)) for value in selection_scores):
            raise RuntimeError("baseline_selector_scores_nonfinite")

        stage = "treatment_attention_bias_forward"
        hooks = install_video_key_bias_hooks(
            model=model, layer_indices=LAYER_INDICES,
            selected_positions=selected_positions,
            query_position=query_position, logit_bias=LOGIT_BIAS,
            torch_module=torch)
        with torch.inference_mode():
            treatment_output = model(**batch, use_cache=False, logits_to_keep=1)
        for handle in hooks:
            handle.remove()
        hooks.clear()
        torch.cuda.synchronize(device)
        baseline_logits = baseline_output.logits[0, -1].float()
        treatment_logits = treatment_output.logits[0, -1].float()
        if (not bool(torch.isfinite(baseline_logits).all())
                or not bool(torch.isfinite(treatment_logits).all())):
            raise RuntimeError("nonfinite_final_logits")
        delta = treatment_logits - baseline_logits
        if not bool(torch.isfinite(delta).all()) or float(delta.abs().max()) == 0.0:
            raise RuntimeError("intervention_no_finite_logit_effect")
        free_after, _ = torch.cuda.mem_get_info(device)
        report = {
            "status": "passed",
            "scope": "synthetic_hook_and_forward_response_only",
            "started_at_utc": started,
            "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "intervention_helper_sha256": hashlib.sha256(
                Path(__file__).with_name("qwen_visual_attention_boost.py").read_bytes()
            ).hexdigest(),
            "model_revision": MODEL_REVISION,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "precision": "bf16_4bit_bf16_compute",
            "synthetic_frames": FRAME_COUNT,
            "video_grid_thw": [int(value) for value in grid],
            "temporal_groups": len(groups),
            "video_token_count": len(visual_positions),
            "selected_video_token_count": len(selected_positions),
            "selected_video_token_ratio": VISUAL_RATIO,
            "intervention_layers_1_based": [PAPER_LAYER_FIRST, PAPER_LAYER_LAST],
            "fixed_answer_query_position": query_position,
            "key_attention_logit_bias": LOGIT_BIAS,
            "odds_multiplier": math.exp(LOGIT_BIAS),
            "forward_calls": 2,
            "finite_final_logits": True,
            "max_abs_logit_change": float(delta.abs().max().item()),
            "rms_logit_change": float(delta.square().mean().sqrt().item()),
            "top_choice_changed": int(baseline_logits.argmax()) != int(treatment_logits.argmax()),
            "free_vram_before_load_gib": free_before / 1024**3,
            "total_vram_gib": total / 1024**3,
            "free_vram_before_forward_gib": free_before_forward / 1024**3,
            "free_vram_after_gib": free_after / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "elapsed_seconds": time.monotonic() - timer,
            "faithfulness_result": False,
            "limitations": [
                "deterministic synthetic video only; no dataset or human review",
                "fixed top-10-percent self-attention selector is an engineering choice",
                "positive ln(2) key-logit bias is a Qwen adaptation, not STEAR",
                "two-forward logit change does not estimate evidence dependence or efficacy",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return 0
    except Exception as exc:
        for handle in hooks:
            try:
                handle.remove()
            except Exception:
                pass
        message = str(exc)
        failure_code = (message if message.isascii() and message.replace("_", "").isalnum()
                        else f"smoke_failed_{stage}_{type(exc).__name__}")
        report.update({"failure_code": failure_code, "failure_stage": stage,
                       "started_at_utc": started,
                       "elapsed_seconds": time.monotonic() - timer,
                       "model_revision": MODEL_REVISION,
                       "forward_calls": 0 if stage in ("initialize", "load_model") else None,
                       "faithfulness_result": False})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return 2
    finally:
        for handle in hooks:
            try:
                handle.remove()
            except Exception:
                pass
        if model is not None:
            del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
