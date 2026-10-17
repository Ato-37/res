#!/usr/bin/env python3
"""Synthetic-only GPU integration smoke for question-selected attention bias.

Runs baseline, question-selector treatment, and deterministic sham across
full/evidence-key-ablation/control-key-ablation branches. It checks hook
composition, immutable inputs, fixed-prediction score extraction, and finite
outputs. Its synthetic score diagnostics are not faithfulness evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_internal_attention_ablation import (  # noqa: E402
    install_video_key_intervention_hooks, remove_hooks,
)
from qwen_question_temporal_attention_selector import (  # noqa: E402
    aggregate_temporal_group_attention, map_question_token_positions,
    select_hash_sham_temporal_groups, select_top_temporal_groups,
)
from fixed_prediction_selective_dependence import (  # noqa: E402
    fixed_prediction_intervention_contrast,
)
from smoke_qwen_question_temporal_selector import (  # noqa: E402
    LAYERS_1_BASED, MODEL_REVISION, _processor_fingerprint, synthetic_records,
)
from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256  # noqa: E402
from run_nextgqa_gold_interval_margin_pilot import (  # noqa: E402
    build_qa_prompt, process_qa_video,
)


LOGIT_BIAS = math.log(2.0)
SHAM_SEED = "qta-sham-seed-20261018-v1"
MIN_FREE_BEFORE_LOAD = 11 * 1024**3
MIN_FREE_BEFORE_FORWARD = 1024**3


def build_smoke_condition_plan(*, target_positions, sham_positions,
                               evidence_positions, control_positions):
    """Return the fixed synthetic baseline/target/sham x full/E/C branch plan."""
    return [
        {"name": "baseline_full", "bias": (), "ablation": ()},
        {"name": "baseline_evidence_ablation", "bias": (),
         "ablation": tuple(evidence_positions)},
        {"name": "baseline_control_ablation", "bias": (),
         "ablation": tuple(control_positions)},
        {"name": "target_full", "bias": tuple(target_positions), "ablation": ()},
        {"name": "target_evidence_ablation", "bias": tuple(target_positions),
         "ablation": tuple(evidence_positions)},
        {"name": "target_control_ablation", "bias": tuple(target_positions),
         "ablation": tuple(control_positions)},
        {"name": "sham_full", "bias": tuple(sham_positions), "ablation": ()},
        {"name": "sham_evidence_ablation", "bias": tuple(sham_positions),
         "ablation": tuple(evidence_positions)},
        {"name": "sham_control_ablation", "bias": tuple(sham_positions),
         "ablation": tuple(control_positions)},
    ]


def _choice_log_probabilities(logits, token_ids, torch_module):
    """Extract full-vocabulary next-token log probabilities for A–E IDs."""
    if (not torch_module.is_tensor(logits) or logits.ndim != 3
            or tuple(logits.shape[:2]) != (1, 1)
            or not torch_module.isfinite(logits).all()
            or len(token_ids) != 5
            or any(type(token_id) is not int or token_id < 0
                   or token_id >= int(logits.shape[-1]) for token_id in token_ids)
            or len(set(token_ids)) != 5):
        raise ValueError("choice_score_logits_or_token_ids_invalid")
    log_probs = torch_module.log_softmax(logits[0, -1].float(), dim=-1)
    selected = log_probs[list(token_ids)].detach().cpu().tolist()
    if len(selected) != 5 or not all(math.isfinite(float(value)) for value in selected):
        raise ValueError("choice_score_nonfinite")
    return [float(value) for value in selected]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new aggregate-only JSON path; existing outputs are refused")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("output_already_exists")

    import numpy as np
    import PIL
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig
    from transformers import Qwen2_5_VLForConditionalGeneration
    bnb_version = importlib.metadata.version("bitsandbytes")
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_model_impl
    import transformers.models.qwen2_5_vl.processing_qwen2_5_vl as qwen_processing_impl
    import transformers.models.qwen2_vl.image_processing_qwen2_vl as qwen_image_impl
    import transformers.models.qwen2_vl.video_processing_qwen2_vl as qwen_video_impl
    import qwen_vl_utils.vision_process as qwen_utils_impl
    from qwen_vl_utils import process_vision_info

    repo = Path(__file__).resolve().parents[1]
    local_sources = {
        "runner": Path(__file__).resolve(),
        "selector_runner_fixture": repo / "scripts/smoke_qwen_question_temporal_selector.py",
        "synthetic_fixture": repo / "scripts/smoke_qwen_internal_attention_ablation.py",
        "selector": repo / "scripts/qwen_question_temporal_attention_selector.py",
        "attention_bias": repo / "scripts/qwen_visual_attention_boost.py",
        "internal_ablation": repo / "scripts/qwen_internal_attention_ablation.py",
        "fixed_prediction_scorer": repo / "scripts/fixed_prediction_selective_dependence.py",
        "typed_batch_hash": repo / "scripts/nextgqa_atomic_batch_freeze.py",
        "prompt_processor_adapter": repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py",
        "processor_batch_mapping": repo / "scripts/qwen_processor_batch_review_mapping.py",
        "temporal_group_mapping": repo / "scripts/qwen_review_interval_token_groups.py",
        "condition_batch_builder": repo / "scripts/nextgqa_condition_batch_builder.py",
        "claim_mask_builder": repo / "scripts/nextgqa_claim_mask_conditions.py",
        "claim_mask_video_decoder": repo / "scripts/claim_mask_video_decoder.py",
        "mask_condition_dependency": repo / "scripts/necessity_choice_margin_utils.py",
        "presentation_time_utils": repo / "scripts/presentation_time_utils.py",
    }
    external_modules = {
        "transformers_qwen_model_impl": qwen_model_impl,
        "transformers_qwen_processing_impl": qwen_processing_impl,
        "transformers_qwen_image_processor_impl": qwen_image_impl,
        "transformers_video_processor_impl": qwen_video_impl,
        "qwen_vl_utils_video_impl": qwen_utils_impl,
    }
    report = {
        "status": "incomplete",
        "scope": "synthetic_question_selector_bias_and_key_ablation_wiring_only",
        "faithfulness_result": False,
        "treatment_frozen": False,
        "execution_authorization": "not_granted",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_revision": MODEL_REVISION,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "qwen_vl_utils_version": importlib.metadata.version("qwen-vl-utils"),
        "bitsandbytes_version": bnb_version,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pillow_version": PIL.__version__,
        "checkpoint_weights_hash_status": "not_computed_local_cached_revision_only",
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in local_sources.items()
        },
    }
    report["source_sha256"].update({
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in external_modules.items()
    })

    timer = time.monotonic()
    model = None
    capture_handles = []
    attempted = completed = 0
    stage = "initialize"
    exit_code = 0
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
        record = synthetic_records(np, Image)[0]
        item = {**record, **record["qa_item"]}
        sample_fps = (record["sample_fps"].numerator,
                      record["sample_fps"].denominator)
        rendered_text, messages = build_qa_prompt(
            processor, item, record["frames"], sample_fps)
        images, videos = process_vision_info(messages, return_video_metadata=True)
        if images is not None or len(videos) != 1:
            raise RuntimeError("synthetic_video_processor_contract_failed")
        pixels, metadata = videos[0]
        batch = FrozenBatch(process_qa_video(processor, rendered_text, images,
                                             pixels, metadata))
        ids = batch["input_ids"]
        if ids.ndim != 2 or int(ids.shape[0]) != 1:
            raise RuntimeError("synthetic_prompt_batch_invalid")
        option_token_ids = []
        for label in "ABCDE":
            encoded = processor.tokenizer.encode(label, add_special_tokens=False)
            if len(encoded) != 1:
                raise RuntimeError("synthetic_answer_label_not_single_token")
            option_token_ids.append(int(encoded[0]))
        if len(set(option_token_ids)) != 5:
            raise RuntimeError("synthetic_answer_label_token_collision")
        question_positions = map_question_token_positions(
            tokenizer=processor.tokenizer, rendered_text=rendered_text,
            question_text=item["question"], processed_input_ids=ids,
            video_token_id=int(processor.video_token_id))
        query_position = int(ids.shape[1]) - 1
        attention_mask = batch.get("attention_mask")
        if (attention_mask is None or tuple(attention_mask.shape) != tuple(ids.shape)
                or not bool((attention_mask == 1).all())
                or max(question_positions) >= query_position):
            raise RuntimeError("synthetic_single_prompt_query_geometry_invalid")
        visual_positions = [i for i, token in enumerate(ids[0].tolist())
                            if token == int(processor.video_token_id)]
        temporal_steps, grid_h, grid_w = map(
            int, batch["video_grid_thw"][0].tolist())
        tokens_per_group = (grid_h * grid_w
                            // int(processor.image_processor.merge_size) ** 2)
        if len(visual_positions) != temporal_steps * tokens_per_group:
            raise RuntimeError("synthetic_temporal_group_geometry_invalid")
        groups = [tuple(visual_positions[i * tokens_per_group:(i + 1) * tokens_per_group])
                  for i in range(temporal_steps)]
        if temporal_steps < 4:
            raise RuntimeError("synthetic_temporal_group_count_invalid")

        free_before, total_vram = torch.cuda.mem_get_info(device)
        if free_before < MIN_FREE_BEFORE_LOAD:
            raise RuntimeError("insufficient_free_vram_before_model_load")
        stage = "load_model"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        ).eval()
        if (int(model.config.video_token_id) != int(processor.video_token_id)
                or len(model.model.language_model.layers) < max(LAYERS_1_BASED)):
            raise RuntimeError("model_processor_or_layer_contract_invalid")
        device_batch = batch.to("cuda")
        batch_digest = batch_sha256(device_batch)
        if batch_digest != batch_sha256(batch):
            raise RuntimeError("synthetic_device_batch_digest_mismatch")
        free_before_forward, _ = torch.cuda.mem_get_info(device)
        if free_before_forward < MIN_FREE_BEFORE_FORWARD:
            raise RuntimeError("synthetic_gpu_resource_floor")

        captured = {}
        stage = "capture_question_attention"
        for layer_1_based in LAYERS_1_BASED:
            layer_index = layer_1_based - 1

            def capture(_module, _inputs, output, *, selected_layer=layer_index):
                if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                    raise RuntimeError("baseline_attention_capture_unavailable")
                captured[selected_layer] = aggregate_temporal_group_attention(
                    attention_by_layer={selected_layer: output[1]},
                    layer_indices=(selected_layer,),
                    question_positions=question_positions,
                    group_positions=groups, torch_module=torch)

            capture_handles.append(
                model.model.language_model.layers[layer_index].self_attn
                .register_forward_hook(capture))
        torch.cuda.reset_peak_memory_stats(device)
        attempted += 1
        with torch.inference_mode():
            baseline_output = model(**device_batch, use_cache=False, logits_to_keep=1)
        if (batch_sha256(device_batch) != batch_digest
                or not torch.isfinite(baseline_output.logits).all()):
            raise RuntimeError("baseline_snapshot_or_logits_invalid")
        baseline_full_choice_log_probs = _choice_log_probabilities(
            baseline_output.logits, option_token_ids, torch)
        del baseline_output
        completed += 1
        if tuple(sorted(captured)) != tuple(i - 1 for i in LAYERS_1_BASED):
            raise RuntimeError("question_attention_capture_incomplete")
        group_scores = [sum(captured[layer][group] for layer in captured) / len(captured)
                        for group in range(temporal_steps)]
        top_k = max(1, math.ceil(temporal_steps * 0.10))
        target_groups = select_top_temporal_groups(group_scores, top_k=top_k)
        sham_groups = select_hash_sham_temporal_groups(
            temporal_group_count=temporal_steps, top_k=top_k,
            protocol_seed=SHAM_SEED, opaque_record_token="synthetic-record-0")
        target_positions = tuple(sorted(p for index in target_groups for p in groups[index]))
        sham_positions = tuple(sorted(p for index in sham_groups for p in groups[index]))

        # Synthetic diagnostic branches only. Group choices have no annotation semantics.
        evidence_positions = tuple(sorted(p for group in groups[:2] for p in group))
        control_positions = tuple(sorted(p for group in groups[-2:] for p in group))
        remove_hooks(capture_handles)
        capture_handles = []
        condition_outputs = {}

        def forward_condition(name, *, bias_positions=(), ablation_positions=()):
            nonlocal attempted, completed
            hooks = []
            try:
                hooks = install_video_key_intervention_hooks(
                    model=model, layer_indices=tuple(i - 1 for i in LAYERS_1_BASED),
                    biased_positions=bias_positions,
                    ablated_positions=ablation_positions,
                    query_position=int(ids.shape[1]) - 1,
                    logit_bias=LOGIT_BIAS, torch_module=torch)
                if batch_sha256(device_batch) != batch_digest:
                    raise RuntimeError("synthetic_batch_digest_changed_before_condition")
                attempted += 1
                with torch.inference_mode():
                    output = model(**device_batch, use_cache=False, logits_to_keep=1)
                if (batch_sha256(device_batch) != batch_digest
                        or not torch.isfinite(output.logits).all()):
                    raise RuntimeError("condition_snapshot_or_logits_invalid")
                condition_outputs[name] = {
                    "logits_finite": True,
                    "logits_shape": list(output.logits.shape),
                    "choice_log_probabilities": _choice_log_probabilities(
                        output.logits, option_token_ids, torch),
                }
                completed += 1
                del output
            finally:
                remove_hooks(hooks)

        stage = "nine_branch_hook_composition"
        branch_plan = build_smoke_condition_plan(
            target_positions=target_positions, sham_positions=sham_positions,
            evidence_positions=evidence_positions, control_positions=control_positions)
        branch_inputs = {branch["name"]: branch for branch in branch_plan}
        # Reuse the already completed baseline full pass.
        condition_outputs["baseline_full"] = {
            "logits_finite": True, "forward_reused": True,
            "choice_log_probabilities": baseline_full_choice_log_probs,
        }
        for branch in branch_plan[1:]:
            forward_condition(branch["name"], bias_positions=branch["bias"],
                              ablation_positions=branch["ablation"])
        if set(condition_outputs) != set(branch_inputs) or completed != 9 or attempted != 9:
            raise RuntimeError("synthetic_nine_condition_incomplete")

        contrast_inputs = {
            name: condition_outputs[name]["choice_log_probabilities"]
            for name in condition_outputs
        }
        target_contrast = fixed_prediction_intervention_contrast({
            "baseline_full": contrast_inputs["baseline_full"],
            "baseline_evidence_ablation": contrast_inputs["baseline_evidence_ablation"],
            "baseline_control_ablation": contrast_inputs["baseline_control_ablation"],
            "intervention_full": contrast_inputs["target_full"],
            "intervention_evidence_ablation": contrast_inputs["target_evidence_ablation"],
            "intervention_control_ablation": contrast_inputs["target_control_ablation"],
        })
        sham_contrast = fixed_prediction_intervention_contrast({
            "baseline_full": contrast_inputs["baseline_full"],
            "baseline_evidence_ablation": contrast_inputs["baseline_evidence_ablation"],
            "baseline_control_ablation": contrast_inputs["baseline_control_ablation"],
            "intervention_full": contrast_inputs["sham_full"],
            "intervention_evidence_ablation": contrast_inputs["sham_evidence_ablation"],
            "intervention_control_ablation": contrast_inputs["sham_control_ablation"],
        })

        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        report.update({
            "status": "passed",
            "device_name": torch.cuda.get_device_name(device),
            "cuda_version": torch.version.cuda,
            "processor_fingerprint_sha256": _processor_fingerprint(processor),
            "precision": "bf16_4bit_bf16_compute_eager",
            "requested_records": 1,
            "completed_records": 1,
            "requested_forwards": 9,
            "attempted_forwards": attempted,
            "completed_forwards": completed,
            "captured_layers_1_based": list(LAYERS_1_BASED),
            "question_token_count": len(question_positions),
            "temporal_group_count": temporal_steps,
            "tokens_per_temporal_group": tokens_per_group,
            "top_k_budget_rule": "ceil(temporal_groups*0.10)_engineering_smoke_only",
            "selected_target_group_indices": target_groups,
            "selected_sham_group_indices": sham_groups,
            "target_sham_overlap_count": len(set(target_groups) & set(sham_groups)),
            "sham_seed": SHAM_SEED,
            "sham_selector": "sha256_utf8_qta-sham-v1_seed_opaque-token_group-index",
            "logit_bias": LOGIT_BIAS,
            "intervention_layers_1_based": list(LAYERS_1_BASED),
            "condition_names": [branch["name"] for branch in branch_plan],
            "condition_forward_checks": condition_outputs,
            "synthetic_fixed_prediction_score_diagnostic": {
                "status": "synthetic_numerical_integration_only",
                "option_label_order": list("ABCDE"),
                "option_token_ids": option_token_ids,
                "target_delta_f": target_contrast["delta_f"],
                "sham_delta_f": sham_contrast["delta_f"],
                "target_minus_sham_delta_f": (
                    target_contrast["delta_f"] - sham_contrast["delta_f"]),
                "answer_index_frozen_from_baseline_full": target_contrast["answer_index"],
                "foil_index_frozen_from_baseline_full": target_contrast["foil_index"],
                "interpretation": "synthetic score path only; no faithfulness result",
            },
            "evidence_control_key_counts_equal": len(evidence_positions) == len(control_positions),
            "typed_batch_sha256": batch_digest,
            "free_vram_before_load_gib": free_before / 1024**3,
            "total_vram_gib": total_vram / 1024**3,
            "free_vram_before_forward_gib": free_before_forward / 1024**3,
            "free_vram_after_gib": free_after / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "elapsed_seconds": time.monotonic() - timer,
            "faithfulness_result": False,
            "treatment_frozen": False,
            "limitations": [
                "synthetic video only; no semantic evidence or treatment efficacy",
                "the selector layer range, 10-percent group budget, ln(2) dose, and sham seed are engineering smoke settings",
                "synthetic evidence/control groups have no annotation meaning",
                "synthetic A-E scores and contrasts are numerical diagnostics only; no real-data effect, interval, or hypothesis test",
            ],
        })
    except Exception as exc:
        remove_hooks(capture_handles)
        code = str(exc)
        if not code.isascii() or not code.replace("_", "").isalnum():
            code = f"question_intervention_smoke_failed_{type(exc).__name__}"
        report.update({"failure_code": code, "failure_stage": stage,
                       "attempted_forwards": attempted,
                       "completed_forwards": completed,
                       "requested_forwards": 9,
                       "elapsed_seconds": time.monotonic() - timer})
        exit_code = 2
    finally:
        remove_hooks(capture_handles)
        if model is not None:
            del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    if exit_code == 0:
        print("synthetic score-path integration passed; no faithfulness result")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
