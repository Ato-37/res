"""Synthetic-only GPU smoke for question-conditioned temporal-group capture.

This verifies that eager Qwen attention hooks can aggregate only the selected
question-query rows for complete temporal groups without retaining full
attention matrices. It does not score an answer or test faithfulness efficacy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_question_temporal_attention_selector import (  # noqa: E402
    aggregate_temporal_group_attention,
    map_question_token_positions,
    select_top_temporal_groups,
)
from smoke_qwen_internal_attention_ablation import (  # noqa: E402
    MODEL_REVISION,
    _processor_fingerprint,
    synthetic_records,
)
from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256  # noqa: E402
from qwen_internal_attention_ablation import remove_hooks  # noqa: E402


LAYERS_1_BASED = tuple(range(5, 17))
MIN_FREE_BEFORE_LOAD = 11 * 1024**3
MIN_FREE_BEFORE_FORWARD = 3 * 1024**3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="unique aggregate-only JSON path; existing files are refused")
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
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_model_impl
    import transformers.models.qwen2_5_vl.processing_qwen2_5_vl as qwen_processing_impl
    import transformers.models.qwen2_vl.image_processing_qwen2_vl as qwen_image_impl
    import transformers.models.qwen2_vl.video_processing_qwen2_vl as qwen_video_impl
    import qwen_vl_utils.vision_process as qwen_utils_impl
    from qwen_vl_utils import process_vision_info
    from run_nextgqa_gold_interval_margin_pilot import (  # noqa: E402
        build_qa_prompt, process_qa_video,
    )

    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    repo = Path(__file__).resolve().parent.parent
    local_sources = {
        "runner": Path(__file__).resolve(),
        "selector": repo / "scripts/qwen_question_temporal_attention_selector.py",
        "synthetic_fixture": repo / "scripts/smoke_qwen_internal_attention_ablation.py",
        "internal_attention_hook_helper": repo / "scripts/qwen_internal_attention_ablation.py",
        "processor_batch_mapping_helper": repo / "scripts/qwen_processor_batch_review_mapping.py",
        "temporal_mapping_helper": repo / "scripts/qwen_review_interval_token_groups.py",
        "condition_batch_builder": repo / "scripts/nextgqa_condition_batch_builder.py",
        "claim_mask_builder": repo / "scripts/nextgqa_claim_mask_conditions.py",
        "batch_digest_manifest": repo / "scripts/nextgqa_batch_digest_freeze.py",
        "prompt_processor_adapter": repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py",
        "video_decoder_dependency": repo / "scripts/claim_mask_video_decoder.py",
        "mask_condition_dependency": repo / "scripts/necessity_choice_margin_utils.py",
        "time_assignment_dependency": repo / "scripts/presentation_time_utils.py",
        "typed_batch_hash": repo / "scripts/nextgqa_atomic_batch_freeze.py",
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
        "scope": "synthetic_question_temporal_selector_capture_only",
        "faithfulness_result": False,
        "treatment_frozen": False,
        "started_at_utc": started,
        "model_revision": MODEL_REVISION,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "qwen_vl_utils_version": importlib.metadata.version("qwen-vl-utils"),
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
    model = None
    handles = []
    layer_group_scores = {}
    attempted = completed = 0
    stage = "initialize"
    exit_code = 0
    device = None
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
        processor_fingerprint = _processor_fingerprint(processor)
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
        batch = process_qa_video(processor, rendered_text, images, pixels, metadata)
        batch = FrozenBatch(batch)
        ids = batch["input_ids"]
        question_positions = map_question_token_positions(
            tokenizer=processor.tokenizer, rendered_text=rendered_text,
            question_text=item["question"], processed_input_ids=ids,
            video_token_id=int(processor.video_token_id))
        visual_positions = [i for i, token in enumerate(ids[0].tolist())
                            if token == int(processor.video_token_id)]
        temporal_steps, grid_h, grid_w = map(
            int, batch["video_grid_thw"][0].tolist())
        tokens_per_group = (grid_h * grid_w
                            // int(processor.image_processor.merge_size) ** 2)
        if (len(visual_positions) != temporal_steps * tokens_per_group
                or not visual_positions):
            raise RuntimeError("synthetic_temporal_group_geometry_invalid")
        groups = [tuple(visual_positions[index * tokens_per_group:
                                          (index + 1) * tokens_per_group])
                  for index in range(temporal_steps)]

        stage = "load_model"
        free_before, total_vram = torch.cuda.mem_get_info(device)
        if free_before < MIN_FREE_BEFORE_LOAD:
            raise RuntimeError("insufficient_free_vram_before_model_load")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        ).eval()
        if max(LAYERS_1_BASED) > len(model.model.language_model.layers):
            raise RuntimeError("required_layers_unavailable")
        device_batch = batch.to("cuda")
        batch_digest = batch_sha256(device_batch)
        if batch_digest != batch_sha256(batch):
            raise RuntimeError("synthetic_device_batch_digest_mismatch")
        free_before_forward, _ = torch.cuda.mem_get_info(device)
        if free_before_forward < MIN_FREE_BEFORE_FORWARD:
            raise RuntimeError("synthetic_gpu_resource_floor")

        stage = "install_attention_capture"
        for layer_1_based in LAYERS_1_BASED:
            layer_index = layer_1_based - 1

            def capture(_module, _inputs, output, *, selected_layer=layer_index):
                if (not isinstance(output, tuple) or len(output) < 2
                        or output[1] is None):
                    raise RuntimeError("baseline_attention_capture_unavailable")
                layer_group_scores[selected_layer] = aggregate_temporal_group_attention(
                    attention_by_layer={selected_layer: output[1]},
                    layer_indices=(selected_layer,),
                    question_positions=question_positions,
                    group_positions=groups, torch_module=torch)

            handles.append(
                model.model.language_model.layers[layer_index].self_attn
                .register_forward_hook(capture))
        stage = "synthetic_baseline_forward"
        if batch_sha256(device_batch) != batch_digest:
            raise RuntimeError("synthetic_batch_digest_changed_before_forward")
        attempted += 1
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            output = model(**device_batch, use_cache=False, logits_to_keep=1)
        if batch_sha256(device_batch) != batch_digest:
            raise RuntimeError("synthetic_batch_digest_changed_after_forward")
        if not torch.isfinite(output.logits).all():
            raise RuntimeError("synthetic_logits_nonfinite")
        completed += 1
        del output
        if tuple(sorted(layer_group_scores)) != tuple(i - 1 for i in LAYERS_1_BASED):
            raise RuntimeError("baseline_attention_capture_incomplete")
        group_scores = [
            sum(layer_group_scores[layer][group] for layer in layer_group_scores)
            / len(layer_group_scores)
            for group in range(temporal_steps)
        ]
        if (len(group_scores) != temporal_steps
                or any(not np.isfinite(score) or score < 0 for score in group_scores)):
            raise RuntimeError("synthetic_temporal_scores_invalid")
        # K=3 is only a capture-path smoke setting; no candidate treatment uses
        # this value until a separate pre-outcome design decision freezes it.
        smoke_selected = select_top_temporal_groups(group_scores, top_k=3)
        remove_hooks(handles)
        handles = []
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        report.update({
            "status": "passed",
            "processor_fingerprint_sha256": processor_fingerprint,
            "device_name": torch.cuda.get_device_name(device),
            "cuda_version": torch.version.cuda,
            "precision": "bf16_4bit_bf16_compute_eager",
            "attention_backend": "eager",
            "synthetic_records_constructed": 3,
            "records_forwarded": 1,
            "requested_forwards": 1,
            "attempted_forwards": attempted,
            "completed_forwards": completed,
            "captured_layers_1_based": list(LAYERS_1_BASED),
            "question_token_count": len(question_positions),
            "question_positions_sha256": hashlib.sha256(json.dumps(
                question_positions, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            "temporal_group_count": temporal_steps,
            "tokens_per_temporal_group": tokens_per_group,
            "group_scores": group_scores,
            "group_attention_sum": sum(group_scores),
            "smoke_selected_group_indices": smoke_selected,
            "smoke_selection_k_only": True,
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
        })
    except Exception as exc:
        remove_hooks(handles)
        code = str(exc)
        if not code.isascii() or not code.replace("_", "").isalnum():
            code = f"question_selector_smoke_failed_{type(exc).__name__}"
        report.update({"failure_code": code, "failure_stage": stage,
                       "attempted_forwards": attempted,
                       "completed_forwards": completed,
                       "requested_forwards": 1,
                       "elapsed_seconds": time.monotonic() - timer})
        exit_code = 2
    finally:
        remove_hooks(handles)
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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
