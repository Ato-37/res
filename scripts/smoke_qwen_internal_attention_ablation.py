#!/usr/bin/env python3
"""Synthetic-only one-record Qwen internal key-ablation wiring smoke.

Runs one full-video forward and one forward with complete temporal-grid groups
mapped from a synthetic interval masked at selected decoder layers. This tests
GPU hook wiring and finite output only; it is not a faithfulness result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from qwen_internal_attention_ablation import (  # noqa: E402
    install_video_key_ablation_hooks, remove_hooks,
)
from qwen_processor_batch_review_mapping import (  # noqa: E402
    map_review_interval_from_processor_batch,
)
from nextgqa_condition_batch_builder import build_exact_triplet_batches  # noqa: E402
from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256  # noqa: E402


MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
LAYERS_1_BASED = (5, 10, 16)
MIN_FREE_BEFORE_LOAD = 12 * 1024**3
MIN_FREE_BEFORE_FORWARD = 768 * 1024**2


def synthetic_records(np_module, image_type):
    """Build deterministic 32-frame 224x224 synthetic RGB clips and QA rows."""
    y, x = np_module.mgrid[0:224, 0:224]
    records = []
    for row_index in range(3):
        frames = []
        for frame_index in range(32):
            rgb = np_module.stack((
                (x * 3 + frame_index * 7 + row_index * 17) % 256,
                (y * 5 + frame_index * 11 + row_index * 29) % 256,
                (x + y * 2 + frame_index * 13 + row_index * 43) % 256,
            ), axis=-1).astype(np_module.uint8)
            frames.append(image_type.fromarray(rgb, mode="RGB"))
        records.append({
            "frames": frames,
            "presentation_times": [Fraction(i) for i in range(32)],
            "sampled_indices": list(range(32)),
            "intervals": [[Fraction(5), Fraction(8)],
                          [Fraction(24), Fraction(27)]],
            "qa_item": {
                "question": "Which synthetic color channel changes most over time?",
                "choices": ["red", "green", "blue", "all equally", "cannot tell"],
                "gold": row_index % 3,
            },
            "sample_fps": Fraction(1),
        })
    return records


def _processor_fingerprint(processor) -> str:
    tokenizer_json = processor.tokenizer.backend_tokenizer.to_str()
    descriptor = {
        "processor_class": type(processor).__name__,
        "image_processor": processor.image_processor.to_dict(),
        "tokenizer_json_sha256": hashlib.sha256(
            tokenizer_json.encode("utf-8")).hexdigest(),
        "video_token_id": int(processor.video_token_id),
        "model_revision": MODEL_REVISION,
    }
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"),
                         allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
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
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_model_impl
    import transformers.models.qwen2_5_vl.processing_qwen2_5_vl as qwen_processing_impl
    import transformers.models.qwen2_vl.image_processing_qwen2_vl as qwen_image_impl
    import transformers.models.qwen2_vl.video_processing_qwen2_vl as qwen_video_impl
    import qwen_vl_utils.vision_process as qwen_utils_impl

    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    repo = Path(__file__).resolve().parent.parent
    source_files = {
        "runner": Path(__file__).resolve(),
        "ablation_helper": repo / "scripts/qwen_internal_attention_ablation.py",
        "temporal_mapping_helper": repo / "scripts/qwen_review_interval_token_groups.py",
        "processor_batch_mapping_helper": repo / "scripts/qwen_processor_batch_review_mapping.py",
        "condition_batch_builder": repo / "scripts/nextgqa_condition_batch_builder.py",
        "batch_digest_manifest": repo / "scripts/nextgqa_batch_digest_freeze.py",
        "claim_mask_builder": repo / "scripts/nextgqa_claim_mask_conditions.py",
        "prompt_and_processor_adapter": repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py",
        "typed_batch_hash": repo / "scripts/nextgqa_atomic_batch_freeze.py",
        "mask_condition_dependency": repo / "scripts/necessity_choice_margin_utils.py",
        "time_assignment_dependency": repo / "scripts/presentation_time_utils.py",
    }
    report = {"status": "incomplete", "scope": "synthetic_internal_ablation_wiring_only",
              "faithfulness_result": False, "execution_authorization": "not_granted",
              "source_sha256": {
                  name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in source_files.items()
              }}
    for source_name, module in (
            ("transformers_qwen_model_impl", qwen_model_impl),
            ("transformers_qwen_processing_impl", qwen_processing_impl),
            ("transformers_qwen_image_processor_impl", qwen_image_impl),
            ("transformers_video_processor_impl", qwen_video_impl),
            ("qwen_vl_utils_video_impl", qwen_utils_impl)):
        report["source_sha256"][source_name] = hashlib.sha256(
            Path(module.__file__).read_bytes()).hexdigest()
    qwen_vl_utils_version = importlib.metadata.version("qwen-vl-utils")
    model = None
    handles = []
    attempted = completed = 0
    stage = "initialize"
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
        records = synthetic_records(np, Image)
        built = build_exact_triplet_batches(records=records, processor=processor)
        batch = built["triplets"][0]["batches"]["full"]
        ids = batch["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or int(ids.shape[1]) < 2:
            raise RuntimeError("synthetic_processor_geometry_invalid")
        temporal_patch_size = int(processor.image_processor.temporal_patch_size)
        mapping = map_review_interval_from_processor_batch(
            batch=batch, torch_module=torch,
            video_token_id=int(processor.video_token_id),
            spatial_merge_size=int(processor.image_processor.merge_size),
            temporal_patch_size=temporal_patch_size,
            sampled_frame_indices=records[0]["sampled_indices"],
            sampled_pts_seconds=records[0]["presentation_times"],
            source_frame_count=32, video_duration_seconds=32,
            sample_fps=records[0]["sample_fps"],
            processor_revision=MODEL_REVISION,
            processor_fingerprint_sha256=processor_fingerprint,
            reviewed_intervals_seconds=records[0]["intervals"],
        )
        batch = mapping.pop("_processor_batch_snapshot")
        ids = batch["input_ids"]
        query_position = int(ids.shape[1]) - 1
        selected_positions = mapping["evidence_token_positions"]
        if (not selected_positions or max(selected_positions) >= query_position
                or not set(selected_positions).issubset(
                    {i for i, token in enumerate(ids[0].tolist())
                     if token == int(processor.video_token_id)})):
            raise RuntimeError("synthetic_evidence_positions_invalid")
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
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise RuntimeError("model_processor_video_token_mismatch")
        if max(LAYERS_1_BASED) > len(model.model.language_model.layers):
            raise RuntimeError("required_layers_unavailable")
        torch.cuda.reset_peak_memory_stats(device)
        device_batch = FrozenBatch(batch).to("cuda")
        if batch_sha256(device_batch) != mapping["typed_batch_sha256"]:
            raise RuntimeError("synthetic_device_batch_digest_mismatch")
        free, _ = torch.cuda.mem_get_info(device)
        if free < MIN_FREE_BEFORE_FORWARD:
            raise RuntimeError("synthetic_gpu_resource_floor")
        stage = "baseline_forward"
        if batch_sha256(device_batch) != mapping["typed_batch_sha256"]:
            raise RuntimeError("synthetic_batch_digest_changed_before_baseline")
        attempted += 1
        with torch.inference_mode():
            baseline = model(**device_batch, use_cache=False, logits_to_keep=1)
        if batch_sha256(device_batch) != mapping["typed_batch_sha256"]:
            raise RuntimeError("synthetic_batch_digest_changed_after_baseline")
        if not torch.isfinite(baseline.logits).all():
            raise RuntimeError("baseline_logits_nonfinite")
        completed += 1
        del baseline
        handles = install_video_key_ablation_hooks(
            model=model, layer_indices=tuple(index - 1 for index in LAYERS_1_BASED),
            selected_positions=selected_positions, query_position=query_position,
            torch_module=torch,
        )
        stage = "ablation_forward"
        if batch_sha256(device_batch) != mapping["typed_batch_sha256"]:
            raise RuntimeError("synthetic_batch_digest_changed_before_ablation")
        attempted += 1
        with torch.inference_mode():
            treated = model(**device_batch, use_cache=False, logits_to_keep=1)
        if batch_sha256(device_batch) != mapping["typed_batch_sha256"]:
            raise RuntimeError("synthetic_batch_digest_changed_after_ablation")
        if not torch.isfinite(treated.logits).all():
            raise RuntimeError("ablated_logits_nonfinite")
        completed += 1
        del treated
        remove_hooks(handles)
        handles = []
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        report.update({
            "status": "passed", "started_at_utc": started,
            "model_revision": MODEL_REVISION, "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "qwen_vl_utils_version": qwen_vl_utils_version,
            "processor_fingerprint_sha256": processor_fingerprint,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "precision": "bf16_4bit_bf16_compute_eager",
            "synthetic_records_constructed": len(records),
            "records_forwarded": 1, "requested_forwards": 2,
            "attempted_forwards": attempted, "completed_forwards": completed,
            "temporal_group_count": mapping["temporal_group_count"],
            "total_temporal_group_count": mapping["total_temporal_group_count"],
            "reviewed_interval_count": mapping["reviewed_interval_count"],
            "reviewed_occurrence_group_indices": mapping[
                "reviewed_occurrence_group_indices"],
            "evidence_group_indices": mapping["evidence_group_indices"],
            "control_group_indices": mapping["control_group_indices"],
            "query_position": query_position,
            "selected_positions_sha256": hashlib.sha256(json.dumps(
                selected_positions, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            "token_budget_per_arm": mapping["token_budget_per_arm"],
            "typed_batch_sha256": mapping["typed_batch_sha256"],
            "sampling_metadata_sha256": mapping["sampling_metadata_sha256"],
            "source_authentication_status": mapping[
                "source_authentication_status"],
            "temporal_mapping_diagnostics": {
                "max_sampled_pts_gap_seconds": mapping["max_sampled_pts_gap_seconds"],
                "first_edge_cell_duration_seconds": mapping[
                    "first_edge_cell_duration_seconds"],
                "last_edge_cell_duration_seconds": mapping[
                    "last_edge_cell_duration_seconds"],
                "max_temporal_group_coverage_seconds": mapping[
                    "max_temporal_group_coverage_seconds"],
                "max_pts_cadence_deviation_from_nominal_seconds": mapping[
                    "max_pts_cadence_deviation_from_nominal_seconds"],
            },
            "intervention_layers_1_based": list(LAYERS_1_BASED),
            "attention_backend": "eager",
            "free_vram_before_load_gib": free_before / 1024**3,
            "total_vram_gib": total_vram / 1024**3,
            "free_vram_after_gib": free_after / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "elapsed_seconds": time.monotonic() - timer,
            "faithfulness_result": False,
            "limitations": [
                "synthetic-only and one-record mechanism wiring test; no efficacy inference",
                "sampling times are synthetic and do not establish real source-clock mapping",
                "masking selected answer-query access does not erase propagated evidence",
                "does not test human review, real QA scoring, or processor-batch provenance gate",
            ],
        })
    except Exception as exc:
        remove_hooks(handles)
        code = str(exc)
        if not code.isascii() or not code.replace("_", "").isalnum():
            code = f"internal_ablation_smoke_failed_{type(exc).__name__}"
        report.update({"failure_code": code, "attempted_forwards": attempted,
                       "completed_forwards": completed,
                       "failure_stage": stage,
                       "requested_forwards": 2,
                       "elapsed_seconds": time.monotonic() - timer})
        status = 2
    else:
        status = 0
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
    return status


if __name__ == "__main__":
    raise SystemExit(main())
