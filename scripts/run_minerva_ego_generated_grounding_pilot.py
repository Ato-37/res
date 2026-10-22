#!/usr/bin/env python3
"""Generate one real-video explanation under baseline/support/sham attention.

The deterministic lexical/time scorer checks whether a named-object timestamp
overlaps annotated movement intervals. It does not judge action semantics,
object identity beyond the annotation name/head alias, or explanation truth.
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

from run_minerva_ego_anchor_gpu_pilot import (
    FRAME_COUNT, MODEL_REVISION, ROW_KEY, VIDEO_ID, VIDEO_MD5,
    _video_groups, decode_uniform_video, sha,
)

ROOT = Path(__file__).resolve().parents[1]
LAYERS_1_BASED = tuple(range(5, 17))
BIAS_NATS = math.log(2.0)


def _observe_shapes(model, layer_indices):
    observations = {index: [] for index in layer_indices}
    handles = []
    for index in layer_indices:
        attention = model.model.language_model.layers[index].self_attn
        def observe(_module, _args, kwargs, *, layer_index=index):
            mask = kwargs.get("attention_mask")
            if (mask is None or not hasattr(mask, "shape") or len(mask.shape) != 4
                    or tuple(mask.shape[:2]) != (1, 1)):
                raise ValueError("generation_shape_observation_invalid")
            observations[layer_index].append([int(mask.shape[-2]), int(mask.shape[-1])])
        handles.append(attention.register_forward_pre_hook(observe, with_kwargs=True))
    return observations, handles


def build_grounding_prompt(row, target_name):
    """Build a per-row prompt using the joined annotation's target name."""
    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError("target_object_name_invalid")
    target_name = target_name.strip()
    target_alias = target_name.split()[-1]
    options = "\n".join(
        f"{chr(65 + i)}. {row[f'answer_choice_{i}']}" for i in range(5))
    return (row["question"] + "\nOptions:\n" + options +
        f"\nDescribe only the movements of the {target_name}, not the "
        "multiple-choice answer. Write one short sentence per movement. "
        "Include a real numeric video time in M:SS form (for example, 2:04); "
        "never copy a placeholder such as 'mm:ss' or '0mm:ss'. "
        f"Repeat '{target_alias}' in every timed sentence. If you cannot "
        "identify an actual time, omit that movement sentence.")


def _generate(*, name, positions, model, processor, batch, prompt_length, torch):
    from qwen_visual_attention_boost import (
        install_generation_video_key_bias_hooks, remove_hooks)
    from run_qta_intervened_interval_generation import (
        validate_dynamic_shape_trace, validate_generation_mask_audit)
    layer_indices = tuple(layer - 1 for layer in LAYERS_1_BASED)
    shape_observations, shape_handles = _observe_shapes(model, layer_indices)
    mask_observations = []
    intervention_handles = []
    try:
        if positions:
            intervention_handles = install_generation_video_key_bias_hooks(
                model=model, layer_indices=layer_indices,
                selected_positions=positions,
                first_query_position=prompt_length - 1,
                logit_bias=BIAS_NATS, torch_module=torch,
                observations=mask_observations)
        moved = {key: value.to("cuda") if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        free, _total = torch.cuda.mem_get_info()
        if free < 2 * 1024**3:
            raise RuntimeError("generation_free_vram_guard_failed")
        with torch.inference_mode():
            generated = model.generate(**moved, do_sample=False, use_cache=True,
                cache_implementation="dynamic", max_new_tokens=96,
                return_dict_in_generate=True, output_logits=False)
        ids = generated.sequences[0].detach().cpu().tolist()
        prompt_ids = batch["input_ids"][0].tolist()
        if ids[:len(prompt_ids)] != prompt_ids or len(ids) <= len(prompt_ids):
            raise ValueError("generation_prefix_contract_failed")
        continuation = ids[len(prompt_ids):]
        text = processor.tokenizer.decode(continuation, skip_special_tokens=True).strip()
        trace = validate_dynamic_shape_trace(shape_observations,
            layer_indices, prompt_length)
        audit = validate_generation_mask_audit(mask_observations, trace,
            layer_indices, expected_applied=bool(positions))
        return text, {"generated_token_count": len(continuation),
            "generation_shape_trace": trace,
            "intervention_audit": audit}
    finally:
        for handle in reversed(shape_handles):
            handle.remove()
        remove_hooks(intervention_handles)
        if "generated" in locals():
            del generated
        if "moved" in locals():
            del moved
        torch.cuda.empty_cache()


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--minerva-json", type=Path, default=Path("/tmp/minerva_ego_20260205.json"))
    parser.add_argument("--time-map", type=Path, required=True)
    parser.add_argument("--assoc-info", type=Path, default=Path("/tmp/hd_epic_assoc_info.json"))
    parser.add_argument("--mask-info", type=Path, default=Path("/tmp/hd_epic_mask_info.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-key", default=ROW_KEY,
        help="exact private Minerva-Ego row key selected before model outcomes")
    parser.add_argument("--expected-video-md5", default=VIDEO_MD5,
        help="official checksum for the selected HD-EPIC MP4")
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    row = next(r for r in json.loads(args.minerva_json.read_text())
               if r.get("key") == args.row_key)
    video_id = row.get("video_id")
    if args.video.stem.removesuffix("_verified") != video_id:
        raise ValueError("video_row_binding_mismatch")
    md5 = hashlib.md5()
    with args.video.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            md5.update(block)
    video_md5 = md5.hexdigest()
    if video_md5 != args.expected_video_md5:
        raise ValueError("video_checksum_mismatch")
    assoc = json.loads(args.assoc_info.read_text())[video_id]
    masks = json.loads(args.mask_info.read_text())[video_id]
    if len(row.get("referenced_object_masks", [])) != 1:
        raise ValueError("single_reference_contract_failed")
    ref = row["referenced_object_masks"][0]
    name, oid = ref.rsplit("-", 1)
    obj = assoc[oid]
    if name.casefold() != obj["name"].casefold():
        raise ValueError("target_display_name_mismatch")
    object_names = sorted({name, name.split()[-1]})
    tracks = [track["time_segment"] for track in obj["tracks"]]
    mask_frames = [masks[mask_id]["frame_number"]
                   for track in obj["tracks"] for mask_id in track["masks"]]
    with args.time_map.open(newline="") as stream:
        import csv
        frame_times = [float(row["mp4_time_ns"]) / 1e9
                       for row in csv.DictReader(stream)]
    frames, sampled = decode_uniform_video(args.video, frame_times)
    from minerva_ego_anchor_utils import (
        evidence_temporal_groups, matched_control_groups,
        score_generated_movement_anchors, track_overlapping_temporal_groups)
    mapped = evidence_temporal_groups(mask_frame_numbers=mask_frames,
        mp4_frame_times_seconds=frame_times, sampled_frames=sampled,
        temporal_patch_size=2, temporal_step_count=FRAME_COUNT // 2)
    evidence_groups = mapped["evidence_temporal_groups"]
    excluded = track_overlapping_temporal_groups(
        sampled_times_seconds=[item["time_seconds"] for item in sampled],
        temporal_patch_size=2, track_segments=tracks)
    control_groups = matched_control_groups(temporal_step_count=FRAME_COUNT // 2,
        evidence_groups=evidence_groups, excluded_groups=excluded,
        seed="minerva-eg-anchor-pilot-v1", record_token=args.row_key)

    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
        / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        raise ValueError("pinned_checkpoint_missing")
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("cuda_bf16_unavailable")
    free, total = torch.cuda.mem_get_info()
    if free < 12 * 1024**3:
        raise ValueError("free_vram_below_12_gib")
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=128 * 28 * 28, max_pixels=128 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    prompt = build_grounding_prompt(row, name)
    messages = [{"role":"user","content":[{"type":"video","video":frames,
        "sample_fps":(FRAME_COUNT-1)/sampled[-1]["time_seconds"]},
        {"type":"text","text":prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or len(videos) != 1:
        raise ValueError("qwen_video_utility_contract_invalid")
    pixel_values, metadata = videos[0]
    if metadata.get("frames_indices") != list(range(FRAME_COUNT)):
        raise ValueError("processor_frame_indices_changed")
    batch = processor(text=[rendered], images=images, videos=[pixel_values],
        video_metadata=[metadata], do_sample_frames=False, cap_pixels_per_frame=True,
        padding=True, return_tensors="pt")
    groups, grid = _video_groups(batch, processor)
    conditions = (("baseline", ()), ("object_time_support_bias", evidence_groups),
                  ("equal_dose_temporal_sham_bias", control_groups))
    torch.manual_seed(20261022); torch.cuda.manual_seed_all(20261022)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    records = {}
    for label, selected_groups in conditions:
        positions = tuple(position for group in selected_groups for position in groups[group])
        generated_text, audit = _generate(name=label, positions=positions,
            model=model, processor=processor, batch=batch,
            prompt_length=int(batch["input_ids"].shape[1]), torch=torch)
        records[label] = {"generated_text": generated_text,
            "anchor_metrics": score_generated_movement_anchors(
                text=generated_text, track_segments=tracks,
                object_names=object_names),
            "generation_audit": audit}
    payload = {
        "schema_version": 1,
        "status": "completed_single_video_generated_grounding_exploratory_pilot",
        "direct_generated_explanation_faithfulness": False,
        "human_review_performed": False,
        "rights_status": "unverified; local-only per user authorization",
        "dataset": "Minerva-Ego / HD-EPIC", "model_revision": MODEL_REVISION,
        "video_id": video_id, "video_md5": video_md5,
        "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "row_key_sha256": sha(args.row_key.encode()), "requested_rows": 1,
        "effective_rows": 1, "generation_calls": 3, "frame_count": FRAME_COUNT,
        "processor_video_grid_thw": grid, "evidence_groups": evidence_groups,
        "control_groups": control_groups, "track_overlapping_groups_excluded": excluded,
        "mapped_mask_frames": mapped["mapped_mask_frames"],
        "generation": records, "biased_layers_1_based": list(LAYERS_1_BASED),
        "bias_nats": BIAS_NATS, "elapsed_seconds": time.monotonic()-started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "visible_total_vram_bytes": total,
        "metric_definition": "time anchors in sentences naming the annotation object label or final-word alias; precision is aligned anchors / all parsed anchors; segment recall is fraction of annotated movement intervals with any overlapping named-object anchor",
        "limitations": ["n=1 and same episode across conditions", "lexical/time alignment only; action semantics not evaluated",
            "no independent human review", "not a generated-explanation faithfulness result"]}
    private = ROOT / "data" / "private_minerva"
    if (args.output.parent.resolve() != private.resolve()
            or not private.is_dir() or private.stat().st_mode & 0o077
            or args.output.exists()):
        raise ValueError("private_output_contract_failed")
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({"status":payload["status"],"generation_calls":3,
        "evidence_groups":evidence_groups,"control_groups":control_groups,
        "grounding_metrics":{k:v["anchor_metrics"] for k,v in records.items()},
        "elapsed_seconds":payload["elapsed_seconds"],"peak_vram_bytes":payload["peak_vram_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
