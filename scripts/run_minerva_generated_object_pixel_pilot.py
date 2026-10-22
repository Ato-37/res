#!/usr/bin/env python3
"""CPU-gated generated-claim object-pixel and visual-key interaction pilot.

The script is local-only. CPU mode builds and fingerprints the exact Qwen
batches; GPU mode requires that receipt, generates one baseline sentence, then
scores its frozen generated token IDs under six prespecified branches.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import time

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"
FRAME_COUNT = 64
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
LAYERS_1_BASED = tuple(range(5, 17))
LOGIT_BIAS_NATS = math.log(2.0)
CPU_STATUS = "completed_minerva_generated_object_pixel_cpu_preflight"
PROMPT_PROTOCOL = "visible_object_observation_v2_exploratory"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value, torch) -> str:
    if not torch.is_tensor(value):
        raise ValueError("tensor_hash_input_invalid")
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return _sha(raw)


def _private_file(path: Path, *, max_bytes: int = 50_000_000) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > max_bytes):
            raise ValueError("private_input_contract_invalid")
        data = bytearray()
        while True:
            chunk = os.read(fd, min(1024 * 1024, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ValueError("private_input_too_large")
        if len(data) != info.st_size:
            raise ValueError("private_input_changed_while_reading")
        return bytes(data)
    finally:
        os.close(fd)


def _validate_private_root() -> None:
    if not PRIVATE.is_dir() or PRIVATE.is_symlink() or PRIVATE.stat().st_mode & 0o077:
        raise ValueError("private_root_contract_invalid")


def _validate_private_path(path: Path, *, must_exist: bool = True) -> None:
    _validate_private_root()
    if path.is_symlink():
        raise ValueError("private_path_symlink_rejected")
    if must_exist and not path.is_file():
        raise ValueError("private_input_missing")
    resolved = path.resolve(strict=must_exist)
    if not resolved.is_relative_to(PRIVATE.resolve()):
        raise ValueError("private_path_outside_root")


def _validate_local_input(path: Path) -> None:
    """Allow read-only annotation/manifest sources outside the private output root."""
    if path.is_symlink() or not path.is_file() or not path.resolve().is_file():
        raise ValueError("local_input_file_contract_invalid")


def _write_private_json(path: Path, payload: dict) -> None:
    _validate_private_path(path, must_exist=False)
    if path.parent.resolve() != PRIVATE.resolve():
        raise ValueError("private_output_must_be_direct_child")
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def compute_pixel_bias_interaction(branch_scores: dict[str, float]) -> dict:
    """Compute the frozen sham-minus-support differences and key-bias DID."""
    expected = {f"{pixel}_{key}" for pixel in ("support", "sham")
                for key in ("none", "bbox_bias", "spatial_sham_bias")}
    if (not isinstance(branch_scores, dict) or set(branch_scores) != expected
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(float(v)) for v in branch_scores.values())):
        raise ValueError("branch_score_set_invalid")
    d = {key: float(branch_scores[f"sham_{key}"])
              - float(branch_scores[f"support_{key}"])
         for key in ("none", "bbox_bias", "spatial_sham_bias")}
    return {
        "D_no_bias_nats_per_token": d["none"],
        "D_bbox_bias_nats_per_token": d["bbox_bias"],
        "D_spatial_sham_bias_nats_per_token": d["spatial_sham_bias"],
        "primary_bbox_minus_spatial_sham_interaction_nats_per_token":
            d["bbox_bias"] - d["spatial_sham_bias"],
        "secondary_bbox_minus_no_bias_interaction_nats_per_token":
            d["bbox_bias"] - d["none"],
        "D_definition": "meanLP(sham-filled frame) - meanLP(annotation-bbox-filled frame)",
        "positive_primary_interaction": "larger object-pixel selectivity under bbox-key bias than under equal-dose spatial-sham-key bias",
    }


def _prompt(row: dict, object_name: str, anchor_clip_seconds: float) -> str:
    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError("target_object_name_invalid")
    if (isinstance(anchor_clip_seconds, bool)
            or not isinstance(anchor_clip_seconds, (int, float))
            or not math.isfinite(float(anchor_clip_seconds))
            or not 0 <= anchor_clip_seconds < FRAME_COUNT / 2):
        raise ValueError("prompt_anchor_clip_time_invalid")
    choices = "\n".join(f"{chr(65+i)}. {row[f'answer_choice_{i}']}" for i in range(5))
    return (row["question"] + "\nOptions:\n" + choices +
            f"\nAt approximately {anchor_clip_seconds:.1f} seconds in this clip, "
            f"describe exactly one short, directly visible observation of the "
            f"{object_name} in one sentence that helps answer the question. "
            f"Name the {object_name} in that sentence. Do not state the "
            "multiple-choice answer or infer an unseen cause. "
            "If the object is not clearly visible at that time, say Unclear.")


def _generated_target(text: str, token_ids: list[int], object_name: str) -> None:
    if not isinstance(text, str) or not text.strip() or not token_ids:
        raise ValueError("generated_target_empty")
    if text.strip().casefold() in ("unclear", "unclear."):
        raise ValueError("generated_target_unclear")
    if len(token_ids) > 64 or "\n" in text.strip():
        raise ValueError("generated_target_not_short_single_line")
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]
    if len(sentences) > 1:
        raise ValueError("generated_target_multiple_sentences")
    alias = object_name.split()[-1]
    if not re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE):
        raise ValueError("generated_target_missing_object_name")


def _record_for_row(receipt: dict, *, cohort_sha256: str, row_key_sha256: str,
                    video_id: str) -> dict:
    if (not isinstance(receipt, dict)
            or receipt.get("status") != "completed_frozen_cohort_cpu_preflight"
            or receipt.get("cohort_sha256") != cohort_sha256):
        raise ValueError("cohort_preflight_binding_mismatch")
    rows = [record for record in receipt.get("records", [])
            if record.get("video_id") == video_id
            and record.get("row_key_sha256") == row_key_sha256]
    if len(rows) != 1 or rows[0].get("status") != "pass":
        raise ValueError("cohort_preflight_row_not_unique_and_passing")
    return rows[0]


def _validate_proxy_anchor_binding(cohort: dict, selected: dict,
                                   actual_anchor: dict) -> None:
    """Require a v2 proxy-selected mask to survive the official-PTS gate."""
    if cohort.get("candidate_preflight_status") != "metadata_only_proxy_not_preflighted":
        return
    proxy = selected.get("proxy_anchor")
    interval = proxy.get("interval_seconds") if isinstance(proxy, dict) else None
    margin = cohort.get("rationale_interval_boundary_margin_seconds")
    if (not isinstance(proxy, dict) or type(proxy.get("frame_number")) is not int
            or not isinstance(interval, list) or len(interval) != 2
            or any(type(x) not in (int, float) for x in interval)
            or any(not math.isfinite(x) for x in interval)
            or interval[1] <= interval[0]
            or type(margin) not in (int, float) or not 0 <= margin < 0.5
            or actual_anchor.get("mask_frame_number") != proxy["frame_number"]
            or actual_anchor.get("timestamp_support_interval_seconds") != interval):
        raise ValueError("proxy_anchor_differs_from_official_pts_anchor")
    actual_pts = actual_anchor.get("mask_time_seconds")
    if (type(actual_pts) not in (int, float) or not math.isfinite(actual_pts)
            or not interval[0] + margin <= actual_pts < interval[1] - margin):
        raise ValueError("proxy_anchor_official_pts_outside_safe_interval")


def _prepare(*, cohort_path: Path, cohort_preflight_path: Path, row_index: int,
             video_path: Path, time_map_path: Path, minerva_json_path: Path,
             assoc_info_path: Path, mask_info_path: Path, manifest_path: Path):
    """Bind cohort, official source files and preflight before building batches."""
    from download_frozen_hd_epic_videos import EXPECTED_MANIFEST_SHA256
    from minerva_object_pixel_utils import (
        clip_bbox_xyxy, mean_fill_box, select_spatial_sham_box)
    from minerva_ego_anchor_utils import (
        bbox_to_video_visual_key_positions, select_rationale_mask_anchor,
        build_anchor_centered_sample_indices, include_exact_anchor_sample)
    from preflight_minerva_grounding_cohort import (
        digest_file, load_time_map_seconds, normalize_manifest_path, parse_manifest)
    from run_minerva_ego_anchor_gpu_pilot import (
        FRAME_COUNT as RUNNER_FRAME_COUNT, MODEL_REVISION as PINNED_REVISION,
        _video_groups, decode_uniform_video)
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    if RUNNER_FRAME_COUNT != FRAME_COUNT or PINNED_REVISION != MODEL_REVISION:
        raise ValueError("minerva_runner_model_contract_changed")
    for path in (cohort_path, cohort_preflight_path, video_path, time_map_path):
        _validate_private_path(path)
    for path in (minerva_json_path, assoc_info_path, mask_info_path, manifest_path):
        _validate_local_input(path)
    cohort_raw = _private_file(cohort_path)
    preflight_raw = _private_file(cohort_preflight_path)
    cohort = json.loads(cohort_raw)
    selection = cohort.get("selection")
    if (cohort.get("status") != "frozen_pre_outcome_exploratory_cohort"
            or not isinstance(selection, list) or type(row_index) is not int
            or not 0 <= row_index < len(selection)):
        raise ValueError("frozen_cohort_contract_invalid")
    selected_video_ids = [item.get("video_id") for item in selection]
    selected_row_hashes = [item.get("row_key_sha256") for item in selection]
    if (len(set(selected_video_ids)) != len(selection)
            or len(set(selected_row_hashes)) != len(selection)):
        raise ValueError("cohort_must_have_one_row_per_video")
    selected = selection[row_index]
    row_key, video_id = selected.get("row_key"), selected.get("video_id")
    if (not isinstance(row_key, str) or not row_key
            or selected.get("row_key_sha256") != _sha(row_key.encode())
            or video_path.stem.removesuffix("_verified") != video_id):
        raise ValueError("cohort_selected_row_binding_invalid")
    preflight = json.loads(preflight_raw)
    record = _record_for_row(preflight, cohort_sha256=_sha(cohort_raw),
        row_key_sha256=_sha(row_key.encode()), video_id=video_id)

    source_paths = {"minerva_json": minerva_json_path,
                    "assoc_info": assoc_info_path, "mask_info": mask_info_path}
    source_hashes = {name: _sha_file(path) for name, path in source_paths.items()}
    if cohort.get("source_sha256") != source_hashes:
        raise ValueError("cohort_annotation_source_hash_mismatch")
    rows = json.loads(minerva_json_path.read_text(encoding="utf-8"))
    matches = [row for row in rows if row.get("key") == row_key]
    if len(matches) != 1 or matches[0].get("video_id") != video_id:
        raise ValueError("cohort_row_source_join_invalid")
    row = matches[0]
    references = row.get("referenced_object_masks") or []
    if len(references) != 1:
        raise ValueError("single_object_reference_required")
    try:
        display_name, object_id = references[0].rsplit("-", 1)
    except (AttributeError, ValueError):
        raise ValueError("object_reference_invalid") from None
    assoc = json.loads(assoc_info_path.read_text(encoding="utf-8"))[video_id]
    masks = json.loads(mask_info_path.read_text(encoding="utf-8"))[video_id]
    obj = assoc.get(object_id)
    if (not isinstance(obj, dict)
            or display_name.casefold() != str(obj.get("name", "")).casefold()):
        raise ValueError("target_display_name_mismatch")
    tracks = [item["time_segment"] for item in obj.get("tracks", [])]
    mask_ids = [mask_id for track in obj.get("tracks", [])
                for mask_id in track.get("masks", [])]
    if not tracks or not mask_ids or any(mask_id not in masks for mask_id in mask_ids):
        raise ValueError("object_track_mask_join_invalid")

    manifest, manifest_sha256 = parse_manifest(manifest_path)
    if (manifest_sha256 != EXPECTED_MANIFEST_SHA256
            or preflight.get("manifest_sha256") != manifest_sha256):
        raise ValueError("official_manifest_binding_invalid")
    if preflight.get("preflight_source_sha256") != _sha_file(
            Path(__file__).with_name("preflight_minerva_grounding_cohort.py")):
        raise ValueError("cohort_preflight_source_changed")
    relative_video = f"Videos/{video_id[:3]}/{video_id}.mp4"
    relative_map = f"Videos/{video_id[:3]}/{video_id}_mp4_to_vrs_time_ns.csv"
    normalized_video = normalize_manifest_path(relative_video)
    normalized_map = normalize_manifest_path(relative_map)
    video_md5, time_map_md5 = digest_file(video_path, "md5"), digest_file(time_map_path, "md5")
    if (manifest.get(normalized_video) != video_md5
            or manifest.get(normalized_map) != time_map_md5):
        raise ValueError("official_media_md5_mismatch")
    video_sha256, time_map_sha256 = _sha_file(video_path), _sha_file(time_map_path)
    if (record.get("video_md5") != video_md5
            or record.get("video_sha256") != video_sha256
            or record.get("time_map_md5") != time_map_md5
            or record.get("time_map_sha256") != time_map_sha256):
        raise ValueError("media_hash_differs_from_cohort_preflight")

    frame_times = load_time_map_seconds(time_map_path)
    if len(frame_times) != record.get("frame_count") or len(frame_times) != record.get("time_map_rows"):
        raise ValueError("time_map_frame_count_mismatch")
    anchor = select_rationale_mask_anchor(reasoning=row.get("reasoning", ""),
        object_id=object_id, tracks=obj["tracks"],
        mask_frame_numbers={mask_id: masks[mask_id]["frame_number"] for mask_id in mask_ids},
        mp4_frame_times_seconds=frame_times)
    _validate_proxy_anchor_binding(cohort, selected, anchor)
    saved_anchor = record.get("rationale_mask_anchor", {})
    anchor_frame = anchor["mask_frame_number"]
    if saved_anchor.get("mask_frame_number") != anchor_frame:
        raise ValueError("rationale_anchor_differs_from_cpu_preflight")
    sampling = record.get("clip_sampling", {})
    sample_count = min(FRAME_COUNT,
        math.floor((frame_times[-1] - frame_times[0]) * 2.0) + 1)
    sample_count -= sample_count % 2
    if sample_count != FRAME_COUNT:
        raise ValueError("frozen_clip_does_not_support_64_frames")
    expected_sampling = build_anchor_centered_sample_indices(
        mp4_frame_times_seconds=frame_times, anchor_frame_number=anchor_frame,
        sample_count=sample_count, sample_fps=2.0)
    expected_sampling.update(include_exact_anchor_sample(
        sample_frame_indices=expected_sampling["sample_frame_indices"],
        mp4_frame_times_seconds=frame_times, anchor_frame_number=anchor_frame))
    expected_sampling["maximum_grid_time_error_seconds"] = max(
        abs(frame_times[source_index]
            - frame_times[expected_sampling["sample_frame_indices"][0]] - rank / 2.0)
        for rank, source_index in enumerate(expected_sampling["sample_frame_indices"]))
    if any(sampling.get(key) != expected_sampling.get(key) for key in (
            "sample_frame_indices", "sample_fps", "clip_start_source_seconds",
            "clip_end_source_seconds", "anchor_sample_rank",
            "maximum_grid_time_error_seconds")):
        raise ValueError("clip_sampling_differs_from_frozen_preflight")
    grid_error = sampling.get("maximum_grid_time_error_seconds")
    if (isinstance(grid_error, bool) or not isinstance(grid_error, (int, float))
            or not math.isfinite(float(grid_error)) or grid_error > 0.25):
        raise ValueError("sampling_grid_time_error_above_0_25s")
    sample_indices = sampling.get("sample_frame_indices")
    anchor_rank = sampling.get("anchor_sample_rank")
    if (record.get("sampled_frame_count") != FRAME_COUNT
            or not isinstance(sample_indices, list) or len(sample_indices) != FRAME_COUNT
            or type(anchor_rank) is not int or not 0 <= anchor_rank < FRAME_COUNT):
        raise ValueError("preflight_sampling_contract_invalid")
    frames, sampled = decode_uniform_video(video_path, frame_times,
        sample_frame_indices=sample_indices, expected_sample_count=FRAME_COUNT)
    if (sampled[anchor_rank]["decoded_frame_index"] != anchor_frame
            or abs(sampled[anchor_rank]["time_seconds"] - frame_times[anchor_frame]) > 0.001):
        raise ValueError("anchor_frame_pts_binding_failed")
    frame = frames[anchor_rank]
    anchor_mask_ids = [mask_id for mask_id in mask_ids
                       if masks[mask_id].get("frame_number") == anchor_frame]
    if len(anchor_mask_ids) != 1:
        raise ValueError("anchor_frame_mask_id_not_unique")
    anchor_mask_id = anchor_mask_ids[0]
    clipped_support = clip_bbox_xyxy(masks[anchor_mask_id].get("bbox"),
        frame.width, frame.height, max_clipped_fraction=0.5)
    support_box = clipped_support["box_xyxy"]
    forbidden_boxes = []
    for mask in masks.values():
        if mask.get("frame_number") != anchor_frame:
            continue
        clipped = clip_bbox_xyxy(mask.get("bbox"), frame.width, frame.height,
                                 max_clipped_fraction=0.5)
        forbidden_boxes.append(clipped["box_xyxy"])

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
        / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        raise ValueError("pinned_checkpoint_missing")
    pixel_cap = 128 * 28 * 28
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=pixel_cap, max_pixels=pixel_cap)
    if int(processor.video_processor.temporal_patch_size) != 2:
        raise ValueError("qwen_temporal_patch_size_mismatch")
    anchor_clip_seconds = anchor_rank / sampling["sample_fps"]
    prompt = _prompt(row, display_name, anchor_clip_seconds)
    rendered, cpu_batches = None, {}
    conditions = {"base": frames,
                  "support_fill": [item.copy() for item in frames]}
    conditions["support_fill"][anchor_rank] = mean_fill_box(frame, support_box)

    video_token_id = int(processor.video_token_id)
    messages_base = [{"role": "user", "content": [
        {"type": "video", "video": frames, "sample_fps": sampling["sample_fps"]},
        {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages_base, tokenize=False,
        add_generation_prompt=True)
    images, videos = process_vision_info(messages_base, return_video_metadata=True)
    if images is not None or not isinstance(videos, list) or len(videos) != 1:
        raise ValueError("qwen_video_utility_contract_invalid")
    _base_pixels, metadata = videos[0]
    if metadata.get("frames_indices") != list(range(FRAME_COUNT)):
        raise ValueError("processor_frame_indices_changed")
    sham_box = None
    pair_start = (anchor_rank // 2) * 2
    import numpy as np
    pair_tensor = torch.from_numpy(np.stack([
        np.asarray(item.convert("RGB")) for item in frames[pair_start:pair_start + 2]]))
    pair_tensor = pair_tensor.permute(0, 3, 1, 2).contiguous()
    pair_features = processor.video_processor([pair_tensor], return_tensors="pt",
                                               cap_pixels_per_frame=True)
    pair_grid = [int(value) for value in pair_features["video_grid_thw"][0].tolist()]
    if pair_grid[0] != 1:
        raise ValueError("anchor_pair_video_grid_invalid")
    merged_cells = (pair_grid[1] // int(processor.image_processor.merge_size)
                    * pair_grid[2] // int(processor.image_processor.merge_size))
    dummy_groups = (tuple(range(merged_cells)),) * (FRAME_COUNT // 2)
    support_mapping_geometry = bbox_to_video_visual_key_positions(
        bbox_xyxy=support_box, frame_rank=anchor_rank, frame_count=FRAME_COUNT,
        temporal_patch_size=2, visual_token_groups=dummy_groups,
        source_width=frame.width, source_height=frame.height,
        grid_height=pair_grid[1], grid_width=pair_grid[2],
        patch_size=int(processor.video_processor.patch_size),
        spatial_merge_size=int(processor.image_processor.merge_size))
    support_offsets = set(support_mapping_geometry["patch_cell_offsets"])
    sham_mapping_geometry = None
    for retry in range(256):
        candidate = select_spatial_sham_box(support_box, frame.width, frame.height,
            forbidden_boxes, seed=(f"minerva-generated-object-pixel-v1-"
                f"{_sha(row_key.encode())}-{retry}"))
        candidate_mapping = bbox_to_video_visual_key_positions(
            bbox_xyxy=candidate, frame_rank=anchor_rank, frame_count=FRAME_COUNT,
            temporal_patch_size=2, visual_token_groups=dummy_groups,
            source_width=frame.width, source_height=frame.height,
            grid_height=pair_grid[1], grid_width=pair_grid[2],
            patch_size=int(processor.video_processor.patch_size),
            spatial_merge_size=int(processor.image_processor.merge_size))
        if (len(candidate_mapping["visual_key_positions"])
                == len(support_mapping_geometry["visual_key_positions"])
                and not (set(candidate_mapping["patch_cell_offsets"]) & support_offsets)):
            sham_box, sham_mapping_geometry = candidate, candidate_mapping
            break
    if sham_box is None:
        raise ValueError("equal_pixel_sham_key_dose_unavailable")
    conditions["sham_fill"] = [item.copy() for item in frames]
    conditions["sham_fill"][anchor_rank] = mean_fill_box(frame, sham_box)

    for condition, condition_frames in conditions.items():
        messages = [{"role": "user", "content": [
            {"type": "video", "video": condition_frames,
             "sample_fps": sampling["sample_fps"]},
            {"type": "text", "text": prompt}]}]
        rendered_condition = processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True)
        if rendered_condition != rendered:
            raise ValueError("pixel_intervention_changed_prompt")
        condition_images, condition_videos = process_vision_info(messages,
            return_video_metadata=True)
        if condition_images is not None or len(condition_videos) != 1:
            raise ValueError("qwen_video_utility_contract_invalid")
        pixels, condition_metadata = condition_videos[0]
        if condition_metadata.get("frames_indices") != list(range(FRAME_COUNT)):
            raise ValueError("processor_frame_indices_changed")
        batch = processor(text=[rendered], images=condition_images,
            videos=[pixels], video_metadata=[condition_metadata],
            do_sample_frames=False, cap_pixels_per_frame=True, padding=True,
            return_tensors="pt")
        cpu_batches[condition] = batch
    base = cpu_batches["base"]
    grid = [int(x) for x in base["video_grid_thw"][0].tolist()]
    groups, _ = _video_groups(base, processor, frame_count=FRAME_COUNT)
    support_mapping = _map_box(support_box, frames=frames,
        anchor_rank=anchor_rank, batch=(base, groups), processor=processor,
        frame_count=FRAME_COUNT)
    sham_mapping = _map_box(sham_box, frames=frames,
        anchor_rank=anchor_rank, batch=(base, groups), processor=processor,
        frame_count=FRAME_COUNT)
    support_positions = tuple(support_mapping["visual_key_positions"])
    sham_positions = tuple(sham_mapping["visual_key_positions"])
    if (not support_positions or len(support_positions) != len(sham_positions)
            or set(support_positions) & set(sham_positions)):
        raise ValueError("support_sham_key_geometry_invalid")
    for condition, batch in cpu_batches.items():
        if [int(x) for x in batch["video_grid_thw"][0].tolist()] != grid:
            raise ValueError("pixel_condition_grid_changed")
        for key, base_value in base.items():
            value = batch[key]
            if key == "pixel_values_videos":
                continue
            if torch.is_tensor(base_value) and (not torch.is_tensor(value)
                    or not torch.equal(base_value, value)):
                raise ValueError("pixel_mask_changed_nonvisual_input")
    tensor_hashes = {name: _tensor_sha256(batch["pixel_values_videos"], torch)
                     for name, batch in cpu_batches.items()}
    if len(set(tensor_hashes.values())) != 3:
        raise ValueError("pixel_intervention_tensor_not_distinct")
    ids_hashes = {name: _sha(json.dumps(batch["input_ids"][0].tolist(),
        separators=(",", ":")).encode("ascii")) for name, batch in cpu_batches.items()}
    if len(set(ids_hashes.values())) != 1:
        raise ValueError("pixel_intervention_input_ids_changed")
    return {
        "cohort_raw": cohort_raw, "cohort": cohort,
        "row_index": row_index,
        "cohort_preflight_raw": preflight_raw, "record": record,
        "row": row, "row_key": row_key, "video_id": video_id,
        "object_name": display_name, "object_id": object_id,
        "video_sha256": video_sha256, "video_md5": video_md5,
        "time_map_sha256": time_map_sha256, "time_map_md5": time_map_md5,
        "annotation_source_sha256": source_hashes,
        "manifest_sha256": manifest_sha256,
        "frames": frames, "sampled": sampled, "frame_times": frame_times,
        "anchor_frame": anchor_frame, "anchor_rank": anchor_rank,
        "anchor_clip_seconds": anchor_clip_seconds,
        "anchor_mask_id_sha256": _sha(anchor_mask_id.encode()),
        "anchor_pts_seconds": sampled[anchor_rank]["time_seconds"],
        "support_box": support_box, "support_clipping": clipped_support,
        "sham_box": sham_box, "support_mapping": support_mapping,
        "sham_mapping": sham_mapping,
        "support_positions": support_positions, "sham_positions": sham_positions,
        "processor": processor, "rendered_prompt": rendered,
        "rendered_prompt_sha256": _sha(rendered.encode("utf-8")),
        "batches": cpu_batches, "video_grid_thw": grid,
        "pixel_tensor_sha256": tensor_hashes,
        "input_ids_sha256": ids_hashes,
        "key_dose": {"bbox": len(support_positions), "spatial_sham": len(sham_positions)},
        "source_sha256": {
            "run_minerva_generated_object_pixel_pilot.py": _sha_file(Path(__file__)),
            "minerva_object_pixel_utils.py": _sha_file(Path(__file__).with_name("minerva_object_pixel_utils.py")),
            "minerva_ego_anchor_utils.py": _sha_file(Path(__file__).with_name("minerva_ego_anchor_utils.py")),
            "preflight_minerva_grounding_cohort.py": _sha_file(Path(__file__).with_name("preflight_minerva_grounding_cohort.py")),
            "run_minerva_ego_anchor_gpu_pilot.py": _sha_file(Path(__file__).with_name("run_minerva_ego_anchor_gpu_pilot.py")),
            "run_minerva_ego_generated_grounding_pilot.py": _sha_file(Path(__file__).with_name("run_minerva_ego_generated_grounding_pilot.py")),
            "run_claim_support_qta_16frame_exploratory.py": _sha_file(Path(__file__).with_name("run_claim_support_qta_16frame_exploratory.py")),
            "qta_multiquery_video_key_intervention.py": _sha_file(Path(__file__).with_name("qta_multiquery_video_key_intervention.py")),
        },
        "runtime_versions": {"python": platform.python_version(),
            **{name: importlib.metadata.version(package) for name, package in
               (("torch", "torch"), ("transformers", "transformers"),
                ("qwen_vl_utils", "qwen-vl-utils"), ("Pillow", "Pillow"),
                ("numpy", "numpy"), ("av", "av"))}},
        "_torch": torch,
    }


def _map_box(box, *, frames, anchor_rank, batch, processor, frame_count):
    from minerva_ego_anchor_utils import bbox_to_video_visual_key_positions
    batch_cpu, groups = batch
    grid = [int(x) for x in batch_cpu["video_grid_thw"][0].tolist()]
    return bbox_to_video_visual_key_positions(bbox_xyxy=box,
        frame_rank=anchor_rank, frame_count=frame_count,
        temporal_patch_size=2, visual_token_groups=groups,
        source_width=frames[anchor_rank].width,
        source_height=frames[anchor_rank].height, grid_height=grid[1],
        grid_width=grid[2], patch_size=int(processor.video_processor.patch_size),
        spatial_merge_size=int(processor.image_processor.merge_size))


def _cpu_receipt(prepared: dict, row_index: int) -> dict:
    return {
        "schema_version": 1,
        "status": CPU_STATUS,
        "prompt_protocol": PROMPT_PROTOCOL,
        "rights_status": "unverified; local-only per user authorization",
        "human_review_performed": False,
        "direct_generated_explanation_faithfulness": False,
        "cohort_sha256": _sha(prepared["cohort_raw"]),
        "cohort_cpu_preflight_sha256": _sha(prepared["cohort_preflight_raw"]),
        "row_index": row_index,
        "video_id_sha256": _sha(prepared["video_id"].encode()),
        "row_key_sha256": _sha(prepared["row_key"].encode()),
        "video_md5": prepared["video_md5"],
        "video_sha256": prepared["video_sha256"],
        "time_map_md5": prepared["time_map_md5"],
        "time_map_sha256": prepared["time_map_sha256"],
        "manifest_sha256": prepared["manifest_sha256"],
        "annotation_source_sha256": prepared["annotation_source_sha256"],
        "anchor_frame_number": prepared["anchor_frame"],
        "anchor_sample_rank": prepared["anchor_rank"],
        "prompt_anchor_clip_seconds": prepared["anchor_clip_seconds"],
        "anchor_pts_seconds": prepared["anchor_pts_seconds"],
        "anchor_mask_id_sha256": prepared["anchor_mask_id_sha256"],
        "source_bbox_integer_xyxy": list(prepared["support_box"]),
        "source_bbox_clipped_area_fraction": prepared["support_clipping"]["clipped_area_fraction"],
        "spatial_sham_integer_xyxy": list(prepared["sham_box"]),
        "temporal_pair_sample_ranks": [prepared["anchor_rank"] // 2 * 2,
                                       prepared["anchor_rank"] // 2 * 2 + 1],
        "temporal_pair_source_frame_numbers": [
            prepared["sampled"][prepared["anchor_rank"] // 2 * 2]["decoded_frame_index"],
            prepared["sampled"][prepared["anchor_rank"] // 2 * 2 + 1]["decoded_frame_index"]],
        "temporal_pair_pts_seconds": [
            prepared["sampled"][prepared["anchor_rank"] // 2 * 2]["time_seconds"],
            prepared["sampled"][prepared["anchor_rank"] // 2 * 2 + 1]["time_seconds"]],
        "processor_video_grid_thw": prepared["video_grid_thw"],
        "rendered_prompt_sha256": prepared["rendered_prompt_sha256"],
        "input_ids_sha256": prepared["input_ids_sha256"],
        "pixel_tensor_sha256": prepared["pixel_tensor_sha256"],
        "condition_key_counts": prepared["key_dose"],
        "bbox_key_positions_sha256": _sha(json.dumps(
            prepared["support_positions"], separators=(",", ":")).encode("ascii")),
        "spatial_sham_key_positions_sha256": _sha(json.dumps(
            prepared["sham_positions"], separators=(",", ":")).encode("ascii")),
        "source_sha256": prepared["source_sha256"],
        "runner_sha256": _sha_file(Path(__file__)),
        "runtime_versions": prepared["runtime_versions"],
        "sampling": {key: prepared["record"]["clip_sampling"][key]
                     for key in ("sample_frame_indices", "sample_fps",
                                 "clip_start_source_seconds", "clip_end_source_seconds",
                                 "anchor_sample_rank", "maximum_grid_time_error_seconds")},
        "estimand": "one generated sentence's teacher-forced mean log probability under object-bbox versus equal-size same-frame spatial sham mean-fill, crossed with fixed bbox-key or equal-dose spatial-sham-key +ln(2) bias",
        "limitations": ["single-video exploratory pilot", "annotation provides bounding boxes, not segmentation masks", "Qwen temporal patch size 2 fuses anchor frame with its unchanged pair frame", "conditional likelihood sensitivity does not establish claim truth, semantic support, or explanation faithfulness"],
    }


def _validate_cpu_receipt(receipt: dict, *, prepared: dict, row_index: int,
                          receipt_raw: bytes) -> None:
    expected = _cpu_receipt(prepared, row_index)
    for key in expected:
        if receipt.get(key) != expected[key]:
            raise ValueError(f"cpu_pixel_receipt_mismatch:{key}")
    if receipt.get("status") != CPU_STATUS:
        raise ValueError("cpu_pixel_receipt_status_invalid")


def _score_and_generate(prepared: dict, *, torch, progress: dict):
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
        / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        raise ValueError("pinned_checkpoint_missing")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("cuda_bf16_unavailable")
    free, total = torch.cuda.mem_get_info()
    if free < 12 * 1024**3:
        raise ValueError("free_vram_below_12_gib")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    batch = prepared["batches"]["base"]
    device_batch = {key: value.to("cuda") if torch.is_tensor(value) else value
                    for key, value in batch.items()}
    torch.manual_seed(20261022)
    torch.cuda.manual_seed_all(20261022)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
        raise RuntimeError("generation_free_vram_guard_failed")
    progress["generation_attempted"] = True
    with torch.inference_mode():
        generated = model.generate(**device_batch, do_sample=False, num_beams=1,
            use_cache=True, cache_implementation="dynamic", max_new_tokens=64,
            return_dict_in_generate=True, output_logits=False)
    progress["generation_completed"] = True
    prompt_ids = batch["input_ids"][0].tolist()
    sequences = generated.sequences[0].detach().cpu().tolist()
    if sequences[:len(prompt_ids)] != prompt_ids:
        raise ValueError("generation_prefix_contract_failed")
    target_ids = sequences[len(prompt_ids):]
    eos = getattr(prepared["processor"].tokenizer, "eos_token_id", None)
    if eos is not None and target_ids and target_ids[-1] == eos:
        # The scored target is the exact generated continuation including the
        # terminal EOS token so the token sequence remains generation-faithful.
        pass
    text = prepared["processor"].tokenizer.decode(target_ids,
        skip_special_tokens=True).strip()
    token_hash = _sha(json.dumps(target_ids, separators=(",", ":")).encode("ascii"))
    progress["frozen_target_token_ids_sha256"] = token_hash
    # Retain a rejected generation only in the private failure receipt so a
    # structural attrition can be diagnosed without silently relaxing it.
    progress["generated_text_for_failure_diagnosis"] = text
    _generated_target(text, target_ids, prepared["object_name"])
    del generated, device_batch
    torch.cuda.empty_cache()

    from run_claim_support_qta_16frame_exploratory import _score_branch
    branches, details = {}, {}
    key_sets = {"none": (), "bbox_bias": prepared["support_positions"],
                "spatial_sham_bias": prepared["sham_positions"]}
    if (len(key_sets["bbox_bias"]) != len(key_sets["spatial_sham_bias"])
            or not key_sets["bbox_bias"]):
        raise ValueError("gpu_key_bias_dose_invalid")
    for fill in ("support", "sham"):
        batch_cpu = prepared["batches"]["support_fill" if fill == "support" else "sham_fill"]
        for key_condition in ("none", "bbox_bias", "spatial_sham_bias"):
            key = f"{fill}_{key_condition}"
            progress["teacher_forced_score_attempted"] += 1
            branches[key], details[key] = _score_branch(model=model,
                batch_cpu=batch_cpu, target_ids=tuple(target_ids),
                biased_positions=key_sets[key_condition], torch=torch,
                logit_bias=LOGIT_BIAS_NATS)
            progress["teacher_forced_score_completed"] += 1
    torch.cuda.synchronize()
    result = {"generated_text": text,
        "generated_token_ids": target_ids,
        "generated_token_ids_sha256": token_hash,
        "generated_text_sha256": _sha(text.encode("utf-8")),
        "target_token_count": len(target_ids),
        "branch_mean_logprob_nats_per_token": branches,
        "branch_audit": details,
        "contrasts": compute_pixel_bias_interaction(branches),
        "model_call_count": {"generation_calls": 1,
            "teacher_forced_score_calls": 6,
            "completed_teacher_forced_score_calls": len(branches)},
        "bias_nats": LOGIT_BIAS_NATS,
        "biased_layers_1_based": list(LAYERS_1_BASED),
        "elapsed_scoring_seconds": time.monotonic() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "visible_total_vram_bytes": total,
    }
    del model
    torch.cuda.empty_cache()
    return result


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-preflight", type=Path, required=True,
        help="existing official-media/PTS CPU receipt from preflight_minerva_grounding_cohort.py")
    parser.add_argument("--row-index", type=int, default=0,
        help="fixed zero-based row in the frozen cohort; default is its first row")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--time-map", type=Path, required=True)
    parser.add_argument("--minerva-json", type=Path,
        default=Path("/tmp/minerva_ego_20260205.json"))
    parser.add_argument("--assoc-info", type=Path,
        default=Path("/tmp/hd_epic_assoc_info.json"))
    parser.add_argument("--mask-info", type=Path,
        default=Path("/tmp/hd_epic_mask_info.json"))
    parser.add_argument("--manifest", type=Path,
        default=Path("/tmp/hd-epic-downloader/data/md5.txt"))
    parser.add_argument("--preflight-only", action="store_true",
        help="build exact CPU batches and write the required private receipt; no CUDA/model load")
    parser.add_argument("--cpu-receipt", type=Path,
        help="exact mode-0600 receipt emitted by --preflight-only; required in GPU mode")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    _validate_private_path(args.output, must_exist=False)
    if args.preflight_only and args.cpu_receipt is not None:
        raise ValueError("cpu_preflight_cannot_consume_receipt")
    if not args.preflight_only:
        if args.cpu_receipt is None:
            raise ValueError("gpu_mode_requires_cpu_receipt")
        _validate_private_path(args.cpu_receipt)
        if args.output.resolve() == args.cpu_receipt.resolve():
            raise ValueError("gpu_output_must_differ_from_cpu_receipt")
    prepared = _prepare(cohort_path=args.cohort,
        cohort_preflight_path=args.cohort_preflight, row_index=args.row_index,
        video_path=args.video, time_map_path=args.time_map,
        minerva_json_path=args.minerva_json, assoc_info_path=args.assoc_info,
        mask_info_path=args.mask_info, manifest_path=args.manifest)
    payload = _cpu_receipt(prepared, args.row_index)
    if args.preflight_only:
        _write_private_json(args.output, payload)
        print(json.dumps({"status": CPU_STATUS, "row_index": args.row_index,
            "processor_video_grid_thw": prepared["video_grid_thw"],
            "key_dose": prepared["key_dose"],
            "pixel_tensor_sha256": prepared["pixel_tensor_sha256"],
            "receipt": str(args.output)}))
        return 0
    cpu_raw = _private_file(args.cpu_receipt)
    cpu_receipt = json.loads(cpu_raw)
    _validate_cpu_receipt(cpu_receipt, prepared=prepared,
                          row_index=args.row_index, receipt_raw=cpu_raw)
    if cpu_receipt.get("row_index") != args.row_index:
        raise ValueError("cpu_pixel_receipt_row_index_mismatch")
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.monotonic()
    progress = {"generation_attempted": False, "generation_completed": False,
        "teacher_forced_score_attempted": 0,
        "teacher_forced_score_completed": 0,
        "frozen_target_token_ids_sha256": None}
    try:
        gpu_result = _score_and_generate(prepared, torch=torch, progress=progress)
    except Exception as exc:
        raw_failure = str(exc)
        failure_code = (raw_failure if isinstance(exc, ValueError)
                        and re.fullmatch(r"[a-z0-9_:]{1,200}", raw_failure)
                        else type(exc).__name__)
        failure = {"schema_version": 1,
            "status": "failed_single_video_generated_object_pixel_exploratory_pilot",
            "prompt_protocol": PROMPT_PROTOCOL,
            "rights_status": "unverified; local-only per user authorization",
            "human_review_performed": False,
            "direct_generated_explanation_faithfulness": False,
            "cohort_sha256": _sha(prepared["cohort_raw"]),
            "cohort_preflight_sha256": _sha(prepared["cohort_preflight_raw"]),
            "cpu_pixel_receipt_sha256": _sha(cpu_raw),
            "row_index": args.row_index,
            "row_key_sha256": _sha(prepared["row_key"].encode()),
            "video_sha256": prepared["video_sha256"],
            "runner_sha256": _sha_file(Path(__file__)),
            "failure_type": type(exc).__name__,
            "failure_code": failure_code, "progress": progress,
            "elapsed_total_seconds": time.monotonic() - started}
        _write_private_json(args.output, failure)
        raise
    gpu_payload = {
        "schema_version": 1,
        "status": "completed_single_video_generated_object_pixel_exploratory_pilot",
        "prompt_protocol": PROMPT_PROTOCOL,
        "purpose": "generated-sentence conditional-likelihood sensitivity to annotation bbox pixel removal and fixed visual-key bias",
        "human_review_performed": False,
        "direct_generated_explanation_faithfulness": False,
        "rights_status": "unverified; local-only per user authorization",
        "cohort_sha256": _sha(prepared["cohort_raw"]),
        "cohort_preflight_sha256": _sha(prepared["cohort_preflight_raw"]),
        "cpu_pixel_receipt_sha256": _sha(cpu_raw),
        "row_index": args.row_index,
        "video_id_sha256": _sha(prepared["video_id"].encode()),
        "row_key_sha256": _sha(prepared["row_key"].encode()),
        "video_md5": prepared["video_md5"],
        "video_sha256": prepared["video_sha256"],
        "time_map_sha256": prepared["time_map_sha256"],
        "annotation_source_sha256": prepared["annotation_source_sha256"],
        "runner_sha256": _sha_file(Path(__file__)),
        "source_sha256": prepared["source_sha256"],
        "model_revision": MODEL_REVISION,
        "runtime_versions": {"python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "qwen-vl-utils": importlib.metadata.version("qwen-vl-utils"),
            "gpu_name": torch.cuda.get_device_name(0)},
        "processor_video_grid_thw": prepared["video_grid_thw"],
        "pixel_tensor_sha256": prepared["pixel_tensor_sha256"],
        "condition_key_counts": prepared["key_dose"],
        "anchor_frame_number": prepared["anchor_frame"],
        "anchor_sample_rank": prepared["anchor_rank"],
        "anchor_pts_seconds": prepared["anchor_pts_seconds"],
        "intervention": "source-frame global-mean RGB fill of clipped annotation bbox versus equal-size same-frame sham; the paired Qwen input frame is unchanged",
        "estimand": "D_key = meanLP(sham-filled input,key condition)-meanLP(object-filled input,key condition); primary interaction D_bbox_bias-D_spatial_sham_bias; secondary D_bbox_bias-D_no_bias",
        "generation": gpu_result,
        "elapsed_total_seconds": time.monotonic() - started,
        "limitations": ["n=1 exploratory; one generated sentence from one video",
            "HD-EPIC records bounding boxes rather than segmentation masks",
            "Qwen temporal patch size 2 fuses the anchor frame with its unchanged pair frame",
            "mean fill may create out-of-distribution pixels",
            "conditional likelihood differences do not establish claim truth, semantic support, or direct explanation faithfulness"],
    }
    _write_private_json(args.output, gpu_payload)
    print(json.dumps({"status": gpu_payload["status"], "row_index": args.row_index,
        "primary_interaction": gpu_result["contrasts"][
            "primary_bbox_minus_spatial_sham_interaction_nats_per_token"],
        "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
