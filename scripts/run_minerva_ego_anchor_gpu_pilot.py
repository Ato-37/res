#!/usr/bin/env python3
"""Local Qwen diagnostic: score annotation-linked rationale under temporal-key bias.

This exploratory diagnostic teacher-forces one dataset-authored object/time
rationale line under baseline, evidence-group bias, and an equal-dose temporal
sham. It is a prediction-dependence proxy, not a generated-explanation
faithfulness score.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import time
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ID = "P01-20240202-161354"
ROW_KEY = VIDEO_ID + ":b9f19f203842665262cfaf978348d269adbcd9da"
VIDEO_MD5 = "a13e897ebcb918b2b22b79df2b89dc02"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
LAYERS = tuple(range(5, 17))
FRAME_COUNT = 64
TEMPORAL_PATCH = 2
BIAS_NATS = math.log(2.0)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cohort_binding(*, cohort: dict, row_key: str, video_id: str) -> dict:
    if not isinstance(cohort, dict) or not isinstance(cohort.get("selection"), list):
        raise ValueError("cohort_selection_invalid")
    matches = [item for item in cohort["selection"]
               if isinstance(item, dict) and item.get("row_key") == row_key]
    if (len(matches) != 1 or matches[0].get("video_id") != video_id
            or matches[0].get("row_key_sha256") != sha(row_key.encode())):
        raise ValueError("cohort_row_video_binding_mismatch")
    return matches[0]


def validate_spatial_token_preflight(receipt: dict, *, cohort_sha256: str,
        preflight_sha256: str, row_key_sha256: str, video_sha256: str,
        time_map_sha256: str, annotation_source_sha256: dict, bias_sign: str,
        bias_nats: float, operation: str, grid: list[int],
        condition_positions: list[tuple[str, tuple[int, ...]]],
        rendered_prompt_sha256: str, runtime_source_sha256: dict) -> None:
    """Bind GPU scoring to the exact full-prompt CPU token/position smoke."""
    expected_status = "completed_full_prompt_cpu_spatial_token_preflight"
    if (not isinstance(receipt, dict) or receipt.get("status") != expected_status
            or receipt.get("cohort_sha256") != cohort_sha256
            or receipt.get("preflight_receipt_sha256") != preflight_sha256
            or receipt.get("row_key_sha256") != row_key_sha256
            or receipt.get("video_sha256") != video_sha256
            or receipt.get("time_map_sha256") != time_map_sha256
            or receipt.get("annotation_source_sha256") != annotation_source_sha256
            or receipt.get("bias_sign") != bias_sign
            or receipt.get("bias_nats") != bias_nats
            or receipt.get("intervention_operation") != operation
            or receipt.get("processor_video_grid_thw") != grid
            or receipt.get("rendered_prompt_sha256") != rendered_prompt_sha256
            or receipt.get("runtime_source_sha256") != runtime_source_sha256):
        raise ValueError("spatial_token_preflight_binding_mismatch")
    counts = {label: len(positions) for label, positions in condition_positions}
    position_hashes = {label: sha(json.dumps(list(positions)).encode())
                       for label, positions in condition_positions}
    if (receipt.get("condition_key_counts") != counts
            or receipt.get("condition_key_position_sha256") != position_hashes):
        raise ValueError("spatial_token_preflight_key_positions_mismatch")


def decode_uniform_video(path: Path, mp4_frame_times_seconds=None,
                         anchor_frame_numbers=(), sample_frame_indices=None,
                         expected_sample_count: int | None = None):
    import av
    from PIL import Image
    container = av.open(str(path))
    stream = container.streams.video[0]
    expected_count = int(stream.frames)
    expected_samples = (FRAME_COUNT if expected_sample_count is None
                        else expected_sample_count)
    if (type(expected_samples) is not int or expected_samples < 1
            or expected_count < expected_samples):
        container.close()
        raise ValueError("video_frame_count_below_sample_count")
    if sample_frame_indices is None:
        from minerva_ego_anchor_utils import build_uniform_sample_indices
        ranks = build_uniform_sample_indices(total_frames=expected_count,
            sample_count=FRAME_COUNT, anchor_frames=anchor_frame_numbers)
    else:
        ranks = list(sample_frame_indices)
        if (anchor_frame_numbers or len(ranks) != expected_samples
                or ranks != sorted(set(ranks))
                or any(type(index) is not int or not 0 <= index < expected_count
                       for index in ranks)):
            container.close()
            raise ValueError("provided_sample_frame_indices_invalid")
    if len(set(ranks)) != expected_samples:
        container.close()
        raise ValueError("uniform_frame_indices_not_unique")
    decoded = {}
    if mp4_frame_times_seconds is not None:
        if len(mp4_frame_times_seconds) != expected_count:
            container.close()
            raise ValueError("video_frame_count_time_map_mismatch")
        time_base = Fraction(stream.time_base)
        iterator = container.decode(stream)
        try:
            first = next(iterator)
        except StopIteration:
            container.close()
            raise ValueError("video_first_frame_missing") from None
        first_time = Fraction(first.pts) * Fraction(first.time_base)
        first_relative = Fraction(str(mp4_frame_times_seconds[0]))
        if abs(first_relative) > Fraction(1, 1000):
            container.close()
            raise ValueError("time_map_first_frame_not_zero")
        decoded[0] = (first, Image.fromarray(first.to_ndarray(format="rgb24")))
        for index in ranks:
            if index == 0:
                continue
            target = first_time + Fraction(str(mp4_frame_times_seconds[index]))
            target_pts = int(round(target / time_base))
            container.seek(target_pts, stream=stream, backward=True)
            found = None
            for frame in container.decode(stream):
                observed = Fraction(frame.pts) * Fraction(frame.time_base)
                delta = observed - target
                if abs(delta) <= Fraction(1, 1000):
                    found = frame
                    break
                if delta > Fraction(1, 1000):
                    break
            if found is None:
                container.close()
                raise ValueError("seeked_frame_pts_not_in_official_time_map")
            decoded[index] = (found, Image.fromarray(found.to_ndarray(format="rgb24")))
        sampled = []
        for index in ranks:
            frame = decoded[index][0]
            relative_time = (Fraction(frame.pts) * Fraction(frame.time_base)
                             - first_time)
            if abs(relative_time - Fraction(str(mp4_frame_times_seconds[index]))) > Fraction(1, 1000):
                container.close()
                raise ValueError("seeked_frame_index_time_mismatch")
            sampled.append({"decoded_frame_index": index,
                "relative_presentation_time": [relative_time.numerator, relative_time.denominator],
                "time_seconds": float(relative_time)})
    else:
        wanted = set(ranks)
        frame_count = 0
        for index, frame in enumerate(container.decode(stream)):
            frame_count += 1
            if index in wanted:
                decoded[index] = (frame, Image.fromarray(frame.to_ndarray(format="rgb24")))
        if frame_count != expected_count or len(decoded) != expected_samples:
            container.close()
            raise ValueError("decoded_frame_count_mismatch")
        first_time = Fraction(decoded[0][0].pts) * Fraction(decoded[0][0].time_base)
        sampled = []
        for index in ranks:
            frame = decoded[index][0]
            relative_time = Fraction(frame.pts) * Fraction(frame.time_base) - first_time
            sampled.append({"decoded_frame_index": index,
                "relative_presentation_time": [relative_time.numerator, relative_time.denominator],
                "time_seconds": float(relative_time)})
    frames = [decoded[index][1] for index in ranks]
    container.close()
    return frames, sampled


def _video_groups(batch, processor, frame_count: int = FRAME_COUNT):
    import torch
    ids = batch["input_ids"][0].tolist()
    temporal, height, width = [int(value) for value in batch["video_grid_thw"][0].tolist()]
    patch = int(processor.video_processor.temporal_patch_size)
    merge = int(processor.image_processor.merge_size)
    positions = [i for i, token in enumerate(ids)
                 if token == int(processor.video_token_id)]
    per_group = (height // merge) * (width // merge)
    if (patch != TEMPORAL_PATCH or temporal != frame_count // TEMPORAL_PATCH
            or frame_count % patch
            or not positions
            or len(positions) != temporal * per_group
            or positions != list(range(positions[0], positions[0] + len(positions)))):
        raise ValueError("processor_video_group_geometry_mismatch")
    return tuple(tuple(positions[i * per_group:(i + 1) * per_group])
                 for i in range(temporal)), [temporal, height, width]


def select_supported_target(*, reasoning: str, object_name: str, object_id: str,
                            track_segments):
    """Select the first numbered object-linked rationale line overlapping a track."""
    from minerva_ego_anchor_utils import supported_rationale_lines
    lines = supported_rationale_lines(reasoning=reasoning,
        object_tracks={object_id: track_segments})
    if not lines:
        raise ValueError("supported_rationale_line_missing")
    token = f"{object_name}-{object_id}"
    target_line = lines[0]
    target = re.sub(r"^\s*\d+[.)]\s*", "", target_line)
    target = re.sub(r"\s*\(" + re.escape(token) + r"\)", "", target)
    target = " ".join(target.split())
    if not target:
        raise ValueError("redacted_rationale_target_empty")
    return target, len(lines)


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--minerva-json", type=Path, default=Path("/tmp/minerva_ego_20260205.json"))
    parser.add_argument("--time-map", type=Path, required=True)
    parser.add_argument("--assoc-info", type=Path, default=Path("/tmp/hd_epic_assoc_info.json"))
    parser.add_argument("--mask-info", type=Path, default=Path("/tmp/hd_epic_mask_info.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", type=Path,
        help="Optional frozen private cohort used to bind row/video provenance")
    parser.add_argument("--preflight", type=Path,
        help="Private CPU preflight receipt required for spatial_bbox intervention")
    parser.add_argument("--token-preflight", type=Path,
        help="Exact full-prompt CPU token receipt required before spatial GPU scoring")
    parser.add_argument("--preflight-only", action="store_true",
        help="Run full CPU processor/token geometry smoke and exit before model loading")
    parser.add_argument("--row-key", default=ROW_KEY)
    parser.add_argument("--expected-video-md5", default=VIDEO_MD5)
    parser.add_argument("--intervention", choices=("temporal_group", "spatial_bbox"),
        default="temporal_group")
    parser.add_argument("--bias-sign", choices=("positive", "negative"),
        default="positive", help="Sign of the fixed-magnitude ln(2) attention-key bias")
    parser.add_argument("--operation", choices=("key_bias", "key_ablation"),
        default="key_bias", help="Internal visual-key operation for spatial_bbox runs")
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.operation == "key_ablation" and args.intervention != "spatial_bbox":
        raise ValueError("key_ablation_requires_spatial_bbox")
    bias_nats = BIAS_NATS if args.bias_sign == "positive" else -BIAS_NATS
    digest = hashlib.md5()
    with args.video.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != args.expected_video_md5:
        raise ValueError("video_checksum_mismatch")
    rows = json.loads(args.minerva_json.read_text())
    row = next(item for item in rows if item.get("key") == args.row_key)
    video_id = row.get("video_id")
    if args.video.stem.removesuffix("_verified") != video_id:
        raise ValueError("video_row_binding_mismatch")
    cohort_raw = None
    preflight_raw = None
    preflight_record = None
    if args.cohort is not None:
        private = ROOT / "data" / "private_minerva"
        if (args.cohort.parent.resolve() != private.resolve()
                or args.cohort.stat().st_mode & 0o077):
            raise ValueError("cohort_path_or_permissions_invalid")
        cohort_raw = args.cohort.read_bytes()
        cohort_data = json.loads(cohort_raw)
        validate_cohort_binding(cohort=cohort_data,
                                row_key=args.row_key, video_id=video_id)
        if args.intervention == "spatial_bbox":
            source_hashes = cohort_data.get("source_sha256", {})
            for key, source_path in (("minerva_json", args.minerva_json),
                    ("assoc_info", args.assoc_info), ("mask_info", args.mask_info)):
                if source_hashes.get(key) != sha_file(source_path):
                    raise ValueError(f"cohort_{key}_source_hash_mismatch")
    if args.intervention == "spatial_bbox":
        private = ROOT / "data" / "private_minerva"
        if (args.cohort is None or args.preflight is None
                or args.preflight.parent.resolve() != private.resolve()
                or args.preflight.stat().st_mode & 0o077):
            raise ValueError("spatial_intervention_requires_private_preflight")
        preflight_raw = args.preflight.read_bytes()
        preflight = json.loads(preflight_raw)
        if preflight.get("cohort_sha256") != sha(cohort_raw):
            raise ValueError("preflight_cohort_binding_mismatch")
        bound_rows = [item for item in preflight.get("records", [])
                      if item.get("video_id") == video_id
                      and item.get("row_key_sha256") == sha(args.row_key.encode())]
        if len(bound_rows) != 1 or bound_rows[0].get("status") != "pass":
            raise ValueError("preflight_row_not_unique_or_passing")
        preflight_record = bound_rows[0]
    references = row.get("referenced_object_masks", [])
    if len(references) != 1:
        raise ValueError("single_reference_contract_failed")
    token = references[0]
    name, oid = token.rsplit("-", 1)
    assoc = json.loads(args.assoc_info.read_text())[video_id]
    masks = json.loads(args.mask_info.read_text())[video_id]
    obj = assoc[oid]
    if name.casefold() != obj["name"].casefold():
        raise ValueError("target_display_name_mismatch")
    track_segments = [track["time_segment"] for track in obj["tracks"]]
    mask_frames_by_id = {mask_id: masks[mask_id]["frame_number"]
                         for track in obj["tracks"] for mask_id in track["masks"]}
    mask_frames = list(mask_frames_by_id.values())
    with args.time_map.open(newline="") as stream:
        time_rows = list(csv.DictReader(stream))
    frame_times = [float(item["mp4_time_ns"]) / 1e9 for item in time_rows]
    from minerva_ego_anchor_utils import (evidence_temporal_groups,
        matched_control_groups, select_rationale_mask_anchor,
        track_overlapping_temporal_groups)
    rationale_anchor = select_rationale_mask_anchor(
        reasoning=row["reasoning"], object_id=oid, tracks=obj["tracks"],
        mask_frame_numbers=mask_frames_by_id,
        mp4_frame_times_seconds=frame_times)
    target, supported_line_count = select_supported_target(
        reasoning=row["reasoning"], object_name=name, object_id=oid,
        track_segments=track_segments)
    from minerva_ego_anchor_utils import (build_anchor_centered_sample_indices,
        include_exact_anchor_sample)
    duration = frame_times[-1] - frame_times[0]
    frame_count = min(FRAME_COUNT, math.floor(duration * 2.0) + 1)
    frame_count -= frame_count % TEMPORAL_PATCH
    if frame_count < 4:
        raise ValueError("video_too_short_for_temporal_patch_sampling")
    clip_sampling = build_anchor_centered_sample_indices(
        mp4_frame_times_seconds=frame_times,
        anchor_frame_number=rationale_anchor["mask_frame_number"],
        sample_count=frame_count, sample_fps=2.0)
    clip_sampling.update(include_exact_anchor_sample(
        sample_frame_indices=clip_sampling["sample_frame_indices"],
        mp4_frame_times_seconds=frame_times,
        anchor_frame_number=rationale_anchor["mask_frame_number"]))
    clip_sampling["maximum_grid_time_error_seconds"] = max(
        abs(frame_times[source_index] - frame_times[clip_sampling["sample_frame_indices"][0]]
            - rank / 2.0)
        for rank, source_index in enumerate(clip_sampling["sample_frame_indices"]))
    if preflight_record is not None:
        expected_sampling = preflight_record["clip_sampling"]
        if (clip_sampling["sample_frame_indices"] != expected_sampling["sample_frame_indices"]
                or clip_sampling["anchor_sample_rank"] != expected_sampling["anchor_sample_rank"]
                or preflight_record["video_md5"] != digest.hexdigest()
                or preflight_record["video_sha256"] != sha_file(args.video)
                or preflight_record["time_map_sha256"] != sha_file(args.time_map)):
            raise ValueError("preflight_media_or_sampling_binding_mismatch")
    frames, sampled = decode_uniform_video(args.video, frame_times,
        sample_frame_indices=clip_sampling["sample_frame_indices"],
        expected_sample_count=frame_count)
    anchor = evidence_temporal_groups(mask_frame_numbers=[rationale_anchor["mask_frame_number"]],
        mp4_frame_times_seconds=frame_times, sampled_frames=sampled,
        temporal_patch_size=TEMPORAL_PATCH,
        temporal_step_count=frame_count // TEMPORAL_PATCH)
    evidence_groups = anchor["evidence_temporal_groups"]
    excluded = track_overlapping_temporal_groups(
        sampled_times_seconds=[item["time_seconds"] for item in sampled],
        temporal_patch_size=TEMPORAL_PATCH, track_segments=track_segments)
    from minerva_ego_anchor_utils import validate_sampled_evidence_alignment
    evidence_alignment = validate_sampled_evidence_alignment(
        mapping=anchor, track_groups=excluded, maximum_error_seconds=1.0)
    control_groups = matched_control_groups(temporal_step_count=frame_count // TEMPORAL_PATCH,
        evidence_groups=evidence_groups, excluded_groups=excluded,
        seed="minerva-eg-anchor-pilot-v1", record_token=args.row_key)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
        / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        raise ValueError("pinned_checkpoint_missing")
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    if args.preflight_only and args.intervention != "spatial_bbox":
        raise ValueError("cpu_preflight_requires_spatial_bbox_intervention")
    if not args.preflight_only and (not torch.cuda.is_available()
                                    or not torch.cuda.is_bf16_supported()):
        raise ValueError("cuda_bf16_unavailable")
    free = total = None
    if not args.preflight_only:
        free, total = torch.cuda.mem_get_info()
        if free < 12 * 1024**3:
            raise ValueError("free_vram_below_12_gib")
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=128 * 28 * 28, max_pixels=128 * 28 * 28)
    prompt = (f"The supplied video is a local clip from source time "
        f"{clip_sampling['clip_start_source_seconds']:.3f} to "
        f"{clip_sampling['clip_end_source_seconds']:.3f} seconds. Clip timestamps "
        "are relative; add the source start time when comparing with the original video.\n"
        + row["question"] + "\nOptions:\n" + "\n".join(
        f"{chr(65+i)}. {row[f'answer_choice_{i}']}" for i in range(5)) +
        f"\nDescribe the {name}'s movement, citing relevant times.")
    messages = [{"role":"user","content":[{"type":"video","video":frames,
        "sample_fps": clip_sampling["sample_fps"]},
        {"type":"text","text":prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or len(videos) != 1:
        raise ValueError("qwen_video_utility_contract_invalid")
    pixel_values, metadata = videos[0]
    if metadata.get("frames_indices") != list(range(frame_count)):
        raise ValueError("processor_frame_indices_changed")
    batch = processor(text=[rendered], images=images, videos=[pixel_values],
        video_metadata=[metadata], do_sample_frames=False, cap_pixels_per_frame=True,
        padding=True, return_tensors="pt")
    groups, grid = _video_groups(batch, processor, frame_count=frame_count)
    second_per_grid_ts_value = batch["second_per_grid_ts"]
    second_per_grid_ts = (second_per_grid_ts_value.tolist()
        if hasattr(second_per_grid_ts_value, "tolist")
        else list(second_per_grid_ts_value))
    target_ids = tuple(processor.tokenizer(target, add_special_tokens=False)["input_ids"])
    if not target_ids or len(target_ids) > 128:
        raise ValueError("target_tokenization_invalid")
    score_values = {}
    score_details = {}
    operation_suffix = "ablation" if args.operation == "key_ablation" else "bias"
    support_label = ("object_time_support_bias" if args.intervention == "temporal_group"
                     else f"object_bbox_support_{operation_suffix}")
    spatial_sham_label = f"equal_dose_same_pair_spatial_sham_{operation_suffix}"
    temporal_sham_label = ("equal_dose_temporal_sham_bias"
        if args.intervention == "temporal_group"
        else f"equal_dose_temporal_bbox_sham_{operation_suffix}")
    condition_positions = [("baseline", ())]
    if args.intervention == "temporal_group":
        condition_positions += [
            ("object_time_support_bias", tuple(
                position for group in evidence_groups for position in groups[group])),
            ("equal_dose_temporal_sham_bias", tuple(
                position for group in control_groups for position in groups[group]))]
    else:
        from minerva_ego_anchor_utils import (bbox_to_video_visual_key_positions,
            matched_spatial_sham_offsets)
        anchor_rank = clip_sampling["anchor_sample_rank"]
        matching_mask_ids = [mask_id for mask_id, frame_number in mask_frames_by_id.items()
                             if frame_number == rationale_anchor["mask_frame_number"]]
        if len(matching_mask_ids) != 1:
            raise ValueError("anchor_frame_mask_id_not_unique")
        anchor_mask_id = matching_mask_ids[0]
        anchor_mask = masks[anchor_mask_id]
        patch_size = int(processor.video_processor.patch_size)
        merge_size = int(processor.image_processor.merge_size)
        mapped_bbox = bbox_to_video_visual_key_positions(
            bbox_xyxy=anchor_mask["bbox"], frame_rank=anchor_rank,
            frame_count=frame_count, temporal_patch_size=TEMPORAL_PATCH,
            visual_token_groups=groups, source_width=frames[anchor_rank].width,
            source_height=frames[anchor_rank].height, grid_height=grid[1],
            grid_width=grid[2], patch_size=patch_size,
            spatial_merge_size=merge_size)
        if preflight_record is not None:
            expected_bbox = preflight_record["bbox_to_qwen_spatiotemporal_mapping"]
            if (list(mapped_bbox["patch_cell_offsets"]) != expected_bbox["patch_cell_offsets"]
                    or mapped_bbox["temporal_group_index"] != expected_bbox["temporal_group_index"]
                    or [1, grid[1], grid[2]] != expected_bbox["video_grid_thw_for_pair"]
                    or anchor_rank != expected_bbox["anchor_sample_rank"]
                    or list(mapped_bbox["source_bbox_xyxy"]) != expected_bbox["source_bbox_xyxy"]
                    or mapped_bbox["clipped_area_fraction"] != expected_bbox["clipped_area_fraction"]):
                raise ValueError("processor_bbox_geometry_differs_from_preflight")
        offsets = mapped_bbox["patch_cell_offsets"]
        spatial_sham_offsets = matched_spatial_sham_offsets(
            merged_cell_count=math.prod(mapped_bbox["merged_grid_hw"]),
            support_offsets=offsets, seed="minerva-eg-spatial-bbox-v1",
            record_token=args.row_key)
        temporal_sham_group = control_groups[0]
        condition_positions += [
            (f"object_bbox_support_{operation_suffix}", mapped_bbox["visual_key_positions"]),
            (f"equal_dose_same_pair_spatial_sham_{operation_suffix}", tuple(
                groups[mapped_bbox["temporal_group_index"]][offset]
                for offset in spatial_sham_offsets)),
            (f"equal_dose_temporal_bbox_sham_{operation_suffix}", tuple(
                groups[temporal_sham_group][offset] for offset in offsets))]
    private_output = ROOT / "data" / "private_minerva"
    if (args.output.parent.resolve() != private_output.resolve()
            or not private_output.is_dir() or private_output.stat().st_mode & 0o077
            or args.output.exists()):
        raise ValueError("output_parent_missing_or_output_exists")
    if args.preflight_only:
        token_position_lists = {label: list(positions)
                                for label, positions in condition_positions}
        runtime_sources = {
            "run_minerva_ego_anchor_gpu_pilot.py": sha(Path(__file__).read_bytes()),
            "minerva_ego_anchor_utils.py": sha_file(Path(__file__).with_name("minerva_ego_anchor_utils.py")),
            "run_claim_support_qta_16frame_exploratory.py": sha_file(
                Path(__file__).with_name("run_claim_support_qta_16frame_exploratory.py")),
            "qta_multiquery_video_key_intervention.py": sha_file(
                Path(__file__).with_name("qta_multiquery_video_key_intervention.py")),
        }
        payload = {
            "schema_version": 1,
            "status": "completed_full_prompt_cpu_spatial_token_preflight",
            "human_review_performed": False,
            "direct_generated_explanation_faithfulness": False,
            "rights_status": "unverified; local-only per user authorization",
            "cohort_sha256": sha(cohort_raw),
            "preflight_receipt_sha256": sha(preflight_raw),
            # Ablation deletes keys with -inf; it does not apply a finite bias.
            "bias_sign": None if args.operation == "key_ablation" else args.bias_sign,
            "bias_nats": None if args.operation == "key_ablation" else bias_nats,
            "intervention_operation": args.operation,
            "annotation_source_sha256": json.loads(cohort_raw)["source_sha256"],
            "video_sha256": sha_file(args.video),
            "time_map_sha256": sha_file(args.time_map),
            "row_key_sha256": sha(args.row_key.encode()),
            "frame_count": frame_count,
            "anchor_sample_rank": clip_sampling["anchor_sample_rank"],
            "processor_video_grid_thw": grid,
            "second_per_grid_ts": second_per_grid_ts,
            "input_token_count": int(batch["input_ids"].shape[1]),
            "video_token_count": sum(len(group) for group in groups),
            "bbox_temporal_group_index": mapped_bbox["temporal_group_index"],
            "bbox_patch_cell_offsets": list(mapped_bbox["patch_cell_offsets"]),
            "bbox_visual_key_positions": list(mapped_bbox["visual_key_positions"]),
            "spatial_sham_offsets": list(spatial_sham_offsets),
            "condition_key_counts": {label: len(positions)
                                      for label, positions in condition_positions},
            "condition_key_position_sha256": {
                label: sha(json.dumps(positions).encode())
                for label, positions in token_position_lists.items()},
            "rendered_prompt_sha256": sha(rendered.encode()),
            "runner_sha256": sha(Path(__file__).read_bytes()),
            "runtime_source_sha256": runtime_sources,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        os.chmod(args.output, 0o600)
        print(json.dumps({key: payload[key] for key in (
            "status", "frame_count", "processor_video_grid_thw",
            "bbox_temporal_group_index", "bbox_patch_cell_offsets",
            "condition_key_counts")}))
        return 0
    if args.intervention == "spatial_bbox":
        private = ROOT / "data" / "private_minerva"
        if (args.token_preflight is None
                or args.token_preflight.parent.resolve() != private.resolve()
                or not args.token_preflight.is_file()
                or args.token_preflight.stat().st_mode & 0o077):
            raise ValueError("spatial_gpu_requires_private_token_preflight")
        token_preflight_raw = args.token_preflight.read_bytes()
        token_preflight = json.loads(token_preflight_raw)
        runtime_sources = {
            "run_minerva_ego_anchor_gpu_pilot.py": sha(Path(__file__).read_bytes()),
            "minerva_ego_anchor_utils.py": sha_file(Path(__file__).with_name("minerva_ego_anchor_utils.py")),
            "run_claim_support_qta_16frame_exploratory.py": sha_file(
                Path(__file__).with_name("run_claim_support_qta_16frame_exploratory.py")),
            "qta_multiquery_video_key_intervention.py": sha_file(
                Path(__file__).with_name("qta_multiquery_video_key_intervention.py")),
        }
        validate_spatial_token_preflight(token_preflight,
            cohort_sha256=sha(cohort_raw), preflight_sha256=sha(preflight_raw),
            row_key_sha256=sha(args.row_key.encode()), video_sha256=sha_file(args.video),
            time_map_sha256=sha_file(args.time_map),
            annotation_source_sha256=json.loads(cohort_raw)["source_sha256"],
            bias_sign=None if args.operation == "key_ablation" else args.bias_sign,
            bias_nats=None if args.operation == "key_ablation" else bias_nats,
            operation=args.operation, grid=grid,
            condition_positions=condition_positions,
            rendered_prompt_sha256=sha(rendered.encode()),
            runtime_source_sha256=runtime_sources)
        token_preflight_sha256 = sha(token_preflight_raw)
    else:
        token_preflight_sha256 = None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    from run_claim_support_qta_16frame_exploratory import _score_branch
    start = time.monotonic()
    torch.manual_seed(20261022); torch.cuda.manual_seed_all(20261022)
    torch.cuda.reset_peak_memory_stats()
    for condition, key_positions in condition_positions:
        biased_positions = (key_positions if condition != "baseline"
                            and args.operation == "key_bias" else ())
        ablated_positions = (key_positions if condition != "baseline"
                             and args.operation == "key_ablation" else ())
        score_values[condition], score_details[condition] = _score_branch(model=model, batch_cpu=batch,
            target_ids=target_ids, biased_positions=biased_positions, torch=torch,
            logit_bias=(0.0 if args.operation == "key_ablation" else bias_nats),
            ablated_positions=ablated_positions)
    payload = {
        "schema_version": 1,
        "status": "completed_single_video_exploratory_reference_likelihood_pilot",
        "purpose": "teacher-forced score of one fixed rationale-like sentence under evidence and matched sham visual-key intervention",
        "direct_generated_explanation_faithfulness": False,
        "human_review_performed": False,
        "rights_status": "unverified; local-only per user authorization",
        "dataset": "Minerva-Ego / HD-EPIC", "model_revision": MODEL_REVISION,
        "video_md5": digest.hexdigest(),
        "video_sha256": sha_file(args.video),
        "time_map_sha256": sha_file(args.time_map),
        "cohort_sha256": sha(cohort_raw) if cohort_raw is not None else None,
        "preflight_receipt_sha256": sha(preflight_raw) if preflight_raw is not None else None,
        "token_preflight_receipt_sha256": token_preflight_sha256,
        "annotation_source_sha256": (json.loads(cohort_raw).get("source_sha256")
            if cohort_raw is not None else None),
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "runtime_source_sha256": runtime_sources,
        "runtime_versions": {"python": platform.python_version(),
            **{name: importlib_metadata.version(package)
            for name, package in (("torch", "torch"),
                ("transformers", "transformers"), ("qwen_vl_utils", "qwen-vl-utils"),
                ("av", "av"), ("pillow", "Pillow"))}},
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "row_key_sha256": sha(args.row_key.encode()), "target_sha256": sha(target.encode()),
        "supported_rationale_line_count": supported_line_count,
        "rationale_mask_anchor": {key: value for key, value in rationale_anchor.items()
                                  if key != "rationale_line"},
        "sampling_strategy": f"largest even frame count up to {FRAME_COUNT} that fits the source duration at nominal 2 fps, centered on an annotated mask frame; exact anchor inserted",
        "clip_sampling": clip_sampling,
        "intervention": args.intervention,
        "intervention_operation": args.operation,
        "mapped_bbox_to_video_visual_keys": mapped_bbox if args.intervention == "spatial_bbox" else None,
        "spatial_sham_offsets": list(spatial_sham_offsets) if args.intervention == "spatial_bbox" else None,
        "temporal_sham_group": temporal_sham_group if args.intervention == "spatial_bbox" else None,
        "target_source": "first dataset rationale line whose exact object ID and timestamp bin/range overlaps its track; opaque object-ID parenthetical removed",
        "requested_rows": 1, "effective_rows": 1,
        "forward_count": len(condition_positions),
        "frame_count": frame_count, "processor_video_grid_thw": grid,
        "second_per_grid_ts": second_per_grid_ts,
        "sampled_frame_count": len(sampled), "evidence_groups": evidence_groups,
        "control_groups": control_groups, "annotated_track_groups_excluded_from_sham": excluded,
        "scoring_audit": score_details,
        "mapped_mask_frames": anchor["mapped_mask_frames"],
        "sampled_evidence_alignment": evidence_alignment,
        "target_token_count": len(target_ids), "target_mean_logprob_nats_per_token": score_values,
        "support_minus_baseline": score_values[support_label] - score_values["baseline"],
        "support_minus_sham": score_values[support_label] - score_values[
            "equal_dose_temporal_sham_bias" if args.intervention == "temporal_group"
            else spatial_sham_label],
        "support_minus_temporal_sham": (score_values[support_label]
            - score_values[temporal_sham_label]
            if args.intervention == "spatial_bbox" else None),
        "bias_sign": None if args.operation == "key_ablation" else args.bias_sign,
        "bias_nats": None if args.operation == "key_ablation" else bias_nats,
        "biased_layers_1_based": list(LAYERS),
        "elapsed_scoring_seconds": time.monotonic() - start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "initial_free_vram_bytes": free,
        "visible_total_vram_bytes": total,
        "limitations": ["n=1 exploratory", "dataset-authored rationale line, not generated explanation",
            "annotation-derived temporal anchors not independently human-reviewed", "does not establish faithfulness improvement"]}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({k:payload[k] for k in ("status","forward_count","processor_video_grid_thw",
        "evidence_groups","control_groups","target_mean_logprob_nats_per_token",
        "support_minus_baseline","support_minus_sham","elapsed_scoring_seconds","peak_vram_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
