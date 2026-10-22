#!/usr/bin/env python3
"""Verify frozen Minerva videos, official time maps and temporal controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from minerva_ego_anchor_utils import (
    evidence_temporal_groups, matched_control_groups,
    build_anchor_centered_sample_indices, select_rationale_mask_anchor,
    include_exact_anchor_sample, bbox_to_merged_patch_offsets,
    track_overlapping_temporal_groups,
    validate_sampled_evidence_alignment)
from run_minerva_ego_anchor_gpu_pilot import FRAME_COUNT, decode_uniform_video
from download_frozen_hd_epic_videos import EXPECTED_MANIFEST_SHA256

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_manifest_path(relative: str) -> str:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("manifest_path_invalid")
    normalized = os.path.normpath(relative.strip().removeprefix("*"))
    if normalized.startswith("../") or normalized == "..":
        raise ValueError("manifest_path_traversal")
    return normalized.removeprefix("./")


def parse_manifest(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    if source_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("official_manifest_hash_mismatch")
    files = {}
    for line in raw.decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        files[normalize_manifest_path(" ".join(parts[1:]))] = parts[0]
    return files, source_hash


def load_time_map_seconds(path: Path) -> list[float]:
    with path.open(newline="") as stream:
        values = [float(item["mp4_time_ns"]) / 1e9
                  for item in csv.DictReader(stream)]
    if (not values or any(not math.isfinite(value) for value in values)
            or any(right <= left for left, right in zip(values, values[1:]))):
        raise ValueError("time_map_invalid")
    return values


def process_anchor_frame_pair(*, frames, anchor_rank: int, video_processor):
    """Run only the anchor's actual two-frame pair through Qwen CPU preprocessing."""
    import numpy as np
    import torch
    pair_start = (anchor_rank // video_processor.temporal_patch_size) * video_processor.temporal_patch_size
    pair_end = pair_start + video_processor.temporal_patch_size
    if pair_end > len(frames):
        raise ValueError("anchor_temporal_pair_out_of_range")
    pair = frames[pair_start:pair_end]
    arrays = [np.asarray(frame.convert("RGB")) for frame in pair]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("anchor_temporal_pair_frame_shape_mismatch")
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
    features = video_processor([tensor], return_tensors="pt",
                               cap_pixels_per_frame=True)
    grid = [int(value) for value in features["video_grid_thw"][0].tolist()]
    if grid[0] != 1:
        raise ValueError("anchor_pair_temporal_grid_mismatch")
    return pair_start, grid, pair


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--minerva-json", type=Path,
        default=Path("/tmp/minerva_ego_20260205.json"))
    parser.add_argument("--assoc-info", type=Path,
        default=Path("/tmp/hd_epic_assoc_info.json"))
    parser.add_argument("--mask-info", type=Path,
        default=Path("/tmp/hd_epic_mask_info.json"))
    parser.add_argument("--manifest", type=Path,
        default=Path("/tmp/hd-epic-downloader/data/md5.txt"))
    parser.add_argument("--media-root", type=Path,
        default=PRIVATE / "range_download/HD-EPIC")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.cohort.parent.resolve() != PRIVATE.resolve()
            or args.output.parent.resolve() != PRIVATE.resolve()
            or not PRIVATE.is_dir() or PRIVATE.stat().st_mode & 0o077
            or args.output.exists()
            or not args.media_root.resolve().is_relative_to(PRIVATE.resolve())):
        raise ValueError("private_preflight_contract_failed")

    cohort_raw = args.cohort.read_bytes()
    cohort = json.loads(cohort_raw)
    rows_by_key = {row["key"]: row for row in json.loads(args.minerva_json.read_text())}
    assoc = json.loads(args.assoc_info.read_text())
    masks = json.loads(args.mask_info.read_text())
    manifest, manifest_sha256 = parse_manifest(args.manifest)
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
    pixel_cap = 128 * 28 * 28
    video_processor = Qwen2VLVideoProcessor(min_pixels=pixel_cap,
                                             max_pixels=pixel_cap)
    records = []
    started = time.monotonic()
    for selected in cohort.get("selection", []):
        video_id = selected["video_id"]
        record = {"video_id": video_id,
                  "row_key_sha256": selected["row_key_sha256"],
                  "status": "started"}
        try:
            row = rows_by_key[selected["row_key"]]
            if row.get("video_id") != video_id or len(row.get("referenced_object_masks", [])) != 1:
                raise ValueError("cohort_row_binding_invalid")
            display_name, object_id = row["referenced_object_masks"][0].rsplit("-", 1)
            obj = assoc[video_id][object_id]
            if display_name.casefold() != obj["name"].casefold():
                raise ValueError("display_name_join_failed")
            participant = video_id[:3]
            relative = f"Videos/{participant}/{video_id}"
            media_path = args.media_root / "Videos" / participant
            video = media_path / f"{video_id}.mp4"
            time_map = media_path / f"{video_id}_mp4_to_vrs_time_ns.csv"
            if not video.is_file():
                raise ValueError("video_file_missing")
            if not time_map.is_file():
                raise ValueError("time_map_file_missing")
            video_rel = relative + ".mp4"
            map_rel = relative + "_mp4_to_vrs_time_ns.csv"
            video_md5 = digest_file(video, "md5")
            map_md5 = digest_file(time_map, "md5")
            if video_md5 != manifest[video_rel]:
                raise ValueError("video_official_md5_mismatch")
            if map_md5 != manifest[map_rel]:
                raise ValueError("time_map_official_md5_mismatch")
            times = load_time_map_seconds(time_map)
            import av
            with av.open(str(video)) as container:
                frame_count = int(container.streams.video[0].frames)
            if len(times) != frame_count:
                raise ValueError("video_frame_count_time_map_mismatch")
            tracks = [track["time_segment"] for track in obj["tracks"]]
            mask_ids = [mask_id for track in obj["tracks"] for mask_id in track["masks"]]
            mask_frames_by_id = {mask_id: masks[video_id][mask_id]["frame_number"]
                                 for mask_id in mask_ids}
            rationale_anchor = select_rationale_mask_anchor(
                reasoning=row["reasoning"], object_id=object_id,
                tracks=obj["tracks"], mask_frame_numbers=mask_frames_by_id,
                mp4_frame_times_seconds=times)
            anchor_frame = rationale_anchor["mask_frame_number"]
            duration_seconds = times[-1] - times[0]
            available_sample_count = math.floor(duration_seconds * 2.0) + 1
            sample_count = min(FRAME_COUNT, available_sample_count)
            sample_count -= sample_count % 2
            if sample_count < 4:
                raise ValueError("video_too_short_for_temporal_patch_sampling")
            clip_sampling = build_anchor_centered_sample_indices(
                mp4_frame_times_seconds=times, anchor_frame_number=anchor_frame,
                sample_count=sample_count, sample_fps=2.0)
            exact_sample = include_exact_anchor_sample(
                sample_frame_indices=clip_sampling["sample_frame_indices"],
                mp4_frame_times_seconds=times,
                anchor_frame_number=anchor_frame)
            clip_sampling.update(exact_sample)
            clip_sampling["maximum_grid_time_error_seconds"] = max(
                abs(times[source_index] - times[clip_sampling["sample_frame_indices"][0]]
                    - rank / 2.0)
                for rank, source_index in enumerate(clip_sampling["sample_frame_indices"]))
            frames, sampled = decode_uniform_video(
                video, times, sample_frame_indices=clip_sampling["sample_frame_indices"],
                expected_sample_count=sample_count)
            anchor_rank = clip_sampling["anchor_sample_rank"]
            if sampled[anchor_rank]["decoded_frame_index"] != anchor_frame:
                raise ValueError("exact_annotation_frame_not_in_decoded_input")
            mapped = evidence_temporal_groups(mask_frame_numbers=[anchor_frame],
                mp4_frame_times_seconds=times, sampled_frames=sampled,
                temporal_patch_size=2, temporal_step_count=sample_count // 2)
            exact_map = mapped["mapped_mask_frames"][0]
            if (exact_map["sample_rank"] != anchor_rank
                    or exact_map["sampled_frame_index"] != anchor_frame
                    or exact_map["absolute_time_error_seconds"] > 0.001):
                raise ValueError("bbox_anchor_not_exactly_mapped")
            anchor_mask_ids = sorted(mask_id for mask_id, frame_number
                                     in mask_frames_by_id.items()
                                     if frame_number == anchor_frame)
            if len(anchor_mask_ids) != 1:
                raise ValueError("anchor_frame_mask_id_not_unique")
            anchor_mask = masks[video_id][anchor_mask_ids[0]]
            bbox_xyxy = anchor_mask.get("bbox")
            frame_width, frame_height = frames[anchor_rank].size
            pair_start, pair_grid, pair_frames = process_anchor_frame_pair(
                frames=frames, anchor_rank=anchor_rank,
                video_processor=video_processor)
            patch_size = int(video_processor.patch_size)
            merge_size = int(video_processor.merge_size)
            spatial_mapping = bbox_to_merged_patch_offsets(
                bbox_xyxy=bbox_xyxy, source_width=frame_width,
                source_height=frame_height, grid_height=pair_grid[1],
                grid_width=pair_grid[2], patch_size=patch_size,
                spatial_merge_size=merge_size)
            anchor_group = anchor_rank // int(video_processor.temporal_patch_size)
            pair_sampled = sampled[pair_start:pair_start + len(pair_frames)]
            spatial_mapping.update({
                "mask_id_sha256": hashlib.sha256(anchor_mask_ids[0].encode()).hexdigest(),
                "anchor_frame_number": anchor_frame,
                "anchor_sample_rank": anchor_rank,
                "temporal_group_index": anchor_group,
                "paired_sample_ranks": [pair_start, pair_start + 1],
                "paired_source_frame_numbers": [item["decoded_frame_index"]
                                                for item in pair_sampled],
                "paired_pts_seconds": [item["time_seconds"] for item in pair_sampled],
                "anchor_temporal_pair_exact": True,
                "video_grid_thw_for_pair": pair_grid,
                "patch_size": patch_size,
                "spatial_merge_size": merge_size,
                "temporal_patch_size": int(video_processor.temporal_patch_size),
                "mapped_visual_token_count": len(spatial_mapping["patch_cell_offsets"]),
                "merged_visual_token_count": math.prod(spatial_mapping["merged_grid_hw"]),
            })
            excluded = track_overlapping_temporal_groups(
                sampled_times_seconds=[sample["time_seconds"] for sample in sampled],
                temporal_patch_size=2, track_segments=tracks)
            mapped_errors = [item["absolute_time_error_seconds"]
                             for item in mapped["mapped_mask_frames"]]
            record.update({
                "frame_count": len(times),
                "sampled_frame_count": len(frames),
                "sample_fps_nominal": 2.0,
                "sample_target_clip_span_seconds": (sample_count - 1) / 2.0,
                "evidence_groups": mapped["evidence_temporal_groups"],
                "track_overlapping_groups": excluded,
                "mapped_mask_frame_count": 1,
                "available_object_mask_frame_count": len(mask_ids),
                "rationale_mask_anchor": {key: value for key, value in rationale_anchor.items()
                                          if key != "rationale_line"},
                "bbox_to_qwen_spatiotemporal_mapping": spatial_mapping,
                "clip_sampling": clip_sampling,
                "mask_to_sample_error_max_seconds": max(mapped_errors),
                "mask_to_sample_error_median_seconds": statistics.median(mapped_errors),
                "sampled_track_evidence_group_count": len(
                    set(mapped["evidence_temporal_groups"]) & set(excluded)),
            })
            alignment = validate_sampled_evidence_alignment(
                mapping=mapped, track_groups=excluded,
                maximum_error_seconds=1.0)
            temporal_steps = sample_count // int(video_processor.temporal_patch_size)
            controls = matched_control_groups(temporal_step_count=temporal_steps,
                evidence_groups=mapped["evidence_temporal_groups"],
                excluded_groups=excluded, seed="minerva-eg-anchor-pilot-v1",
                record_token=selected["row_key"])
            if any(type(group) is not int or not 0 <= group < pair_grid[0] * temporal_steps
                   for group in mapped["evidence_temporal_groups"] + controls):
                raise ValueError("temporal_group_out_of_processor_grid")
            pts_drift = max(abs(sample["time_seconds"]
                - times[sample["decoded_frame_index"]]) for sample in sampled)
            record.update({"status": "pass", "frame_count": frame_count,
                "time_map_rows": len(times), "sampled_frame_count": len(frames),
                "sample_pts_max_abs_drift_seconds": pts_drift,
                "video_md5": video_md5, "video_sha256": digest_file(video),
                "time_map_md5": map_md5, "time_map_sha256": digest_file(time_map),
                "evidence_groups": mapped["evidence_temporal_groups"],
                "track_overlapping_groups": excluded,
                "sampled_evidence_alignment": alignment,
                "control_groups": controls, "mapped_mask_frame_count": 1,
                "available_object_mask_frame_count": len(mask_ids),
                "rationale_mask_anchor": {key: value for key, value in rationale_anchor.items()
                                          if key != "rationale_line"},
                "bbox_to_qwen_spatiotemporal_mapping": spatial_mapping,
                "clip_sampling": clip_sampling})
            del frames, sampled
        except Exception as exc:
            record.update({"status": "fail",
                "failure_code": str(exc) if str(exc) else type(exc).__name__})
        records.append(record)
        print(json.dumps({"video_token": hashlib.sha256(video_id.encode()).hexdigest()[:12],
            "status": record["status"], "frame_count": record.get("frame_count"),
            "evidence_groups": len(record.get("evidence_groups", [])),
            "failure_code": record.get("failure_code")}), flush=True)
    payload = {"schema_version": 1,
        "status": "completed_frozen_cohort_cpu_preflight",
        "requested_rows": len(cohort.get("selection", [])),
        "effective_rows": sum(record["status"] == "pass" for record in records),
        "records": records,
        "cohort_sha256": hashlib.sha256(cohort_raw).hexdigest(),
        "manifest_sha256": manifest_sha256,
        "preflight_source_sha256": digest_file(Path(__file__)),
        "decoder_source_sha256": digest_file(ROOT / "scripts/run_minerva_ego_anchor_gpu_pilot.py"),
        "generated_runner_source_sha256": digest_file(ROOT / "scripts/run_minerva_ego_generated_grounding_pilot.py"),
        "bbox_mapper_source_sha256": digest_file(ROOT / "scripts/minerva_ego_anchor_utils.py"),
        "qwen_video_processor": {"transformers_version": __import__("transformers").__version__,
            "min_pixels": pixel_cap, "max_pixels": pixel_cap,
            "patch_size": int(video_processor.patch_size),
            "merge_size": int(video_processor.merge_size),
            "temporal_patch_size": int(video_processor.temporal_patch_size)},
        "elapsed_seconds": time.monotonic() - started,
        "human_review_performed": False,
        "rights_status": "unverified; local-only"}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({"receipt": str(args.output),
        "requested": payload["requested_rows"], "effective": payload["effective_rows"],
        "elapsed_seconds": round(payload["elapsed_seconds"], 2)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
