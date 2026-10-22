#!/usr/bin/env python3
"""Screen Minerva rows for timestamp-bin mask anchors using frame_number/30 only.

This is a metadata-only candidate freezer. It does not inspect videos or time
maps and its output is not eligible for inference until an official-media CPU
preflight confirms the proxy-selected frame against source PTS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from minerva_ego_anchor_utils import (
    parse_rationale_time_intervals,
    supported_rationale_lines,
    timestamp_bins_overlap_segment,
)

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"
SELECTION_SEED = "minerva-metadata-proxy-anchor-cohort-v2-seed-20261022"
FRAME_RATE_PROXY = 30.0
BOUNDARY_MARGIN_SECONDS = 0.05
DEFAULT_GPU_EXPOSED_VIDEO_IDS = frozenset({
    "P03-20240216-084005",
    "P03-20240217-210126",
    "P04-20240413-145245",
    "P04-20240413-151722",
    "P05-20240425-125459",
    "P06-20240510-140459",
    "P07-20240529-131737",
    "P09-20240623-120359",
    "P09-20240623-154547",
})


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _rank(kind: str, value: str, seed: str = SELECTION_SEED) -> str:
    return _sha(f"{seed}|{kind}|{value}".encode("utf-8"))


def _proxy_anchor_for_row(row: dict, object_id: str, tracks: list[dict],
                          masks: dict, *, boundary_margin_seconds: float
                          ) -> dict | None:
    object_tracks = [track.get("time_segment") for track in tracks
                     if isinstance(track, dict) and track.get("time_segment")]
    lines = supported_rationale_lines(
        reasoning=row.get("reasoning", ""), object_tracks={object_id: object_tracks})
    if not lines:
        return None
    line = lines[0]
    intervals = parse_rationale_time_intervals(line)
    candidates = []
    for left, right in intervals:
        safe_left = float(left) + boundary_margin_seconds
        safe_right = float(right) - boundary_margin_seconds
        if safe_right <= safe_left:
            continue
        for track in tracks:
            if not isinstance(track, dict):
                continue
            segment = track.get("time_segment")
            if not segment or not timestamp_bins_overlap_segment([(left, right)], segment):
                continue
            for mask_id in track.get("masks", []):
                mask = masks.get(mask_id)
                frame_number = mask.get("frame_number") if isinstance(mask, dict) else None
                if type(frame_number) is not int or frame_number < 0:
                    continue
                proxy_seconds = frame_number / FRAME_RATE_PROXY
                # Preserve the existing anchor helper's tight track-boundary
                # tolerance while adding a margin from the rationale-bin edge.
                if (safe_left <= proxy_seconds < safe_right
                        and float(segment[0]) - 0.05 <= proxy_seconds
                        <= float(segment[1]) + 0.05):
                    candidates.append({
                        "frame_number": frame_number,
                        "proxy_seconds": proxy_seconds,
                        "interval_seconds": [left, right],
                        "interval_center_error_seconds": abs(
                            proxy_seconds - (left + right) / 2.0),
                        "support_line_sha256": _sha(line.encode("utf-8")),
                        "mask_id_sha256": _sha(str(mask_id).encode("utf-8")),
                    })
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (
        candidate["interval_center_error_seconds"],
        candidate["frame_number"], candidate["mask_id_sha256"]))


def discover_gpu_exposures(private_dir: Path = PRIVATE) -> dict[str, list[dict[str, str]]]:
    """Find locally recorded one-video model outputs and map row hashes to IDs."""
    row_hash_to_videos: dict[str, set[str]] = {}
    payloads: list[tuple[Path, dict, bytes]] = []
    for path in sorted(private_dir.glob("*.json")):
        try:
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                    or info.st_size > 50_000_000):
                continue
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payloads.append((path, payload, raw))
        selection = payload.get("selection")
        if isinstance(selection, list):
            for item in selection:
                if (isinstance(item, dict) and isinstance(item.get("row_key_sha256"), str)
                        and isinstance(item.get("video_id"), str)):
                    row_hash_to_videos.setdefault(item["row_key_sha256"], set()).add(
                        item["video_id"])
        if (isinstance(payload.get("row_key_sha256"), str)
                and isinstance(payload.get("video_id"), str)):
            row_hash_to_videos.setdefault(payload["row_key_sha256"], set()).add(
                payload["video_id"])

    exposures: dict[str, list[dict[str, str]]] = {}
    for path, payload, raw in payloads:
        status = payload.get("status", "")
        # Single-video inference outputs, including failed attempts, count as
        # exposure. CPU preflights are cohort-level and do not match this status.
        if not isinstance(status, str) or "single_video" not in status:
            continue
        video_ids = set()
        if isinstance(payload.get("video_id"), str):
            video_ids.add(payload["video_id"])
        row_hash = payload.get("row_key_sha256")
        if isinstance(row_hash, str):
            video_ids.update(row_hash_to_videos.get(row_hash, set()))
        for video_id in video_ids:
            exposures.setdefault(video_id, []).append({
                "artifact": path.name, "artifact_sha256": _sha(raw),
                "status": status,
            })
    return {video_id: sorted(rows, key=lambda row: row["artifact"])
            for video_id, rows in sorted(exposures.items())}


def select_metadata_candidates(rows: list[dict], assoc_by_video: dict,
                               masks_by_video: dict, *, requested_rows: int = 3,
                               excluded_video_ids: set[str] | None = None,
                               boundary_margin_seconds: float = BOUNDARY_MARGIN_SECONDS,
                               seed: str = SELECTION_SEED) -> dict:
    """Count eligibility, then freeze distinct videos using only annotations."""
    if (type(requested_rows) is not int or requested_rows < 1
            or requested_rows > 3):
        raise ValueError("requested_rows_must_be_between_1_and_3")
    if (isinstance(boundary_margin_seconds, bool)
            or not isinstance(boundary_margin_seconds, (int, float))
            or not 0 <= boundary_margin_seconds < 0.5):
        raise ValueError("boundary_margin_invalid")
    if excluded_video_ids is None:
        excluded_video_ids = set()
    if (not isinstance(excluded_video_ids, set)
            or any(not isinstance(video_id, str) or not video_id
                   for video_id in excluded_video_ids)):
        raise ValueError("excluded_video_ids_invalid")

    counts = {
        "all_rows": len(rows),
        "single_reference_rows": 0,
        "exact_name_join_rows": 0,
        "action_recognition_non_counterfactual_rows": 0,
        "single_word_object_rows": 0,
        "supported_time_line_rows": 0,
        "mask_join_rows": 0,
        "proxy_interval_anchor_rows": 0,
        "eligible_rows_before_exposure_exclusion": 0,
        "exposed_rows_excluded": 0,
        "eligible_rows_after_exposure_exclusion": 0,
    }
    candidates = []
    pre_exposure_video_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        refs = row.get("referenced_object_masks") or []
        if len(refs) != 1:
            continue
        counts["single_reference_rows"] += 1
        try:
            display_name, object_id = refs[0].rsplit("-", 1)
        except (AttributeError, ValueError):
            continue
        video_id = row.get("video_id")
        assoc = assoc_by_video.get(video_id, {})
        obj = assoc.get(object_id) if isinstance(assoc, dict) else None
        if (not isinstance(obj, dict)
                or display_name.casefold() != str(obj.get("name", "")).casefold()):
            continue
        counts["exact_name_join_rows"] += 1
        tags = set(row.get("question_type") or [])
        if "Action Recognition" not in tags or "Counterfactual" in tags:
            continue
        counts["action_recognition_non_counterfactual_rows"] += 1
        if len(display_name.split()) != 1:
            continue
        counts["single_word_object_rows"] += 1
        tracks = [track for track in obj.get("tracks", [])
                  if isinstance(track, dict) and track.get("time_segment")]
        if not tracks:
            continue
        lines = supported_rationale_lines(reasoning=row.get("reasoning", ""),
            object_tracks={object_id: [track["time_segment"] for track in tracks]})
        if not lines:
            continue
        counts["supported_time_line_rows"] += 1
        mask_ids = [mask_id for track in tracks for mask_id in track.get("masks", [])]
        masks = masks_by_video.get(video_id, {})
        if (not mask_ids or any(not isinstance(masks.get(mask_id), dict)
                                for mask_id in mask_ids)):
            continue
        counts["mask_join_rows"] += 1
        if not isinstance(row.get("key"), str) or not row["key"]:
            continue
        anchor = _proxy_anchor_for_row(row, object_id, tracks, masks,
            boundary_margin_seconds=float(boundary_margin_seconds))
        if anchor is None:
            continue
        counts["proxy_interval_anchor_rows"] += 1
        counts["eligible_rows_before_exposure_exclusion"] += 1
        pre_exposure_video_ids.add(video_id)
        if video_id in excluded_video_ids:
            counts["exposed_rows_excluded"] += 1
            continue
        counts["eligible_rows_after_exposure_exclusion"] += 1
        candidates.append({"video_id": video_id, "row_key": row["key"],
            "row_key_sha256": _sha(row["key"].encode("utf-8")),
            "object_name": display_name, "object_id_sha256": _sha(object_id.encode()),
            "proxy_anchor": anchor})

    # Select a stable representative row per distinct video before ranking
    # videos. No inference outcome or media content enters these hashes.
    per_video: dict[str, dict] = {}
    for item in candidates:
        video_id = item["video_id"]
        current = per_video.get(video_id)
        rank = _rank("row", item["row_key"], seed)
        if current is None or (rank, item["row_key"]) < (
                _rank("row", current["row_key"], seed), current["row_key"]):
            per_video[video_id] = item
    counts["eligible_distinct_videos_before_exposure_exclusion"] = len(
        pre_exposure_video_ids)
    counts["eligible_distinct_videos_after_exposure_exclusion"] = len(per_video)
    if len(per_video) < requested_rows:
        raise ValueError("eligible_distinct_video_count_below_requested_rows")
    ordered_videos = sorted(per_video,
        key=lambda video_id: (_rank("video", video_id, seed), video_id))
    participants = {video_id.split("-", 1)[0] for video_id in ordered_videos}
    counts["eligible_distinct_participants_after_exposure_exclusion"] = len(participants)
    if len(participants) < requested_rows:
        raise ValueError("eligible_distinct_participant_count_below_requested_rows")
    selection, chosen_participants = [], set()
    for video_id in ordered_videos:
        participant = video_id.split("-", 1)[0]
        if participant in chosen_participants:
            continue
        selection.append(per_video[video_id])
        chosen_participants.add(participant)
        if len(selection) == requested_rows:
            break
    return {
        "selection_seed": seed,
        "frame_rate_proxy_fps": FRAME_RATE_PROXY,
        "rationale_interval_boundary_margin_seconds": float(boundary_margin_seconds),
        "candidate_counts_before_selection": counts,
        "eligible_candidate_rows_before_exposure_exclusion":
            counts["eligible_rows_before_exposure_exclusion"],
        "eligible_candidate_rows_after_exposure_exclusion":
            counts["eligible_rows_after_exposure_exclusion"],
        "eligible_distinct_videos": len(per_video),
        "selection_diversity_rule": "one_video_per_participant_prefix",
        "requested_rows": requested_rows,
        "selection": selection,
        "temporal_proxy_caveat": (
            "Mask times are approximated as frame_number/30 fps. This metadata screen "
            "does not read official time maps or video, and does not establish that the "
            "selected frame PTS is correct. Every selected row must pass official "
            "time-map/media CPU preflight before inference; retain failures without replacement."),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minerva-json", type=Path,
        default=Path("/tmp/minerva_ego_20260205.json"))
    parser.add_argument("--assoc-info", type=Path,
        default=Path("/tmp/hd_epic_assoc_info.json"))
    parser.add_argument("--mask-info", type=Path,
        default=Path("/tmp/hd_epic_mask_info.json"))
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--exclude-video-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (not PRIVATE.is_dir() or PRIVATE.is_symlink()
            or stat.S_IMODE(PRIVATE.stat().st_mode) != 0o700
            or args.output.parent.resolve() != PRIVATE.resolve()
            or args.output.exists() or args.output.is_symlink()):
        raise ValueError("private_output_contract_failed")
    source_paths = {"minerva_json": args.minerva_json,
                    "assoc_info": args.assoc_info, "mask_info": args.mask_info}
    source_raw = {}
    for name, path in source_paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError("annotation_source_file_contract_invalid")
        source_raw[name] = path.read_bytes()
    rows = json.loads(source_raw["minerva_json"])
    assoc = json.loads(source_raw["assoc_info"])
    masks = json.loads(source_raw["mask_info"])
    exposure_records = discover_gpu_exposures(PRIVATE)
    excluded = set(DEFAULT_GPU_EXPOSED_VIDEO_IDS) | set(exposure_records)
    excluded.update(args.exclude_video_id)
    chosen = select_metadata_candidates(rows, assoc, masks,
        requested_rows=args.rows, excluded_video_ids=excluded)
    exposure_audit = []
    for video_id in sorted(excluded):
        exposure_audit.append({"video_id": video_id,
            "source": "local_single_video_output" if video_id in exposure_records
                else "prior_exposure_default_exclusion" if video_id in DEFAULT_GPU_EXPOSED_VIDEO_IDS
                else "explicit_cli_exclusion",
            "artifacts": exposure_records.get(video_id, [])})
    payload = {"schema_version": 2,
        "status": "frozen_pre_outcome_exploratory_cohort",
        "candidate_preflight_status": "metadata_only_proxy_not_preflighted",
        "dataset": "Minerva-Ego / HD-EPIC",
        "rights_status": "unverified; local-only per user authorization",
        "exposure_exclusion_video_ids": sorted(excluded),
        "exposure_audit": exposure_audit,
        "exposure_audit_sha256": _sha(json.dumps(exposure_audit,
            sort_keys=True, separators=(",", ":")).encode("utf-8")),
        **chosen,
        "source_sha256": {name: _sha(raw) for name, raw in source_raw.items()},
        "selector_sha256": _sha(Path(__file__).read_bytes()),
        "anchor_utils_sha256": _sha(Path(__file__).with_name(
            "minerva_ego_anchor_utils.py").read_bytes()),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False,
        allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            count = os.write(fd, view)
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"status": payload["status"],
        "requested_rows": args.rows,
        "candidate_counts_before_selection": payload["candidate_counts_before_selection"],
        "eligible_distinct_videos": payload["eligible_distinct_videos"],
        "selected_count": len(payload["selection"]),
        "selected_video_ids_private": [row["video_id"] for row in payload["selection"]],
        "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
