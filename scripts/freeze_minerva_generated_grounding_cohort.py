#!/usr/bin/env python3
"""Freeze a deterministic, local-only Minerva-Ego movement grounding cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from minerva_ego_anchor_utils import supported_rationale_lines

ROOT = Path(__file__).resolve().parents[1]
SEED = "minerva-generated-grounding-singleton-v1"


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _rank(kind: str, value: str) -> str:
    return sha_bytes(f"{SEED}|{kind}|{value}".encode())


def select_cohort(rows: list[dict], assoc_by_video: dict,
                  masks_by_video: dict, requested_rows: int,
                  excluded_video_ids: set[str] | None = None) -> dict:
    if type(requested_rows) is not int or requested_rows < 1:
        raise ValueError("requested_rows_invalid")
    if excluded_video_ids is None:
        excluded_video_ids = set()
    if (not isinstance(excluded_video_ids, set)
            or any(not isinstance(value, str) or not value
                   for value in excluded_video_ids)):
        raise ValueError("excluded_video_ids_invalid")
    eligible = []
    counts = {"all_rows": len(rows), "single_reference": 0,
              "exact_name": 0, "action_recognition_non_counterfactual": 0,
              "supported_time_line": 0, "mask_join": 0,
              "single_word_object": 0}
    for row in rows:
        refs = row.get("referenced_object_masks") or []
        if len(refs) != 1:
            continue
        counts["single_reference"] += 1
        try:
            display_name, object_id = refs[0].rsplit("-", 1)
        except (AttributeError, ValueError):
            continue
        video_id = row.get("video_id")
        obj = assoc_by_video.get(video_id, {}).get(object_id)
        if not isinstance(obj, dict) or display_name.casefold() != str(obj.get("name", "")).casefold():
            continue
        counts["exact_name"] += 1
        tags = set(row.get("question_type") or [])
        if "Counterfactual" in tags or "Action Recognition" not in tags:
            continue
        counts["action_recognition_non_counterfactual"] += 1
        # A one-token object label makes the generated-object anchor test
        # explicit and avoids choosing an unvalidated alias/head noun.
        if len(display_name.split()) != 1:
            continue
        counts["single_word_object"] += 1
        tracks = [item.get("time_segment") for item in obj.get("tracks", [])
                  if isinstance(item, dict) and item.get("time_segment")]
        if not tracks:
            continue
        lines = supported_rationale_lines(
            reasoning=row.get("reasoning", ""), object_tracks={object_id: tracks})
        if not lines:
            continue
        counts["supported_time_line"] += 1
        object_tracks = obj.get("tracks", [])
        mask_ids = [mask_id for track in object_tracks
                    for mask_id in track.get("masks", [])]
        video_masks = masks_by_video.get(video_id, {})
        if not mask_ids or any(mask_id not in video_masks for mask_id in mask_ids):
            continue
        counts["mask_join"] += 1
        if not isinstance(row.get("key"), str) or not row["key"]:
            continue
        eligible.append({"video_id": video_id, "row_key": row["key"],
                         "row_key_sha256": sha_bytes(row["key"].encode()),
                         "object_name": display_name,
                         "support_line_count": len(lines)})

    # Exactly one metadata-selected row per video; then a stable hash order
    # selects the requested number of distinct videos before any inference.
    per_video = {}
    for item in eligible:
        if item["video_id"] in excluded_video_ids:
            continue
        current = per_video.get(item["video_id"])
        rank = _rank("row", item["row_key"])
        if current is None or (rank, item["row_key"]) < (current[0], current[1]["row_key"]):
            per_video[item["video_id"]] = (rank, item)
    video_order = sorted(per_video,
        key=lambda video_id: (_rank("video", video_id), video_id))
    selected = [per_video[video_id][1]
                for video_id in video_order[:requested_rows]]
    if len(selected) != requested_rows:
        raise ValueError("eligible_video_count_below_requested_rows")
    return {"counts": counts, "eligible_rows": len(eligible),
            "eligible_videos": len(per_video), "selection": selected}


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
    private = ROOT / "data" / "private_minerva"
    if (args.output.parent.resolve() != private.resolve()
            or not private.is_dir() or private.stat().st_mode & 0o077
            or args.output.exists()):
        raise ValueError("private_output_contract_failed")
    raw_rows = args.minerva_json.read_bytes()
    raw_assoc = args.assoc_info.read_bytes()
    raw_masks = args.mask_info.read_bytes()
    chosen = select_cohort(json.loads(raw_rows), json.loads(raw_assoc),
                           json.loads(raw_masks), args.rows,
                           excluded_video_ids=set(args.exclude_video_id))
    payload = {"schema_version": 1,
        "status": "frozen_pre_outcome_exploratory_cohort",
        "dataset": "Minerva-Ego / HD-EPIC",
        "rights_status": "unverified; local-only per user authorization",
        "selection_seed": SEED,
        "selection_excluded_video_ids": args.exclude_video_id,
        "eligibility": ["exactly one referenced object", "exact display-name/association join",
            "Action Recognition tag and no Counterfactual tag", "single-word object name",
            "same-line exact-object-ID and rounded timestamp-bin/track overlap",
            "all referenced object mask IDs join to mask metadata"],
        "requested_rows": args.rows, **chosen,
        "source_sha256": {"minerva_json": sha_bytes(raw_rows),
                          "assoc_info": sha_bytes(raw_assoc),
                          "mask_info": sha_bytes(raw_masks)},
        "selector_sha256": sha_bytes(Path(__file__).read_bytes()),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({"status": payload["status"], "requested_rows": args.rows,
        "eligible_rows": payload["eligible_rows"],
        "eligible_videos": payload["eligible_videos"],
        "selected_row_count": len(payload["selection"]),
        "selected_video_tokens": [sha_bytes(x["video_id"].encode())[:12]
                                  for x in payload["selection"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
