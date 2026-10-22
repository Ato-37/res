#!/usr/bin/env python3
"""Build a private frame-review page from an exact generated-claim packet.

Only valid, structured claims are shown. Failed rows remain in the source
manifest as attrition; this tool never replaces them or scores faithfulness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from build_test_split_claim_support_frame_review import (
    FRAME_COUNT, PRIVATE, ROOT, _safe_write, frame_jpeg_data_url, render,
)
from freeze_temporal_order_claim_cohort import _private_json_sha256
from load_frozen_claim_packet_samples import reconstruct_samples


def _read_private(path: Path) -> tuple[bytes, dict]:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError("private_input_permissions_invalid")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private_input_json_invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("private_input_json_shape_invalid")
    return raw, payload


def _inside_private(relative: Path) -> Path:
    if relative.is_absolute() or not relative.is_relative_to(Path("data/private_nextgqa")):
        raise ValueError("private_input_path_invalid")
    path = ROOT
    for part in relative.parts:
        path = path / part
        try:
            if stat.S_ISLNK(path.lstat().st_mode):
                raise ValueError("private_input_path_invalid")
        except OSError as exc:
            raise ValueError("private_input_path_invalid") from exc
    path = path.resolve(strict=True)
    if not path.is_relative_to(PRIVATE.resolve(strict=True)):
        raise ValueError("private_input_path_invalid")
    return path


def _private_output(relative: Path) -> Path:
    """Resolve an output under PRIVATE and reject traversal/symlink components."""
    if relative.is_absolute() or not relative.is_relative_to(Path("data/private_nextgqa")):
        raise ValueError("review_output_or_video_root_invalid")
    path = ROOT
    for part in relative.parts:
        path = path / part
        try:
            if stat.S_ISLNK(path.lstat().st_mode):
                raise ValueError("review_output_or_video_root_invalid")
        except FileNotFoundError:
            # Output itself may not exist; its containing directory must.
            if path != ROOT / relative:
                raise ValueError("review_output_or_video_root_invalid")
        except OSError as exc:
            raise ValueError("review_output_or_video_root_invalid") from exc
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(PRIVATE.resolve(strict=True)):
        raise ValueError("review_output_or_video_root_invalid")
    if not resolved.parent.is_dir() or not resolved.parent.is_relative_to(PRIVATE.resolve(strict=True)):
        raise ValueError("review_output_or_video_root_invalid")
    return resolved


def _validate_lock_pair_binding(lock: dict, pair_filter_sha256: str) -> None:
    if (type(lock.get("schema_version")) is not int or lock.get("schema_version") != 1
            or lock.get("status") != "frozen_before_claim_generation_and_scoring"
            or lock.get("selected_pair_filter_sha256") != pair_filter_sha256
            or type(lock.get("local_scored_claim_overlap_count")) is not int
            or lock.get("local_scored_claim_overlap_count") != 0):
        raise ValueError("packet_lock_input_binding_invalid")


def _validate_structured_rows(packet: dict, records: list) -> int:
    if (type(packet.get("fully_structured_claim_records")) is not int
            or any(not isinstance(row, dict) for row in records)):
        raise ValueError("packet_structured_count_invalid")
    observed = sum(isinstance(row.get("generated_parsed"), dict)
                   and row["generated_parsed"].get("valid") is True for row in records)
    if (packet.get("fully_structured_claim_records") != observed
            or (packet.get("status") in {"complete", "passed"}
                and observed != len(records))
            or (packet.get("status") == "partial" and observed >= len(records))):
        raise ValueError("packet_structured_count_invalid")
    return observed


def build(*, packet_rel: Path, lock_rel: Path, pair_filter_rel: Path,
          exclusion_rel: Path, video_root_rel: Path, output_rel: Path) -> dict:
    packet_path = _inside_private(packet_rel)
    lock_path = _inside_private(lock_rel)
    pair_path = _inside_private(pair_filter_rel)
    exclusion_path = _inside_private(exclusion_rel)
    video_root = _inside_private(video_root_rel)
    if not video_root.is_dir() or output_rel.suffix != ".html":
        raise ValueError("review_output_or_video_root_invalid")
    output = _private_output(output_rel)
    if output.exists():
        raise FileExistsError("review_artifact_already_exists")

    bound_samples, replay_binding = reconstruct_samples(
        packet_path=packet_path, lock_path=lock_path,
        pair_filter_path=pair_path, exclusion_path=exclusion_path,
        video_root=video_root, repo_root=ROOT, frame_count=FRAME_COUNT)

    packet_raw, packet = _read_private(packet_path)
    lock_raw, lock = _read_private(lock_path)
    pair_raw, pair_filter = _read_private(pair_path)
    exclusion_raw, exclusions = _read_private(exclusion_path)
    if (hashlib.sha256(packet_raw).hexdigest() != replay_binding["source_packet_sha256"]
            or hashlib.sha256(lock_raw).hexdigest() != replay_binding["source_lock_sha256"]
            or hashlib.sha256(pair_raw).hexdigest() != replay_binding["pair_filter_sha256"]
            or hashlib.sha256(exclusion_raw).hexdigest()
            != replay_binding["exclusion_manifest_sha256"]):
        raise ValueError("packet_review_snapshot_changed")
    if (packet.get("purpose") != "private_local_nextgqa_model_generated_claim_review_packet"
            or packet.get("dataset_split") not in {"val", "test"}
            or packet.get("dataset_split") != lock.get("dataset_split")
            or packet.get("effective_videos") != 3
            or packet.get("requested_videos") != 3
            or packet.get("status") not in {"complete", "partial", "passed"}
            or packet.get("cohort_sha256") != lock.get("cohort_sha256")
            or lock.get("selected_pair_filter_sha256") != hashlib.sha256(pair_raw).hexdigest()
            or packet.get("candidate_pair_filter_sha256") != hashlib.sha256(pair_raw).hexdigest()
            or packet.get("private_exclusion_manifest_sha256") != hashlib.sha256(exclusion_raw).hexdigest()
            or not isinstance(pair_filter.get("pair_sha256"), list)
            or not isinstance(exclusions.get("media_sha256"), list)
            ):
        raise ValueError("packet_lock_input_binding_invalid")
    _validate_lock_pair_binding(lock, hashlib.sha256(pair_raw).hexdigest())
    records = packet.get("records")
    assignments = lock.get("assignments")
    if (not isinstance(records, list) or len(records) != 3
            or not isinstance(assignments, list) or len(assignments) != 3
            or len(pair_filter["pair_sha256"]) != 3):
        raise ValueError("packet_lock_row_count_invalid")

    exclusion_media = set(exclusions["media_sha256"])
    media = [row.get("video_sha256") for row in records]
    qids = [hashlib.sha256(str(row.get("dataset_question_id", "")).encode()).hexdigest()
            for row in records]
    pairs = [row.get("candidate_pair_sha256") for row in records]
    if (media != [row.get("media_sha256") for row in assignments]
            or qids != [row.get("qid_sha256") for row in assignments]
            or pairs != pair_filter["pair_sha256"]
            or len(set(media)) != 3 or set(media) & exclusion_media):
        raise ValueError("packet_order_or_exposure_binding_invalid")

    valid_rows = []
    invalid_rows = 0
    observed_structured = _validate_structured_rows(packet, records)
    source_manifest = {
        "schema_version": 1,
        "purpose": "generated_claim_support_frame_review_source_manifest",
        "packet_sha256": hashlib.sha256(packet_raw).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_raw).hexdigest(),
        "pair_filter_sha256": hashlib.sha256(pair_raw).hexdigest(),
        "exclusion_manifest_sha256": hashlib.sha256(exclusion_raw).hexdigest(),
        "runner_sha256": packet.get("runner_sha256"),
        "model_revision": packet.get("model_revision"),
        "generation_prompt_version": packet.get("generation_prompt_version"),
        "requested_rows": len(records),
        "structured_rows": packet.get("fully_structured_claim_records"),
        "unreviewable_attrition_rows": 0,
        "items": [],
    }
    for index, row in enumerate(records):
        parsed = row.get("generated_parsed")
        if not isinstance(parsed, dict) or parsed.get("valid") is not True:
            invalid_rows += 1
            continue
        claim, question, video_path_text = parsed.get("claim"), row.get("question"), row.get("local_video_path")
        if (not isinstance(claim, str) or not claim.strip()
                or not isinstance(question, str) or not question.strip()
                or not isinstance(video_path_text, str)):
            raise ValueError("valid_packet_row_fields_missing")
        bound = bound_samples[index]
        video_path = Path(video_path_text).resolve(strict=True)
        if (bound["video_path"] != str(video_path)
                or bound["candidate_pair_sha256"] != pairs[index]):
            raise ValueError("packet_video_frame_binding_invalid")
        frames, metadata = bound["frames"], bound["decode"]
        sampled_times = row["sampled_frame_relative_pts_seconds"]
        item_id = hashlib.sha256(
            (str(row["video_sha256"]) + str(row["dataset_question_id"])
             + "generated-claim-frame-review-v1").encode()).hexdigest()[:24]
        valid_rows.append({
            "item_id": item_id,
            "question": question,
            "claim": claim,
            "frames": [{"data": frame_jpeg_data_url(frame), "time": point}
                       for frame, point in zip(frames, sampled_times, strict=True)],
        })
        source_manifest["items"].append({
            "record_index": index,
            "item_id": item_id,
            "video_sha256": row["video_sha256"],
            "qid_sha256": qids[index],
            "candidate_pair_sha256": pairs[index],
        })
    source_manifest["unreviewable_attrition_rows"] = invalid_rows
    if len(valid_rows) < 1 or len(valid_rows) > 3:
        raise ValueError("no_reviewable_claim_rows")
    html = render(valid_rows, provenance=source_manifest,
                  title="動画説明の根拠フレーム確認（生成claim pilot）")
    _safe_write(output, html)
    return {
        "status": "built_private_generated_claim_frame_review",
        "requested_rows": len(records),
        "reviewable_rows": len(valid_rows),
        "unreviewable_attrition_rows": invalid_rows,
        "frames_per_reviewable_row": FRAME_COUNT,
        "artifact": str(output.relative_to(ROOT)),
        "artifact_sha256": hashlib.sha256(html).hexdigest(),
        "artifact_mode": "0600",
        "source_manifest_sha256": _private_json_sha256(source_manifest),
        "direct_faithfulness_measured": False,
        "rights_status": "unverified_local_only",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "lock", "pair-filter", "exclusion-manifest", "video-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = build(
            packet_rel=args.packet, lock_rel=args.lock,
            pair_filter_rel=args.pair_filter,
            exclusion_rel=args.exclusion_manifest,
            video_root_rel=args.video_root, output_rel=args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "generated_claim_review_build_failed")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
