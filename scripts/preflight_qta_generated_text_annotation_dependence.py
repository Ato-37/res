#!/usr/bin/env python3
"""CPU-only prompt/video-geometry gate for the generated-text QTA proxy."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from freeze_qta_generated_text_candidate_cohort import sha256_file
from qta_multiquery_video_key_intervention import validate_evidence_control_key_sets
from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX
from run_nextgqa_gold_interval_margin_pilot import (
    MODEL_REVISION, UNIFORM_TOLERANCE,
)
from run_nextgqa_validation_qta_annotation_proxy import temporal_support_end


LOCK_NAME = "qta_generated_text_candidate_cohort_local_20261019_01.json"
FRAMES = 16
SOURCE_FILES = (
    "scripts/preflight_qta_generated_text_annotation_dependence.py",
    "scripts/run_qta_generated_text_annotation_dependence.py",
    "scripts/freeze_qta_generated_text_candidate_cohort.py",
    "scripts/qta_multiquery_video_key_intervention.py",
    "scripts/generate_nextgqa_claim_review_pilot.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/qwen_question_temporal_attention_selector.py",
    "scripts/qwen_review_interval_token_groups.py",
    "scripts/qta_attention_forward_capture.py",
    "scripts/qta_attention_capture_contract.py",
    "scripts/qta_attention_capture_sidecars.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/run_nextgqa_validation_qta_annotation_proxy.py",
)
DATA_FILES = ("test.csv", "gsub_test.json", "map_vid_vidorID.json", "frame2time_test.json")


def _read_private(path: Path, private_root: Path) -> tuple[dict, str]:
    root = private_root.resolve(strict=True)
    if stat.S_IMODE(root.stat().st_mode) != 0o700 or path.resolve(strict=True).parent != root:
        raise ValueError("private_input_root_or_path_invalid")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("private_input_permissions_invalid")
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def build_text(processor, row: dict) -> tuple[str, list[dict]]:
    choices = [f"{chr(65 + i)}. {str(row[f'a{i}']).strip()}" for i in range(5)]
    question = str(row["question"]).strip()
    prompt = question + "\nOptions:\n" + "\n".join(choices) + PROMPT_SUFFIX
    messages = [{"role": "user", "content": [
        {"type": "video", "video": None}, {"type": "text", "text": prompt}]}]
    # Callers insert decoded frames and exact sample FPS before rendering.
    return prompt, messages


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def prepare_candidate_geometry(*, dataset: Path, video_root: Path, processor,
                               candidate: dict, rows: dict, gsub: dict) -> dict:
    from qwen_vl_utils import process_vision_info
    from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
    from qwen_question_temporal_attention_selector import map_question_token_positions
    from qwen_review_interval_token_groups import map_review_intervals_to_qwen_groups

    with (dataset / "test.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = {(str(row["video_id"]), str(row["qid"])): row
                for row in csv.DictReader(stream)}
    gsub = json.loads((dataset / "gsub_test.json").read_text(encoding="utf-8"))
    vid, qid = candidate["video_id"], candidate["qid"]
    row = rows.get((vid, qid))
    annotation = gsub.get(vid)
    if row is None or not isinstance(annotation, dict):
        raise ValueError("candidate_dataset_join_invalid")
    spans = annotation.get("location", {}).get(qid)
    duration = annotation.get("duration")
    if (not isinstance(spans, list) or not spans
            or isinstance(duration, bool) or not isinstance(duration, (int, float))):
        raise ValueError("candidate_annotation_invalid")
    media_path = video_root / candidate["video_relative_path"]
    if (not media_path.is_file() or media_path.is_symlink()
            or sha256_file(media_path) != candidate["video_sha256"]):
        raise ValueError("candidate_media_binding_invalid")
    frames, decode = decode_sampled_video(media_path, num_frames=FRAMES,
        uniform_tolerance_seconds=UNIFORM_TOLERANCE)
    verify_sampled_frame_pixels(frames, decode)
    fps = decode["effective_sample_fps"]
    prompt, messages = build_text(processor, row)
    messages[0]["content"][0]["video"] = frames
    messages[0]["content"][0]["sample_fps"] = fps[0] / fps[1]
    rendered = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or len(videos) != 1:
        raise ValueError("qwen_video_utility_contract_invalid")
    pixels, metadata = videos[0]
    if (int(pixels.shape[0]) != FRAMES
            or metadata.get("frames_indices") != list(range(FRAMES))
            or abs(float(metadata.get("fps", 0)) - fps[0] / fps[1]) > 1e-7):
        raise ValueError("qwen_video_frame_grid_invalid")
    batch = processor(text=[rendered], images=images, videos=[pixels],
        video_metadata=[metadata], do_sample_frames=False,
        cap_pixels_per_frame=True, padding=True, return_tensors="pt")
    grid = [int(x) for x in batch["video_grid_thw"][0].tolist()]
    pts = [Fraction(*record["relative_presentation_time"])
           for record in decode["sampled_frames"]]
    support_end = temporal_support_end(Fraction(str(duration)), pts)
    mapped = map_review_intervals_to_qwen_groups(
        input_ids=batch["input_ids"][0].tolist(),
        video_token_id=int(processor.video_token_id), video_grid_thw=grid,
        spatial_merge_size=int(processor.image_processor.merge_size),
        temporal_patch_size=int(processor.video_processor.temporal_patch_size),
        sampled_pts_seconds=pts, video_duration_seconds=support_end,
        annotation_duration_seconds=duration,
        reviewed_intervals_seconds=spans,
        interval_boundary_guard_seconds=0,
        require_processor_uniform_grid_assignment=True,
        evidence_selection_policy="all_annotations")
    budget = validate_evidence_control_key_sets(
        evidence_positions=tuple(mapped["evidence_token_positions"]),
        control_positions=tuple(mapped["control_token_positions"]))
    qpos = map_question_token_positions(tokenizer=processor.tokenizer,
        rendered_text=rendered, question_text=str(row["question"]).strip(),
        processed_input_ids=batch["input_ids"],
        video_token_id=int(processor.video_token_id))
    fingerprint = _canonical_hash({
        "candidate_video_sha256": candidate["video_sha256"],
        "qid_sha256": hashlib.sha256(qid.encode()).hexdigest(),
        "question_sha256": hashlib.sha256(str(row["question"]).strip().encode()).hexdigest(),
        "choices_sha256": _canonical_hash([str(row[f"a{i}"]).strip() for i in range(5)]),
        "input_ids_sha256": hashlib.sha256(batch["input_ids"].cpu().contiguous().numpy().tobytes()).hexdigest(),
        "video_grid_thw": grid, "sampled_pts": [str(x) for x in pts],
        "evidence_groups": mapped["evidence_group_indices"],
        "control_groups": mapped["control_group_indices"],
        "evidence_key_count": budget["evidence_key_count"],
        "question_positions": qpos})
    return {"row": row, "rendered": rendered, "batch": batch, "frames": frames,
        "decode": decode, "mapped": mapped, "question_positions": qpos,
        "geometry_sha256": fingerprint, "evidence_control_key_budget": budget,
        "video_grid_thw": grid, "sampled_pts": pts, "interval_count": len(spans)}


def preflight(*, repo: Path, requested_n: int) -> dict:
    private = repo / "data/private_nextgqa"
    dataset = repo / "NExT-GQA/datasets/nextgqa"
    video_root = Path.home() / "nextgqa_sample/NExTVideo"
    lock, lock_sha = _read_private(private / LOCK_NAME, private)
    if (lock.get("schema_version") != 1
            or lock.get("status") != "candidate_order_frozen_geometry_not_preflighted_not_authorizing"
            or not isinstance(lock.get("candidates"), list)):
        raise ValueError("candidate_lock_contract_invalid")
    from transformers import AutoProcessor
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        raise ValueError("pinned_processor_snapshot_missing")
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)

    with (dataset / "test.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = {(str(row["video_id"]), str(row["qid"])): row
                for row in csv.DictReader(stream)}
    gsub = json.loads((dataset / "gsub_test.json").read_text(encoding="utf-8"))
    plans = []
    failures: dict[str, int] = {}
    attempted = 0
    for candidate in lock["candidates"]:
        if len(plans) >= requested_n:
            break
        attempted += 1
        try:
            prepared = prepare_candidate_geometry(dataset=dataset, video_root=video_root,
                processor=processor, candidate=candidate, rows=rows, gsub=gsub)
            mapped = prepared["mapped"]
            plans.append({"video_id": candidate["video_id"], "qid": candidate["qid"],
                "video_relative_path": candidate["video_relative_path"],
                "video_sha256": candidate["video_sha256"],
                "geometry_sha256": prepared["geometry_sha256"],
                "video_grid_thw": prepared["video_grid_thw"],
                "temporal_group_count": len(mapped["groups"]),
                "evidence_group_count": len(mapped["evidence_group_indices"]),
                "control_group_count": len(mapped["control_group_indices"]),
                "visual_key_count_each": prepared["evidence_control_key_budget"]["evidence_key_count"],
                "question_token_count": len(prepared["question_positions"]),
                "annotation_interval_count": prepared["interval_count"]})
            del prepared
        except Exception as exc:
            name = type(exc).__name__ + ":" + (str(exc).split(":", 1)[0][:100])
            failures[name] = failures.get(name, 0) + 1
    if len(plans) != requested_n:
        raise ValueError(f"geometry_candidate_count_insufficient:{len(plans)}")
    sources = {name: sha256_file(repo / name) for name in SOURCE_FILES}
    inputs = {name: sha256_file(dataset / name) for name in DATA_FILES}
    return {"schema_version": 1,
        "status": "geometry_preflight_passed_not_scoring_authorization",
        "scope": "exploratory generated-text benchmark-annotation dependence proxy",
        "requested_n": requested_n, "effective_n": len(plans),
        "candidate_order_sha256": lock_sha,
        "candidate_count_attempted_until_n": attempted,
        "geometry_failures_by_type": failures,
        "selected_candidates": plans,
        "dataset_sha256": inputs, "source_sha256": sources,
        "model_revision": MODEL_REVISION, "frame_count": FRAMES,
        "uniform_tolerance_seconds": [UNIFORM_TOLERANCE.numerator, UNIFORM_TOLERANCE.denominator],
        "selection_rule": "first requested_n candidates in frozen hash order passing outcome-blind decode, processor geometry and exact equal-key control checks",
        "execution_authorized": False, "scoring_authorized": False,
        "direct_faithfulness": False, "faithfulness_result": False,
        "rights_status": "unverified_local_only"}


def write_private(path: Path, payload: dict, private_root: Path) -> None:
    root = private_root.resolve(strict=True)
    if stat.S_IMODE(root.stat().st_mode) != 0o700 or path.resolve().parent != root or path.exists():
        raise ValueError("preflight_output_path_invalid")
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(fd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(1, 3, 4), default=3)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    repo = Path(__file__).resolve().parents[1]
    private = repo / "data/private_nextgqa"
    output = private / args.output_name
    if Path(args.output_name).name != args.output_name:
        parser.error("output_name_must_be_basename")
    started = time.monotonic()
    try:
        payload = preflight(repo=repo, requested_n=args.n)
        payload["elapsed_seconds"] = time.monotonic() - started
        write_private(output, payload, private)
    except Exception as exc:
        parser.error(f"geometry_preflight_failed:{type(exc).__name__}:{str(exc).split(':',1)[0]}")
    print(json.dumps({"status": payload["status"], "requested_n": args.n,
        "effective_n": payload["effective_n"],
        "candidate_count_attempted": payload["candidate_count_attempted_until_n"],
        "geometry_failure_types": payload["geometry_failures_by_type"],
        "private_output": args.output_name, "elapsed_seconds": payload["elapsed_seconds"]},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
