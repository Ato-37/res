#!/usr/bin/env python3
"""Exploratory fixed-claim likelihood sensitivity to QA-interval key bias.

This local-only runner scores frozen claim text under six
teacher-forced conditions: QA-interval or equal-budget matched-control black
pixel masking, crossed with no bias, QA-interval +ln(2) key bias, or a disjoint
third-window +ln(2) sham bias. It is a conditional text-sensitivity proxy, not direct
faithfulness, claim truth, or proof that the QA interval supports the claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Sequence

from private_output_paths import validate_private_output
from run_machine_localized_claim_mask_pilot import (
    FRAME_COUNT, MODEL_REVISION, MIN_FREE_VRAM, MIN_HOST_AVAILABLE,
    _host_available_bytes, _sha256,
)
from run_nextgqa_gold_interval_margin_pilot import process_qa_video
from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_BY_VERSION
from run_saved_claim_video_content_dependence import resolve_claim_prompt_suffix
from run_saved_claim_frame_permutation_attention_smoke import _build_prompt
from qta_multiquery_video_key_intervention import (
    append_target_to_unpadded_batch, install_multiquery_video_key_intervention_hooks,
    remove_hooks, score_target_logits_window,
)

PACKET_REL = Path("data/private_nextgqa/claim_likelihood_claims_remaining_20261021_30.json")
LAYERS_1_BASED = tuple(range(5, 17))
LOGIT_BIAS = math.log(2.0)
SEED = 20261021
TEMPORAL_PATCH_SIZE = 2
EXPECTED_GROUP_COUNT = FRAME_COUNT // TEMPORAL_PATCH_SIZE
BRANCHES = (
    "baseline_qa_pixel_mask", "baseline_control_pixel_mask",
    "qa_bias_qa_pixel_mask", "qa_bias_control_pixel_mask",
    "sham_bias_qa_pixel_mask", "sham_bias_control_pixel_mask",
)
SHAM_PROTOCOL = "qa-interval-claim-key-sham-v2-hash-ranked-groups"
SOURCE_REL = (
    "scripts/run_claim_text_qa_interval_key_bias.py",
    "scripts/run_machine_localized_claim_mask_pilot.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/run_nextgqa_real_video_gpu_smoke.py",
    "scripts/qta_multiquery_video_key_intervention.py",
    "scripts/presentation_time_utils.py",
    "scripts/necessity_choice_margin_utils.py",
    "scripts/private_output_paths.py",
    "scripts/generate_nextgqa_claim_review_pilot.py",
    "scripts/run_saved_claim_video_content_dependence.py",
    "scripts/run_saved_claim_frame_permutation_attention_smoke.py",
    "scripts/build_claim_text_qa_internal_bias_16frame_cohort.py",
)
DATA_REL = (
    "NExT-GQA/datasets/nextgqa/test.csv",
    "NExT-GQA/datasets/nextgqa/map_vid_vidorID.json",
    "NExT-GQA/datasets/nextgqa/gsub_test.json",
    "NExT-GQA/datasets/nextgqa/frame2time_test.json",
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_private_json(path: Path, private_root: Path) -> tuple[dict, bytes]:
    private = private_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(private):
        raise ValueError("private_input_path_invalid")
    fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_input_permissions_invalid")
        parts = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            parts.append(block)
    finally:
        os.close(fd)
    raw = b"".join(parts)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("private_json_object_required")
    return value, raw


def validate_packet(payload: Any, *, expected_frame_count: int | None = None,
                    expected_split: str = "test",
                    allow_ineligible_rows: bool = False) -> list[dict]:
    frame_count = payload.get("frames_per_video") if isinstance(payload, dict) else None
    expected_records = payload.get("requested_videos") if isinstance(payload, dict) else None
    structured_records = payload.get("fully_structured_claim_records") if isinstance(payload, dict) else None
    expected_status = ("passed" if structured_records == expected_records else "partial")
    if (not isinstance(payload, dict) or payload.get("status") != "passed"
            and not (allow_ineligible_rows and payload.get("status") == "partial")
            or expected_split not in {"val", "test"}
            or payload.get("dataset_split") != expected_split
            or payload.get("model_revision") != MODEL_REVISION
            or payload.get("generation_prompt_version") != "v2_short"
            or type(frame_count) is not int or frame_count < 4 or frame_count % TEMPORAL_PATCH_SIZE
            or (expected_frame_count is not None and frame_count != expected_frame_count)
            or payload.get("claim_review_completed") is not False
            or type(expected_records) is not int or expected_records < 1
            or payload.get("effective_videos") != expected_records
            or type(structured_records) is not int or not 0 <= structured_records <= expected_records
            or (structured_records != expected_records and not allow_ineligible_rows)
            or payload.get("status") != expected_status
            or (structured_records == expected_records and payload.get("status") != "passed")
            or (structured_records < expected_records and not allow_ineligible_rows)):
        raise ValueError("frozen_claim_packet_contract_invalid")
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != expected_records:
        raise ValueError("frozen_claim_packet_record_count_invalid")
    hashes = []
    valid_rows = []
    valid_claim_count = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("frozen_claim_record_invalid")
        parsed = row.get("generated_parsed") if isinstance(row, dict) else None
        claim = parsed.get("claim") if isinstance(parsed, dict) else None
        pts = row.get("sampled_frame_relative_pts_seconds") if isinstance(row, dict) else None
        gold = row.get("gold_interval_seconds") if isinstance(row, dict) else None
        common_invalid = (not isinstance(parsed, dict)
                or row.get("generation_prompt_version") != "v2_short"
                or not isinstance(row.get("dataset_question_id"), str)
                or not row.get("dataset_question_id")
                or not isinstance(row.get("question"), str) or not row["question"].strip()
                or not isinstance(row.get("choices"), list) or len(row["choices"]) != 5
                or not isinstance(row.get("local_video_path"), str)
                or not isinstance(row.get("video_sha256"), str)
                or len(row["video_sha256"]) != 64
                or not isinstance(pts, list) or len(pts) != frame_count
                or any(isinstance(t, bool) or not isinstance(t, (int, float))
                       or not math.isfinite(float(t)) for t in pts)
                or not isinstance(gold, list) or not gold
                or any(not valid_interval(span) for span in gold))
        if common_invalid:
            raise ValueError("frozen_claim_record_invalid")
        if any(b <= a for a, b in zip(pts, pts[1:])) or abs(float(pts[0])) > 1e-7:
            raise ValueError("frozen_claim_pts_invalid")
        hashes.append(row["video_sha256"])
        claim_valid = (parsed.get("valid") is True and isinstance(claim, str) and bool(claim.strip()))
        if claim_valid:
            valid_claim_count += 1
            if len(gold) == 1:
                valid_rows.append(row)
            elif not allow_ineligible_rows:
                raise ValueError("single_qa_interval_required")
        elif (not allow_ineligible_rows or parsed.get("valid") is not False
              or not isinstance(parsed.get("claim_text_quality_failure"), str)
              or not parsed.get("claim_text_quality_failure")):
            raise ValueError("frozen_claim_record_invalid")
    if (len(set(hashes)) != expected_records or valid_claim_count != structured_records
            or (not allow_ineligible_rows and len(valid_rows) != structured_records)):
        raise ValueError("frozen_claim_media_not_unique")
    return valid_rows


def valid_interval(span: Any) -> bool:
    return (isinstance(span, (list, tuple)) and len(span) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(float(x)) for x in span)
            and 0 <= float(span[0]) < float(span[1]))


def align_interval_and_control_groups(*, sampled_pts: Sequence[float], intervals: list,
                                      temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
                                      group_count: int | None = None,
                                      frame_count: int = FRAME_COUNT) -> dict:
    """Outward-expand QA-covered frames to whole groups; nearest equal run is control."""
    if group_count is None:
        group_count = frame_count // temporal_patch_size if temporal_patch_size > 0 else 0
    if (not isinstance(sampled_pts, (list, tuple)) or len(sampled_pts) != frame_count
            or type(temporal_patch_size) is not int or temporal_patch_size < 1
            or type(group_count) is not int or group_count < 2
            or group_count * temporal_patch_size != len(sampled_pts)
            or any(isinstance(t, bool) or not isinstance(t, (int, float))
                   or not math.isfinite(float(t)) for t in sampled_pts)
            or any(b <= a for a, b in zip(sampled_pts, sampled_pts[1:]))
            or not isinstance(intervals, list) or len(intervals) != 1
            or any(not valid_interval(span) for span in intervals)):
        raise ValueError("interval_group_mapping_inputs_invalid")
    frame_hits = tuple(i for i, t in enumerate(sampled_pts)
                       if any(float(lo) <= float(t) <= float(hi) for lo, hi in intervals))
    if not frame_hits:
        raise ValueError("qa_interval_covers_no_sampled_frame")
    group_start = frame_hits[0] // temporal_patch_size
    group_stop = frame_hits[-1] // temporal_patch_size + 1
    evidence = tuple(range(group_start, group_stop))
    length = len(evidence)
    candidates = [tuple(range(start, start + length))
                  for start in range(group_count - length + 1)
                  if set(range(start, start + length)).isdisjoint(evidence)]
    if not candidates:
        raise ValueError("equal_count_group_control_unavailable")
    center = (evidence[0] + evidence[-1]) / 2
    control = min(candidates, key=lambda values: (
        abs((values[0] + values[-1]) / 2 - center), values[0]))
    if (len(evidence) != len(control) or set(evidence) & set(control)
            or any(i < 0 or i >= group_count for i in (*evidence, *control))):
        raise ValueError("group_control_contract_invalid")
    return {
        "sampled_qa_frame_indices": list(frame_hits),
        "evidence_group_indices": list(evidence),
        "control_group_indices": list(control),
        "evidence_frame_window": [group_start * temporal_patch_size,
                                   group_stop * temporal_patch_size],
        "control_frame_window": [control[0] * temporal_patch_size,
                                  (control[-1] + 1) * temporal_patch_size],
        "resolution_rule": "outward expansion to whole temporal groups; equal-count nearest disjoint contiguous control, lower-start tie",
        "expansion_frames": (group_stop - group_start) * temporal_patch_size - len(frame_hits),
    }


def map_groups_to_visual_keys(*, input_ids: Sequence[int], video_token_id: int,
                              video_grid_thw: Sequence[int], spatial_merge_size: int,
                              temporal_patch_size: int, frame_count: int,
                              evidence_group_indices: Sequence[int],
                              control_group_indices: Sequence[int]) -> dict:
    """Map complete temporal groups to Qwen video-token indices, fail closed."""
    if (len(video_grid_thw) != 3 or any(type(v) is not int or v <= 0 for v in video_grid_thw)
            or frame_count % temporal_patch_size):
        raise ValueError("processor_video_grid_invalid")
    temporal_steps, height, width = video_grid_thw
    if temporal_steps != frame_count // temporal_patch_size or height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("processor_video_grid_frame_contract_invalid")
    positions = tuple(i for i, token in enumerate(input_ids) if token == video_token_id)
    per_group = (height // spatial_merge_size) * (width // spatial_merge_size)
    if (not positions or len(positions) != temporal_steps * per_group
            or positions != tuple(range(positions[0], positions[0] + len(positions)))):
        raise ValueError("processor_video_token_block_invalid")
    groups = tuple(tuple(positions[i * per_group:(i + 1) * per_group])
                   for i in range(temporal_steps))
    evidence_groups = tuple(evidence_group_indices)
    control_groups = tuple(control_group_indices)
    for name, selected in (("evidence", evidence_groups), ("control", control_groups)):
        if (not selected or tuple(sorted(set(selected))) != selected
                or any(type(i) is not int or not 0 <= i < temporal_steps for i in selected)):
            raise ValueError(f"{name}_groups_invalid")
    if (len(evidence_groups) != len(control_groups)
            or set(evidence_groups) & set(control_groups)):
        raise ValueError("evidence_control_group_budget_invalid")
    ev = tuple(sorted(p for i in evidence_groups for p in groups[i]))
    ctl = tuple(sorted(p for i in control_groups for p in groups[i]))
    if not ev or len(ev) != len(ctl) or set(ev) & set(ctl):
        raise ValueError("evidence_control_key_budget_invalid")
    return {"temporal_group_count": temporal_steps,
            "tokens_per_temporal_group": per_group,
            "temporal_groups": groups,
            "evidence_key_positions": ev, "control_key_positions": ctl,
            "equal_key_count": len(ev)}


def select_disjoint_sham_groups(*, group_count: int, group_budget: int,
                                evidence_groups: Sequence[int],
                                control_groups: Sequence[int], row_token: str) -> tuple[int, ...]:
    """Hash-rank an equal-size set of temporal groups disjoint from both masks."""
    evidence, control = set(evidence_groups), set(control_groups)
    if (type(group_count) is not int or group_count < 1
            or type(group_budget) is not int or group_budget < 1
            or not row_token or evidence & control
            or any(type(i) is not int or not 0 <= i < group_count
                   for i in evidence | control)):
        raise ValueError("sham_group_inputs_invalid")
    occupied = evidence | control
    prefix = f"{SHAM_PROTOCOL}\0{row_token}\0"
    candidates = [i for i in range(group_count) if i not in occupied]
    if len(candidates) < group_budget:
        raise ValueError("disjoint_sham_group_set_unavailable")
    ranked = sorted(candidates, key=lambda group: (
        hashlib.sha256((prefix + str(group)).encode("ascii")).digest(), group))
    return tuple(sorted(ranked[:group_budget]))


def compute_text_dependence(branch_means: dict[str, float]) -> dict:
    if (not isinstance(branch_means, dict) or set(branch_means) != set(BRANCHES)
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(float(v)) for v in branch_means.values())):
        raise ValueError("branch_score_set_invalid")
    f_base = float(branch_means["baseline_control_pixel_mask"])
    f_base -= float(branch_means["baseline_qa_pixel_mask"])
    f_qa = float(branch_means["qa_bias_control_pixel_mask"]) - float(branch_means["qa_bias_qa_pixel_mask"])
    f_control = float(branch_means["sham_bias_control_pixel_mask"]) - float(branch_means["sham_bias_qa_pixel_mask"])
    return {"f_baseline_nats": f_base, "f_qa_bias_nats": f_qa,
            "f_sham_bias_nats": f_control,
            "delta_qa_bias_minus_baseline_nats": f_qa - f_base,
            "specificity_qa_minus_sham_bias_nats": f_qa - f_control,
            "score_semantics": "mean conditional logprob of re-tokenized saved claim; control-pixel-mask minus QA-pixel-mask under bias condition"}


def validate_smoke_receipt(smoke: Any, *, input_hashes: dict,
                           source_hashes: dict, model_revision: str,
                           weight_sha256: dict,
                           expected_row: dict) -> None:
    """Fail closed unless a smoke receipt contains a complete, self-consistent run."""
    if (not isinstance(smoke, dict)
            or smoke.get("status") != "completed_exploratory_claim_text_qa_interval_key_bias_smoke"
            or smoke.get("mode") != "smoke"
            or smoke.get("model_revision") != model_revision
            or smoke.get("model_weight_sha256") != weight_sha256
            or smoke.get("input_sha256") != input_hashes
            or smoke.get("source_sha256") != source_hashes
            or smoke.get("requested_records") != 1
            or smoke.get("effective_records") != 1
            or smoke.get("requested_forwards") != len(BRANCHES)
            or smoke.get("attempted_forwards") != len(BRANCHES)
            or smoke.get("completed_forwards") != len(BRANCHES)
            or smoke.get("direct_faithfulness") is not False
            or smoke.get("faithfulness_result") is not False
            or not isinstance(smoke.get("human_review"), dict)
            or smoke["human_review"].get("review_actually_performed") is not False):
        raise ValueError("successful_matching_smoke_receipt_required")
    records = smoke.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("smoke_receipt_record_count_invalid")
    record = records[0]
    if (not isinstance(expected_row, dict)
            or record.get("row_token_sha256") != expected_row.get("row_token_sha256")
            or record.get("source_video_sha256") != expected_row.get("source_video_sha256")
            or record.get("saved_claim_sha256") != expected_row.get("claim_hash")
            or record.get("target_token_ids_sha256") != expected_row.get("target_hash")):
        raise ValueError("smoke_receipt_row_identity_mismatch")
    means = record.get("branch_mean_logprob_nats_per_token") if isinstance(record, dict) else None
    tokens = record.get("branch_token_logprobs_nats") if isinstance(record, dict) else None
    calls = record.get("calls") if isinstance(record, dict) else None
    forward = record.get("forward_count") if isinstance(record, dict) else None
    if (not isinstance(means, dict) or set(means) != set(BRANCHES)
            or not isinstance(tokens, dict) or set(tokens) != set(BRANCHES)
            or not isinstance(calls, dict) or set(calls) != set(BRANCHES)
            or forward != {"attempted": len(BRANCHES), "completed": len(BRANCHES)}):
        raise ValueError("smoke_receipt_branch_set_invalid")
    target_count = record.get("target_token_count")
    if type(target_count) is not int or target_count < 1:
        raise ValueError("smoke_receipt_target_count_invalid")
    for branch in BRANCHES:
        score = means[branch]
        token_scores = tokens[branch]
        call = calls[branch]
        expected_hooks = 0 if branch.startswith("baseline_") else 1
        expected_bias = 0 if expected_hooks == 0 else record.get("qa_bias_key_count")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not isinstance(token_scores, list) or len(token_scores) != target_count
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(float(x)) for x in token_scores)
                or not math.isclose(float(score), sum(map(float, token_scores)) / target_count,
                                    rel_tol=0, abs_tol=1e-6)
                or not isinstance(call, dict)
                or call.get("target_token_count") != target_count
                or call.get("query_token_count") != target_count
                or call.get("biased_key_count") != expected_bias
                or call.get("surviving_bias_key_count") != expected_bias
                or call.get("ablated_key_count") != 0):
            raise ValueError("smoke_receipt_branch_score_invalid")
        hooks = call.get("hook_calls_by_layer")
        expected_layers = {str(layer): expected_hooks for layer in LAYERS_1_BASED}
        if hooks != expected_layers:
            raise ValueError("smoke_receipt_hook_observation_invalid")
    recomputed = compute_text_dependence(means)
    reported = record.get("primary")
    if not isinstance(reported, dict):
        raise ValueError("smoke_receipt_primary_missing")
    for name in ("f_baseline_nats", "f_qa_bias_nats", "f_sham_bias_nats",
                 "delta_qa_bias_minus_baseline_nats", "specificity_qa_minus_sham_bias_nats"):
        if (name not in reported or isinstance(reported[name], bool)
                or not isinstance(reported[name], (int, float))
                or not math.isfinite(float(reported[name]))
                or not math.isclose(float(reported[name]), recomputed[name],
                                    rel_tol=0, abs_tol=1e-7)):
            raise ValueError("smoke_receipt_primary_arithmetic_invalid")


def masked_group_frame_indices(groups: Sequence[int], *, temporal_patch_size: int,
                               frame_count: int) -> tuple[int, ...]:
    if (not groups or type(temporal_patch_size) is not int or temporal_patch_size < 1
            or type(frame_count) is not int or frame_count < 1
            or tuple(sorted(set(groups))) != tuple(groups)
            or any(type(i) is not int or i < 0 or i * temporal_patch_size >= frame_count for i in groups)):
        raise ValueError("masked_group_indices_invalid")
    result = tuple(frame for group in groups
                   for frame in range(group * temporal_patch_size,
                                      min((group + 1) * temporal_patch_size, frame_count)))
    if not result or result[-1] >= frame_count:
        raise ValueError("masked_frame_indices_invalid")
    return result


def _prepare_rows(repo: Path, packet_path: Path, packet: dict, packet_raw: bytes,
                  processor, process_vision_info, *, frame_count: int,
                  allow_ineligible_rows: bool = False):
    import csv
    from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
    from necessity_choice_margin_utils import apply_mask_fill
    from presentation_time_utils import uniform_sampling_rate_from_times
    from run_nextgqa_real_video_gpu_smoke import video_key
    from run_nextgqa_gold_interval_margin_pilot import candidate_pair_sha256

    split = packet.get("dataset_split")
    if split not in {"val", "test"}:
        raise ValueError("packet_dataset_split_invalid")
    dataset = repo / "NExT-GQA/datasets/nextgqa"
    data_rel = tuple(f"NExT-GQA/datasets/nextgqa/{name}" for name in
        (f"{split}.csv", "map_vid_vidorID.json", f"gsub_{split}.json",
         f"frame2time_{split}.json"))
    if set(packet.get("dataset_inputs_sha256", {})) != {Path(x).name for x in data_rel}:
        raise ValueError("packet_dataset_input_manifest_invalid")
    for name, expected in packet["dataset_inputs_sha256"].items():
        path = dataset / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError("packet_dataset_input_digest_mismatch")
    with (dataset / f"{split}.csv").open(newline="", encoding="utf-8-sig") as stream:
        dataset_rows = list(csv.DictReader(stream))
    video_map = json.loads((dataset / "map_vid_vidorID.json").read_text(encoding="utf-8"))
    gsub = json.loads((dataset / f"gsub_{split}.json").read_text(encoding="utf-8"))
    prepared = []
    hashes = {packet_path.relative_to(repo).as_posix(): sha256_bytes(packet_raw)}
    preflight_geometry_exclusions = []
    for position, row in enumerate(validate_packet(packet, expected_split=split,
            allow_ineligible_rows=allow_ineligible_rows), start=1):
        media = Path(row["local_video_path"])
        if media.is_symlink() or _sha256(media) != row["video_sha256"]:
            raise ValueError("source_video_digest_mismatch")
        relative = "/".join(media.parts[-2:])
        video_ids = [str(video_id) for video_id, rel in video_map.items()
                     if video_key(rel) == video_key(relative)
                     and candidate_pair_sha256(str(video_id), str(row["dataset_question_id"]))
                         == row["candidate_pair_sha256"]]
        if len(video_ids) != 1:
            raise ValueError("dataset_video_pair_join_invalid")
        matches = [candidate for candidate in dataset_rows
                   if str(candidate["video_id"]) == video_ids[0]
                   and str(candidate["qid"]) == str(row["dataset_question_id"])]
        if len(matches) != 1:
            raise ValueError("dataset_question_join_invalid")
        dataset_row = matches[0]
        vid, qid = video_ids[0], str(row["dataset_question_id"])
        info = gsub.get(vid)
        loc = info.get("location") if isinstance(info, dict) else None
        raw_spans = loc.get(qid) if isinstance(loc, dict) else None
        try:
            dataset_spans = [[float(span[0]), float(span[1])] for span in raw_spans]
            dataset_duration = float(info["duration"])
        except (TypeError, ValueError, IndexError):
            raise ValueError("dataset_qa_interval_join_invalid") from None
        if (not dataset_spans or not math.isfinite(dataset_duration) or dataset_duration <= 0
                or any(lo < 0 or hi > dataset_duration for lo, hi in dataset_spans)
                or dataset_spans != [[float(x) for x in span]
                for span in row["gold_interval_seconds"]]
                or str(dataset_row["question"]).strip() != row["question"].strip()
                or [str(dataset_row[f"a{i}"]).strip() for i in range(5)] != row["choices"]
                or str(dataset_row["answer"]).strip() != str(row.get("gold_answer", "")).strip()):
            raise ValueError("packet_qa_fields_differ_from_dataset")

        frames, decoded = decode_sampled_video(media, num_frames=frame_count,
            uniform_tolerance_seconds=Fraction(1, 20), include_timing_map=True)
        verify_sampled_frame_pixels(frames, decoded)
        sampled = decoded["sampled_frames"]
        pts = [float(Fraction(*item["relative_presentation_time"])) for item in sampled]
        saved_pts = row["sampled_frame_relative_pts_seconds"]
        if (decoded["source_sha256"] != row["video_sha256"] or len(pts) != frame_count
                or len(saved_pts) != frame_count
                or any(abs(a - float(b)) > 1e-6 for a, b in zip(pts, saved_pts, strict=True))):
            raise ValueError("packet_decoded_pts_mismatch")
        fps_ratio = decoded["effective_sample_fps"]
        fps = Fraction(*fps_ratio)
        uniform_sampling_rate_from_times(tuple(Fraction(str(x)) for x in pts), Fraction(1, 20))
        if len(row["gold_interval_seconds"]) != 1:
            raise ValueError("single_qa_interval_required")
        try:
            mapping = align_interval_and_control_groups(sampled_pts=pts,
                intervals=row["gold_interval_seconds"], frame_count=frame_count)
        except ValueError as exc:
            code = str(exc).splitlines()[0]
            if allow_ineligible_rows and code in {
                    "qa_interval_covers_no_sampled_frame",
                    "equal_count_group_control_unavailable"}:
                preflight_geometry_exclusions.append({
                    "row_token_sha256": sha256_bytes((row["video_sha256"]
                        + str(row["dataset_question_id"])).encode()),
                    "reason_code": code})
                continue
            raise
        prompt_version, prompt_suffix = resolve_claim_prompt_suffix(row, PROMPT_SUFFIX_BY_VERSION)
        if prompt_version != "v2_short":
            raise ValueError("claim_generation_prompt_version_not_frozen_v2_short")
        row_token = sha256_bytes((row["video_sha256"] + str(row["dataset_question_id"])).encode())
        try:
            sham_groups = select_disjoint_sham_groups(
                group_count=frame_count // TEMPORAL_PATCH_SIZE,
                group_budget=len(mapping["evidence_group_indices"]),
                evidence_groups=mapping["evidence_group_indices"],
                control_groups=mapping["control_group_indices"], row_token=row_token)
        except ValueError as exc:
            code = str(exc).splitlines()[0]
            if allow_ineligible_rows and code == "disjoint_sham_group_set_unavailable":
                preflight_geometry_exclusions.append({
                    "row_token_sha256": row_token, "reason_code": code})
                continue
            raise
        mask_frames = {
            "qa_pixel_mask": masked_group_frame_indices(
                mapping["evidence_group_indices"], temporal_patch_size=TEMPORAL_PATCH_SIZE,
                frame_count=frame_count),
            "control_pixel_mask": masked_group_frame_indices(
                mapping["control_group_indices"], temporal_patch_size=TEMPORAL_PATCH_SIZE,
                frame_count=frame_count),
        }
        batches, render_hash = {}, None
        for condition, masked in mask_frames.items():
            masked_set = set(masked)
            condition_frames = [apply_mask_fill(frame, "black") if i in masked_set else frame.copy()
                                for i, frame in enumerate(frames)]
            rendered, messages = _build_prompt(processor, row, condition_frames,
                                                float(fps), prompt_suffix)
            if render_hash is not None and rendered != render_hash:
                raise ValueError("pixel_mask_changed_rendered_prompt")
            render_hash = rendered
            images, videos = process_vision_info(messages, return_video_metadata=True)
            if images is not None or not isinstance(videos, list) or len(videos) != 1:
                raise ValueError("video_processor_utility_contract_invalid")
            pixels, metadata = videos[0]
            if (metadata.get("frames_indices") != list(range(frame_count))
                    or not math.isclose(float(metadata.get("fps", 0)), float(fps), rel_tol=0, abs_tol=1e-7)):
                raise ValueError("processor_video_metadata_invalid")
            batches[condition] = process_qa_video(processor, rendered, images, pixels, metadata)
        temporal_patch = int(processor.video_processor.temporal_patch_size)
        if temporal_patch != TEMPORAL_PATCH_SIZE:
            raise ValueError("processor_temporal_patch_size_unexpected")
        batch = batches["qa_pixel_mask"]
        grid = [int(x) for x in batch["video_grid_thw"][0].tolist()]
        other_batch = batches["control_pixel_mask"]
        for key in batch:
            left, right = batch[key], other_batch[key]
            if hasattr(left, "shape") and hasattr(right, "shape"):
                if tuple(left.shape) != tuple(right.shape):
                    raise ValueError("pixel_condition_tensor_shape_mismatch")
                if key != "pixel_values_videos" and not __import__("torch").equal(left, right):
                    raise ValueError("pixel_mask_changed_nonvisual_input")
        group_map = map_groups_to_visual_keys(input_ids=batch["input_ids"][0].tolist(),
            video_token_id=int(processor.video_token_id), video_grid_thw=grid,
            spatial_merge_size=int(processor.image_processor.merge_size),
            temporal_patch_size=temporal_patch, frame_count=frame_count,
            evidence_group_indices=mapping["evidence_group_indices"],
            control_group_indices=mapping["control_group_indices"])
        claim = row["generated_parsed"]["claim"].strip()
        control_grid = [int(x) for x in other_batch["video_grid_thw"][0].tolist()]
        control_map = map_groups_to_visual_keys(input_ids=other_batch["input_ids"][0].tolist(),
            video_token_id=int(processor.video_token_id), video_grid_thw=control_grid,
            spatial_merge_size=int(processor.image_processor.merge_size),
            temporal_patch_size=temporal_patch, frame_count=frame_count,
            evidence_group_indices=mapping["evidence_group_indices"],
            control_group_indices=mapping["control_group_indices"])
        if group_map != control_map:
            raise ValueError("masked_condition_visual_key_geometry_changed")
        sham_key_positions = tuple(sorted(p for i in sham_groups for p in group_map["temporal_groups"][i]))
        target_ids = processor.tokenizer(claim, add_special_tokens=False)["input_ids"]
        if not target_ids or any(type(token) is not int or token < 0 for token in target_ids):
            raise ValueError("saved_claim_tokenization_invalid")
        row_hash = sha256_bytes((row["video_sha256"] + str(row["dataset_question_id"])).encode())
        prepared.append({"position": position, "row": row, "batches": batches,
            "mapping": mapping, "group_map": group_map, "target_ids": tuple(target_ids),
            "target_hash": sha256_bytes(json.dumps(target_ids, separators=(",", ":")).encode("ascii")),
            "claim_hash": sha256_bytes(claim.encode("utf-8")), "row_token_sha256": row_hash,
            "prompt_hash": sha256_bytes(rendered.encode("utf-8")), "sham_groups": list(sham_groups),
            "sham_key_positions": sham_key_positions, "masked_frame_indices": mask_frames,
            "fps": float(fps), "video_grid_thw": grid,
            "qa_pixel_tensor_sha256": _tensor_sha256(batches["qa_pixel_mask"]["pixel_values_videos"]),
            "control_pixel_tensor_sha256": _tensor_sha256(batches["control_pixel_mask"]["pixel_values_videos"])})
    for relative in data_rel:
        path = repo / relative
        hashes[relative] = _sha256(path)
    return prepared, hashes, preflight_geometry_exclusions


def _run_row(prepared: dict, model, torch, counters: dict) -> dict:
    ev = prepared["group_map"]["evidence_key_positions"]
    sham = prepared["sham_key_positions"]
    ctl = prepared["group_map"]["control_key_positions"]
    if len(ev) != len(sham) or set(ev) & set(sham) or set(ctl) & set(sham):
        raise ValueError("qa_control_sham_key_sets_invalid")
    branches = {}
    branch_token_scores = {}
    call_log = {}
    experiment = (
        ("baseline_qa_pixel_mask", "qa_pixel_mask", ()),
        ("baseline_control_pixel_mask", "control_pixel_mask", ()),
        ("qa_bias_qa_pixel_mask", "qa_pixel_mask", ev),
        ("qa_bias_control_pixel_mask", "control_pixel_mask", ev),
        ("sham_bias_qa_pixel_mask", "qa_pixel_mask", sham),
        ("sham_bias_control_pixel_mask", "control_pixel_mask", sham),
    )
    query_positions = None
    prompt_length = None
    for name, input_condition, biased in experiment:
        branch_cpu = prepared["batches"][input_condition]
        branch = {key: (value.to("cuda") if torch.is_tensor(value) else value)
                  for key, value in branch_cpu.items()}
        extended_branch, branch_queries = append_target_to_unpadded_batch(
            batch=branch, target_token_ids=prepared["target_ids"], torch_module=torch)
        branch_prompt_length = int(branch["input_ids"].shape[1])
        if branch_queries != tuple(range(branch_prompt_length - 1,
                branch_prompt_length + len(prepared["target_ids"]) - 1)):
            raise ValueError("target_query_rows_invalid")
        if query_positions is not None and branch_queries != query_positions:
            raise ValueError("condition_target_query_rows_changed")
        query_positions = branch_queries
        prompt_length = branch_prompt_length
        free, _total = torch.cuda.mem_get_info()
        if free < 2 * 1024**3 or _host_available_bytes() < 3 * 1024**3:
            raise RuntimeError("per_forward_resource_guard_failed")
        observed = {layer: 0 for layer in LAYERS_1_BASED}
        observer_handles = []
        layer_modules = model.model.language_model.layers
        hook_handles = (install_multiquery_video_key_intervention_hooks(
            model=model, layer_indices=tuple(layer - 1 for layer in LAYERS_1_BASED),
            biased_positions=biased, ablated_positions=(),
            query_positions=query_positions, logit_bias=LOGIT_BIAS,
            torch_module=torch) if biased else [])
        try:
            if biased:
                for layer in LAYERS_1_BASED:
                    module = layer_modules[layer - 1].self_attn
                    def observe(_module, args, kwargs, *, current_layer=layer,
                                current_biased=tuple(biased)):
                        mask = kwargs.get("attention_mask")
                        if not torch.is_tensor(mask) or mask.ndim != 4 or mask.shape[-2] != mask.shape[-1]:
                            raise ValueError("attention_mask_observer_invalid")
                        for qpos in query_positions:
                            for pos in current_biased:
                                if not math.isclose(float(mask[0, 0, qpos, pos].item()),
                                        float(torch.tensor(LOGIT_BIAS, dtype=mask.dtype).item()),
                                        rel_tol=0, abs_tol=1e-6):
                                    raise ValueError("visual_bias_not_observed")
                        observed[current_layer] += 1
                        return args, kwargs
                    observer_handles.append(module.register_forward_pre_hook(observe, with_kwargs=True))
            counters["attempted"] += 1
            with torch.inference_mode():
                output = model(**extended_branch, use_cache=False,
                               logits_to_keep=len(prepared["target_ids"]) + 1)
            counters["completed"] += 1
            score = score_target_logits_window(logits=output.logits,
                target_token_ids=prepared["target_ids"], prompt_length=prompt_length,
                first_query_position=prompt_length - 1, torch_module=torch)
            if biased and observed != {layer: 1 for layer in LAYERS_1_BASED}:
                raise ValueError("hook_layer_observation_count_invalid")
            branches[name] = score["target_logprob_mean"]
            branch_token_scores[name] = score["target_token_logprobs"]
            call_log[name] = {"hook_calls_by_layer": {str(k): v for k, v in observed.items()},
                "biased_key_count": len(biased), "ablated_key_count": 0,
                "query_token_count": len(query_positions),
                "surviving_bias_key_count": len(biased),
                "target_token_count": len(prepared["target_ids"])}
            del output
        finally:
            for handle in reversed(observer_handles):
                handle.remove()
            remove_hooks(hook_handles)
        del extended_branch, branch
        torch.cuda.empty_cache()
    dependence = compute_text_dependence(branches)
    record = prepared["row"]
    result = {"row_token_sha256": prepared["row_token_sha256"],
        "source_video_sha256": record["video_sha256"],
        "saved_claim_sha256": prepared["claim_hash"],
        "generation_prompt_version": "v2_short",
        "target_token_ids_sha256": prepared["target_hash"],
        "target_token_count": len(prepared["target_ids"]),
        "rendered_prompt_sha256": prepared["prompt_hash"],
        "input_ids_video_batch_sha256": _tensor_sha256(prepared["batches"]["qa_pixel_mask"]["input_ids"]),
        "qa_intervals_seconds": record["gold_interval_seconds"],
        "sampled_qa_frame_indices": prepared["mapping"]["sampled_qa_frame_indices"],
        "evidence_group_indices": prepared["mapping"]["evidence_group_indices"],
        "control_group_indices": prepared["mapping"]["control_group_indices"],
        "sham_bias_group_indices": prepared["sham_groups"],
        "evidence_frame_window_half_open": prepared["mapping"]["evidence_frame_window"],
        "control_frame_window_half_open": prepared["mapping"]["control_frame_window"],
        "resolution_rule": prepared["mapping"]["resolution_rule"],
        "expansion_frames": prepared["mapping"]["expansion_frames"],
        "temporal_group_count": prepared["group_map"]["temporal_group_count"],
        "tokens_per_temporal_group": prepared["group_map"]["tokens_per_temporal_group"],
        "qa_bias_key_count": prepared["group_map"]["equal_key_count"],
        "sham_bias_key_count": len(prepared["sham_key_positions"]),
        "masked_frame_indices": {key: list(value) for key, value in prepared["masked_frame_indices"].items()},
        "pixel_values_sha256": {"qa_pixel_mask": prepared["qa_pixel_tensor_sha256"],
            "control_pixel_mask": prepared["control_pixel_tensor_sha256"]},
        "processor_video_grid_thw": prepared["video_grid_thw"],
        "effective_sample_fps": prepared["fps"],
        "branch_mean_logprob_nats_per_token": branches,
        "branch_token_logprobs_nats": branch_token_scores,
        "primary": dependence,
        "calls": call_log,
        "forward_count": {"attempted": len(BRANCHES), "completed": len(BRANCHES)},
        "claim_review_completed": False, "direct_faithfulness": False}
    gc.collect(); torch.cuda.empty_cache()
    return result


def _tensor_sha256(value) -> str:
    import torch
    if not torch.is_tensor(value):
        raise ValueError("tensor_hash_input_invalid")
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return sha256_bytes(raw)


def pinned_model_weight_sha256(model_path: Path) -> dict[str, str]:
    """Hash every weight shard selected by the pinned HF safetensors index."""
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("pinned_model_weight_index_invalid")
    shards = sorted(set(weight_map.values()))
    # HF snapshots symlink into the sibling hub/blobs store.
    cache_root = model_path.parent.parent.parent.resolve(strict=True)
    manifest = {}
    for relative in shards:
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise ValueError("pinned_model_weight_shard_name_invalid")
        path = model_path / relative
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(cache_root) or not resolved.is_file():
            raise ValueError("pinned_model_weight_shard_path_invalid")
        manifest[relative] = sha256_file_streaming(resolved)
    return manifest


def sha256_file_streaming(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_private_exclusive(path: Path, payload: dict) -> None:
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--packet", default=str(PACKET_REL),
        help="frozen owner-only cohort JSON under data/private_nextgqa")
    parser.add_argument("--dataset-split", choices=("val", "test"), default=None,
        help="require the packet's split explicitly; default reads the split from the frozen packet")
    parser.add_argument("--allow-ineligible-rows", action="store_true",
        help="retain the frozen denominator, count generation/geometry exclusions, score eligible claims only, and never replace rows")
    parser.add_argument("--output", required=True, help="new mode-0600 JSON under data/private_nextgqa")
    parser.add_argument("--smoke-receipt", help="required for full mode: successful matching smoke JSON")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    repo = Path(__file__).resolve().parents[1]
    output = validate_private_output(args.output, repo)
    private = repo / "data/private_nextgqa"
    if output.parent != private or output.suffix != ".json":
        raise ValueError("output_must_be_private_json")
    if args.mode == "full" and not args.smoke_receipt:
        parser.error("matching_successful_smoke_receipt_required_for_full_mode")
    started = time.monotonic(); started_utc = datetime.now(timezone.utc).isoformat()
    counters = {"attempted": 0, "completed": 0}
    requested_rows = 3
    stage = "packet_preflight"
    try:
        packet_path = Path(args.packet)
        if not packet_path.is_absolute():
            packet_path = repo / packet_path
        packet_path = packet_path.resolve(strict=True)
        if not packet_path.is_relative_to(private.resolve(strict=True)):
            raise ValueError("packet_must_be_inside_private_data")
        packet, packet_raw = read_private_json(packet_path, private)
        frame_count = packet.get("frames_per_video")
        dataset_split = args.dataset_split or packet.get("dataset_split")
        records = validate_packet(packet, expected_frame_count=frame_count,
            expected_split=dataset_split,
            allow_ineligible_rows=args.allow_ineligible_rows)
        requested_rows = packet.get("requested_videos", len(records))
        if args.allow_ineligible_rows and len(records) < 6:
            raise ValueError("scorable_rows_below_minimum_6")
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info(); host_free = _host_available_bytes()
        if free < MIN_FREE_VRAM or host_free < MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_model_resource_guard_failed")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
            / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise ValueError("pinned_model_checkpoint_missing")
        stage = "pinned_model_weight_fingerprint"
        weight_sha256 = pinned_model_weight_sha256(model_path)
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        stage = "cpu_video_prompt_and_geometry_preflight"
        prepared, input_hashes, cpu_geometry_exclusions = _prepare_rows(repo, packet_path, packet, packet_raw,
            processor, process_vision_info, frame_count=frame_count,
            allow_ineligible_rows=args.allow_ineligible_rows)
        if len(prepared) + len(cpu_geometry_exclusions) != len(records):
            raise ValueError("all_packet_rows_cpu_preflight_required")
        if args.allow_ineligible_rows and len(prepared) < 6:
            raise ValueError("cpu_scorable_rows_below_minimum_6")
        source_hashes = {name: _sha256(repo / name) for name in SOURCE_REL}
        stage = "smoke_receipt_validation"
        if args.mode == "full":
            smoke_path = Path(args.smoke_receipt)
            if not smoke_path.is_absolute():
                smoke_path = private / smoke_path
            smoke, _smoke_raw = read_private_json(smoke_path, private)
            validate_smoke_receipt(smoke, input_hashes=input_hashes,
                source_hashes=source_hashes, model_revision=MODEL_REVISION,
                weight_sha256=weight_sha256,
                expected_row={"row_token_sha256": prepared[0]["row_token_sha256"],
                    "source_video_sha256": prepared[0]["row"]["video_sha256"],
                    "claim_hash": prepared[0]["claim_hash"],
                    "target_hash": prepared[0]["target_hash"]})
        stage = "load_pinned_model"
        free, total = torch.cuda.mem_get_info(); host_free = _host_available_bytes()
        if free < MIN_FREE_VRAM or host_free < MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_model_resource_guard_failed")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("model_processor_video_token_mismatch")
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        selected = prepared[:1] if args.mode == "smoke" else prepared
        results = []
        stage = "six_branch_claim_target_scoring"
        for row in selected:
            results.append(_run_row(row, model, torch, counters))
        expected_rows = 1 if args.mode == "smoke" else len(prepared)
        expected_forwards = expected_rows * len(BRANCHES)
        if len(results) != expected_rows or counters["attempted"] != expected_forwards or counters["completed"] != expected_forwards:
            raise RuntimeError("incomplete_requested_forward_set")
        values = [r["primary"]["delta_qa_bias_minus_baseline_nats"] for r in results]
        payload = {"schema_version": 1,
            "status": ("completed_exploratory_claim_text_qa_interval_key_bias_smoke" if args.mode == "smoke"
                       else "completed_exploratory_claim_text_qa_interval_key_bias_cohort"),
            "mode": args.mode, "purpose": "frozen_saved_claim_conditional_likelihood_sensitivity_to_QA_interval_visual_keys",
            "claim_generation_prompt_version": packet["generation_prompt_version"],
            "estimand": "F_X = meanLP(control-pixel-mask + X key bias) - meanLP(QA-pixel-mask + X key bias); primary F_QA_bias - F_baseline; specificity F_QA_bias - F_sham_bias",
            "bias_dose": LOGIT_BIAS, "layers_1_based": list(LAYERS_1_BASED),
            "pixel_mask": "black-fill all frames in an outward-expanded QA temporal-group window or equal-count control window",
            "internal_attention_operation": "positive +ln(2) visual-key bias only; no attention-key ablation",
            "cohort": packet_path.name, "cohort_sha256": sha256_bytes(packet_raw),
            "rights_status": "unverified; local-only per user instruction",
            "human_review": {"collection_waived_by_user": True, "review_actually_performed": False,
                "claim_review_completed": False, "claim_truth_reviewed": False},
            "requested_records": expected_rows if args.mode == "smoke" else requested_rows,
            "effective_records": len(results),
            "cohort_requested_videos": requested_rows,
            "cohort_structured_claims": packet.get("fully_structured_claim_records"),
            "cohort_scorable_records": len(prepared),
            "claim_generation_failure_count": requested_rows - packet.get(
                "fully_structured_claim_records", 0),
            "preflight_geometry_exclusion_count": packet.get(
                "fully_structured_claim_records", 0) - len(prepared),
            "claim_generation_failure_row_tokens_sha256": [
                sha256_bytes((row["video_sha256"] + str(row["dataset_question_id"])).encode())
                for row in packet.get("records", [])
                if not (isinstance(row.get("generated_parsed"), dict)
                    and row["generated_parsed"].get("valid") is True
                    and isinstance(row["generated_parsed"].get("claim"), str)
                    and row["generated_parsed"]["claim"].strip())],
            "preflight_geometry_exclusion_row_tokens_sha256": [
                sha256_bytes((row["video_sha256"] + str(row["dataset_question_id"])).encode())
                for row in packet.get("records", [])
                if (isinstance(row.get("generated_parsed"), dict)
                    and row["generated_parsed"].get("valid") is True
                    and isinstance(row["generated_parsed"].get("claim"), str)
                    and row["generated_parsed"]["claim"].strip()
                    and len(row.get("gold_interval_seconds", [])) != 1)]
                + [row["row_token_sha256"] for row in cpu_geometry_exclusions],
            "requested_forwards": expected_forwards, "attempted_forwards": counters["attempted"],
            "completed_forwards": counters["completed"], "input_sha256": input_hashes,
            "source_sha256": source_hashes, "records": results,
            "model_weight_sha256": weight_sha256,
            "finite_cohort_summary": None if args.mode == "smoke" else {
                "n": len(values), "mean_delta_qa_bias_minus_baseline_nats": sum(values)/len(values),
                "positive_rows": sum(x > 0 for x in values),
                "interpretation": "descriptive fixed-cohort result only; no p-value or population CI"},
            "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": MODEL_REVISION,
            "pinned_snapshot_json_sha256": {
                path.relative_to(model_path).as_posix(): _sha256(path)
                for path in sorted(model_path.rglob("*.json")) if path.is_file()},
            "precision": "4-bit weights, BF16 compute, eager attention", "seed": SEED,
            "frame_count": frame_count, "temporal_patch_size": TEMPORAL_PATCH_SIZE,
            "expected_temporal_group_count": frame_count // TEMPORAL_PATCH_SIZE,
            "key_bias_windows": "QA-aligned visual keys versus deterministic hash-ranked equal-count temporal-group sham set disjoint from both pixel-mask windows",
            "environment": {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")},
            "hardware": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "free_vram_before_load_gib": free / 1024**3, "total_vram_gib": total / 1024**3,
                "host_available_before_load_gib": host_free / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
            "elapsed_seconds": time.monotonic() - started,
            "direct_faithfulness": False, "faithfulness_result": False,
            "limitations": [
                "QA interval is a question-answer benchmark annotation, not a reviewed support interval for the saved claim",
                "claim text is re-tokenized from the saved parsed string; original generation token IDs are unavailable",
                "cohort is previously selected test videos and exploratory, not confirmatory",
                "outward temporal-group expansion changes exact QA interval boundaries at 2-frame granularity",
                "conditional log-likelihood sensitivity under pixel masking and key bias does not establish claim truth, semantic support, or direct explanation faithfulness",
                "attention-mask interventions can create distribution shift; result is model- and protocol-specific",
                "human review was not performed; waiver only removes a run prerequisite",
                "rights remain unverified; protected data and derived outputs remain local"]}
        _write_private_exclusive(output, payload)
        print(json.dumps({"status": payload["status"], "mode": args.mode,
            "effective_records": len(results), "completed_forwards": counters["completed"],
            "output_name": output.name}, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {"schema_version": 1, "status": "failed_exploratory_claim_text_qa_interval_key_bias",
            "mode": args.mode, "stage": stage, "failure_type": type(exc).__name__,
            "failure_code": str(exc).splitlines()[0][:160], "started_utc": started_utc,
            "requested_records": 1 if args.mode == "smoke" else requested_rows,
            "cohort_requested_videos": requested_rows,
            "effective_claim_candidates": (len(records)
                if isinstance(locals().get("records"), list) else None),
            "claim_generation_failure_count": (requested_rows - packet.get(
                "fully_structured_claim_records", 0) if isinstance(locals().get("packet"), dict) else None),
            "preflight_geometry_exclusion_count": (packet.get(
                "fully_structured_claim_records", 0) - len(records)
                if isinstance(locals().get("records"), list)
                and isinstance(locals().get("packet"), dict) else None),
            "requested_forwards": (1 if args.mode == "smoke" else requested_rows) * len(BRANCHES),
            "attempted_forwards": counters["attempted"], "completed_forwards": counters["completed"],
            "effective_records": 0, "direct_faithfulness": False,
            "claim_review_completed": False, "rights_status": "unverified; local-only",
            "source_sha256": {str(Path(__file__).relative_to(repo)): _sha256(Path(__file__))}}
        try:
            _write_private_exclusive(output, payload)
        except FileExistsError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(run())
