#!/usr/bin/env python3
"""Local-only exploratory QA-annotation temporal-key bias experiment.

This measures sensitivity of the NExT-GQA restricted-choice answer margin to
an internal visual-key bias on temporal groups overlapping the dataset's QA
time annotation, versus an equal-size disjoint hash sham. It is not a test of
claim truth, explanation grounding, or explanation faithfulness. The QA time
annotation is used to choose intervention keys and has no independent clock
calibration to decoded-video PTS.

Workflow: ``preflight`` freezes an exact hash-only cohort lock and validates
the processor/media path without loading model weights; ``smoke`` requires
that preflight and runs the first locked row; ``full`` requires a successful
matching smoke and scores the entire frozen cohort without row replacement.
All inputs and outputs must remain in data/private_nextgqa/ and mode 0600.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import time
from typing import Any, Sequence

import run_nextgqa_gold_interval_margin_pilot as cohort

MODEL_REVISION = cohort.MODEL_REVISION
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
LAYERS_1_BASED = tuple(range(5, 17))
BIAS_NATS = math.log(2.0)
SEED = 20261021
PURPOSE = "exploratory_nextgqa_qa_annotation_temporal_key_bias_sensitivity"
CONDITIONS = ("baseline", "annotation_interval_key_bias", "hash_sham_key_bias")
MIN_FREE_VRAM = 12 * 1024**3
MIN_HOST_AVAILABLE = 8 * 1024**3
MIN_FORWARD_VRAM = 2 * 1024**3
MIN_FORWARD_HOST = 3 * 1024**3
SOURCE_PATHS = (
    "scripts/run_nextgqa_annotation_key_bias.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/necessity_choice_margin_utils.py",
    "scripts/presentation_time_utils.py",
    "scripts/qta_multiquery_video_key_intervention.py",
    "scripts/private_output_paths.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


WEIGHT_SHARD_NAMES = tuple(
    f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6))


def hash_weight_shards(snapshot_path: Path) -> dict:
    """Hash the exact five expected checkpoint shards without exposing paths."""
    snapshot = Path(snapshot_path).resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError("pinned_snapshot_not_directory")
    # HF repos share one content-addressed cache at hub/blobs. A repo-local
    # ``models--.../blobs`` may itself be a symlink into that shared store.
    hub_cache = snapshot.parents[2].resolve(strict=True)
    blobs = (hub_cache / "blobs").resolve(strict=True)
    shard_entries = {path.name for path in snapshot.glob("model-*-of-*.safetensors")}
    if shard_entries != set(WEIGHT_SHARD_NAMES):
        raise ValueError("expected_five_weight_shards_not_present")
    hashes = {}
    for name in WEIGHT_SHARD_NAMES:
        entry = snapshot / name
        try:
            resolved = entry.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("expected_weight_shard_missing_or_unresolvable") from exc
        try:
            resolved.relative_to(blobs)
        except ValueError as exc:
            raise ValueError("weight_shard_target_outside_pinned_model_blobs") from exc
        info = resolved.stat()
        if not resolved.is_file() or not stat.S_ISREG(info.st_mode):
            raise ValueError("expected_weight_shard_not_regular_file")
        hashes[name] = sha256_file(resolved)
    if tuple(hashes) != WEIGHT_SHARD_NAMES:
        raise ValueError("weight_shard_set_invalid")
    return {"shards_sha256": hashes,
            "shards_digest_sha256": canonical_sha256(hashes),
            "shard_count": len(hashes)}


def source_snapshot(repo: Path) -> dict:
    """Bind the exact current code and Git base; dirty files remain hash-pinned."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return {"git_head": commit,
            "source_sha256": {name: sha256_file(repo / name)
                              for name in SOURCE_PATHS}}


def read_private_json(path: Path, private_root: Path) -> tuple[dict, str]:
    private_root = private_root.resolve(strict=True)
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=True)
    else:
        root_candidate = (private_root.parent.parent / candidate
                         if candidate.parts[:2] == ("data", "private_nextgqa")
                         else private_root / candidate)
        resolved = root_candidate.resolve(strict=True)
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError("private_input_path_invalid") from exc
    fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_input_permissions_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def private_output_path(value: str, repo: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("private_output_path_invalid")
    root = (repo / "data/private_nextgqa").resolve(strict=True)
    output = (repo / candidate).absolute()
    try:
        output.parent.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError("private_output_path_invalid") from exc
    if output.suffix != ".json" or os.path.lexists(output):
        raise ValueError("private_json_output_must_be_new")
    return output


def write_private_json_exclusive(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def row_token(media_sha256: str, qid_sha256: str) -> str:
    for value in (media_sha256, qid_sha256):
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("row_binding_hash_invalid")
    return hashlib.sha256(f"annotation-key-bias-v1\0{media_sha256}\0{qid_sha256}".encode()).hexdigest()


def map_annotation_frames_to_groups(*, evidence_frame_indices: Sequence[int],
                                    frame_count: int, temporal_patch_size: int,
                                    temporal_group_count: int) -> tuple[int, ...]:
    """Map PTS-selected annotation frames to complete Qwen temporal groups.

    A group is selected when at least one of its constituent uniformly sampled
    frames is inside the QA annotation according to the existing strict PTS
    inclusion helper. This is explicit coarse group overlap, not a clock
    calibration or claim-support judgment.
    """
    if (type(frame_count) is not int or frame_count <= 0
            or type(temporal_patch_size) is not int or temporal_patch_size <= 0
            or frame_count % temporal_patch_size
            or temporal_group_count != frame_count // temporal_patch_size
            or not isinstance(evidence_frame_indices, Sequence)
            or isinstance(evidence_frame_indices, (str, bytes))
            or not evidence_frame_indices
            or any(type(i) is not int or i < 0 or i >= frame_count
                   for i in evidence_frame_indices)
            or tuple(sorted(set(evidence_frame_indices))) != tuple(evidence_frame_indices)):
        raise ValueError("annotation_frame_group_inputs_invalid")
    groups = tuple(sorted({i // temporal_patch_size for i in evidence_frame_indices}))
    if not groups or len(groups) >= temporal_group_count:
        raise ValueError("annotation_temporal_group_budget_invalid")
    return groups


def choose_disjoint_hash_sham(*, temporal_group_count: int,
                              annotation_groups: Sequence[int],
                              opaque_row_token: str,
                              protocol_seed: str = "nextgqa-annotation-key-sham-v1") -> tuple[int, ...]:
    """Pick equal-count temporal groups, disjoint from annotation groups."""
    selected = tuple(annotation_groups)
    if (type(temporal_group_count) is not int or temporal_group_count <= 0
            or not selected or tuple(sorted(set(selected))) != selected
            or any(type(i) is not int or not 0 <= i < temporal_group_count for i in selected)
            or temporal_group_count - len(selected) < len(selected)
            or not isinstance(opaque_row_token, str) or len(opaque_row_token) != 64
            or not protocol_seed):
        raise ValueError("annotation_sham_inputs_invalid")
    excluded = set(selected)
    prefix = f"{protocol_seed}\0{opaque_row_token}\0"
    ranked = sorted((i for i in range(temporal_group_count) if i not in excluded),
                    key=lambda i: (hashlib.sha256(f"{prefix}{i}".encode()).digest(), i))
    result = tuple(sorted(ranked[:len(selected)]))
    if len(result) != len(selected) or set(result) & excluded:
        raise ValueError("annotation_sham_contract_failed")
    return result


def video_group_key_positions(*, input_ids: Sequence[int], video_token_id: int,
                              video_grid_thw: Sequence[int], spatial_merge_size: int,
                              temporal_patch_size: int) -> tuple[tuple[int, ...], ...]:
    """Validate one contiguous Qwen video-token run and return group positions."""
    if (not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes))
            or not input_ids or any(type(x) is not int or x < 0 for x in input_ids)
            or type(video_token_id) is not int or video_token_id < 0
            or not isinstance(video_grid_thw, Sequence) or len(video_grid_thw) != 3
            or any(type(x) is not int or x <= 0 for x in video_grid_thw)
            or type(spatial_merge_size) is not int or spatial_merge_size <= 0
            or type(temporal_patch_size) is not int or temporal_patch_size <= 0):
        raise ValueError("video_group_geometry_invalid")
    temporal, height, width = video_grid_thw
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("video_spatial_grid_invalid")
    positions = tuple(i for i, token in enumerate(input_ids) if token == video_token_id)
    per_group = (height // spatial_merge_size) * (width // spatial_merge_size)
    if (len(positions) != temporal * per_group or not positions
            or positions != tuple(range(positions[0], positions[0] + len(positions)))):
        raise ValueError("video_placeholder_geometry_mismatch")
    return tuple(tuple(positions[g * per_group:(g + 1) * per_group])
                 for g in range(temporal))


def flatten_selected_key_positions(*, temporal_groups: Sequence[Sequence[int]],
                                   annotation_group_indices: Sequence[int],
                                   sham_group_indices: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Expand groups and require exact equal key budget plus disjointness."""
    groups = tuple(tuple(group) for group in temporal_groups)
    annotation = tuple(annotation_group_indices)
    sham = tuple(sham_group_indices)
    if (not groups or not annotation or not sham
            or any(not group or tuple(sorted(set(group))) != group for group in groups)
            or any(type(i) is not int or not 0 <= i < len(groups) for i in annotation + sham)
            or tuple(sorted(set(annotation))) != annotation
            or tuple(sorted(set(sham))) != sham
            or set(annotation) & set(sham)):
        raise ValueError("annotation_sham_group_selection_invalid")
    annotation_keys = tuple(sorted(position for i in annotation for position in groups[i]))
    sham_keys = tuple(sorted(position for i in sham for position in groups[i]))
    if (not annotation_keys or len(annotation_keys) != len(sham_keys)
            or len(set(annotation_keys)) != len(annotation_keys)
            or len(set(sham_keys)) != len(sham_keys)
            or set(annotation_keys) & set(sham_keys)):
        raise ValueError("annotation_sham_flattened_key_budget_mismatch")
    return annotation_keys, sham_keys


def fixed_gold_foil_margin(choice_logprobs: Sequence[float], *, gold_index: int,
                           baseline_foil_index: int) -> float:
    """Gold minus the foil frozen from baseline; never reselect foil per arm."""
    if (not isinstance(choice_logprobs, Sequence) or isinstance(choice_logprobs, (str, bytes))
            or len(choice_logprobs) != 5
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(float(x)) for x in choice_logprobs)
            or type(gold_index) is not int or not 0 <= gold_index < 5
            or type(baseline_foil_index) is not int or not 0 <= baseline_foil_index < 5
            or baseline_foil_index == gold_index):
        raise ValueError("fixed_gold_foil_inputs_invalid")
    return float(choice_logprobs[gold_index]) - float(choice_logprobs[baseline_foil_index])


def audit_attention_mask_delta(*, original_mask: Any, actual_mask: Any,
                               query_position: int, key_positions: Sequence[int],
                               logit_bias: float, torch_module: Any) -> dict:
    """Prove a mask differs only at intended cells and report effective dtype dose."""
    if (torch_module is None or not torch_module.is_tensor(original_mask)
            or not torch_module.is_tensor(actual_mask)
            or original_mask.ndim != 4 or tuple(original_mask.shape[:2]) != (1, 1)
            or tuple(actual_mask.shape) != tuple(original_mask.shape)
            or actual_mask.dtype != original_mask.dtype
            or type(query_position) is not int
            or not isinstance(key_positions, Sequence) or isinstance(key_positions, (str, bytes))
            or not key_positions or any(type(k) is not int for k in key_positions)
            or tuple(sorted(set(key_positions))) != tuple(key_positions)
            or any(k < 0 or k >= int(original_mask.shape[-1]) or k >= query_position
                   for k in key_positions)
            or not 0 <= query_position < int(original_mask.shape[-2])
            or isinstance(logit_bias, bool) or not isinstance(logit_bias, (int, float))
            or not math.isfinite(float(logit_bias)) or float(logit_bias) <= 0):
        raise ValueError("attention_mask_delta_audit_inputs_invalid")
    original_mask = original_mask.to(device=actual_mask.device)
    if (bool(torch_module.isnan(original_mask).any())
            or bool(torch_module.isnan(actual_mask).any())):
        raise ValueError("attention_mask_delta_contains_nan")
    indices = torch_module.tensor(key_positions, device=actual_mask.device,
                                  dtype=torch_module.long)
    original_target = original_mask[0, 0, query_position, indices]
    if not bool(torch_module.isfinite(original_target).all()) or not bool(
            (original_target == 0).all()):
        raise ValueError("annotation_keys_not_originally_visible_zero")
    expected = original_mask.clone()
    expected[0, 0, query_position, indices] += float(logit_bias)
    effective = float(torch_module.tensor(float(logit_bias),
                                          dtype=original_mask.dtype).item())
    if effective <= 0 or not math.isfinite(effective):
        raise ValueError("effective_mask_bias_invalid")
    if not torch_module.equal(actual_mask, expected):
        raise ValueError("attention_mask_changed_outside_or_wrong_at_intended_cells")
    changed = actual_mask != original_mask
    changed_count = int(torch_module.count_nonzero(changed).item())
    if changed_count != len(key_positions):
        raise ValueError("attention_mask_changed_cell_count_mismatch")
    observed = actual_mask[0, 0, query_position, indices] - original_target
    if not bool(torch_module.equal(observed,
            torch_module.full_like(observed, effective))):
        raise ValueError("attention_mask_effective_delta_mismatch")
    return {"attention_mask_dtype": str(actual_mask.dtype),
            "requested_bias": float(logit_bias),
            "effective_bias": effective,
            "intended_changed_cells": len(key_positions),
            "observed_changed_cells": changed_count,
            "all_and_only_intended_cells_changed": True}


def baseline_best_foil(choice_logprobs: Sequence[float], gold_index: int) -> int:
    if (not isinstance(choice_logprobs, Sequence) or isinstance(choice_logprobs, (str, bytes))
            or len(choice_logprobs) != 5 or type(gold_index) is not int
            or not 0 <= gold_index < 5
            or any(not math.isfinite(float(v)) for v in choice_logprobs)):
        raise ValueError("baseline_choice_scores_invalid")
    return max((i for i in range(5) if i != gold_index),
               key=lambda i: (float(choice_logprobs[i]), -i))


def dataset_filenames(dataset_split: str) -> dict[str, str]:
    if dataset_split not in {"train", "val", "test"}:
        raise ValueError("nextgqa_dataset_split_invalid")
    return {"csv": f"{dataset_split}.csv",
            "gsub": f"gsub_{dataset_split}.json",
            "mapping": "map_vid_vidorID.json"}


def validate_split_bound_lock(payload: dict, *, requested: int,
                              dataset_split: str) -> list[dict]:
    """Validate v2 lock and reject assignments frozen under another split."""
    if (not isinstance(payload, dict)
            or set(payload) != {"schema_version", "dataset_split", "assignments"}
            or payload.get("schema_version") != 2
            or payload.get("dataset_split") != dataset_split):
        raise ValueError("cohort_lock_dataset_split_mismatch")
    legacy_shape = {"schema_version": 1, "assignments": payload["assignments"]}
    return cohort.validate_cohort_lock_payload(legacy_shape, requested)


def validate_run_binding(preflight: dict, *, args_n: int, cohort_hash: str,
                         lock_sha256: str, source: dict, data_hashes: dict,
                         exclusion_sha256: str, model_config_sha256: str,
                         versions: dict, dataset_split: str, mode: str,
                         candidate_manifest_digest: str,
                         candidate_manifest_count: int,
                         weight_shard_manifest: dict) -> None:
    if (not isinstance(preflight, dict) or preflight.get("status") != "passed"
            or preflight.get("purpose") != PURPOSE
            or preflight.get("mode") != "outcome_blind_preflight"
            or preflight.get("dataset_split") != dataset_split
            or preflight.get("requested_videos") != args_n
            or preflight.get("effective_videos") != args_n
            or preflight.get("cohort_sha256") != cohort_hash
            or preflight.get("cohort_lock_sha256") != lock_sha256
            or preflight.get("source_snapshot") != source
            or preflight.get("dataset_sha256") != data_hashes
            or preflight.get("exclusion_manifest_sha256") != exclusion_sha256
            or preflight.get("model_config_sha256") != model_config_sha256
            or preflight.get("package_versions") != versions
            or preflight.get("candidate_manifest_digest_sha256") != candidate_manifest_digest
            or preflight.get("candidate_manifest_count") != candidate_manifest_count
            or preflight.get("weight_shards_sha256") != weight_shard_manifest.get("shards_sha256")
            or preflight.get("weight_shards_digest_sha256") != weight_shard_manifest.get("shards_digest_sha256")
            or preflight.get("weight_shard_count") != 5
            or weight_shard_manifest.get("shard_count") != 5
            or mode not in {"smoke", "full"}):
        raise ValueError("outcome_blind_preflight_binding_mismatch")


def validate_smoke_receipt(smoke: dict, *, args_n: int, lock_sha256: str,
                           preflight_sha256: str, source: dict,
                           data_hashes: dict, model_config_sha256: str,
                           first_row_binding: dict, dataset_split: str,
                           candidate_manifest_digest: str,
                           candidate_manifest_count: int,
                           weight_shard_manifest: dict,
                           torch_module: Any) -> None:
    records = smoke.get("records") if isinstance(smoke, dict) else None
    record = records[0] if isinstance(records, list) and len(records) == 1 else None
    group_count = first_row_binding.get("annotation_group_count") if isinstance(
        first_row_binding, dict) else None
    token_count_per_group = first_row_binding.get("video_keys_per_temporal_group") if isinstance(
        first_row_binding, dict) else None
    expected_key_count = (group_count * token_count_per_group
                          if type(group_count) is int
                          and type(token_count_per_group) is int else None)
    groups = first_row_binding.get("annotation_group_indices", []) if isinstance(
        first_row_binding, dict) else []
    sham_groups = first_row_binding.get("sham_group_indices", []) if isinstance(
        first_row_binding, dict) else []
    temporal_group_count = first_row_binding.get("temporal_group_count") if isinstance(
        first_row_binding, dict) else None
    expected_row_fields = ("row_token_sha256", "media_sha256", "qid_sha256",
        "annotation_group_indices", "sham_group_indices", "video_grid_thw",
        "input_batch_sha256")
    expected_row = {name: first_row_binding.get(name) for name in expected_row_fields}
    actual_row = ({name: record.get(name) for name in expected_row_fields}
                  if isinstance(record, dict) else {})
    audits = ({"annotation_interval_key_bias": record.get("annotation_mask_audit"),
               "hash_sham_key_bias": record.get("sham_mask_audit")}
              if isinstance(record, dict) else {})
    conditions = record.get("conditions") if isinstance(record, dict) else None
    scores_by_condition = {}
    arithmetic_valid = isinstance(conditions, dict) and set(conditions) == set(CONDITIONS)
    if arithmetic_valid:
        for name in CONDITIONS:
            condition = conditions[name]
            scores = condition.get("choice_logprobs_nats") if isinstance(condition, dict) else None
            if (not isinstance(scores, list) or len(scores) != 5
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(float(value)) for value in scores)):
                arithmetic_valid = False
                break
            scores_by_condition[name] = [float(value) for value in scores]
    foil = record.get("baseline_best_foil_index") if isinstance(record, dict) else None
    gold = record.get("gold_choice_index") if isinstance(record, dict) else None
    if (arithmetic_valid and type(gold) is int and 0 <= gold < 5
            and type(foil) is int and 0 <= foil < 5 and foil != gold):
        arithmetic_valid = foil == baseline_best_foil(scores_by_condition["baseline"], gold)
        for name in CONDITIONS:
            condition = conditions[name]
            score = scores_by_condition[name]
            expected_margin = score[gold] - score[foil]
            arithmetic_valid = arithmetic_valid and math.isclose(
                float(condition.get("gold_choice_logprob_nats", math.nan)),
                score[gold], rel_tol=0, abs_tol=1e-8)
            arithmetic_valid = arithmetic_valid and math.isclose(
                float(condition.get("baseline_best_foil_logprob_nats", math.nan)),
                score[foil], rel_tol=0, abs_tol=1e-8)
            arithmetic_valid = arithmetic_valid and math.isclose(
                float(condition.get("fixed_gold_minus_baseline_foil_margin_nats", math.nan)),
                expected_margin, rel_tol=0, abs_tol=1e-8)
    else:
        arithmetic_valid = False

    def audit_valid(audit):
        if not isinstance(audit, dict):
            return False
        layers = audit.get("per_layer")
        dtype_name = audit.get("attention_mask_dtype")
        dtype = getattr(torch_module, dtype_name.removeprefix("torch."), None) \
            if isinstance(dtype_name, str) else None
        expected_effective = (float(torch_module.tensor(BIAS_NATS, dtype=dtype).item())
                              if dtype is not None else math.nan)
        return (audit.get("biased_key_count") == expected_key_count
                and audit.get("intended_changed_cells") == expected_key_count
                and audit.get("observed_changed_cells") == expected_key_count
                and audit.get("all_and_only_intended_cells_changed") is True
                and isinstance(audit.get("attention_mask_dtype"), str)
                and audit.get("requested_bias") == BIAS_NATS
                and audit.get("effective_bias") == audit.get("effective_bias_per_layer")
                and audit.get("effective_bias") == expected_effective
                and math.isfinite(float(audit.get("effective_bias", math.nan)))
                and audit.get("effective_bias", 0) > 0
                and isinstance(layers, dict)
                and set(layers) == {str(i) for i in LAYERS_1_BASED}
                and all(isinstance(value, dict)
                    and value.get("attention_mask_dtype") == audit.get("attention_mask_dtype")
                    and value.get("observed_changed_cells") == expected_key_count
                    and value.get("all_and_only_intended_cells_changed") is True
                    and value.get("requested_bias") == BIAS_NATS
                    and value.get("effective_bias") == audit.get("effective_bias")
                    for value in layers.values()))

    if (not isinstance(smoke, dict) or smoke.get("status") != "smoke_passed"
            or smoke.get("purpose") != PURPOSE or smoke.get("mode") != "smoke"
            or smoke.get("dataset_split") != dataset_split
            or smoke.get("requested_videos") != args_n
            or smoke.get("effective_videos") != 1
            or smoke.get("requested_forwards") != 3
            or smoke.get("attempted_forwards") != 3
            or smoke.get("completed_forwards") != 3
            or smoke.get("cohort_lock_sha256") != lock_sha256
            or smoke.get("preflight_receipt_sha256") != preflight_sha256
            or smoke.get("source_snapshot") != source
            or smoke.get("dataset_sha256") != data_hashes
            or smoke.get("model_config_sha256") != model_config_sha256
            or smoke.get("candidate_manifest_digest_sha256") != candidate_manifest_digest
            or smoke.get("candidate_manifest_count") != candidate_manifest_count
            or smoke.get("weight_shards_sha256") != weight_shard_manifest.get("shards_sha256")
            or smoke.get("weight_shards_digest_sha256") != weight_shard_manifest.get("shards_digest_sha256")
            or smoke.get("weight_shard_count") != 5
            or weight_shard_manifest.get("shard_count") != 5
            or actual_row != expected_row
            or not expected_key_count or expected_key_count <= 0
            or type(temporal_group_count) is not int
            or not groups or len(groups) != group_count
            or len(sham_groups) != group_count
            or len(set(groups)) != len(groups) or len(set(sham_groups)) != len(sham_groups)
            or set(groups) & set(sham_groups)
            or any(type(i) is not int or not 0 <= i < temporal_group_count
                   for i in groups + sham_groups)
            or not arithmetic_valid
            or not math.isclose(float(record.get("primary_annotation_minus_sham_nats", math.nan)),
                float(conditions["annotation_interval_key_bias"][
                    "fixed_gold_minus_baseline_foil_margin_nats"])
                - float(conditions["hash_sham_key_bias"][
                    "fixed_gold_minus_baseline_foil_margin_nats"]), rel_tol=0, abs_tol=1e-8)
            or not math.isclose(float(record.get("annotation_minus_baseline_nats", math.nan)),
                float(conditions["annotation_interval_key_bias"][
                    "fixed_gold_minus_baseline_foil_margin_nats"])
                - float(conditions["baseline"][
                    "fixed_gold_minus_baseline_foil_margin_nats"]), rel_tol=0, abs_tol=1e-8)
            or not math.isclose(float(record.get("sham_minus_baseline_nats", math.nan)),
                float(conditions["hash_sham_key_bias"][
                    "fixed_gold_minus_baseline_foil_margin_nats"])
                - float(conditions["baseline"][
                    "fixed_gold_minus_baseline_foil_margin_nats"]), rel_tol=0, abs_tol=1e-8)
            or not audit_valid(audits.get("annotation_interval_key_bias"))
            or not audit_valid(audits.get("hash_sham_key_bias"))):
        raise ValueError("smoke_receipt_binding_or_completion_invalid")


def package_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")}


def summarize(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(float(x) for x in values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {"n": n, "mean": sum(ordered) / n, "median": median,
            "min": ordered[0], "max": ordered[-1]}


def _load_checkpoint_config(repo: Path) -> tuple[Path, str]:
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        raise ValueError("pinned_local_checkpoint_missing_downloads_disabled")
    return model_path, sha256_file(model_path / "config.json")


def _data_hashes(dataset_dir: Path, dataset_split: str) -> dict:
    filenames = dataset_filenames(dataset_split)
    return {"dataset_split": dataset_split,
            "csv_filename": filenames["csv"],
            "csv_sha256": sha256_file(dataset_dir / filenames["csv"]),
            "gsub_filename": filenames["gsub"],
            "gsub_sha256": sha256_file(dataset_dir / filenames["gsub"]),
            "mapping_filename": filenames["mapping"],
            "mapping_sha256": sha256_file(dataset_dir / filenames["mapping"])}


def _freeze_exact(repo: Path, *, video_root: Path, n: int, frame_count: int,
                  exclusion: list[str], lock_assignments: list[dict] | None,
                  dataset_split: str):
    cohort.MAX_VIDEOS = n
    selected, counts, meta, fingerprint = cohort.freeze_cohort(
        repo, video_root, exclusion, frame_count, lock_assignments,
        allowed_pair_sha256=None, dataset_split=dataset_split)
    cohort.validate_cohort_cardinality(n, len(selected))
    return selected, counts, meta, fingerprint


def _preflight_or_score(args) -> int:
    repo = Path(__file__).resolve().parents[1]
    private_root = (repo / "data/private_nextgqa").resolve(strict=True)
    output = private_output_path(args.output, repo)
    source = source_snapshot(repo)
    video_root = Path(args.video_root).resolve(strict=True)
    dataset_dir = repo / "NExT-GQA/datasets/nextgqa"
    data_hashes = _data_hashes(dataset_dir, args.dataset_split)
    model_path, model_config_sha256 = _load_checkpoint_config(repo)
    weight_shard_manifest = hash_weight_shards(model_path)
    try:
        exclusion_payload, exclusion_sha256 = read_private_json(
            Path(args.exclude_media_sha256), private_root)
        exclusion_hashes = cohort.validate_private_media_exclusion(exclusion_payload)
    except (OSError, ValueError) as exc:
        raise ValueError(f"private_media_exclusion_invalid:{exc}") from exc

    lock_assignments = None
    lock_sha256 = None
    if args.mode == "preflight":
        if not args.cohort_lock_output:
            raise ValueError("preflight_requires_cohort_lock_output")
        lock_output = private_output_path(args.cohort_lock_output, repo)
        if lock_output == output:
            raise ValueError("lock_and_receipt_outputs_must_differ")
    else:
        if not args.cohort_lock or not args.preflight_receipt:
            raise ValueError("scoring_requires_exact_lock_and_preflight")
        lock_payload, lock_sha256 = read_private_json(Path(args.cohort_lock), private_root)
        lock_assignments = validate_split_bound_lock(lock_payload,
            requested=args.n, dataset_split=args.dataset_split)

    selected, counts, meta, cohort_hash = _freeze_exact(
        repo, video_root=video_root, n=args.n, frame_count=args.frames,
        exclusion=exclusion_hashes, lock_assignments=lock_assignments,
        dataset_split=args.dataset_split)
    if args.mode == "preflight":
        lock_payload = {"schema_version": 2,
                        "dataset_split": args.dataset_split,
                        "assignments": cohort.build_cohort_lock_assignments(selected)}
        # The lock must exist before the preflight receipt references its byte hash.
        write_private_json_exclusive(lock_output, lock_payload)
        lock_sha256 = sha256_file(lock_output)

    import torch
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    if not torch.cuda.is_available() and args.mode in {"smoke", "full"}:
        raise RuntimeError("cuda_unavailable_no_cpu_fallback")
    versions = package_versions()
    label_ids = []
    for label in cohort.LABELS:
        ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"choice_label_tokenization_invalid:{label}")
        label_ids.append(ids[0])
    if len(set(label_ids)) != 5:
        raise ValueError("choice_labels_not_distinct_tokens")

    row_bindings = []
    patch = int(processor.video_processor.temporal_patch_size)
    if args.frames % patch:
        raise ValueError("frame_count_not_aligned_to_temporal_patch")
    for sample in selected:
        prepared_row = cohort.prepare_conditions([sample], processor, torch,
            args.frames, retain_batches=True)[0]
        batch = prepared_row["batches"]["full"]
        batch_hash = _batch_digest(batch, torch)
        grid = prepared_row["grid"]
        temporal_count = int(grid[0])
        if temporal_count != args.frames // patch:
            raise ValueError("processor_temporal_group_count_mismatch")
        groups = map_annotation_frames_to_groups(
            evidence_frame_indices=sample["evidence_positions"],
            frame_count=args.frames, temporal_patch_size=patch,
            temporal_group_count=temporal_count)
        qid_hash = hashlib.sha256(str(sample["qid"]).encode("utf-8")).hexdigest()
        media_hash = sample["decode"]["source_sha256"]
        token = row_token(media_hash, qid_hash)
        sham = choose_disjoint_hash_sham(temporal_group_count=temporal_count,
            annotation_groups=groups, opaque_row_token=token)
        row_bindings.append({"row_token_sha256": token,
            "media_sha256": media_hash, "qid_sha256": qid_hash,
            "annotation_group_indices": list(groups),
            "sham_group_indices": list(sham),
            "annotation_group_count": len(groups),
            "temporal_group_count": temporal_count,
            "sampled_frame_count": len(sample["frames"]),
            "video_grid_thw": grid,
            "input_batch_sha256": batch_hash,
            "video_keys_per_temporal_group": (
                (int(grid[1]) // int(processor.video_processor.merge_size))
                * (int(grid[2]) // int(processor.video_processor.merge_size)))})
        del prepared_row, batch
        gc.collect()
    if args.mode == "preflight":
        payload = {"schema_version": 1, "purpose": PURPOSE,
            "mode": "outcome_blind_preflight", "status": "passed",
            "dataset_split": args.dataset_split,
            "source_snapshot": source, "dataset_sha256": data_hashes,
            "exclusion_manifest_sha256": exclusion_sha256,
            "exclusion_media_digest_sha256": canonical_sha256(sorted(exclusion_hashes)),
            "candidate_manifest_digest_sha256": meta["private_exclusion_manifest_digest"],
            "candidate_manifest_count": meta["private_exclusion_manifest_count"],
            "cohort_lock_sha256": lock_sha256,
            "cohort_sha256": cohort_hash, "requested_videos": args.n,
            "effective_videos": len(selected), "requested_frames_per_video": args.frames,
            "temporal_patch_size": patch, "row_bindings": row_bindings,
            "candidate_video_count": meta["candidate_video_count"],
            "candidate_scan_count": counts["candidate_scan_count"],
            "timing_pass_count": counts["timing_pass_videos"],
            "timing_rejected_count": counts["timing_rejected_videos"],
            "decode_failure_count": counts["decode_failures"],
            "rows_without_equal_count_disjoint_frame_control": counts["videos_without_interval_control_pair"],
            "model": MODEL_NAME, "model_revision": MODEL_REVISION,
            "model_config_sha256": model_config_sha256,
            "weight_shards_sha256": weight_shard_manifest["shards_sha256"],
            "weight_shards_digest_sha256": weight_shard_manifest["shards_digest_sha256"],
            "weight_shard_count": weight_shard_manifest["shard_count"],
            "package_versions": versions,
            "source_clock_calibration_performed": False,
            "human_review_performed": False,
            "endpoint": "full-video five-choice gold-minus-baseline-best-foil logprob margin under annotation-key bias versus equal-count disjoint sham",
            "interpretation_limit": "QA time-annotation internal-intervention sensitivity proxy only; not claim support, claim truth, or explanation faithfulness",
            "rights_status": "unverified; local-only operation",
            "privacy": "Hash-only cohort data; no text, IDs, paths, or media emitted"}
        write_private_json_exclusive(output, payload)
        print(f"Outcome-blind preflight passed: {args.n} exact rows; private receipt {output.name}")
        return 0

    preflight, preflight_sha256 = read_private_json(Path(args.preflight_receipt), private_root)
    validate_run_binding(preflight, args_n=args.n, cohort_hash=cohort_hash,
        lock_sha256=lock_sha256, source=source, data_hashes=data_hashes,
        exclusion_sha256=exclusion_sha256, model_config_sha256=model_config_sha256,
        versions=versions, dataset_split=args.dataset_split, mode=args.mode,
        candidate_manifest_digest=meta["private_exclusion_manifest_digest"],
        candidate_manifest_count=meta["private_exclusion_manifest_count"],
        weight_shard_manifest=weight_shard_manifest)
    if preflight.get("row_bindings") != row_bindings:
        raise ValueError("preflight_row_mapping_changed")
    smoke_payload = None
    smoke_sha256 = None
    if args.mode == "full":
        if not args.smoke_receipt:
            raise ValueError("full_requires_successful_smoke_receipt")
        smoke_payload, smoke_sha256 = read_private_json(Path(args.smoke_receipt), private_root)
        validate_smoke_receipt(smoke_payload, args_n=args.n,
            lock_sha256=lock_sha256, preflight_sha256=preflight_sha256,
            source=source, data_hashes=data_hashes,
            model_config_sha256=model_config_sha256,
            first_row_binding=row_bindings[0],
            dataset_split=args.dataset_split,
            candidate_manifest_digest=meta["private_exclusion_manifest_digest"],
            candidate_manifest_count=meta["private_exclusion_manifest_count"],
            weight_shard_manifest=weight_shard_manifest,
            torch_module=torch)

    free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    host_avail = cohort.host_available_bytes()
    if args.mode in {"smoke", "full"} and (free < MIN_FREE_VRAM or host_avail < MIN_HOST_AVAILABLE):
        raise RuntimeError("pre_model_resource_guard_failed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if int(model.config.video_token_id) != int(processor.video_token_id):
        raise ValueError("processor_checkpoint_video_token_mismatch")
    from qta_multiquery_video_key_intervention import (
        install_multiquery_video_key_intervention_hooks, remove_hooks)

    row_indices = range(args.n) if args.mode == "full" else range(1)
    records = []
    attempted = completed = 0
    failed_condition = None
    start = time.monotonic()

    def forward_scores(batch, label_tensor, expected_batch_hash, biased_positions=()):
        nonlocal attempted, completed, failed_condition
        torch.cuda.empty_cache()
        if (torch.cuda.mem_get_info()[0] < MIN_FORWARD_VRAM
                or cohort.host_available_bytes() < MIN_FORWARD_HOST):
            raise RuntimeError("per_forward_resource_guard_failed")
        device_batch = batch.to("cuda")
        seq_len = int(device_batch["input_ids"].shape[1])
        handles = []
        calls = {i: 0 for i in LAYERS_1_BASED}
        before_observers = []
        after_observers = []
        original_masks = {}
        layer_audits = {}
        attempted += 1
        try:
            if biased_positions:
                layers = model.model.language_model.layers
                for layer_index in LAYERS_1_BASED:
                    module = layers[layer_index - 1].self_attn
                    def capture_original(_module, args, kwargs, *, layer=layer_index):
                        mask = kwargs.get("attention_mask")
                        if not torch.is_tensor(mask) or mask.ndim != 4:
                            raise ValueError("attention_mask_original_capture_invalid")
                        original_masks[layer] = mask.detach().to("cpu").clone()
                        return args, kwargs
                    before_observers.append(module.register_forward_pre_hook(
                        capture_original, with_kwargs=True))

                handles = install_multiquery_video_key_intervention_hooks(
                    model=model, layer_indices=tuple(i - 1 for i in LAYERS_1_BASED),
                    biased_positions=biased_positions, ablated_positions=(),
                    query_positions=(seq_len - 1,), logit_bias=BIAS_NATS,
                    torch_module=torch)
                for layer_index in LAYERS_1_BASED:
                    module = layers[layer_index - 1].self_attn
                    def audit_intervened(_module, args, kwargs, *, layer=layer_index):
                        mask = kwargs.get("attention_mask")
                        original = original_masks.pop(layer, None)
                        if original is None or not torch.is_tensor(mask):
                            raise ValueError("attention_mask_delta_capture_missing")
                        layer_audits[layer] = audit_attention_mask_delta(
                            original_mask=original, actual_mask=mask,
                            query_position=seq_len - 1,
                            key_positions=biased_positions,
                            logit_bias=BIAS_NATS, torch_module=torch)
                        calls[layer] += 1
                        return args, kwargs
                    after_observers.append(module.register_forward_pre_hook(
                        audit_intervened, with_kwargs=True))
            if _batch_digest(batch, torch) != expected_batch_hash:
                raise ValueError("frozen_batch_changed_before_forward")
            with torch.inference_mode():
                output_obj = model(**device_batch, use_cache=False, logits_to_keep=1)
            completed += 1
            logits = output_obj.logits[0, -1].float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite_answer_logits")
            values = torch.log_softmax(logits, dim=-1)[label_tensor].detach().cpu().tolist()
            if biased_positions and calls != {i: 1 for i in LAYERS_1_BASED}:
                raise ValueError("intervention_layer_calls_mismatch")
            if biased_positions:
                if set(layer_audits) != set(LAYERS_1_BASED):
                    raise ValueError("intervention_mask_audit_layer_set_mismatch")
                dtypes = {item["attention_mask_dtype"] for item in layer_audits.values()}
                doses = {item["effective_bias"] for item in layer_audits.values()}
                changed = {item["observed_changed_cells"] for item in layer_audits.values()}
                if len(dtypes) != 1 or len(doses) != 1 or changed != {len(biased_positions)}:
                    raise ValueError("intervention_mask_audit_inconsistent_across_layers")
                dtype, dose = next(iter(dtypes)), next(iter(doses))
                audit = {"hook_calls_by_layer": {str(i): calls[i] for i in LAYERS_1_BASED},
                    "biased_key_count": len(biased_positions),
                    "requested_bias": BIAS_NATS,
                    "intended_changed_cells": len(biased_positions),
                    "observed_changed_cells": len(biased_positions),
                    "all_and_only_intended_cells_changed": True,
                    "attention_mask_dtype": dtype, "effective_bias": dose,
                    "effective_bias_per_layer": dose,
                    "per_layer": {str(i): layer_audits[i] for i in LAYERS_1_BASED}}
            else:
                audit = {"hook_calls_by_layer": None, "biased_key_count": 0}
            del output_obj, logits
            return [float(x) for x in values], audit
        finally:
            for handle in reversed(after_observers + before_observers):
                handle.remove()
            remove_hooks(handles)
            del device_batch
            torch.cuda.empty_cache()

    try:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        for index in row_indices:
            sample = selected[index]
            failed_condition = "prepare_full_batch"
            row_prepared = cohort.prepare_conditions([sample], processor, torch,
                args.frames, retain_batches=True)[0]
            batch = row_prepared["batches"]["full"]
            batch_hash = _batch_digest(batch, torch)
            if (batch_hash != row_bindings[index]["input_batch_sha256"]
                    or row_prepared["grid"] != row_bindings[index]["video_grid_thw"]):
                raise ValueError("preflight_full_video_batch_binding_mismatch")
            input_ids = batch["input_ids"][0].tolist()
            geometry = batch["video_grid_thw"][0].tolist()
            groups = video_group_key_positions(input_ids=input_ids,
                video_token_id=int(processor.video_token_id), video_grid_thw=geometry,
                spatial_merge_size=int(processor.video_processor.merge_size),
                temporal_patch_size=patch)
            annotation_group_ids = tuple(row_bindings[index]["annotation_group_indices"])
            sham_group_ids = tuple(row_bindings[index]["sham_group_indices"])
            if (len(annotation_group_ids) != len(sham_group_ids)
                    or set(annotation_group_ids) & set(sham_group_ids)
                    or len(groups) != row_bindings[index]["temporal_group_count"]):
                raise ValueError("annotation_sham_key_budget_mismatch")
            annotation_keys, sham_keys = flatten_selected_key_positions(
                temporal_groups=groups,
                annotation_group_indices=annotation_group_ids,
                sham_group_indices=sham_group_ids)
            label_tensor = torch.tensor(label_ids, device="cuda")

            failed_condition = "baseline"
            baseline_scores, _ = forward_scores(batch, label_tensor, batch_hash)
            gold = int(sample["gold"])
            foil = baseline_best_foil(baseline_scores, gold)
            failed_condition = "annotation_interval_key_bias"
            annotation_scores, annotation_audit = forward_scores(batch, label_tensor,
                batch_hash, annotation_keys)
            failed_condition = "hash_sham_key_bias"
            sham_scores, sham_audit = forward_scores(batch, label_tensor,
                batch_hash, sham_keys)
            margins = {"baseline": fixed_gold_foil_margin(baseline_scores,
                         gold_index=gold, baseline_foil_index=foil),
                       "annotation_interval_key_bias": fixed_gold_foil_margin(
                           annotation_scores, gold_index=gold, baseline_foil_index=foil),
                       "hash_sham_key_bias": fixed_gold_foil_margin(
                           sham_scores, gold_index=gold, baseline_foil_index=foil)}
            records.append({"row_token_sha256": row_bindings[index]["row_token_sha256"],
                "media_sha256": row_bindings[index]["media_sha256"],
                "qid_sha256": row_bindings[index]["qid_sha256"],
                "gold_choice_index": gold, "baseline_best_foil_index": foil,
                "conditions": {name: {"choice_logprobs_nats": scores,
                    "gold_choice_logprob_nats": scores[gold],
                    "baseline_best_foil_logprob_nats": scores[foil],
                    "fixed_gold_minus_baseline_foil_margin_nats": margins[name]}
                    for name, scores in (("baseline", baseline_scores),
                         ("annotation_interval_key_bias", annotation_scores),
                         ("hash_sham_key_bias", sham_scores))},
                "primary_annotation_minus_sham_nats": (
                    margins["annotation_interval_key_bias"] - margins["hash_sham_key_bias"]),
                "annotation_minus_baseline_nats": (
                    margins["annotation_interval_key_bias"] - margins["baseline"]),
                "sham_minus_baseline_nats": margins["hash_sham_key_bias"] - margins["baseline"],
                "annotation_group_indices": list(annotation_group_ids),
                "sham_group_indices": list(sham_group_ids),
                "video_grid_thw": geometry, "input_batch_sha256": batch_hash,
                "annotation_mask_audit": annotation_audit,
                "sham_mask_audit": sham_audit,
                "gold_prediction_correct_by_condition": {
                    name: max(range(5), key=lambda j: (scores[j], -j)) == gold
                    for name, scores in (("baseline", baseline_scores),
                         ("annotation_interval_key_bias", annotation_scores),
                         ("hash_sham_key_bias", sham_scores))}})
            del row_prepared, batch, label_tensor
            gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:
        failed_payload = {"schema_version": 1, "purpose": PURPOSE,
            "mode": args.mode, "dataset_split": args.dataset_split,
            "status": "failed", "failure_type": type(exc).__name__,
            "failure_code": str(exc)[:240], "failed_condition": failed_condition,
            "requested_videos": args.n,
            "completed_full_condition_sets": len(records),
            "attempted_forwards": attempted, "completed_forwards": completed,
            "source_snapshot": source, "cohort_lock_sha256": lock_sha256,
            "preflight_receipt_sha256": preflight_sha256,
            "candidate_manifest_digest_sha256": meta["private_exclusion_manifest_digest"],
            "candidate_manifest_count": meta["private_exclusion_manifest_count"],
            "weight_shards_sha256": weight_shard_manifest["shards_sha256"],
            "weight_shards_digest_sha256": weight_shard_manifest["shards_digest_sha256"],
            "weight_shard_count": weight_shard_manifest["shard_count"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rights_status": "unverified; local-only operation",
            "human_review_performed": False, "source_clock_calibration_performed": False}
        write_private_json_exclusive(output, failed_payload)
        raise
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()

    expected_rows = 1 if args.mode == "smoke" else args.n
    expected_forwards = expected_rows * 3
    if len(records) != expected_rows or attempted != expected_forwards or completed != expected_forwards:
        raise RuntimeError("run_completion_arithmetic_mismatch")
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError("final_cuda_synchronize_failed; scores discarded") from exc
    payload = {"schema_version": 1, "purpose": PURPOSE,
        "mode": args.mode, "status": "smoke_passed" if args.mode == "smoke" else "completed",
        "dataset_split": args.dataset_split,
        "source_snapshot": source, "dataset_sha256": data_hashes,
        "model": MODEL_NAME, "model_revision": MODEL_REVISION,
        "model_config_sha256": model_config_sha256,
        "precision": "4-bit weights, BF16 compute, eager attention",
        "layers_1_based": list(LAYERS_1_BASED), "requested_bias_nats": BIAS_NATS,
        "seed": SEED, "cohort_lock_sha256": lock_sha256,
        "cohort_sha256": cohort_hash, "preflight_receipt_sha256": preflight_sha256,
        "candidate_manifest_digest_sha256": meta["private_exclusion_manifest_digest"],
        "candidate_manifest_count": meta["private_exclusion_manifest_count"],
        "weight_shards_sha256": weight_shard_manifest["shards_sha256"],
        "weight_shards_digest_sha256": weight_shard_manifest["shards_digest_sha256"],
        "weight_shard_count": weight_shard_manifest["shard_count"],
        "smoke_receipt_sha256": smoke_sha256,
        "requested_videos": args.n, "effective_videos": len(records),
        "requested_forwards": expected_forwards, "attempted_forwards": attempted,
        "completed_forwards": completed,
        "primary_estimand": "per-row annotation-interval-key-bias fixed gold-minus-baseline-best-foil margin minus equal-size disjoint hash-sham margin",
        "primary_summary": None if args.mode == "smoke" else summarize(
            [r["primary_annotation_minus_sham_nats"] for r in records]),
        "annotation_minus_baseline_summary": None if args.mode == "smoke" else summarize(
            [r["annotation_minus_baseline_nats"] for r in records]),
        "records": records, "statistical_test": "none; exploratory finite-cohort description",
        "confidence_interval": None, "claim_specific_faithfulness_estimate": None,
        "human_review_performed": False, "source_clock_calibration_performed": False,
        "rights_status": "unverified; protected data and outputs kept local",
        "interpretation_limit": "NExT-GQA QA-time annotation key-bias sensitivity only; not claim support or explanation faithfulness",
        "runtime_seconds": time.monotonic() - start,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
            "gpu": torch.cuda.get_device_name(0), "initial_free_vram_bytes": int(free),
            "total_vram_bytes": int(total), "package_versions": versions}}
    write_private_json_exclusive(output, payload)
    print(json.dumps({"status": payload["status"], "records": len(records),
        "forwards": completed, "output": output.name,
        "output_sha256": sha256_file(output),
        "output_mode": oct(stat.S_IMODE(output.stat().st_mode))}, sort_keys=True))
    return 0


def sha256_file_from_tensor(tensor: Any) -> str:
    """Digest tensor bytes on CPU for a stable per-batch receipt (no text)."""
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.reshape(-1).view(__import__("torch").uint8)
                          .numpy().tobytes()).hexdigest()


def _batch_digest(batch: dict, torch_module) -> str:
    descriptor = {}
    for key, tensor in sorted(batch.items()):
        if not torch_module.is_tensor(tensor):
            raise ValueError("processor_batch_contains_non_tensor")
        descriptor[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                           "sha256": sha256_file_from_tensor(tensor)}
    return canonical_sha256(descriptor)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "smoke", "full"))
    parser.add_argument("--dataset-split", choices=("train", "val", "test"),
                        default="test")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--frames", type=int, choices=(16, 32), default=32)
    parser.add_argument("--exclude-media-sha256", required=True,
                        help="owner-only manifest of previously exposed/scored media hashes")
    parser.add_argument("--cohort-lock", help="hash-only exact lock; required for smoke/full")
    parser.add_argument("--cohort-lock-output", help="new lock path; required for preflight")
    parser.add_argument("--preflight-receipt", help="passed outcome-blind preflight; required for smoke/full")
    parser.add_argument("--smoke-receipt", help="successful matching smoke; required for full")
    parser.add_argument("--output", required=True, help="new local JSON under data/private_nextgqa")
    parser.add_argument("--video-root", default=str(Path.home() / "nextgqa_sample/NExTVideo"))
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_user_authorized_local_only_required")
    if not 3 <= args.n <= 20:
        parser.error("n_must_be_between_3_and_20")
    if args.mode == "preflight" and (args.cohort_lock or args.preflight_receipt or args.smoke_receipt):
        parser.error("preflight_accepts_no_prior_lock_or_receipt")
    if args.mode == "smoke" and (not args.cohort_lock or not args.preflight_receipt or args.smoke_receipt):
        parser.error("smoke_requires_lock_and_preflight_only")
    if args.mode == "full" and (not args.cohort_lock or not args.preflight_receipt or not args.smoke_receipt):
        parser.error("full_requires_lock_preflight_and_smoke")
    try:
        return _preflight_or_score(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
