#!/usr/bin/env python3
"""Run a frozen local-only QTA same-video question-swap diagnostic.

This measures selector sensitivity only. It never reads answers, choices,
temporal annotations, generated text, or prediction logits. The finite-cohort
result is not a faithfulness estimate. Protected inputs and all output remain
local under data/private_nextgqa.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Any

from qta_question_swap_metrics import (
    forward_accounting, question_repeat_schedule, summarize_video_contrasts,
    video_contrast,
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
FRAME_COUNT = 32
TOLERANCE = Fraction(1, 20)
LAYERS = tuple(range(5, 17))
VISUAL_RATIO = 0.10
RANDOM_SEED = 20261018
MIN_FREE_BEFORE_LOAD = 12 * 1024**3
MIN_FREE_BEFORE_FORWARD = 768 * 1024**2
MIN_AVAILABLE_RAM_BEFORE_LOAD = 6 * 1024**3
MIN_AVAILABLE_RAM_AFTER_LOAD = 2 * 1024**3
EXPECTED_COHORT = 80
_STATE = {
    "started_monotonic": None, "started_at_utc": None,
    "requested_videos": 0, "requested_forwards": 0,
    "decoded_videos": 0, "effective_videos": 0, "attempted_forwards": 0,
    "completed_forwards": 0, "current_video_ordinal": None,
    "failed_video_ordinal": None, "phase": "not_started",
    "free_vram_at_failure_bytes": None,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def tensor_sha256(tensor, torch_module) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return sha256_bytes(value.view(torch_module.uint8).numpy().tobytes(order="C"))


def memory_snapshot() -> dict[str, int | None]:
    mem_available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1]) * 1024
            break
    if mem_available is None:
        raise ValueError("host_memory_available_unreadable")
    current = maximum = None
    cgroup = Path("/sys/fs/cgroup")
    try:
        current_raw = (cgroup / "memory.current").read_text().strip()
        max_raw = (cgroup / "memory.max").read_text().strip()
        current = int(current_raw)
        maximum = None if max_raw == "max" else int(max_raw)
    except (OSError, ValueError):
        # cgroup v1 or no cgroup limit: host MemAvailable still applies.
        pass
    cgroup_available = None if maximum is None or current is None else max(0, maximum - current)
    effective = mem_available if cgroup_available is None else min(mem_available, cgroup_available)
    return {"mem_available_bytes": mem_available,
            "cgroup_current_bytes": current,
            "cgroup_max_bytes": maximum,
            "effective_available_bytes": effective}


def sampled_decode_signature(metadata: dict[str, Any]) -> str:
    rows = metadata.get("sampled_frames")
    if not isinstance(rows, list) or len(rows) != FRAME_COUNT:
        raise ValueError("sampled_frame_metadata_count_invalid")
    canonical = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("sampled_frame_metadata_invalid")
        canonical.append({
            "decoded_frame_index": row.get("decoded_frame_index"),
            "relative_presentation_time": row.get("relative_presentation_time"),
            "rgb_pixel_sha256": row.get("rgb_pixel_sha256"),
        })
    return sha256_bytes(json.dumps(canonical, sort_keys=True,
        separators=(",", ":")).encode("ascii"))


def _read_private_plan(path: Path, private_root: Path) -> tuple[bytes, dict[str, Any]]:
    root = private_root.resolve(strict=True)
    if (path.is_symlink() or path.parent.resolve(strict=True) != root
            or stat.S_IMODE(root.stat().st_mode) != 0o700
            or root.stat().st_uid != os.getuid()):
        raise ValueError("cohort_plan_private_path_invalid")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size < 1 or before.st_size > 64 * 1024 * 1024):
            raise ValueError("cohort_plan_owner_mode_or_size_invalid")
        raw = stream.read(64 * 1024 * 1024 + 1)
        after = os.fstat(stream.fileno())
        if (len(raw) != after.st_size or before.st_ino != after.st_ino
                or before.st_mtime_ns != after.st_mtime_ns):
            raise ValueError("cohort_plan_changed_during_read")
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("cohort_plan_json_invalid") from exc
    return raw, plan


def _verify_frozen_plan(*, repo: Path, dataset: Path, video_root: Path,
                        plan_path: Path, expected_count: int):
    from freeze_qta_question_swap_cohort import (
        build_plan, csv_question_rows, prior_repeatability_exclusions,
    )

    private = repo / "data/private_nextgqa"
    raw, plan = _read_private_plan(plan_path, private)
    if (not isinstance(plan, dict)
            or plan.get("protocol") != "qta-question-swap-repeat-controlled-v1"
            or plan.get("status") != "frozen_local_question_swap_selector_diagnostic_not_faithfulness"
            or plan.get("authorization") != "non_authorizing_local_finite_cohort_diagnostic"
            or plan.get("requested_videos") != EXPECTED_COHORT
            or plan.get("effective_videos") != EXPECTED_COHORT
            or plan.get("new_question_eligibility_and_pair_selection_use_only_question_columns") is not True
            or plan.get("historical_repeatability_exclusion_used_annotation_validity") is not True
            or not isinstance(plan.get("fixed_records"), list)
            or len(plan["fixed_records"]) != EXPECTED_COHORT):
        raise ValueError("cohort_plan_status_or_count_invalid")
    records = plan["fixed_records"][:1] if expected_count == 1 else plan["fixed_records"]
    if len(records) != expected_count:
        raise ValueError("cohort_plan_smoke_or_full_count_invalid")
    excluded = plan.get("explicit_prior_capture_excluded_video_keys")
    if (not isinstance(excluded, list) or len(excluded) != 3
            or any(not isinstance(key, str) or not key for key in excluded)):
        raise ValueError("prior_capture_exclusion_list_invalid")
    verified_exclusions, verified_provenance = prior_repeatability_exclusions(
        repo, dataset, video_root)
    if (set(excluded) != verified_exclusions
            or plan.get("prior_capture_exclusion_provenance") != verified_provenance):
        raise ValueError("prior_capture_exclusion_reconstruction_mismatch")
    # Rebuild the exact outcome-blind cohort from the bound current inputs,
    # including every source/media byte hash, before model loading.
    rebuilt = build_plan(repo=repo, dataset=dataset, video_root=video_root,
                         requested_videos=EXPECTED_COHORT,
                         exclude_keys=set(excluded),
                         prior_exclusion_provenance=plan.get(
                             "prior_capture_exclusion_provenance"))
    if rebuilt != plan:
        raise ValueError("cohort_plan_rebuild_mismatch")
    if plan.get("source_fingerprints", {}).get("test_csv_selected_columns_sha256") != \
            csv_question_rows(dataset / "test.csv")[1]:
        raise ValueError("question_source_columns_changed")
    for overlap in plan.get("prior_media_overlap_audit", {}).values():
        if (not isinstance(overlap, dict) or type(overlap.get("overlap_count")) is not int
                or type(overlap.get("prior_count")) is not int
                or not isinstance(overlap.get("source_sha256"), str)
                or len(overlap["source_sha256"]) != 64):
            raise ValueError("prior_overlap_audit_receipt_invalid")
    return raw, plan, records


def _runtime_hash(directory: Path) -> str:
    digest = hashlib.sha256(b"qta-question-swap-model-snapshot-v1\0")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError("model_snapshot_empty")
    for path in files:
        rel = path.relative_to(directory).as_posix().encode("utf-8")
        file_hash = sha256_file(path).encode("ascii")
        digest.update(len(rel).to_bytes(4, "big")); digest.update(rel)
        digest.update(file_hash)
    return digest.hexdigest()


def _run(args) -> dict[str, Any]:
    import torch
    import transformers
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_model_module
    import transformers.models.qwen2_5_vl.processing_qwen2_5_vl as qwen_processor_module
    import transformers.models.qwen2_vl.video_processing_qwen2_vl as qwen_video_processor_module
    import qwen_vl_utils.vision_process as qwen_video_utils_module
    from importlib.metadata import version as package_version
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
    from qta_attention_forward_capture import (
        aggregate_captured_qta_rows, capture_qta_attention_forward_once,
    )
    from qwen_question_temporal_attention_selector import (
        map_question_token_positions, select_top_temporal_groups,
    )
    from run_nextgqa_real_video_gpu_smoke import video_key
    from smoke_qta_resolution_stability import _write_private, processor_group_support
    from smoke_qwen_real_video_attention_boost import (
        PROMPT_TEMPLATE, _make_batch, _processor_fingerprint,
        _temporal_visual_groups,
    )
    repo = Path(__file__).resolve().parents[1]
    dataset = repo / "NExT-GQA/datasets/nextgqa"
    video_root = args.video_root.resolve(strict=True)
    plan_path = (repo / "data/private_nextgqa" / args.plan_name)
    if Path(args.plan_name).name != args.plan_name:
        raise ValueError("plan_name_must_be_basename")
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        raise ValueError("pinned_checkpoint_not_cached")
    plan_raw, plan, records = _verify_frozen_plan(
        repo=repo, dataset=dataset, video_root=video_root,
        plan_path=plan_path, expected_count=1 if args.smoke else EXPECTED_COHORT)
    # Decode every frozen row before loading weights. Any failure preserves the
    # exact denominator and aborts the all-or-none run; there are no replacements.
    prepared = []
    decode_failures = 0
    _STATE["phase"] = "decode"
    for record in records:
        path = (video_root / record["video_relative_path"]).resolve(strict=True)
        if not path.is_relative_to(video_root) or path.is_symlink():
            raise ValueError("frozen_media_path_escape")
        if sha256_file(path) != record["media_sha256"]:
            raise ValueError("frozen_media_hash_changed")
        try:
            frames, metadata = decode_sampled_video(
                path, num_frames=FRAME_COUNT,
                uniform_tolerance_seconds=TOLERANCE)
            verify_sampled_frame_pixels(frames, metadata)
        except (ValueError, OSError):
            decode_failures += 1
            raise ValueError("frozen_video_decode_or_sampling_failure_no_replacement")
        prepared.append((record, path, sampled_decode_signature(metadata)))
        del frames, metadata
        _STATE["decoded_videos"] += 1
    if len(prepared) != len(records) or decode_failures:
        raise ValueError("frozen_cohort_decode_denominator_invalid")
    ram_before_load = memory_snapshot()
    if ram_before_load["effective_available_bytes"] < MIN_AVAILABLE_RAM_BEFORE_LOAD:
        raise ValueError("insufficient_available_ram_before_model_load")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("cuda_bf16_unavailable")
    device = torch.cuda.current_device()
    free_before, total_vram = torch.cuda.mem_get_info(device)
    if free_before < MIN_FREE_BEFORE_LOAD:
        raise ValueError("insufficient_free_vram_before_model_load")
    _STATE["phase"] = "model_load"
    model_hash_before = _runtime_hash(model_path)
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if (int(model.config.video_token_id) != int(processor.video_token_id)
            or len(model.model.language_model.layers) < max(LAYERS)):
        raise ValueError("model_processor_layer_contract_invalid")
    temporal_patch_size = int(processor.image_processor.temporal_patch_size)
    video_processor = getattr(processor, "video_processor", processor.image_processor)
    if int(video_processor.temporal_patch_size) != temporal_patch_size:
        raise ValueError("processor_temporal_patch_size_disagreement")
    config_patch = getattr(model.config.vision_config, "temporal_patch_size", None)
    if config_patch is None or int(config_patch) != temporal_patch_size:
        raise ValueError("model_processor_temporal_patch_size_disagreement")
    free_after_load, _ = torch.cuda.mem_get_info(device)
    if free_after_load < MIN_FREE_BEFORE_FORWARD:
        raise ValueError("insufficient_free_vram_after_model_load")
    ram_after_load = memory_snapshot()
    if ram_after_load["effective_available_bytes"] < MIN_AVAILABLE_RAM_AFTER_LOAD:
        raise ValueError("insufficient_available_ram_after_model_load")

    rng = torch.Generator(device="cpu"); rng.manual_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED); torch.cuda.manual_seed_all(RANDOM_SEED)
    contrasts = []
    private_rows = []
    pixel_hashes = []
    sampled_pts_hashes = []
    capture_row_hashes = []
    capture_shape_counts: dict[str, int] = {}
    capture_dtype_counts: dict[str, int] = {}
    verified_layer_call_receipts = 0
    attempted = completed = 0
    max_row_deviation = 0.0
    forward_seconds = 0.0
    source_paths = {
        "runner": Path(__file__),
        "cohort_freezer": repo / "scripts/freeze_qta_question_swap_cohort.py",
        "metrics": repo / "scripts/qta_question_swap_metrics.py",
        "decoder": repo / "scripts/claim_mask_video_decoder.py",
        "qwen_batch_adapter": repo / "scripts/smoke_qwen_real_video_attention_boost.py",
        "question_selector": repo / "scripts/qwen_question_temporal_attention_selector.py",
        "attention_capture": repo / "scripts/qta_attention_forward_capture.py",
        "attention_capture_contract": repo / "scripts/qta_attention_capture_contract.py",
        "attention_capture_sidecars": repo / "scripts/qta_attention_capture_sidecars.py",
        "candidate_exclusion_selector": repo / "scripts/run_nextgqa_real_video_gpu_smoke.py",
        "temporal_support": repo / "scripts/smoke_qta_resolution_stability.py",
        "qwen_model_implementation": Path(qwen_model_module.__file__),
        "qwen_processor_implementation": Path(qwen_processor_module.__file__),
        "qwen_video_processor": Path(qwen_video_processor_module.__file__),
        "qwen_video_utils": Path(qwen_video_utils_module.__file__),
    }
    source_hashes_before = {key: sha256_file(path) for key, path in source_paths.items()}
    annotation_column_hash = plan["source_fingerprints"]["test_csv_selected_columns_sha256"]
    mapping_hash = plan["source_fingerprints"]["video_mapping_sha256"]
    temporal_group_count = None
    start = time.monotonic()
    _STATE["phase"] = "forward_and_selector"

    for ordinal, (record, path, expected_decode_signature) in enumerate(prepared, 1):
        _STATE["current_video_ordinal"] = ordinal
        if sha256_file(path) != record["media_sha256"]:
            raise ValueError("frozen_video_media_changed_after_preflight")
        frames, metadata = decode_sampled_video(
            path, num_frames=FRAME_COUNT,
            uniform_tolerance_seconds=TOLERANCE)
        verify_sampled_frame_pixels(frames, metadata)
        if sampled_decode_signature(metadata) != expected_decode_signature:
            raise ValueError("frozen_video_decode_changed_after_preflight")
        fps = Fraction(*metadata["effective_sample_fps"])
        batch_a, video_metadata_a, rendered_a = _make_batch(
            frames=frames, sample_fps=fps,
            question=record["question_a"]["question"], processor=processor,
            torch_module=torch, frame_count=FRAME_COUNT)
        batch_b, video_metadata_b, rendered_b = _make_batch(
            frames=frames, sample_fps=fps,
            question=record["question_b"]["question"], processor=processor,
            torch_module=torch, frame_count=FRAME_COUNT)
        pixels_a, pixels_b = batch_a.get("pixel_values_videos"), batch_b.get("pixel_values_videos")
        if (not torch.is_tensor(pixels_a) or not torch.is_tensor(pixels_b)
                or not torch.equal(pixels_a, pixels_b)
                or not torch.equal(batch_a.get("video_grid_thw"), batch_b.get("video_grid_thw"))
                or video_metadata_a != video_metadata_b):
            raise ValueError("question_swap_visual_batch_mismatch")
        pixel_hash = tensor_sha256(pixels_a, torch)
        pixel_hashes.append(pixel_hash)
        selections = {}
        token_lengths = {}
        batch_hashes = {}
        for condition, batch, question_label, question in question_repeat_schedule(
                batch_a, batch_b, record["question_a"]["question"],
                record["question_b"]["question"]):
            rendered = rendered_a if question_label == "A" else rendered_b
            ids = batch.get("input_ids")
            grid = batch.get("video_grid_thw")
            if (not torch.is_tensor(ids) or ids.ndim != 2 or tuple(ids.shape[:1]) != (1,)
                    or not torch.is_tensor(grid) or tuple(grid.shape) != (1, 3)):
                raise ValueError("processor_batch_shape_invalid")
            input_length = int(ids.shape[1])
            if int(batch["attention_mask"].sum()) != input_length:
                raise ValueError("unexpected_padding_not_supported")
            temporal_steps, grid_h, grid_w = map(int, grid[0].tolist())
            if temporal_steps * temporal_patch_size != FRAME_COUNT:
                raise ValueError("processor_temporal_patch_count_invalid")
            group_positions = _temporal_visual_groups(
                token_ids=[int(x) for x in ids[0].tolist()],
                video_token_id=int(processor.video_token_id),
                temporal_steps=temporal_steps, grid_h=grid_h, grid_w=grid_w,
                merge_size=int(processor.image_processor.merge_size))
            if temporal_group_count is None:
                temporal_group_count = len(group_positions)
            elif len(group_positions) != temporal_group_count:
                raise ValueError("temporal_group_count_changed")
            qpos = map_question_token_positions(
                tokenizer=processor.tokenizer, rendered_text=rendered,
                question_text=question, processed_input_ids=ids,
                video_token_id=int(processor.video_token_id))
            token_lengths[question_label] = len(qpos)
            batch_hashes[condition] = sha256_bytes(
                json.dumps({"input_ids": ids.detach().cpu().tolist(),
                            "visual_pixels": pixel_hash,
                            "video_grid": grid.detach().cpu().tolist()},
                           sort_keys=True, separators=(",", ":")).encode("utf-8"))
            if condition.endswith("2"):
                # The repeat must reuse the exact same batch object and prompt.
                if batch is not (batch_a if condition == "A2" else batch_b):
                    raise ValueError("repeat_batch_object_not_reused")
            free_before_forward = torch.cuda.mem_get_info(device)[0]
            if free_before_forward < MIN_FREE_BEFORE_FORWARD:
                _STATE["free_vram_at_failure_bytes"] = int(free_before_forward)
                raise ValueError("insufficient_free_vram_before_forward")
            attempted += 1
            _STATE["attempted_forwards"] = attempted
            capture_start = time.monotonic()
            output, rows, report = capture_qta_attention_forward_once(
                model=model,
                forward=lambda current=batch: current_model_forward(
                    model, current),
                question_positions=qpos, input_length=input_length,
                torch_module=torch)
            torch.cuda.synchronize()
            forward_seconds += time.monotonic() - capture_start
            completed += 1
            _STATE["completed_forwards"] = completed
            if (report.get("status") != "one_forward_attention_rows_captured_non_authorizing"
                    or report.get("forward_calls") != 1
                    or report.get("attention_rows_satisfy_source_dtype_normalization_tolerance") is not True):
                raise ValueError("attention_capture_report_invalid")
            expected_layers = list(LAYERS)
            layer_calls = report.get("layer_calls")
            shape = report.get("attention_shape")
            source_dtypes = report.get("attention_source_dtypes")
            if (not isinstance(shape, list) or len(shape) != 4
                    or any(type(value) is not int or value < 1 for value in shape)
                    or shape[0] != len(LAYERS) or shape[1] < 1
                    or shape[2] != len(qpos) or shape[3] != input_length
                    or not isinstance(layer_calls, dict)
                    or layer_calls != {str(layer): 1 for layer in expected_layers}
                    or not isinstance(source_dtypes, dict)
                    or set(source_dtypes) != {str(layer) for layer in expected_layers}
                    or len(set(source_dtypes.values())) != 1
                    or report.get("layer_indices_1_based") != expected_layers):
                raise ValueError("attention_capture_shape_dtype_or_layer_receipt_invalid")
            shape_key = "x".join(str(value) for value in shape)
            dtype_key = next(iter(source_dtypes.values()))
            capture_shape_counts[shape_key] = capture_shape_counts.get(shape_key, 0) + 1
            capture_dtype_counts[dtype_key] = capture_dtype_counts.get(dtype_key, 0) + 1
            verified_layer_call_receipts += 1
            max_row_deviation = max(max_row_deviation,
                                    float(report["attention_row_sum_max_abs_deviation"]))
            row_payload = b"".join(
                rows[layer].detach().to(device="cpu", dtype=torch.float32)
                .contiguous().numpy().tobytes(order="C") for layer in LAYERS)
            capture_row_hashes.append(sha256_bytes(row_payload))
            scores = aggregate_captured_qta_rows(
                rows=rows, group_positions=group_positions, torch_module=torch)
            if any(not math.isfinite(value) or value < 0 for value in scores):
                raise ValueError("selector_scores_invalid")
            selected = select_top_temporal_groups(
                scores, top_k=max(1, math.ceil(VISUAL_RATIO * len(group_positions))))
            selections[condition] = tuple(sorted(selected))
            # The forward's token logits and generated predictions are never read.
            del output, rows, scores
            # Eager attention may leave large freed blocks in the CUDA caching
            # allocator. Return them before the next repeat so free-VRAM gates
            # reflect live model state rather than prior forward peak caches.
            torch.cuda.empty_cache()
        if set(selections) != {"A1", "A2", "B1", "B2"}:
            raise ValueError("question_swap_forward_set_incomplete")
        contrast = video_contrast(selections, group_count=temporal_group_count)
        contrasts.append(contrast)
        _STATE["effective_videos"] = len(contrasts)
        pts_payload = json.dumps(
            [row["relative_presentation_time"] for row in metadata["sampled_frames"]],
            sort_keys=True, separators=(",", ":")).encode("ascii")
        sampled_pts_hashes.append(sha256_bytes(pts_payload))
        private_rows.append({"fixed_video_ordinal": ordinal,
                             **contrast,
                             "question_a_token_count": token_lengths["A"],
                             "question_b_token_count": token_lengths["B"],
                             "pixel_tensor_sha256": pixel_hash,
                             "batch_sha256": batch_hashes,
                             "selected_group_sets": {k: list(v) for k, v in selections.items()}})
        del batch_a, batch_b, pixels_a, pixels_b, frames
        torch.cuda.empty_cache()
        _STATE["current_video_ordinal"] = None

    model_hash_after = _runtime_hash(model_path)
    if model_hash_before != model_hash_after:
        raise ValueError("model_snapshot_changed_during_run")
    if (sha256_file(Path(__file__)) != source_hashes_before["runner"]
            or any(sha256_file(path) != source_hashes_before[name]
                   for name, path in source_paths.items())):
        raise ValueError("source_files_changed_during_run")
    metric_summary = summarize_video_contrasts(contrasts)
    payload = {
        "schema_version": 1,
        "purpose": "local_qta_question_swap_repeat_controlled_selector_diagnostic_not_faithfulness",
        "status": "passed",
        "started_at_utc": _STATE["started_at_utc"],
        "smoke_only": bool(args.smoke),
        "cohort_plan_sha256": sha256_bytes(plan_raw),
        "source_sha256": source_hashes_before,
        "dataset_split": "NExT-GQA test; test-associated, prior-exposure and source/event independence unresolved",
        "test_csv_selected_question_columns_sha256": annotation_column_hash,
        "video_mapping_sha256": mapping_hash,
        "prior_media_overlap_audit": plan["prior_media_overlap_audit"],
        "prior_media_overlap_scope": "exact local cohort and pool manifest byte snapshots; not an authenticated historical-universe or freshness claim",
        "model_revision": MODEL_REVISION,
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "model_snapshot_content_sha256_before": model_hash_before,
        "model_snapshot_content_sha256_after": model_hash_after,
        "processor_fingerprint_sha256": _processor_fingerprint(processor),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "cuda_version": str(torch.version.cuda),
        "device_name": torch.cuda.get_device_name(device),
        "precision": "bf16_weights_4bit_bf16_compute_eager_attention",
        "random_seed": RANDOM_SEED,
        "requested_videos": 1 if args.smoke else EXPECTED_COHORT,
        "effective_videos": len(contrasts),
        "failed_videos": (1 if args.smoke else EXPECTED_COHORT) - len(contrasts),
        **forward_accounting(4 * (1 if args.smoke else EXPECTED_COHORT), attempted, completed),
        "frame_count": FRAME_COUNT,
        "uniform_pts_tolerance_seconds": [TOLERANCE.numerator, TOLERANCE.denominator],
        "selector": "untreated question-token attention averaged over heads/layers 5-16; top ceil(0.10*T) temporal groups",
        "question_pair_repeat_control": "A1,A2,B1,B2; each prompt reuses one exact processor batch object twice; A/B visual pixel tensors, video metadata, and processor grid required exactly equal",
        "primary_contrast": "mean video-level [J_repeat - J_cross]; J_repeat=mean(J(A1,A2),J(B1,B2)); J_cross=mean of all four A-to-B Jaccards",
        "metrics": metric_summary,
        "fixed_ordinal_video_metrics_local_only": private_rows,
        "group_count": temporal_group_count,
        "visual_tensor_hash_aggregate_sha256": sha256_bytes("".join(sorted(pixel_hashes)).encode("ascii")),
        "sampled_pts_hash_aggregate_sha256": sha256_bytes("".join(sorted(sampled_pts_hashes)).encode("ascii")),
        "attention_capture_row_hash_aggregate_sha256": sha256_bytes("".join(capture_row_hashes).encode("ascii")),
        "attention_capture_receipts_verified": completed,
        "attention_capture_shape_counts": capture_shape_counts,
        "attention_capture_source_dtype_counts": capture_dtype_counts,
        "attention_capture_all_expected_layer_calls_once": verified_layer_call_receipts == completed,
        "max_attention_row_sum_deviation": max_row_deviation,
        "free_vram_before_model_load_gib": free_before / 1024**3,
        "free_vram_after_model_load_gib": free_after_load / 1024**3,
        "host_and_cgroup_memory_before_model_load": ram_before_load,
        "host_and_cgroup_memory_after_model_load": ram_after_load,
        "total_vram_gib": total_vram / 1024**3,
        "full_elapsed_seconds": time.monotonic() - start,
        "forward_and_capture_seconds": forward_seconds,
        "answers_choices_new_gold_intervals_claims_generated_text_and_prediction_logits_used": False,
        "historical_prior_capture_exclusion_used_annotation_validity_only": True,
        "faithfulness_result": False,
        "rights_status": "unverified_local_only",
        "local_only": True,
    }
    output = repo / "data/private_nextgqa" / args.output_name
    if Path(args.output_name).name != args.output_name:
        raise ValueError("output_name_must_be_basename")
    _write_private(output, payload)
    return payload


def current_model_forward(model, batch):
    return model(**batch, use_cache=False, logits_to_keep=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-name", required=True,
                        help="frozen private cohort plan basename")
    parser.add_argument("--output-name", required=True,
                        help="new private result basename; overwrite refused")
    parser.add_argument("--video-root", type=Path,
                        default=Path.home() / "nextgqa_sample" / "NExTVideo")
    parser.add_argument("--smoke", action="store_true",
                        help="run only the first frozen video (4 forwards), no replacement")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    private = repo / "data/private_nextgqa"
    if (Path(args.plan_name).name != args.plan_name
            or Path(args.output_name).name != args.output_name):
        parser.error("plan_and_output_names_must_be_basenames")
    output_path = private / args.output_name
    if output_path.exists() or output_path.is_symlink():
        parser.error("output_exists_choose_unique_name")
    expected_videos = 1 if args.smoke else EXPECTED_COHORT
    _STATE.update(started_monotonic=time.monotonic(),
                  started_at_utc=datetime.now(timezone.utc).isoformat(),
                  requested_videos=expected_videos,
                  requested_forwards=4 * expected_videos,
                  decoded_videos=0, effective_videos=0,
                  attempted_forwards=0, completed_forwards=0,
                  current_video_ordinal=None, failed_video_ordinal=None,
                  free_vram_at_failure_bytes=None,
                  phase="preflight")
    try:
        payload = _run(args)
    except Exception as exc:
        detail = str(exc)
        code = detail if re.fullmatch(r"[a-zA-Z0-9_]+", detail) else \
            f"local_run_failed_{type(exc).__name__}"
        _STATE["failed_video_ordinal"] = _STATE["current_video_ordinal"]
        failure = {
            "schema_version": 1,
            "purpose": "local_qta_question_swap_selector_diagnostic_failure_not_faithfulness",
            "status": "failed",
            "failure_code": code,
            "phase": _STATE["phase"],
            "smoke_only": bool(args.smoke),
            "requested_videos": _STATE["requested_videos"],
            "decoded_videos": _STATE["decoded_videos"],
            "effective_videos": _STATE["effective_videos"],
            "failed_or_incomplete_video_ordinal": _STATE["failed_video_ordinal"],
            "free_vram_at_failure_bytes": _STATE["free_vram_at_failure_bytes"],
            **forward_accounting(_STATE["requested_forwards"],
                                 _STATE["attempted_forwards"],
                                 _STATE["completed_forwards"]),
            "full_elapsed_seconds": (time.monotonic() - _STATE["started_monotonic"]
                                     if _STATE["started_monotonic"] is not None else 0.0),
            "faithfulness_result": False,
            "answers_choices_new_gold_intervals_claims_generated_text_and_prediction_logits_used": False,
            "rights_status": "unverified_local_only",
            "local_only": True,
        }
        try:
            from freeze_qta_question_swap_cohort import write_private_exclusive
            write_private_exclusive(output_path, private, failure)
            artifact = "written"
        except Exception:
            artifact = "not_written_path_unsafe_or_already_exists"
        print(json.dumps({"status": "failed", "failure_code": code,
            "failure_artifact": artifact,
            "requested_videos": failure["requested_videos"],
            "effective_videos": failure["effective_videos"],
            "requested_forwards": failure["requested_forwards"],
            "attempted_forwards": failure["attempted_forwards"],
            "completed_forwards": failure["completed_forwards"]}, sort_keys=True))
        return 2
    print(json.dumps({"status": payload["status"], "smoke_only": payload["smoke_only"],
        "requested_videos": payload["requested_videos"], "effective_videos": payload["effective_videos"],
        "requested_forwards": payload["requested_forwards"], "completed_forwards": payload["completed_forwards"],
        "metrics": payload["metrics"], "faithfulness_result": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
