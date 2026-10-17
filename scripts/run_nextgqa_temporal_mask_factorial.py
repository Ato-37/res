#!/usr/bin/env python3
"""Local exploratory 2x3 temporal-orientation by black-mask sensitivity run.

Uses the exact hash-locked NExT-GQA media/QID assignments (ten by default;
one explicitly locked assignment for a smoke run). Conditions are
chronological/reversed crossed with full, black gold-evidence mask, and black
same-count control mask. This post-hoc fixed-cohort diagnostic makes no
population, confidence-interval, or p-value claim. All output stays local.
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
import random
import subprocess
import time

import torch
from PIL import Image

import run_nextgqa_gold_interval_margin_pilot as base

FRAMES = 32
N = 10
ORIENTATIONS = ("chronological", "reversed")
MASKS = ("full", "black_gold_mask", "black_control_mask")
CONDITIONS = tuple(f"{orientation}_{mask}" for orientation in ORIENTATIONS
                   for mask in MASKS)
SOURCE_PATHS = (
    "scripts/run_nextgqa_temporal_mask_factorial.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/necessity_choice_margin_utils.py",
    "scripts/presentation_time_utils.py",
)
PURPOSE = "exploratory_nextgqa_temporal_orientation_mask_factorial"


def orient_masked_frames(frames, positions, fill="black"):
    """Mask original source positions first, then optionally reverse the list."""
    masked_positions = set(positions)
    transformed = [
        (base.apply_mask_fill(frame, fill, base.BLUR_RADIUS)
         if index in masked_positions else frame.copy())
        for index, frame in enumerate(frames)
    ]
    if fill != "black":
        raise ValueError("factorial_fill_must_be_black")
    return transformed


def orient_completed_frames(frames, orientation):
    """Reverse already-created frames; processor time metadata stays ascending."""
    if orientation == "chronological":
        return [frame.copy() for frame in frames]
    if orientation == "reversed":
        return [frame.copy() for frame in reversed(frames)]
    raise ValueError("unknown_temporal_orientation")


def source_mask_then_orient(frames, source_positions, orientation):
    masked = orient_masked_frames(frames, source_positions)
    return orient_completed_frames(masked, orientation)


def frame_signatures(frames):
    """Per-frame RGB mean signatures used for a readable pixel/order check."""
    import numpy as np
    return [tuple(float(x) for x in np.asarray(frame).astype("float64").mean((0, 1)))
            for frame in frames]


def validate_signature_order(source_frames, processed_tensor, orientation,
                             atol=2.0):
    """Require processor signatures to match the intended exact source order."""
    expected = frame_signatures(orient_completed_frames(source_frames, orientation))
    actual = processed_tensor.float().mean(dim=(-1, -2)).detach().cpu().tolist()
    if len(actual) != len(expected) or any(len(a) != len(e) for a, e in zip(actual, expected)):
        raise ValueError("factorial_frame_signature_shape_mismatch")
    if any(abs(float(a[c]) - e[c]) > atol for a, e in zip(actual, expected)
           for c in range(3)):
        raise ValueError("factorial_frame_pixel_identity_or_order_mismatch")


def processor_frame_hashes(processed_tensor):
    """Cryptographically bind each processor tensor frame including pixel order."""
    frames = processed_tensor.detach().cpu().contiguous()
    if frames.ndim != 4:
        raise ValueError("factorial_processor_frame_tensor_shape_invalid")
    digests = []
    for index in range(frames.shape[0]):
        frame = frames[index].contiguous()
        header = f"{frame.dtype}:{tuple(frame.shape)}:".encode("ascii")
        raw = frame.view(torch.uint8).numpy().tobytes()
        digests.append(hashlib.sha256(header + raw).hexdigest())
    return digests


def validate_reversal_hash_pairs(frame_hashes):
    """Each reversed-cell tensor sequence must be the exact reverse of its pair."""
    for mask in MASKS:
        chronological = frame_hashes.get(f"chronological_{mask}")
        reversed_hashes = frame_hashes.get(f"reversed_{mask}")
        if (not isinstance(chronological, list) or not isinstance(reversed_hashes, list)
                or reversed_hashes != list(reversed(chronological))):
            raise ValueError("factorial_reversed_processor_hash_mismatch")


def require_identical_prompts(prompts):
    if len(prompts) != len(CONDITIONS) or len(set(prompts.values())) != 1:
        raise ValueError("factorial_prompt_text_mismatch")


def condition_order(sample_index):
    """Deterministic cyclic counterbalance of the six inference positions."""
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("factorial_sample_index_invalid")
    offset = sample_index % len(CONDITIONS)
    return CONDITIONS[offset:] + CONDITIONS[:offset]


def validate_source_binding_receipt(receipt, lock_sha256, assignments):
    if (not isinstance(receipt, dict)
            or receipt.get("schema_version") != 1
            or receipt.get("purpose") != "private_source_binding_for_locked_32f_exploratory_run"
            or receipt.get("lock_sha256") != lock_sha256
            or not isinstance(receipt.get("locked_media"), list)
            or not isinstance(receipt.get("archive_sha256"), str)
            or len(receipt["archive_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in receipt["archive_sha256"])
            or not isinstance(receipt.get("source_commit"), str)
            or len(receipt["source_commit"]) != 40
            or any(char not in "0123456789abcdef" for char in receipt["source_commit"])):
        raise ValueError("source_binding_receipt_contract_invalid")
    receipt_media = []
    receipt_members = []
    for item in receipt["locked_media"]:
        if (not isinstance(item, dict)
                or set(item) != {"archive_member", "extracted_bytes_sha256", "media_sha256"}):
            raise ValueError("source_binding_receipt_media_row_invalid")
        media_hash = item.get("media_sha256")
        extracted_hash = item.get("extracted_bytes_sha256")
        member = item.get("archive_member")
        if (not isinstance(media_hash, str) or len(media_hash) != 64
                or any(char not in "0123456789abcdef" for char in media_hash)
                or extracted_hash != media_hash
                or not isinstance(member, str) or member.startswith("/")
                or ".." in Path(member).parts
                or not member.startswith("NExTVideo/") or not member.endswith(".mp4")):
            raise ValueError("source_binding_receipt_media_row_invalid")
        receipt_media.append(media_hash)
        receipt_members.append(member)
    expected_media = [item["media_sha256"] for item in assignments]
    if (len(receipt_media) != len(receipt["locked_media"])
            or len(set(receipt_media)) != len(receipt_media)
            or len(set(receipt_members)) != len(receipt_members)
            or sorted(receipt_media) != sorted(expected_media)):
        raise ValueError("source_binding_receipt_media_set_mismatch")
    return receipt_media


def validate_private_output_path(path, repo):
    output = Path(path)
    if not output.is_absolute():
        output = Path(repo) / output
    output = Path(os.path.abspath(output))
    private_root = (Path(repo) / "data/private_nextgqa").resolve(strict=True)
    try:
        resolved_parent = output.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("output_parent_must_exist_in_private_directory") from exc
    if resolved_parent != private_root or output.parent != resolved_parent:
        raise ValueError("output_must_be_in_private_nextgqa_directory")
    if os.path.lexists(output):
        raise ValueError("refusing_to_overwrite_output")
    return output


def read_private_preflight(path, repo):
    gate_path = Path(path)
    if not gate_path.is_absolute():
        gate_path = Path(repo) / gate_path
    gate_path = Path(os.path.abspath(gate_path))
    private_root = (Path(repo) / "data/private_nextgqa").resolve(strict=True)
    if gate_path.parent.resolve(strict=True) != private_root or gate_path.parent != private_root:
        raise ValueError("preflight_must_be_in_private_nextgqa_directory")
    return base.read_private_json_snapshot(gate_path)


def require_complete_factorial(records, failures, requested=N):
    """Return only complete 6-cell sample sets; reject all partial aggregates."""
    if failures or len(records) != requested:
        raise ValueError("factorial_requires_all_samples_complete")
    for record in records:
        if set(record.get("margins", {})) != set(CONDITIONS):
            raise ValueError("factorial_requires_six_conditions_per_sample")
    return records


def derive_factorial_effects(margins):
    """Return per-orientation control-minus-gold and chrono-minus-reverse."""
    if set(margins) != set(CONDITIONS):
        raise ValueError("factorial_margin_cells_invalid")
    chrono = margins["chronological_black_control_mask"] - margins[
        "chronological_black_gold_mask"]
    reverse = margins["reversed_black_control_mask"] - margins["reversed_black_gold_mask"]
    return {
        "chronological_control_minus_gold": chrono,
        "reversed_control_minus_gold": reverse,
        "interaction_chronological_minus_reversed": chrono - reverse,
    }


def verify_committed_sources(repo):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                     text=True).strip()
    hashes = {}
    for name in SOURCE_PATHS:
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=repo)
        current = (repo / name).read_bytes()
        digest = hashlib.sha256(current).hexdigest()
        if digest != hashlib.sha256(committed).hexdigest():
            raise ValueError(f"uncommitted_research_source:{name}")
        hashes[name] = digest
    return commit, hashes


def prepare_six_conditions(sample, processor, torch_module):
    from qwen_vl_utils import process_vision_info
    prepared = {}
    signatures = {}
    prompts = {}
    processor_hashes = {}
    for orientation in ORIENTATIONS:
        for mask in MASKS:
            positions = (() if mask == "full" else sample[
                "evidence_positions" if mask == "black_gold_mask" else "control_positions"])
            source = source_mask_then_orient(sample["frames"], positions, orientation)
            # Original frame index/fps metadata are intentionally retained even for reversed pixels.
            fps = sample["decode"]["effective_sample_fps"]
            text, messages = base.build_qa_prompt(processor, sample, source, fps)
            images, videos = process_vision_info(messages, return_video_metadata=True)
            if images is not None or len(videos) != 1:
                raise RuntimeError("qwen_video_utility_contract_failed")
            pixels, metadata = videos[0]
            if (not isinstance(pixels, torch_module.Tensor) or pixels.shape[0] != FRAMES
                    or metadata.get("frames_indices") != list(range(FRAMES))
                    or not math.isclose(float(metadata.get("fps", 0)),
                                         fps[0] / fps[1], rel_tol=1e-9)):
                raise RuntimeError("qwen_video_metadata_mismatch")
            validate_signature_order(source, pixels, "chronological")
            processor_hashes[f"{orientation}_{mask}"] = processor_frame_hashes(pixels)
            prompts[f"{orientation}_{mask}"] = text
            batch = base.process_qa_video(processor, text, images, pixels, metadata)
            grid = batch["video_grid_thw"][0].tolist()
            seconds_per_grid = float(batch["second_per_grid_ts"][0])
            expected_grid = FRAMES // int(processor.video_processor.temporal_patch_size)
            if grid[0] != expected_grid:
                raise RuntimeError("qwen_temporal_grid_mismatch")
            if abs(seconds_per_grid - 2 / (fps[0] / fps[1])) > 1e-6:
                raise RuntimeError("qwen_second_per_grid_mismatch")
            condition = f"{orientation}_{mask}"
            prepared[condition] = batch
            signatures[condition] = {"grid": grid, "seconds_per_grid": seconds_per_grid}
    require_identical_prompts(prompts)
    validate_reversal_hash_pairs(processor_hashes)
    if len({(tuple(v["grid"]), v["seconds_per_grid"]) for v in signatures.values()}) != 1:
        raise RuntimeError("factorial_condition_geometry_mismatch")
    return prepared, signatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "score"), required=True)
    parser.add_argument("--output", required=True, help="new private local JSON path")
    parser.add_argument("--preflight-summary")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--n", type=int, choices=(1, 10), default=N,
                        help="one explicitly locked sample for smoke, or the exact ten-row cohort")
    parser.add_argument("--cohort-lock",
                        help="private hash-only lock; required explicitly when --n 1")
    parser.add_argument("--exclude-media-sha256", default="data/private_nextgqa/exclude_prior_goldpilot_smoke_20261017_02.json")
    parser.add_argument("--candidate-pair-filter", default="data/private_nextgqa/zip_gold_candidates_20261017_01/eligible_pairs.json")
    parser.add_argument("--source-binding-receipt",
        default="data/private_nextgqa/answer_margin_fresh10_32f_source_binding_20261017_resume_01.json")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args()
    if args.n == 1 and not args.cohort_lock:
        parser.error("--n 1 requires an explicit one-assignment --cohort-lock")
    cohort_lock_arg = args.cohort_lock or "data/private_nextgqa/answer_margin_fresh10_lock_20261017_01.json"
    if not args.user_authorized_local_only:
        parser.error("explicit local-only authorization flag required")
    if args.mode == "score" and not args.preflight_summary:
        parser.error("score mode requires an outcome-blind --preflight-summary")
    repo = Path(__file__).resolve().parents[1]
    try:
        output = validate_private_output_path(args.output, repo)
        source_commit, source_hashes = verify_committed_sources(repo)
        lock, lock_sha = base.read_private_json_snapshot(Path(cohort_lock_arg))
        assignments = base.validate_cohort_lock_payload(lock, args.n)
        source_receipt, source_receipt_sha = base.read_private_json_snapshot(
            Path(args.source_binding_receipt))
        validate_source_binding_receipt(source_receipt, lock_sha, assignments)
        exclude_payload, exclude_sha = base.read_private_json_snapshot(Path(args.exclude_media_sha256))
        excluded = base.validate_private_media_exclusion(exclude_payload)
        filter_payload, filter_sha = base.read_private_json_snapshot(Path(args.candidate_pair_filter))
        allowed = base.validate_candidate_pair_filter(filter_payload)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(f"factorial_input_or_source_gate_failed:{exc}")
    base.MAX_VIDEOS = args.n
    selected, counts, candidate_meta, cohort_hash = base.freeze_cohort(
        repo, Path(args.video_root).resolve(), excluded, FRAMES, assignments, allowed)
    base.validate_cohort_cardinality(args.n, len(selected))
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / base.MODEL_REVISION)
    if not model_path.is_dir():
        parser.error("pinned local checkpoint missing; downloads disabled")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA and BF16 required; CPU fallback disabled")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    host_free = base.host_available_bytes()
    if free_bytes < base.MIN_FREE_VRAM or host_free < base.MIN_HOST_AVAILABLE:
        parser.error("fixed VRAM/host-memory preflight failed")
    random.seed(base.SEED)
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)
    started = datetime.now(timezone.utc).isoformat()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    token_ids = []
    for label in base.LABELS:
        ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            parser.error(f"choice_label_tokenization_invalid:{label}")
        token_ids.append(ids[0])
    if len(set(token_ids)) != 5:
        parser.error("choice_labels_not_distinct_tokens")
    grids = []
    for sample in selected:
        batches, geometry = prepare_six_conditions(sample, processor, torch)
        grids.append(next(iter(geometry.values()))["grid"])
        del batches
        gc.collect()
    binding = {
        "schema_version": 1, "purpose": PURPOSE, "status": "passed",
        "source_commit": source_commit, "source_file_sha256": source_hashes,
        "cohort_lock_sha256": lock_sha, "exclusion_file_sha256": exclude_sha,
        "source_binding_receipt_sha256": source_receipt_sha,
        "exclusion_media_digest_sha256": base._hash(sorted(excluded)),
        "candidate_filter_sha256": filter_sha,
        "candidate_filter_digest_sha256": base._hash(sorted(allowed)),
        "candidate_input_sha256": {name: base.sha256_file(candidate_meta["dataset"] / name)
            for name in ("test.csv", "gsub_test.json", "map_vid_vidorID.json")},
        "cohort_sha256": cohort_hash, "requested_samples": args.n,
        "effective_samples": len(selected), "requested_frames": FRAMES,
        "effective_frames": [len(item["frames"]) for item in selected],
        "requested_forwards": args.n * len(CONDITIONS), "conditions": list(CONDITIONS),
        "inference_order_contract": "cyclic rotation of six fixed cells by sample_index modulo six",
        "condition_grids": grids, "candidate_video_count": candidate_meta["candidate_video_count"],
        "candidate_videos_scanned": counts["candidate_scan_count"],
        "prior_cohort_media_excluded": counts["prior_cohort_media_excluded"],
        "model_revision": base.MODEL_REVISION,
        "model_config_sha256": base.sha256_file(model_path / "config.json"),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")},
        "seed": base.SEED,
        "orientation_contract": "mask original source indices, then reverse completed pixel-frame list; processor indices/fps remain ascending/original",
        "effect_contract": "for each orientation: black-control margin minus black-gold-mask margin; interaction: chronological effect minus reversed effect",
        "statistical_scope": "post-hoc exploratory fixed cohort; no CI, p-value, or population claim",
    }
    if args.mode == "preflight":
        payload = dict(binding, mode="outcome_blind_preflight", started_utc=started,
            gpu_free_before_gib=free_bytes / 1024**3, gpu_total_gib=total_bytes / 1024**3,
            host_available_gib=host_free / 1024**3,
            authorization_scope="user-authorized local-only; rights status unverified",
            privacy="No raw QA, media, IDs, absolute paths, logits, or generated text.")
        base.write_private_json_exclusive(output, payload)
        print(f"Factorial preflight passed: {args.n} samples, {args.n*len(CONDITIONS)} planned forwards; {output.name}")
        return
    try:
        gate, gate_sha = read_private_preflight(args.preflight_summary, repo)
        gate_path = Path(args.preflight_summary)
        if not gate_path.is_absolute():
            gate_path = repo / gate_path
        gate_path = Path(os.path.abspath(gate_path))
    except (OSError, ValueError) as exc:
        parser.error(f"preflight_summary_invalid:{exc}")
    for key, value in binding.items():
        if gate.get(key) != value:
            parser.error(f"frozen_preflight_binding_mismatch:{key}; no score was run")
    if (gate.get("mode") != "outcome_blind_preflight"
            or gate.get("status") != "passed"):
        parser.error("frozen_preflight_not_passed; no score was run")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if int(model.config.video_token_id) != int(processor.video_token_id):
        parser.error("processor_checkpoint_video_token_mismatch")
    label_tensor = torch.tensor(token_ids, device="cuda")
    records, failures, attempted = [], [], 0
    inference_orders = []
    torch.cuda.reset_peak_memory_stats()
    for sample_index, sample in enumerate(selected):
        metrics = {}
        failed = None
        batches = None
        try:
            batches, geometry = prepare_six_conditions(sample, processor, torch)
            if next(iter(geometry.values()))["grid"] != gate["condition_grids"][sample_index]:
                raise RuntimeError("processor_grid_changed_after_preflight")
            order = condition_order(sample_index)
            inference_orders.append(list(order))
            for condition in order:
                failed = condition
                if base.host_available_bytes() < base.MIN_HOST_AVAILABLE:
                    raise RuntimeError("per_forward_host_memory_guard_failed")
                if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
                    raise RuntimeError("per_forward_free_vram_guard_failed")
                attempted += 1
                batch = batches[condition].to("cuda")
                with torch.inference_mode():
                    out = model(**batch, use_cache=False, logits_to_keep=1)
                logits = out.logits[0, -1].float()
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("nonfinite_next_token_logits")
                metrics[condition] = torch.log_softmax(logits, dim=-1)[label_tensor].cpu().tolist()
                del batch, out, logits
                torch.cuda.empty_cache()
            margins = {key: base.gold_vs_best_foil_margin(value, sample["gold"])
                       for key, value in metrics.items()}
            effects = derive_factorial_effects(margins)
            records.append({"sample_index": sample_index, "margins": margins,
                "effects": effects, "evidence_frame_count": len(sample["evidence_positions"]),
                "control_frame_count": len(sample["control_positions"])})
        except Exception as exc:
            failures.append({"sample_index": sample_index, "failure_type": type(exc).__name__,
                             "failed_condition": failed, "completed_conditions": sorted(metrics)})
            break
        finally:
            del batches
            gc.collect()
    sync_passed, discarded = base.finalize_cuda_scoring(torch, records, failures)
    complete = not failures and len(records) == args.n and sync_passed
    if complete:
        try:
            require_complete_factorial(records, failures, args.n)
        except ValueError:
            complete = False
            records.clear()
    discarded_incomplete_records = 0
    if not complete:
        discarded_incomplete_records = len(records)
        records.clear()
    payload = dict(binding, mode="score", status="passed" if complete else "partial",
        started_utc=started, records=records, failures=failures,
        attempted_model_forwards=attempted, completed_full_condition_sets=len(records),
        inference_orders=inference_orders,
        final_cuda_synchronize_passed=sync_passed,
        discarded_records_after_sync_failure=discarded,
        discarded_incomplete_records=discarded_incomplete_records,
        preflight_summary_sha256=gate_sha,
        effect_definition=binding["effect_contract"], statistical_test="none",
        p_value=None, ci=None,
        gpu=torch.cuda.get_device_name(0) if sync_passed else None,
        gpu_free_before_gib=free_bytes / 1024**3, gpu_total_gib=total_bytes / 1024**3,
        host_available_before_gib=host_free / 1024**3,
        cuda_version=torch.version.cuda,
        versions={name: importlib.metadata.version(name) for name in
                  ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")},
        authorization_scope="user-authorized local-only; rights status unverified",
        privacy="No raw QA, media, IDs, absolute paths, logits, or generated text.")
    base.write_private_json_exclusive(output, payload)
    print(f"Factorial score {payload['status']}: {len(records)}/{args.n} complete sets, "
          f"{attempted}/{args.n*len(CONDITIONS)} forwards; {output.name}")
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
