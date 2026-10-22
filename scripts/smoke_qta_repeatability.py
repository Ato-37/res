#!/usr/bin/env python3
"""Local-only repeatability smoke for the frozen 32-frame QTA selector.

Run two untreated attention captures on the same model.eval processor batch
for each fixed NExT-GQA video/question. `--smoke` uses only the first frozen
candidate (2 forwards); the full exploratory run uses three (6 forwards).
Outputs contain fixed-ordinal metrics and non-authorizing hook-call receipts
bound to hashes of captured attention rows. No IDs, questions, answers, group
selections, logits, or raw attention/video rows are saved. This measures
selector repeatability and capture wiring only, not faithfulness. Rights
unverified; keep outputs local.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import re
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
FRAME_COUNT = 32
FULL_RECORDS = 3
TOLERANCE = Fraction(1, 20)
RANDOM_SEED = 20261018
MIN_FREE_BEFORE_LOAD = 12 * 1024**3
MIN_FREE_BEFORE_FORWARD = 768 * 1024**2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_hash(values) -> str:
    return hashlib.sha256("".join(sorted(values)).encode("ascii")).hexdigest()


def _directory_content_sha256(root: Path) -> str:
    """Hash exact bytes of every regular file in the pinned model snapshot."""
    digest = hashlib.sha256(b"qta-repeatability-model-snapshot-v1\0")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("model_snapshot_files_unavailable")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        file_digest = _sha256(path).encode("ascii")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(file_digest)
    return digest.hexdigest()


def batch_identity_sha256(batch_digest: str, video_metadata: dict) -> str:
    if (not isinstance(batch_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", batch_digest)
            or not isinstance(video_metadata, dict)):
        raise ValueError("batch_identity_input_invalid")
    metadata_bytes = json.dumps(video_metadata, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=True, allow_nan=False).encode("ascii")
    digest = hashlib.sha256(b"qta-repeatability-batch-identity-v1\0")
    digest.update(batch_digest.encode("ascii"))
    digest.update(metadata_bytes)
    return digest.hexdigest()


def initialize_rng_once(torch_module, seed: int):
    """Initialize run RNGs exactly once; callers must not invoke per repeat."""
    if type(seed) is not int or seed < 0:
        raise ValueError("random_seed_invalid")
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)


def require_same_snapshot(before: str, after: str):
    if (not isinstance(before, str) or not re.fullmatch(r"[0-9a-f]{64}", before)
            or before != after):
        raise RuntimeError("model_snapshot_changed_during_run")


def capture_same_batch_twice(*, model, batch, batch_identity, capture_once,
                              on_attempt, on_complete):
    """Capture twice from the identical batch object and validate each receipt."""
    if not all(callable(function) for function in
               (batch_identity, capture_once, on_attempt, on_complete)):
        raise ValueError("repeat_capture_callbacks_invalid")
    initial_identity = batch_identity(batch)
    selections = []
    maximum_deviation = 0.0
    for _ in range(2):
        on_attempt()
        selection, receipt = capture_once(model, batch, on_forward_complete=on_complete)
        if (not isinstance(receipt, dict)
                or receipt.get("status") != "one_forward_attention_rows_captured_non_authorizing"
                or receipt.get("forward_calls") != 1
                or receipt.get("attention_rows_satisfy_source_dtype_normalization_tolerance") is not True):
            raise RuntimeError("attention_capture_receipt_invalid")
        deviation = receipt.get("attention_row_sum_max_abs_deviation")
        if isinstance(deviation, bool) or not isinstance(deviation, (int, float)) \
                or not math.isfinite(float(deviation)) or deviation < 0:
            raise RuntimeError("attention_capture_deviation_invalid")
        selections.append(selection)
        maximum_deviation = max(maximum_deviation, float(deviation))
        if batch_identity(batch) != initial_identity:
            raise RuntimeError("batch_identity_changed_after_repeat")
    return selections, initial_identity, maximum_deviation, 2


def repeatability_pair(intervals, centers, selection_a, selection_b):
    """Compute pair metrics without retaining selected group indices."""
    from smoke_qta_resolution_stability import compare_selections
    return compare_selections(intervals, centers, selection_a,
                             intervals, centers, selection_b)


def summarize_pairs(pairs):
    if not pairs or any(len(pair) != 2 or any(not math.isfinite(value) for value in pair)
                        for pair in pairs):
        raise ValueError("repeatability_pair_metrics_invalid")
    return {
        "interval_jaccard_mean": sum(pair[0] for pair in pairs) / len(pairs),
        "interval_jaccard_min": min(pair[0] for pair in pairs),
        "selected_center_wasserstein1_mean": sum(pair[1] for pair in pairs) / len(pairs),
        "selected_center_wasserstein1_max": max(pair[1] for pair in pairs),
    }


def ordinal_rows(pairs, batch_hashes):
    if not pairs or len(pairs) != len(batch_hashes):
        raise ValueError("repeatability_pair_batch_count_mismatch")
    return [{"fixed_pair_ordinal": i + 1, "batch_sha256": batch_hashes[i],
             "interval_jaccard": pair[0],
             "selected_center_wasserstein1": pair[1]}
            for i, pair in enumerate(pairs)]


def _write_private(path: Path, payload: dict):
    from smoke_qta_resolution_stability import _write_private as private_writer
    return private_writer(path, payload)


_STATE = {"requested_videos": FULL_RECORDS, "requested_forwards": FULL_RECORDS * 2,
          "candidate_videos_inspected": 0, "effective_video_pairs": 0,
          "attempted_forwards": 0, "completed_forwards": 0,
          "capture_max_deviation": 0.0, "snapshot_hash_before_load": None,
          "snapshot_hash_after_forwards": None, "snapshot_hash_seconds": 0.0,
          "capture_seconds": 0.0, "selection_seconds": 0.0,
          "capture_call_receipts": [],
          "expected_call_bindings": [],
          "smoke_only": False,
          "started_at_utc": None, "started_monotonic": None}


def _execute(args):
    import torch
    import transformers
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_model_module
    import transformers.models.qwen2_5_vl.processing_qwen2_5_vl as qwen_processor_module
    import transformers.models.qwen2_vl.image_processing_qwen2_vl as qwen_image_processor_module
    import transformers.models.qwen2_vl.video_processing_qwen2_vl as qwen_video_processor_module
    import qwen_vl_utils.vision_process as qwen_video_utils_module
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
    from nextgqa_atomic_batch_freeze import batch_sha256
    from qta_attention_forward_capture import (
        aggregate_captured_qta_rows,
        capture_qta_attention_forward_once,
    )
    from qta_attention_call_receipts import (
        build_qta_attention_call_receipt,
        build_qta_attention_call_receipt_index,
        validate_qta_attention_call_receipt_index,
    )
    from qta_sampler_runtime_descriptor import (
        validate_qta_sampler_model_runtime_descriptor,
    )
    from qwen_question_temporal_attention_selector import (
        map_question_token_positions,
        select_top_temporal_groups,
    )
    from run_nextgqa_real_video_gpu_smoke import choose_videos
    from smoke_qwen_real_video_attention_boost import (
        _make_batch, _temporal_visual_groups, _processor_fingerprint,
        question_for_video,
    )

    repo = Path(__file__).resolve().parents[1]
    dataset = repo / "NExT-GQA/datasets/nextgqa"
    count = _STATE["requested_videos"]
    candidates = choose_videos(repo, dataset)
    video_root = Path.home() / "nextgqa_sample/NExTVideo"
    examples = []
    for video in candidates[:count]:
        _STATE["candidate_videos_inspected"] += 1
        try:
            frames, metadata = decode_sampled_video(
                video, num_frames=FRAME_COUNT, uniform_tolerance_seconds=TOLERANCE)
            verify_sampled_frame_pixels(frames, metadata)
            question = question_for_video(dataset, video, video_root)
        except ValueError as exc:
            raise RuntimeError("frozen_candidate_32frame_sampler_or_question_failure") from exc
        examples.append((video, frames, metadata, question))
    if len(examples) != count:
        raise RuntimeError("fixed_candidate_count_unavailable_no_replacement")

    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        raise RuntimeError("pinned_checkpoint_not_cached")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("cuda_bf16_unavailable")
    device = torch.cuda.current_device()
    free_before, total_vram = torch.cuda.mem_get_info(device)
    if free_before < MIN_FREE_BEFORE_LOAD:
        raise RuntimeError("insufficient_free_vram_before_model_load")
    snapshot_timer = time.monotonic()
    snapshot_hash_before = _directory_content_sha256(model_path)
    _STATE["snapshot_hash_before_load"] = snapshot_hash_before
    _STATE["snapshot_hash_seconds"] += time.monotonic() - snapshot_timer
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    model.eval()
    free_after_load, _ = torch.cuda.mem_get_info(device)
    if free_after_load < MIN_FREE_BEFORE_FORWARD:
        raise RuntimeError("insufficient_free_vram_after_model_load")
    if (int(model.config.video_token_id) != int(processor.video_token_id)
            or len(model.model.language_model.layers) < 16):
        raise RuntimeError("pinned_model_processor_contract_invalid")

    temporal_patch_size = int(processor.image_processor.temporal_patch_size)
    video_processor = getattr(processor, "video_processor", processor.image_processor)
    if int(video_processor.temporal_patch_size) != temporal_patch_size:
        raise RuntimeError("processor_temporal_patch_size_disagreement")
    model_patch_size = getattr(model.config.vision_config, "temporal_patch_size", None)
    if model_patch_size is None or int(model_patch_size) != temporal_patch_size:
        raise RuntimeError("model_processor_temporal_patch_size_disagreement")

    from smoke_qta_resolution_stability import processor_group_support

    source_paths = {
        "runner": Path(__file__),
        "decoder": repo / "scripts/claim_mask_video_decoder.py",
        "candidate_selector": repo / "scripts/run_nextgqa_real_video_gpu_smoke.py",
        "question_selector": repo / "scripts/qwen_question_temporal_attention_selector.py",
        "attention_capture": repo / "scripts/qta_attention_forward_capture.py",
        "attention_call_receipts": repo / "scripts/qta_attention_call_receipts.py",
        "sampler_runtime_descriptor": repo / "scripts/qta_sampler_runtime_descriptor.py",
        "batch_hasher": repo / "scripts/nextgqa_atomic_batch_freeze.py",
        "qta_batch_adapter": repo / "scripts/smoke_qwen_real_video_attention_boost.py",
        "temporal_support_mapper": repo / "scripts/smoke_qta_resolution_stability.py",
        "qwen_model_implementation": Path(qwen_model_module.__file__),
        "qwen_processor_implementation": Path(qwen_processor_module.__file__),
        "qwen_image_processor": Path(qwen_image_processor_module.__file__),
        "qwen_video_processor": Path(qwen_video_processor_module.__file__),
        "qwen_video_utils": Path(qwen_video_utils_module.__file__),
    }
    manifests = sorted((repo / "data").glob("nextgqa_samples_*.json"))
    manifest_hashes = [(path.name, _sha256(path)) for path in manifests]
    annotation_hashes = {name: _sha256(dataset / name)
                         for name in ("test.csv", "gsub_test.json", "map_vid_vidorID.json")}

    initialize_rng_once(torch, RANDOM_SEED)
    pair_metrics = []
    batch_hashes = []
    source_video_hashes = []
    full_pts_hashes = []
    sampled_pts_hashes = []
    sampled_rgb_hashes = []
    runtime_descriptor = {
        "schema_version": 1,
        "purpose": "qta_sampler_model_runtime_descriptor_non_authorizing_v1",
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda) if torch.version.cuda else None,
        "transformers_version": str(transformers.__version__),
        "bitsandbytes_version": package_version("bitsandbytes"),
        "device_type": "cuda",
        "precision": "bfloat16",
        "quantization": "4bit_nf4",
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    runtime_descriptor_raw = json.dumps(
        runtime_descriptor, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")
    runtime_descriptor_report = validate_qta_sampler_model_runtime_descriptor(
        runtime_descriptor_raw)
    runtime_sha256 = runtime_descriptor_report["runtime_descriptor_sha256"]
    started_processing = time.monotonic()
    for _ordinal, (video, frames, metadata, question) in enumerate(examples, 1):
        batch, _video_meta, rendered = _make_batch(
            frames=frames,
            sample_fps=Fraction(*metadata["effective_sample_fps"]),
            question=question, processor=processor, torch_module=torch,
            frame_count=FRAME_COUNT)
        ids = batch.get("input_ids")
        grid = batch.get("video_grid_thw")
        if (not torch.is_tensor(ids) or ids.ndim != 2 or int(ids.shape[0]) != 1
                or not torch.is_tensor(grid) or tuple(grid.shape) != (1, 3)):
            raise RuntimeError("processor_batch_shape_invalid")
        temporal_steps, grid_h, grid_w = map(int, grid[0].tolist())
        if temporal_steps * temporal_patch_size != FRAME_COUNT:
            raise RuntimeError("processor_temporal_patch_count_contract_invalid")
        if int(batch["attention_mask"].sum()) != int(ids.shape[1]):
            raise RuntimeError("unexpected_padding_not_supported")
        ids_list = [int(value) for value in ids[0].tolist()]
        groups = _temporal_visual_groups(
            token_ids=ids_list, video_token_id=int(processor.video_token_id),
            temporal_steps=temporal_steps, grid_h=grid_h, grid_w=grid_w,
            merge_size=int(processor.image_processor.merge_size))
        frame_times = [Fraction(row["relative_presentation_time"][0],
                                row["relative_presentation_time"][1])
                       for row in metadata["sampled_frames"]]
        intervals, centers = processor_group_support(
            frame_times, temporal_steps, temporal_patch_size)
        if len(groups) != len(intervals) or len(groups) != len(centers):
            raise RuntimeError("processor_temporal_mapping_shape_mismatch")
        qpos = map_question_token_positions(
            tokenizer=processor.tokenizer, rendered_text=rendered,
            question_text=question, processed_input_ids=ids,
            video_token_id=int(processor.video_token_id))
        capture_run = {"index": 0}
        def identity_for(current_batch):
            return batch_identity_sha256(batch_sha256(current_batch), _video_meta)

        def capture_and_select(current_model, current_batch, *, on_forward_complete):
            free, _ = torch.cuda.mem_get_info(device)
            if free < MIN_FREE_BEFORE_FORWARD:
                raise RuntimeError("insufficient_free_vram_before_forward")
            capture_started = time.monotonic()
            _, rows, receipt = capture_qta_attention_forward_once(
                model=current_model,
                forward=lambda: current_model(**current_batch, use_cache=False, logits_to_keep=1),
                question_positions=qpos, input_length=int(ids.shape[1]),
                torch_module=torch)
            # Count the model forward as complete immediately after its capture
            # helper returns, even if downstream aggregation/selection fails.
            on_forward_complete()
            _STATE["capture_seconds"] += time.monotonic() - capture_started
            capture_run["index"] += 1
            stacked = torch.stack([rows[layer] for layer in range(5, 17)], dim=0)
            payload_bytes = stacked.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype(
                ">f4", copy=True).tobytes(order="C")
            record_token = hashlib.sha256(
                f"qta-repeatability-row-{_ordinal}-{capture_run['index']}".encode("ascii")
            ).hexdigest()
            attention_bytes_sha256 = hashlib.sha256(payload_bytes).hexdigest()
            expected_call_binding = {
                "record_token": record_token,
                "runtime_sha256": runtime_sha256,
                "input_length": int(ids.shape[1]),
                "question_positions": list(qpos),
                "attention_bytes_sha256": attention_bytes_sha256,
            }
            _STATE["expected_call_bindings"].append(expected_call_binding)
            call_receipt = build_qta_attention_call_receipt(
                report=receipt, record_token=expected_call_binding["record_token"],
                runtime_sha256=expected_call_binding["runtime_sha256"],
                input_length=expected_call_binding["input_length"],
                question_positions=expected_call_binding["question_positions"],
                attention_bytes_sha256=expected_call_binding["attention_bytes_sha256"],
                expected_module_classes=receipt["layer_module_classes"])
            _STATE["capture_call_receipts"].append(call_receipt)
            selection_started = time.monotonic()
            scores = aggregate_captured_qta_rows(
                rows=rows, group_positions=groups, torch_module=torch)
            selected_groups = select_top_temporal_groups(
                scores, top_k=max(1, math.ceil(temporal_steps * 0.10)))
            _STATE["selection_seconds"] += time.monotonic() - selection_started
            del rows
            return selected_groups, receipt

        selections, initial_batch_hash, max_deviation, verified_captures = capture_same_batch_twice(
            model=model, batch=batch, batch_identity=identity_for,
            capture_once=capture_and_select,
            on_attempt=lambda: _STATE.__setitem__("attempted_forwards", _STATE["attempted_forwards"] + 1),
            on_complete=lambda: _STATE.__setitem__("completed_forwards", _STATE["completed_forwards"] + 1))
        if verified_captures != 2:
            raise RuntimeError("repeat_capture_count_invalid")
        _STATE["capture_max_deviation"] = max(_STATE["capture_max_deviation"], max_deviation)
        pair_metrics.append(repeatability_pair(intervals, centers, selections[0], selections[1]))
        batch_hashes.append(initial_batch_hash)
        # Keep only the aggregate capture validity statistic, never row payloads.
        source_video_hashes.append(metadata["source_sha256"])
        full_pts_hashes.append(metadata["decoded_pts_sequence_sha256"])
        sampled_pts_hashes.append(hashlib.sha256(json.dumps(
            [row["relative_presentation_time"] for row in metadata["sampled_frames"]],
            separators=(",", ":")).encode("ascii")).hexdigest())
        sampled_rgb_hashes.append(hashlib.sha256(json.dumps(
            [row["rgb_pixel_sha256"] for row in metadata["sampled_frames"]],
            separators=(",", ":")).encode("ascii")).hexdigest())
        _STATE["effective_video_pairs"] += 1
        del batch
        torch.cuda.empty_cache()

    snapshot_timer = time.monotonic()
    model_snapshot_hash = _directory_content_sha256(model_path)
    _STATE["snapshot_hash_seconds"] += time.monotonic() - snapshot_timer
    _STATE["snapshot_hash_after_forwards"] = model_snapshot_hash
    require_same_snapshot(_STATE["snapshot_hash_before_load"], model_snapshot_hash)
    call_receipt_index_raw = build_qta_attention_call_receipt_index(
        receipts=_STATE["capture_call_receipts"])
    call_receipt_validation = validate_qta_attention_call_receipt_index(
        index_raw=call_receipt_index_raw,
        expected_rows=_STATE["expected_call_bindings"])
    payload = {
        "status": "passed",
        "scope": ("exploratory_qta_repeatability_smoke" if args.smoke
                  else "exploratory_qta_repeatability"),
        "smoke_only": bool(args.smoke),
        "started_at_utc": _STATE["started_at_utc"],
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "dataset_input_sha256": annotation_hashes,
        "candidate_exclusion_manifest_count": len(manifest_hashes),
        "candidate_exclusion_manifests_sha256": hashlib.sha256(json.dumps(
            manifest_hashes, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "model_revision": MODEL_REVISION,
        "model_config_sha256": _sha256(model_path / "config.json"),
        "model_snapshot_content_sha256": model_snapshot_hash,
        "model_snapshot_content_sha256_before_load": _STATE["snapshot_hash_before_load"],
        "model_snapshot_content_sha256_after_forwards": _STATE["snapshot_hash_after_forwards"],
        "model_snapshot_content_unchanged": True,
        "processor_revision": MODEL_REVISION,
        "processor_temporal_patch_size": temporal_patch_size,
        "processor_geometry_sha256": hashlib.sha256(json.dumps({
            "temporal_patch_size": temporal_patch_size,
            "patch_size": int(processor.image_processor.patch_size),
            "merge_size": int(processor.image_processor.merge_size),
            "image_processor": processor.image_processor.to_dict(),
        }, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest(),
        "processor_fingerprint_sha256": _processor_fingerprint(processor),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "precision": "bf16_weights_4bit_bf16_compute",
        "random_seed": RANDOM_SEED,
        "dataset_split": "NExT-GQA test",
        "candidate_selection": f"first {_STATE['requested_videos']} from deterministic choose_videos; same qid-sorted question; cohort frozen before decoding; no replacement",
        "requested_videos": _STATE["requested_videos"],
        "effective_video_pairs": len(pair_metrics),
        "failed_video_pairs": _STATE["requested_videos"] - len(pair_metrics),
        "candidate_videos_inspected": _STATE["candidate_videos_inspected"],
        "requested_forwards": _STATE["requested_forwards"],
        "attempted_forwards": _STATE["attempted_forwards"],
        "completed_forwards": _STATE["completed_forwards"],
        "failed_forwards": _STATE["attempted_forwards"] - _STATE["completed_forwards"],
        "not_attempted_forwards": _STATE["requested_forwards"] - _STATE["attempted_forwards"],
        "frame_count": FRAME_COUNT,
        "uniform_grid_tolerance_seconds": [TOLERANCE.numerator, TOLERANCE.denominator],
        "selector": "untreated question-token attention averaged over heads/layers 5-16; top ceil(0.10*T) complete temporal groups",
        "repeat_condition": "same model.eval instance and same exact in-memory processor batch; two independent untreated forwards",
        "batch_identity_verified_per_pair": True,
        "batch_identity_sha256_aggregate": _aggregate_hash(batch_hashes),
        "attention_capture_receipts_verified": _STATE["completed_forwards"],
        "attention_call_receipts": json.loads(call_receipt_index_raw),
        # Persist the separately constructed capture-time join keys so a later
        # auditor can verify the receipt index without deriving expectations
        # from the receipts themselves. These contain only hashes and token
        # geometry, not video, question, answer, or attention payload bytes.
        "expected_capture_call_bindings": list(_STATE["expected_call_bindings"]),
        "attention_call_receipt_validation": call_receipt_validation,
        "sampler_runtime_descriptor_sha256": runtime_sha256,
        "sampler_runtime_descriptor": runtime_descriptor,
        "maximum_attention_row_sum_deviation": _STATE.get("capture_max_deviation", 0.0),
        "primary_statistic": "mean per-pair interval Jaccard over fixed ordinal pairs",
        "metrics": summarize_pairs(pair_metrics),
        "fixed_pair_metrics": ordinal_rows(pair_metrics, batch_hashes),
        "input_hash_aggregates": {
            "source_video_sha256": _aggregate_hash(source_video_hashes),
            "full_decoded_pts_sha256": _aggregate_hash(full_pts_hashes),
            "sampled_pts_sha256": _aggregate_hash(sampled_pts_hashes),
            "sampled_rgb_sha256": _aggregate_hash(sampled_rgb_hashes),
        },
        "free_vram_before_model_load_gib": free_before / 1024**3,
        "free_vram_after_model_load_gib": free_after_load / 1024**3,
        "total_vram_gib": total_vram / 1024**3,
        "attention_capture_and_metric_seconds": time.monotonic() - started_processing,
        "capture_seconds": _STATE["capture_seconds"],
        "selection_seconds": _STATE["selection_seconds"],
        "model_snapshot_hash_seconds_total": _STATE["snapshot_hash_seconds"],
        "full_elapsed_seconds": time.monotonic() - _STATE["started_monotonic"],
        "faithfulness_result": False,
        "ids_questions_answers_group_selections_and_raw_logits_saved": False,
        "local_only": True,
        "rights_status": "unverified_local_only",
    }
    from smoke_qta_resolution_stability import _write_private
    _write_private(args.output, payload)
    print(json.dumps({key: payload[key] for key in (
        "status", "scope", "requested_videos", "effective_video_pairs",
        "requested_forwards", "attempted_forwards", "completed_forwards", "metrics",
        "faithfulness_result")}, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new private local JSON summary; overwrite refused")
    parser.add_argument("--smoke", action="store_true",
                        help="first frozen candidate only; 2 forwards, descriptive wiring smoke")
    args = parser.parse_args(argv)
    record_count = 1 if args.smoke else FULL_RECORDS
    _STATE.update(requested_videos=record_count, requested_forwards=record_count * 2,
                  candidate_videos_inspected=0, effective_video_pairs=0,
                  attempted_forwards=0, completed_forwards=0,
                  capture_max_deviation=0.0, smoke_only=bool(args.smoke),
                  snapshot_hash_before_load=None, snapshot_hash_after_forwards=None,
                  snapshot_hash_seconds=0.0, capture_seconds=0.0,
                  selection_seconds=0.0,
                  capture_call_receipts=[],
                  expected_call_bindings=[],
                  started_at_utc=datetime.now(timezone.utc).isoformat(),
                  started_monotonic=time.monotonic())
    try:
        _execute(args)
        return 0
    except Exception as exc:
        detail = str(exc)
        failure_code = (detail if re.fullmatch(r"[A-Za-z0-9_]+", detail)
                        else f"local_run_failed_{type(exc).__name__}")
        requested = _STATE["requested_forwards"]
        failed_pairs = _STATE["requested_videos"] - _STATE["effective_video_pairs"]
        failure = {
            "status": "failed",
            "scope": "exploratory_qta_repeatability_smoke" if args.smoke else "exploratory_qta_repeatability",
            "smoke_only": bool(args.smoke),
            "started_at_utc": _STATE["started_at_utc"],
            "full_elapsed_seconds": time.monotonic() - _STATE["started_monotonic"],
            "runner_sha256": _sha256(Path(__file__)),
            "failure_code": failure_code,
            "requested_videos": _STATE["requested_videos"],
            "effective_video_pairs": _STATE["effective_video_pairs"],
            "failed_video_pairs": failed_pairs,
            "candidate_videos_inspected": _STATE["candidate_videos_inspected"],
            "requested_forwards": requested,
            "attempted_forwards": _STATE["attempted_forwards"],
            "completed_forwards": _STATE["completed_forwards"],
            "failed_forwards": _STATE["attempted_forwards"] - _STATE["completed_forwards"],
            "not_attempted_forwards": requested - _STATE["attempted_forwards"],
            "model_snapshot_content_sha256_before_load": _STATE["snapshot_hash_before_load"],
            "model_snapshot_content_sha256_after_forwards": _STATE["snapshot_hash_after_forwards"],
            "model_snapshot_hash_seconds_total": _STATE["snapshot_hash_seconds"],
            "capture_seconds": _STATE["capture_seconds"],
            "selection_seconds": _STATE["selection_seconds"],
            "faithfulness_result": False,
            "local_only": True,
            "rights_status": "unverified_local_only",
        }
        try:
            from smoke_qta_resolution_stability import _write_private
            _write_private(args.output, failure)
            artifact = "written"
        except Exception:
            artifact = "unavailable_output_path_unsafe_or_exists"
        print(json.dumps({"status": "failed", "failure_code": failure_code,
            "failure_artifact": artifact,
            "requested_videos": _STATE["requested_videos"],
            "effective_video_pairs": _STATE["effective_video_pairs"],
            "failed_video_pairs": failed_pairs,
            "requested_forwards": requested,
            "attempted_forwards": _STATE["attempted_forwards"],
            "completed_forwards": _STATE["completed_forwards"]}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
