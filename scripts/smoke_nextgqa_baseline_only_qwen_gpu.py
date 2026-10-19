#!/usr/bin/env python3
"""Synthetic-only Qwen compatibility smoke for the baseline scorer.

Builds three deterministic 32-frame RGB clips with the locally cached pinned
processor, then either writes a CPU-only preflight receipt or loads the local
pinned 7B model and invokes exactly three baseline-only forwards. The report is
aggregate-only and always non-authorizing; it is not a faithfulness experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_DIRECTORY = ("models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
                   + MODEL_REVISION)
MIN_FREE_BEFORE_LOAD = 12 * 1024**3
MIN_FREE_BEFORE_FORWARD = 3 * 1024**3
SENSITIVE_KEYS = frozenset({
    "question", "question_text", "qa", "choices", "gold", "answer",
    "frames", "frame", "pixels", "pixel_values", "pixel_values_videos",
    "input_ids", "records", "prompt", "prompt_text", "messages",
})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new aggregate-only JSON path under repository results/")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and validate CPU processor inputs, without model/GPU")
    return parser


def validate_output_path(path: Path, repo: Path) -> Path:
    results_dir = (repo / "results").resolve()
    output = path if path.is_absolute() else repo / path
    resolved_parent = output.parent.resolve()
    if (not resolved_parent.is_relative_to(results_dir)
            or output.suffix.lower() != ".json"
            or output.name in ("", ".", "..")):
        raise ValueError("output_must_be_json_under_results")
    output = resolved_parent / output.name
    if output.exists() or output.is_symlink():
        raise FileExistsError("output_already_exists")
    return output


def _assert_aggregate_only(value: Any, path: str = "report") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("aggregate_report_contains_sensitive_field")
            normalized = key.lower()
            if (normalized in SENSITIVE_KEYS or normalized.startswith("raw_")
                    or normalized.endswith("_text")
                    or any(term in normalized for term in (
                        "question", "prompt", "message", "pixel_values",
                        "token_ids", "record_id", "choice_text", "gold_answer"))):
                raise ValueError("aggregate_report_contains_sensitive_field")
            _assert_aggregate_only(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_aggregate_only(child, f"{path}.{index}")
    elif isinstance(value, str):
        if len(value) > 2048:
            raise ValueError("aggregate_report_string_too_long")


def write_manifest_exclusive(path: Path, report: dict[str, Any]) -> None:
    _assert_aggregate_only(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload + b"\n")
        os.chmod(path, 0o600)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_content_sha256(snapshot: Path) -> tuple[str, int, int]:
    """Hash local snapshot file content with bounded memory, without network."""
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    if not files:
        raise ValueError("pinned_checkpoint_empty")
    entries = []
    total_bytes = 0
    for path in files:
        relative = path.relative_to(snapshot).as_posix()
        size = path.stat().st_size
        entries.append({"path": relative, "size": size,
                        "sha256": _sha256_file(path)})
        total_bytes += size
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(files), total_bytes


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_video_sha256(record: dict) -> str:
    digest = hashlib.sha256()
    digest.update(b"synthetic-rgb-clip-v1\0")
    for index, frame in enumerate(record["frames"]):
        digest.update(index.to_bytes(4, "big"))
        digest.update(frame.convert("RGB").tobytes())
    return digest.hexdigest()


def derive_answer_token_ids(tokenizer: Any, rendered_prompt: str) -> list[int]:
    """Resolve A–E as one-token continuations of this exact QA prompt."""
    from qta_choice_token_binding import resolve_choice_continuation_ids

    token_ids, _prefix_ids = resolve_choice_continuation_ids(tokenizer, rendered_prompt)
    return token_ids


def build_synthetic_rows(*, processor: Any, records: list[dict],
                         vision_info_fn: Any, torch_module: Any,
                         processor_fingerprint: str,
                         model_revision: str = MODEL_REVISION) -> tuple[list, list, dict]:
    """Build exact same-source QA batches and row pins, without a model call."""
    from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256
    from qwen_question_temporal_attention_selector import map_question_token_positions
    from qwen_processor_batch_review_mapping import map_review_interval_from_processor_batch
    from run_nextgqa_gold_interval_margin_pilot import (
        build_qa_prompt, process_qa_video,
    )

    if (len(records) != 3 or any(len(record.get("frames", ())) != 32
                                 for record in records)):
        raise ValueError("synthetic_three_by_32_frame_inventory_required")
    rows, pins = [], []
    grid_counts = []
    source_hashes = set()
    for position, record in enumerate(records, start=1):
        item = {**record, **record["qa_item"]}
        fps = record["sample_fps"]
        sample_fps = (fps.numerator, fps.denominator)
        rendered_text, messages = build_qa_prompt(
            processor, item, record["frames"], sample_fps)
        token_ids = derive_answer_token_ids(processor.tokenizer, rendered_text)
        if len(token_ids) != 5 or len(set(token_ids)) != 5:
            raise ValueError("answer_label_token_ids_not_distinct")
        images, videos = vision_info_fn(messages, return_video_metadata=True)
        if images is not None or not isinstance(videos, list) or len(videos) != 1:
            raise ValueError("synthetic_video_processor_contract_failed")
        pixels, metadata = videos[0]
        batch = FrozenBatch(process_qa_video(
            processor, rendered_text, images, pixels, metadata))
        mapping = map_review_interval_from_processor_batch(
            batch=batch, torch_module=torch_module,
            video_token_id=int(processor.video_token_id),
            spatial_merge_size=int(processor.image_processor.merge_size),
            temporal_patch_size=int(processor.image_processor.temporal_patch_size),
            sampled_frame_indices=record["sampled_indices"],
            sampled_pts_seconds=record["presentation_times"],
            source_frame_count=32, video_duration_seconds=32,
            sample_fps=fps, processor_revision=model_revision,
            processor_fingerprint_sha256=processor_fingerprint,
            reviewed_intervals_seconds=record["intervals"],
            interval_boundary_guard_seconds=0.0,
        )
        if mapping["typed_batch_sha256"] != batch_sha256(batch):
            raise ValueError("synthetic_processor_mapping_batch_mismatch")
        ids = batch["input_ids"]
        question_positions = map_question_token_positions(
            tokenizer=processor.tokenizer, rendered_text=rendered_text,
            question_text=item["question"], processed_input_ids=ids,
            video_token_id=int(processor.video_token_id))
        temporal_steps, grid_h, grid_w = map(int, batch["video_grid_thw"][0].tolist())
        merge = int(processor.image_processor.merge_size)
        tokens_per_group = (grid_h * grid_w) // (merge * merge)
        visual_positions = [index for index, token in enumerate(ids[0].tolist())
                            if token == int(processor.video_token_id)]
        if len(visual_positions) != temporal_steps * tokens_per_group:
            raise ValueError("synthetic_visual_group_geometry_invalid")
        temporal_groups = [
            visual_positions[index * tokens_per_group:(index + 1) * tokens_per_group]
            for index in range(temporal_steps)
        ]
        source_hash = _source_video_sha256(record)
        if source_hash in source_hashes:
            raise ValueError("synthetic_source_hash_collision")
        source_hashes.add(source_hash)
        input_ids_list = ids[0].tolist()
        input_ids_sha = hashlib.sha256(json.dumps(
            input_ids_list, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        batch_sha = batch_sha256(batch)
        opaque_token = hashlib.sha256(
            f"synthetic-qta-review-token-v1:{position}".encode()).hexdigest()[:24]
        row_id = f"synthetic-{position:02d}"
        binding = {
            "record_id": row_id,
            "source_video_sha256": source_hash,
            "prompt_sha256": hashlib.sha256(rendered_text.encode("utf-8")).hexdigest(),
            "input_ids_sha256": input_ids_sha,
            "batch_sha256": batch_sha,
            "question_positions": question_positions,
            "temporal_groups": temporal_groups,
            "opaque_record_token": opaque_token,
            "answer_token_ids": token_ids,
        }
        rows.append({
            **binding, "condition": "full", "answer_token_ids": token_ids,
            "batch": batch,
        })
        pins.append(binding)
        grid_counts.append(temporal_steps)
    if len(set(row["source_video_sha256"] for row in pins)) != 3:
        raise ValueError("synthetic_source_inventory_not_distinct")
    summary = {
        "synthetic_clip_count": 3,
        "synthetic_frames_per_clip": 32,
        "processor_grid_group_counts": grid_counts,
        "distinct_source_count": len(source_hashes),
        "answer_token_count": 5,
    }
    return rows, pins, summary


def _base_report(started_utc: str, dry_run: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "synthetic_three_clip_baseline_scorer_compatibility_smoke",
        "status": "incomplete",
        "scope": "synthetic_only_no_faithfulness_evidence",
        "started_utc": started_utc,
        "model_revision": MODEL_REVISION,
        "dry_run": dry_run,
        "requested_records": 3,
        "requested_frames_per_clip": 32,
        "requested_forwards": 0 if dry_run else 3,
        "attempted_forwards": 0,
        "completed_forwards": 0,
        "faithfulness_result": False,
        "model_load_permitted": False,
        "scoring_authorized": False,
        "execution_authorization": "not_granted",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo = Path(__file__).resolve().parent.parent
    output = validate_output_path(args.output, repo)
    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    report = _base_report(started, args.dry_run)
    stage = "imports"
    model = None
    try:
        import numpy as np
        import torch
        import transformers
        import PIL
        from PIL import Image
        from transformers import AutoProcessor, BitsAndBytesConfig
        from transformers import Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from smoke_qwen_internal_attention_ablation import (
            MODEL_REVISION as FIXTURE_REVISION,
            _processor_fingerprint, synthetic_records,
        )
        from nextgqa_baseline_only_scoring import score_full_video_baseline_three_rows

        if FIXTURE_REVISION != MODEL_REVISION:
            raise ValueError("synthetic_fixture_revision_mismatch")
        snapshot = Path.home() / ".cache/huggingface/hub" / MODEL_DIRECTORY
        if not snapshot.is_dir():
            raise ValueError("pinned_checkpoint_not_cached_downloads_disabled")
        stage = "load_local_processor"
        processor = AutoProcessor.from_pretrained(
            str(snapshot), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        processor_fingerprint = _processor_fingerprint(processor)
        source_paths = {
            "runner": Path(__file__).resolve(),
            "synthetic_fixture": Path(__file__).resolve().parent / "smoke_qwen_internal_attention_ablation.py",
            "scorer": Path(__file__).resolve().parent / "nextgqa_baseline_only_scoring.py",
            "qa_video_batch_builder": Path(__file__).resolve().parent / "run_nextgqa_gold_interval_margin_pilot.py",
            "answer_token_binding": Path(__file__).resolve().parent / "qta_choice_token_binding.py",
            "selector": Path(__file__).resolve().parent / "qwen_question_temporal_attention_selector.py",
            "atomic_batch_freeze": Path(__file__).resolve().parent / "nextgqa_atomic_batch_freeze.py",
            "processor_mapping": Path(__file__).resolve().parent / "qwen_processor_batch_review_mapping.py",
            "temporal_mapping": Path(__file__).resolve().parent / "qwen_review_interval_token_groups.py",
            "attention_capture": Path(__file__).resolve().parent / "qta_attention_forward_capture.py",
            "attention_capture_contract": Path(__file__).resolve().parent / "qta_attention_capture_contract.py",
            "attention_capture_sidecars": Path(__file__).resolve().parent / "qta_attention_capture_sidecars.py",
            "model_call_boundary": Path(__file__).resolve().parent / "nextgqa_model_call_boundary.py",
            "protocol": repo / "docs/qwen_question_attention_faithfulness_protocol_v1_draft.md",
        }
        report["source_sha256"] = {
            name: _sha256_file(path) for name, path in source_paths.items()
        }
        runtime_descriptor = {
            "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
            "numpy": np.__version__, "pillow": PIL.__version__,
        }
        report["runtime_descriptor"] = runtime_descriptor
        report["runtime_sha256"] = _canonical_sha256(runtime_descriptor)
        records = synthetic_records(np, Image)
        rows, pins, input_summary = build_synthetic_rows(
            processor=processor, records=records,
            vision_info_fn=process_vision_info, torch_module=torch,
            processor_fingerprint=processor_fingerprint)
        report.update(input_summary)
        report["processor_fingerprint_sha256"] = processor_fingerprint
        report["preflight_elapsed_seconds"] = time.monotonic() - timer
        if args.dry_run:
            report["status"] = "preflight_passed_no_model_forward"
            report["requested_forwards"] = 3
            report["processor_mapping_checked"] = True
            report["model_weights_loaded"] = False
            report["checkpoint_hash_status"] = "not_computed_dry_run"
            report["elapsed_seconds"] = time.monotonic() - timer
            write_manifest_exclusive(output, report)
            return 0

        stage = "cuda_preflight"
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda_bf16_unavailable")
        device = torch.cuda.current_device()
        free_before, total_vram = torch.cuda.mem_get_info(device)
        report.update({
            "device_name": torch.cuda.get_device_name(device),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pillow_version": PIL.__version__,
            "free_vram_before_load_gib": free_before / 1024**3,
            "total_vram_gib": total_vram / 1024**3,
        })
        if free_before < MIN_FREE_BEFORE_LOAD:
            raise RuntimeError("insufficient_free_vram_before_model_load")

        stage = "hash_local_checkpoint"
        checkpoint_sha, checkpoint_file_count, checkpoint_bytes = checkpoint_content_sha256(snapshot)
        tokenizer_sha = hashlib.sha256(
            processor.tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest()
        runtime_descriptor["cuda"] = torch.version.cuda
        report["runtime_descriptor"] = runtime_descriptor
        report["runtime_sha256"] = _canonical_sha256(runtime_descriptor)
        scripts = Path(__file__).resolve().parent
        protocol_path = repo / "docs/qwen_question_attention_faithfulness_protocol_v1_draft.md"
        scorer_path = scripts / "nextgqa_baseline_only_scoring.py"
        run_digests = {
            "model_sha256": checkpoint_sha,
            "processor_sha256": processor_fingerprint,
            "tokenizer_sha256": tokenizer_sha,
            "runtime_sha256": _canonical_sha256(runtime_descriptor),
            "protocol_sha256": _sha256_file(protocol_path),
            "scorer_sha256": _sha256_file(scorer_path),
        }
        report.update({
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_hash_status": "all_snapshot_files_content_hashed",
            "checkpoint_file_count": checkpoint_file_count,
            "checkpoint_bytes_hashed": checkpoint_bytes,
            "runtime_descriptor": runtime_descriptor,
            "runtime_sha256": run_digests["runtime_sha256"],
            "model_sha256": checkpoint_sha,
            "tokenizer_sha256": tokenizer_sha,
            "protocol_sha256": run_digests["protocol_sha256"],
            "scorer_sha256": run_digests["scorer_sha256"],
        })

        stage = "load_local_model"
        load_started = time.monotonic()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(snapshot), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        ).eval()
        report["model_load_seconds"] = time.monotonic() - load_started
        if (int(model.config.video_token_id) != int(processor.video_token_id)
                or getattr(model.config, "_attn_implementation", None) != "eager"):
            raise RuntimeError("model_processor_or_attention_config_mismatch")
        if len(model.model.language_model.layers) < 16:
            raise RuntimeError("required_attention_layers_unavailable")
        free_before_forward, _ = torch.cuda.mem_get_info(device)
        report["free_vram_before_forward_gib"] = free_before_forward / 1024**3
        if free_before_forward < MIN_FREE_BEFORE_FORWARD:
            raise RuntimeError("insufficient_free_vram_before_forward")

        stage = "score_three_full_video_rows"
        torch.cuda.reset_peak_memory_stats(device)
        score_started = time.monotonic()
        scoring = score_full_video_baseline_three_rows(
            model=model, rows=rows, expected_record_bindings=pins,
            run_digests=run_digests, torch_module=torch, device="cuda")
        report["status"] = "passed" if scoring["status"] == "completed_non_authorizing" else "incomplete"
        report["requested_forwards"] = scoring["requested_forwards"]
        report["attempted_forwards"] = scoring["attempted_forwards"]
        report["completed_forwards"] = scoring["completed_forwards"]
        report["root_call_entries"] = scoring["root_call_entries"]
        report["verified_root_calls"] = scoring["verified_root_calls"]
        report["completed_boundary_calls"] = scoring["completed_boundary_calls"]
        report["failure_code"] = scoring["failure_code"]
        # Persist counters before synchronize: if CUDA reports a deferred error,
        # the receipt must retain forwards that the scorer already completed.
        torch.cuda.synchronize(device)
        report["scoring_elapsed_seconds"] = time.monotonic() - score_started
        report["capture_layers_1_based"] = list(range(5, 17))
        report["attention_capture_rows_requested"] = 3
        report["attention_capture_passed_rows"] = (
            3 if scoring["status"] == "completed_non_authorizing" else 0)
        report["peak_allocated_vram_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
        report["peak_reserved_vram_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
        report["free_vram_after_gib"] = torch.cuda.mem_get_info(device)[0] / 1024**3
        report["model_weights_loaded"] = True
        report["precision"] = "bf16_4bit_bf16_compute_eager"
        report["elapsed_seconds"] = time.monotonic() - timer
        # Do not serialize per-row scores, token IDs, prompts, frames, or records.
        if scoring["status"] == "completed_non_authorizing":
            margins = [
                row["answer_label_log_probabilities_nats"][row["p_label"]]
                - row["answer_label_log_probabilities_nats"][row["q_label"]]
                for row in scoring["records"]
            ]
            report["aggregate_pq_log_probability_margin_nats"] = {
                "n": len(margins), "mean": sum(margins) / len(margins),
                "min": min(margins), "max": max(margins),
            }
        write_manifest_exclusive(output, report)
        return 0 if report["status"] == "passed" else 1
    except Exception as exc:
        report["status"] = "failed"
        report["failure_stage"] = stage
        report["failure_code"] = "error_" + type(exc).__name__
        report["elapsed_seconds"] = time.monotonic() - timer
        report["model_weights_loaded"] = model is not None
        write_manifest_exclusive(output, report)
        return 1
    finally:
        if model is not None:
            del model


if __name__ == "__main__":
    raise SystemExit(main())
