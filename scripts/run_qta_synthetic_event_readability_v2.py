#!/usr/bin/env python3
"""Run a two-clip, synthetic Qwen construct-readability smoke.

The normal run is exactly two full-video next-token forwards on the fixed
circle/triangle transient-event pair. It uses the pinned local Qwen2.5-VL
checkpoint in offline 4-bit/BF16 mode. ``--dry-run`` loads only the local
processor and validates both CPU video batches. This is construct readability,
not faithfulness or evidence about natural videos.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import stat
import time
from collections.abc import Mapping
from typing import Any, Callable

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from qta_synthetic_event_video_fixture import (  # noqa: E402
    GENERATOR_VERSION,
    SEED,
    EventPair,
    build_synthetic_event_pair,
)

MODEL_REPOSITORY = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_SNAPSHOT_RELATIVE = Path(
    ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
) / "snapshots" / MODEL_REVISION
MIN_FREE_VRAM_BEFORE_LOAD = int(13.5 * 1024**3)
MIN_FREE_VRAM_BEFORE_FORWARD = 3 * 1024**3
MIN_HOST_AVAILABLE_BEFORE_PROCESSOR = 4 * 1024**3
MIN_HOST_AVAILABLE_BEFORE_MODEL = 6 * 1024**3
MIN_HOST_AVAILABLE_BEFORE_FORWARD = 2 * 1024**3
CHOICE_LABELS = "ABCDE"
CHOICE_TO_INDEX = {label: index for index, label in enumerate(CHOICE_LABELS)}
FORBIDDEN_REPORT_KEYS = {
    "question", "question_text", "choices", "choice_text", "answer",
    "answers", "gold", "frames", "frame", "pixels", "pixel_values",
    "pixel_values_videos", "input_ids", "messages", "prompt", "prompt_text",
    "records", "transcript", "nickname",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path,
        help="new aggregate-only JSON directly under data/private_nextgqa/",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="CPU processor preflight only; no model weights or forward calls",
    )
    return parser


def validate_output_path(path: Path, repo_root: Path) -> Path:
    private_dir = repo_root / "data" / "private_nextgqa"
    try:
        private_info = private_dir.lstat()
    except OSError as exc:
        raise ValueError("private_output_directory_unavailable") from exc
    if (not stat.S_ISDIR(private_info.st_mode)
            or stat.S_IMODE(private_info.st_mode) & 0o077
            or private_info.st_uid != os.geteuid()):
        raise ValueError("private_output_directory_permissions_invalid")
    resolved_private = private_dir.resolve(strict=True)
    output = path if path.is_absolute() else repo_root / path
    if output.parent.resolve(strict=False) != resolved_private:
        raise ValueError("output_must_be_direct_child_of_private_directory")
    if output.suffix.lower() != ".json" or output.name in ("", ".", ".."):
        raise ValueError("output_must_be_json")
    if output.exists() or output.is_symlink():
        raise FileExistsError("output_already_exists")
    return resolved_private / output.name


def _assert_aggregate_only(value: Any, path: str = "report") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key.lower() in FORBIDDEN_REPORT_KEYS:
                raise ValueError("aggregate_report_contains_private_field")
            if (key.startswith("raw_") or key.endswith("_text")
                    or any(term in key.lower() for term in (
                        "prompt", "pixel_values", "token_ids", "frame_bytes",
                    ))):
                raise ValueError("aggregate_report_contains_private_field")
            _assert_aggregate_only(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_aggregate_only(child, f"{path}.{index}")
    elif isinstance(value, str):
        if len(value) > 2048 or value.startswith(("/", "\\")):
            raise ValueError("aggregate_report_string_invalid")


def write_report_exclusive(path: Path, report: dict[str, Any]) -> None:
    _assert_aggregate_only(report)
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
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


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _hash_pair_frames(pair: EventPair) -> str:
    digest = hashlib.sha256(b"qta-synthetic-event-pair-v2\0")
    for clip in pair.clips:
        for index, frame in enumerate(clip.frames):
            digest.update(index.to_bytes(4, "big"))
            digest.update(frame.convert("RGB").tobytes())
    return digest.hexdigest()


def host_available_memory_bytes(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    try:
        lines = meminfo_path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise RuntimeError("host_memory_probe_failed") from exc
    for line in lines:
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1]) * 1024
    raise RuntimeError("host_available_memory_unavailable")


def _memory_gib(value: int) -> float:
    return round(value / 1024**3, 3)


def _release_cuda_after_forward(torch_module: Any, device: int) -> None:
    """Synchronize and return unused cached blocks before the next guard."""
    torch_module.cuda.synchronize(device)
    torch_module.cuda.empty_cache()


def _runtime_descriptor(torch_module: Any, transformers_module: Any,
                        pillow_version: str, qwen_utils_version: str,
                        *, cuda: bool) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(torch_module.__version__),
        "transformers": str(transformers_module.__version__),
        "pillow": pillow_version,
        "qwen_vl_utils": qwen_utils_version,
        "cuda": str(torch_module.version.cuda or "unavailable") if cuda else "not_used",
    }


def _processor_fingerprint(processor: Any) -> tuple[str, str]:
    tokenizer_json = processor.tokenizer.backend_tokenizer.to_str()
    tokenizer_sha = hashlib.sha256(tokenizer_json.encode("utf-8")).hexdigest()
    image_processor = processor.image_processor
    descriptor = {
        "processor_class": type(processor).__name__,
        "image_processor": image_processor.to_dict(),
        "tokenizer_sha256": tokenizer_sha,
        "video_token_id": int(processor.video_token_id),
        "model_revision": MODEL_REVISION,
    }
    return _canonical_sha256(descriptor), tokenizer_sha


def prepare_pair_inputs(*, pair: EventPair, processor: Any, torch_module: Any,
                        vision_info_fn: Callable[..., Any],
                        build_prompt_fn: Callable[..., Any],
                        process_video_fn: Callable[..., Any]) -> list[dict[str, Any]]:
    """Render and CPU-process exactly two fixture clips, with A–E token binding."""
    from qta_choice_token_binding import resolve_choice_continuation_ids

    if len(pair.clips) != 2:
        raise ValueError("exact_two_clip_pair_required")
    prepared = []
    seen_prompt_hashes = set()
    for clip in pair.clips:
        if len(clip.frames) != 32 or any(
                frame.mode != "RGB" for frame in clip.frames):
            raise ValueError("event_fixture_frame_contract_invalid")
        if tuple(label for label, _ in clip.choices) != tuple(CHOICE_LABELS):
            raise ValueError("event_fixture_choice_order_invalid")
        item = {"question": clip.question,
                "choices": [text for _label, text in clip.choices]}
        rendered_text, messages = build_prompt_fn(
            processor, item, clip.frames, (1, 1))
        prompt_sha256 = hashlib.sha256(rendered_text.encode("utf-8")).hexdigest()
        seen_prompt_hashes.add(prompt_sha256)
        choice_token_ids, _prefix_ids = resolve_choice_continuation_ids(
            processor.tokenizer, rendered_text)
        if len(choice_token_ids) != 5 or len(set(choice_token_ids)) != 5:
            raise ValueError("choice_continuation_binding_invalid")
        images, videos = vision_info_fn(messages, return_video_metadata=True)
        if images is not None or not isinstance(videos, list) or len(videos) != 1:
            raise ValueError("video_processor_contract_invalid")
        pixels, metadata = videos[0]
        batch = process_video_fn(processor, rendered_text, images, pixels, metadata)
        if not isinstance(batch, Mapping):
            raise ValueError("processor_batch_schema_invalid")
        input_ids = batch.get("input_ids")
        if (not isinstance(input_ids, torch_module.Tensor) or input_ids.device.type != "cpu"
                or input_ids.ndim != 2 or input_ids.shape[0] != 1
                or input_ids.shape[1] < 1):
            raise ValueError("processor_input_ids_contract_invalid")
        video_grid = batch.get("video_grid_thw")
        if (not isinstance(video_grid, torch_module.Tensor) or video_grid.device.type != "cpu"
                or video_grid.ndim != 2 or video_grid.shape[0] != 1
                or video_grid.shape[1] != 3 or bool((video_grid <= 0).any().item())):
            raise ValueError("processor_video_grid_contract_invalid")
        if any(isinstance(value, torch_module.Tensor) and value.device.type != "cpu"
               for value in batch.values()):
            raise ValueError("processor_batch_must_remain_on_cpu")
        target_index = CHOICE_TO_INDEX[clip.correct_choice_label]
        prepared.append({
            "batch": batch,
            "choice_token_ids": choice_token_ids,
            "target_index": target_index,
            "prompt_sha256": prompt_sha256,
        })
    if len(prepared) != 2 or len(seen_prompt_hashes) != 1:
        raise ValueError("paired_prompt_identity_mismatch")
    return prepared


def processor_shape_summary(prepared: list[dict[str, Any]], *, merge_size: int
                            ) -> list[dict[str, Any]]:
    """Expose only derived tensor/grid counts needed for resource review."""
    if len(prepared) != 2 or merge_size < 1:
        raise ValueError("processor_summary_contract_invalid")
    summaries = []
    for row in prepared:
        batch = row["batch"]
        input_ids = batch["input_ids"]
        grid = batch["video_grid_thw"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("processor_summary_input_shape_invalid")
        if grid.ndim != 2 or grid.shape != (1, 3):
            raise ValueError("processor_summary_grid_shape_invalid")
        temporal, height, width = (int(value) for value in grid[0].tolist())
        if (temporal <= 0 or height <= 0 or width <= 0
                or height % merge_size or width % merge_size):
            raise ValueError("processor_summary_grid_values_invalid")
        summaries.append({
            "sequence_tokens": int(input_ids.shape[1]),
            "video_grid_thw": [temporal, height, width],
            "merged_visual_tokens": temporal * (height // merge_size)
            * (width // merge_size),
        })
    return summaries


def score_two_inputs(*, model: Any, prepared: list[dict[str, Any]],
                     torch_module: Any, device: str,
                     before_forward: Callable[[], dict[str, float]] | None = None,
                     after_forward_cleanup: Callable[[], None] | None = None,
                     progress: Callable[[int, int], None] | None = None,
                     ) -> dict[str, Any]:
    """Make exactly one direct forward per clip; no generation/intervention."""
    if len(prepared) != 2:
        raise ValueError("exact_two_prepared_inputs_required")
    completed_rows = []
    attempted = 0
    completed_forward_calls = 0
    for row in prepared:
        if before_forward is not None:
            before_forward()
        batch = row["batch"]
        model_inputs = {
            key: value.to(device) if isinstance(value, torch_module.Tensor) else value
            for key, value in batch.items()
        }
        model_inputs["use_cache"] = False
        model_inputs["logits_to_keep"] = 1
        attempted += 1
        if progress is not None:
            progress(attempted, completed_forward_calls)
        with torch_module.inference_mode():
            output = model(**model_inputs)
        completed_forward_calls += 1
        if progress is not None:
            progress(attempted, completed_forward_calls)
        logits = getattr(output, "logits", None)
        if (not isinstance(logits, torch_module.Tensor) or logits.ndim != 3
                or logits.shape[0] != 1 or logits.shape[1] < 1):
            raise ValueError("model_logits_schema_invalid")
        last_logits = logits[0, -1].float()
        if not bool(torch_module.isfinite(last_logits).all().item()):
            raise ValueError("model_logits_nonfinite")
        log_probs = torch_module.log_softmax(last_logits, dim=-1)
        token_ids = row["choice_token_ids"]
        if (len(token_ids) != 5 or any(token < 0 or token >= log_probs.numel()
                                      for token in token_ids)):
            raise ValueError("choice_token_id_out_of_vocab")
        option_scores = [float(log_probs[token].item()) for token in token_ids]
        if not all(math.isfinite(value) for value in option_scores):
            raise ValueError("choice_log_probability_nonfinite")
        target_index = row["target_index"]
        top_index = max(range(5), key=lambda index: (option_scores[index], -index))
        foil_score = max(score for index, score in enumerate(option_scores)
                         if index != target_index)
        completed_rows.append({
            "correct": top_index == target_index,
            "target_vs_best_foil_margin": option_scores[target_index] - foil_score,
        })
        # Only scalar metrics cross the iteration boundary. The video tensors,
        # output and derived logits otherwise keep substantial CUDA allocations
        # live while the next clip is prepared.
        del output, logits, last_logits, log_probs, model_inputs, batch
        if after_forward_cleanup is not None:
            after_forward_cleanup()
    if attempted != 2 or completed_forward_calls != 2 or len(completed_rows) != 2:
        raise RuntimeError("exact_two_forward_invariant_failed")
    margins = [row["target_vs_best_foil_margin"] for row in completed_rows]
    return {
        "attempted_forwards": attempted,
        "completed_forwards": completed_forward_calls,
        "top_choice_correct_count": sum(int(row["correct"]) for row in completed_rows),
        "top_choice_accuracy": sum(int(row["correct"]) for row in completed_rows) / 2,
        "target_vs_best_foil_margin_nats": {
            "n": len(margins),
            "mean": sum(margins) / len(margins),
            "min": min(margins),
            "max": max(margins),
        },
    }


def _base_report(*, started_utc: str, dry_run: bool,
                 pair_sha256: str, source_sha256: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "qta_synthetic_event_construct_readability_smoke",
        "status": "incomplete",
        "scope": "synthetic_transient_event_readability_only",
        "started_utc": started_utc,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "generator_version": GENERATOR_VERSION,
        "random_seed": SEED,
        "pair_frames_sha256": pair_sha256,
        "source_sha256": source_sha256,
        "requested_videos": 2,
        "effective_videos": 0,
        "frames_per_video": 32,
        "planned_model_forwards": 2,
        "minimum_free_vram_gib_before_model_load": 13.5,
        "minimum_free_vram_gib_before_each_forward": _memory_gib(
            MIN_FREE_VRAM_BEFORE_FORWARD),
        "attempted_forwards": 0,
        "completed_forwards": 0,
        "dry_run": dry_run,
        "model_weights_loaded": False,
        "faithfulness_result": False,
        "faithfulness_scoring_authorized": False,
        "authorization_scope": "user_authorized_synthetic_construct_readability_only",
    }


def _source_hashes(repo_root: Path) -> dict[str, str]:
    scripts = Path(__file__).resolve().parent
    files = {
        "runner": Path(__file__).resolve(),
        "fixture": scripts / "qta_synthetic_event_video_fixture.py",
        "qa_render_video_batch_builder": repo_root / "scripts/run_nextgqa_gold_interval_margin_pilot.py",
        "choice_token_binding": scripts / "qta_choice_token_binding.py",
        "experiment_protocol": repo_root / "experiments/2026-10-19_qta_synthetic_event_readability_smoke.md",
    }
    return {name: _sha256_file(path) for name, path in files.items()}


def _checkpoint_hash(snapshot: Path) -> tuple[str, int, int]:
    entries = []
    total_size = 0
    for path in sorted(snapshot.rglob("*")):
        if path.is_symlink():
            # HF snapshots commonly link to content-addressed blobs; hash the
            # resolved bytes while retaining only the relative logical name.
            resolved = path.resolve(strict=True)
            source = resolved
        elif path.is_file():
            source = path
        else:
            continue
        if not source.is_file():
            raise ValueError("checkpoint_entry_not_regular_file")
        size = source.stat().st_size
        entries.append({"path": path.relative_to(snapshot).as_posix(),
                        "size_bytes": size, "sha256": _sha256_file(source)})
        total_size += size
    if not entries:
        raise ValueError("pinned_checkpoint_empty")
    return _canonical_sha256(entries), len(entries), total_size


def _failure_report(report: dict[str, Any], *, stage: str,
                    exc: Exception, started_monotonic: float) -> dict[str, Any]:
    report.update({
        "status": "failed",
        "failure_stage": stage,
        "failure_type": type(exc).__name__,
        "attempted_forwards": report.get("attempted_forwards", 0),
        "completed_forwards": report.get("completed_forwards", 0),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
    })
    return report


def run(*, output: Path, dry_run: bool, repo_root: Path | None = None) -> int:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    output_path = validate_output_path(output, repo_root)
    start = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    pair = build_synthetic_event_pair(seed=SEED)
    report = _base_report(started_utc=started_utc, dry_run=dry_run,
                          pair_sha256=_hash_pair_frames(pair),
                          source_sha256=_source_hashes(repo_root))
    stage = "imports"
    model = None
    try:
        host_before = host_available_memory_bytes()
        required_host = (MIN_HOST_AVAILABLE_BEFORE_PROCESSOR if dry_run
                         else MIN_HOST_AVAILABLE_BEFORE_MODEL)
        report["host_available_gib_before_processor"] = _memory_gib(host_before)
        if host_before < required_host:
            raise RuntimeError("insufficient_host_available_memory_before_processor")

        import torch
        import transformers
        import PIL
        from PIL import Image  # noqa: F401 - confirms PIL is available locally
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info
        from run_nextgqa_gold_interval_margin_pilot import (
            build_qa_prompt, process_qa_video,
        )

        snapshot = Path.home() / MODEL_SNAPSHOT_RELATIVE
        if not snapshot.is_dir():
            raise FileNotFoundError("pinned_checkpoint_not_cached_downloads_disabled")
        stage = "load_local_processor"
        processor = AutoProcessor.from_pretrained(
            str(snapshot), revision=MODEL_REVISION, local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
            trust_remote_code=False,
        )
        processor_sha, tokenizer_sha = _processor_fingerprint(processor)
        stage = "cpu_processor_preflight"
        prepared = prepare_pair_inputs(
            pair=pair, processor=processor, torch_module=torch,
            vision_info_fn=process_vision_info,
            build_prompt_fn=build_qa_prompt,
            process_video_fn=process_qa_video,
        )
        report["effective_videos"] = len(prepared)
        merge_size = int(processor.image_processor.merge_size)
        report["processor_input_shape_summary"] = processor_shape_summary(
            prepared, merge_size=merge_size)
        report["processor_spatial_merge_size"] = merge_size
        report["processor_sha256"] = processor_sha
        report["tokenizer_sha256"] = tokenizer_sha
        prompt_sha = prepared[0]["prompt_sha256"]
        if any(row["prompt_sha256"] != prompt_sha for row in prepared):
            raise ValueError("pair_prompt_hash_mismatch")
        report["input_bundle_sha256"] = prompt_sha
        runtime = _runtime_descriptor(
            torch, transformers, PIL.__version__,
            importlib.metadata.version("qwen-vl-utils"), cuda=not dry_run,
        )
        report["runtime_descriptor"] = runtime
        report["runtime_sha256"] = _canonical_sha256(runtime)
        report["cpu_processor_preflight"] = "passed"
        if dry_run:
            report.update({
                "status": "preflight_passed_no_model_forward",
                "model_weights_loaded": False,
                "gpu_used": False,
                "checkpoint_hash_status": "not_computed_dry_run",
                "elapsed_seconds": round(time.monotonic() - start, 3),
            })
            write_report_exclusive(output_path, report)
            return 0

        stage = "cuda_and_resource_preflight"
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda_bf16_unavailable")
        device = torch.cuda.current_device()
        free_vram, total_vram = torch.cuda.mem_get_info(device)
        host_before_load = host_available_memory_bytes()
        report.update({
            "gpu_device_name": torch.cuda.get_device_name(device),
            "cuda_version": str(torch.version.cuda),
            "free_vram_gib_before_load": _memory_gib(free_vram),
            "total_vram_gib": _memory_gib(total_vram),
            "host_available_gib_before_model_load": _memory_gib(host_before_load),
        })
        if free_vram < MIN_FREE_VRAM_BEFORE_LOAD:
            raise RuntimeError("insufficient_vram_before_model_load")
        if host_before_load < MIN_HOST_AVAILABLE_BEFORE_MODEL:
            raise RuntimeError("insufficient_host_memory_before_model_load")

        stage = "hash_local_checkpoint"
        checkpoint_sha, checkpoint_files, checkpoint_bytes = _checkpoint_hash(snapshot)
        report.update({
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_file_count": checkpoint_files,
            "checkpoint_bytes_hashed": checkpoint_bytes,
            "checkpoint_hash_status": "local_snapshot_files_hashed",
        })

        stage = "load_local_4bit_bf16_model"
        from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(snapshot), revision=MODEL_REVISION, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        ).eval()
        if (int(model.config.video_token_id) != int(processor.video_token_id)
                or getattr(model.config, "_attn_implementation", None) != "eager"):
            raise RuntimeError("model_processor_contract_mismatch")
        report["model_weights_loaded"] = True
        report["precision"] = "4bit_weights_bf16_compute_eager"
        report["gpu_used"] = True
        report["model_load_permitted"] = True
        report["model_load_seconds"] = round(time.monotonic() - start, 3)
        stage = "exact_two_forward_scoring"
        torch.cuda.reset_peak_memory_stats(device)
        before_forward_values = []

        def guard_forward() -> dict[str, float]:
            free, _total = torch.cuda.mem_get_info(device)
            host = host_available_memory_bytes()
            before_forward_values.append({
                "free_vram_gib": _memory_gib(free),
                "host_available_gib": _memory_gib(host),
            })
            report["forward_resource_checks"] = list(before_forward_values)
            if free < MIN_FREE_VRAM_BEFORE_FORWARD:
                raise RuntimeError("insufficient_vram_before_forward")
            if host < MIN_HOST_AVAILABLE_BEFORE_FORWARD:
                raise RuntimeError("insufficient_host_memory_before_forward")
            return before_forward_values[-1]

        scoring = score_two_inputs(
            model=model, prepared=prepared, torch_module=torch, device="cuda",
            before_forward=guard_forward,
            after_forward_cleanup=lambda: _release_cuda_after_forward(torch, device),
            progress=lambda attempted, completed: report.update({
                "attempted_forwards": attempted,
                "completed_forwards": completed,
            }),
        )
        report.update(scoring)
        report["forward_resource_checks"] = before_forward_values
        torch.cuda.synchronize(device)
        report["peak_allocated_vram_gib"] = _memory_gib(
            torch.cuda.max_memory_allocated(device))
        report["peak_reserved_vram_gib"] = _memory_gib(
            torch.cuda.max_memory_reserved(device))
        report["free_vram_gib_after_scoring"] = _memory_gib(
            torch.cuda.mem_get_info(device)[0])
        report["status"] = "passed" if scoring["completed_forwards"] == 2 else "incomplete"
        report["elapsed_seconds"] = round(time.monotonic() - start, 3)
        write_report_exclusive(output_path, report)
        return 0 if report["status"] == "passed" else 1
    except Exception as exc:
        report = _failure_report(report, stage=stage, exc=exc,
                                 started_monotonic=start)
        if "attempted_forwards" not in report:
            report["attempted_forwards"] = 0
        if "completed_forwards" not in report:
            report["completed_forwards"] = 0
        report["model_weights_loaded"] = model is not None
        if "gpu_used" not in report:
            report["gpu_used"] = False
        write_report_exclusive(output_path, report)
        return 1
    finally:
        if model is not None:
            del model


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run(output=args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
