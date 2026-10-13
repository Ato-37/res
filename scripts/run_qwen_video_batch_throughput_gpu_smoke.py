"""Bounded Qwen2.5-VL processor/generation smoke for batch sizes one and two.

The two-item batch contains two independently processor-encoded copies of one
fixed, cleared NOAA clip. This is a batch-path and throughput engineering
diagnostic, not a representative heterogeneous-video benchmark or a QA/
faithfulness evaluation. It is intentionally fixed to four frames and SDPA.

Usage::

    PYTHONPATH=scripts .venv/bin/python scripts/run_qwen_video_batch_throughput_gpu_smoke.py \
      --output results/qwen_video_batch_throughput_gpu_smoke_YYYYMMDD_NN.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from run_dataset_video_gpu_smoke import sha256
from run_qwen_video_cache_gpu_smoke import GpuTelemetrySampler


REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MEDIA_SHA256 = "13ab1d0e12d9c6b9e776890d123f363c12c572d52d1f58464b57a2e98667f126"
SEED = 20261014
FRAME_COUNT = 4
FRAME_PROFILE = {
    "grid_thw": (2, 24, 46),
    "input_tokens": 584,
    "frame_size": 224,
    "max_pixels": 256 * 28 * 28,
    "min_free_vram_bytes": 12 * 1024**3,
}
MERGE_SIZE = 2
MAX_NEW_TOKENS = 17
DECODE_STEPS = 16
PROMPT = "Describe the visible scene briefly. Use only visual information."
RUN = {"stage": "preflight", "completed_model_calls": 0, "effective_batch_sizes": []}


def resolve_output_path(repo: Path, output_arg: str) -> Path:
    output = (repo / output_arg).resolve()
    results_root = (repo / "results").resolve()
    if output.parent != results_root or output.exists():
        raise ValueError("exclusive new direct-child output under results/ required; overwrite refused")
    return output


def _to_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return list(value)


def _tensor_digest(tensor) -> str:
    raw = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _grid_rows(batch, batch_size: int):
    grid = batch.get("video_grid_thw")
    if grid is None or getattr(grid, "ndim", None) != 2 or tuple(grid.shape) != (batch_size, 3):
        raise ValueError("processor video_grid_thw must contain one (T,H,W) row per item")
    return tuple(tuple(int(x) for x in row) for row in grid.detach().cpu().tolist())


def validate_processor_batch(batch, batch_size: int, video_token_id: int,
                             expected_grid=FRAME_PROFILE["grid_thw"]):
    """Fail closed on per-item token/grid/patch correspondence for Qwen video batches."""
    import torch

    input_ids = batch.get("input_ids")
    attention = batch.get("attention_mask")
    token_types = batch.get("mm_token_type_ids")
    pixels = batch.get("pixel_values_videos")
    seconds = batch.get("second_per_grid_ts")
    if input_ids is None or tuple(input_ids.shape) != (batch_size, FRAME_PROFILE["input_tokens"]):
        raise ValueError("processor input_ids shape differs from the fixed per-item profile")
    if attention is None or tuple(attention.shape) != tuple(input_ids.shape):
        raise ValueError("processor attention_mask shape differs from input_ids")
    if token_types is None or tuple(token_types.shape) != tuple(input_ids.shape):
        raise ValueError("processor mm_token_type_ids shape differs from input_ids")
    if not bool(attention.bool().all()):
        raise ValueError("fixed duplicate inputs unexpectedly contain padding")
    grids = _grid_rows(batch, batch_size)
    if any(grid != tuple(expected_grid) for grid in grids):
        raise ValueError("processor video grid differs from the fixed four-frame profile")
    if seconds is None or tuple(seconds.shape) != (batch_size,):
        raise ValueError("processor second_per_grid_ts must have one value per video")
    if not bool(torch.isfinite(seconds).all()) or bool((seconds <= 0).any()):
        raise ValueError("processor second_per_grid_ts contains invalid values")
    if pixels is None or getattr(pixels, "ndim", 0) < 2:
        raise ValueError("processor omitted flattened pixel_values_videos")

    video_id = int(video_token_id)
    if video_id < 0:
        raise ValueError("valid video token ID is required")
    expected_video_rows = []
    expected_patch_rows = []
    for row, grid in enumerate(grids):
        video_mask = token_types[row] == 2
        token_mask = input_ids[row] == video_id
        expected_tokens = grid[0] * (grid[1] // MERGE_SIZE) * (grid[2] // MERGE_SIZE)
        expected_patches = grid[0] * grid[1] * grid[2]
        positions = token_mask.nonzero(as_tuple=True)[0]
        if (not bool(video_mask.any()) or not torch.equal(video_mask, token_mask)
                or int(positions.numel()) != expected_tokens):
            raise ValueError("per-item video token/type/grid counts disagree")
        if not torch.equal(positions, torch.arange(
                int(positions[0]), int(positions[-1]) + 1, device=positions.device)):
            raise ValueError("per-item video tokens are not one contiguous block")
        expected_video_rows.append(expected_tokens)
        expected_patch_rows.append(expected_patches)
    if int(pixels.shape[0]) != sum(expected_patch_rows):
        raise ValueError("flattened video patch count does not match video_grid_thw")
    return {
        "batch_size": batch_size,
        "input_tokens_per_item": int(input_ids.shape[1]),
        "video_tokens_per_item": expected_video_rows,
        "grid_thw_per_item": [list(row) for row in grids],
        "patch_rows_per_item": expected_patch_rows,
        "pixel_values_video_shape": list(pixels.shape),
        "seconds_per_grid_count": int(seconds.numel()),
        "all_attention_positions_valid": True,
    }


def validate_duplicate_processor_batches(batch1, batch2, video_token_id: int):
    """Check actual B=2 processor output matches two separately encoded B=1 items."""
    import torch

    report1 = validate_processor_batch(batch1, 1, video_token_id)
    report2 = validate_processor_batch(batch2, 2, video_token_id)
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if not torch.equal(batch1[key][0].cpu(), batch2[key][0].cpu()) or not torch.equal(
                batch1[key][0].cpu(), batch2[key][1].cpu()):
            raise ValueError(f"duplicated processor item differs in {key}")
    grids1 = _grid_rows(batch1, 1)
    grids2 = _grid_rows(batch2, 2)
    if grids2 != (grids1[0], grids1[0]):
        raise ValueError("B=2 video grids do not duplicate the B=1 grid")
    seconds1 = batch1["second_per_grid_ts"].detach().cpu()
    seconds2 = batch2["second_per_grid_ts"].detach().cpu()
    if not torch.equal(seconds2, seconds1.repeat(2)):
        raise ValueError("B=2 per-video temporal intervals differ from B=1")
    pixels1, pixels2 = batch1["pixel_values_videos"], batch2["pixel_values_videos"]
    patch_rows = report1["patch_rows_per_item"][0]
    if (not torch.equal(pixels2[:patch_rows].cpu(), pixels1.cpu())
            or not torch.equal(pixels2[patch_rows:2 * patch_rows].cpu(), pixels1.cpu())):
        raise ValueError("B=2 flattened video patches differ from two B=1 processor outputs")
    return {"batch1": report1, "batch2": report2, "duplicate_items_match": True}


def _position_rows(position_ids, batch_size: int):
    if position_ids is None or position_ids.ndim != 3 or position_ids.shape[1] != batch_size:
        raise RuntimeError("expected batched three-dimensional rotary position IDs")
    if position_ids.shape[0] == 4:
        return position_ids[1:]
    if position_ids.shape[0] == 3:
        return position_ids
    raise RuntimeError("expected three rotary rows, optionally preceded by text row")


def _cache_batch_size(cache):
    if cache is None:
        return None
    layers = getattr(cache, "layers", None)
    if layers:
        keys = getattr(layers[0], "keys", None)
        if keys is not None and getattr(keys, "ndim", 0) >= 1 and keys.numel():
            return int(keys.shape[0])
    key_cache = getattr(cache, "key_cache", None)
    if key_cache:
        keys = key_cache[0]
        if getattr(keys, "ndim", 0) >= 1 and keys.numel():
            return int(keys.shape[0])
    return None


def _output_cache(output):
    cache = getattr(output, "past_key_values", None)
    if cache is not None:
        return cache
    if isinstance(output, (tuple, list)):
        return next((item for item in output if hasattr(item, "get_seq_length")), None)
    return None


class BatchGenerationTrace:
    def __init__(self, language_model, batch_size: int):
        self.batch_size = batch_size
        self.calls = []
        self.pending = []
        self.handles = [
            language_model.register_forward_pre_hook(self._pre, with_kwargs=True),
            language_model.register_forward_hook(self._post, with_kwargs=True),
        ]

    def _pre(self, _module, _args, kwargs):
        ids, embeds = kwargs.get("input_ids"), kwargs.get("inputs_embeds")
        batch_size = int(ids.shape[0]) if ids is not None else int(embeds.shape[0])
        if batch_size != self.batch_size:
            raise RuntimeError("language-model input batch size changed during generation")
        seq_len = int(ids.shape[-1]) if ids is not None else int(embeds.shape[-2])
        positions = _position_rows(kwargs.get("position_ids"), batch_size).detach().cpu().clone()
        past = kwargs.get("past_key_values")
        self.pending.append({
            "input_length": seq_len,
            "past_length": int(past.get_seq_length()) if past is not None else 0,
            "past_batch_size": _cache_batch_size(past),
            "position_ids": positions,
        })

    def _post(self, _module, _args, kwargs, output):
        if not self.pending:
            raise RuntimeError("language-model post-hook has no matching pre-hook")
        row = self.pending.pop(0)
        cache = _output_cache(output)
        row["cache_length_after"] = int(cache.get_seq_length()) if cache is not None else -1
        row["cache_batch_size_after"] = _cache_batch_size(cache)
        self.calls.append(row)

    def close(self):
        for handle in self.handles:
            handle.remove()


def validate_batch_trace(trace: BatchGenerationTrace, expected_prefill, decode_coordinates,
                         sequence_length: int, decode_steps: int = DECODE_STEPS):
    import torch

    if trace.pending or len(trace.calls) != 1 + decode_steps:
        raise RuntimeError("unexpected number of batched prefill/decode calls")
    prefill, *decodes = trace.calls
    if (prefill["input_length"], prefill["past_length"], prefill["cache_length_after"],
            prefill["cache_batch_size_after"]) != (sequence_length, 0, sequence_length, trace.batch_size):
        raise RuntimeError("batched prefill/cache lengths or cache batch dimension are invalid")
    if not torch.equal(prefill["position_ids"], expected_prefill.detach().cpu()):
        raise RuntimeError("captured batched prefill positions differ from native positions")
    for step, row in enumerate(decodes):
        expected_past = sequence_length + step
        if (row["input_length"], row["past_length"], row["past_batch_size"],
                row["cache_length_after"], row["cache_batch_size_after"]) != (
                1, expected_past, trace.batch_size, expected_past + 1, trace.batch_size):
            raise RuntimeError("batched cached decode did not advance sequence/cache batch correctly")
        coords = torch.as_tensor(decode_coordinates, dtype=row["position_ids"].dtype).view(1, -1, 1)
        expected = coords.expand(3, -1, -1).clone().add_(step)
        if not torch.equal(row["position_ids"], expected):
            raise RuntimeError("batched decode positions did not advance every rotary row/item")
    return {
        "model_calls": len(trace.calls),
        "batch_size_per_call": trace.batch_size,
        "prefill_input_length": prefill["input_length"],
        "cache_lengths_after": [row["cache_length_after"] for row in trace.calls],
        "cache_batch_size_after": [row["cache_batch_size_after"] for row in trace.calls],
        "positions_verified_per_item": True,
    }


def _finite_generation(output, batch_size: int, expected_tokens: int):
    import torch

    scores, logits = getattr(output, "scores", None), getattr(output, "logits", None)
    if scores is None or logits is None or len(scores) != expected_tokens or len(logits) != expected_tokens:
        raise RuntimeError("generation did not return the requested score/logit horizon")
    for tensor in (*scores, *logits):
        if tensor.ndim != 2 or tensor.shape[0] != batch_size or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("generation scores/logits have invalid batch shape or nonfinite values")
    if output.sequences.ndim != 2 or output.sequences.shape[0] != batch_size:
        raise RuntimeError("generated sequences have an invalid batch dimension")
    return True


def compare_generation_outputs(reference, candidate, batch_size: int):
    """Return scalar comparison summaries and never retain/export token/logit arrays."""
    import torch

    if len(reference.logits) != len(candidate.logits):
        raise ValueError("B=1 and B=2 generation horizons differ")
    if len(reference.logits) != MAX_NEW_TOKENS:
        raise ValueError("generation comparison did not contain the fixed 17-token horizon")

    def compare_tensor(left, right):
        left = left.detach().float().cpu()
        right = right.detach().float().cpu()
        difference = (left - right).abs()
        return {
            "maximum_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "top1_equal": bool(torch.equal(left.argmax(-1), right.argmax(-1))),
            "bitwise_equal": bool(torch.equal(left, right)),
        }

    steps = []
    reference_sequences = reference.sequences.detach().cpu()
    candidate_sequences = candidate.sequences.detach().cpu()
    prompt_length = int(reference_sequences.shape[-1]) - len(reference.logits)
    for index, (one, two) in enumerate(zip(reference.logits, candidate.logits)):
        if one.shape[0] != 1 or two.shape[0] != batch_size:
            raise ValueError("generation logits do not have expected batch dimensions")
        steps.append({
            "generated_step": index + 1,
            "b1_vs_b2_item0": compare_tensor(one[0], two[0]),
            "b1_vs_b2_item1": compare_tensor(one[0], two[1]),
            "b2_duplicate_items": compare_tensor(two[0], two[1]),
            "context_prefixes_equal": [
                bool(torch.equal(reference_sequences[0, :prompt_length + index],
                                 candidate_sequences[row, :prompt_length + index]))
                for row in range(batch_size)
            ],
        })
    sequence_checks = {
        "b2_duplicate_sequences_equal": bool(torch.equal(candidate_sequences[0], candidate_sequences[1])),
        "b1_sequence_equals_b2_item0": bool(torch.equal(reference_sequences[0], candidate_sequences[0])),
        "b1_sequence_equals_b2_item1": bool(torch.equal(reference_sequences[0], candidate_sequences[1])),
    }
    return {"per_generated_step_logits": steps, "sequence_equality": sequence_checks}


def output_equivalence_gate(comparison):
    """Require identical greedy sequences and each-step top-1 choices; retain scalar deltas."""
    sequences = comparison["sequence_equality"]
    sequences_equal = all(bool(value) for value in sequences.values())
    top1_equal = all(
        all(bool(step[key]["top1_equal"]) for key in (
            "b1_vs_b2_item0", "b1_vs_b2_item1", "b2_duplicate_items"
        ))
        for step in comparison["per_generated_step_logits"]
    )
    contexts_equal = all(
        all(step["context_prefixes_equal"])
        for step in comparison["per_generated_step_logits"]
    )
    return {
        "passed": sequences_equal and top1_equal and contexts_equal,
        "greedy_sequences_equal": sequences_equal,
        "per_step_top1_equal": top1_equal,
        "compared_context_prefixes_equal": contexts_equal,
        "logit_bitwise_equality_required": False,
        "numeric_logit_deltas_reported_descriptively": True,
    }


def _copy_logits_to_cpu(output):
    """Keep only temporary in-memory comparison tensors, never serialize them."""
    from types import SimpleNamespace

    return SimpleNamespace(
        logits=tuple(value.detach().float().cpu().clone() for value in output.logits),
        sequences=output.sequences.detach().cpu().clone(),
    )


def _write_result(path: Path, record):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _source_hashes(repo: Path, processor):
    from qwen_vl_utils import process_vision_info
    from transformers.generation import utils as generation_utils
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

    paths = {
        "runner": Path(__file__).resolve(),
        "cache_runner_helpers": Path(__file__).with_name("run_qwen_video_cache_gpu_smoke.py"),
        "frame_extractor": repo / "scripts/grounding_check.py",
        "frame_smoke_helper": repo / "scripts/run_dataset_video_gpu_smoke.py",
        "installed_qwen_model": Path(inspect.getfile(modeling_qwen2_5_vl)).resolve(),
        "installed_generation_utils": Path(inspect.getfile(generation_utils)).resolve(),
        "installed_processor": Path(inspect.getfile(type(processor))).resolve(),
        "installed_video_processor": Path(inspect.getfile(type(processor.video_processor))).resolve(),
        "installed_vision_utils": Path(inspect.getfile(process_vision_info)).resolve(),
    }
    return {key: sha256(path) for key, path in paths.items()}


def _failure_code(error, stage: str):
    name = type(error).__name__.lower()
    message = str(error).lower()
    if "outofmemory" in name or "out of memory" in message or "cuda oom" in message:
        return "cuda_oom"
    return {
        "preflight": "preflight_failed",
        "load_model": "model_load_failed",
        "prepare_batch_1": "processor_batch1_failed",
        "generate_batch_1": "generation_batch1_failed",
        "prepare_batch_2": "processor_batch2_failed",
        "generate_batch_2": "generation_batch2_failed",
    }.get(stage, "runtime_failed")


def _message(frame_paths, fps):
    return {"role": "user", "content": [
        {"type": "video", "video": frame_paths, "sample_fps": fps},
        {"type": "text", "text": PROMPT},
    ]}


def _prepare_processor_batch(processor, conversations, frame_paths, fps):
    from qwen_vl_utils import process_vision_info

    texts = [processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    ) for conversation in conversations]
    images, videos = process_vision_info(conversations, return_video_metadata=True)
    if images is not None or videos is None or len(videos) != len(conversations):
        raise RuntimeError("processor vision helper did not return one video per conversation")
    video_tensors = [entry[0] for entry in videos]
    video_metadata = [entry[1] for entry in videos]
    return processor(
        text=texts, images=images, videos=video_tensors, video_metadata=video_metadata,
        do_sample_frames=False, cap_pixels_per_frame=True,
        padding=True, return_tensors="pt",
    )


def _run_generation(model, batch, batch_size: int, native_positions, decode_coords,
                    max_new_tokens: int = MAX_NEW_TOKENS, collect_telemetry: bool = True,
                    measured: bool = True):
    import torch
    from transformers.cache_utils import DynamicCache

    device_batch = {
        key: value.to("cuda") if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    sequence_length = int(device_batch["input_ids"].shape[1])
    decode_steps = max_new_tokens - 1
    cache = DynamicCache(config=model.config)
    trace = BatchGenerationTrace(model.model.language_model, batch_size)
    telemetry = GpuTelemetrySampler(interval_seconds=1.0) if collect_telemetry else None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    if telemetry is not None:
        telemetry.start()
    try:
        with torch.inference_mode():
            output = model.generate(
                **device_batch, past_key_values=cache, use_cache=True,
                do_sample=False, max_new_tokens=max_new_tokens,
                eos_token_id=None,
                pad_token_id=model.generation_config.pad_token_id,
                forced_eos_token_id=None,
                return_dict_in_generate=True, output_scores=True, output_logits=True,
            )
        torch.cuda.synchronize()
    finally:
        telemetry_record = telemetry.stop() if telemetry is not None else None
        trace.close()
        RUN["completed_model_calls"] += len(trace.calls)
        if measured:
            RUN["measured_model_calls"] += len(trace.calls)
        else:
            RUN["warmup_model_calls"] += len(trace.calls)
    duration = time.monotonic() - started
    trace_record = validate_batch_trace(
        trace, native_positions, decode_coords, sequence_length, decode_steps
    )
    _finite_generation(output, batch_size, max_new_tokens)
    if int(output.sequences.shape[-1]) != sequence_length + max_new_tokens:
        raise RuntimeError("generated sequence length differs from requested token horizon")
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    record = {
        "batch_size": batch_size,
        "measured": measured,
        "model_calls": len(trace.calls),
        "generation_seconds": duration,
        "clips_per_second": batch_size / duration if duration > 0 else None,
        "seconds_per_clip": duration / batch_size if duration > 0 else None,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "trace": trace_record,
        "gpu_telemetry": telemetry_record,
    }
    del device_batch, cache
    return record, output


def run():
    repo = Path(__file__).resolve().parents[1]
    media_path = repo / "datasets/noaa_qa_pilot_20261011/media/source.mp4"
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"
                  / REVISION)
    results = []
    prepared1_cpu = None
    started_utc = datetime.now(timezone.utc).isoformat()
    run_started = time.monotonic()
    RUN.update(stage="preflight", started_utc=started_utc, batch_runs=[], warmup_runs=[],
               processor_comparison=None,
               current_batch_size=None,
               completed_model_calls=0, measured_model_calls=0, warmup_model_calls=0,
               effective_batch_sizes=[])
    source_commit = None
    version_record = {}
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from grounding_check import extract_frames

        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        version_record = {name: importlib.metadata.version(name) for name in
                          ["torch", "transformers", "bitsandbytes", "qwen-vl-utils", "opencv-python-headless"]}
        RUN.update(stage="preflight", completed_model_calls=0, measured_model_calls=0,
                   warmup_model_calls=0, effective_batch_sizes=[])
        if not media_path.is_file() or sha256(media_path) != MEDIA_SHA256:
            raise RuntimeError("fixed cleared media missing or hash mismatch")
        if not model_path.is_dir():
            raise RuntimeError("pinned local checkpoint missing; downloads are disabled")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support is required")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < FRAME_PROFILE["min_free_vram_bytes"]:
            raise RuntimeError("insufficient free VRAM for fixed four-frame batch smoke")
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        frame_paths, temp_dir, timestamps, fps = extract_frames(
            str(media_path), FRAME_COUNT, FRAME_PROFILE["frame_size"]
        )
        model = None
        try:
            RUN["stage"] = "load_model"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", device_map="cuda",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            ).eval()
            processor = AutoProcessor.from_pretrained(
                str(model_path), local_files_only=True,
                min_pixels=256 * 28 * 28, max_pixels=FRAME_PROFILE["max_pixels"],
            )
            RUN["source_sha256"] = _source_hashes(repo, processor)
            RUN["checkpoint_config_sha256"] = sha256(model_path / "config.json")
            RUN["checkpoint_generation_config_sha256"] = sha256(model_path / "generation_config.json")
            RUN["checkpoint_processor_config_sha256"] = sha256(model_path / "preprocessor_config.json")
            if len(timestamps) != FRAME_COUNT:
                raise RuntimeError("fixed clip extractor returned an unexpected frame count")
            # Encode B=1 via the actual processor API.
            RUN["stage"] = "prepare_batch_1"
            RUN["current_batch_size"] = 1
            conversations1 = [[_message(frame_paths, fps)]]
            prep1_start = time.monotonic()
            prepared1_cpu = _prepare_processor_batch(processor, conversations1, frame_paths, fps)
            prep1_seconds = time.monotonic() - prep1_start
            report1 = validate_processor_batch(
                prepared1_cpu, 1, int(model.config.video_token_id)
            )
            grid1 = _grid_rows(prepared1_cpu, 1)
            native1, _ = model.model.get_rope_index(
                input_ids=prepared1_cpu["input_ids"].to("cuda"),
                mm_token_type_ids=prepared1_cpu["mm_token_type_ids"].to("cuda"),
                video_grid_thw=prepared1_cpu["video_grid_thw"].to("cuda"),
                second_per_grid_ts=prepared1_cpu["second_per_grid_ts"].to("cuda"),
                attention_mask=prepared1_cpu["attention_mask"].to("cuda"),
            )
            coords1 = [int(native1[:, index, -1].max().item()) + 1 for index in range(1)]
            model.model.rope_deltas = None
            if model.model.rope_deltas is not None:
                raise RuntimeError("rope_deltas was not cleared before batch-one generation")
            RUN["stage"] = "warmup_batch_1"
            warmup1, warmup_output1 = _run_generation(
                model, prepared1_cpu, 1, native1, coords1,
                max_new_tokens=2, collect_telemetry=False, measured=False,
            )
            RUN["warmup_runs"].append(warmup1)
            del warmup_output1
            gc.collect()
            torch.cuda.empty_cache()
            model.model.rope_deltas = None

            # Encode B=2 by passing two conversations through process_vision_info and processor.
            RUN["stage"] = "prepare_batch_2"
            RUN["current_batch_size"] = 2
            conversations2 = [[_message(frame_paths, fps)], [_message(frame_paths, fps)]]
            prep2_start = time.monotonic()
            prepared2_cpu = _prepare_processor_batch(processor, conversations2, frame_paths, fps)
            prep2_seconds = time.monotonic() - prep2_start
            duplicate_report = validate_duplicate_processor_batches(
                prepared1_cpu, prepared2_cpu, int(model.config.video_token_id)
            )
            RUN["processor_comparison"] = duplicate_report
            if _grid_rows(prepared2_cpu, 2) != (grid1[0], grid1[0]):
                raise RuntimeError("B=2 processor grid differs from independently encoded B=1 profile")
            native2, _ = model.model.get_rope_index(
                input_ids=prepared2_cpu["input_ids"].to("cuda"),
                mm_token_type_ids=prepared2_cpu["mm_token_type_ids"].to("cuda"),
                video_grid_thw=prepared2_cpu["video_grid_thw"].to("cuda"),
                second_per_grid_ts=prepared2_cpu["second_per_grid_ts"].to("cuda"),
                attention_mask=prepared2_cpu["attention_mask"].to("cuda"),
            )
            if not torch.equal(native2[:, 0, :], native2[:, 1, :]):
                raise RuntimeError("duplicate native position rows are not identical")
            coords2 = [int(native2[:, index, -1].max().item()) + 1 for index in range(2)]
            model.model.rope_deltas = None
            if model.model.rope_deltas is not None:
                raise RuntimeError("rope_deltas was not cleared before batch-two generation")
            RUN["stage"] = "warmup_batch_2"
            warmup2, warmup_output2 = _run_generation(
                model, prepared2_cpu, 2, native2, coords2,
                max_new_tokens=2, collect_telemetry=False, measured=False,
            )
            RUN["warmup_runs"].append(warmup2)
            del warmup_output2
            gc.collect()
            torch.cuda.empty_cache()

            # Both batch shapes have now completed the same excluded warmup.
            RUN["current_batch_size"] = 1
            model.model.rope_deltas = None
            if model.model.rope_deltas is not None:
                raise RuntimeError("rope_deltas was not cleared before measured batch-one generation")
            RUN["stage"] = "generate_batch_1"
            run1, output1 = _run_generation(model, prepared1_cpu, 1, native1, coords1)
            run1["processor_seconds"] = prep1_seconds
            run1["processor_geometry"] = report1
            RUN["batch_runs"].append(run1)
            RUN["effective_batch_sizes"].append(1)
            logits1_cpu = _copy_logits_to_cpu(output1)
            del output1, native1
            gc.collect()
            torch.cuda.empty_cache()

            RUN["current_batch_size"] = 2
            model.model.rope_deltas = None
            if model.model.rope_deltas is not None:
                raise RuntimeError("rope_deltas was not cleared before measured batch-two generation")
            RUN["stage"] = "generate_batch_2"
            run2, output2 = _run_generation(model, prepared2_cpu, 2, native2, coords2)
            run2["processor_seconds"] = prep2_seconds
            run2["processor_geometry"] = duplicate_report["batch2"]
            RUN["batch_runs"].append(run2)
            RUN["effective_batch_sizes"].append(2)
            if (RUN["measured_model_calls"] != 2 * (1 + DECODE_STEPS)
                    or RUN["warmup_model_calls"] != 2 * (1 + 1)
                    or RUN["completed_model_calls"] != RUN["measured_model_calls"] + RUN["warmup_model_calls"]):
                raise RuntimeError("measured/warmup model call accounting is inconsistent")
            comparisons = compare_generation_outputs(logits1_cpu, output2, 2)
            equivalence = output_equivalence_gate(comparisons)
            RUN["processor_comparison"] = duplicate_report
            del output2, native2, prepared2_cpu, logits1_cpu
            RUN["stage"] = "complete"

            return {
                "schema_version": 1,
                "status": "ok" if equivalence["passed"] else "completed_with_output_mismatch",
                "purpose": "fixed_noaa_qwen_batch1_vs_batch2_processor_throughput_smoke",
                "started_utc": started_utc,
                "elapsed_seconds": time.monotonic() - run_started,
                "source_commit": source_commit,
                "source_sha256": RUN["source_sha256"],
                "checkpoint_revision": REVISION,
                "checkpoint_config_sha256": RUN["checkpoint_config_sha256"],
                "checkpoint_generation_config_sha256": RUN["checkpoint_generation_config_sha256"],
                "checkpoint_processor_config_sha256": RUN["checkpoint_processor_config_sha256"],
                "media_sha256": MEDIA_SHA256,
                "seed": SEED,
                "frame_count": FRAME_COUNT,
                "frame_timestamps_seconds": [float(value) for value in timestamps],
                "sampled_fps": float(fps),
                "attention_implementation_requested": "sdpa",
                "quantization": "4-bit",
                "compute_dtype": "bfloat16",
                "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                               "cached_decode_steps": DECODE_STEPS,
                               "output_scores": True, "output_logits": True,
                               "outputs_exported": False,
                               "batch_execution_order": ["warmup1", "warmup2", "measure1", "measure2"],
                               "warmup_new_tokens_per_batch": 2},
                "requested_batch_sizes": [1, 2],
                "requested_conditions": 2,
                "effective_batch_sizes": RUN["effective_batch_sizes"],
                "effective_conditions": len(RUN["effective_batch_sizes"]),
                "completed_model_calls": RUN["completed_model_calls"],
                "measured_model_calls": RUN["measured_model_calls"],
                "warmup_model_calls": RUN["warmup_model_calls"],
                "expected_model_calls": {
                    "measured": 2 * (1 + DECODE_STEPS),
                    "warmup": 2 * (1 + 1),
                    "total": 2 * (1 + DECODE_STEPS) + 2 * (1 + 1),
                },
                "warmup_runs": RUN["warmup_runs"],
                "processor_batch_comparison": duplicate_report,
                "batch_runs": [run1, run2],
                "batch2_throughput_ratio_vs_batch1": (
                    run2["clips_per_second"] / run1["clips_per_second"]
                    if run1["clips_per_second"] else None
                ),
                "numerical_comparison": comparisons,
                "output_equivalence_gate": equivalence,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_free_before_load_gib": free_bytes / 1024**3,
                "gpu_total_gib": total_bytes / 1024**3,
                "versions": version_record,
                "cuda_version": torch.version.cuda,
                "limitations": [
                    "One fixed four-frame NOAA clip is duplicated; this does not represent heterogeneous videos.",
                    "Throughput is an engineering observation for this clip, model, backend, and runtime only.",
                    "Requested SDPA is recorded; selected per-layer kernels are not introspected.",
                    "Numerical comparisons do not validate QA, temporal semantics, grounding, or faithfulness.",
                ],
                "privacy": "Aggregate only; no media, text, tokens, raw logits, QA labels, or local paths exported.",
                "failures": 0 if equivalence["passed"] else 1,
                "failure_codes": [] if equivalence["passed"] else ["batch_greedy_output_mismatch"],
            }
        finally:
            if model is not None:
                del model
            shutil = __import__("shutil")
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as error:
        code = _failure_code(error, RUN.get("stage", "unknown"))
        current_batch = RUN.get("current_batch_size")
        if current_batch in (1, 2) and current_batch not in RUN["effective_batch_sizes"]:
            RUN["batch_runs"].append({
                "batch_size": current_batch,
                "status": "failed",
                "n_requested": 1,
                "n_effective": 0,
                "model_calls_total_at_failure": RUN.get("completed_model_calls", 0),
                "failure_codes": [code],
                "failure_stage": RUN.get("stage", "unknown"),
            })
        return {
            "schema_version": 1,
            "status": "failed",
            "purpose": "fixed_noaa_qwen_batch1_vs_batch2_processor_throughput_smoke",
            "started_utc": started_utc,
            "elapsed_seconds": time.monotonic() - run_started,
            "source_sha256": RUN.get("source_sha256", {"runner": sha256(Path(__file__).resolve())}),
            "source_commit": source_commit,
            "versions": version_record,
            "checkpoint_revision": REVISION,
            "checkpoint_config_sha256": RUN.get("checkpoint_config_sha256"),
            "checkpoint_generation_config_sha256": RUN.get("checkpoint_generation_config_sha256"),
            "checkpoint_processor_config_sha256": RUN.get("checkpoint_processor_config_sha256"),
            "media_sha256": MEDIA_SHA256,
            "requested_batch_sizes": [1, 2],
            "requested_conditions": 2,
            "effective_batch_sizes": list(RUN["effective_batch_sizes"]),
            "effective_conditions": len(RUN["effective_batch_sizes"]),
            "attention_implementation_requested": "sdpa",
            "frame_count": FRAME_COUNT,
            "seed": SEED,
            "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                           "cached_decode_steps": DECODE_STEPS, "outputs_exported": False},
            "completed_model_calls": int(RUN["completed_model_calls"]),
            "measured_model_calls": int(RUN["measured_model_calls"]),
            "warmup_model_calls": int(RUN["warmup_model_calls"]),
            "expected_model_calls": {
                "measured": 2 * (1 + DECODE_STEPS),
                "warmup": 2 * (1 + 1),
                "total": 2 * (1 + DECODE_STEPS) + 2 * (1 + 1),
            },
            "failures": 1,
            "failure_codes": [code],
            "failure_stage": RUN.get("stage", "unknown"),
            "batch_runs": list(RUN.get("batch_runs", [])),
            "warmup_runs": list(RUN.get("warmup_runs", [])),
            "processor_batch_comparison": RUN.get("processor_comparison"),
            "privacy": "Sanitized aggregate failure only; exception text and local paths excluded.",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
                        help="exclusive new direct-child result path under results/")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output = resolve_output_path(repo, args.output)
    record = run()
    _write_result(output, record)
    print(json.dumps({
        "status": record["status"], "output": str(output.relative_to(repo)),
        "effective_batch_sizes": record.get("effective_batch_sizes", []),
        "completed_model_calls": record.get("completed_model_calls", 0),
        "failure_codes": record.get("failure_codes", []),
    }, sort_keys=True))
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
