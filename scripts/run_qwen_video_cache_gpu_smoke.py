"""Fixed NOAA video smoke for the pinned Qwen KV-cache generation path.

This checks cached generation wiring only. It is not QA, efficacy, or semantic
validation. The run loads the pinned checkpoint offline, samples a fixed 4- or
32-frame profile from the NOAA clip, and compares native automatic,
explicit-native, and temporal-row-only candidate prefill positions. The
    default generates two greedy tokens; optional fixed 16- or 128-step cached decode
horizon exercises repeated cache growth without exporting generated content.

Usage::

    PYTHONPATH=scripts .venv/bin/python scripts/run_qwen_video_cache_gpu_smoke.py \
      --frames 32 --cached-decode-steps 16 \
      --output results/qwen_video_cache_gpu_smoke_YYYYMMDD_NN.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from qwen_temporal_geometry import replace_video_temporal_rope_row
from run_dataset_video_gpu_smoke import sha256
from run_qwen_temporal_position_gpu_check import assert_fractional_float32_compatibility


SEED = 20261012
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MEDIA_SHA256 = "13ab1d0e12d9c6b9e776890d123f363c12c572d52d1f58464b57a2e98667f126"
PROFILES = {
    4: {"grid": (2, 24, 46), "input_tokens": 584, "frame_size": 224,
        "max_pixels": 256 * 28 * 28, "min_free_vram_bytes": 12 * 1024**3},
    32: {"grid": (16, 24, 44), "input_tokens": 4256, "frame_size": 1024,
         "max_pixels": 200704, "min_free_vram_bytes": 13 * 1024**3},
}
MAX_NEW_TOKENS = 2
DEFAULT_CACHED_DECODE_STEPS = 1
PROMPT = "Describe the visible scene briefly. Use only visual information."
RUN_PROGRESS = {
    "stage": "preflight", "completed_model_calls": 0,
    "effective_conditions": 0, "effective_frames": 0,
}


def resolve_output_path(repo: Path, output_arg: str) -> Path:
    output = (repo / output_arg).resolve()
    results_root = (repo / "results").resolve()
    if output.parent != results_root or output.exists():
        raise ValueError("exclusive new direct-child output under results/ required; overwrite refused")
    return output


def normalize_rope_positions(position_ids):
    """Return the three rotary rows from native 3-row or packed 4-row IDs."""
    if position_ids is None or position_ids.ndim != 3 or position_ids.shape[1] != 1:
        raise ValueError("expected single-batch 3D position IDs")
    if position_ids.shape[0] == 4:
        return position_ids[1:]
    if position_ids.shape[0] == 3:
        return position_ids
    raise ValueError("expected three rotary rows, optionally preceded by text row")


def pack_generation_positions(rope_positions, sequence_length: int):
    """Prepend GenerationMixin's text coordinate row to three MRoPE rows."""
    if rope_positions.ndim != 3 or tuple(rope_positions.shape[:2]) != (3, 1):
        raise ValueError("rope positions must have shape [3, 1, sequence]")
    if rope_positions.shape[-1] != sequence_length or sequence_length < 1:
        raise ValueError("rope position length must equal the nonempty prompt length")
    text_positions = __import__("torch").arange(
        sequence_length, dtype=rope_positions.dtype, device=rope_positions.device
    ).view(1, 1, sequence_length)
    return __import__("torch").cat((text_positions, rope_positions), dim=0)


def cache_length(cache) -> int:
    if cache is None:
        return 0
    if not hasattr(cache, "get_seq_length"):
        raise ValueError("past_key_values does not expose get_seq_length")
    return int(cache.get_seq_length())


def _output_cache(output):
    cache = getattr(output, "past_key_values", None)
    if cache is not None:
        return cache
    if isinstance(output, (tuple, list)):
        for value in output:
            if hasattr(value, "get_seq_length"):
                return value
    return None


class GenerationCacheTrace:
    """Capture aggregate lengths and internal positions at the text model."""

    def __init__(self, language_model):
        self.calls = []
        self._handles = [
            language_model.register_forward_pre_hook(self._pre, with_kwargs=True),
            language_model.register_forward_hook(self._post, with_kwargs=True),
        ]
        self._pending = []

    def _pre(self, _module, _args, kwargs):
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        if input_ids is not None:
            input_length = int(input_ids.shape[-1])
        elif inputs_embeds is not None:
            input_length = int(inputs_embeds.shape[-2])
        else:
            raise RuntimeError("text-model call had neither token IDs nor embeddings")
        pids = kwargs.get("position_ids")
        if pids is None:
            raise RuntimeError("text-model call omitted position IDs")
        rope = normalize_rope_positions(pids.detach().cpu())
        past = kwargs.get("past_key_values")
        self._pending.append({
            "input_length": input_length,
            "past_length": cache_length(past),
            "position_ids": rope.clone(),
        })

    def _post(self, _module, _args, kwargs, output):
        if not self._pending:
            raise RuntimeError("post-hook observed a call without matching pre-hook")
        record = self._pending.pop(0)
        record["cache_length_after"] = cache_length(_output_cache(output))
        self.calls.append(record)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def validate_generation_trace(trace: GenerationCacheTrace, expected_prefill, expected_seq_len: int,
                              expected_decode_coordinate: int, cached_decode_steps: int = 1):
    """Fail closed unless every cached decode advances positions and KV length."""
    expected_call_count = 1 + cached_decode_steps
    if trace._pending or len(trace.calls) != expected_call_count:
        raise RuntimeError("unexpected number of completed prefill/decode calls")
    prefill, *decodes = trace.calls
    if (prefill["input_length"], prefill["past_length"], prefill["cache_length_after"]) != (
        expected_seq_len, 0, expected_seq_len
    ):
        raise RuntimeError("prefill input/cache lengths did not match the full prompt")
    if not __import__("torch").equal(prefill["position_ids"], expected_prefill.detach().cpu()):
        raise RuntimeError("captured prefill positions differ from requested positions")
    for step, decode in enumerate(decodes):
        expected = (1, expected_seq_len + step, expected_seq_len + step + 1)
        if (decode["input_length"], decode["past_length"], decode["cache_length_after"]) != expected:
            raise RuntimeError("cached decode did not process one token with expected KV lengths")
        expected_decode = __import__("torch").full(
            (3, 1, 1), expected_decode_coordinate + step, dtype=decode["position_ids"].dtype
        )
        if not __import__("torch").equal(decode["position_ids"], expected_decode):
            raise RuntimeError("cached generation did not advance all rotary rows at every decode step")
    return {
        "model_calls": len(trace.calls),
        "input_lengths": [prefill["input_length"], *[call["input_length"] for call in decodes]],
        "past_cache_lengths": [prefill["past_length"], *[call["past_length"] for call in decodes]],
        "cache_lengths_after": [prefill["cache_length_after"], *[call["cache_length_after"] for call in decodes]],
        "decode_steps": cached_decode_steps,
        "decode_start_coordinate": expected_decode_coordinate,
        "decode_end_coordinate": expected_decode_coordinate + cached_decode_steps - 1,
        "positions_verified": True,
    }


def _generation_outputs_finite(output, expected_new_tokens: int = MAX_NEW_TOKENS):
    import torch

    scores = getattr(output, "scores", None)
    logits = getattr(output, "logits", None)
    if scores is None or logits is None or len(scores) != expected_new_tokens or len(logits) != expected_new_tokens:
        raise RuntimeError("generation did not return the requested score and raw-logit tensors")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in (*scores, *logits)):
        raise RuntimeError("nonfinite generated scores or raw logits")
    return True


def _equal_tensor_tuples(left, right):
    import torch

    return len(left) == len(right) and all(torch.equal(a, b) for a, b in zip(left, right))


def _write_result(path: Path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def kv_cache_payload_estimate(config, sequence_length: int, dtype_bytes: int = 2):
    """Approximate dense full-attention K/V payload from pinned text config."""
    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    attention_heads = int(config.num_attention_heads)
    hidden_size = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden_size // attention_heads))
    payload_bytes = 2 * layers * kv_heads * head_dim * sequence_length * dtype_bytes
    return {
        "layers": layers,
        "key_value_heads": kv_heads,
        "head_dim": head_dim,
        "sequence_length": sequence_length,
        "bytes_per_element": dtype_bytes,
        "estimated_prefill_kv_payload_bytes": payload_bytes,
        "estimated_prefill_kv_payload_mib": payload_bytes / 1024**2,
        "scope": "dense BF16 K/V payload estimate; excludes allocator/workspace and implementation overhead",
    }


def run_gpu_smoke(output: Path, frame_count: int, cached_decode_steps: int = DEFAULT_CACHED_DECODE_STEPS):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from transformers.cache_utils import DynamicCache
    from qwen_vl_utils import process_vision_info
    from grounding_check import extract_frames

    repo = Path(__file__).resolve().parents[1]
    model_path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" / REVISION
    video = repo / "datasets/noaa_qa_pilot_20261011/media/source.mp4"
    if not video.is_file() or sha256(video) != MEDIA_SHA256:
        raise RuntimeError("fixed NOAA media missing or hash mismatch")
    if not model_path.is_dir():
        raise RuntimeError("pinned cached checkpoint missing; downloads are disabled")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    profile = PROFILES[frame_count]
    expected_new_tokens = cached_decode_steps + 1
    expected_model_calls = 3 * expected_new_tokens
    if free_bytes < profile["min_free_vram_bytes"]:
        raise RuntimeError("insufficient free VRAM for the selected fixed frame profile")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    RUN_PROGRESS.update(
        stage="extract_frames", completed_model_calls=0,
        effective_conditions=0, effective_frames=0,
        requested_model_calls=expected_model_calls,
    )
    frame_paths, temp_dir, timestamps, fps = extract_frames(
        str(video), frame_count, profile["frame_size"]
    )
    RUN_PROGRESS["effective_frames"] = len(timestamps)
    try:
        RUN_PROGRESS["stage"] = "load_model_and_processor"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            ),
        ).eval()
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=profile["max_pixels"],
        )
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frame_paths, "sample_fps": fps},
            {"type": "text", "text": PROMPT},
        ]}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages, return_video_metadata=True)
        if len(videos) != 1 or videos[0][0] is None:
            raise RuntimeError("processor did not yield exactly one video")
        video_tensor, metadata = videos[0]
        batch = processor(
            text=[prompt_text], images=images, videos=[video_tensor],
            video_metadata=[metadata], do_sample_frames=False,
            cap_pixels_per_frame=True, padding=True, return_tensors="pt",
        ).to("cuda")
        grid = tuple(int(value) for value in batch["video_grid_thw"][0].tolist())
        input_length = int(batch["input_ids"].shape[1])
        if (len(timestamps) != frame_count or grid != profile["grid"]
                or input_length != profile["input_tokens"]):
            raise RuntimeError("fixed NOAA processor geometry changed")
        video_mask = batch["mm_token_type_ids"][0] == 2
        image_mask = batch["mm_token_type_ids"][0] == 1
        token_mask = batch["input_ids"][0] == model.config.video_token_id
        attention_mask = batch["attention_mask"][0].bool()
        merge_size = int(processor.video_processor.merge_size)
        video_positions = (batch["input_ids"][0] == model.config.video_token_id).nonzero(as_tuple=True)[0]
        expected_video_tokens = grid[0] * (grid[1] // merge_size) * (grid[2] // merge_size)
        if (not bool(video_mask.any()) or bool(image_mask.any())
                or not torch.equal(video_mask, token_mask)
                or not bool(attention_mask.all())
                or len(video_positions) != expected_video_tokens
                or not torch.equal(video_positions, torch.arange(
                    int(video_positions[0]), int(video_positions[-1]) + 1,
                    device=video_positions.device,
                ))):
            raise RuntimeError("video-token mapping is not one contiguous video block")

        model_config = model.config.vision_config
        native, native_delta = model.model.get_rope_index(
            input_ids=batch["input_ids"], mm_token_type_ids=batch["mm_token_type_ids"],
            video_grid_thw=batch["video_grid_thw"],
            second_per_grid_ts=batch["second_per_grid_ts"],
            attention_mask=batch["attention_mask"],
        )
        span = (int(video_positions[0]), int(video_positions[-1]) + 1)
        candidate_rows = replace_video_temporal_rope_row(
            native[:, 0, :].detach().cpu().tolist(),
            video_token_spans=[span], image_token_spans=[],
            attention_mask=batch["attention_mask"][0].detach().cpu().tolist(),
            video_grid_thw=grid,
            spatial_merge_size=int(processor.video_processor.merge_size),
            second_per_grid_seconds=float(batch["second_per_grid_ts"][0]),
            tokens_per_second=int(model_config.tokens_per_second),
        )
        candidate = torch.tensor(candidate_rows, dtype=native.dtype, device=native.device).unsqueeze(1)
        native_single = native[:, 0, :].contiguous()
        seq_len = int(batch["input_ids"].shape[1])
        native_packed = pack_generation_positions(native_single.unsqueeze(1), seq_len)
        candidate_packed = pack_generation_positions(candidate[:, 0, :].unsqueeze(1), seq_len)
        if (not torch.equal(candidate[1:], native[1:])
                or not torch.equal(candidate[0, 0, ~video_mask], native[0, 0, ~video_mask])):
            raise RuntimeError("candidate modified non-temporal or non-video positions")
        last_text_coordinate = int(native_single[:, -1].max().item())
        decode_coordinate = last_text_coordinate + 1
        step_tokens = (grid[1] // merge_size) * (grid[2] // merge_size)
        native_anchors = [int(native_single[0, span[0] + i * step_tokens]) for i in range(grid[0])]
        candidate_anchors = [int(candidate[0, 0, span[0] + i * step_tokens]) for i in range(grid[0])]
        if (merge_size != 2 or len(native_anchors) != grid[0]
                or len(candidate_anchors) != grid[0]
                or native_anchors[0] != 15 or candidate_anchors[0] != 15
                or decode_coordinate != last_text_coordinate + 1):
            raise RuntimeError("fixed NOAA position geometry/continuation coordinates changed")
        expected_native_anchors = [
            native_anchors[0] + step * int(float(batch["second_per_grid_ts"][0]))
            * int(model_config.tokens_per_second)
            for step in range(grid[0])
        ]
        if native_anchors != expected_native_anchors:
            raise RuntimeError("native temporal anchors differ from pinned Transformers integer-interval rule")
        expected_candidate_anchors = torch.arange(
            grid[0], dtype=torch.float32
        ).mul_(torch.tensor(
            float(batch["second_per_grid_ts"][0]), dtype=torch.float32
        ) * int(model_config.tokens_per_second)).long().add_(candidate_anchors[0]).tolist()
        if candidate_anchors != expected_candidate_anchors:
            raise RuntimeError("candidate temporal anchors differ from declared float32 fractional rule")
        assert_fractional_float32_compatibility(
            candidate_anchors, float(batch["second_per_grid_ts"][0]),
            int(model_config.tokens_per_second),
        )

        conditions = []
        outputs = {}
        torch.cuda.empty_cache()
        peak_allocated = 0
        for name, supplied_positions, expected_prefill in (
            ("native_auto", None, native_single.unsqueeze(1)),
            ("explicit_native_noop", native_packed, native_single.unsqueeze(1)),
            ("temporal_candidate", candidate_packed, candidate[:, 0, :].unsqueeze(1)),
        ):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.model.rope_deltas = None
            if model.model.rope_deltas is not None:
                raise RuntimeError("rope_deltas was not cleared before generation")
            fresh_cache = DynamicCache(config=model.config)
            if cache_length(fresh_cache) != 0:
                raise RuntimeError("new generation cache was not empty")
            trace = GenerationCacheTrace(model.model.language_model)
            RUN_PROGRESS["stage"] = "generate_" + name
            begin = time.monotonic()
            try:
                with torch.inference_mode():
                    result = model.generate(
                        **batch,
                        **({} if supplied_positions is None else {"position_ids": supplied_positions}),
                        past_key_values=fresh_cache,
                        use_cache=True, do_sample=False, max_new_tokens=expected_new_tokens,
                        eos_token_id=None, pad_token_id=model.generation_config.pad_token_id,
                        forced_eos_token_id=None,
                        return_dict_in_generate=True, output_scores=True, output_logits=True,
                    )
                torch.cuda.synchronize()
            finally:
                trace.close()
                RUN_PROGRESS["completed_model_calls"] += len(trace.calls)
            trace_report = validate_generation_trace(
                trace, expected_prefill, expected_seq_len=seq_len,
                expected_decode_coordinate=decode_coordinate,
                cached_decode_steps=cached_decode_steps,
            )
            finite = _generation_outputs_finite(result, expected_new_tokens)
            if int(result.sequences.shape[-1]) != seq_len + expected_new_tokens:
                raise RuntimeError("generation returned an unexpected sequence length")
            peak = torch.cuda.max_memory_allocated() / 1024**3
            peak_allocated = max(peak_allocated, peak)
            conditions.append({
                "condition": name,
                "finite_generation_scores_and_raw_logits": finite,
                "trace": trace_report,
                "forward_seconds": time.monotonic() - begin,
                "peak_allocated_gib": peak,
            })
            RUN_PROGRESS["effective_conditions"] = len(conditions)
            outputs[name] = {
                "sequences": result.sequences.detach().cpu().clone(),
                "scores": tuple(tensor.detach().cpu().clone() for tensor in result.scores),
                "logits": tuple(tensor.detach().cpu().clone() for tensor in result.logits),
            }
            del result
            del fresh_cache, trace

        auto = outputs["native_auto"]
        noop = outputs["explicit_native_noop"]
        exact_noop = (
            torch.equal(auto["sequences"], noop["sequences"])
            and _equal_tensor_tuples(auto["scores"], noop["scores"])
            and _equal_tensor_tuples(auto["logits"], noop["logits"])
        )
        if not exact_noop:
            raise RuntimeError("native automatic and explicit-native generation were not identical")
        if (len(conditions) != 3
                or RUN_PROGRESS["completed_model_calls"] != expected_model_calls):
            raise RuntimeError("fixed run did not complete every requested model call")
        RUN_PROGRESS["stage"] = "write_success_result"
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
        from transformers.generation import utils as generation_utils
        source_paths = {
            "runner": Path(__file__).resolve(),
            "geometry_helper": repo / "scripts/qwen_temporal_geometry.py",
            "frame_smoke_helper": repo / "scripts/run_dataset_video_gpu_smoke.py",
            "frame_extractor": repo / "scripts/grounding_check.py",
            "video_time_mapping": repo / "scripts/video_time_mapping.py",
            "fractional_precision_guard": repo / "scripts/run_qwen_temporal_position_gpu_check.py",
            "installed_qwen_model": Path(inspect.getfile(modeling_qwen2_5_vl)).resolve(),
            "installed_generation_utils": Path(inspect.getfile(generation_utils)).resolve(),
            "installed_processor": Path(inspect.getfile(type(processor))).resolve(),
            "installed_video_processor": Path(inspect.getfile(type(processor.video_processor))).resolve(),
            "installed_vision_utils": Path(inspect.getfile(process_vision_info)).resolve(),
        }
        result_record = {
            "schema_version": 1,
            "status": "ok",
            "purpose": "fixed_noaa_qwen_video_cache_generation_compatibility_smoke",
            "started_utc": started,
            "elapsed_seconds": time.monotonic() - timer,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "source_sha256": {key: sha256(path) for key, path in source_paths.items()},
            "checkpoint_revision": REVISION,
            "checkpoint_config_sha256": sha256(model_path / "config.json"),
            "checkpoint_generation_config_sha256": sha256(model_path / "generation_config.json"),
            "checkpoint_processor_config_sha256": sha256(model_path / "preprocessor_config.json"),
            "media_sha256": MEDIA_SHA256,
            "seed": SEED,
            "quantization": "4-bit",
            "compute_dtype": "bfloat16",
            "attention_implementation": "eager",
            "cache": "fresh DynamicCache per condition; rope_deltas cleared before each generation",
            "kv_cache_payload_estimate": kv_cache_payload_estimate(
                model.config.get_text_config(decoder=True), seq_len, dtype_bytes=2
            ),
            "generation": {"do_sample": False, "max_new_tokens": expected_new_tokens,
                           "cached_decode_steps": cached_decode_steps,
                           "output_scores": True, "output_logits": True},
            "requested_samples": 1,
            "effective_samples": 1,
            "requested_conditions": 3,
            "effective_conditions": len(conditions),
            "requested_model_calls": expected_model_calls,
            "completed_model_calls": RUN_PROGRESS["completed_model_calls"],
            "failures": 0,
            "requested_frames": frame_count,
            "effective_frames": len(timestamps),
            "frame_max_dimension": profile["frame_size"],
            "max_pixels_per_frame": profile["max_pixels"],
            "sample_timestamps_seconds": [float(value) for value in timestamps],
            "sampled_fps": float(fps),
            "grid_thw": list(grid),
            "video_tokens": int(len(video_positions)),
            "input_tokens": seq_len,
            "second_per_grid_seconds": float(batch["second_per_grid_ts"][0]),
            "native_temporal_anchors": native_anchors,
            "candidate_temporal_anchors": candidate_anchors,
            "changed_video_temporal_position_entries": int((native[0] != candidate[0]).sum().item()),
            "non_temporal_positions_preserved": True,
            "native_auto_vs_explicit_native_exact_sequences_scores_logits": exact_noop,
            "last_text_coordinate": last_text_coordinate,
            "uninterrupted_decode_coordinate": decode_coordinate,
            "conditions": conditions,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_free_before_gib": free_bytes / 1024**3,
            "gpu_total_gib": total_bytes / 1024**3,
            "peak_allocated_gib": peak_allocated,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["torch", "transformers", "bitsandbytes", "qwen-vl-utils", "opencv-python-headless"]
            },
            "cuda_version": torch.version.cuda,
            "limitations": [
                f"One fixed {frame_count}-frame NOAA input; engineering compatibility only.",
                "Generation and cache wiring do not validate candidate temporal semantics, video understanding, QA, grounding, or faithfulness.",
                "No answer text, token IDs, raw scores, raw logits, annotations, or human QA labels are exported.",
                "The candidate is a temporal-row-only prefill edit; uninterrupted generation advances from the last prompt coordinate, not max(candidate position).",
            ],
            "privacy": "Derived aggregate only; no raw media, generated text, token IDs, logits, QA labels, or local absolute paths.",
        }
        _write_result(output, result_record)
        print(json.dumps({
            "status": "ok", "output": str(output.relative_to(repo)),
            "conditions": len(conditions), "model_calls": RUN_PROGRESS["completed_model_calls"],
            "exact_native_noop": exact_noop, "decode_position": decode_coordinate,
            "peak_allocated_gib": round(peak_allocated, 3),
            "elapsed_seconds": round(result_record["elapsed_seconds"], 2),
        }, sort_keys=True))
        return result_record
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="exclusive new summary JSON under results/")
    parser.add_argument("--frames", type=int, choices=tuple(PROFILES), default=4,
                        help="fixed input profile (4=baseline, 32=longer temporal context)")
    parser.add_argument("--cached-decode-steps", type=int, choices=(1, 16, 128), default=1,
                        help="cached one-token decode steps after prefill (1=baseline, 16=smoke, 128=long horizon)")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    try:
        output = resolve_output_path(repo, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    process_started = time.monotonic()
    try:
        run_gpu_smoke(output, args.frames, args.cached_decode_steps)
    except Exception as exc:
        if not output.exists():
            model_path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" / REVISION
            try:
                commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                commit = None
            local_sources = {
                "runner": Path(__file__).resolve(),
                "geometry_helper": repo / "scripts/qwen_temporal_geometry.py",
                "frame_smoke_helper": repo / "scripts/run_dataset_video_gpu_smoke.py",
                "frame_extractor": repo / "scripts/grounding_check.py",
                "video_time_mapping": repo / "scripts/video_time_mapping.py",
                "fractional_precision_guard": repo / "scripts/run_qwen_temporal_position_gpu_check.py",
            }
            record = {
                "schema_version": 1,
                "status": "failed",
                "purpose": "fixed_noaa_qwen_video_cache_generation_compatibility_smoke",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - process_started,
                "source_commit": commit,
                "runner_sha256": sha256(Path(__file__).resolve()),
                "available_source_sha256": {
                    name: sha256(path) for name, path in local_sources.items() if path.is_file()
                },
                "checkpoint_revision": REVISION,
                "checkpoint_config_sha256": sha256(model_path / "config.json") if (model_path / "config.json").is_file() else None,
                "checkpoint_generation_config_sha256": sha256(model_path / "generation_config.json") if (model_path / "generation_config.json").is_file() else None,
                "checkpoint_processor_config_sha256": sha256(model_path / "preprocessor_config.json") if (model_path / "preprocessor_config.json").is_file() else None,
                "media_sha256_expected": MEDIA_SHA256,
                "requested_conditions": 3,
                "effective_conditions": RUN_PROGRESS["effective_conditions"],
                "cached_decode_steps": args.cached_decode_steps,
                "requested_new_tokens": args.cached_decode_steps + 1,
                "requested_model_calls": 3 * (args.cached_decode_steps + 1),
                "completed_model_calls": RUN_PROGRESS["completed_model_calls"],
                "requested_frames": args.frames,
                "effective_frames": RUN_PROGRESS["effective_frames"],
                "failures": 1,
                "failure_type": type(exc).__name__,
                "stage": RUN_PROGRESS["stage"],
                "privacy": "Sanitized failure summary; no exception text, media, prompt, tokens, or logits.",
            }
            _write_result(output, record)
        print(json.dumps({
            "status": "failed", "failure_type": type(exc).__name__,
            "stage": RUN_PROGRESS["stage"], "completed_model_calls": RUN_PROGRESS["completed_model_calls"],
        }), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
