"""Compare generated raw logits with fresh full-sequence teacher forcing.

One fixed, label-free engineering check: a four-frame NOAA EX2503 dive03
prompt is run once through the pinned Qwen2.5-VL checkpoint. Its two greedy
raw output tokens are frozen, then rescored in one fresh full-sequence forward
with native 3-row positions. This checks generation/teacher-forcing alignment
and model-path compatibility; it is not QA, claim validity, faithfulness, or an
intervention test.

The raw-logit tolerance (atol=0.1, rtol=0.01) and maximum per-token frozen
target log-probability difference tolerance (0.02 nats) are fixed engineering
tolerances, not efficacy thresholds. The
wrong-offset control compares rows P and P+1 instead of P-1 and P; it must
differ by more than 1e-6 nats in maximum selected-token log probability. No output token
IDs, text, raw logits, or absolute target scores are written.

Usage::

    PYTHONPATH=scripts .venv/bin/python scripts/run_qwen_frozen_teacher_forcing_gpu_smoke.py \
      --output results/qwen_frozen_teacher_forcing_gpu_smoke_YYYYMMDD_NN.json
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
import time

from run_dataset_video_gpu_smoke import sha256
from frozen_teacher_forced_score import freeze_generation_tail


SEED = 20261012
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MEDIA_SHA256 = "13ab1d0e12d9c6b9e776890d123f363c12c572d52d1f58464b57a2e98667f126"
EXPECTED_GRID = (2, 24, 46)
EXPECTED_PROMPT_TOKENS = 584
FRAME_COUNT = 4
GENERATED_TOKENS = 2
MIN_FREE_VRAM_BYTES = 12 * 1024**3
RAW_LOGIT_ATOL = 0.1
RAW_LOGIT_RTOL = 0.01
MAX_TOKEN_LOGPROB_DIFFERENCE_TOLERANCE_NATS = 0.02
WRONG_OFFSET_MIN_LOGPROB_DIFFERENCE_NATS = 1e-6
PROMPT = "Describe the visible scene briefly. Use only visual information."
RUN_PROGRESS = {
    "stage": "preflight",
    "completed_model_calls": 0,
    "effective_frames": 0,
    "generation_tokens_frozen": 0,
    "partial_evidence": {},
}


def resolve_output_path(repo: Path, output_arg: str) -> Path:
    output = (repo / output_arg).resolve()
    results_root = (repo / "results").resolve()
    if output.parent != results_root or output.exists():
        raise ValueError("exclusive new direct-child output under results/ required; overwrite refused")
    return output


def prediction_rows(full_logits, prompt_length: int, target_length: int):
    """Select full-forward next-token rows P-1..P+L-2 from [1,S,V] logits."""
    import torch

    if type(prompt_length) is not int or prompt_length <= 0:
        raise ValueError("prompt_length must be a positive integer")
    if type(target_length) is not int or target_length <= 0:
        raise ValueError("target_length must be a positive integer")
    if not isinstance(full_logits, torch.Tensor) or full_logits.ndim != 3:
        raise ValueError("full logits must be a tensor shaped [1, sequence, vocabulary]")
    if full_logits.shape[0] != 1 or full_logits.shape[1] != prompt_length + target_length:
        raise ValueError("full logits must cover one exact prompt-plus-target sequence")
    if full_logits.shape[2] <= 0 or not full_logits.dtype.is_floating_point:
        raise ValueError("full logits must have floating values and nonempty vocabulary")
    return full_logits[:, prompt_length - 1 : prompt_length + target_length - 1, :]


def _target_logprobs(logits, target_ids):
    """Return only the selected raw-target token logprobs in float64 math."""
    import torch

    if logits.ndim != 3 or tuple(logits.shape[:2]) != (1, len(target_ids)):
        raise ValueError("target logits must have shape [1, target length, vocabulary]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("target logits must be finite")
    ids = torch.as_tensor(target_ids, dtype=torch.long, device=logits.device).view(1, -1, 1)
    if bool((ids < 0).any()) or bool((ids >= logits.shape[-1]).any()):
        raise ValueError("frozen target ID is outside the logits vocabulary")
    values = logits.double().log_softmax(-1).gather(-1, ids).squeeze(0).squeeze(-1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("target log probabilities must be finite")
    return values


def compare_generation_to_teacher_forcing(generation_logits, full_logits, target, prompt_length):
    """Compare aligned raw logits and frozen-target sums; return aggregates only."""
    import torch

    if not isinstance(target.token_ids, tuple) or len(target.token_ids) != GENERATED_TOKENS:
        raise ValueError("frozen target must contain exactly two generated IDs")
    if target.prompt_length != prompt_length:
        raise ValueError("frozen target prompt length does not match input sequence")
    if not isinstance(generation_logits, torch.Tensor) or generation_logits.ndim != 3:
        raise ValueError("generation logits must have shape [1, target length, vocabulary]")
    if tuple(generation_logits.shape[:2]) != (1, len(target.token_ids)):
        raise ValueError("generation logits do not align with frozen target length")
    teacher_rows = prediction_rows(full_logits, prompt_length, len(target.token_ids))
    if generation_logits.shape[-1] != teacher_rows.shape[-1]:
        raise ValueError("generation and teacher-forced vocabulary widths differ")
    if not bool(torch.isfinite(generation_logits).all()) or not bool(torch.isfinite(teacher_rows).all()):
        raise ValueError("generation and teacher-forced logits must be finite")

    generation_logits = generation_logits.detach().to(device="cpu", dtype=torch.float64)
    teacher_rows = teacher_rows.detach().to(device="cpu", dtype=torch.float64)
    aligned_abs = (generation_logits - teacher_rows).abs()
    wrong_offset_rows = full_logits[:, prompt_length : prompt_length + len(target.token_ids), :]
    if wrong_offset_rows.shape != generation_logits.shape:
        raise ValueError("wrong-offset control rows are not available")
    wrong_offset_rows = wrong_offset_rows.detach().to(device="cpu", dtype=torch.float64)
    wrong_abs = (generation_logits - wrong_offset_rows).abs()
    generated_token_logprobs = _target_logprobs(generation_logits, target.token_ids)
    teacher_token_logprobs = _target_logprobs(teacher_rows, target.token_ids)
    token_score_abs_differences = (generated_token_logprobs - teacher_token_logprobs).abs()
    generated_sum = float(generated_token_logprobs.sum().item())
    teacher_sum = float(teacher_token_logprobs.sum().item())
    wrong_token_logprobs = _target_logprobs(wrong_offset_rows, target.token_ids)
    wrong_token_score_max_difference = float(
        (teacher_token_logprobs - wrong_token_logprobs).abs().max().item()
    )
    max_abs = float(aligned_abs.max().item())
    mean_abs = float(aligned_abs.mean().item())
    score_abs_difference = abs(generated_sum - teacher_sum)
    wrong_max_abs = float(wrong_abs.max().item())
    return {
        "target_sha256": target.target_sha256,
        "target_token_count": len(target.token_ids),
        "prediction_logit_rows": list(range(prompt_length - 1, prompt_length + len(target.token_ids) - 1)),
        "raw_logits_finite": True,
        "raw_logits_allclose_atol_0_1_rtol_0_01": bool(
            torch.allclose(generation_logits, teacher_rows, atol=RAW_LOGIT_ATOL, rtol=RAW_LOGIT_RTOL)
        ),
        "maximum_absolute_raw_logit_difference": max_abs,
        "mean_absolute_raw_logit_difference": mean_abs,
        "maximum_absolute_per_token_logprob_difference_nats": float(token_score_abs_differences.max().item()),
        "frozen_target_score_absolute_difference_nats": score_abs_difference,
        "max_per_token_logprob_within_0_02_nats": (
            float(token_score_abs_differences.max().item())
            <= MAX_TOKEN_LOGPROB_DIFFERENCE_TOLERANCE_NATS
        ),
        "wrong_offset_prediction_rows": list(range(prompt_length, prompt_length + len(target.token_ids))),
        "generation_vs_wrong_offset_maximum_absolute_raw_logit_difference": wrong_max_abs,
        "teacher_correct_vs_wrong_offset_maximum_absolute_per_token_logprob_difference_nats": wrong_token_score_max_difference,
        "wrong_offset_control_discriminates": wrong_token_score_max_difference > WRONG_OFFSET_MIN_LOGPROB_DIFFERENCE_NATS,
    }


def _normalize_native_rope(position_ids):
    if position_ids is None or position_ids.ndim != 3 or position_ids.shape[1] != 1:
        raise RuntimeError("generation omitted single-batch position IDs")
    if position_ids.shape[0] == 4:
        return position_ids[1:]
    if position_ids.shape[0] == 3:
        return position_ids
    raise RuntimeError("generation position IDs must contain three native rotary rows")


def _cache_len(cache):
    if cache is None:
        return 0
    if not hasattr(cache, "get_seq_length"):
        raise RuntimeError("generation cache did not expose sequence length")
    return int(cache.get_seq_length())


def _output_cache(output):
    cache = getattr(output, "past_key_values", None)
    if cache is not None:
        return cache
    if isinstance(output, (tuple, list)):
        return next((value for value in output if hasattr(value, "get_seq_length")), None)
    return None


class NativeGenerationTrace:
    """Capture only sequence lengths and native rotary positions at text LM."""

    def __init__(self, language_model):
        self.calls = []
        self.pending = []
        self.handles = [
            language_model.register_forward_pre_hook(self._pre, with_kwargs=True),
            language_model.register_forward_hook(self._post, with_kwargs=True),
        ]

    def _pre(self, _module, _args, kwargs):
        ids, embeds = kwargs.get("input_ids"), kwargs.get("inputs_embeds")
        length = int(ids.shape[-1]) if ids is not None else int(embeds.shape[-2])
        pids = kwargs.get("position_ids")
        if pids is None:
            raise RuntimeError("text model call omitted native position IDs")
        cache = kwargs.get("past_key_values")
        self.pending.append({
            "input_length": length,
            "past_length": _cache_len(cache),
            "position_ids": _normalize_native_rope(pids.detach().cpu()).clone(),
        })

    def _post(self, _module, _args, kwargs, output):
        if not self.pending:
            raise RuntimeError("text-model output had no paired input record")
        record = self.pending.pop(0)
        record["cache_length_after"] = _cache_len(_output_cache(output))
        self.calls.append(record)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def validate_native_trace(trace, prompt_positions, prompt_length):
    import torch

    if trace.pending or len(trace.calls) != GENERATED_TOKENS:
        raise RuntimeError("generation did not complete exactly two text-model calls")
    prefill, decode = trace.calls
    if (prefill["input_length"], prefill["past_length"], prefill["cache_length_after"]) != (
        prompt_length, 0, prompt_length
    ):
        raise RuntimeError("generation prefill cache lengths do not match prompt")
    if (decode["input_length"], decode["past_length"], decode["cache_length_after"]) != (
        1, prompt_length, prompt_length + 1
    ):
        raise RuntimeError("generation decode cache lengths do not match frozen target")
    if not torch.equal(prefill["position_ids"], prompt_positions.detach().cpu()):
        raise RuntimeError("generation prefill positions differ from native full prompt positions")
    if not torch.equal(decode["position_ids"], torch.full_like(decode["position_ids"], 55)):
        raise RuntimeError("first generated-token position is not the fixed native coordinate 55")
    return {
        "calls": len(trace.calls),
        "input_lengths": [prefill["input_length"], decode["input_length"]],
        "past_cache_lengths": [prefill["past_length"], decode["past_length"]],
        "cache_lengths_after": [prefill["cache_length_after"], decode["cache_length_after"]],
        "native_prefill_positions_match": True,
        "first_generated_token_position": 55,
    }


def validate_teacher_forcing_trace(trace, expected_positions, sequence_length):
    import torch

    if trace.pending or len(trace.calls) != 1:
        raise RuntimeError("teacher-forced model forward was not observed exactly once")
    record = trace.calls[0]
    if (record["input_length"], record["past_length"], record["cache_length_after"]) != (
        sequence_length, 0, 0
    ):
        raise RuntimeError("teacher-forced model call did not use one full uncached sequence")
    if not torch.equal(record["position_ids"], expected_positions.detach().cpu()):
        raise RuntimeError("teacher-forced text-model input positions differ from native positions")
    return {"model_calls": 1, "input_length": sequence_length,
            "positions_match_at_text_model": True, "cache_length": 0}


def _write_result(path: Path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_gate_result(path: Path, record, gate_failures):
    """Persist all completed evidence even when a fixed gate fails."""
    codes = tuple(gate_failures)
    record["gate_failures"] = list(codes)
    record["status"] = "ok" if not codes else "gate_failed"
    record["failures"] = len(codes)
    _write_result(path, record)
    return record


def _source_hashes(repo: Path, processor, process_vision_info):
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    from transformers.generation import utils as generation_utils

    return {
        **_source_code_hashes(repo),
        "installed_qwen_model": sha256(Path(inspect.getfile(modeling_qwen2_5_vl)).resolve()),
        "installed_generation_utils": sha256(Path(inspect.getfile(generation_utils)).resolve()),
        "installed_processor": sha256(Path(inspect.getfile(type(processor))).resolve()),
        "installed_video_processor": sha256(Path(inspect.getfile(type(processor.video_processor))).resolve()),
        "installed_vision_utils": sha256(Path(inspect.getfile(process_vision_info)).resolve()),
    }


def _source_code_hashes(repo: Path):
    return {
        "runner": sha256(Path(__file__).resolve()),
        "frame_sha_helper": sha256(repo / "scripts/run_dataset_video_gpu_smoke.py"),
        "frame_extractor": sha256(repo / "scripts/grounding_check.py"),
        "video_time_mapping": sha256(repo / "scripts/video_time_mapping.py"),
        "frozen_target_scorer": sha256(repo / "scripts/frozen_teacher_forced_score.py"),
    }


def _failure_provenance(repo: Path):
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        source_commit = None
    checkpoint = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" / REVISION
    checkpoint_hashes = {}
    for name, filename in (
        ("checkpoint_config_sha256", "config.json"),
        ("checkpoint_generation_config_sha256", "generation_config.json"),
        ("checkpoint_processor_config_sha256", "preprocessor_config.json"),
        ("checkpoint_tokenizer_config_sha256", "tokenizer_config.json"),
        ("checkpoint_tokenizer_sha256", "tokenizer.json"),
    ):
        path = checkpoint / filename
        if path.is_file():
            try:
                checkpoint_hashes[name] = sha256(path)
            except OSError:
                pass
    return {
        "source_commit": source_commit,
        "source_sha256": _source_code_hashes(repo),
        "checkpoint_revision": REVISION,
        "checkpoint_hashes_available_at_failure": checkpoint_hashes,
        "expected_media_sha256": MEDIA_SHA256,
        "expected_media_hash_validated": False,
    }


def run_gpu_smoke(output: Path):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from grounding_check import extract_frames

    repo = Path(__file__).resolve().parents[1]
    model_path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" / REVISION
    video = repo / "datasets/noaa_qa_pilot_20261011/media/source.mp4"
    RUN_PROGRESS.update(stage="preflight", completed_model_calls=0,
                        effective_frames=0, generation_tokens_frozen=0,
                        partial_evidence={})
    if not video.is_file() or sha256(video) != MEDIA_SHA256:
        raise RuntimeError("fixed NOAA source media missing or hash mismatch")
    RUN_PROGRESS["partial_evidence"].update({
        "expected_media_hash_validated": True,
        "media_sha256": MEDIA_SHA256,
    })
    if not model_path.is_dir():
        raise RuntimeError("pinned cached checkpoint missing; downloads are disabled")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < MIN_FREE_VRAM_BYTES:
        raise RuntimeError("at least 12 GiB free VRAM is required")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    RUN_PROGRESS.update(stage="extract_frames", completed_model_calls=0,
                        effective_frames=0, generation_tokens_frozen=0,
                        partial_evidence={
                            "started_utc": started,
                            "expected_media_sha256": MEDIA_SHA256,
                            "media_sha256": MEDIA_SHA256,
                            "media_hash_validated": True,
                            "expected_media_hash_validated": True,
                            "checkpoint_revision": REVISION,
                        })
    frames, temp_dir, timestamps, fps = extract_frames(str(video), FRAME_COUNT, 224)
    RUN_PROGRESS["effective_frames"] = len(timestamps)
    try:
        if len(frames) != FRAME_COUNT or len(timestamps) != FRAME_COUNT:
            raise RuntimeError("fixed run requires exactly four decoded frames")
        RUN_PROGRESS["stage"] = "load_model_processor"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        RUN_PROGRESS["partial_evidence"].update({
            "gpu_peak_tracking_started_before_model_load": True,
            "gpu_free_before_gib": free_bytes / 1024**3,
            "gpu_total_gib": total_bytes / 1024**3,
            "media_sha256": MEDIA_SHA256,
            "checkpoint_revision": REVISION,
            "quantization": "4-bit",
            "compute_dtype": "bfloat16",
            "attention_implementation": "eager",
            "seed": SEED,
            "requested_frames": FRAME_COUNT,
            "effective_frames": len(timestamps),
            "sampled_fps": float(fps),
            "sample_timestamps_seconds": [float(value) for value in timestamps],
        })
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            ),
        ).eval()
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
        )
        source_hashes = _source_hashes(repo, processor, process_vision_info)
        config_hashes = {
            "checkpoint_config_sha256": sha256(model_path / "config.json"),
            "checkpoint_generation_config_sha256": sha256(model_path / "generation_config.json"),
            "checkpoint_processor_config_sha256": sha256(model_path / "preprocessor_config.json"),
            "checkpoint_tokenizer_config_sha256": sha256(model_path / "tokenizer_config.json"),
            "checkpoint_tokenizer_sha256": sha256(model_path / "tokenizer.json"),
        }
        RUN_PROGRESS["partial_evidence"].update({
            "source_sha256": source_hashes,
            **config_hashes,
        })
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames, "sample_fps": fps},
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
        prompt_length = int(batch["input_ids"].shape[1])
        prompt_ids_sha256 = hashlib.sha256(
            json.dumps(
                tuple(int(value) for value in batch["input_ids"][0].detach().cpu().tolist()),
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        grid = tuple(int(v) for v in batch["video_grid_thw"][0].tolist())
        if prompt_length != EXPECTED_PROMPT_TOKENS or grid != EXPECTED_GRID:
            raise RuntimeError("fixed four-frame processor geometry changed")
        if not bool(batch["attention_mask"].bool().all()):
            raise RuntimeError("single prompt unexpectedly contains padding")
        if batch["mm_token_type_ids"].shape != batch["input_ids"].shape:
            raise RuntimeError("multimodal token map does not match prompt token shape")
        video_token_id = int(model.config.video_token_id)
        video_mask = batch["mm_token_type_ids"] == 2
        if not torch.equal(video_mask, batch["input_ids"] == video_token_id):
            raise RuntimeError("video marker positions disagree with processor token-type map")
        RUN_PROGRESS["partial_evidence"].update({
            "prompt_ids_sha256": prompt_ids_sha256,
            "prompt_tokens": prompt_length,
            "video_grid_thw": list(grid),
        })

        native_prompt_positions, prompt_delta = model.model.get_rope_index(
            input_ids=batch["input_ids"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            video_grid_thw=batch["video_grid_thw"],
            second_per_grid_ts=batch["second_per_grid_ts"],
            attention_mask=batch["attention_mask"],
        )
        if native_prompt_positions.shape != (3, 1, prompt_length):
            raise RuntimeError("native prompt positions have unexpected shape")
        if int(native_prompt_positions[0, 0, -1].item()) != 54:
            raise RuntimeError("fixed prompt last text coordinate differs from expected 54")

        RUN_PROGRESS["stage"] = "native_greedy_generation"
        model.model.rope_deltas = None
        if model.model.rope_deltas is not None:
            raise RuntimeError("rope deltas were not cleared before native generation")
        from transformers.cache_utils import DynamicCache
        fresh_cache = DynamicCache(config=model.config)
        trace = NativeGenerationTrace(model.model.language_model)
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    past_key_values=fresh_cache,
                    use_cache=True,
                    do_sample=False,
                    max_new_tokens=GENERATED_TOKENS,
                    eos_token_id=None,
                    pad_token_id=model.generation_config.pad_token_id,
                    forced_eos_token_id=None,
                    return_dict_in_generate=True,
                    output_logits=True,
                )
                torch.cuda.synchronize()
        finally:
            trace.close()
            RUN_PROGRESS["completed_model_calls"] += len(trace.calls)
        trace_summary = validate_native_trace(trace, native_prompt_positions, prompt_length)
        generation_logits_cpu = torch.stack(tuple(generated.logits), dim=1).detach().to(
            device="cpu", dtype=torch.float32
        )
        if tuple(generated.sequences.shape) != (1, prompt_length + GENERATED_TOKENS):
            raise RuntimeError("generation did not produce exactly two tokens")
        raw_sequence_cpu = generated.sequences[0].detach().to(device="cpu")
        frozen = freeze_generation_tail(
            raw_sequence_cpu, prompt_length=prompt_length, eos_token_id=None
        )
        RUN_PROGRESS["generation_tokens_frozen"] = len(frozen.token_ids)
        if len(frozen.token_ids) != GENERATED_TOKENS:
            raise RuntimeError("raw generation tail did not contain exactly two tokens")
        RUN_PROGRESS["partial_evidence"].update({
            "frozen_target_sha256": frozen.target_sha256,
            "frozen_target_token_count": len(frozen.token_ids),
            "generation_trace": trace_summary,
        })

        RUN_PROGRESS["stage"] = "native_full_teacher_forcing"
        target_ids_cuda = torch.tensor(
            frozen.token_ids, dtype=batch["input_ids"].dtype, device=batch["input_ids"].device
        ).view(1, -1)
        extended_ids = torch.cat((batch["input_ids"], target_ids_cuda), dim=1)
        extra = torch.zeros((1, GENERATED_TOKENS), dtype=batch["mm_token_type_ids"].dtype,
                            device=batch["mm_token_type_ids"].device)
        teacher_types = torch.cat((batch["mm_token_type_ids"], extra), dim=1)
        teacher_attention = torch.ones_like(extended_ids, dtype=batch["attention_mask"].dtype)
        teacher_positions, teacher_delta = model.model.get_rope_index(
            input_ids=extended_ids,
            mm_token_type_ids=teacher_types,
            video_grid_thw=batch["video_grid_thw"],
            second_per_grid_ts=batch["second_per_grid_ts"],
            attention_mask=teacher_attention,
        )
        if teacher_positions.shape != (3, 1, prompt_length + GENERATED_TOKENS):
            raise RuntimeError("full teacher-forcing positions have unexpected shape")
        if not torch.equal(teacher_positions[:, :, :prompt_length], native_prompt_positions):
            raise RuntimeError("native teacher-forcing prompt positions differ from generation prefill")
        expected_tail_positions = torch.tensor(
            [[[55, 56]], [[55, 56]], [[55, 56]]],
            dtype=teacher_positions.dtype,
            device=teacher_positions.device,
        )
        if not torch.equal(teacher_positions[:, :, prompt_length:], expected_tail_positions):
            raise RuntimeError("all three teacher-forced native position rows must be [55,56]")
        generated_positions = teacher_positions[:, 0, prompt_length:].detach().cpu().tolist()
        # Release generation's mutable KV cache and device output tensors before
        # the fresh full teacher-forcing pass. The frozen raw ID tuple and two
        # small CPU logit rows are the only generation outputs retained.
        del generated, fresh_cache, trace, raw_sequence_cpu, target_ids_cuda
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        teacher_batch = dict(batch)
        teacher_batch["input_ids"] = extended_ids
        teacher_batch["mm_token_type_ids"] = teacher_types
        teacher_batch["attention_mask"] = teacher_attention
        model.model.rope_deltas = None
        teacher_trace = NativeGenerationTrace(model.model.language_model)
        try:
            with torch.inference_mode():
                teacher_output = model(
                    **teacher_batch,
                    position_ids=teacher_positions,
                    use_cache=False,
                    return_dict=True,
                )
                torch.cuda.synchronize()
        finally:
            teacher_trace.close()
            RUN_PROGRESS["completed_model_calls"] += len(teacher_trace.calls)
        teacher_trace_summary = validate_teacher_forcing_trace(
            teacher_trace, teacher_positions, prompt_length + GENERATED_TOKENS
        )
        teacher_logits = teacher_output.logits
        if teacher_logits.shape[1] != prompt_length + GENERATED_TOKENS:
            raise RuntimeError("teacher-forced output logit sequence length changed")
        comparison = compare_generation_to_teacher_forcing(
            generation_logits_cpu, teacher_logits, frozen, prompt_length
        )
        del teacher_output, teacher_logits
        gc.collect()
        torch.cuda.synchronize()
        peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
        RUN_PROGRESS["peak_allocated_gib"] = peak_allocated_gib
        RUN_PROGRESS["partial_evidence"].update({
            "comparison": comparison,
            "teacher_forced_trace": teacher_trace_summary,
            "teacher_forced_native_positions_match_prompt": True,
            "teacher_forced_generated_token_positions": generated_positions,
            "completed_model_calls": RUN_PROGRESS["completed_model_calls"],
            "peak_allocated_gib": peak_allocated_gib,
        })

        gate_failures = []
        if not comparison["wrong_offset_control_discriminates"]:
            gate_failures.append("wrong_offset_token_score_control_not_discriminating")
        if not comparison["raw_logits_allclose_atol_0_1_rtol_0_01"]:
            gate_failures.append("raw_logits_outside_fixed_tolerance")
        if not comparison["max_per_token_logprob_within_0_02_nats"]:
            gate_failures.append("max_per_token_logprob_difference_outside_0_02_nats")
        if RUN_PROGRESS["completed_model_calls"] != 3:
            gate_failures.append("completed_model_call_count_not_three")
        record = {
            "schema_version": 1,
            "status": "ok",
            "purpose": "fixed_native_generation_vs_full_teacher_forcing_engineering_smoke",
            "started_utc": started,
            "elapsed_seconds": time.monotonic() - timer,
            "elapsed_scope": "runner workload from frame extraction through result write; excludes imports and initial preflight",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "source_sha256": source_hashes,
            "checkpoint_revision": REVISION,
            **config_hashes,
            "media_sha256": MEDIA_SHA256,
            "seed": SEED,
            "quantization": "4-bit",
            "compute_dtype": "bfloat16",
            "attention_implementation": "eager",
            "prompt_policy": "fixed generic video description prompt; prompt text not exported",
            "generation": {"do_sample": False, "max_new_tokens": GENERATED_TOKENS,
                           "eos_token_id": None, "forced_eos_token_id": None,
                           "output_logits": True, "use_cache": True},
            "teacher_forcing": {"one fresh full-sequence forward": True,
                                "position_ids": "native get_rope_index for prompt plus fixed generated tail",
                                "use_cache": False},
            "requested_frames": FRAME_COUNT,
            "effective_frames": len(timestamps),
            "sample_timestamps_seconds": [float(value) for value in timestamps],
            "sampled_fps": float(fps),
            "video_grid_thw": list(grid),
            "prompt_tokens": prompt_length,
            "prompt_ids_sha256": prompt_ids_sha256,
            "frozen_target_token_count": len(frozen.token_ids),
            "frozen_target_sha256": frozen.target_sha256,
            "target_provenance": "raw generation tail; caller provenance not independently verified",
            "generation_trace": trace_summary,
            "teacher_forced_trace": teacher_trace_summary,
            "teacher_forced_native_positions_match_prompt": True,
            "teacher_forced_generated_token_positions": generated_positions,
            "position_delta_summary": {"prompt_rope_delta": prompt_delta.detach().cpu().tolist(),
                                       "teacher_rope_delta": teacher_delta.detach().cpu().tolist()},
            "comparison": comparison,
            "raw_logit_tolerance_fixed": {"absolute": RAW_LOGIT_ATOL, "relative": RAW_LOGIT_RTOL},
            "max_per_token_logprob_tolerance_nats": MAX_TOKEN_LOGPROB_DIFFERENCE_TOLERANCE_NATS,
            "wrong_offset_control_minimum_max_token_logprob_difference_nats": WRONG_OFFSET_MIN_LOGPROB_DIFFERENCE_NATS,
            "requested_model_calls": 3,
            "completed_model_calls": RUN_PROGRESS["completed_model_calls"],
            "requested_samples": 1,
            "effective_samples": 1,
            "gpu_peak_tracking_started_before_model_load": True,
            "peak_allocated_gib": peak_allocated_gib,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_free_before_gib": free_bytes / 1024**3,
            "gpu_total_gib": total_bytes / 1024**3,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "opencv-python-headless")},
            "cuda_version": torch.version.cuda,
            "limitations": [
                "One licensed NOAA source, four uniform samples, one generic prompt and two greedy output tokens.",
                "Raw-logit and maximum per-token log-probability tolerances are fixed engineering checks, not efficacy criteria.",
                "This does not validate answer semantics, claim boundaries, QA, explanation faithfulness, or an intervention.",
                "Generated text, token IDs, absolute scores, logits, annotations and QA labels are not exported.",
            ],
            "privacy": "Derived metrics only; no local paths, prompt text, generated text, token IDs, logits, absolute target scores or annotations.",
        }
        record = _write_gate_result(output, record, gate_failures)
        RUN_PROGRESS["stage"] = "write_success_result" if not gate_failures else "write_gate_failure_result"
        print(json.dumps({
            "status": record["status"], "output": str(output.relative_to(repo)),
            "frames": len(timestamps), "model_calls": RUN_PROGRESS["completed_model_calls"],
            "max_logit_difference": round(comparison["maximum_absolute_raw_logit_difference"], 6),
            "max_token_logprob_abs_difference_nats": round(
                comparison["maximum_absolute_per_token_logprob_difference_nats"], 8
            ),
            "gate_failures": len(gate_failures),
            "elapsed_seconds": round(record["elapsed_seconds"], 2),
        }, sort_keys=True))
        return record
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="exclusive new summary JSON under results/")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    try:
        output = resolve_output_path(repo, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    cli_started_utc = datetime.now(timezone.utc).isoformat()
    cli_timer = time.monotonic()
    try:
        result = run_gpu_smoke(output)
        if result["status"] != "ok":
            raise SystemExit(1)
    except Exception as exc:
        partial = dict(RUN_PROGRESS.get("partial_evidence", {}))
        try:
            import torch
            if partial.get("gpu_peak_tracking_started_before_model_load") is True and torch.cuda.is_available():
                torch.cuda.synchronize()
                partial.setdefault("peak_allocated_gib", torch.cuda.max_memory_allocated() / 1024**3)
        except Exception:
            pass
        provenance = _failure_provenance(repo)
        provenance.update(partial)
        failure = {
            "schema_version": 1,
            "status": "failed",
            "purpose": "fixed_native_generation_vs_full_teacher_forcing_engineering_smoke",
            "stage": RUN_PROGRESS["stage"],
            "failure_started_utc": cli_started_utc,
            "elapsed_seconds": time.monotonic() - cli_timer,
            "elapsed_scope": "CLI after argument validation; excludes Python/module startup",
            "requested_frames": FRAME_COUNT,
            "effective_frames": RUN_PROGRESS["effective_frames"],
            "requested_model_calls": 3,
            "completed_model_calls": RUN_PROGRESS["completed_model_calls"],
            "requested_samples": 1,
            "effective_samples": int(RUN_PROGRESS["completed_model_calls"] == 3),
            "generation_tokens_frozen": RUN_PROGRESS["generation_tokens_frozen"],
            "failures": 1,
            "failure_type": type(exc).__name__,
            "exception_message_exported": False,
            "evidence": provenance,
        }
        _write_result(output, failure)
        print(json.dumps({"status": "failed", "stage": failure["stage"],
                          "completed_model_calls": failure["completed_model_calls"],
                          "failure_type": failure["failure_type"],
                          "output": str(output.relative_to(repo))}, sort_keys=True))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
