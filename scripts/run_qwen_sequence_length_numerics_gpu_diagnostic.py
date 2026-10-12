"""Describe cached-vs-full-sequence numerical differences on one frozen target.

This diagnostic preserves the two-token target and fixed input from the prior
NOAA GPU01 smoke. It runs native cached generation once, then three fresh
native-position no-cache forwards at prompt lengths P=584, P+1=585 and P+2=586.
The nine fixed row comparisons characterize numerical paths; ten further
comparisons isolate output-projection input shape. They have no efficacy/pass
threshold and do not validate QA, claim meaning or faithfulness. The prior
raw-logit gate-failed artifact is left unchanged.

The diagnostic also records actual lm_head input sequence widths. Transformers
generation requests logits_to_keep=1 while direct Qwen forwards default to all
sequence rows, so comparing their logits alone conflates cache/sequence effects
with output-projection shape. Five full-forward hidden rows are therefore passed
through lm_head individually once; these scalar comparisons characterize that
shape path without changing the model or target.

No token IDs, prompt/output text, raw logits, per-token absolute scores or
arrays are exported. Scalar logit, log-probability and KL summaries are
diagnostic only. The NumPy/Torch/Python scorer tolerance checks arithmetic
agreement, not model outcomes.

Usage::

    PYTHONPATH=scripts .venv/bin/python scripts/run_qwen_sequence_length_numerics_gpu_diagnostic.py \
      --output results/qwen_sequence_length_numerics_gpu_diagnostic_YYYYMMDD_NN.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time

from run_qwen_frozen_teacher_forcing_gpu_smoke import (
    EXPECTED_GRID,
    EXPECTED_PROMPT_TOKENS,
    FRAME_COUNT,
    GENERATED_TOKENS,
    MEDIA_SHA256,
    MIN_FREE_VRAM_BYTES,
    PROMPT,
    REVISION,
    SEED,
    NativeGenerationTrace,
    _failure_provenance,
    _source_hashes,
    _source_code_hashes,
    _write_result,
    resolve_output_path,
    sha256,
    validate_native_trace,
    validate_teacher_forcing_trace,
)
from frozen_teacher_forced_score import (
    _logprob as python_stable_logprob,
    freeze_generation_tail,
    score_frozen_target,
)


BASELINE_RESULT = Path("results/qwen_frozen_teacher_forcing_gpu_smoke_20261012_01.json")
EXPECTED_TARGET_SHA256 = "6d46d93d937a1c6bc789da3de5f72bb8a2a8d44f47b30ffc408020734946ab0d"
EXPECTED_PROMPT_IDS_SHA256 = "24145443dae43eb82050d90c838d0cfad6b3aac86faf1e70bdbd9dc66eb5a18a"
CHECKPOINT_AUDIT_PATH = Path("results/qwen_checkpoint_bytes_cpu_20261012_01.json")
CHECKPOINT_AUDIT_ARTIFACT_SHA256 = "1892543e5dad3cba212b912fde19380c994f5b153da9a00281ba0f90faad8263"
CHECKPOINT_AUDIT_SOURCE_COMMIT = "bb4773b9128bf6be3585f5661e09ca2b3a951660"
CHECKPOINT_AUDIT_RUNNER_SHA256 = "eb2fd915f7139fdb2b3f21d5333a23d21bd3faf95ca435b2419a1bf918d923aa"
CHECKPOINT_AUDIT_EXPECTED_BYTES = 16_584_414_560
CHECKPOINT_AUDIT_SHARDS = tuple(f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6))
CHECKPOINT_AUDIT_API_SHA256 = "4b62c781f1f1e70ff315d119853e65f59f7103db52fbf11e0bd8eed3242ea06d"
CHECKPOINT_AUDIT_INDEX_BLOB_SHA1 = "f0c9e9b8e043f5de0ac99fd8b2ea75515d811e7b"
CHECKPOINT_AUDIT_INDEX_BYTES = 57_619
CHECKPOINT_AUDIT_INDEX_TENSOR_BYTES = 16_584_333_312
CHECKPOINT_AUDIT_EXPECTED_SHARD_METADATA = {
    "model-00001-of-00005.safetensors": (3_900_233_256, "e97b877e47fde53a6c6e77aafb36e58e91ee9d95c4a3eeac6f1b5c0e6a1c986e"),
    "model-00002-of-00005.safetensors": (3_864_726_320, "a9a300a43b4724eee2abe7c18ceb26768d0ab011eb0cad19d9bfd2476a24d024"),
    "model-00003-of-00005.safetensors": (3_864_726_424, "111223d173e00bbee81cba1216fad28668df3476706b7fd26f4d5b50f8b3a507"),
    "model-00004-of-00005.safetensors": (3_864_733_680, "ef47f634fa57d46ee134edcc09f34085a47da1e16c12a2abe0d67118be6d72ed"),
    "model-00005-of-00005.safetensors": (1_089_994_880, "0c859795ad3a627a9b95bcb762e059d5b768a4a36fdd4affeff269d93fdecc67"),
}
NUMERIC_CROSSCHECK_TOLERANCE = 1e-10
EXPECTED_PREFIX_LENGTHS = (584, 585, 586)
RUN_PROGRESS = {
    "stage": "preflight",
    "completed_model_calls": 0,
    "effective_frames": 0,
    "completed_teacher_prefixes": 0,
    "frozen_target_sha256": None,
    "partial_evidence": {},
}


def require_expected_target_hash(target_sha256: str) -> None:
    if target_sha256 != EXPECTED_TARGET_SHA256:
        raise ValueError("generated target SHA does not match fixed GPU01 baseline; stop without retry")


def build_comparison_plan(prompt_length: int = EXPECTED_PROMPT_TOKENS) -> tuple[dict, ...]:
    """Return the exact fixed 9-pair comparison matrix for suffix lengths 0/1/2."""
    if type(prompt_length) is not int or prompt_length <= 0:
        raise ValueError("prompt_length must be a positive strict integer")
    lengths = (prompt_length, prompt_length + 1, prompt_length + 2)
    rows = {0: prompt_length - 1, 1: prompt_length}
    pairs = []
    for token_index in (0, 1):
        for prefix_length in lengths[token_index:]:
            pairs.append({
                "left": f"generation_token_{token_index}",
                "right": f"full_{prefix_length}_row_{rows[token_index]}",
                "token_index": token_index,
                "prediction_row": rows[token_index],
                "comparison_type": "generation_vs_full_sequence",
            })
    for left_length, right_length in (
        (lengths[0], lengths[1]),
        (lengths[0], lengths[2]),
        (lengths[1], lengths[2]),
    ):
        pairs.append({
            "left": f"full_{left_length}_row_{rows[0]}",
            "right": f"full_{right_length}_row_{rows[0]}",
            "token_index": 0,
            "prediction_row": rows[0],
            "comparison_type": "full_sequence_length_pair",
        })
    pairs.append({
        "left": f"full_{lengths[1]}_row_{rows[1]}",
        "right": f"full_{lengths[2]}_row_{rows[1]}",
        "token_index": 1,
        "prediction_row": rows[1],
        "comparison_type": "full_sequence_length_pair",
    })
    if len(pairs) != 9:
        raise AssertionError("fixed comparison matrix must contain exactly nine pairs")
    return tuple(pairs)


def _numpy_logprobs(row):
    import numpy as np

    values = np.asarray(row, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("logit row must be a nonempty finite one-dimensional vector")
    peak = float(values.max())
    return (values - peak) - np.log(np.exp(values - peak).sum())


def _torch_metrics(left, right, target_token_id):
    import torch

    x = torch.as_tensor(left, dtype=torch.float64, device="cpu")
    y = torch.as_tensor(right, dtype=torch.float64, device="cpu")
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.numel() == 0:
        raise ValueError("paired logits must be same-shape nonempty one-dimensional vectors")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("paired logits must be finite")
    if type(target_token_id) is not int or target_token_id < 0 or target_token_id >= x.numel():
        raise ValueError("target token index is outside vocabulary")

    mean_x, mean_y = x.mean(), y.mean()
    diff = (x - y).abs()
    centered_diff = ((x - mean_x) - (y - mean_y)).abs()
    logp, logq = x.log_softmax(0), y.log_softmax(0)
    p, q = logp.exp(), logq.exp()
    kl_pq = (p * (logp - logq)).sum()
    kl_qp = (q * (logq - logp)).sum()
    return {
        "raw_max_abs": float(diff.max().item()),
        "raw_mean_abs": float(diff.mean().item()),
        "vocab_mean_offset_delta": float((mean_y - mean_x).item()),
        "centered_max_abs": float(centered_diff.max().item()),
        "centered_mean_abs": float(centered_diff.mean().item()),
        "logprob_max_abs": float((logp - logq).abs().max().item()),
        "logprob_mean_abs": float((logp - logq).abs().mean().item()),
        "selected_token_logprob_abs_difference_nats": float(abs((logp[target_token_id] - logq[target_token_id]).item())),
        "kl_left_to_right_nats": float(kl_pq.item()),
        "kl_right_to_left_nats": float(kl_qp.item()),
        "kl_symmetric_mean_nats": float(((kl_pq + kl_qp) / 2).item()),
        "left_selected_logprob_nats": float(logp[target_token_id].item()),
        "right_selected_logprob_nats": float(logq[target_token_id].item()),
    }


def _numpy_metrics(left, right, target_token_id):
    import numpy as np

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError("paired logits must be same-shape nonempty one-dimensional vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("paired logits must be finite")
    if type(target_token_id) is not int or target_token_id < 0 or target_token_id >= x.size:
        raise ValueError("target token index is outside vocabulary")
    logp, logq = _numpy_logprobs(x), _numpy_logprobs(y)
    p, q = np.exp(logp), np.exp(logq)
    kl_pq = float(np.sum(p * (logp - logq)))
    kl_qp = float(np.sum(q * (logq - logp)))
    raw_diff = np.abs(x - y)
    centered_diff = np.abs((x - x.mean()) - (y - y.mean()))
    logprob_diff = np.abs(logp - logq)
    return {
        "raw_max_abs": float(raw_diff.max()),
        "raw_mean_abs": float(raw_diff.mean()),
        "vocab_mean_offset_delta": float(y.mean() - x.mean()),
        "centered_max_abs": float(centered_diff.max()),
        "centered_mean_abs": float(centered_diff.mean()),
        "logprob_max_abs": float(logprob_diff.max()),
        "logprob_mean_abs": float(logprob_diff.mean()),
        "selected_token_logprob_abs_difference_nats": float(abs(logp[target_token_id] - logq[target_token_id])),
        "kl_left_to_right_nats": kl_pq,
        "kl_right_to_left_nats": kl_qp,
        "kl_symmetric_mean_nats": (kl_pq + kl_qp) / 2,
        "left_selected_logprob_nats": float(logp[target_token_id]),
        "right_selected_logprob_nats": float(logq[target_token_id]),
    }


def summarize_pair(pair, left, right, target_token_id):
    """Compute scalar pair metrics and independently cross-check with NumPy/Python."""
    torch_values = _torch_metrics(left, right, target_token_id)
    numpy_values = _numpy_metrics(left, right, target_token_id)
    python_left = python_stable_logprob([float(value) for value in left], target_token_id)
    python_right = python_stable_logprob([float(value) for value in right], target_token_id)
    errors = [abs(torch_values[key] - numpy_values[key]) for key in torch_values]
    errors.extend((
        abs(torch_values["left_selected_logprob_nats"] - python_left),
        abs(torch_values["right_selected_logprob_nats"] - python_right),
        abs(torch_values["selected_token_logprob_abs_difference_nats"] - abs(python_left - python_right)),
    ))
    max_error = max(errors)
    return {
        **{key: value for key, value in torch_values.items() if not key.endswith("selected_logprob_nats")},
        "left": pair["left"],
        "right": pair["right"],
        "comparison_type": pair["comparison_type"],
        "token_index": pair["token_index"],
        "prediction_row": pair["prediction_row"],
        "torch64_numpy_python_max_abs_error": max_error,
        "numeric_crosscheck_within_1e_10": max_error <= NUMERIC_CROSSCHECK_TOLERANCE,
    }


class LMHeadInputTrace:
    """Capture lm_head input shapes and detached hidden states for diagnostics."""

    def __init__(self, lm_head, selected_rows=None):
        self.calls = []
        self.selected_rows = selected_rows
        self.handle = lm_head.register_forward_pre_hook(self._pre)

    def _pre(self, _module, args):
        import torch

        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("lm_head input was not a tensor")
        hidden = args[0]
        if (hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] <= 0
                or not hidden.dtype.is_floating_point):
            raise RuntimeError("lm_head input must have shape [1, sequence, hidden]")
        if self.selected_rows is None:
            rows = tuple(range(int(hidden.shape[1])))
        else:
            rows_list = []
            for requested in self.selected_rows:
                if type(requested) is not int:
                    raise RuntimeError("lm_head capture row indices must be strict integers")
                row = requested if requested >= 0 else int(hidden.shape[1]) + requested
                if row < 0 or row >= hidden.shape[1]:
                    raise RuntimeError("lm_head capture row index is outside sequence")
                rows_list.append(row)
            if not rows_list or len(set(rows_list)) != len(rows_list):
                raise RuntimeError("lm_head capture rows must be nonempty and unique")
            rows = tuple(rows_list)
        selected = hidden[:, list(rows), :].detach()
        if not bool(torch.isfinite(selected).all()):
            raise RuntimeError("captured lm_head hidden rows must be finite")
        self.calls.append({
            "shape": tuple(int(value) for value in hidden.shape),
            "dtype": str(hidden.dtype),
            "captured_rows": rows,
            "hidden_cpu": selected.to(device="cpu").clone(),
        })

    def close(self):
        self.handle.remove()


def _captured_hidden_row(record, absolute_row: int):
    try:
        offset = record["captured_rows"].index(absolute_row)
    except (KeyError, ValueError) as exc:
        raise ValueError("requested hidden row was not captured") from exc
    return record["hidden_cpu"][:, offset:offset + 1, :]


class OuterForwardConfigTrace:
    """Capture effective and explicitly passed logits_to_keep per model call."""

    def __init__(self, model):
        self.calls = []
        self.handle = model.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, _module, _args, kwargs):
        value = kwargs.get("logits_to_keep", 0)
        if type(value) is not int or value < 0:
            raise RuntimeError("forward logits_to_keep must be a nonnegative integer")
        self.calls.append({
            "effective_logits_to_keep": value,
            "explicit_kwarg": "logits_to_keep" in kwargs,
        })

    def close(self):
        self.handle.remove()


def projection_comparison_plan(prompt_length: int = EXPECTED_PROMPT_TOKENS):
    """Five predeclared full rows, each paired with single-row and generation logits."""
    if type(prompt_length) is not int or prompt_length <= 0:
        raise ValueError("prompt_length must be a positive strict integer")
    specs = []
    for full_length, row in (
        (prompt_length, prompt_length - 1),
        (prompt_length + 1, prompt_length - 1),
        (prompt_length + 1, prompt_length),
        (prompt_length + 2, prompt_length - 1),
        (prompt_length + 2, prompt_length),
    ):
        specs.append({
            "full_length": full_length,
            "row": row,
            "full_name": f"full_{full_length}_row_{row}",
            "generation_name": f"generation_token_{row - (prompt_length - 1)}",
        })
    return tuple(specs)


def summarize_hidden_pair(pair, left, right):
    """Compare same-dtype finite hidden rows using Torch64 and NumPy scalars."""
    import numpy as np
    import torch

    x, y = torch.as_tensor(left), torch.as_tensor(right)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.numel() == 0:
        raise ValueError("paired hidden rows must be same-shape nonempty vectors")
    if x.dtype != y.dtype or not x.dtype.is_floating_point:
        raise ValueError("paired hidden rows must share a floating dtype")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("paired hidden rows must be finite")
    torch_diff = (x.to(dtype=torch.float64) - y.to(dtype=torch.float64)).abs()
    x_numpy = x.to(dtype=torch.float64).cpu().numpy()
    y_numpy = y.to(dtype=torch.float64).cpu().numpy()
    numpy_diff = np.abs(x_numpy - y_numpy)
    raw_max = float(torch_diff.max().item())
    raw_mean = float(torch_diff.mean().item())
    max_error = max(abs(raw_max - float(numpy_diff.max())), abs(raw_mean - float(numpy_diff.mean())))
    return {
        "left": pair["left"],
        "right": pair["right"],
        "prediction_row": pair["prediction_row"],
        "comparison_type": "hidden_row_sequence_length_pair",
        "hidden_dtype": str(x.dtype),
        "hidden_width": int(x.numel()),
        "raw_max_abs": raw_max,
        "raw_mean_abs": raw_mean,
        "torch64_numpy_max_abs_error": max_error,
        "numeric_crosscheck_within_1e_10": max_error <= NUMERIC_CROSSCHECK_TOLERANCE,
    }


def validate_checkpoint_audit(report: dict) -> None:
    """Fail closed unless Cycle14 authenticated the exact current checkpoint bytes."""
    if not isinstance(report, dict):
        raise ValueError("checkpoint byte audit must be a JSON object")
    required = {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "revision": REVISION,
        "byte_authentication_status": "verified_current_local_bytes",
        "historical_gpu_runs_byte_verified": False,
        "requested_shards": 5,
        "effective_shards": 5,
        "expected_shard_bytes": CHECKPOINT_AUDIT_EXPECTED_BYTES,
        "bytes_read": CHECKPOINT_AUDIT_EXPECTED_BYTES,
        "failure_count": 0,
        "failure_codes": [],
        "api_metadata_hash_matches_pinned_artifact": True,
        "api_metadata_sha256": CHECKPOINT_AUDIT_API_SHA256,
    }
    for key, expected in required.items():
        observed = report.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError("checkpoint byte audit does not satisfy the fixed provenance gate")
    if report.get("source_commit") != CHECKPOINT_AUDIT_SOURCE_COMMIT:
        raise ValueError("checkpoint byte audit source commit differs from the pinned artifact")
    if report.get("runner_sha256") != CHECKPOINT_AUDIT_RUNNER_SHA256:
        raise ValueError("checkpoint byte audit runner hash differs from the pinned artifact")
    shards = report.get("shards")
    if not isinstance(shards, list) or len(shards) != 5:
        raise ValueError("checkpoint byte audit must contain exactly five shard records")
    if tuple(item.get("filename") for item in shards if isinstance(item, dict)) != CHECKPOINT_AUDIT_SHARDS:
        raise ValueError("checkpoint byte audit shard identity/order differs")
    for item in shards:
        expected_size, expected_sha = CHECKPOINT_AUDIT_EXPECTED_SHARD_METADATA[item["filename"]]
        if (item.get("sha256_matches") is not True
                or item.get("stable_during_read") is not True
                or item.get("failure_code") is not None
                or type(item.get("observed_size_bytes")) is not int
                or item.get("observed_size_bytes") != expected_size
                or type(item.get("expected_size_bytes")) is not int
                or item.get("expected_size_bytes") != expected_size
                or type(item.get("bytes_read")) is not int
                or item.get("bytes_read") != expected_size
                or item.get("observed_sha256") != expected_sha
                or item.get("expected_sha256") != expected_sha):
            raise ValueError("checkpoint byte audit contains an unauthenticated shard")
    if sum(item["bytes_read"] for item in shards) != CHECKPOINT_AUDIT_EXPECTED_BYTES:
        raise ValueError("checkpoint byte audit shard byte total differs from fixed metadata")
    index = report.get("index")
    if (not isinstance(index, dict) or index.get("git_blob_sha1_matches") is not True
            or index.get("failure_code") is not None
            or index.get("expected_git_blob_sha1") != CHECKPOINT_AUDIT_INDEX_BLOB_SHA1
            or index.get("observed_git_blob_sha1") != CHECKPOINT_AUDIT_INDEX_BLOB_SHA1
            or type(index.get("expected_size_bytes")) is not int
            or index.get("expected_size_bytes") != CHECKPOINT_AUDIT_INDEX_BYTES
            or type(index.get("observed_size_bytes")) is not int
            or index.get("observed_size_bytes") != CHECKPOINT_AUDIT_INDEX_BYTES
            or type(report.get("index_weight_map_entries")) is not int
            or report.get("index_weight_map_entries") != 729
            or type(report.get("index_tensor_payload_bytes")) is not int
            or report.get("index_tensor_payload_bytes") != CHECKPOINT_AUDIT_INDEX_TENSOR_BYTES):
        raise ValueError("checkpoint byte audit index provenance is incomplete")


def load_checkpoint_audit(repo: Path):
    path = repo / CHECKPOINT_AUDIT_PATH
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CHECKPOINT_AUDIT_ARTIFACT_SHA256:
        raise ValueError("checkpoint byte audit artifact SHA differs from the published fixed artifact")
    report = json.loads(raw)
    validate_checkpoint_audit(report)
    return report, digest


def hash_checkpoint_shards(model_path: Path, *, status: str) -> dict:
    """Hash the exact shard paths immediately before ``from_pretrained``.

    The older byte-audit artifact authenticates the cache at audit time, but it
    cannot prove which bytes a historical model load consumed.  This in-run
    witness binds the current load to the fixed shard names, sizes and hashes.
    """
    records = []
    for filename in CHECKPOINT_AUDIT_SHARDS:
        path = model_path / filename
        if not path.is_file():
            raise RuntimeError(f"checkpoint shard missing before model load: {filename}")
        size = path.stat().st_size
        expected_size, expected_sha256 = CHECKPOINT_AUDIT_EXPECTED_SHARD_METADATA[filename]
        observed_sha256 = sha256(path)
        if size != expected_size or observed_sha256 != expected_sha256:
            raise RuntimeError(f"checkpoint shard changed or mismatched before model load: {filename}")
        records.append({"filename": filename, "bytes": size, "sha256": observed_sha256})
    return {
        "status": status,
        "shard_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "shards": records,
    }


def require_expected_target(target) -> None:
    require_expected_target_hash(target.target_sha256)
    if len(target.token_ids) != GENERATED_TOKENS or target.prompt_length != EXPECTED_PROMPT_TOKENS:
        raise ValueError("frozen target length/prompt boundary differs from GPU01 baseline")


def _position_tensor(model, batch, full_ids, suffix_count):
    import torch

    extra_types = torch.zeros(
        (1, suffix_count), dtype=batch["mm_token_type_ids"].dtype,
        device=batch["mm_token_type_ids"].device,
    )
    types = torch.cat((batch["mm_token_type_ids"], extra_types), dim=1)
    attention = torch.ones_like(full_ids, dtype=batch["attention_mask"].dtype)
    positions, delta = model.model.get_rope_index(
        input_ids=full_ids,
        mm_token_type_ids=types,
        video_grid_thw=batch["video_grid_thw"],
        second_per_grid_ts=batch["second_per_grid_ts"],
        attention_mask=attention,
    )
    prompt_length = int(batch["input_ids"].shape[1])
    if positions.shape != (3, 1, prompt_length + suffix_count):
        raise RuntimeError("native full-prefix positions have unexpected shape")
    return positions, types, attention, delta


def run_gpu_diagnostic(output: Path):
    RUN_PROGRESS.update(
        stage="preflight",
        completed_model_calls=0,
        effective_frames=0,
        completed_teacher_prefixes=0,
        frozen_target_sha256=None,
        partial_evidence={},
    )
    import numpy as np
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from transformers.cache_utils import DynamicCache
    from qwen_vl_utils import process_vision_info
    from grounding_check import extract_frames

    repo = Path(__file__).resolve().parents[1]
    baseline_path = repo / BASELINE_RESULT
    if not baseline_path.is_file():
        raise RuntimeError("fixed GPU01 baseline artifact is missing")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("frozen_target_sha256") != EXPECTED_TARGET_SHA256:
        raise RuntimeError("GPU01 artifact target SHA does not match the predeclared target")
    if baseline.get("prompt_ids_sha256") != EXPECTED_PROMPT_IDS_SHA256:
        raise RuntimeError("GPU01 artifact prompt ID digest differs from the fixed input")
    if baseline.get("media_sha256") != MEDIA_SHA256 or baseline.get("prompt_tokens") != EXPECTED_PROMPT_TOKENS:
        raise RuntimeError("GPU01 input provenance differs from the fixed diagnostic fixture")
    RUN_PROGRESS["partial_evidence"].update({
        "baseline_artifact_sha256": sha256(baseline_path),
        "baseline_source_commit": baseline.get("source_commit"),
        "baseline_target_sha256": baseline.get("frozen_target_sha256"),
        "baseline_prompt_ids_sha256": baseline.get("prompt_ids_sha256"),
        "expected_media_sha256": MEDIA_SHA256,
    })
    checkpoint_audit, checkpoint_audit_sha256 = load_checkpoint_audit(repo)
    RUN_PROGRESS["partial_evidence"].update({
        "checkpoint_byte_audit_path": CHECKPOINT_AUDIT_PATH.as_posix(),
        "checkpoint_byte_audit_sha256": checkpoint_audit_sha256,
        "checkpoint_byte_audit_source_commit": checkpoint_audit["source_commit"],
        "checkpoint_byte_audit_status": checkpoint_audit["byte_authentication_status"],
        "checkpoint_byte_audit_current_bytes_only": True,
        "checkpoint_byte_audit_historical_gpu_limit": True,
    })

    model_path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" / REVISION
    video = repo / "datasets/noaa_qa_pilot_20261011/media/source.mp4"
    if not video.is_file() or sha256(video) != MEDIA_SHA256:
        raise RuntimeError("fixed NOAA source media missing or hash mismatch")
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
                        effective_frames=0, completed_teacher_prefixes=0,
                        frozen_target_sha256=None)
    frames, temp_dir, timestamps, fps = extract_frames(str(video), FRAME_COUNT, 224)
    RUN_PROGRESS["effective_frames"] = len(timestamps)
    RUN_PROGRESS["partial_evidence"].update({
        "started_utc": started,
        "media_sha256": MEDIA_SHA256,
        "media_hash_validated": True,
        "checkpoint_revision": REVISION,
        "requested_frames": FRAME_COUNT,
        "effective_frames": len(timestamps),
        "sampled_fps": float(fps),
        "sample_timestamps_seconds": [float(value) for value in timestamps],
    })
    try:
        if len(frames) != FRAME_COUNT or len(timestamps) != FRAME_COUNT:
            raise RuntimeError("fixed diagnostic requires exactly four decoded frames")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        RUN_PROGRESS["partial_evidence"]["gpu_peak_tracking_started_before_model_load"] = True
        checkpoint_shards_before_model_load = hash_checkpoint_shards(
            model_path, status="verified_immediately_before_model_load"
        )
        RUN_PROGRESS["partial_evidence"]["checkpoint_shards_before_model_load"] = checkpoint_shards_before_model_load
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            ),
        ).eval()
        checkpoint_shards_after_model_load = hash_checkpoint_shards(
            model_path, status="verified_immediately_after_model_load"
        )
        if checkpoint_shards_before_model_load["shards"] != checkpoint_shards_after_model_load["shards"]:
            raise RuntimeError("checkpoint shard bytes changed during model load")
        RUN_PROGRESS["partial_evidence"]["checkpoint_shards_after_model_load"] = checkpoint_shards_after_model_load
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
        )
        source_hashes = _source_hashes(repo, processor, process_vision_info)
        source_hashes["sequence_length_diagnostic_runner"] = sha256(Path(__file__).resolve())
        source_hashes["base_teacher_forcing_runner"] = source_hashes.pop("runner")
        config_hashes = {
            "checkpoint_config_sha256": sha256(model_path / "config.json"),
            "checkpoint_generation_config_sha256": sha256(model_path / "generation_config.json"),
            "checkpoint_processor_config_sha256": sha256(model_path / "preprocessor_config.json"),
            "checkpoint_tokenizer_config_sha256": sha256(model_path / "tokenizer_config.json"),
            "checkpoint_tokenizer_sha256": sha256(model_path / "tokenizer.json"),
        }
        RUN_PROGRESS["partial_evidence"].update({"source_sha256": source_hashes, **config_hashes})
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames, "sample_fps": fps},
            {"type": "text", "text": PROMPT},
        ]}]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            json.dumps(tuple(int(x) for x in batch["input_ids"][0].detach().cpu().tolist()),
                       separators=(",", ":")).encode("ascii")
        ).hexdigest()
        grid = tuple(int(x) for x in batch["video_grid_thw"][0].tolist())
        if prompt_length != EXPECTED_PROMPT_TOKENS or grid != EXPECTED_GRID:
            raise RuntimeError("processor geometry differs from the fixed GPU01 input")
        if prompt_ids_sha256 != EXPECTED_PROMPT_IDS_SHA256:
            raise RuntimeError("prompt ID digest differs from GPU01; refuse a replacement input")
        if not bool(batch["attention_mask"].bool().all()):
            raise RuntimeError("fixed single prompt unexpectedly contains padding")
        if not torch.equal(batch["mm_token_type_ids"] == 2,
                           batch["input_ids"] == int(model.config.video_token_id)):
            raise RuntimeError("processor video-token map differs from input marker IDs")
        native_prompt_positions, _prompt_delta = model.model.get_rope_index(
            input_ids=batch["input_ids"], mm_token_type_ids=batch["mm_token_type_ids"],
            video_grid_thw=batch["video_grid_thw"], second_per_grid_ts=batch["second_per_grid_ts"],
            attention_mask=batch["attention_mask"],
        )
        RUN_PROGRESS["partial_evidence"].update({
            "prompt_tokens": prompt_length,
            "prompt_ids_sha256": prompt_ids_sha256,
            "video_grid_thw": list(grid),
        })

        RUN_PROGRESS["stage"] = "native_generation"
        model.model.rope_deltas = None
        generation_cache = DynamicCache(config=model.config)
        generation_trace = NativeGenerationTrace(model.model.language_model)
        generation_projection_trace = LMHeadInputTrace(model.lm_head, selected_rows=(-1,))
        generation_forward_config_trace = OuterForwardConfigTrace(model)
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    past_key_values=generation_cache,
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
            generation_trace.close()
            generation_projection_trace.close()
            generation_forward_config_trace.close()
            RUN_PROGRESS["completed_model_calls"] += len(generation_trace.calls)
        if len(generation_projection_trace.calls) != GENERATED_TOKENS:
            raise RuntimeError("generation did not invoke lm_head exactly once per token")
        if len(generation_forward_config_trace.calls) != GENERATED_TOKENS:
            raise RuntimeError("generation did not invoke outer model exactly once per token")
        generation_projection_inputs = generation_projection_trace.calls
        generation_forward_configs = generation_forward_config_trace.calls
        generation_trace_summary = validate_native_trace(
            generation_trace, native_prompt_positions, prompt_length
        )
        if tuple(generated.sequences.shape) != (1, prompt_length + GENERATED_TOKENS):
            raise RuntimeError("generation did not produce exactly two raw tokens")
        frozen_target = freeze_generation_tail(
            generated.sequences[0].detach().cpu(), prompt_length=prompt_length, eos_token_id=None
        )
        require_expected_target(frozen_target)
        RUN_PROGRESS["frozen_target_sha256"] = frozen_target.target_sha256
        RUN_PROGRESS["partial_evidence"].update({
            "frozen_target_sha256": frozen_target.target_sha256,
            "frozen_target_token_count": len(frozen_target.token_ids),
            "generation_trace": generation_trace_summary,
        })
        generated_rows = torch.stack(tuple(generated.logits), dim=1).detach().to(
            device="cpu", dtype=torch.float64
        )[0]
        if generated_rows.shape[0] != GENERATED_TOKENS or not bool(torch.isfinite(generated_rows).all()):
            raise RuntimeError("generation output logits are missing, malformed or nonfinite")
        generated_by_name = {
            f"generation_token_{index}": generated_rows[index].clone()
            for index in range(GENERATED_TOKENS)
        }
        del generated, generation_cache, generation_trace, generated_rows
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        teacher_rows = {}
        teacher_projection_inputs = {}
        teacher_forward_configs = []
        scorer_crosscheck_max_abs_error = 0.0
        teacher_trace_summaries = []
        rope_delta_summaries = []
        for suffix_count in (0, 1, 2):
            full_length = prompt_length + suffix_count
            if full_length != EXPECTED_PREFIX_LENGTHS[suffix_count]:
                raise RuntimeError("fixed teacher prefix length changed")
            RUN_PROGRESS["stage"] = f"teacher_forcing_prefix_{full_length}"
            target_prefix = torch.tensor(
                frozen_target.token_ids[:suffix_count], dtype=batch["input_ids"].dtype,
                device=batch["input_ids"].device,
            ).view(1, -1)
            full_ids = torch.cat((batch["input_ids"], target_prefix), dim=1)
            positions, mm_types, attention, delta = _position_tensor(
                model, batch, full_ids, suffix_count
            )
            if not torch.equal(positions[:, :, :prompt_length], native_prompt_positions):
                raise RuntimeError("teacher prefix prompt positions differ from native generation prefill")
            expected_suffix = torch.tensor(
                [[[55 + i for i in range(suffix_count)]] for _axis in range(3)],
                dtype=positions.dtype, device=positions.device,
            )
            if not torch.equal(positions[:, :, prompt_length:], expected_suffix):
                raise RuntimeError("teacher suffix native positions differ from fixed [55,56] progression")
            teacher_batch = dict(batch)
            teacher_batch["input_ids"] = full_ids
            teacher_batch["mm_token_type_ids"] = mm_types
            teacher_batch["attention_mask"] = attention
            model.model.rope_deltas = None
            trace = NativeGenerationTrace(model.model.language_model)
            capture_prediction_rows = tuple(
                range(prompt_length - 1, min(full_length, prompt_length + 1))
            )
            projection_trace = LMHeadInputTrace(
                model.lm_head, selected_rows=capture_prediction_rows
            )
            forward_config_trace = OuterForwardConfigTrace(model)
            try:
                with torch.inference_mode():
                    teacher_output = model(
                        **teacher_batch, position_ids=positions, use_cache=False, return_dict=True
                    )
                    torch.cuda.synchronize()
            finally:
                trace.close()
                projection_trace.close()
                forward_config_trace.close()
                RUN_PROGRESS["completed_model_calls"] += len(trace.calls)
            trace_summary = validate_teacher_forcing_trace(trace, positions, full_length)
            if len(forward_config_trace.calls) != 1:
                raise RuntimeError("teacher-forced call did not invoke outer model once")
            teacher_forward_configs.append({
                "prefix_length": full_length,
                **forward_config_trace.calls[0],
            })
            if len(projection_trace.calls) != 1:
                raise RuntimeError("teacher-forced call did not invoke lm_head exactly once")
            projection_input = projection_trace.calls[0]
            if projection_input["shape"][1] != full_length:
                raise RuntimeError("teacher-forced lm_head did not receive all sequence rows")
            if projection_input["captured_rows"] != capture_prediction_rows:
                raise RuntimeError("teacher-forced lm_head hidden-row capture differs from fixed eligible rows")
            teacher_projection_inputs[full_length] = projection_input
            if teacher_output.logits.shape[1] != full_length:
                raise RuntimeError("teacher-forced logits do not cover the fixed prefix")
            # Capture the prediction row for each target token already
            # generated plus the next row at the end of this prefix. The P-only
            # forward must contribute row P-1 even though its target suffix is
            # empty; likewise P+1 contributes rows P-1 and P.
            prediction_count = min(suffix_count + 1, GENERATED_TOKENS)
            for token_index in range(prediction_count):
                row_index = prompt_length - 1 + token_index
                if row_index >= full_length:
                    raise RuntimeError("fixed next-token prediction row is outside prefix logits")
                values = teacher_output.logits[0, row_index].detach().to(
                    device="cpu", dtype=torch.float64
                )
                if not bool(torch.isfinite(values).all()):
                    raise RuntimeError("selected teacher-forced full-vocabulary row is nonfinite")
                teacher_rows[f"full_{full_length}_row_{row_index}"] = values.clone()
            if suffix_count == 2:
                library_score = score_frozen_target(teacher_output.logits, frozen_target)
                for token_index, token_id in enumerate(frozen_target.token_ids):
                    row_index = prompt_length - 1 + token_index
                    torch_logprob = float(
                        teacher_rows[f"full_{full_length}_row_{row_index}"].log_softmax(0)[token_id].item()
                    )
                    scorer_crosscheck_max_abs_error = max(
                        scorer_crosscheck_max_abs_error,
                        abs(torch_logprob - library_score["token_logprobs_nats"][token_index]),
                    )
            teacher_trace_summaries.append({"prefix_length": full_length, **trace_summary})
            rope_delta_summaries.append({
                "prefix_length": full_length,
                "delta": delta.detach().cpu().tolist(),
            })
            RUN_PROGRESS["completed_teacher_prefixes"] += 1
            RUN_PROGRESS["partial_evidence"].update({
                "completed_teacher_prefixes": RUN_PROGRESS["completed_teacher_prefixes"],
                "completed_model_calls": RUN_PROGRESS["completed_model_calls"],
                "teacher_trace_summaries": list(teacher_trace_summaries),
                "scorer_crosscheck_max_abs_error": scorer_crosscheck_max_abs_error,
            })
            del teacher_output, target_prefix, full_ids, teacher_batch, positions, mm_types, attention
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        vectors = {**generated_by_name, **teacher_rows}
        plan = build_comparison_plan(prompt_length)
        required_teacher_rows = {
            name
            for pair in plan
            for name in (pair["left"], pair["right"])
            if name.startswith("full_")
        }
        if not required_teacher_rows.issubset(teacher_rows):
            raise RuntimeError("fixed full-sequence comparison rows were not captured")
        comparisons = []
        for pair in plan:
            comparisons.append(summarize_pair(
                pair,
                vectors[pair["left"]],
                vectors[pair["right"]],
                frozen_target.token_ids[pair["token_index"]],
            ))
        hidden_vectors = {}
        for index, record in enumerate(generation_projection_inputs):
            if record["captured_rows"] != (record["shape"][1] - 1,) or record["hidden_cpu"].shape[1] != 1:
                raise RuntimeError("generation lm_head did not capture exactly its selected final hidden row")
            hidden_vectors[f"generation_token_{index}"] = record["hidden_cpu"][0, 0]
        for spec in projection_comparison_plan(prompt_length):
            record = teacher_projection_inputs[spec["full_length"]]
            hidden_vectors[spec["full_name"]] = _captured_hidden_row(record, spec["row"])[0, 0]
        hidden_comparisons = [
            summarize_hidden_pair(
                pair, hidden_vectors[pair["left"]], hidden_vectors[pair["right"]]
            )
            for pair in plan
        ]
        hidden_comparison_crosscheck_max = max(
            item["torch64_numpy_max_abs_error"] for item in hidden_comparisons
        )
        projection_inputs_by_name = {
            f"generation_token_{index}": record["hidden_cpu"]
            for index, record in enumerate(generation_projection_inputs)
        }
        projection_input_shapes = [
            {"condition": f"generation_token_{index}", "shape": list(record["shape"]),
             "captured_rows": list(record["captured_rows"]), "dtype": record["dtype"]}
            for index, record in enumerate(generation_projection_inputs)
        ]
        for full_length, record in teacher_projection_inputs.items():
            projection_input_shapes.append({
                "condition": f"full_{full_length}",
                "shape": list(record["shape"]),
                "captured_rows": list(record["captured_rows"]),
                "dtype": record["dtype"],
            })

        reprojection_trace = LMHeadInputTrace(model.lm_head, selected_rows=(0,))
        reprojected = {}
        projection_specs = projection_comparison_plan(prompt_length)
        lm_head_device = next(model.lm_head.parameters()).device
        try:
            with torch.inference_mode():
                for spec in projection_specs:
                    full_record = teacher_projection_inputs[spec["full_length"]]
                    hidden = _captured_hidden_row(full_record, spec["row"])
                    expected_name = spec["full_name"]
                    if expected_name in projection_inputs_by_name:
                        raise RuntimeError("projection comparison name collision")
                    projected = model.lm_head(hidden.to(device=lm_head_device))
                    if tuple(projected.shape[:2]) != (1, 1):
                        raise RuntimeError("one-row lm_head projection returned unexpected shape")
                    vector = projected[0, 0].detach().to(device="cpu", dtype=torch.float64)
                    if not bool(torch.isfinite(vector).all()):
                        raise RuntimeError("one-row lm_head projection returned nonfinite logits")
                    reprojected[expected_name] = vector
                    projection_inputs_by_name[expected_name] = hidden
        finally:
            reprojection_trace.close()
        if len(reprojection_trace.calls) != len(projection_specs):
            raise RuntimeError("single-row lm_head reprojection call count differs from fixed five")
        for spec, actual in zip(projection_specs, reprojection_trace.calls):
            expected_hidden = projection_inputs_by_name[spec["full_name"]]
            if actual["shape"][:2] != (1, 1) or not torch.equal(actual["hidden_cpu"], expected_hidden):
                raise RuntimeError("single-row projection did not receive the selected unchanged hidden row")
            projection_input_shapes.append({
                "condition": f"reproject_{spec['full_name']}",
                "shape": list(actual["shape"]),
                "captured_rows": list(actual["captured_rows"]),
                "dtype": actual["dtype"],
            })
        for spec in projection_specs:
            full_logits = teacher_rows[spec["full_name"]]
            one_row_logits = reprojected[spec["full_name"]]
            generation_logits = generated_by_name[spec["generation_name"]]
            target_id = frozen_target.token_ids[spec["row"] - (prompt_length - 1)]
            projection_metrics = (
                ("full_output_vs_one_row_reprojection", full_logits, one_row_logits),
                ("one_row_reprojection_vs_generation", one_row_logits, generation_logits),
            )
            for kind, left, right in projection_metrics:
                comparison = summarize_pair({
                    "left": spec["full_name"] if kind.startswith("full_") else f"reproject_{spec['full_name']}",
                    "right": f"reproject_{spec['full_name']}" if kind.startswith("full_") else spec["generation_name"],
                    "comparison_type": kind,
                    "token_index": spec["row"] - (prompt_length - 1),
                    "prediction_row": spec["row"],
                }, left, right, target_id)
                comparison.update({
                    "full_prefix_length": spec["full_length"],
                    "lm_head_projection_shape_diagnostic": True,
                })
                comparisons.append(comparison)
        projection_call_count = (
            len(generation_projection_inputs)
            + len(teacher_projection_inputs)
            + len(reprojection_trace.calls)
        )
        if projection_call_count != 10:
            raise RuntimeError("fixed lm_head invocation count differs from ten")
        comparison_crosscheck_max = max(
            item["torch64_numpy_python_max_abs_error"] for item in comparisons
        )
        overall_crosscheck = max(
            comparison_crosscheck_max,
            hidden_comparison_crosscheck_max,
            scorer_crosscheck_max_abs_error,
        )
        crosscheck_ok = overall_crosscheck <= NUMERIC_CROSSCHECK_TOLERANCE
        peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
        RUN_PROGRESS["partial_evidence"].update({
            "comparison_pair_count": len(comparisons),
            "comparisons": comparisons,
            "hidden_row_comparison_count": len(hidden_comparisons),
            "hidden_row_comparisons": hidden_comparisons,
            "hidden_row_numeric_crosscheck_max_abs_error": hidden_comparison_crosscheck_max,
            "lm_head_projection_input_shapes": projection_input_shapes,
            "lm_head_projection_call_count": projection_call_count,
            "numpy_torch_python_crosscheck_max_abs_error": comparison_crosscheck_max,
            "existing_scorer_crosscheck_max_abs_error": scorer_crosscheck_max_abs_error,
            "peak_allocated_gib": peak_allocated_gib,
        })
        if RUN_PROGRESS["completed_model_calls"] != 5 or RUN_PROGRESS["completed_teacher_prefixes"] != 3:
            raise RuntimeError("fixed diagnostic did not complete exactly five calls and three prefixes")

        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
        from transformers.generation import utils as generation_utils
        source_hashes.update({
            "installed_qwen_model": sha256(Path(__import__("inspect").getfile(modeling_qwen2_5_vl)).resolve()),
            "installed_generation_utils": sha256(Path(__import__("inspect").getfile(generation_utils)).resolve()),
        })
        record = {
            "schema_version": 1,
            "status": "diagnostic_complete" if crosscheck_ok else "numerical_crosscheck_failed",
            "purpose": "fixed_cached_vs_full_sequence_numerical_path_diagnostic",
            "started_utc": started,
            "elapsed_seconds": time.monotonic() - timer,
            "elapsed_scope": "runner workload from frame extraction through scalar summary construction; excludes final filesystem serialization, imports and initial preflight",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "source_sha256": source_hashes,
            "baseline_result_sha256": sha256(baseline_path),
            "baseline_source_commit": baseline.get("source_commit"),
            "baseline_source_sha256": baseline.get("source_sha256"),
            "baseline_target_sha256": EXPECTED_TARGET_SHA256,
            "checkpoint_revision": REVISION,
            **config_hashes,
            "media_sha256": MEDIA_SHA256,
            "seed": SEED,
            "quantization": "4-bit",
            "compute_dtype": "bfloat16",
            "attention_implementation": "eager",
            "prompt_policy": "same fixed generic video description prompt as baseline; text not exported",
            "prompt_tokens": prompt_length,
            "prompt_ids_sha256": prompt_ids_sha256,
            "requested_frames": FRAME_COUNT,
            "effective_frames": len(timestamps),
            "sample_timestamps_seconds": [float(value) for value in timestamps],
            "sampled_fps": float(fps),
            "video_grid_thw": list(grid),
            "frozen_target_sha256": frozen_target.target_sha256,
            "frozen_target_token_count": len(frozen_target.token_ids),
            "generation_trace": generation_trace_summary,
            "generation_forward_logits_to_keep": generation_forward_configs,
            "teacher_prefix_lengths": list(EXPECTED_PREFIX_LENGTHS),
            "teacher_trace_summaries": teacher_trace_summaries,
            "teacher_forward_logits_to_keep": teacher_forward_configs,
            "teacher_rope_delta_summaries": rope_delta_summaries,
            "requested_model_calls": 5,
            "completed_model_calls": RUN_PROGRESS["completed_model_calls"],
            "completed_teacher_prefixes": RUN_PROGRESS["completed_teacher_prefixes"],
            "requested_comparison_pairs": 19,
            "effective_comparison_pairs": len(comparisons),
            "sequence_length_comparison_pairs": len(plan),
            "projection_shape_comparison_pairs": len(comparisons) - len(plan),
            "comparisons": comparisons,
            "requested_hidden_row_comparisons": 9,
            "effective_hidden_row_comparisons": len(hidden_comparisons),
            "hidden_row_comparisons": hidden_comparisons,
            "hidden_row_numeric_crosscheck_max_abs_error": hidden_comparison_crosscheck_max,
            "lm_head_projection_input_shapes": projection_input_shapes,
            "requested_lm_head_projection_calls": 10,
            "effective_lm_head_projection_calls": projection_call_count,
            "requested_single_row_reprojections": 5,
            "effective_single_row_reprojections": len(reprojection_trace.calls),
            "fixed_numeric_crosscheck_tolerance": NUMERIC_CROSSCHECK_TOLERANCE,
            "numeric_crosscheck_is_outcome_gate": False,
            "numpy_torch_python_crosscheck_max_abs_error": comparison_crosscheck_max,
            "existing_frozen_scorer_crosscheck_max_abs_error": scorer_crosscheck_max_abs_error,
            "numeric_crosscheck_passed": crosscheck_ok,
            "checkpoint_byte_audit": {
                "artifact_path": CHECKPOINT_AUDIT_PATH.as_posix(),
                "artifact_sha256": checkpoint_audit_sha256,
                "source_commit": checkpoint_audit["source_commit"],
                "status": checkpoint_audit["byte_authentication_status"],
                "current_local_bytes_only": True,
                "historical_gpu_runs_byte_verified": False,
                "continuous_authentication_of_later_load": False,
            },
            "checkpoint_shards_before_model_load": checkpoint_shards_before_model_load,
            "checkpoint_shards_after_model_load": checkpoint_shards_after_model_load,
            "checkpoint_provenance_limitations": [
                "The byte audit authenticates the local checkpoint bytes during its audit time, not historical GPU runs.",
                "This run hashes the exact five shard paths immediately before and after model load; it detects a changed cache state across loading but cannot prevent or observe a mutation between individual shard reads.",
            ],
            "gpu_peak_tracking_started_before_model_load": True,
            "gpu_peak_allocated_gib": peak_allocated_gib,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_free_before_gib": free_bytes / 1024**3,
            "gpu_total_gib": total_bytes / 1024**3,
            "cuda_version": torch.version.cuda,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "opencv-python-headless", "numpy")
            },
            "limitations": [
                "One fixed NOAA video, four sampled frames, one generic prompt, and the exact two-token GPU01 target.",
                "Nine scalar comparisons describe numerical path sensitivity; they have no model-outcome threshold.",
                "No QA, intervention, claim semantics, faithfulness, or human-annotation validity is measured.",
                "KL is exploratory descriptive characterization; it does not identify a causal source of numerical drift.",
                "No raw token IDs, logits, text, absolute target scores or arrays are exported.",
            ],
            "privacy": "Aggregate numeric diagnostics only; no absolute paths, prompt/output text, raw token IDs, arrays, logits or absolute scores.",
        }
        record["failures"] = 0 if crosscheck_ok else 1
        _write_result(output, record)
        RUN_PROGRESS["stage"] = "write_result"
        print(json.dumps({
            "status": record["status"], "output": str(output.relative_to(repo)),
            "calls": RUN_PROGRESS["completed_model_calls"],
            "prefixes": RUN_PROGRESS["completed_teacher_prefixes"],
            "pairs": len(comparisons),
            "max_numeric_crosscheck_error": round(overall_crosscheck, 14),
            "peak_allocated_gib": round(peak_allocated_gib, 3),
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
        result = run_gpu_diagnostic(output)
        if result["status"] != "diagnostic_complete":
            raise SystemExit(1)
    except Exception as exc:
        evidence = dict(RUN_PROGRESS.get("partial_evidence", {}))
        evidence.update(_failure_provenance(repo))
        failure = {
            "schema_version": 1,
            "status": "failed",
            "purpose": "fixed_cached_vs_full_sequence_numerical_path_diagnostic",
            "stage": RUN_PROGRESS["stage"],
            "failure_started_utc": cli_started_utc,
            "elapsed_seconds": time.monotonic() - cli_timer,
            "elapsed_scope": "CLI after argument validation; excludes Python/module startup",
            "expected_target_sha256": EXPECTED_TARGET_SHA256,
            "requested_frames": FRAME_COUNT,
            "effective_frames": RUN_PROGRESS["effective_frames"],
            "requested_model_calls": 5,
            "completed_model_calls": RUN_PROGRESS["completed_model_calls"],
            "completed_teacher_prefixes": RUN_PROGRESS["completed_teacher_prefixes"],
            "frozen_target_sha256": RUN_PROGRESS["frozen_target_sha256"],
            "failures": 1,
            "failure_type": type(exc).__name__,
            "exception_message_exported": False,
            "evidence": evidence,
        }
        _write_result(output, failure)
        print(json.dumps({"status": "failed", "stage": failure["stage"],
                          "calls": failure["completed_model_calls"],
                          "prefixes": failure["completed_teacher_prefixes"],
                          "output": str(output.relative_to(repo))}, sort_keys=True))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
