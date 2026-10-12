"""Fixed synthetic BNB row-count diagnostic (not a Qwen/model experiment).

The script compares one shared output row for M=1,584,585,586 under four
fixed linear implementations. Only aggregate scalar differences and hashes
are written. It contains no causal mask and cannot establish that a real
model's row-count drift has the same cause.

Usage::

    .venv/bin/python scripts/run_bnb_row_count_gpu_diagnostic.py \
      --output results/bnb_row_count_gpu_diagnostic_YYYYMMDD_NN.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time


SEED = 20261012
WIDTH = 3584
BASE_ROWS = 584
COMMON_ROW = 583
ROW_COUNTS = (1, 584, 585, 586)
CONDITIONS = (
    "original_bf16_f_linear",
    "original_fp32_f_linear",
    "bnb_fp4_bf16_compute",
    "bnb_dequantized_bf16_f_linear",
)
REPEAT_M = 584
EXPECTED_FORWARD_CALLS = len(ROW_COUNTS) * len(CONDITIONS) + len(CONDITIONS)
EXPECTED_COMPARISONS = 32


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(tensor) -> str:
    """Hash a detached contiguous tensor's exact bytes without exporting it."""
    import torch

    value = tensor.detach().contiguous().cpu()
    return sha256_bytes(value.view(torch.uint8).numpy().tobytes())


def quant_state_witness(layer, quantized_weight):
    """Capture stable quantization metadata and tensor digests for before/after checks."""
    state = quantized_weight.quant_state
    if state is None:
        raise RuntimeError("bnb_quantization_state_missing")
    tensor_fields = {}
    for name in ("absmax", "code", "offset", "state2", "absmax2", "code2"):
        value = getattr(state, name, None)
        if value is not None:
            tensor_fields[name] = {
                "shape": list(value.shape), "dtype": str(value.dtype),
                "device": value.device.type, "contiguous": bool(value.is_contiguous()),
                "sha256": tensor_sha256(value),
            }
    return {
        "quant_type": str(state.quant_type), "dtype": str(state.dtype),
        "blocksize": int(state.blocksize), "nested": bool(state.nested),
        "compute_dtype": str(layer.compute_dtype),
        "weight_parameter_dtype": str(layer.weight.dtype),
        "weight_parameter_class": type(layer.weight).__name__,
        "weight_parameter_shape": list(layer.weight.shape),
        "packed_weight_sha256": tensor_sha256(quantized_weight.data),
        "state_tensors": tensor_fields,
    }


def build_fixed_inputs(width: int = WIDTH, seed: int = SEED):
    """Create deterministic CPU BF16 W/X; no global RNG state is touched."""
    import torch

    if type(width) is not int or width < 1:
        raise ValueError("width must be a positive strict integer")
    if type(seed) is not int:
        raise ValueError("seed must be a strict integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    weight = (torch.randn((width, width), generator=generator, dtype=torch.float32, device="cpu") / math.sqrt(width)).to(torch.bfloat16)
    inputs = torch.randn((BASE_ROWS, width), generator=generator, dtype=torch.float32, device="cpu").to(torch.bfloat16)
    # Freeze the appended rows as duplicates, rather than drawing additional
    # random values after choosing the observed common row.
    inputs_extended = torch.cat((inputs, inputs[0:1], inputs[0:1]), dim=0)
    return weight, inputs, inputs_extended


def make_row_inputs(inputs, inputs_extended, counts=ROW_COUNTS):
    """Return strict fixed M inputs and their common-row indices."""
    import torch

    result = {}
    for count in counts:
        if type(count) is not int or count not in ROW_COUNTS:
            raise ValueError("unsupported row count")
        if count == 1:
            value, selected_index = inputs[COMMON_ROW:COMMON_ROW + 1], 0
        else:
            value, selected_index = inputs_extended[:count], COMMON_ROW
        result[count] = (value, selected_index)
    return result


def build_comparison_plan(counts=ROW_COUNTS):
    """Six row-count pairs and one repeat per condition, plus four BNB/dequant pairs."""
    if tuple(counts) != ROW_COUNTS:
        raise ValueError("comparison plan is fixed to M=(1,584,585,586)")
    pairs = []
    for condition in CONDITIONS:
        for left_i, left_m in enumerate(ROW_COUNTS):
            for right_m in ROW_COUNTS[left_i + 1:]:
                pairs.append({
                    "comparison_type": "row_count_within_condition",
                    "condition": condition,
                    "left_m": left_m,
                    "right_m": right_m,
                    "selected_row": "common_row_583" if left_m != 1 and right_m != 1 else "same_source_row_583",
                })
        pairs.append({
            "comparison_type": "same_shape_repeat",
            "condition": condition,
            "m": REPEAT_M,
        })
    for count in ROW_COUNTS:
        pairs.append({
            "comparison_type": "bnb_vs_dequantized_same_quantized_weight",
            "condition": "bnb_fp4_bf16_compute_vs_bnb_dequantized_bf16_f_linear",
            "m": count,
        })
    if len(pairs) != EXPECTED_COMPARISONS:
        raise AssertionError("fixed plan must contain 32 comparisons")
    return tuple(pairs)


def scalar_metrics(left, right):
    """Independent CPU float64 aggregate comparison; never returns arrays."""
    import numpy as np
    import torch

    x = torch.as_tensor(left).detach().to(device="cpu", dtype=torch.float64)
    y = torch.as_tensor(right).detach().to(device="cpu", dtype=torch.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.numel() == 0:
        raise ValueError("paired outputs must have identical nonempty one-dimensional shape")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("paired outputs must be finite")
    difference = x - y
    if not bool(torch.isfinite(difference).all()):
        raise ValueError("paired output difference overflowed float64")
    denominator = float(torch.linalg.vector_norm(x).item())
    numerator = float(torch.linalg.vector_norm(difference).item())
    if not math.isfinite(denominator) or not math.isfinite(numerator):
        raise ValueError("paired output norm overflowed float64")
    if denominator > 0:
        relative_l2, relative_status = numerator / denominator, "defined"
    elif numerator == 0:
        relative_l2, relative_status = None, "zero_reference_equal"
    else:
        relative_l2, relative_status = None, "zero_reference_different"
    metrics = {
        "raw_max_abs": float(difference.abs().max().item()),
        "raw_mean_abs": float(difference.abs().mean().item()),
        "raw_rms": float(torch.sqrt(torch.mean(difference.square())).item()),
        "relative_l2_to_left": relative_l2,
        "relative_l2_status": relative_status,
    }
    if any(value is not None and isinstance(value, (int, float)) and not math.isfinite(value)
           for value in metrics.values()):
        raise ValueError("paired output aggregate overflowed float64")
    return metrics


def numpy_scalar_metrics(left, right):
    """Independent NumPy float64 implementation used as arithmetic cross-check."""
    import numpy as np

    def as_float64_cpu(value):
        if hasattr(value, "detach"):
            import torch
            # NumPy does not reliably support torch.bfloat16; convert on CPU
            # before exposing the values to NumPy.
            value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
        return np.asarray(value, dtype=np.float64)

    x = as_float64_cpu(left)
    y = as_float64_cpu(right)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("paired outputs must be same-shape nonempty 1d vectors and finite")
    delta = x - y
    if not np.isfinite(delta).all():
        raise ValueError("paired output difference overflowed float64")
    denom = float(np.linalg.norm(x))
    numerator = float(np.linalg.norm(delta))
    if not math.isfinite(denom) or not math.isfinite(numerator):
        raise ValueError("paired output norm overflowed float64")
    if denom > 0:
        relative_l2, relative_status = numerator / denom, "defined"
    elif numerator == 0:
        relative_l2, relative_status = None, "zero_reference_equal"
    else:
        relative_l2, relative_status = None, "zero_reference_different"
    metrics = {
        "raw_max_abs": float(np.max(np.abs(delta))),
        "raw_mean_abs": float(np.mean(np.abs(delta))),
        "raw_rms": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2_to_left": relative_l2,
        "relative_l2_status": relative_status,
    }
    if any(value is not None and isinstance(value, (int, float)) and not math.isfinite(value)
           for value in metrics.values()):
        raise ValueError("paired output aggregate overflowed float64")
    return metrics


def summarize_pair(left, right, tolerance=1e-10):
    torch_values = scalar_metrics(left, right)
    numpy_values = numpy_scalar_metrics(left, right)
    errors = []
    for key in torch_values:
        left_value, right_value = torch_values[key], numpy_values[key]
        if left_value is None or right_value is None:
            if left_value is not None or right_value is not None:
                raise ValueError("torch and numpy relative norm definedness disagree")
        elif isinstance(left_value, (bool, str)):
            if left_value != right_value:
                raise ValueError("torch and numpy metric status disagrees")
        else:
            errors.append(abs(left_value - right_value))
    max_error = max(errors, default=0.0)
    return {**torch_values, "torch64_numpy_max_abs_error": max_error,
            "numeric_crosscheck_within_1e_10": max_error <= tolerance}


def source_dispatch_prediction(m, n=WIDTH, k=WIDTH, dtype_name="bfloat16", sm=(12, 0)):
    """Report the installed Python-dispatch expectation, not observed kernel use."""
    if type(m) is not int or m <= 0:
        raise ValueError("M must be a positive strict integer")
    if m > 1536:
        route = "dequantize_then_f_linear_by_custom_max"
    elif k % 64:
        route = "dequantize_then_f_linear_by_block_alignment"
    elif m <= 4:
        route = "custom_kernel_candidate_small_m_floor"
    elif m > 512 and n * k >= 4 * 1024 * 1024:
        route = "dequantize_then_f_linear_by_non_small_weight_m_gt_512"
    else:
        route = "backend_arch_shape_predicate_needed"
    return {
        "m": m,
        "route_prediction": route,
        "actual_kernel_observed": False,
        "source_python_dispatch_version": "bitsandbytes 0.50.2 installed CUDA ops.py",
        "dtype_name": dtype_name,
        "sm": list(sm),
    }


def installed_source_hashes():
    """Hash installed BNB Python dispatch sources without exposing local paths."""
    import bitsandbytes
    root = Path(bitsandbytes.__file__).resolve().parent
    files = {
        "nn_modules.py": root / "nn" / "modules.py",
        "functional.py": root / "functional.py",
        "cuda_ops.py": root / "backends" / "cuda" / "ops.py",
        "autograd_functions.py": root / "autograd" / "_functions.py",
    }
    return {name: sha256_bytes(path.read_bytes()) for name, path in files.items() if path.is_file()}


def native_library_summary():
    """Return basename and content hash only; binary path is intentionally omitted."""
    try:
        from bitsandbytes.cextension import lib

        native = getattr(lib, "_lib", None)
        native_path = getattr(native, "_name", None)
        if not native_path:
            return {"available": False, "failure_code": "native_library_path_unavailable"}
        path = Path(native_path)
        if not path.is_file():
            return {"available": False, "failure_code": "native_library_file_unavailable"}
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"available": True, "basename": path.name, "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest()}
    except Exception:
        return {"available": False, "failure_code": "native_library_hash_unavailable"}


class BNBDispatchTrace:
    """Witness Python fallback/custom dispatch and fallback dequantized weights.

    This does not identify the inner compiled CUDA kernel implementation.
    """

    def __init__(self, ops_module, expected_dequantized):
        self.ops = ops_module
        self.expected = expected_dequantized
        self.active_m = None
        self.active_label = None
        self.python_routes = []
        self.fallback_weights = {}
        self._originals = {}

    def __enter__(self):
        names = ("_gemm_4bit_kernel_impl", "_dequant_linear_fallback", "_dequantize_4bit_impl")
        self._originals = {name: getattr(self.ops, name) for name in names}
        original_kernel = self._originals["_gemm_4bit_kernel_impl"]
        original_fallback = self._originals["_dequant_linear_fallback"]
        original_dequant = self._originals["_dequantize_4bit_impl"]
        trace = self

        def kernel(*args, **kwargs):
            a = args[0]
            trace.python_routes.append({"m": int(a.numel() // a.shape[-1]), "python_path": "custom_kernel_impl"})
            return original_kernel(*args, **kwargs)

        def fallback(*args, **kwargs):
            a = args[0]
            trace.python_routes.append({"m": int(a.numel() // a.shape[-1]), "python_path": "dequantize_then_f_linear"})
            return original_fallback(*args, **kwargs)

        def dequant(*args, **kwargs):
            result = original_dequant(*args, **kwargs)
            out = kwargs.get("out")
            if out is None and len(args) >= 5:
                out = args[4]
            if out is not None and trace.active_m is not None:
                expected = trace.expected
                if out.shape != expected.shape or out.dtype != expected.dtype or out.device != expected.device:
                    raise RuntimeError("fallback_dequant_metadata_mismatch")
                if not out.is_contiguous() or not torch_equal(out, expected):
                    raise RuntimeError("fallback_dequant_value_mismatch")
                trace.fallback_weights[trace.active_label] = {
                    "shape_out_in": list(out.shape), "dtype": str(out.dtype),
                    "contiguous": bool(out.is_contiguous()),
                    "sha256": tensor_sha256(out), "bitwise_equal_public_dequant": True,
                }
            return result

        self.ops._gemm_4bit_kernel_impl = kernel
        self.ops._dequant_linear_fallback = fallback
        self.ops._dequantize_4bit_impl = dequant
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, value in self._originals.items():
            setattr(self.ops, name, value)
        return False


def torch_equal(left, right):
    import torch
    return bool(torch.equal(left, right))


def exclusive_result_path(repo: Path, output_arg: str) -> Path:
    requested = Path(output_arg)
    if not requested.is_absolute():
        requested = repo / requested
    requested = requested.resolve()
    if requested.parent != (repo / "results").resolve() or requested.suffix != ".json":
        raise ValueError("output must be a direct results/*.json path")
    return requested


def reserve_exclusive_private_output(path: Path) -> int:
    """Create and hold a private output before any expensive/GPU work begins."""
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)


def write_reserved_private_json(fd: int, record) -> None:
    payload = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_exclusive_private_json(path: Path, record) -> None:
    write_reserved_private_json(reserve_exclusive_private_output(path), record)


def _git_commit(repo: Path):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def run_diagnostic(device="cuda:0", progress=None):
    """Execute fixed 20 calls. Caller writes returned safe summary exclusively."""
    import torch
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_functional
    import bitsandbytes.backends.cuda.ops as bnb_cuda_ops

    bnb_version = importlib.metadata.version("bitsandbytes")
    if bnb_version != "0.50.2":
        raise RuntimeError("unsupported_bitsandbytes_version")

    progress = progress if progress is not None else {
        "attempted_forward_calls": 0, "completed_forward_calls": 0, "completed_comparisons": 0,
    }
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("device must be CUDA")
    torch.cuda.synchronize(target)
    free_bytes, total_bytes = torch.cuda.mem_get_info(target)
    if free_bytes < 2 * 1024**3:
        raise RuntimeError("insufficient_free_vram")
    torch.cuda.reset_peak_memory_stats(target)

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    old_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        math_settings = {
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_bf16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
            "split_k_setting": "not_exposed_by_fixed_torch_linear_api",
        }

        weight_bf16, inputs_bf16, extended = build_fixed_inputs()
        weight_sha = tensor_sha256(weight_bf16)
        input_sha = tensor_sha256(inputs_bf16)
        extended_sha = tensor_sha256(extended)
        rows = make_row_inputs(inputs_bf16, extended)
        device_weight_bf16 = weight_bf16.to(target)
        device_weight_fp32 = weight_bf16.to(device=target, dtype=torch.float32)

        rng_state = torch.random.get_rng_state()
        try:
            layer = bnb.nn.Linear4bit(
                WIDTH, WIDTH, bias=False, compute_dtype=torch.bfloat16,
                compress_statistics=False, quant_type="fp4",
            )
        finally:
            torch.random.set_rng_state(rng_state)
        # Replace the fresh FP32 initialization with the exact fixed BF16 W.
        # Params4bit retains this source dtype until CUDA conversion quantizes it.
        layer.weight = bnb.nn.Params4bit(
            weight_bf16.clone(), requires_grad=False, compress_statistics=False,
            quant_type="fp4", module=layer,
        )
        if layer.weight.dtype != torch.bfloat16:
            raise RuntimeError("bnb_source_weight_dtype_not_bf16")
        layer = layer.to(target)
        quantized_weight = layer.weight
        quant_state = quantized_weight.quant_state
        if quant_state is None:
            raise RuntimeError("bnb_quantization_state_missing")
        if (quant_state.dtype != torch.bfloat16 or int(quant_state.blocksize) != 64
                or bool(quant_state.nested) or str(quant_state.quant_type) != "fp4"):
            raise RuntimeError("bnb_quant_state_unexpected")
        dequant_bf16 = bnb_functional.dequantize_4bit(
            quantized_weight.data, quant_state=quant_state, quant_type="fp4"
        ).to(dtype=torch.bfloat16)
        if (tuple(dequant_bf16.shape) != (WIDTH, WIDTH)
                or dequant_bf16.dtype != torch.bfloat16
                or not dequant_bf16.is_contiguous()):
            raise RuntimeError("bnb_public_dequant_metadata_unexpected")
        dequant_sha = tensor_sha256(dequant_bf16)
        packed_sha_before = tensor_sha256(quantized_weight.data)
        quant_state_before = quant_state_witness(layer, quantized_weight)

        condition_outputs = {name: {} for name in CONDITIONS}
        dispatch_trace = BNBDispatchTrace(bnb_cuda_ops, dequant_bf16)
        with dispatch_trace, torch.inference_mode():
            for condition in CONDITIONS:
                calls = [(count, rows[count][0], rows[count][1], str(count)) for count in ROW_COUNTS]
                calls.insert(2, (REPEAT_M, rows[REPEAT_M][0], rows[REPEAT_M][1], "584_repeat"))
                for count, cpu_input, selected_idx, label in calls:
                    x_bf16 = cpu_input.to(target)
                    progress["attempted_forward_calls"] = progress.get("attempted_forward_calls", 0) + 1
                    if condition == "original_bf16_f_linear":
                        output = torch.nn.functional.linear(x_bf16, device_weight_bf16)
                    elif condition == "original_fp32_f_linear":
                        output = torch.nn.functional.linear(x_bf16.float(), device_weight_fp32)
                    elif condition == "bnb_fp4_bf16_compute":
                        dispatch_trace.active_m = count
                        dispatch_trace.active_label = label
                        output = layer(x_bf16)
                        dispatch_trace.active_m = None
                        dispatch_trace.active_label = None
                    else:
                        output = torch.nn.functional.linear(x_bf16, dequant_bf16)
                    if output.shape != (count, WIDTH):
                        raise RuntimeError("unexpected_output_shape")
                    if not bool(torch.isfinite(output).all()):
                        raise RuntimeError("nonfinite_output")
                    condition_outputs[condition][label] = output[selected_idx].detach().cpu().clone()
                    progress["completed_forward_calls"] += 1
                    del output, x_bf16

        expected_routes = [
            {"m": 1, "python_path": "custom_kernel_impl"},
            {"m": 584, "python_path": "dequantize_then_f_linear"},
            {"m": 584, "python_path": "dequantize_then_f_linear"},
            {"m": 585, "python_path": "dequantize_then_f_linear"},
            {"m": 586, "python_path": "dequantize_then_f_linear"},
        ]
        if dispatch_trace.python_routes != expected_routes:
            raise RuntimeError("bnb_dispatch_path_unexpected")
        if set(dispatch_trace.fallback_weights) != {"584", "584_repeat", "585", "586"}:
            raise RuntimeError("fallback_dequant_capture_incomplete")
        quant_state_after = quant_state_witness(layer, quantized_weight)
        if quant_state_after != quant_state_before:
            raise RuntimeError("bnb_quantized_state_changed_during_forwards")
        packed_sha_after = quant_state_after["packed_weight_sha256"]

        comparisons = []
        for condition in CONDITIONS:
            for i, left_m in enumerate(ROW_COUNTS):
                for right_m in ROW_COUNTS[i + 1:]:
                    pair = {
                        "comparison_type": "row_count_within_condition",
                        "condition": condition,
                        "left_m": left_m,
                        "right_m": right_m,
                    }
                    comparisons.append({**pair, **summarize_pair(
                        condition_outputs[condition][str(left_m)], condition_outputs[condition][str(right_m)]
                    )})
                    progress["completed_comparisons"] += 1
            pair = {"comparison_type": "same_shape_repeat", "condition": condition, "m": REPEAT_M}
            comparisons.append({**pair, **summarize_pair(
                condition_outputs[condition]["584"], condition_outputs[condition]["584_repeat"]
            )})
            progress["completed_comparisons"] += 1
        for count in ROW_COUNTS:
            pair = {
                "comparison_type": "bnb_vs_dequantized_same_quantized_weight",
                "condition": "bnb_fp4_bf16_compute_vs_bnb_dequantized_bf16_f_linear",
                "m": count,
            }
            comparisons.append({**pair, **summarize_pair(
                condition_outputs["bnb_fp4_bf16_compute"][str(count)],
                condition_outputs["bnb_dequantized_bf16_f_linear"][str(count)],
            )})
            progress["completed_comparisons"] += 1

        if progress["completed_forward_calls"] != EXPECTED_FORWARD_CALLS:
            raise RuntimeError("forward_call_count_mismatch")
        if progress["completed_comparisons"] != EXPECTED_COMPARISONS:
            raise RuntimeError("comparison_count_mismatch")
        numeric_crosscheck_ok = all(item["numeric_crosscheck_within_1e_10"] for item in comparisons)
        torch.cuda.synchronize(target)
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(target))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(target))

        device_props = torch.cuda.get_device_properties(target)
        sm = [int(device_props.major), int(device_props.minor)]
        source_predictions = [source_dispatch_prediction(m, sm=tuple(sm)) for m in ROW_COUNTS]
        return {
            "status": "complete" if numeric_crosscheck_ok else "failed",
            "failure_code": None if numeric_crosscheck_ok else "numeric_crosscheck_failed",
            "purpose": "synthetic_isolated_linear_row_count_characterization",
            "interpretation_limit": "does not reproduce causal masking, Qwen activations, or establish cause of model drift",
            "seed": SEED,
            "width": WIDTH,
            "weight_shape_out_in": [WIDTH, WIDTH],
            "input_shapes": {"M1": [1, WIDTH], "M584": [584, WIDTH], "M585": [585, WIDTH], "M586": [586, WIDTH]},
            "common_selected_input_row": COMMON_ROW,
            "M1_selected_input_row": COMMON_ROW,
            "appended_duplicate_row_source_index": 0,
            "appended_duplicate_row_count": 2,
            "appended_duplicate_row_indices": [584, 585],
            "appended_duplicate_rows_match_source_bitwise": True,
            "hashes": {"bf16_weight_sha256": weight_sha, "bf16_base_input_sha256": input_sha,
                       "bf16_extended_input_sha256": extended_sha,
                       "bf16_common_selected_input_row_sha256": tensor_sha256(inputs_bf16[COMMON_ROW:COMMON_ROW + 1]),
                       "bf16_appended_duplicate_rows_sha256": tensor_sha256(extended[584:586]),
                       "bnb_packed_weight_sha256_before_forwards": packed_sha_before,
                       "bnb_packed_weight_sha256_after_forwards": packed_sha_after,
                       "bnb_dequantized_weight_bf16_sha256": dequant_sha,
                       "bnb_fallback_dequantized_weight_bf16_sha256": dispatch_trace.fallback_weights["584"]["sha256"]},
            "condition_settings": {
                "original_bf16": "torch.nn.functional.linear, W/X BF16",
                "original_fp32": "torch.nn.functional.linear, exact BF16 W/X values cast FP32; TF32 disabled; highest precision",
                "bnb": "Linear4bit FP4, BF16 compute, compress_statistics=false, bias=false",
                "dequantized": "torch.nn.functional.linear using this BNB quantized weight dequantized to BF16",
                "bnb_quant_state_before_forwards": quant_state_before,
                "bnb_quant_state_after_forwards": quant_state_after,
                "bnb_quant_state_unchanged_during_forwards": True,
            },
            "dispatch_predictions_not_runtime_observations": source_predictions,
            "python_dispatch_observation_not_inner_kernel_identity": {
                "routes": dispatch_trace.python_routes,
                "fallback_dequant_weight_witnesses": dispatch_trace.fallback_weights,
                "fallback_public_dequant_bitwise_match": True,
                "compiled_cuda_kernel_identity_observed": False,
            },
            "comparison_count": len(comparisons),
            "numeric_crosscheck_all_32_passed": numeric_crosscheck_ok,
            "same_shape_repeat_m": REPEAT_M,
            "same_shape_repeat_added_four_fixed_forwards": True,
            "comparisons": comparisons,
            "numeric_crosscheck_tolerance": 1e-10,
            "numeric_crosscheck_is_outcome_gate": False,
            "cuda": {"name": device_props.name, "compute_capability": sm,
                     "free_bytes_before_work": int(free_bytes), "total_bytes": int(total_bytes),
                     "peak_memory_stats_reset_before_allocation": True,
                     "peak_allocated_bytes": peak_allocated_bytes,
                     "peak_reserved_bytes": peak_reserved_bytes,
                     **math_settings},
            "software": {"python": platform.python_version(), "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda, "bitsandbytes": bnb_version,
                         "bnb_python_source_sha256": installed_source_hashes(),
                         "bnb_native_library": native_library_summary()},
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new direct results/*.json path")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output = exclusive_result_path(repo, args.output)
    try:
        output_fd = reserve_exclusive_private_output(output)
    except FileExistsError:
        raise SystemExit("refusing to overwrite existing result")
    progress = {"attempted_forward_calls": 0, "completed_forward_calls": 0, "completed_comparisons": 0}
    started = time.monotonic()
    record = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_commit(repo),
        "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "status": "failed",
        "failure_code": None,
        "progress": progress,
    }
    try:
        result = run_diagnostic(args.device, progress)
        record.update(result)
    except Exception as exc:
        # Persist only a fixed stage/code, never exception text or local paths.
        record["failure_code"] = str(exc) if str(exc) in {
            "cuda_unavailable", "insufficient_free_vram", "bnb_quantization_state_missing",
            "unexpected_output_shape", "nonfinite_output",
        } else "diagnostic_runtime_failure"
    finally:
        record["progress"] = dict(progress)
        record["elapsed_seconds_excluding_result_write"] = time.monotonic() - started
        record["failure_count"] = int(record["status"] != "complete")
        write_reserved_private_json(output_fd, record)
    print(json.dumps({"status": record["status"], "output": str(output.relative_to(repo)),
                      "failure_count": record["failure_count"], "progress": record["progress"]}, sort_keys=True))
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
