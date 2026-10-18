"""Compare CPU capture recomputation with CUDA-staged, CPU-canonical reductions.

This synthetic numerical diagnostic loads no model and reads no dataset. It
stages identical float32 attention rows through CUDA, then compares the capture
contract's CPU reduction with the selector's CPU-canonical reduction. It does
not test native CUDA reductions or establish model faithfulness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct

from qta_attention_capture_contract import LAYERS, build_attention_capture
from qwen_question_temporal_attention_selector import (
    aggregate_temporal_group_attention,
    select_top_temporal_groups,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def parity_passed(result: dict) -> bool:
    """Return true only when all exact arithmetic/selection checks agree."""
    if not isinstance(result, dict):
        return False
    integer_fields = (
        "trials", "temporal_group_count",
        "per_layer_group_mass_comparisons",
        "per_layer_mass_mismatch_count", "aggregate_score_comparisons",
        "aggregate_score_mismatch_count", "topk_selection_mismatch_count",
    )
    if any(type(result.get(field)) is not int for field in integer_fields):
        return False
    trials = result["trials"]
    groups = result["temporal_group_count"]
    if (trials < 1 or groups < 1
            or result["per_layer_group_mass_comparisons"] != trials * len(LAYERS) * groups
            or result["aggregate_score_comparisons"] != trials * groups):
        return False
    for field in ("maximum_absolute_layer_mass_difference",
                  "maximum_absolute_aggregate_score_difference"):
        value = result.get(field)
        if type(value) is not float or not math.isfinite(value) or value != 0.0:
            return False
    return (result["per_layer_mass_mismatch_count"] == 0
            and result["aggregate_score_mismatch_count"] == 0
            and result["topk_selection_mismatch_count"] == 0)


def run(*, seed: int, trials: int, device: str = "cuda") -> dict:
    import torch

    if not torch.cuda.is_available() or device != "cuda":
        raise RuntimeError("cuda_required")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dev = torch.device(device)
    heads, query_count, input_length = 8, 4, 64
    qpos = list(range(40, 44))
    groups = [list(range(i * 4, i * 4 + 4)) for i in range(8)]
    binding = {
        "record_token": _sha("synthetic-record"),
        "row_binding_sha256": _sha("synthetic-row"),
        "typed_batch_sha256": _sha("synthetic-batch"),
        "baseline_pq_row_sha256": _sha("synthetic-baseline"),
        "prompt_sha256": _sha("synthetic-prompt"),
        "input_ids_sha256": _sha("synthetic-input-ids"),
        "model_sha256": _sha("synthetic-model"),
        "processor_sha256": _sha("synthetic-processor"),
        "tokenizer_sha256": _sha("synthetic-tokenizer"),
        "runtime_sha256": _sha("synthetic-runtime"),
        "question_positions": qpos,
        "temporal_groups": groups,
    }
    per_layer_mass_mismatches = 0
    aggregate_score_mismatches = 0
    topk_selection_mismatches = 0
    maximum_absolute_score_difference = 0.0
    maximum_absolute_layer_mass_difference = 0.0
    top_k = max(1, math.ceil(0.10 * len(groups)))

    for trial in range(trials):
        # Generate normalized synthetic attention on CPU, then send the exact
        # float32 values to CUDA so both paths receive the same input bytes.
        logits = torch.randn((len(LAYERS), heads, query_count, input_length),
                             generator=torch.Generator(device="cpu").manual_seed(
                                 seed + trial), dtype=torch.float32)
        cpu_attention = torch.softmax(logits, dim=-1)
        capture_input = cpu_attention.tolist()
        capture = build_attention_capture(binding=binding, attention=capture_input,
                                          input_length=input_length)
        cpu_mass_rows = [[struct.unpack(">f", bytes.fromhex(value))[0]
                          for value in row]
                         for row in capture["layer_group_masses_f32_be_hex"]]
        gpu_attention = cpu_attention.to(dev)
        by_layer = {}
        for layer_index, layer_id in enumerate(LAYERS):
            full = torch.zeros((1, heads, input_length, input_length),
                               dtype=torch.float32, device=dev)
            full[0, :, qpos, :] = gpu_attention[layer_index]
            by_layer[layer_id - 1] = full
        gpu_scores = aggregate_temporal_group_attention(
            attention_by_layer=by_layer,
            layer_indices=[layer - 1 for layer in LAYERS],
            question_positions=qpos,
            group_positions=groups,
            torch_module=torch,
        )
        cpu_scores = [sum(row[group_index] for row in cpu_mass_rows) / len(LAYERS)
                      for group_index in range(len(groups))]
        for layer_idx, row in enumerate(cpu_mass_rows):
            gpu_single = aggregate_temporal_group_attention(
                attention_by_layer={layer_idx + 4: by_layer[layer_idx + 4]},
                layer_indices=[layer_idx + 4], question_positions=qpos,
                group_positions=groups, torch_module=torch)
            for group_idx, cpu_mass in enumerate(row):
                diff = abs(cpu_mass - gpu_single[group_idx])
                maximum_absolute_layer_mass_difference = max(
                    maximum_absolute_layer_mass_difference, diff)
                if cpu_mass != gpu_single[group_idx]:
                    per_layer_mass_mismatches += 1
        for cpu_score, gpu_score in zip(cpu_scores, gpu_scores, strict=True):
            diff = abs(cpu_score - gpu_score)
            maximum_absolute_score_difference = max(maximum_absolute_score_difference,
                                                    diff)
            if cpu_score != gpu_score:
                aggregate_score_mismatches += 1
        if (select_top_temporal_groups(cpu_scores, top_k=top_k)
                != select_top_temporal_groups(gpu_scores, top_k=top_k)):
            topk_selection_mismatches += 1

    result = {
        "comparison": "CPU capture-contract vs CUDA-staged-input CPU-canonical selector",
        "seed": seed,
        "trials": trials,
        "attention_shape_per_trial": [len(LAYERS), heads, query_count, input_length],
        "temporal_group_count": len(groups),
        "top_k": top_k,
        "per_layer_group_mass_comparisons": trials * len(LAYERS) * len(groups),
        "per_layer_mass_mismatch_count": per_layer_mass_mismatches,
        "aggregate_score_comparisons": trials * len(groups),
        "aggregate_score_mismatch_count": aggregate_score_mismatches,
        "topk_selection_mismatch_count": topk_selection_mismatches,
        "maximum_absolute_layer_mass_difference": maximum_absolute_layer_mass_difference,
        "maximum_absolute_aggregate_score_difference": maximum_absolute_score_difference,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(dev),
        "model_loaded": False,
        "dataset_read": False,
        "faithfulness_result": False,
        "execution_authorization": "not_granted",
    }
    result["parity_passed"] = parity_passed(result)
    result["status"] = (
        "passed_synthetic_cuda_to_cpu_canonical_transfer_parity"
        if result["parity_passed"] else
        "failed_synthetic_cuda_to_cpu_canonical_transfer_parity")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--trials", type=int, default=32)
    args = parser.parse_args()
    if args.seed < 0 or args.trials < 1:
        parser.error("seed must be nonnegative and trials must be positive")
    try:
        result = run(seed=args.seed, trials=args.trials)
    except Exception as exc:
        print(json.dumps({"status": "failed", "failure_code": type(exc).__name__},
                         sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["parity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
