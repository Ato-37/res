"""Compare native teacher-forced Qwen logits with fresh-cache decoding on CPU.

Fixed synthetic tiny-model diagnostic only: no video pixels, QA, labels, GPU,
or candidate temporal-position policy. It checks that direct model.forward
calls use the same installed native 3-row position IDs for one fixed suffix.

Usage::

    PYTHONPATH=scripts .venv/bin/python scripts/audit_qwen_teacher_forcing_cache_cpu.py \
      --output results/qwen_teacher_forcing_cache_cpu_20261012_01.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

import torch

from audit_qwen_cache_positions_cpu import (
    SEED,
    _fixture,
    _sha256,
    _source_commit,
    _tiny_model,
)


SUFFIX_IDS = (12, 13, 14)
ATOL = 1e-5
RTOL = 1e-4
WRONG_POSITION_SHIFT = 1
EXPECTED_MODEL_FORWARD_CALLS = 6


def _protocol_summary() -> dict:
    return {
        "position_source": "installed native model.model.get_rope_index on prompt plus fixed suffix",
        "forward_api": "direct model.forward with explicit native three-row positions; no GenerationMixin packed text row",
        "cache_path": "fresh prompt prefill followed by two sequential one-token calls",
        "fixed_suffix_length": len(SUFFIX_IDS),
        "expected_model_forward_calls": EXPECTED_MODEL_FORWARD_CALLS,
        "score_alignment": "full-forward rows P-1,P,P+1 vs prefill-final and decode outputs after suffix tokens 0,1",
        "atol": ATOL,
        "rtol": RTOL,
        "negative_control": {
            "position_shift_all_three_axes": WRONG_POSITION_SHIFT,
            "comparison": "logits predicting suffix token 1 against native teacher-forced row P",
            "required_result": "not allclose at the same fixed tolerance",
            "sensitivity_failure_action": "record as diagnostic failure; no offset/token/seed search",
        },
    }


class _PositionCapture:
    def __init__(self, model):
        self.values = []
        self.handle = model.model.language_model.register_forward_pre_hook(
            self._capture, with_kwargs=True
        )

    def _capture(self, _module, _args, kwargs):
        value = kwargs.get("position_ids")
        self.values.append(None if value is None else value.detach().cpu().clone())

    def close(self):
        self.handle.remove()


def _cache_length(cache) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    # Compatibility with legacy tuple caches used by older Transformers APIs.
    return int(cache[0][0].shape[-2])


def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _allclose(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(left, right, atol=ATOL, rtol=RTOL))


def _tracked_forward(context: dict, stage: str, forward):
    context["stage"] = stage
    context["model_forward_calls_attempted"] += 1
    context["call_counts_by_stage_attempted"][stage] = (
        context["call_counts_by_stage_attempted"].get(stage, 0) + 1
    )
    output = forward()
    context["model_forward_calls_completed"] += 1
    context["call_counts_by_stage_completed"][stage] = (
        context["call_counts_by_stage_completed"].get(stage, 0) + 1
    )
    return output


def _run_diagnostic_impl(context: dict) -> dict:
    context["stage"] = "model_setup"
    model = _tiny_model()
    prompt_kwargs, _span = _fixture(model)
    context["model_config"] = {
        "config_sha256": hashlib.sha256(
            model.config.to_json_string(use_diff=False).encode()
        ).hexdigest(),
        "text_hidden_size": model.config.text_config.hidden_size,
        "text_layers": model.config.text_config.num_hidden_layers,
        "text_attention_heads": model.config.text_config.num_attention_heads,
        "text_key_value_heads": model.config.text_config.num_key_value_heads,
        "vocab_size": model.config.text_config.vocab_size,
        "vision_layers": model.config.vision_config.depth,
    }
    prompt_ids = prompt_kwargs["input_ids"]
    prompt_length = int(prompt_ids.shape[1])
    context["prompt_length"] = prompt_length
    context["stage"] = "native_position_setup"
    suffix = torch.tensor([SUFFIX_IDS], dtype=torch.long)
    full_ids = torch.cat((prompt_ids, suffix), dim=1)
    full_attention = torch.ones_like(full_ids)
    full_mm_types = torch.cat(
        (prompt_kwargs["mm_token_type_ids"], torch.zeros_like(suffix, dtype=torch.int32)),
        dim=1,
    )
    full_kwargs = {
        "input_ids": full_ids,
        "attention_mask": full_attention,
        "mm_token_type_ids": full_mm_types,
        "video_grid_thw": prompt_kwargs["video_grid_thw"],
        "second_per_grid_ts": prompt_kwargs["second_per_grid_ts"],
    }
    full_positions, _full_delta = model.model.get_rope_index(**full_kwargs)
    prompt_positions, _prompt_delta = model.model.get_rope_index(**prompt_kwargs)
    if full_positions.ndim != 3 or full_positions.shape[0] != 3:
        raise RuntimeError("expected native 3-row Qwen positions, without GenerationMixin text row")
    if not torch.equal(full_positions[:, :, :prompt_length], prompt_positions):
        raise RuntimeError("native positions for prompt changed when fixed suffix was appended")

    # One full teacher-forced pass. Row P-1 predicts suffix token 0; rows P and
    # P+1 predict suffix tokens 1 and 2 respectively.
    teacher_capture = _PositionCapture(model)
    try:
        with torch.inference_mode():
            teacher = _tracked_forward(
                context,
                "teacher_forced_forward",
                lambda: model(
                    **full_kwargs,
                    position_ids=full_positions,
                    use_cache=False,
                ),
            )
    finally:
        teacher_capture.close()
    if len(teacher_capture.values) != 1 or not torch.equal(
        teacher_capture.values[0], full_positions
    ):
        raise RuntimeError("teacher-forced native position IDs did not reach language model")

    # Fresh-cache path: prompt prefill followed by two sequential one-token
    # forwards. Each decode position is sliced from the same native full pass.
    def prefill():
        capture = _PositionCapture(model)
        try:
            with torch.inference_mode():
                output = _tracked_forward(
                    context,
                    "cache_prefill",
                    lambda: model(
                        **prompt_kwargs,
                        position_ids=full_positions[:, :, :prompt_length],
                        use_cache=True,
                    ),
                )
        finally:
            capture.close()
        if len(capture.values) != 1 or not torch.equal(
            capture.values[0], full_positions[:, :, :prompt_length]
        ):
            raise RuntimeError("prefill native position IDs did not reach language model")
        return output

    prefill_output = prefill()
    cache_lengths = [_cache_length(prefill_output.past_key_values)]
    context["cache_lengths"] = cache_lengths
    sequential_outputs = [prefill_output.logits[:, -1, :].detach().cpu()]
    sequential_positions = []
    current_cache = prefill_output.past_key_values
    for offset in range(2):
        token = suffix[:, offset : offset + 1]
        native_position = full_positions[:, :, prompt_length + offset : prompt_length + offset + 1]
        mask = torch.ones((1, prompt_length + offset + 1), dtype=torch.long)
        capture = _PositionCapture(model)
        try:
            with torch.inference_mode():
                output = _tracked_forward(
                    context,
                    "sequential_cached_decode",
                    lambda: model(
                        input_ids=token,
                        attention_mask=mask,
                        position_ids=native_position,
                        past_key_values=current_cache,
                        use_cache=True,
                    ),
                )
        finally:
            capture.close()
        if len(capture.values) != 1 or not torch.equal(capture.values[0], native_position):
            raise RuntimeError("cached decode position slice did not reach language model")
        sequential_positions.append(capture.values[0])
        current_cache = output.past_key_values
        cache_lengths.append(_cache_length(current_cache))
        context["cache_lengths"] = cache_lengths
        sequential_outputs.append(output.logits[:, -1, :].detach().cpu())

    teacher_scores = [
        teacher.logits[:, prompt_length - 1 + offset, :].detach().cpu()
        for offset in range(3)
    ]
    finite_logits = all(bool(torch.isfinite(score).all()) for score in teacher_scores + sequential_outputs)
    differences = [
        (_max_abs_difference(expected, actual)
         if bool(torch.isfinite(expected).all()) and bool(torch.isfinite(actual).all())
         else None)
        for expected, actual in zip(teacher_scores, sequential_outputs)
    ]
    matches = [
        (bool(torch.isfinite(expected).all()) and bool(torch.isfinite(actual).all())
         and _allclose(expected, actual))
        for expected, actual in zip(teacher_scores, sequential_outputs)
    ]
    context["prediction_matches"] = matches
    context["prediction_max_abs_errors"] = differences
    context["all_scores_finite"] = finite_logits

    # Fixed sensitivity control: start from an independent fresh prefill and
    # shift only the first suffix token's three native coordinates by +1.
    wrong_prefill = prefill()
    wrong_token = suffix[:, :1]
    wrong_position = full_positions[:, :, prompt_length : prompt_length + 1] + WRONG_POSITION_SHIFT
    wrong_mask = torch.ones((1, prompt_length + 1), dtype=torch.long)
    wrong_capture = _PositionCapture(model)
    try:
        with torch.inference_mode():
            wrong = _tracked_forward(
                context,
                "negative_control_decode",
                lambda: model(
                    input_ids=wrong_token,
                    attention_mask=wrong_mask,
                    position_ids=wrong_position,
                    past_key_values=wrong_prefill.past_key_values,
                    use_cache=True,
                ),
            )
    finally:
        wrong_capture.close()
    if len(wrong_capture.values) != 1 or not torch.equal(wrong_capture.values[0], wrong_position):
        raise RuntimeError("fixed wrong-offset control did not reach language model")
    wrong_logits = wrong.logits[:, -1, :].detach().cpu()
    wrong_finite = bool(torch.isfinite(wrong_logits).all())
    wrong_difference = (
        _max_abs_difference(teacher_scores[1], wrong_logits)
        if wrong_finite and bool(torch.isfinite(teacher_scores[1]).all())
        else None
    )
    wrong_is_close = (
        wrong_finite and bool(torch.isfinite(teacher_scores[1]).all())
        and _allclose(teacher_scores[1], wrong_logits)
    )
    context["wrong_offset_control_finite"] = wrong_finite
    context["wrong_offset_control_max_abs_error"] = wrong_difference
    context["wrong_offset_control_not_allclose"] = not wrong_is_close

    expected_cache_lengths = [prompt_length, prompt_length + 1, prompt_length + 2]
    cache_lengths_match = cache_lengths == expected_cache_lengths
    gate_checks = {
        "native_position_captures_match": True,
        "teacher_forced_and_cache_logits_finite": finite_logits,
        "teacher_forced_cache_predictions_match_fixed_tolerance": all(matches),
        "cache_lengths_match_expected": cache_lengths_match,
        "model_forward_calls_match_expected": context["model_forward_calls_completed"] == EXPECTED_MODEL_FORWARD_CALLS,
        "wrong_offset_control_finite": wrong_finite,
        "wrong_offset_control_separates_at_fixed_tolerance": not wrong_is_close,
    }
    status = "ok" if all(gate_checks.values()) else "diagnostic_gate_failed"
    context["stage"] = "gate_validation"

    return {
        "schema_version": 1,
        "purpose": "native_teacher_forcing_vs_fresh_cache_cpu_api_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "fixture": {
            "batch_size": 1,
            "prompt_tokens": prompt_length,
            "fixed_suffix_length": len(SUFFIX_IDS),
            "video_grid_thw": [8, 4, 4],
            "video_placeholder_tokens": int(prompt_kwargs["video_grid_thw"].prod().item() // 4),
            "synthetic_token_ids_omitted": True,
            "attention_masks_all_ones": True,
        },
        "protocol": _protocol_summary(),
        "observations": {
            "model_forward_calls_attempted": context["model_forward_calls_attempted"],
            "model_forward_calls_completed": context["model_forward_calls_completed"],
            "model_forward_calls_expected": EXPECTED_MODEL_FORWARD_CALLS,
            "model_forward_calls_by_stage_attempted": context["call_counts_by_stage_attempted"],
            "model_forward_calls_by_stage_completed": context["call_counts_by_stage_completed"],
            "teacher_forced_forward_calls": context["call_counts_by_stage_completed"].get("teacher_forced_forward", 0),
            "cache_prefill_calls": context["call_counts_by_stage_completed"].get("cache_prefill", 0),
            "sequential_decode_calls": context["call_counts_by_stage_completed"].get("sequential_cached_decode", 0),
            "negative_control_decode_calls": context["call_counts_by_stage_completed"].get("negative_control_decode", 0),
            "cache_lengths_after_prefill_and_decode": cache_lengths,
            "cache_lengths_expected": [prompt_length, prompt_length + 1, prompt_length + 2],
            "captured_decode_positions_match_native_slices": True,
            "all_scores_finite": finite_logits and wrong_finite,
            "teacher_forced_vs_cache_allclose_each_suffix_prediction": matches,
            "teacher_forced_vs_cache_max_abs_error_each_suffix_prediction": differences,
            "wrong_offset_control_not_allclose": not wrong_is_close,
            "wrong_offset_control_max_abs_error": wrong_difference,
            "gate_checks": gate_checks,
        },
        "model_config": {**context["model_config"], "attention_implementation": "eager"},
        "scope": "fixed synthetic placeholder geometry; tiny random CPU FP32 model; native positions only; no visual pixels, QA, candidate policy, or efficacy",
        "limitations": "does not establish 7B BF16/4-bit behavior, visual encoder compatibility, semantic validity of continuation coordinates, or a universal cache contract",
        "status": status,
        "failure_stage": None if status == "ok" else "gate_validation",
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "device": "cpu",
            "dtype": "float32",
            "model": "random tiny Qwen2.5-VL configuration reused from CPU cache-position audit",
        },
    }


def run_diagnostic() -> dict:
    """Use fixed seed/settings and restore process-global CPU state afterward."""
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_threads = torch.get_num_threads()
    try:
        random.seed(SEED)
        torch.random.default_generator.manual_seed(SEED)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_num_threads(1)
        started = time.monotonic()
        context = {
            "stage": "model_setup",
            "model_forward_calls_attempted": 0,
            "model_forward_calls_completed": 0,
            "call_counts_by_stage_attempted": {},
            "call_counts_by_stage_completed": {},
            "cache_lengths": [],
        }
        try:
            result = _run_diagnostic_impl(context)
        except Exception as exc:
            # Persist only a sanitized class/stage and aggregate partial counts;
            # exception messages may contain input details and are not needed.
            result = {
                "schema_version": 1,
                "purpose": "native_teacher_forcing_vs_fresh_cache_cpu_api_diagnostic",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": SEED,
                "status": "execution_failed",
                "failure_stage": context["stage"],
                "failure_exception_type": type(exc).__name__,
                "protocol": _protocol_summary(),
                "fixture": {
                    "batch_size": 1,
                    "prompt_tokens": context.get("prompt_length"),
                    "fixed_suffix_length": len(SUFFIX_IDS),
                    "video_grid_thw": [8, 4, 4],
                    "synthetic_token_ids_omitted": True,
                    "attention_masks_all_ones": True,
                },
                "observations": {
                    "model_forward_calls_attempted": context["model_forward_calls_attempted"],
                    "model_forward_calls_completed": context["model_forward_calls_completed"],
                    "model_forward_calls_expected": EXPECTED_MODEL_FORWARD_CALLS,
                    "model_forward_calls_by_stage_attempted": context[
                        "call_counts_by_stage_attempted"
                    ],
                    "model_forward_calls_by_stage_completed": context[
                        "call_counts_by_stage_completed"
                    ],
                    "cache_lengths_observed": context["cache_lengths"],
                    "cache_lengths_expected": (
                        [context["prompt_length"], context["prompt_length"] + 1,
                         context["prompt_length"] + 2]
                        if "prompt_length" in context else None
                    ),
                    "teacher_forced_vs_cache_allclose_each_suffix_prediction": context.get(
                        "prediction_matches"
                    ),
                    "teacher_forced_vs_cache_max_abs_error_each_suffix_prediction": context.get(
                        "prediction_max_abs_errors"
                    ),
                    "wrong_offset_control_finite": context.get("wrong_offset_control_finite"),
                    "wrong_offset_control_not_allclose": context.get(
                        "wrong_offset_control_not_allclose"
                    ),
                    "wrong_offset_control_max_abs_error": context.get(
                        "wrong_offset_control_max_abs_error"
                    ),
                },
                "model_config": context.get("model_config"),
                "scope": "fixed synthetic placeholder geometry; tiny random CPU FP32 model; native positions only; no visual pixels, QA, candidate policy, or efficacy",
                "limitations": "does not establish 7B BF16/4-bit behavior, visual encoder compatibility, semantic validity of continuation coordinates, or a universal cache contract",
                "versions": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "transformers": importlib.metadata.version("transformers"),
                    "device": "cpu",
                    "dtype": "float32",
                },
            }
        result["elapsed_seconds"] = time.monotonic() - started
        return result
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def main(argv=None, repo_path: Path | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="exclusive new JSON path directly under results/")
    args = parser.parse_args(argv)
    repo = Path(repo_path).resolve() if repo_path is not None else Path(__file__).resolve().parents[1]
    results = (repo / "results").resolve()
    output = (repo / args.output).resolve()
    if output.parent != results or output.exists():
        parser.error("exclusive new output directly under results/ required; overwrite refused")

    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    generation_utils = __import__("transformers.generation.utils", fromlist=["utils"])
    prior_cpu_audit = repo / "scripts" / "audit_qwen_cache_positions_cpu.py"
    result = run_diagnostic()
    result["source_commit"] = _source_commit(repo)
    result["source_sha256"] = {
        "runner": _sha256(Path(__file__).resolve()),
        "reused_cpu_fixture_and_model_config": _sha256(prior_cpu_audit),
        "installed_qwen_model": _sha256(Path(modeling_qwen2_5_vl.__file__).resolve()),
        "installed_generation_utils": _sha256(Path(generation_utils.__file__).resolve()),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(output, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    output.chmod(0o600)
    print(json.dumps({
        "status": result["status"],
        "output": str(output.relative_to(repo)),
        "stage": result.get("failure_stage"),
        "model_forward_calls": result["observations"].get("model_forward_calls_completed"),
        "cache_lengths": result["observations"].get("cache_lengths_after_prefill_and_decode",
                                                        result["observations"].get("cache_lengths_observed")),
        "matched_predictions": sum(result["observations"].get("teacher_forced_vs_cache_allclose_each_suffix_prediction") or []),
        "negative_control_separated": result["observations"].get("wrong_offset_control_not_allclose"),
        "elapsed_seconds": round(result.get("elapsed_seconds", 0.0), 3),
    }))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
