"""Audit installed Qwen 2.5-VL position and cache plumbing on a tiny CPU model.

This is a deterministic API/shape diagnostic, not a numerical proxy for the
cached 7B checkpoint. It uses no video pixels, labels, external data, or GPU.

Usage::

    PYTHONPATH=scripts .venv/bin/python scripts/audit_qwen_cache_positions_cpu.py \
      --output results/qwen_cache_positions_cpu_20261012_01.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import os

import torch

from qwen_temporal_geometry import replace_video_temporal_rope_row


SEED = 20261012
GRID_T_H_W = (8, 4, 4)
SECOND_PER_GRID = 0.75
TOKENS_PER_SECOND = 2
SPATIAL_MERGE_SIZE = 2
VIDEO_TOKEN_COUNT = GRID_T_H_W[0] * (GRID_T_H_W[1] // 2) * (GRID_T_H_W[2] // 2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tiny_model():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    vision = {
        "depth": 1,
        "hidden_size": 48,
        "intermediate_size": 96,
        "num_heads": 2,
        "in_channels": 3,
        "patch_size": 14,
        "spatial_merge_size": SPATIAL_MERGE_SIZE,
        "temporal_patch_size": 2,
        "tokens_per_second": TOKENS_PER_SECOND,
        "window_size": 14,
        "out_hidden_size": 48,
        "fullatt_block_indexes": [],
    }
    text = {
        "vocab_size": 64,
        "hidden_size": 48,
        "intermediate_size": 96,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "use_sliding_window": False,
        "layer_types": ["full_attention"],
        "attention_dropout": 0.0,
        "rope_parameters": {"rope_theta": 1000000.0, "mrope_section": [4, 4, 4]},
        "bos_token_id": 1,
        "eos_token_id": None,
        "pad_token_id": 0,
    }
    config = Qwen2_5_VLConfig(
        vision_config=vision,
        text_config=text,
        image_token_id=61,
        video_token_id=62,
        vision_start_token_id=60,
        vision_end_token_id=59,
    )
    config._attn_implementation = "eager"
    model = Qwen2_5_VLForConditionalGeneration(config).to(device="cpu", dtype=torch.float32).eval()
    return model


def _fixture(model):
    # Text prefix, one video block, video end and one text suffix. Video tokens
    # have placeholder embeddings only; this deliberately tests position/cache
    # plumbing rather than the vision encoder or visual content.
    prefix = [10, model.config.vision_start_token_id]
    suffix = [model.config.vision_end_token_id, 11]
    video = [model.config.video_token_id] * VIDEO_TOKEN_COUNT
    ids = torch.tensor([prefix + video + suffix], dtype=torch.long)
    mm_types = torch.tensor([[0] * len(prefix) + [2] * len(video) + [0] * len(suffix)], dtype=torch.int32)
    mask = torch.ones_like(ids)
    span_start = len(prefix)
    span_stop = span_start + len(video)
    kwargs = {
        "input_ids": ids,
        "attention_mask": mask,
        "mm_token_type_ids": mm_types,
        "video_grid_thw": torch.tensor([GRID_T_H_W], dtype=torch.long),
        "second_per_grid_ts": torch.tensor([SECOND_PER_GRID], dtype=torch.float32),
    }
    return kwargs, (span_start, span_stop)


class _PositionCapture:
    def __init__(self, model):
        self.values = []
        self.text_rows = []
        self.handle = model.model.language_model.register_forward_pre_hook(
            self._capture, with_kwargs=True
        )

    def _capture(self, _module, _args, kwargs):
        value = kwargs.get("position_ids")
        if value is None:
            self.values.append(None)
            self.text_rows.append(None)
        else:
            value = value.detach().cpu().clone()
            if value.ndim == 3 and value.shape[0] == 4:
                self.text_rows.append(value[0].clone())
                value = value[1:]
            else:
                self.text_rows.append(None)
            self.values.append(value)

    def close(self):
        self.handle.remove()


def _packed_positions(native, candidate, seq_len):
    text_positions = torch.arange(seq_len, dtype=torch.long).view(1, 1, seq_len)
    return (
        torch.cat((text_positions, native), dim=0),
        torch.cat((text_positions, candidate), dim=0),
    )


def _run_generate(model, kwargs, position_ids=None):
    model.model.rope_deltas = None
    capture = _PositionCapture(model)
    try:
        with torch.inference_mode():
            output = model.generate(
                **kwargs,
                **({} if position_ids is None else {"position_ids": position_ids}),
                use_cache=True,
                do_sample=False,
                max_new_tokens=2,
                eos_token_id=None,
                pad_token_id=0,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        capture.close()
    return output, capture.values, model.model.rope_deltas


def _delta_for(position_ids, prompt_length):
    # This mirrors the installed get_rope_index mechanical convention. It is
    # reported as a cache-coordinate rule, not an endorsed semantic policy.
    return int(position_ids.max().item()) + 1 - int(prompt_length)


def _resume_generate(model, ids, cache, delta, prompt_length, token_id):
    model.model.rope_deltas = (
        None if delta is None else torch.tensor([[delta]], dtype=torch.long)
    )
    attention_mask = torch.ones((1, prompt_length + 1), dtype=torch.long)
    capture = _PositionCapture(model)
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=torch.tensor([[token_id]], dtype=torch.long),
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                do_sample=False,
                max_new_tokens=1,
                eos_token_id=None,
                pad_token_id=0,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        capture.close()
    return output, capture.values


def run_diagnostic():
    """Run in an isolated RNG/determinism context and restore caller state."""
    python_rng_state = random.getstate()
    torch_rng_state = torch.get_rng_state()
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        random.seed(SEED)
        torch.random.default_generator.manual_seed(SEED)
        torch.use_deterministic_algorithms(True, warn_only=False)
        started = time.monotonic()
        result = _run_diagnostic_impl()
        result["elapsed_seconds"] = time.monotonic() - started
        return result
    finally:
        random.setstate(python_rng_state)
        torch.set_rng_state(torch_rng_state)
        torch.use_deterministic_algorithms(
            deterministic_enabled, warn_only=deterministic_warn_only
        )


def _run_diagnostic_impl():
    model = _tiny_model()
    kwargs, span = _fixture(model)
    ids = kwargs["input_ids"]
    prompt_length = int(ids.shape[1])

    native, native_delta = model.model.get_rope_index(**kwargs)
    native_single = native[:, 0, :].contiguous()
    corrected_rows = replace_video_temporal_rope_row(
        native_single.tolist(),
        video_token_spans=[span],
        image_token_spans=[],
        attention_mask=kwargs["attention_mask"][0].tolist(),
        video_grid_thw=GRID_T_H_W,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        second_per_grid_seconds=SECOND_PER_GRID,
        tokens_per_second=TOKENS_PER_SECOND,
    )
    candidate_single = torch.tensor(corrected_rows, dtype=torch.long)
    candidate = candidate_single.unsqueeze(1)
    packed_native, packed_candidate = _packed_positions(native, candidate, prompt_length)
    native_delta_scalar = int(native_delta[0, 0].item())
    candidate_delta_scalar = _delta_for(candidate_single, prompt_length)
    candidate_equals_native = torch.equal(candidate_single, native_single)

    # Uninterrupted generate: the generic GenerationMixin extends supplied
    # positions from the last prompt token, while Qwen auto-preparation sets
    # rope_deltas only when it is asked to create positions itself.
    baseline, baseline_captures, baseline_state = _run_generate(model, kwargs)
    explicit_native, native_captures, explicit_native_state = _run_generate(
        model, kwargs, packed_native
    )
    explicit_candidate, candidate_captures, candidate_state = _run_generate(
        model, kwargs, packed_candidate
    )

    def equal_score_sets(left, right):
        return (len(left.scores) == len(right.scores)
                and all(torch.equal(a, b) for a, b in zip(left.scores, right.scores)))

    auto_pids = baseline_captures[0]
    native_pids = native_captures[0]
    candidate_pids = candidate_captures[0]
    if any(value is None for value in (auto_pids, native_pids, candidate_pids)):
        raise RuntimeError("language model did not receive position IDs")
    if not torch.equal(auto_pids, native_single.unsqueeze(1)):
        raise RuntimeError("native auto prefill IDs differ from actual get_rope_index output")
    if not torch.equal(native_pids, native_single.unsqueeze(1)):
        raise RuntimeError("explicit-native prefill IDs were not passed through")
    if not torch.equal(candidate_pids, candidate_single.unsqueeze(1)):
        raise RuntimeError("candidate prefill IDs were not passed through")
    if len(baseline_captures) != 2 or len(native_captures) != 2 or len(candidate_captures) != 2:
        raise RuntimeError("two-token generation did not produce one prefill and one cached decode")
    prompt_last = int(packed_native[1:, 0, -1].max().item())
    expected_continuous_decode = prompt_last + 1
    if not torch.equal(native_captures[1], torch.full((3, 1, 1), expected_continuous_decode)):
        raise RuntimeError("explicit-native continuous decode did not extend last prompt position by one")
    if not torch.equal(candidate_captures[1], native_captures[1]):
        raise RuntimeError("candidate continuous decode changed the last-text continuation position")
    if not torch.equal(baseline_captures[1], native_captures[1]):
        raise RuntimeError("automatic and explicit-native continuous decode positions differ")

    # Prove the model state behavior of explicit prefill: the forward bypasses
    # compute_3d_position_ids, leaving a cleared state cleared and a stale state
    # stale. Use separate fresh models/caches so cache mutations cannot leak.
    def prefill(position_ids, initial_delta=None):
        fresh = _tiny_model()
        fresh.model.rope_deltas = initial_delta
        cap = _PositionCapture(fresh)
        with torch.inference_mode():
            out = fresh(**kwargs, position_ids=position_ids, use_cache=True, logits_to_keep=1)
        cap.close()
        return fresh, out, cap.values

    cleared_model, cleared_prefill, cleared_cap = prefill(native_single.unsqueeze(1), None)
    stale_start = torch.tensor([[native_delta_scalar + 5]], dtype=torch.long)
    stale_model, stale_prefill, stale_cap = prefill(native_single.unsqueeze(1), stale_start)
    if cleared_model.model.rope_deltas is not None:
        raise RuntimeError("explicit prefill unexpectedly set a cleared rope_deltas state")
    if not torch.equal(stale_model.model.rope_deltas, stale_start):
        raise RuntimeError("explicit prefill unexpectedly replaced stale rope_deltas state")

    # Source-level cached computation shows a shape concern: with a full S+1
    # attention mask and one-token inputs_embeds, compute_3d_position_ids builds
    # S+1 positions. Exercise the direct forward and record its actual outcome.
    shape_model, shape_prefill, _ = prefill(native_single.unsqueeze(1), native_delta.clone())
    next_token = torch.tensor([[13]], dtype=torch.long)
    full_mask = torch.ones((1, prompt_length + 1), dtype=torch.long)
    next_embeds = shape_model.model.get_input_embeddings()(next_token)
    computed = shape_model.model.compute_3d_position_ids(
        input_ids=next_token,
        image_grid_thw=None,
        video_grid_thw=None,
        inputs_embeds=next_embeds,
        attention_mask=full_mask,
        past_key_values=shape_prefill.past_key_values,
        mm_token_type_ids=None,
    )
    direct_position_length = int(computed.shape[-1]) if computed is not None else None
    direct_outcome = "returned"
    direct_error_type = None
    try:
        with torch.inference_mode():
            direct_out = shape_model(
                input_ids=next_token,
                attention_mask=full_mask,
                past_key_values=shape_prefill.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        direct_finite = bool(torch.isfinite(direct_out.logits).all())
        if not direct_finite:
            direct_outcome = "nonfinite_logits"
    except Exception as exc:  # Record class only; exception text can contain details.
        direct_outcome = "raised"
        direct_error_type = type(exc).__name__
        direct_finite = False

    # Paired cached-forward check: keep the exact prefill/cache and full mask,
    # then pass a single explicit native-policy continuation coordinate. This
    # tests the shape contract only; it does not establish semantic correctness
    # for any cache-resume coordinate policy.
    explicit_model, explicit_prefill, _ = prefill(native_single.unsqueeze(1), native_delta.clone())
    expected_native_position = int(native_single[:, -1].max().item()) + 1
    explicit_next_position = torch.full(
        (3, 1, 1), expected_native_position, dtype=torch.long
    )
    explicit_cached_outcome = "returned"
    explicit_cached_error_type = None
    try:
        with torch.inference_mode():
            explicit_cached_out = explicit_model(
                input_ids=next_token,
                attention_mask=full_mask,
                past_key_values=explicit_prefill.past_key_values,
                position_ids=explicit_next_position,
                use_cache=True,
                logits_to_keep=1,
            )
        explicit_cached_finite = bool(torch.isfinite(explicit_cached_out.logits).all())
        if not explicit_cached_finite:
            explicit_cached_outcome = "nonfinite_logits"
    except Exception as exc:
        explicit_cached_outcome = "raised"
        explicit_cached_error_type = type(exc).__name__
        explicit_cached_finite = False

    # Fixed long-interval geometry shows the distinction between the native
    # maximum-based cache-resume coordinate and ordinary generation's
    # last-packed-token+1 continuation. This is a coordinate-policy comparison,
    # not an assertion that either policy is semantically correct.
    long_interval_kwargs = dict(kwargs)
    long_interval_kwargs["second_per_grid_ts"] = torch.tensor([5.0], dtype=torch.float32)
    long_native, long_native_delta = model.model.get_rope_index(**long_interval_kwargs)
    long_native_single = long_native[:, 0, :]
    long_last_token_next = int(long_native_single[:, -1].max().item()) + 1
    long_cache_resume = int(long_native.max().item()) + 1

    # Test GenerationMixin's cache-resume route, which slices the full-mask
    # positions down to the one current token before calling Qwen forward.
    resume_cleared_model, resume_cleared_prefill, _ = prefill(native_single.unsqueeze(1), None)
    cleared_out, cleared_resume_caps = _resume_generate(
        resume_cleared_model,
        ids,
        resume_cleared_prefill.past_key_values,
        delta=None,
        prompt_length=prompt_length,
        token_id=13,
    )
    resume_native_model, resume_native_prefill, _ = prefill(native_single.unsqueeze(1), None)
    native_out, native_resume_caps = _resume_generate(
        resume_native_model,
        ids,
        resume_native_prefill.past_key_values,
        delta=native_delta_scalar,
        prompt_length=prompt_length,
        token_id=13,
    )
    resume_candidate_model, resume_candidate_prefill, _ = prefill(candidate_single.unsqueeze(1), None)
    candidate_out, candidate_resume_caps = _resume_generate(
        resume_candidate_model,
        ids,
        resume_candidate_prefill.past_key_values,
        delta=candidate_delta_scalar,
        prompt_length=prompt_length,
        token_id=13,
    )
    resume_stale_model, resume_stale_prefill, _ = prefill(
        native_single.unsqueeze(1), stale_start
    )
    stale_out, stale_resume_caps = _resume_generate(
        resume_stale_model,
        ids,
        resume_stale_prefill.past_key_values,
        delta=native_delta_scalar + 5,
        prompt_length=prompt_length,
        token_id=13,
    )

    expected_cleared = prompt_length
    expected_native_resume = int(native_single.max().item()) + 1
    expected_candidate_resume = int(candidate_single.max().item()) + 1
    for label, captures, expected, expected_shape in (
        ("cleared", cleared_resume_caps, expected_cleared, (3, 1, 1)),
        ("native", native_resume_caps, expected_native_resume, (1, 1, 1)),
        ("candidate", candidate_resume_caps, expected_candidate_resume, (1, 1, 1)),
        ("stale", stale_resume_caps, prompt_length + native_delta_scalar + 5, (1, 1, 1)),
    ):
        if len(captures) != 1 or captures[0] is None or captures[0].shape != expected_shape:
            raise RuntimeError(f"{label} resumed generation did not slice positions to one token")
        if not torch.equal(captures[0], torch.full(expected_shape, expected)):
            raise RuntimeError(f"{label} resumed generation used an unexpected cache coordinate")
    for out in (cleared_out, native_out, candidate_out, stale_out):
        if not out.scores or not bool(torch.isfinite(out.scores[0]).all()):
            raise RuntimeError("resumed cache generation returned missing/nonfinite scores")

    # Also verify the explicit candidate values were a temporal-row-only edit.
    video_mask = kwargs["mm_token_type_ids"][0] == 2
    non_temporal_preserved = (
        torch.equal(candidate_single[1:], native_single[1:])
        and torch.equal(candidate_single[0, ~video_mask], native_single[0, ~video_mask])
    )
    if not non_temporal_preserved:
        raise RuntimeError("candidate changed a non-temporal or non-video position")

    return {
        "schema_version": 1,
        "purpose": "installed_qwen_cache_position_cpu_api_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "model_config": {
            "text_hidden_size": model.config.text_config.hidden_size,
            "text_intermediate_size": model.config.text_config.intermediate_size,
            "text_layers": model.config.text_config.num_hidden_layers,
            "text_attention_heads": model.config.text_config.num_attention_heads,
            "text_key_value_heads": model.config.text_config.num_key_value_heads,
            "vocab_size": model.config.text_config.vocab_size,
            "vision_layers": model.config.vision_config.depth,
            "attention_implementation": "eager",
            "config_sha256": hashlib.sha256(model.config.to_json_string(use_diff=False).encode()).hexdigest(),
        },
        "scope": "tiny random CPU model; synthetic token geometry; no visual encoder, QA, labels, or efficacy",
        "precision_note": "tiny randomly initialized FP32 model; not a numerical proxy for pinned 7B BF16/4-bit GPU behavior",
        "fixture": {
            "batch_size": 1,
            "sequence_length": prompt_length,
            "video_grid_thw": list(GRID_T_H_W),
            "video_token_count": VIDEO_TOKEN_COUNT,
            "spatial_merge_size": SPATIAL_MERGE_SIZE,
            "second_per_grid_seconds": SECOND_PER_GRID,
            "tokens_per_second": TOKENS_PER_SECOND,
            "attention_mask_all_ones": True,
            "trailing_text_tokens_after_video_end": 1,
        },
        "position_geometry": {
            "native_temporal_anchors": [int(native_single[0, span[0] + i * 4]) for i in range(GRID_T_H_W[0])],
            "candidate_temporal_anchors": [int(candidate_single[0, span[0] + i * 4]) for i in range(GRID_T_H_W[0])],
            "native_rope_delta": native_delta_scalar,
            "candidate_mechanical_rope_delta": candidate_delta_scalar,
            "candidate_delta_differs": candidate_delta_scalar != native_delta_scalar,
            "fractional_candidate_equals_native": candidate_equals_native,
            "non_temporal_positions_preserved": non_temporal_preserved,
            "delta_convention": "max(position IDs)+1−valid prompt length; source mechanical rule only, not semantic validation",
        },
        "uninterrupted_generate": {
            "native_auto_prefill_matches_get_rope_index": True,
            "explicit_native_prefill_matches_get_rope_index": True,
            "explicit_candidate_prefill_matches_helper": True,
            "native_auto_vs_explicit_native_scores_bitwise_equal": equal_score_sets(baseline, explicit_native),
            "native_auto_vs_explicit_native_sequences_equal": torch.equal(baseline.sequences, explicit_native.sequences),
            "candidate_prefill_changes_only_video_temporal_row": non_temporal_preserved,
            "decode_positions_are_last_prompt_coordinate_plus_one": expected_continuous_decode,
            "candidate_decode_equals_native_decode": torch.equal(candidate_captures[1], native_captures[1]),
            "explicit_native_generation_left_rope_delta_none": explicit_native_state is None,
            "explicit_candidate_generation_left_rope_delta_none": candidate_state is None,
            "native_auto_populated_rope_delta": baseline_state is not None,
        },
        "explicit_prefill_state": {
            "cleared_rope_delta_stayed_cleared": cleared_model.model.rope_deltas is None,
            "stale_rope_delta_survived_explicit_prefill": torch.equal(stale_model.model.rope_deltas, stale_start),
            "native_delta_from_actual_get_rope_index": native_delta_scalar,
            "candidate_delta_manually_set_for_cache_resume": candidate_delta_scalar,
        },
        "direct_cached_forward": {
            "past_length": prompt_length,
            "current_input_length": 1,
            "computed_position_ids_length_from_full_mask": direct_position_length,
            "shape_mismatch_observed": direct_position_length not in (None, 1),
            "outcome": direct_outcome,
            "error_type": direct_error_type,
            "finite_logits": direct_finite,
            "paired_explicit_one_token_positions": {
                "position_coordinate": expected_native_position,
                "position_ids_length": int(explicit_next_position.shape[-1]),
                "outcome": explicit_cached_outcome,
                "error_type": explicit_cached_error_type,
                "finite_logits": explicit_cached_finite,
                "interpretation": "explicit one-token positions satisfy the observed shape contract only; no semantic cache-position claim",
            },
        },
        "long_interval_native_policy": {
            "second_per_grid_seconds": 5.0,
            "video_grid_thw": list(GRID_T_H_W),
            "native_rope_delta": int(long_native_delta[0, 0].item()),
            "cache_resume_position_from_native_max_plus_one": long_cache_resume,
            "continuous_generate_position_from_last_packed_coordinate_plus_one": long_last_token_next,
            "positions_differ": long_cache_resume != long_last_token_next,
            "interpretation": "fixed synthetic geometry exposes distinct native cache-resume and uninterrupted-generation coordinate policies; neither is endorsed as universally correct",
        },
        "cache_resume_generate": {
            "cleared_delta_decode_position": expected_cleared,
            "native_delta_decode_position": expected_native_resume,
            "candidate_mechanical_delta_decode_position": expected_candidate_resume,
            "stale_delta_decode_position": prompt_length + native_delta_scalar + 5,
            "positions_after_generation_slice": {
                "cleared": list(cleared_resume_caps[0].shape),
                "native_delta": list(native_resume_caps[0].shape),
                "candidate_delta": list(candidate_resume_caps[0].shape),
                "stale_delta": list(stale_resume_caps[0].shape),
            },
            "one_token_positions_after_generation_slice": True,
            "finite_scores_all_cases": True,
            "interpretation": "different continuation coordinates expose state policy; max-based candidate is not claimed semantically correct",
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "device": "cpu",
            "dtype": "float32",
            "model": "random tiny Qwen2.5-VL configuration",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new JSON path under results/")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output = (repo / args.output).resolve()
    if not output.is_relative_to((repo / "results").resolve()) or output.exists():
        parser.error("exclusive new output under results/ required; overwrite refused")

    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    from transformers import generation
    generation_utils = __import__("transformers.generation.utils", fromlist=["utils"])
    files = {
        "runner": Path(__file__).resolve(),
        "geometry_helper": repo / "scripts" / "qwen_temporal_geometry.py",
        "installed_qwen_model": Path(modeling_qwen2_5_vl.__file__).resolve(),
        "installed_generation_utils": Path(generation_utils.__file__).resolve(),
    }
    result = run_diagnostic()
    result["source_commit"] = _source_commit(repo)
    result["source_sha256"] = {key: _sha256(path) for key, path in files.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.chmod(output, 0o600)
    print(json.dumps({
        "status": "ok",
        "output": str(output.relative_to(repo)),
        "fixture": result["fixture"],
        "native_delta": result["position_geometry"]["native_rope_delta"],
        "candidate_delta": result["position_geometry"]["candidate_mechanical_rope_delta"],
        "direct_cached_forward": result["direct_cached_forward"]["outcome"],
        "generate_noop_equal": result["uninterrupted_generate"]["native_auto_vs_explicit_native_scores_bitwise_equal"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
