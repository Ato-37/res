#!/usr/bin/env python3
"""Verify a local Minerva VISTA-style smoke and write a privacy-safe summary.

This verifier reads only mode-0600 local receipts and current source files. Its
summary omits all raw text, video/row IDs, hashes that identify a specific row,
and absolute paths. The one-row likelihood contrasts are descriptive only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"
RESULTS = ROOT / "results"
DEFAULTS = {
    "gpu": PRIVATE / "minerva_vista_residual_oldrow0_gpu_20261022_01.json",
    "cpu": PRIVATE / "minerva_vista_residual_oldrow0_cpu_20261022_03.json",
    "prior_gpu": PRIVATE / "minerva_visible_object_v2_generation_bias_row0_gpu_20261022_01.json",
    "prior_cpu": PRIVATE / "minerva_visible_object_v2_fresh_row0_cpu_20261022_01.json",
    "prior_preflight": PRIVATE / "minerva_visible_object_v2_cohort_cpu_preflight_20261022_01.json",
    "cohort": PRIVATE / "minerva_metadata_proxy_anchor_cohort_v2_20261022_03.json",
    "preflight": PRIVATE / "minerva_visible_object_v2_cohort_cpu_preflight_20261022_02.json",
}
GPU_SOURCE_FILES = (
    "run_minerva_vista_residual_smoke.py",
    "run_minerva_generated_object_pixel_pilot.py",
    "minerva_object_pixel_utils.py",
    "minerva_ego_anchor_utils.py",
    "preflight_minerva_grounding_cohort.py",
    "run_minerva_ego_anchor_gpu_pilot.py",
    "run_claim_support_qta_16frame_exploratory.py",
    "qta_multiquery_video_key_intervention.py",
)
PREFLIGHT_SOURCE_FIELDS = {
    "preflight_source_sha256": "preflight_minerva_grounding_cohort.py",
    "decoder_source_sha256": "run_minerva_ego_anchor_gpu_pilot.py",
    "generated_runner_source_sha256": "run_minerva_ego_generated_grounding_pilot.py",
    "bbox_mapper_source_sha256": "minerva_ego_anchor_utils.py",
}
ABS_TOL = 1e-8


class VerificationError(ValueError):
    """A failed provenance, arithmetic, or protocol assertion."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise VerificationError(f"{label}_not_finite_number")
    return float(value)


def _eq_number(left: Any, right: Any, label: str, *, tolerance: float = ABS_TOL) -> float:
    a, b = _finite(left, label), _finite(right, label)
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance):
        raise VerificationError(f"{label}_arithmetic_mismatch")
    return a


def _private_path(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise VerificationError("private_input_missing_or_symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(PRIVATE.resolve()):
        raise VerificationError("private_input_outside_root")
    return resolved


def _read_private(path: Path, max_bytes: int = 50_000_000) -> tuple[bytes, dict]:
    resolved = _private_path(path)
    fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o600 or before.st_size > max_bytes):
            raise VerificationError("private_receipt_mode_or_size_invalid")
        blocks, total = [], 0
        while True:
            block = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
            if not block:
                break
            blocks.append(block)
            total += len(block)
            if total > max_bytes:
                raise VerificationError("private_receipt_too_large")
        after = os.fstat(fd)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or total != before.st_size):
            raise VerificationError("private_receipt_changed_while_reading")
    finally:
        os.close(fd)
    raw = b"".join(blocks)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise VerificationError("private_receipt_json_invalid") from None
    if not isinstance(value, dict):
        raise VerificationError("private_receipt_root_invalid")
    return raw, value


def _record_for_hash(receipt: dict, row_hash: str) -> dict:
    records = receipt.get("records")
    if not isinstance(records, list):
        raise VerificationError("preflight_records_missing")
    matches = [r for r in records if isinstance(r, dict) and r.get("row_key_sha256") == row_hash]
    if len(matches) != 1 or matches[0].get("status") != "pass":
        raise VerificationError("row_preflight_not_unique_and_passing")
    return matches[0]


def _manifest_source_checks(receipt: dict, *, label: str,
                            expected_names: tuple[str, ...] | None = None) -> dict:
    manifest = receipt.get("source_sha256")
    if not isinstance(manifest, dict) or not manifest:
        raise VerificationError(f"{label}_source_manifest_missing")
    names = tuple(manifest.keys())
    if expected_names is not None and set(names) != set(expected_names):
        raise VerificationError(f"{label}_source_manifest_file_set_mismatch")
    checks = {}
    for name, expected in manifest.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise VerificationError(f"{label}_source_manifest_entry_invalid")
        path = ROOT / "scripts" / name
        if not path.is_file() or path.is_symlink() or sha256(path.read_bytes()) != expected:
            raise VerificationError(f"{label}_source_hash_mismatch")
        checks[name] = True
    runner_hash = receipt.get("runner_sha256")
    if runner_hash is not None:
        runner_manifest_hash = manifest.get("run_minerva_vista_residual_smoke.py")
        if runner_manifest_hash is not None and runner_hash != runner_manifest_hash:
            raise VerificationError(f"{label}_runner_hash_mismatch")
    return checks


def _check_preflight_sources(preflight: dict, label: str) -> None:
    for field, name in PREFLIGHT_SOURCE_FIELDS.items():
        expected = preflight.get(field)
        path = ROOT / "scripts" / name
        if (not isinstance(expected, str) or not path.is_file() or path.is_symlink()
                or sha256(path.read_bytes()) != expected):
            raise VerificationError(f"{label}_source_hash_mismatch")


def _verify_hook_audit(gpu: dict) -> None:
    capture = gpu.get("capture_hook_audit")
    score_hooks = gpu.get("score_hook_audit")
    if not isinstance(capture, dict) or set(capture) != {
            "baseline", "anchor_pair_mean_fill", "temporal_sham_pair_mean_fill"}:
        raise VerificationError("capture_hook_condition_set_invalid")
    for value in capture.values():
        if (value.get("hook_call_count") != 1
                or value.get("hook_registry_before") != value.get("hook_registry_after_removal")):
            raise VerificationError("capture_hook_count_or_cleanup_invalid")
    if not isinstance(score_hooks, dict) or set(score_hooks) != {"anchor_steering", "temporal_sham_steering"}:
        raise VerificationError("score_hook_condition_set_invalid")
    for value in score_hooks.values():
        if (value.get("hook_call_count") != 1
                or value.get("hook_registry_before") != value.get("hook_registry_after_removal")):
            raise VerificationError("score_hook_count_or_cleanup_invalid")


def _verify_dose_audit(gpu: dict) -> dict:
    dose = gpu.get("vector_dose_audit")
    score_hooks = gpu.get("score_hook_audit")
    if not isinstance(dose, dict) or set(dose) != {
            "anchor_steering", "temporal_sham_steering", "anchor_sham_cosine_raw"}:
        raise VerificationError("vector_dose_condition_set_invalid")
    full_rms_values, effective_fractions, target_rms = [], {}, {}
    for key in ("anchor_steering", "temporal_sham_steering"):
        values = dose[key]
        full_rms = _finite(values.get("full_residual_rms_fp32"), "full_residual_rms")
        raw_rms = _finite(values.get("raw_full_minus_filled_rms_fp32"), "raw_delta_rms")
        normalized = _finite(values.get("normalized_delta_rms_fp32"), "normalized_delta_rms")
        fraction = _finite(values.get("dose_fraction_of_full_residual_rms"), "dose_fraction")
        target = _finite(values.get("target_delta_rms_fp32"), "target_delta_rms")
        if full_rms <= 0 or raw_rms <= 0 or not math.isclose(fraction, 0.1, abs_tol=1e-12):
            raise VerificationError("vector_dose_zero_or_wrong_fraction")
        _eq_number(target, 0.1 * full_rms, "target_delta_rms")
        _eq_number(normalized, target, "normalized_delta_rms")
        hook = score_hooks[key]
        effective_rms = _finite(hook.get("effective_displacement_rms_fp32"), "effective_displacement_rms")
        effective_fraction = _finite(hook.get("effective_displacement_fraction_of_pre_rms"), "effective_dose_fraction")
        pre_update_rms = _eq_number(hook.get("pre_update_rms_fp32"), full_rms,
            "hook_pre_update_rms", tolerance=2e-6)
        _eq_number(effective_fraction, effective_rms / pre_update_rms,
            "effective_bf16_fraction", tolerance=1e-8)
        if effective_rms <= 0 or abs(effective_fraction - 0.1) > 0.002:
            raise VerificationError("effective_bf16_dose_out_of_tolerance")
        effective_fractions[key] = effective_fraction
        target_rms[key] = target
        full_rms_values.append(full_rms)
    _eq_number(full_rms_values[0], full_rms_values[1], "shared_baseline_rms", tolerance=2e-6)
    cosine = _finite(dose.get("anchor_sham_cosine_raw"), "anchor_sham_cosine")
    if not -1.0 <= cosine <= 1.0:
        raise VerificationError("anchor_sham_cosine_out_of_range")
    return {"effective_fractions": effective_fractions, "target_rms": target_rms,
            "raw_anchor_sham_cosine": cosine}


def _verify_score_arithmetic(gpu: dict) -> dict:
    scores = gpu.get("mean_object_name_logprob_nats_per_token")
    if not isinstance(scores, dict) or set(scores) != {"baseline", "anchor_steering", "temporal_sham_steering"}:
        raise VerificationError("score_condition_set_invalid")
    baseline = _finite(scores["baseline"], "baseline_score")
    anchor = _finite(scores["anchor_steering"], "anchor_score")
    sham = _finite(scores["temporal_sham_steering"], "sham_score")
    anchor_delta, sham_delta = anchor - baseline, sham - baseline
    interaction = anchor_delta - sham_delta
    _eq_number(gpu.get("delta_anchor_minus_baseline_nats_per_token"), anchor_delta,
               "anchor_minus_baseline")
    _eq_number(gpu.get("delta_sham_minus_baseline_nats_per_token"), sham_delta,
               "sham_minus_baseline")
    _eq_number(gpu.get("anchor_minus_sham_change_nats_per_token"), interaction,
               "anchor_minus_sham")
    return {"scores": {"baseline": baseline, "anchor_steering": anchor,
                       "temporal_sham_steering": sham},
            "contrasts": {"anchor_minus_baseline": anchor_delta,
                          "temporal_sham_minus_baseline": sham_delta,
                          "anchor_change_minus_sham_change": interaction}}


def _assert_privacy(summary: dict, private_literals: tuple[str, ...]) -> None:
    serialized = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    for literal in private_literals:
        if literal and literal in serialized:
            raise VerificationError("privacy_literal_leak_detected")


def _summary_from_receipts(*, gpu: dict, cpu: dict, prior_gpu: dict,
                           prior_cpu: dict, prior_preflight: dict,
                           cohort: dict, preflight: dict,
                           receipt_hashes: dict[str, str],
                           gpu_source_checks: dict, cpu_source_checks: dict,
                           prior_cpu_source_checks: dict,
                           private_literals: tuple[str, ...]) -> dict:
    # Primary receipt and protocol status.
    if gpu.get("status") != "completed_vista_residual_one_row_gpu_technical_smoke":
        raise VerificationError("gpu_receipt_not_completed")
    if cpu.get("status") != "completed_vista_residual_cpu_preflight":
        raise VerificationError("cpu_receipt_not_completed")
    if prior_gpu.get("status") != "completed_minerva_visible_object_generation_bias_v1":
        raise VerificationError("prior_generation_receipt_not_completed")
    if prior_cpu.get("status") != "completed_minerva_generated_object_pixel_cpu_preflight":
        raise VerificationError("prior_cpu_receipt_not_completed")
    if (gpu.get("human_review_performed") is not False
            or gpu.get("direct_generated_explanation_faithfulness") is not False
            or gpu.get("rights_status") != "unverified; local-only per user authorization"):
        raise VerificationError("gpu_scope_or_rights_declaration_invalid")

    # Exact receipt, frozen cohort, source, and prior-generation binding chain.
    if (gpu.get("cohort_sha256") != cpu.get("cohort_sha256")
            or gpu.get("cohort_cpu_preflight_sha256") != cpu.get("cohort_cpu_preflight_sha256")
            or gpu.get("cpu_receipt_sha256") != receipt_hashes["cpu"]
            or gpu.get("prior_baseline_receipt_sha256") != receipt_hashes["prior_gpu"]
            or cpu.get("prior_baseline_receipt_sha256") != receipt_hashes["prior_gpu"]):
        raise VerificationError("gpu_cpu_prior_receipt_join_mismatch")
    if (gpu.get("cohort_sha256") != receipt_hashes["cohort"]
            or gpu.get("cohort_cpu_preflight_sha256") != receipt_hashes["preflight"]
            or prior_gpu.get("cohort_sha256") != gpu.get("cohort_sha256")
            or prior_gpu.get("cohort_preflight_sha256") != receipt_hashes["prior_preflight"]
            or prior_gpu.get("cpu_receipt_sha256") != receipt_hashes["prior_cpu"]
            or prior_cpu.get("cohort_sha256") != gpu.get("cohort_sha256")
            or prior_cpu.get("cohort_cpu_preflight_sha256") != receipt_hashes["prior_preflight"]):
        raise VerificationError("cohort_or_prior_preflight_join_mismatch")
    if (preflight.get("status") != "completed_frozen_cohort_cpu_preflight"
            or prior_preflight.get("status") != "completed_frozen_cohort_cpu_preflight"
            or preflight.get("cohort_sha256") != gpu.get("cohort_sha256")
            or prior_preflight.get("cohort_sha256") != gpu.get("cohort_sha256")
            or cohort.get("status") != "frozen_pre_outcome_exploratory_cohort"):
        raise VerificationError("frozen_cohort_preflight_join_mismatch")
    if not isinstance(cohort.get("selection"), list) or len(cohort["selection"]) < 1:
        raise VerificationError("frozen_cohort_selection_missing")
    row = cohort["selection"][0]
    row_hash = row.get("row_key_sha256")
    if (not isinstance(row_hash, str) or gpu.get("row_key_sha256") != row_hash
            or cpu.get("row_key_sha256") != row_hash
            or prior_gpu.get("row_key_sha256") != row_hash
            or prior_cpu.get("row_key_sha256") != row_hash
            or prior_cpu.get("row_index") != 0 or gpu.get("row_index") != 0
            or cpu.get("row_index") != 0):
        raise VerificationError("row_zero_binding_mismatch")
    current_row = _record_for_hash(preflight, row_hash)
    prior_row = _record_for_hash(prior_preflight, row_hash)
    for receipt in (gpu, cpu, prior_gpu, prior_cpu):
        for key in ("video_sha256", "time_map_sha256"):
            if receipt.get(key) != current_row.get(key) or receipt.get(key) != prior_row.get(key):
                raise VerificationError("row_media_hash_join_mismatch")
    if (prior_gpu.get("prompt_sha256") != cpu.get("rendered_prompt_sha256")
            or prior_cpu.get("rendered_prompt_sha256") != cpu.get("rendered_prompt_sha256")
            or gpu.get("target_token_ids_sha256") != prior_gpu.get("outcomes", {}).get("baseline", {}).get("generated_token_ids_sha256")
            or gpu.get("target_text_sha256") != prior_gpu.get("outcomes", {}).get("baseline", {}).get("generated_text_sha256")
            or prior_gpu.get("outcomes", {}).get("baseline", {}).get("mechanical_diagnostic", {}).get("status") != "mechanically_valid"):
        raise VerificationError("prior_baseline_prompt_or_generation_join_mismatch")
    if (gpu.get("model_revision") != cpu.get("model_revision")
            or gpu.get("model_revision") != prior_gpu.get("model_revision")):
        raise VerificationError("model_revision_mismatch")
    if gpu.get("source_sha256") != cpu.get("source_sha256"):
        raise VerificationError("gpu_cpu_source_manifest_mismatch")
    runner_source = (ROOT / "scripts" / "run_minerva_vista_residual_smoke.py").read_text()
    if ("if args.row_index != 0:" not in runner_source
            or "technical_smoke_is_frozen_to_row0" not in runner_source
            or gpu.get("row_index") != 0 or cpu.get("row_index") != 0):
        raise VerificationError("row_zero_only_runner_contract_missing")

    # Fixed protocol and target/query alignment.
    if (gpu.get("frame_count") != 64 or cpu.get("frame_count") != 64
            or gpu.get("video_grid_thw") != cpu.get("video_grid_thw")
            or gpu.get("video_grid_thw") != [32, 22, 22]
            or gpu.get("fixed_block_1_based") != 14
            or gpu.get("fixed_block_index_zero_based") != 13
            or cpu.get("fixed_block_1_based") != 14
            or cpu.get("fixed_block_index_zero_based") != 13
            or not math.isclose(_finite(gpu.get("dose_rms_fraction"), "dose_fraction"), 0.10,
                                rel_tol=0.0, abs_tol=1e-12)
            or cpu.get("dose_rms_fraction") != gpu.get("dose_rms_fraction")):
        raise VerificationError("fixed_intervention_protocol_mismatch")
    prompt_len, target_len = gpu.get("prompt_token_count"), gpu.get("target_token_count")
    name_count, indices = gpu.get("object_name_token_count"), gpu.get("object_name_token_indices")
    query_positions = gpu.get("teacher_forcing_query_positions")
    target_queries = gpu.get("object_name_target_query_positions")
    if (type(prompt_len) is not int or type(target_len) is not int or target_len < 1
            or type(name_count) is not int or name_count < 1
            or not isinstance(indices, list) or len(indices) != name_count
            or any(type(i) is not int or not 0 <= i < target_len for i in indices)
            or query_positions != list(range(prompt_len - 1, prompt_len + target_len - 1))
            or target_queries != [prompt_len + i - 1 for i in indices]
            or target_len != prior_gpu.get("outcomes", {}).get("baseline", {}).get("generated_token_count")):
        raise VerificationError("target_or_teacher_forcing_position_mismatch")

    # Hook call counts, hook cleanup, condition set, and dose arithmetic.
    _verify_hook_audit(gpu)
    dose_result = _verify_dose_audit(gpu)
    score_result = _verify_score_arithmetic(gpu)
    model_calls = gpu.get("model_call_counts", {})
    tf_calls = gpu.get("teacher_forced_forward_counts", {})
    if (model_calls.get("generate_api_invocations") != 1
            or model_calls.get("explicit_teacher_forced_model_forward_calls") != 5
            or model_calls.get("top_level_model_invocations") != 6
            or model_calls.get("generation_internal_decoder_forward_count") != "not instrumented"
            or tf_calls.get("total") != 5
            or sum(tf_calls.get(k, 0) for k in ("baseline_capture_and_score", "anchor_mask_capture",
                "sham_mask_capture", "anchor_steered_score", "sham_steered_score")) != 5):
        raise VerificationError("model_call_count_audit_invalid")

    # Textual privacy fence. No raw IDs, strings, paths, or token IDs are copied.
    summary = {
        "schema_version": 1,
        "status": "verified_minerva_vista_residual_one_row_technical_smoke",
        "verification": {
            "provenance_chain": "passed",
            "gpu_cpu_current_source_manifest": f"{len(gpu_source_checks)}/{len(GPU_SOURCE_FILES)} files matched",
            "prior_cpu_current_source_manifest": f"{len(prior_cpu_source_checks)}/{len(prior_cpu_source_checks)} files matched",
            "gpu_to_cpu_and_prior_receipts": "passed",
            "cohort_and_both_official_preflights": "passed",
            "row_media_hash_joins": "passed",
            "prior_greedy_generation_ids_text_and_prompt_hashes": "passed",
            "score_arithmetic_recalculated": "passed",
            "hook_count_and_removal_checks": "passed",
            "dose_arithmetic_recalculated": "passed",
            "effective_dose_tolerance_screen": "post-hoc diagnostic only; absolute fraction tolerance 0.002; not a preregistered gate",
        },
        "counts": {
            "requested_rows": 1,
            "effective_rows": 1,
            "failed_rows": 0,
            "count_basis": "row-0-only runner contract and completed one-row receipt; the GPU receipt itself omits explicit requested/effective counters",
        },
        "experiment": {
            "dataset": "Minerva-Ego / HD-EPIC",
            "model_revision": gpu["model_revision"],
            "gpu": gpu.get("runtime_versions", {}).get("gpu_name"),
            "torch_version": gpu.get("runtime_versions", {}).get("torch"),
            "cuda_version": gpu.get("runtime_versions", {}).get("cuda"),
            "precision": "4-bit weights; BF16 compute/activation",
            "frames": gpu["frame_count"],
            "video_grid_thw": gpu["video_grid_thw"],
            "prompt_token_count": prompt_len,
            "generated_target_token_count": target_len,
            "scored_object_name_token_count": name_count,
            "fixed_decoder_block_1_based": gpu["fixed_block_1_based"],
            "dose_rms_fraction": 0.10,
            "anchor_and_temporal_sham_each_replaced_adjacent_frames": 2,
        },
        "scores_mean_object_name_logprob_nats_per_token": {
            "baseline": score_result["scores"]["baseline"],
            "anchor_steering": score_result["scores"]["anchor_steering"],
            "temporal_sham_steering": score_result["scores"]["temporal_sham_steering"],
        },
        "contrasts_nats_per_token": {
            **score_result["contrasts"],
        },
        "dose_audit": {
            "requested_fraction_of_baseline_residual_rms": 0.10,
            "anchor_target_delta_rms": dose_result["target_rms"]["anchor_steering"],
            "sham_target_delta_rms": dose_result["target_rms"]["temporal_sham_steering"],
            "anchor_effective_bf16_fraction": dose_result["effective_fractions"]["anchor_steering"],
            "sham_effective_bf16_fraction": dose_result["effective_fractions"]["temporal_sham_steering"],
            "raw_anchor_sham_vector_cosine": dose_result["raw_anchor_sham_cosine"],
        },
        "runtime": {
            "seconds": _finite(gpu.get("elapsed_seconds"), "elapsed_seconds"),
            "peak_allocated_vram_gib": _finite(gpu.get("peak_vram_bytes"), "peak_vram") / 1024**3,
        },
        "inferential_status": {
            "sample_size": 1,
            "statistical_test_or_interval": "not applicable; single-row engineering smoke",
            "significance_or_generalization_claimed": False,
            "direct_faithfulness_established": False,
            "human_review_performed": False,
            "rights_status": "unverified; local-only per user authorization",
        },
        "limitations": [
            "one-row technical smoke; no efficacy or statistical inference",
            "object-name-token likelihood is prediction dependence, not claim truth or semantic support",
            "no human review; direct explanation faithfulness is not established",
            "whole-pair global mean-fill may produce out-of-distribution inputs",
            "farthest-endpoint sham is an extreme temporal control and is not representative or exchangeable",
            "generation's internal decoder-forward count was not instrumented",
        ],
    }
    _assert_privacy(summary, private_literals)
    return summary


def verify(args) -> dict:
    raw = {}
    data = {}
    for key, path in (("gpu", args.gpu_receipt), ("cpu", args.cpu_receipt),
        ("prior_gpu", args.prior_baseline_receipt), ("prior_cpu", args.prior_cpu_receipt),
        ("prior_preflight", args.prior_preflight), ("cohort", args.cohort),
        ("preflight", args.preflight)):
        raw[key], data[key] = _read_private(path)
    gpu_source = _manifest_source_checks(data["gpu"], label="gpu", expected_names=GPU_SOURCE_FILES)
    cpu_source = _manifest_source_checks(data["cpu"], label="cpu", expected_names=GPU_SOURCE_FILES)
    prior_cpu_source = _manifest_source_checks(data["prior_cpu"], label="prior_cpu")
    if gpu_source != cpu_source:
        raise VerificationError("gpu_cpu_source_hashes_differ")
    _check_preflight_sources(data["preflight"], "current_preflight")
    _check_preflight_sources(data["prior_preflight"], "prior_preflight")
    row = data["cohort"].get("selection", [{}])[0]
    private_literals = (
        str(row.get("video_id", "")), str(row.get("row_key", "")),
        str(data["gpu"].get("row_key_sha256", "")), str(data["gpu"].get("video_sha256", "")),
        str(data["gpu"].get("time_map_sha256", "")),
        str(data["gpu"].get("target_text_sha256", "")),
        str(data["gpu"].get("target_token_ids_sha256", "")),
        str(PRIVATE.resolve()),
    )
    hashes = {key: sha256(value) for key, value in raw.items()}
    return _summary_from_receipts(gpu=data["gpu"], cpu=data["cpu"],
        prior_gpu=data["prior_gpu"], prior_cpu=data["prior_cpu"],
        prior_preflight=data["prior_preflight"], cohort=data["cohort"],
        preflight=data["preflight"], receipt_hashes=hashes,
        gpu_source_checks=gpu_source, cpu_source_checks=cpu_source,
        prior_cpu_source_checks=prior_cpu_source, private_literals=private_literals)


def _write_results(path: Path, summary: dict) -> None:
    if RESULTS.is_symlink() or not RESULTS.is_dir():
        raise VerificationError("results_directory_invalid")
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(RESULTS.resolve()) or resolved.parent != RESULTS.resolve():
        raise VerificationError("summary_output_must_be_direct_results_child")
    if path.exists() or path.is_symlink():
        raise VerificationError("summary_output_exists_or_symlink")
    payload = (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            n = os.write(fd, view)
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-receipt", type=Path, default=DEFAULTS["gpu"])
    parser.add_argument("--cpu-receipt", type=Path, default=DEFAULTS["cpu"])
    parser.add_argument("--prior-baseline-receipt", type=Path, default=DEFAULTS["prior_gpu"])
    parser.add_argument("--prior-cpu-receipt", type=Path, default=DEFAULTS["prior_cpu"])
    parser.add_argument("--prior-preflight", type=Path, default=DEFAULTS["prior_preflight"])
    parser.add_argument("--cohort", type=Path, default=DEFAULTS["cohort"])
    parser.add_argument("--preflight", type=Path, default=DEFAULTS["preflight"])
    parser.add_argument("--output", type=Path,
        default=RESULTS / "minerva_vista_residual_smoke_summary_20261022_01.json")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise VerificationError("summary_output_exists_or_symlink")
    summary = verify(args)
    _write_results(args.output, summary)
    print(json.dumps({"status": summary["status"],
        "provenance_chain": summary["verification"]["provenance_chain"],
        "requested_rows": summary["counts"]["requested_rows"],
        "effective_rows": summary["counts"]["effective_rows"],
        "anchor_minus_sham_change_nats_per_token":
            summary["contrasts_nats_per_token"]["anchor_change_minus_sham_change"],
        "summary": str(args.output)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        raise SystemExit(f"verification_failed:{exc}") from None
