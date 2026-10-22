#!/usr/bin/env python3
"""Score the frozen synthetic transient-event QTA study (local GPU only).

Modes are intentionally separate: ``smoke`` scores the reserved external seed
only (two clips, 18 forwards); ``full`` requires a passing, hash-bound all-
cohort CPU processor preflight and a passing smoke receipt (80 clips, 720
forwards). Receipts contain aggregate sufficient statistics only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import time
from typing import Any, Callable

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
LAYERS_1_BASED = tuple(range(5, 17))
LOGIT_BIAS = math.log(2.0)
SHAM_SEED = "qta-sham-seed-20261018-v1"
MIN_FREE_BEFORE_LOAD = int(13.5 * 1024**3)
MIN_FREE_BEFORE_FORWARD = 3 * 1024**3
MIN_HOST_AVAILABLE_BEFORE_LOAD = 8 * 1024**3
MIN_HOST_AVAILABLE_BEFORE_FORWARD = 3 * 1024**3
PRIVATE_RELATIVE = Path("data/private_nextgqa")
PROTOCOL_RELATIVE = Path("experiments/2026-10-19_synthetic_event_qta_construct_validation_v1.md")
PREFLIGHT_RELATIVE = Path("scripts/run_qta_synthetic_event_cohort_preflight.py")
FIXTURE_RELATIVE = Path("scripts/qta_synthetic_event_construct_v1.py")
SOURCE_RELATIVES = (
    PROTOCOL_RELATIVE,
    PREFLIGHT_RELATIVE,
    FIXTURE_RELATIVE,
    Path("scripts/qwen_question_temporal_attention_selector.py"),
    Path("scripts/qta_attention_forward_capture.py"),
    Path("scripts/qta_attention_capture_contract.py"),
    Path("scripts/qta_attention_capture_sidecars.py"),
    Path("scripts/qwen_internal_attention_ablation.py"),
    Path("scripts/qwen_visual_attention_boost.py"),
    Path("scripts/qta_choice_token_binding.py"),
    Path("scripts/run_nextgqa_gold_interval_margin_pilot.py"),
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(repo_root: Path) -> dict[str, str]:
    hashes = {path.as_posix(): sha256_file(repo_root / path)
              for path in SOURCE_RELATIVES}
    hashes["scripts/run_qta_synthetic_event_construct_gpu.py"] = sha256_file(
        Path(__file__).resolve())
    return hashes


def _safe_private_json(path: Path, private_root: Path) -> tuple[dict[str, Any], str]:
    absolute = Path(os.path.abspath(path))
    root = Path(os.path.abspath(private_root))
    if absolute.parent != root:
        raise ValueError("receipt_must_be_direct_child_of_private_directory")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(absolute, flags)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("receipt_file_permissions_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("receipt_schema_invalid")
    return payload, _sha_bytes(raw)


def validate_preflight_receipt(receipt: Mapping[str, Any], *,
                               expected_protocol_sha256: str,
                               expected_source_sha256: Mapping[str, str],
                               expected_assignment_sha256: str,
                               mode: str) -> None:
    if mode not in {"smoke", "full"}:
        raise ValueError("run_mode_invalid")
    if (not isinstance(receipt, Mapping)
            or receipt.get("schema_version") != 1
            or receipt.get("status") != "passed_cpu_processor_preflight_no_model_forward"
            or receipt.get("purpose") != "qta_synthetic_event_construct_processor_preflight"
            or receipt.get("scope") != "synthetic_processor_geometry_only_no_qta_overlap"
            or receipt.get("requested_pairs") != 40
            or receipt.get("effective_pairs") != 40
            or receipt.get("requested_clips") != 80
            or receipt.get("effective_clips") != 80
            or receipt.get("model_revision") != MODEL_REVISION
            or receipt.get("model_weights_loaded") is not False
            or receipt.get("gpu_used") is not False
            or receipt.get("model_forwards") != 0
            or receipt.get("qta_selector_overlap_measured") is not False
            or receipt.get("faithfulness_result") is not False
            or receipt.get("authorization") != "processor_preflight_only_no_scoring_authorization"
            or receipt.get("protocol_sha256") != expected_protocol_sha256
            or receipt.get("source_sha256") != expected_source_sha256
            or receipt.get("assignment_manifest_sha256") != expected_assignment_sha256):
        raise ValueError(f"{mode}_cpu_preflight_binding_invalid")
    checks = receipt.get("checks")
    expected_checks = {
        "paired_rendered_qa_equal": 40,
        "paired_choice_lists_equal": 40,
        "paired_processor_geometry_equal": 40,
        "clips_with_exactly_two_event_groups": 80,
        "clips_with_two_disjoint_nearest_controls": 80,
    }
    if checks != expected_checks:
        raise ValueError(f"{mode}_cpu_preflight_checks_invalid")
    balance = receipt.get("design_balance")
    if (not isinstance(balance, Mapping)
            or [(row.get("start_frame"), row.get("pairs"))
                for row in balance.get("onset_pair_counts", ())]
            != [(4, 10), (10, 10), (18, 10), (24, 10)]
            or [(row.get("start_frame"), row.get("circle_left"), row.get("circle_right"))
                for row in balance.get("orientation_pair_counts_by_onset", ())]
            != [(4, 5, 5), (10, 5, 5), (18, 5, 5), (24, 5, 5)]
            or balance.get("sampled_cohort_seeds_unique") != 40
            or balance.get("orientation_rotation_combinations_per_onset") != 10
            or balance.get("circle_target_correct_label_counts") != {key: 8 for key in "ABCDE"}
            or balance.get("triangle_target_correct_label_counts") != {key: 8 for key in "ABCDE"}):
        raise ValueError(f"{mode}_cpu_preflight_balance_invalid")
    geometry = receipt.get("aggregate_processor_geometry")
    grid_counts = geometry.get("grid_counts") if isinstance(geometry, Mapping) else None
    if (not isinstance(grid_counts, list) or len(grid_counts) != 1
            or not isinstance(grid_counts[0], Mapping)
            or grid_counts[0].get("video_grid_thw", [None])[0] != 16
            or grid_counts[0].get("clips") != 80):
        raise ValueError(f"{mode}_cpu_preflight_geometry_invalid")


def validate_smoke_receipt(receipt: Mapping[str, Any], *,
                           expected_protocol_sha256: str,
                           expected_sources: Mapping[str, str],
                           expected_assignment_sha256: str,
                           expected_checkpoint_sha256: str,
                           expected_model_config_sha256: str,
                           expected_smoke_seed_sha256: str,
                           expected_preflight_receipt_sha256: str,
                           expected_source_commit: str) -> None:
    if (not isinstance(receipt, Mapping)
            or receipt.get("schema_version") != 1
            or receipt.get("status") != "passed_external_engineering_smoke_only"
            or receipt.get("mode") != "smoke"
            or receipt.get("requested_pairs") != 1
            or receipt.get("effective_pairs") != 1
            or receipt.get("requested_clips") != 2
            or receipt.get("effective_clips") != 2
            or receipt.get("requested_forwards") != 18
            or receipt.get("completed_forwards") != 18
            or receipt.get("attempted_forwards") != 18
            or receipt.get("failure_count") != 0
            or receipt.get("model_revision") != MODEL_REVISION
            or receipt.get("protocol_sha256") != expected_protocol_sha256
            or receipt.get("source_sha256") != expected_sources
            or receipt.get("assignment_manifest_sha256") != expected_assignment_sha256
            or receipt.get("checkpoint_sha256") != expected_checkpoint_sha256
            or receipt.get("model_config_sha256") != expected_model_config_sha256
            or receipt.get("external_smoke_seed_sha256") != expected_smoke_seed_sha256
            or receipt.get("preflight_receipt_sha256") != expected_preflight_receipt_sha256
            or receipt.get("source_commit") != expected_source_commit
            or receipt.get("model_weights_loaded") is not True
            or receipt.get("gpu_used") is not True
            or receipt.get("cohort_included") is not False
            or receipt.get("faithfulness_result") is not False):
        raise ValueError("full_mode_external_smoke_binding_invalid")


def fixed_gold_foil_margin(values: Sequence[float], *, gold_index: int,
                           foil_index: int) -> float:
    if (not isinstance(values, Sequence) or isinstance(values, (str, bytes))
            or len(values) != 5
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in values)
            or type(gold_index) is not int or type(foil_index) is not int
            or not 0 <= gold_index < 5 or not 0 <= foil_index < 5
            or gold_index == foil_index):
        raise ValueError("fixed_gold_foil_input_invalid")
    return float(values[gold_index]) - float(values[foil_index])


def baseline_foil_index(values: Sequence[float], *, gold_index: int) -> int:
    if (not isinstance(values, Sequence) or isinstance(values, (str, bytes))
            or len(values) != 5 or type(gold_index) is not int
            or not 0 <= gold_index < 5
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in values)):
        raise ValueError("baseline_choice_scores_invalid")
    choices = [index for index in range(5) if index != gold_index]
    return min(choices, key=lambda index: (-float(values[index]), index))


def clip_primary_delta(branch_scores: Mapping[str, Sequence[float]], *,
                       gold_index: int) -> tuple[float, float]:
    """Return QTA DeltaF and sham DeltaF using baseline-full frozen foil."""
    required = {
        "baseline_full", "baseline_evidence_ablation", "baseline_control_ablation",
        "qta_full", "qta_evidence_ablation", "qta_control_ablation",
        "sham_full", "sham_evidence_ablation", "sham_control_ablation",
    }
    if not isinstance(branch_scores, Mapping) or set(branch_scores) != required:
        raise ValueError("nine_branch_score_set_invalid")
    foil = baseline_foil_index(branch_scores["baseline_full"], gold_index=gold_index)
    margins = {name: fixed_gold_foil_margin(values, gold_index=gold_index,
                                             foil_index=foil)
               for name, values in branch_scores.items()}
    baseline_f = (margins["baseline_control_ablation"]
                  - margins["baseline_evidence_ablation"])
    qta_f = margins["qta_control_ablation"] - margins["qta_evidence_ablation"]
    sham_f = margins["sham_control_ablation"] - margins["sham_evidence_ablation"]
    return qta_f - baseline_f, sham_f - baseline_f


def run_nine_branch_adapter(*, gold_index: int, evidence_positions: Sequence[int],
                            control_positions: Sequence[int],
                            temporal_groups: Sequence[Sequence[int]],
                            opaque_token: str,
                            forward: Callable[..., Any]) -> dict[str, Any]:
    """Testable protocol adapter; forward returns (A-E scores, Q-group masses).

    Q-group masses are required only for baseline_full. Calls proceed in the
    predeclared nine-branch order. The returned clip-level deltas are intended
    only for immediate in-memory pair aggregation, never artifact storage.
    """
    from qwen_question_temporal_attention_selector import (
        select_hash_sham_temporal_groups, select_top_temporal_groups,
    )
    if (type(gold_index) is not int or not 0 <= gold_index < 5
            or not isinstance(temporal_groups, Sequence) or not temporal_groups
            or any(not isinstance(group, Sequence) or isinstance(group, (str, bytes))
                   or not group or any(type(position) is not int for position in group)
                   for group in temporal_groups)
            or not isinstance(opaque_token, str) or not opaque_token
            or not callable(forward)):
        raise ValueError("nine_branch_adapter_input_invalid")
    groups = tuple(tuple(group) for group in temporal_groups)
    t = len(groups)
    k = max(1, math.ceil(0.10 * t))
    evidence = tuple(evidence_positions)
    control = tuple(control_positions)
    if (len(evidence) != 2 or len(control) != 2
            or tuple(sorted(set(evidence))) != evidence
            or tuple(sorted(set(control))) != control
            or set(evidence) & set(control)
            or any(type(value) is not int or value < 0 or value >= t
                   for value in (*evidence, *control))
            or any(not group or tuple(sorted(set(group))) != group
                   or tuple(group) != tuple(range(group[0], group[-1] + 1))
                   for group in groups)
            or len({len(group) for group in groups}) != 1
            or len({value for group in groups for value in group}) != sum(map(len, groups))):
        raise ValueError("nine_branch_adapter_mask_positions_invalid")
    visual_positions = tuple(position for group in groups for position in group)
    if (not visual_positions
            or visual_positions != tuple(range(visual_positions[0], visual_positions[-1] + 1))):
        raise ValueError("nine_branch_adapter_temporal_partition_invalid")
    evidence_keys = tuple(sorted(position for index in evidence for position in groups[index]))
    control_keys = tuple(sorted(position for index in control for position in groups[index]))

    scores: dict[str, Sequence[float]] = {}
    target_groups: tuple[int, ...] | None = None
    sham_groups = tuple(select_hash_sham_temporal_groups(
        temporal_group_count=t, top_k=k, protocol_seed=SHAM_SEED,
        opaque_record_token=opaque_token))
    branches = (
        ("baseline_full", (), ()),
        ("baseline_evidence_ablation", (), evidence_keys),
        ("baseline_control_ablation", (), control_keys),
        ("qta_full", None, ()),
        ("qta_evidence_ablation", None, evidence_keys),
        ("qta_control_ablation", None, control_keys),
        ("sham_full", tuple(position for index in sham_groups for position in groups[index]), ()),
        ("sham_evidence_ablation", tuple(position for index in sham_groups for position in groups[index]), evidence_keys),
        ("sham_control_ablation", tuple(position for index in sham_groups for position in groups[index]), control_keys),
    )
    for name, bias, ablation in branches:
        if name.startswith("qta_"):
            if target_groups is None:
                raise ValueError("qta_selector_not_frozen_from_baseline_full")
            bias = tuple(position for index in target_groups for position in groups[index])
        result = forward(name=name, biased_positions=tuple(bias or ()),
                         ablated_positions=tuple(ablation),
                         capture_question_attention=(name == "baseline_full"))
        if (not isinstance(result, tuple) or len(result) != 2
                or not isinstance(result[0], (list, tuple)) or len(result[0]) != 5):
            raise ValueError("nine_branch_forward_result_invalid")
        branch_scores = result[0]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in branch_scores):
            raise ValueError("nine_branch_forward_score_nonfinite")
        scores[name] = tuple(float(value) for value in branch_scores)
        if name == "baseline_full":
            masses = result[1]
            if (not isinstance(masses, (list, tuple)) or len(masses) != t
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(float(value)) or float(value) < 0
                           for value in masses)):
                raise ValueError("baseline_q_attention_group_masses_invalid")
            target_groups = tuple(sorted(select_top_temporal_groups(
                tuple(float(value) for value in masses), top_k=k)))
        elif result[1] is not None:
            raise ValueError("unexpected_attention_capture_outside_baseline_full")
    if tuple(scores) != tuple(name for name, _bias, _abl in branches):
        raise ValueError("nine_branch_order_or_completion_invalid")
    qta_delta, sham_delta = clip_primary_delta(scores, gold_index=gold_index)
    return {
        "qta_delta_f": qta_delta,
        "sham_delta_f": sham_delta,
        "target_group_indices": target_groups,
        "sham_group_indices": sham_groups,
        "target_evidence_overlap_count": len(set(target_groups or ()) & set(evidence)),
        "target_control_overlap_count": len(set(target_groups or ()) & set(control)),
        "sham_evidence_overlap_count": len(set(sham_groups) & set(evidence)),
        "sham_control_overlap_count": len(set(sham_groups) & set(control)),
        "forward_count": len(scores),
    }


def _aggregate_pair_deltas(pair_deltas: Sequence[tuple[int, float, float]]) -> dict[str, Any]:
    """Build privacy-safe onset-stratum sufficient statistics; no row scores."""
    strata: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for onset, qta, sham in pair_deltas:
        if (onset not in (4, 10, 18, 24) or not math.isfinite(qta)
                or not math.isfinite(sham)):
            raise ValueError("pair_delta_invalid")
        strata[onset].append((float(qta), float(sham)))
    if set(strata) != {4, 10, 18, 24} or any(len(strata[onset]) != 10 for onset in strata):
        raise ValueError("stratified_pair_denominator_invalid")
    return {
        str(onset): {
            "n_pairs": len(strata[onset]),
            "qta_delta_sum": sum(qta for qta, _ in strata[onset]),
            "qta_delta_sum_squares": sum(qta * qta for qta, _ in strata[onset]),
            "sham_delta_sum": sum(sham for _, sham in strata[onset]),
            "sham_delta_sum_squares": sum(sham * sham for _, sham in strata[onset]),
        }
        for onset in (4, 10, 18, 24)
    }


def _write_private_json(path: Path, payload: Mapping[str, Any], private_root: Path) -> None:
    absolute = Path(os.path.abspath(path))
    root = Path(os.path.abspath(private_root))
    try:
        absolute.relative_to(root)
    except ValueError:
        raise ValueError("output_outside_private_directory") from None
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private_directory_permissions_invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(absolute, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            absolute.unlink()
        except OSError:
            pass
        raise


def _validate_output_path(path: Path, private_root: Path) -> Path:
    root = Path(os.path.abspath(private_root))
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private_directory_permissions_invalid")
    output = Path(os.path.abspath(path))
    if output.parent != root or output.suffix.lower() != ".json":
        raise ValueError("output_must_be_new_json_direct_child")
    if os.path.lexists(output):
        raise ValueError("refusing_to_overwrite_output")
    return output


class ScoringRunError(RuntimeError):
    def __init__(self, code: str, *, attempted: int, completed: int):
        super().__init__(code)
        self.code = code
        self.attempted = attempted
        self.completed = completed


def _host_available_bytes() -> int:
    with open("/proc/meminfo", "r", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("host_available_memory_unavailable")


def _checkpoint_shards(model_path: Path) -> dict[str, dict[str, Any]]:
    shards = sorted(model_path.glob("model-*-of-*.safetensors"))
    if len(shards) != 5:
        raise ValueError("checkpoint_shard_inventory_invalid")
    result = {}
    for path in shards:
        result[path.name] = {"size_bytes": path.stat().st_size,
                             "sha256": sha256_file(path)}
    return result


def _checkpoint_digest(shards: Mapping[str, Mapping[str, Any]]) -> str:
    return _sha_bytes(_canonical(shards))


def _processor_fingerprint(processor: Any) -> str:
    tokenizer_sha = _sha_bytes(processor.tokenizer.backend_tokenizer.to_str().encode("utf-8"))
    descriptor = {
        "processor_class": type(processor).__name__,
        "image_processor": processor.image_processor.to_dict(),
        "tokenizer_sha256": tokenizer_sha,
        "video_token_id": int(processor.video_token_id),
        "model_revision": MODEL_REVISION,
    }
    return _sha_bytes(_canonical(descriptor))


def _event_controls(onset: int) -> tuple[tuple[int, int], tuple[int, int]]:
    evidence = (onset // 2, onset // 2 + 1)
    event_center_twice = onset * 2 + 4
    candidates = sorted((index for index in range(16) if index not in evidence),
                        key=lambda index: (abs((2 * index + 1) * 2 - event_center_twice), index))
    controls = tuple(sorted(candidates[:2]))
    if len(controls) != 2 or set(evidence) & set(controls):
        raise ValueError("event_control_groups_invalid")
    return evidence, controls


def _load_attention_components():
    """Import the selector and capture/aggregation helpers from their owners."""
    from qwen_question_temporal_attention_selector import map_question_token_positions
    from qta_attention_forward_capture import (
        aggregate_captured_qta_rows, capture_qta_attention_forward_once,
    )
    return map_question_token_positions, aggregate_captured_qta_rows, capture_qta_attention_forward_once


def _make_external_design(smoke_seed: int):
    from qta_synthetic_event_construct_v1 import PairDesign, FROZEN_CHOICE_TEXTS, LABELS
    rotation = 0
    choices = tuple(zip(LABELS, FROZEN_CHOICE_TEXTS, strict=True))
    return PairDesign(-1, -1, 14, smoke_seed, True, rotation, choices)


def _preflight_expected_source_subset(hashes: Mapping[str, str]) -> dict[str, str]:
    # The CPU receipt binds its own runner, fixture, QA-video builder and protocol.
    return {
        "runner": hashes[PREFLIGHT_RELATIVE.as_posix()],
        "fixture": hashes[FIXTURE_RELATIVE.as_posix()],
        "protocol": hashes[PROTOCOL_RELATIVE.as_posix()],
        "qa_video_builder": hashes["scripts/run_nextgqa_gold_interval_margin_pilot.py"],
    }


def execute(*, mode: str, output: Path, preflight_receipt_path: Path,
            smoke_receipt_path: Path | None = None,
            repo_root: Path | None = None) -> dict[str, Any]:
    """Run one separate smoke or full cohort transaction (GPU CLI only)."""
    if mode not in {"smoke", "full"}:
        raise ValueError("run_mode_invalid")
    started_at = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    repo = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    private = repo / PRIVATE_RELATIVE
    output = _validate_output_path(output, private)
    hashes = source_hashes(repo)
    from qta_synthetic_event_construct_v1 import (
        PAIR_COUNT, CLIP_COUNT, FRAME_COUNT, ONSET_STRATA,
        assignment_manifest, build_assignment, build_construct_clip,
        external_smoke_seed, sha256_canonical,
    )
    assignment = assignment_manifest(python_version=platform.python_version())
    assignment_sha = sha256_canonical(assignment)
    smoke_seed = external_smoke_seed()
    smoke_seed_sha = _sha_bytes(str(smoke_seed).encode("ascii"))
    protocol_sha = hashes[PROTOCOL_RELATIVE.as_posix()]
    preflight, preflight_sha = _safe_private_json(preflight_receipt_path, private)
    preflight_source_hashes = _preflight_expected_source_subset(hashes)
    validate_preflight_receipt(
        preflight, expected_protocol_sha256=protocol_sha,
        expected_source_sha256=preflight_source_hashes,
        expected_assignment_sha256=assignment_sha, mode=mode)
    if mode == "full" and smoke_receipt_path is None:
        raise ValueError("full_mode_requires_external_smoke_receipt")

    import torch
    import transformers
    import PIL
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    map_question_token_positions, aggregate_captured_qta_rows, capture_qta_attention_forward_once = (
        _load_attention_components())
    from qwen_internal_attention_ablation import (
        install_video_key_intervention_hooks, remove_hooks,
    )
    from qta_choice_token_binding import resolve_choice_continuation_ids
    from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video

    snapshot = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                / "snapshots" / MODEL_REVISION)
    if not snapshot.is_dir() or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("pinned_cuda_bf16_runtime_unavailable")
    vram_before, _total_vram = torch.cuda.mem_get_info()
    host_before = _host_available_bytes()
    if vram_before < MIN_FREE_BEFORE_LOAD or host_before < MIN_HOST_AVAILABLE_BEFORE_LOAD:
        raise RuntimeError("preload_resource_guard_failed")
    checkpoint_shards = _checkpoint_shards(snapshot)
    checkpoint_sha = _checkpoint_digest(checkpoint_shards)
    model_config_sha = sha256_file(snapshot / "config.json")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    smoke_receipt_sha = None
    if mode == "full":
        smoke, smoke_receipt_sha = _safe_private_json(smoke_receipt_path, private)
        validate_smoke_receipt(
            smoke, expected_protocol_sha256=protocol_sha,
            expected_sources=hashes, expected_assignment_sha256=assignment_sha,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_model_config_sha256=model_config_sha,
            expected_smoke_seed_sha256=smoke_seed_sha,
            expected_preflight_receipt_sha256=preflight_sha,
            expected_source_commit=source_commit)

    processor = AutoProcessor.from_pretrained(
        str(snapshot), revision=MODEL_REVISION, local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
        trust_remote_code=False,
    )
    processor_sha = _processor_fingerprint(processor)
    if processor_sha != preflight.get("processor_sha256"):
        raise ValueError("processor_fingerprint_preflight_mismatch")
    expected_grid = tuple(preflight["aggregate_processor_geometry"]["grid_counts"][0][
        "video_grid_thw"])
    tokenizer_sha = _sha_bytes(processor.tokenizer.backend_tokenizer.to_str().encode("utf-8"))
    if mode == "full":
        if (smoke.get("processor_sha256") != processor_sha
                or smoke.get("tokenizer_sha256") != tokenizer_sha):
            raise ValueError("external_smoke_processor_binding_invalid")
    vram_immediately_before_load, _ = torch.cuda.mem_get_info()
    host_immediately_before_load = _host_available_bytes()
    if (vram_immediately_before_load < MIN_FREE_BEFORE_LOAD
            or host_immediately_before_load < MIN_HOST_AVAILABLE_BEFORE_LOAD):
        raise RuntimeError("immediate_pre_model_load_resource_guard_failed")
    workspace_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(snapshot), revision=MODEL_REVISION, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    ).eval()
    if (int(model.config.video_token_id) != int(processor.video_token_id)
            or len(model.model.language_model.layers) < max(LAYERS_1_BASED)):
        raise RuntimeError("model_processor_or_layer_contract_invalid")
    torch.cuda.reset_peak_memory_stats()

    designs = ([_make_external_design(smoke_seed)] if mode == "smoke"
               else list(build_assignment()))
    pair_sums: list[tuple[int, float, float]] = []
    attempted = completed = 0
    overlap_counts = Counter()
    stage = "score_clips"
    try:
        for design in designs:
            pair_delta = []
            for changed_object in ("circle", "triangle"):
                clip = build_construct_clip(design, changed_object)
                item = {"question": clip.question,
                        "choices": [text for _label, text in clip.choices]}
                rendered, messages = build_qa_prompt(processor, item, clip.frames, (1, 1))
                choice_ids, _prefix = resolve_choice_continuation_ids(
                    processor.tokenizer, rendered, labels="ABCDE")
                images, videos = process_vision_info(messages, return_video_metadata=True)
                if images is not None or not isinstance(videos, list) or len(videos) != 1:
                    raise ValueError("video_input_contract_invalid")
                pixels, metadata = videos[0]
                if len(pixels) != FRAME_COUNT:
                    raise ValueError("processor_frame_count_invalid")
                cpu_batch = process_qa_video(processor, rendered, images, pixels, metadata)
                ids = cpu_batch.get("input_ids")
                grid = cpu_batch.get("video_grid_thw")
                attention_mask = cpu_batch.get("attention_mask")
                if (not torch.is_tensor(ids) or ids.ndim != 2 or tuple(ids.shape[:1]) != (1,)
                        or not torch.is_tensor(grid) or tuple(grid.shape) != (1, 3)
                        or not torch.is_tensor(attention_mask) or attention_mask.shape != ids.shape
                        or not bool((attention_mask == 1).all())):
                    raise ValueError("processor_batch_geometry_invalid")
                # Confirm every answer is one processor continuation token.
                for label, expected_id in zip("ABCDE", choice_ids, strict=True):
                    extended = process_qa_video(processor, rendered + label,
                                                images, pixels, metadata)
                    extended_ids = extended["input_ids"][0].detach().cpu().tolist()
                    base_ids = ids[0].detach().cpu().tolist()
                    if extended_ids != base_ids + [expected_id]:
                        raise ValueError("processed_choice_continuation_mismatch")
                    del extended, extended_ids
                t, grid_h, grid_w = map(int, grid[0].tolist())
                merge = int(processor.image_processor.merge_size)
                tokens_per_group = grid_h * grid_w // merge**2
                visual_positions = [i for i, value in enumerate(ids[0].tolist())
                                    if int(value) == int(processor.video_token_id)]
                if (tuple(map(int, grid[0].tolist())) != expected_grid
                        or t != 16 or len(visual_positions) != t * tokens_per_group
                        or tokens_per_group < 1):
                    raise ValueError("processor_temporal_group_geometry_invalid")
                groups = tuple(tuple(visual_positions[g * tokens_per_group:(g + 1) * tokens_per_group])
                               for g in range(t))
                if any(not group for group in groups):
                    raise ValueError("processor_temporal_group_empty")
                question_positions = map_question_token_positions(
                    tokenizer=processor.tokenizer, rendered_text=rendered,
                    question_text=clip.question, processed_input_ids=ids,
                    video_token_id=int(processor.video_token_id))
                if max(question_positions) >= int(ids.shape[1]) - 1:
                    raise ValueError("question_attention_query_position_invalid")
                evidence, controls = _event_controls(design.onset_frame)
                query_position = int(ids.shape[1]) - 1
                device_batch = {key: value.to("cuda") if torch.is_tensor(value) else value
                                for key, value in cpu_batch.items()}
                score_cache: dict[str, Any] = {}

                def forward(*, name, biased_positions, ablated_positions,
                            capture_question_attention):
                    nonlocal attempted, completed
                    free, _ = torch.cuda.mem_get_info()
                    host = _host_available_bytes()
                    if free < MIN_FREE_BEFORE_FORWARD or host < MIN_HOST_AVAILABLE_BEFORE_FORWARD:
                        raise RuntimeError("per_forward_resource_guard_failed")
                    if capture_question_attention:
                        attempted += 1
                        output, rows, capture_report = capture_qta_attention_forward_once(
                            model=model, forward=lambda: model(
                                **device_batch, use_cache=False, logits_to_keep=1),
                            question_positions=question_positions,
                            input_length=int(ids.shape[1]), torch_module=torch)
                        if (capture_report.get("forward_calls") != 1
                                or capture_report.get("layer_indices_1_based") != list(LAYERS_1_BASED)
                                or capture_report.get("attention_rows_satisfy_source_dtype_normalization_tolerance") is not True):
                            raise RuntimeError("baseline_attention_capture_invalid")
                        masses_by_group = aggregate_captured_qta_rows(
                            rows=rows, group_positions=groups, torch_module=torch)
                        del rows, capture_report
                    else:
                        hooks = install_video_key_intervention_hooks(
                            model=model, layer_indices=tuple(layer - 1 for layer in LAYERS_1_BASED),
                            biased_positions=biased_positions,
                            ablated_positions=ablated_positions,
                            query_position=query_position, logit_bias=LOGIT_BIAS,
                            torch_module=torch)
                        try:
                            attempted += 1
                            with torch.inference_mode():
                                output = model(**device_batch, use_cache=False, logits_to_keep=1)
                        finally:
                            remove_hooks(hooks)
                        masses_by_group = None
                    logits = getattr(output, "logits", None)
                    if (not torch.is_tensor(logits) or logits.ndim != 3
                            or tuple(logits.shape[:2]) != (1, 1)
                            or not bool(torch.isfinite(logits).all())):
                        del output
                        raise RuntimeError("forward_logits_invalid")
                    log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
                    values = [float(log_probs[token].item()) for token in choice_ids]
                    if any(not math.isfinite(value) for value in values):
                        del output, logits, log_probs
                        raise RuntimeError("choice_log_probability_nonfinite")
                    completed += 1
                    score_cache[name] = values
                    del output, logits, log_probs
                    torch.cuda.empty_cache()
                    return values, masses_by_group

                attempted += 0
                adapter = run_nine_branch_adapter(
                    gold_index=next(index for index, (label, _text) in enumerate(clip.choices)
                                    if label == clip.correct_choice_label),
                    evidence_positions=evidence, control_positions=controls,
                    temporal_groups=groups,
                    opaque_token=f"qta-synth-event-v1:{design.scene_seed}:{changed_object}",
                    forward=forward,
                )
                pair_delta.append((adapter["qta_delta_f"], adapter["sham_delta_f"]))
                for key in ("target_evidence_overlap_count", "target_control_overlap_count",
                            "sham_evidence_overlap_count", "sham_control_overlap_count"):
                    overlap_counts[key] += adapter[key]
                if adapter["forward_count"] != 9:
                    raise RuntimeError("nine_forward_clip_incomplete")
                del adapter, score_cache, device_batch, cpu_batch, ids, grid, attention_mask
                del groups, question_positions, clip, item, rendered, messages, images, videos
                del pixels, metadata, choice_ids, _prefix
                gc.collect()
                torch.cuda.empty_cache()
            if len(pair_delta) != 2:
                raise RuntimeError("scene_pair_clip_count_invalid")
            pair_sums.append((design.onset_frame,
                              sum(row[0] for row in pair_delta) / 2,
                              sum(row[1] for row in pair_delta) / 2))
            del pair_delta, design
        requested_pairs = 1 if mode == "smoke" else PAIR_COUNT
        requested_clips = requested_pairs * 2
        requested_forwards = requested_clips * 9
        torch.cuda.synchronize()
        if len(pair_sums) != requested_pairs or completed != requested_forwards:
            raise RuntimeError("requested_forward_denominator_incomplete")
        status = ("passed_external_engineering_smoke_only" if mode == "smoke"
                  else "passed_full_synthetic_construct_cohort")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "purpose": "synthetic_qta_construct_check_not_faithfulness",
            "status": status,
            "mode": mode,
            "scope": "synthetic_procedural_event_dependency_mechanism_check",
            "started_utc": started_utc,
            "requested_pairs": requested_pairs,
            "effective_pairs": len(pair_sums),
            "requested_clips": requested_clips,
            "effective_clips": len(pair_sums) * 2,
            "requested_forwards": requested_forwards,
            "completed_forwards": completed,
            "attempted_forwards": attempted,
            "failure_count": 0,
            "model_revision": MODEL_REVISION,
            "precision": "4bit_bf16_compute_bf16_eager",
            "qta_layers_1_based": list(LAYERS_1_BASED),
            "qta_logit_bias": LOGIT_BIAS,
            "qta_top_group_rule": "max(1,ceil(0.10*T)); earlier-index tie break",
            "hash_sham_seed": SHAM_SEED,
            "score": "gold_choice_next_token_logprob_minus_baseline_full_best_non_gold_choice_logprob",
            "primary": "scene-pair averaged DeltaF_QTA = F_QTA - F_baseline",
            "analysis_sufficient_statistics_by_onset": (
                None if mode == "smoke" else _aggregate_pair_deltas(pair_sums)),
            "secondary_selector_overlap_group_count_sums": dict(sorted(overlap_counts.items())),
            "assignment_manifest_sha256": assignment_sha,
            "external_smoke_seed_sha256": smoke_seed_sha,
            "preflight_receipt_sha256": preflight_sha,
            "smoke_receipt_sha256": smoke_receipt_sha,
            "protocol_sha256": protocol_sha,
            "source_sha256": hashes,
            "source_commit": source_commit,
            "workspace_dirty": workspace_dirty,
            "checkpoint_shards": checkpoint_shards,
            "checkpoint_sha256": checkpoint_sha,
            "processor_sha256": processor_sha,
            "tokenizer_sha256": tokenizer_sha,
            "model_config_sha256": model_config_sha,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "runtime": {
                "python": platform.python_version(), "torch": str(torch.__version__),
                "transformers": str(transformers.__version__), "pillow": str(PIL.__version__),
                "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
                "bitsandbytes": importlib.metadata.version("bitsandbytes"),
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "free_vram_before_load_bytes": vram_before,
                "host_available_before_load_bytes": host_before,
                "free_vram_immediately_before_load_bytes": vram_immediately_before_load,
                "host_available_immediately_before_load_bytes": host_immediately_before_load,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "cohort_included": mode == "full",
            "model_weights_loaded": True,
            "gpu_used": True,
            "faithfulness_result": False,
            "rights_status": "unverified_local_only",
            "limitations": [
                "Synthetic construct validation only; no generated explanation or claim was evaluated.",
                "Internal attention bias and key ablations can interact and can shift inputs out of distribution.",
                "The nearest temporal non-event controls are not visual placebos.",
                "Results do not establish natural-video evidence dependence or faithfulness.",
            ],
        }
        return payload
    except ScoringRunError:
        raise
    except Exception as exc:
        code = str(exc)
        if not code.isidentifier():
            code = "scoring_failed"
        raise ScoringRunError(code, attempted=attempted,
                              completed=completed) from exc
    finally:
        try:
            del model
        except UnboundLocalError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--preflight-receipt", required=True, type=Path)
    parser.add_argument("--smoke-receipt", type=Path,
                        help="required in full mode; successful reserved-seed smoke JSON")
    parser.add_argument("--output", required=True, type=Path,
                        help="new private JSON path under data/private_nextgqa/")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    private = repo / PRIVATE_RELATIVE
    started = datetime.now(timezone.utc).isoformat()
    output_path = None
    try:
        output_path = _validate_output_path(args.output, private)
        payload = execute(mode=args.mode, output=args.output,
                          preflight_receipt_path=args.preflight_receipt,
                          smoke_receipt_path=args.smoke_receipt, repo_root=repo)
        _write_private_json(output_path, payload, private)
    except Exception as exc:
        attempted = getattr(exc, "attempted", 0)
        completed = getattr(exc, "completed", 0)
        code = getattr(exc, "code", str(exc))
        failure_name = None
        if isinstance(exc, ImportError):
            candidate = getattr(exc, "name", None)
            if isinstance(candidate, str) and candidate.isidentifier():
                failure_name = candidate
                code = "import_" + candidate
        if not isinstance(code, str) or not code.isidentifier():
            code = "scoring_failed"
        failure = {
            "schema_version": 1,
            "purpose": "synthetic_qta_construct_check_not_faithfulness",
            "status": "failed_no_cohort_estimate",
            "mode": args.mode,
            "started_utc": started,
            "effective_pairs": 0,
            "effective_clips": 0,
            "requested_forwards": 18 if args.mode == "smoke" else 720,
            "attempted_forwards": attempted if type(attempted) is int else 0,
            "completed_forwards": completed if type(completed) is int else 0,
            "failure_count": 1,
            "failure_type": type(exc).__name__,
            "failure_code": code,
            "failure_import_name": failure_name,
            "cohort_estimate": None,
            "faithfulness_result": False,
            "model_revision": MODEL_REVISION,
            "rights_status": "unverified_local_only",
        }
        if output_path is None:
            parser.error(f"synthetic_construct_scoring_failed:{type(exc).__name__}:{code}; no cohort estimate")
        try:
            _write_private_json(output_path, failure, private)
        except Exception:
            parser.error("synthetic_construct_scoring_failed; failure_receipt_not_written")
        parser.error(f"synthetic_construct_scoring_failed:{type(exc).__name__}:{code}; no cohort estimate")
    print(json.dumps({
        "status": payload["status"], "mode": args.mode,
        "effective_pairs": payload["effective_pairs"],
        "effective_clips": payload["effective_clips"],
        "completed_forwards": payload["completed_forwards"],
        "faithfulness_result": False,
        "private_output_name": args.output.name,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
