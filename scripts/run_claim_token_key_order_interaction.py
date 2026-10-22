#!/usr/bin/env python3
"""Exploratory claim-token-conditioned temporal visual-key intervention.

This module freezes deterministic claim-query key/sham plans and validates the
private lock/packet. Its current capture protocol is non-authorizing, so GPU
smoke/full scoring is fail-closed; preflight remains available for input checks.
The measured endpoint is a conditional text-likelihood sensitivity proxy,
not semantic support or direct explanation faithfulness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Sequence
from fractions import Fraction
import gc
import time

from private_output_paths import validate_private_output
from qwen_question_temporal_attention_selector import select_top_temporal_groups


LAYERS_1_BASED = tuple(range(5, 17))
LOGIT_BIAS = math.log(2.0)
TOP_FRACTION_NUMERATOR = 1
TOP_FRACTION_DENOMINATOR = 10
BRANCHES = frozenset({
    "natural_baseline", "permuted_baseline",
    "natural_claim_key", "permuted_claim_key",
    "natural_hash_sham", "permuted_hash_sham",
})
PLAN_PURPOSE = "claim_token_conditioned_temporal_key_order_interaction_plan"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
FRAME_COUNT = 16
LOCK_REL = "data/private_nextgqa/temporal_order_claim_cohort_20261021_07.json"
PAIR_REL = "data/private_nextgqa/temporal_order_claim_pair_filter_20261021_07.json"
PACKET_REL = "data/private_nextgqa/claim_generation_claim_conditioned_temporal_test_exploratory_20261021_07.json"
MIN_FREE_VRAM = 12 * 1024**3
MIN_HOST_AVAILABLE = 8 * 1024**3
MIN_FORWARD_FREE_VRAM = 2 * 1024**3
# This runner's capture contract explicitly reports no authenticated source,
# capture, model-load permission, or scoring authorization. A future authorized
# study must use a separately versioned runner and pre-load transaction.
GPU_EXECUTION_AUTHORIZATION = "not_granted"


def require_gpu_execution_authorization() -> None:
    if GPU_EXECUTION_AUTHORIZATION != "granted":
        raise ValueError("capture_protocol_authorization_not_granted")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def deterministic_reverse_group_indices(group_count: int) -> tuple[int, ...]:
    """Return new group-index order for a reversal of complete temporal groups."""
    if type(group_count) is not int or group_count < 2:
        raise ValueError("temporal_group_count_invalid")
    return tuple(reversed(range(group_count)))


def map_group_keys_to_permuted_positions(*, selected_group_indices: Sequence[int],
        original_groups: Sequence[Sequence[int]],
        permuted_groups: Sequence[Sequence[int]],
        permuted_group_for_original: Sequence[int]) -> tuple[int, ...]:
    """Map selected content groups to their positions after temporal reversal.

    ``permuted_group_for_original[i]`` names the permuted position containing
    the same frame content as original group ``i``. This prevents a treatment
    from silently steering different frames after the order perturbation.
    """
    groups = _validate_groups(original_groups)
    permuted = _validate_groups(permuted_groups)
    if len(groups) != len(permuted):
        raise ValueError("permuted_group_count_mismatch")
    mapping = tuple(permuted_group_for_original)
    if (len(mapping) != len(groups)
            or any(type(x) is not int for x in mapping)
            or tuple(sorted(mapping)) != tuple(range(len(groups)))):
        raise ValueError("permuted_group_mapping_invalid")
    selected = _validate_group_indices(selected_group_indices, len(groups))
    return tuple(sorted(position for index in selected
                        for position in permuted[mapping[index]]))


def _validate_groups(groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("temporal_groups_invalid")
    normalized = []
    for group in groups:
        if (not isinstance(group, (list, tuple)) or not group
                or any(type(x) is not int or x < 0 for x in group)
                or tuple(group) != tuple(range(group[0], group[-1] + 1))):
            raise ValueError("temporal_group_positions_invalid")
        normalized.append(tuple(group))
    flattened = tuple(position for group in normalized for position in group)
    if (normalized != sorted(normalized, key=lambda x: x[0])
            or len({len(group) for group in normalized}) != 1
            or len(set(flattened)) != len(flattened)
            or flattened != tuple(range(flattened[0], flattened[-1] + 1))):
        raise ValueError("temporal_group_partition_invalid")
    return tuple(normalized)


def _validate_group_indices(indices: Sequence[int], group_count: int) -> tuple[int, ...]:
    if (not isinstance(indices, (list, tuple)) or not indices
            or any(type(index) is not int or not 0 <= index < group_count
                   for index in indices)
            or tuple(sorted(set(indices))) != tuple(indices)):
        raise ValueError("selected_group_indices_invalid")
    return tuple(indices)


def select_disjoint_hash_sham_groups(*, group_count: int, selected_groups: Sequence[int],
        opaque_row_token: str, protocol_seed: str = "claim-query-key-sham-v1") -> tuple[int, ...]:
    """Choose a deterministic equal-budget sham from the selected complement."""
    if type(group_count) is not int or group_count < 1:
        raise ValueError("temporal_group_count_invalid")
    selected = _validate_group_indices(selected_groups, group_count)
    if (not isinstance(opaque_row_token, str) or len(opaque_row_token) != 64
            or any(c not in "0123456789abcdef" for c in opaque_row_token)
            or not isinstance(protocol_seed, str) or not protocol_seed
            or "\x00" in protocol_seed):
        raise ValueError("sham_binding_invalid")
    available = [i for i in range(group_count) if i not in set(selected)]
    if len(available) < len(selected):
        raise ValueError("disjoint_equal_budget_sham_unavailable")
    prefix = f"{protocol_seed}\0{opaque_row_token}\0"
    ranked = sorted(available, key=lambda i: (
        hashlib.sha256(f"{prefix}{i}".encode("utf-8")).digest(), i))
    return tuple(sorted(ranked[:len(selected)]))


def build_claim_key_plan(*, attention_scores: Sequence[float],
        temporal_groups: Sequence[Sequence[int]], query_positions: Sequence[int],
        opaque_row_token: str, protocol_seed: str = "claim-query-key-order-v1") -> dict:
    """Freeze top-10%-claim-attended groups and a disjoint hash sham."""
    groups = _validate_groups(temporal_groups)
    if (not isinstance(attention_scores, (list, tuple))
            or len(attention_scores) != len(groups)):
        raise ValueError("claim_attention_group_score_count_mismatch")
    if (not isinstance(query_positions, (list, tuple)) or not query_positions
            or any(type(x) is not int or x < 0 for x in query_positions)
            or tuple(query_positions) != tuple(range(query_positions[0], query_positions[-1] + 1))):
        raise ValueError("claim_query_positions_invalid")
    if groups[-1][-1] >= query_positions[0]:
        raise ValueError("visual_keys_not_causal_for_claim_queries")
    if (not isinstance(opaque_row_token, str) or len(opaque_row_token) != 64
            or any(c not in "0123456789abcdef" for c in opaque_row_token)):
        raise ValueError("opaque_row_token_invalid")
    # The 10% rule is fixed by protocol; ceil and minimum one group.
    top_k = max(1, math.ceil(len(groups) * TOP_FRACTION_NUMERATOR
                             / TOP_FRACTION_DENOMINATOR))
    selected = tuple(sorted(select_top_temporal_groups(attention_scores, top_k=top_k)))
    sham = select_disjoint_hash_sham_groups(group_count=len(groups),
        selected_groups=selected, opaque_row_token=opaque_row_token,
        protocol_seed=protocol_seed)
    natural_selected_keys = tuple(sorted(p for i in selected for p in groups[i]))
    natural_sham_keys = tuple(sorted(p for i in sham for p in groups[i]))
    return {
        "layers_1_based": list(LAYERS_1_BASED), "logit_bias_requested": LOGIT_BIAS,
        "query_positions": list(query_positions), "temporal_group_count": len(groups),
        "top_fraction": [TOP_FRACTION_NUMERATOR, TOP_FRACTION_DENOMINATOR],
        "selected_group_indices": list(selected), "sham_group_indices": list(sham),
        "selected_key_positions": list(natural_selected_keys),
        "sham_key_positions": list(natural_sham_keys),
        "selected_group_scores": [float(attention_scores[i]) for i in selected],
        "sham_protocol_seed": protocol_seed,
        "selection_scope": "baseline teacher-forced claim-token query attention; exploratory and self-conditioned",
    }


def build_six_forward_schedule() -> tuple[dict[str, str], ...]:
    """Natural baseline also supplies the claim-query attention capture."""
    return (
        {"branch": "natural_baseline", "order": "natural", "intervention": "none",
         "capture_claim_query_attention": "yes"},
        {"branch": "permuted_baseline", "order": "reversed_complete_temporal_groups",
         "intervention": "none", "capture_claim_query_attention": "no"},
        {"branch": "natural_claim_key", "order": "natural", "intervention": "selected_claim_key_ln2",
         "capture_claim_query_attention": "no"},
        {"branch": "permuted_claim_key", "order": "reversed_complete_temporal_groups",
         "intervention": "same_content_selected_claim_key_ln2", "capture_claim_query_attention": "no"},
        {"branch": "natural_hash_sham", "order": "natural", "intervention": "disjoint_equal_budget_hash_sham_ln2",
         "capture_claim_query_attention": "no"},
        {"branch": "permuted_hash_sham", "order": "reversed_complete_temporal_groups",
         "intervention": "same_content_hash_sham_ln2", "capture_claim_query_attention": "no"},
    )


def compute_interactions(branch_scores: dict[str, float]) -> dict:
    if not isinstance(branch_scores, dict) or set(branch_scores) != BRANCHES:
        raise ValueError("branch_score_set_invalid")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(float(x)) for x in branch_scores.values()):
        raise ValueError("branch_score_nonfinite")
    baseline = float(branch_scores["natural_baseline"] - branch_scores["permuted_baseline"])
    treatment = float(branch_scores["natural_claim_key"] - branch_scores["permuted_claim_key"])
    sham = float(branch_scores["natural_hash_sham"] - branch_scores["permuted_hash_sham"])
    return {
        "baseline_natural_minus_permuted_nats_per_token": baseline,
        "claim_key_order_interaction_nats_per_token": treatment - baseline,
        "hash_sham_order_interaction_nats_per_token": sham - baseline,
        "claim_key_minus_sham_specificity_nats_per_token": treatment - sham,
        "interpretation": "conditional claim-token likelihood sensitivity proxy; not semantic faithfulness",
    }


def _read_private_json(repo: Path, relpath: str) -> tuple[dict, str]:
    relative = Path(relpath)
    if (relative.is_absolute() or len(relative.parts) < 3
            or relative.parts[:2] != ("data", "private_nextgqa")
            or any(part in {"", ".", ".."} for part in relative.parts)):
        raise ValueError("private_input_path_invalid")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    opened = []
    try:
        current = os.open(repo, os.O_RDONLY | directory)
        opened.append(current)
        for component in relative.parts[:-1]:
            current = os.open(component, os.O_RDONLY | directory | nofollow,
                               dir_fd=current)
            opened.append(current)
        fd = os.open(relative.parts[-1], os.O_RDONLY | nofollow, dir_fd=current)
    except OSError:
        raise ValueError("private_input_open_failed") from None
    finally:
        for directory_fd in reversed(opened):
            try:
                os.close(directory_fd)
            except OSError:
                pass
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_input_permissions_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("private_input_object_invalid")
    return payload, sha256_bytes(raw)


def validate_locked_claim_packets(*, lock: dict, pair_filter: dict,
        packet_payloads: Sequence[dict]) -> list[dict]:
    """Check exactly one valid generated claim per locked media/QID row."""
    assignments = lock.get("assignments") if isinstance(lock, dict) else None
    pairs = pair_filter.get("pair_sha256") if isinstance(pair_filter, dict) else None
    if (lock.get("status") != "frozen_before_claim_generation_and_scoring"
            or lock.get("dataset_split") not in {"test", "val"}
            or lock.get("requested_videos") != 3 or lock.get("effective_videos") != 3
            or not _is_sha(lock.get("cohort_sha256"))
            or not isinstance(assignments, list) or len(assignments) != 3
            or pair_filter.get("schema_version") != 1 or not isinstance(pairs, list)
            or len(pairs) != 3 or len(set(pairs)) != 3
            or any(not _is_sha(value) for value in pairs)):
        raise ValueError("cohort_lock_or_pair_filter_invalid")
    if sha256_bytes(canonical_json_bytes(pair_filter)) != lock.get("selected_pair_filter_sha256"):
        raise ValueError("selected_pair_filter_digest_mismatch")
    by_media = {}
    for index, row in enumerate(assignments):
        media, qid = row.get("media_sha256"), row.get("qid_sha256")
        if (not _is_sha(media) or not _is_sha(qid) or media in by_media):
            raise ValueError("cohort_assignment_invalid")
        by_media[media] = (index, qid)
    valid = {}
    for packet in packet_payloads:
        if (packet.get("dataset_split") != lock.get("dataset_split")
                or packet.get("cohort_sha256") != lock.get("cohort_sha256")
                or packet.get("status") not in {"passed", "partial"}
                or not isinstance(packet.get("records"), list)):
            raise ValueError("claim_packet_contract_invalid")
        packet_records = packet["records"]
        packet_valid_count = sum(
            isinstance(row, dict)
            and isinstance(row.get("generated_parsed"), dict)
            and row["generated_parsed"].get("valid") is True
            and isinstance(row["generated_parsed"].get("claim"), str)
            and bool(row["generated_parsed"]["claim"].strip())
            for row in packet_records)
        expected_status = "passed" if packet_valid_count == len(packet_records) else "partial"
        if (not packet_records
                or packet.get("requested_videos") != len(packet_records)
                or packet.get("effective_videos") != len(packet_records)
                or packet.get("fully_structured_claim_records") != packet_valid_count
                or packet.get("status") != expected_status):
            raise ValueError("claim_packet_counts_or_status_invalid")
        for record in packet_records:
            if not isinstance(record, dict):
                raise ValueError("claim_packet_record_invalid")
            media = record.get("video_sha256")
            binding = by_media.get(media)
            qid = record.get("dataset_question_id")
            if (binding is None or isinstance(qid, bool) or not isinstance(qid, (str, int))
                    or sha256_bytes(str(qid).encode("utf-8")) != binding[1]
                    or record.get("candidate_pair_sha256") != pairs[binding[0]]):
                raise ValueError("claim_packet_row_not_in_frozen_cohort")
            parsed = record.get("generated_parsed")
            claim = parsed.get("claim") if isinstance(parsed, dict) else None
            if (not isinstance(parsed, dict) or parsed.get("valid") is not True
                    or not isinstance(claim, str) or not claim.strip()):
                raise ValueError("all_frozen_claims_must_be_valid")
            index = binding[0]
            if index in valid:
                raise ValueError("duplicate_packet_row")
            valid[index] = record
    if set(valid) != {0, 1, 2}:
        raise ValueError("claim_packets_do_not_cover_frozen_cohort")
    return [valid[index] for index in range(3)]


def _is_sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def build_preflight_receipt(*, lock_hash: str, pair_filter_hash: str,
        packet_hashes: dict[str, str], row_count: int, source_hashes: dict[str, str],
        packet_declared_video_hashes: Sequence[str]) -> dict:
    if (not _is_sha(lock_hash) or not _is_sha(pair_filter_hash)
            or row_count != 3 or not packet_hashes
            or any(not _is_sha(value) for value in packet_hashes.values())
            or len(packet_declared_video_hashes) != 3
            or any(not _is_sha(value) for value in packet_declared_video_hashes)
            or len(set(packet_declared_video_hashes)) != 3
            or any(not _is_sha(value) for value in source_hashes.values())):
        raise ValueError("preflight_receipt_binding_invalid")
    return {
        "schema_version": 1, "status": "inputs_validated_gpu_not_run",
        "purpose": PLAN_PURPOSE, "started_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_lock_sha256": lock_hash, "pair_filter_sha256": pair_filter_hash,
        "claim_packet_sha256": packet_hashes, "source_sha256": source_hashes,
        "packet_declared_video_sha256": list(packet_declared_video_hashes),
        "video_bytes_rehashed": False,
        "requested_rows": 3, "validated_rows": row_count,
        "planned_forward_count_per_row": 6, "planned_forward_count_full": 18,
        "layers_1_based": list(LAYERS_1_BASED), "logit_bias_requested": LOGIT_BIAS,
        "human_review_performed": False, "direct_faithfulness": False,
        "gpu_execution_performed": False,
        "blocker": "preflight mode validates locked inputs only and does not execute GPU scoring",
    }


def write_private_json_exclusive(path: Path, payload: dict) -> None:
    raw = canonical_json_bytes(payload)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _host_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("host_memory_probe_unavailable")


def _safe_failure_code(exc: Exception) -> str:
    """Emit only path-free stable codes; never serialize OS exception text."""
    if isinstance(exc, ValueError):
        candidate = str(exc)
        if candidate and all(char.isascii() and (char.islower() or char.isdigit() or char == "_")
                             for char in candidate):
            return candidate[:96]
    return "operation_failed_" + type(exc).__name__.lower()


def _move_batch(batch: dict, torch) -> dict:
    return {key: value.to("cuda") if torch.is_tensor(value) else value
            for key, value in batch.items()}


def _validate_bias_rows(mask, *, keys: Sequence[int], queries: Sequence[int], torch) -> None:
    if (not torch.is_tensor(mask) or mask.ndim != 4
            or tuple(mask.shape[:2]) != (1, 1) or not keys or not queries):
        raise ValueError("biased_mask_audit_shape_invalid")
    q = torch.tensor(tuple(queries), dtype=torch.long, device=mask.device)
    k = torch.tensor(tuple(keys), dtype=torch.long, device=mask.device)
    observed = mask[0, 0][q[:, None], k[None, :]]
    expected = torch.full_like(observed, LOGIT_BIAS)
    if not torch.allclose(observed, expected, rtol=0.0, atol=1e-6):
        raise ValueError("biased_mask_audit_value_mismatch")


def _score_claim_branch(*, model, batch_cpu, target_ids, video_token_id,
                        query_positions, temporal_groups, key_positions=(),
                        capture=False, torch):
    from qta_multiquery_video_key_intervention import (
        append_target_to_unpadded_batch,
        install_multiquery_video_key_intervention_hooks,
        remove_hooks, score_target_logits_window)
    from qta_attention_forward_capture import capture_qta_attention_forward_once
    from qta_attention_capture_contract import LAYERS as CAPTURE_LAYERS

    batch = _move_batch(batch_cpu, torch)
    prompt_length = int(batch["input_ids"].shape[1])
    extended, predicted_queries = append_target_to_unpadded_batch(
        batch=batch, target_token_ids=target_ids, torch_module=torch)
    if tuple(predicted_queries) != tuple(query_positions):
        raise ValueError("claim_prediction_query_binding_mismatch")
    visual = tuple(int(x) for x in (batch["input_ids"][0] == video_token_id)
                   .nonzero(as_tuple=True)[0].detach().cpu().tolist())
    if not visual or max(visual) >= min(query_positions):
        raise ValueError("visual_keys_not_causal_for_claim_queries")
    handles, audit_handles = [], []
    invocation_counts = {layer: 0 for layer in LAYERS_1_BASED}
    try:
        if key_positions:
            if not set(key_positions).issubset(visual):
                raise ValueError("intervention_keys_not_visual_tokens")
            handles = install_multiquery_video_key_intervention_hooks(
                model=model, layer_indices=tuple(layer - 1 for layer in LAYERS_1_BASED),
                biased_positions=key_positions, ablated_positions=(),
                query_positions=query_positions, logit_bias=LOGIT_BIAS,
                torch_module=torch)
            layers = model.model.language_model.layers
            for layer_1based in LAYERS_1_BASED:
                def audit(_module, args, kwargs, *, layer=layer_1based):
                    del args
                    if "attention_mask" not in kwargs:
                        raise ValueError("biased_mask_audit_mask_missing")
                    _validate_bias_rows(kwargs["attention_mask"], keys=key_positions,
                                        queries=query_positions, torch=torch)
                    invocation_counts[layer] += 1
                audit_handles.append(layers[layer_1based - 1].self_attn
                    .register_forward_pre_hook(audit, with_kwargs=True))
        free, _ = torch.cuda.mem_get_info()
        if free < MIN_FORWARD_FREE_VRAM or _host_available_bytes() < MIN_HOST_AVAILABLE:
            raise RuntimeError("per_forward_resource_guard_failed")
        with torch.inference_mode():
            if capture:
                output, rows, report = capture_qta_attention_forward_once(
                    model=model, forward=lambda: model(**extended, use_cache=False,
                        logits_to_keep=len(target_ids) + 1),
                    question_positions=query_positions,
                    input_length=int(extended["input_ids"].shape[1]),
                    torch_module=torch)
                if (report.get("layer_indices_1_based") != list(CAPTURE_LAYERS)
                        or report.get("forward_calls") != 1
                        or report.get("attention_rows_satisfy_source_dtype_normalization_tolerance") is not True):
                    raise ValueError("claim_attention_capture_contract_invalid")
            else:
                output = model(**extended, use_cache=False,
                    logits_to_keep=len(target_ids) + 1)
                rows, report = None, None
        logits = getattr(output, "logits", None)
        if (not torch.is_tensor(logits) or logits.ndim != 3
                or tuple(logits.shape[:2]) != (1, len(target_ids) + 1)
                or not bool(torch.isfinite(logits).all())):
            raise ValueError("claim_logits_window_invalid")
        scored = score_target_logits_window(logits=logits,
            target_token_ids=target_ids, prompt_length=prompt_length,
            first_query_position=prompt_length - 1, torch_module=torch)
        if key_positions and any(count != 1 for count in invocation_counts.values()):
            raise ValueError("intervention_hook_invocation_count_mismatch")
        scored["intervention_hook_invocations_by_layer_1_based"] = invocation_counts
        scored["visual_key_position_count"] = len(visual)
        scored["intervention_key_position_count"] = len(key_positions)
        scored["intervention_key_positions_sha256"] = sha256_bytes(
            json.dumps(list(key_positions), separators=(",", ":")).encode("ascii"))
        if capture:
            from qta_attention_forward_capture import aggregate_captured_qta_rows
            scores = aggregate_captured_qta_rows(rows=rows,
                group_positions=temporal_groups, torch_module=torch)
            scored["_attention_group_scores"] = scores
            scored["_attention_capture_report"] = report
        del output, logits
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        return scored
    finally:
        for handle in reversed(audit_handles):
            handle.remove()
        remove_hooks(handles)
        del extended, batch
        torch.cuda.empty_cache()


def _prepare_temporal_batch(processor, process_vision_info, record, frames,
                            sample_fps, prompt_suffix, torch):
    from run_saved_claim_frame_permutation_attention_smoke_20261021_v1 import (
        _prepare_processor_batch)
    return _prepare_processor_batch(processor, process_vision_info, record,
                                    frames, sample_fps, prompt_suffix, torch)


def _processor_temporal_groups(batch: dict, processor, video_token_id: int,
                               frame_count: int) -> tuple[tuple[int, ...], ...]:
    grid = batch.get("video_grid_thw")
    if grid is None or tuple(grid.shape) != (1, 3):
        raise ValueError("processor_video_grid_shape_invalid")
    steps, height, width = (int(x) for x in grid[0].tolist())
    merge = int(getattr(processor.image_processor, "merge_size",
                         getattr(processor.video_processor, "spatial_merge_size", 0)))
    temporal_patch = int(getattr(processor.video_processor, "temporal_patch_size", 0))
    if (temporal_patch < 1 or frame_count % temporal_patch
            or steps != frame_count // temporal_patch or merge < 1
            or height % merge or width % merge):
        raise ValueError("processor_video_grid_frame_contract_invalid")
    ids = batch["input_ids"][0].tolist()
    positions = tuple(i for i, value in enumerate(ids) if int(value) == video_token_id)
    per_group = (height // merge) * (width // merge)
    if not positions or len(positions) != steps * per_group or positions != tuple(
            range(positions[0], positions[0] + len(positions))):
        raise ValueError("processor_video_token_block_invalid")
    return tuple(tuple(positions[i * per_group:(i + 1) * per_group])
                 for i in range(steps))


def _run_gpu(args) -> int:
    """Run one-row smoke; full mode requires caller-validated smoke digest."""
    repo = Path(__file__).resolve().parents[1]
    output = __import__("private_output_paths").validate_private_output(args.output, repo)
    if output.parent != repo / "data/private_nextgqa" or output.suffix != ".json":
        raise ValueError("gpu_output_must_be_json_child_of_private_data_root")
    started_utc = datetime.now(timezone.utc).isoformat(); started = time.monotonic()
    attempted = completed = 0; stage = "private_lock_packet_validation"
    row_token = None; packet_hashes = {}; video_hash = None
    try:
        stage = "capture_protocol_preload_authorization"
        require_gpu_execution_authorization()
        stage = "private_lock_packet_validation"
        lock, lock_hash = _read_private_json(repo, args.cohort_lock)
        pairs, pair_hash = _read_private_json(repo, args.pair_filter)
        packets = []
        for rel in args.claim_packet:
            packet_payload, packet_digest = _read_private_json(repo, rel)
            if Path(rel).name in packet_hashes:
                raise ValueError("claim_packet_basename_collision")
            packets.append(packet_payload)
            packet_hashes[Path(rel).name] = packet_digest
        if not packets:
            raise ValueError("claim_packet_required")
        packet_hash = sha256_bytes(canonical_json_bytes(packet_hashes))
        records = validate_locked_claim_packets(lock=lock, pair_filter=pairs,
                                                packet_payloads=packets)
        packet = packets[0]
        if (lock.get("cohort_sha256") != packet.get("cohort_sha256")
                or any(packet_row.get("model_revision") != MODEL_REVISION
                       or packet_row.get("frames_per_video") != FRAME_COUNT
                       or packet_row.get("generation_prompt_version") != "v1"
                       for packet_row in packets)):
            raise ValueError("locked_claim_packet_model_or_frame_contract_invalid")
        source_rel = ("scripts/run_claim_token_key_order_interaction.py",
            "scripts/run_saved_claim_frame_permutation_attention_smoke_20261021_v1.py",
            "scripts/qta_multiquery_video_key_intervention.py",
            "scripts/qta_attention_forward_capture.py",
            "scripts/qta_attention_capture_contract.py",
            "scripts/qwen_question_temporal_attention_selector.py",
            "scripts/claim_mask_video_decoder.py",
            "scripts/generate_nextgqa_claim_review_pilot.py")
        source_hashes = {name: _sha256_file(repo / name) for name in source_rel}
        if args.mode == "full":
            if not args.smoke_receipt:
                raise ValueError("full_run_requires_primary_validated_smoke_receipt")
            smoke_payload, smoke_hash = _read_private_json(repo, args.smoke_receipt)
            if smoke_hash != args.smoke_receipt_sha256:
                raise ValueError("smoke_receipt_binding_invalid")
            if (smoke_payload.get("status") != "completed_one_row_exploratory_smoke"
                    or smoke_payload.get("schema_version") != 1
                    or smoke_payload.get("cohort_lock_sha256") != lock_hash
                    or smoke_payload.get("pair_filter_sha256") != pair_hash
                    or smoke_payload.get("claim_packet_sha256") != packet_hash
                    or smoke_payload.get("effective_rows") != 1
                    or smoke_payload.get("completed_forwards") != 6
                    or not isinstance(smoke_payload.get("rows"), list)
                    or len(smoke_payload["rows"]) != 1
                    or not isinstance(smoke_payload["rows"][0], dict)
                    or not isinstance(smoke_payload["rows"][0].get("row"), dict)
                    or smoke_payload["rows"][0]["row"].get("row_token_sha256") !=
                       sha256_bytes((records[0]["video_sha256"] +
                           str(records[0]["dataset_question_id"])).encode())
                    or smoke_payload.get("source_sha256") != source_hashes):
                raise ValueError("smoke_receipt_not_matching_validated_cohort")
        else:
            smoke_hash = None
        selected_records = records[:1] if args.mode == "smoke" else records
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info(); host_free = _host_available_bytes()
        if free < MIN_FREE_VRAM or host_free < MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_model_resource_guard_failed")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir() or model_path.name != MODEL_REVISION:
            raise ValueError("pinned_local_checkpoint_missing_or_wrong_revision")
        checkpoint_files = ("config.json", "preprocessor_config.json", "tokenizer_config.json")
        if any(not (model_path / name).is_file() for name in checkpoint_files):
            raise ValueError("pinned_checkpoint_config_incomplete")
        checkpoint_hashes = {name: _sha256_file(model_path / name) for name in checkpoint_files}
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
        from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_V1
        from run_saved_claim_frame_permutation_attention_smoke_20261021_v1 import (
            deterministic_reverse_block_indices, reorder_frames, _tensor_sha256)

        stage = "load_processor_decode_and_prepare"
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        if int(processor.video_token_id) < 0:
            raise ValueError("processor_video_token_id_invalid")
        rows_payload = []
        batches = []
        for row_index, record in enumerate(selected_records):
            video_path, video_hash = record.get("local_video_path"), record.get("video_sha256")
            if not isinstance(video_path, str) or not _is_sha(video_hash):
                raise ValueError("locked_packet_video_binding_missing")
            frames, decoded = decode_sampled_video(video_path, num_frames=FRAME_COUNT,
                uniform_tolerance_seconds=Fraction(1, 20))
            verify_sampled_frame_pixels(frames, decoded)
            if decoded.get("source_sha256") != video_hash or len(frames) != FRAME_COUNT:
                raise ValueError("locked_packet_video_bytes_or_frame_count_mismatch")
            saved_pts = record.get("sampled_frame_relative_pts_seconds")
            actual_pts = [float(Fraction(*entry["relative_presentation_time"]))
                          for entry in decoded["sampled_frames"]]
            if (not isinstance(saved_pts, list) or len(saved_pts) != FRAME_COUNT
                    or any(abs(float(a)-b) > 1e-6 for a,b in zip(saved_pts, actual_pts, strict=True))):
                raise ValueError("locked_packet_video_timing_mismatch")
            fps = Fraction(*decoded["effective_sample_fps"])
            temporal_patch = int(processor.video_processor.temporal_patch_size)
            permutation = deterministic_reverse_block_indices(FRAME_COUNT, temporal_patch)
            reversed_frames = reorder_frames(frames, permutation)
            natural_cpu, natural_meta = _prepare_temporal_batch(processor, process_vision_info,
                record, frames, float(fps), PROMPT_SUFFIX_V1, torch)
            reversed_cpu, reversed_meta = _prepare_temporal_batch(processor, process_vision_info,
                record, reversed_frames, float(fps), PROMPT_SUFFIX_V1, torch)
            if natural_meta != reversed_meta or set(natural_cpu) != set(reversed_cpu):
                raise ValueError("reversed_processor_metadata_mismatch")
            for key in natural_cpu:
                left, right = natural_cpu[key], reversed_cpu[key]
                if torch.is_tensor(left) and torch.is_tensor(right):
                    if tuple(left.shape) != tuple(right.shape):
                        raise ValueError("reversed_processor_tensor_shape_mismatch")
                    if key != "pixel_values_videos" and not torch.equal(left, right):
                        raise ValueError("reversed_nonvisual_tensor_changed")
                elif left != right:
                    raise ValueError("reversed_processor_non_tensor_changed")
            groups = _processor_temporal_groups(natural_cpu, processor,
                int(processor.video_token_id), FRAME_COUNT)
            reverse_groups = _processor_temporal_groups(reversed_cpu, processor,
                int(processor.video_token_id), FRAME_COUNT)
            if len(groups) != len(reverse_groups):
                raise ValueError("reversed_temporal_group_count_changed")
            natural_pixels_hash = _tensor_sha256(natural_cpu["pixel_values_videos"], torch)
            reversed_pixels_hash = _tensor_sha256(reversed_cpu["pixel_values_videos"], torch)
            if natural_pixels_hash == reversed_pixels_hash:
                raise ValueError("frame_group_reversal_did_not_change_processor_pixels")
            claim = record["generated_parsed"]["claim"]
            target_ids = tuple(processor.tokenizer(claim, add_special_tokens=False)["input_ids"])
            if not target_ids or len(target_ids) > 512:
                raise ValueError("claim_retokenization_invalid")
            row_token = sha256_bytes((video_hash + str(record["dataset_question_id"])).encode())
            batches.append((row_index, record, natural_cpu, reversed_cpu, groups,
                            target_ids, row_token, natural_pixels_hash, reversed_pixels_hash))
            rows_payload.append({"row_token_sha256": row_token,
                "video_sha256": video_hash, "dataset_question_id_sha256": sha256_bytes(
                    str(record["dataset_question_id"]).encode()),
                "claim_text_sha256": sha256_bytes(claim.encode()),
                "claim_token_ids_sha256": sha256_bytes(json.dumps(target_ids,
                    separators=(",", ":")).encode("ascii")),
                "claim_token_count": len(target_ids), "sampled_video_frame_count": FRAME_COUNT,
                "temporal_group_count": len(groups), "natural_processor_pixels_sha256": batches[-1][-2],
                "reversed_processor_pixels_sha256": batches[-1][-1]})
        stage = "load_pinned_qwen_model"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
            device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if (int(model.config.video_token_id) != int(processor.video_token_id)
                or len(model.model.language_model.layers) < max(LAYERS_1_BASED)):
            raise ValueError("pinned_model_processor_architecture_mismatch")
        torch.manual_seed(20261021); torch.cuda.manual_seed_all(20261021)
        torch.cuda.reset_peak_memory_stats()
        all_results = []
        for row_tuple, row_meta in zip(batches, rows_payload, strict=True):
            row_index, record, natural_cpu, reversed_cpu, groups, target_ids, row_token, pix_n, pix_r = row_tuple
            prompt_length = int(natural_cpu["input_ids"].shape[1])
            query_positions = tuple(range(prompt_length - 1, prompt_length + len(target_ids) - 1))
            schedule = ("natural_baseline", "permuted_baseline")
            attempted += 1
            scores = {"natural_baseline": _score_claim_branch(model=model,
                batch_cpu=natural_cpu, target_ids=target_ids,
                video_token_id=int(processor.video_token_id), query_positions=query_positions,
                temporal_groups=groups, capture=True, torch=torch)}
            completed += 1
            plan = build_claim_key_plan(attention_scores=scores["natural_baseline"].pop(
                "_attention_group_scores"), temporal_groups=groups,
                query_positions=query_positions, opaque_row_token=row_token)
            group_order = deterministic_reverse_group_indices(len(groups))
            # Temporal reversal moves original content group i to position N-1-i.
            selected_reversed = map_group_keys_to_permuted_positions(
                selected_group_indices=plan["selected_group_indices"],
                original_groups=groups, permuted_groups=groups,
                permuted_group_for_original=group_order)
            sham_reversed = map_group_keys_to_permuted_positions(
                selected_group_indices=plan["sham_group_indices"],
                original_groups=groups, permuted_groups=groups,
                permuted_group_for_original=group_order)
            plan["permuted_selected_key_positions"] = list(selected_reversed)
            plan["permuted_sham_key_positions"] = list(sham_reversed)
            branch_specs = (
                ("permuted_baseline", reversed_cpu, ()),
                ("natural_claim_key", natural_cpu, tuple(plan["selected_key_positions"])),
                ("permuted_claim_key", reversed_cpu, selected_reversed),
                ("natural_hash_sham", natural_cpu, tuple(plan["sham_key_positions"])),
                ("permuted_hash_sham", reversed_cpu, sham_reversed))
            for branch, batch_cpu, keys in branch_specs:
                attempted += 1
                scores[branch] = _score_claim_branch(model=model,
                    batch_cpu=batch_cpu, target_ids=target_ids,
                    video_token_id=int(processor.video_token_id),
                    query_positions=query_positions, temporal_groups=groups,
                    key_positions=keys, torch=torch)
                completed += 1
            interaction = compute_interactions({name: score["target_logprob_mean"]
                for name, score in scores.items()})
            all_results.append({"row": row_meta, "plan": plan,
                "branch_scores": scores, "interaction": interaction,
                "baseline_attention_capture": scores["natural_baseline"].pop(
                    "_attention_capture_report")})
            del scores
            gc.collect(); torch.cuda.empty_cache()
        torch.cuda.synchronize()
        payload = {"schema_version": 1,
            "status": "completed_one_row_exploratory_smoke" if args.mode == "smoke"
                     else "completed_three_row_exploratory_run",
            "started_utc": started_utc, "purpose": PLAN_PURPOSE,
            "scope": "claim-token-conditioned temporal-key steering × complete temporal-group reversal",
            "direct_faithfulness": False, "human_review_performed": False,
            "rights_status": "unverified; local-only per user authorization",
            "mode": args.mode, "requested_rows": 1 if args.mode == "smoke" else 3,
            "effective_rows": len(all_results), "requested_forwards": (1 if args.mode == "smoke" else 3)*6,
            "attempted_forwards": attempted, "completed_forwards": completed,
            "cohort_lock_sha256": lock_hash, "pair_filter_sha256": pair_hash,
            "claim_packet_sha256": packet_hash, "claim_packet_hashes": packet_hashes,
            "smoke_receipt_sha256": smoke_hash,
            "rows": all_results, "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": MODEL_REVISION, "checkpoint_config_sha256": checkpoint_hashes,
            "precision": "4-bit weights; BF16 compute; eager attention",
            "seed": 20261021, "layers_1_based": list(LAYERS_1_BASED),
            "logit_bias": LOGIT_BIAS, "temporal_patch_size": int(processor.video_processor.temporal_patch_size),
            "hardware": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "free_vram_before_load_gib": free/1024**3, "total_vram_gib": total/1024**3,
                "host_available_before_load_gib": host_free/1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3},
            "source_sha256": source_hashes, "elapsed_seconds": time.monotonic()-started,
            "limitations": ["n=1 smoke or n=3 descriptive only", "claim was generated by same model and is not human-verified",
                "claim is deterministically re-tokenized, not original generation IDs",
                "query selection uses baseline attention to the teacher-forced claim tokens (self-conditioned)",
                "frame-order reversal is a temporal/OOD perturbation", "fixed top-10% and +ln(2), no tuning",
                "likelihood sensitivity does not establish semantic support or direct faithfulness"]}
        write_private_json_exclusive(output, payload)
        print(json.dumps({"status": payload["status"], "effective_rows": len(all_results),
            "completed_forwards": completed, "output_name": output.name}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema_version": 1, "status": "failed_claim_token_key_order_interaction",
            "stage": stage, "exception_type": type(exc).__name__,
            "failure_code": _safe_failure_code(exc), "started_utc": started_utc,
            "mode": args.mode, "requested_rows": 1 if args.mode == "smoke" else 3,
            "effective_rows": 0, "attempted_forwards": attempted,
            "completed_forwards": completed, "cohort_lock_sha256": locals().get("lock_hash"),
            "pair_filter_sha256": locals().get("pair_hash"),
            "claim_packet_sha256": locals().get("packet_hash"),
            "claim_packet_hashes": packet_hashes,
            "source_video_sha256": locals().get("video_hash"),
            "row_token_sha256": row_token, "direct_faithfulness": False,
            "human_review_performed": False, "rights_status": "unverified; local-only"}
        try:
            write_private_json_exclusive(output, failure)
        except FileExistsError:
            pass
        print(json.dumps({"status": failure["status"], "stage": stage,
            "failure_code": failure["failure_code"], "completed_forwards": completed,
            "output_name": output.name}, sort_keys=True))
        return 2


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"),
                        default="preflight")
    parser.add_argument("--cohort-lock", default=LOCK_REL,
                        help="private lock JSON path relative to repository")
    parser.add_argument("--pair-filter", default=PAIR_REL,
                        help="private pair-filter JSON path relative to repository")
    parser.add_argument("--claim-packet", action="append", default=None,
                        help="private packet JSON path; repeat for split packets")
    parser.add_argument("--output", required=True,
                        help="new owner-only receipt under data/private_nextgqa/")
    parser.add_argument("--smoke-receipt",
                        help="full mode only: private smoke receipt reviewed by the primary")
    parser.add_argument("--smoke-receipt-sha256",
                        help="SHA256 supplied after the primary validates that smoke receipt")
    return parser


def _cli() -> int:
    args = _argument_parser().parse_args()
    if args.claim_packet is None:
        args.claim_packet = [PACKET_REL]
    if args.mode != "preflight":
        if args.mode == "full" and (not args.smoke_receipt or not _is_sha(args.smoke_receipt_sha256)):
            raise ValueError("full_mode_requires_primary_validated_smoke_digest")
        return _run_gpu(args)
    repo = Path(__file__).resolve().parents[1]
    lock, lock_hash = _read_private_json(repo, args.cohort_lock)
    pair_filter, pair_hash = _read_private_json(repo, args.pair_filter)
    packets, packet_hashes = [], {}
    for rel in args.claim_packet:
        payload, digest = _read_private_json(repo, rel)
        packets.append(payload)
        packet_hashes[Path(rel).name] = digest
    records = validate_locked_claim_packets(lock=lock, pair_filter=pair_filter,
                                             packet_payloads=packets)
    source_names = (
        "scripts/run_claim_token_key_order_interaction.py",
        "scripts/qta_multiquery_video_key_intervention.py",
        "scripts/qta_attention_forward_capture.py",
        "scripts/qwen_question_temporal_attention_selector.py",
        "scripts/qwen_frozen_frame_window_groups.py",
        "scripts/run_saved_claim_frame_permutation_attention_smoke.py",
        "scripts/claim_mask_video_decoder.py",
    )
    source_hashes = {name: sha256_bytes((repo / name).read_bytes()) for name in source_names}
    receipt = build_preflight_receipt(lock_hash=lock_hash,
        pair_filter_hash=pair_hash, packet_hashes=packet_hashes,
        row_count=len(records), source_hashes=source_hashes,
        packet_declared_video_hashes=[record["video_sha256"] for record in records])
    output = validate_private_output(args.output, repo)
    if output.parent != repo / "data/private_nextgqa" or output.suffix != ".json":
        raise ValueError("preflight_output_must_be_json_child_of_private_data_root")
    write_private_json_exclusive(output, receipt)
    print(json.dumps({"status": receipt["status"], "validated_rows": 3,
        "gpu_execution_performed": False, "output_sha256": sha256_bytes(output.read_bytes())},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
