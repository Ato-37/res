#!/usr/bin/env python3
"""Score one frozen claim under original/frame-mean video, with/without key bias.

This is a local claim-token likelihood interaction proxy. It is not a direct
measure of semantic explanation faithfulness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from private_output_paths import validate_private_output
from run_saved_claim_frame_mean_control import make_frame_mean_control
from run_saved_claim_frame_permutation_attention_smoke import (
    FRAME_COUNT, LABELS, LAYERS_1_BASED, LOGIT_BIAS, MODEL_REVISION,
    MIN_FREE_VRAM, MIN_FORWARD_FREE_VRAM, MIN_HOST_AVAILABLE,
    _host_available_bytes, _prepare_processor_batch,
    _score_branch, _tensor_sha256, _write_receipt_exclusive,
)
from run_saved_claim_video_content_dependence import resolve_claim_prompt_suffix

PRIVATE = Path("data/private_nextgqa")
LOCK_PURPOSE = "exploratory_saved_claim_video_content_frame_mean_split_check"
PACKET_PURPOSE = "private_local_nextgqa_model_generated_claim_review_packet"
PROMPT_VERSION = "v2_short"
LIMITATIONS = [
    "per-frame RGB-mean video is an out-of-distribution control that removes spatial layout and other pixel statistics",
    "global visual-key bias is not claim-specific or independently grounded",
    "claim text is model-generated and is re-tokenized for teacher-forced scoring",
    "interaction measures conditional claim-token likelihood sensitivity, not claim truth or semantic support",
    "local exposure scans cannot establish complete historical or pretrained-model non-exposure",
    "human review was not performed; rights remain unverified; inputs and receipts stay local",
]
DIRECT_FAITHFULNESS = False


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_private_json(repo: Path, relative: Path) -> tuple[dict, str, Path]:
    if relative.is_absolute() or not relative.is_relative_to(PRIVATE):
        raise ValueError("private_input_path_invalid")
    root = (repo / PRIVATE).resolve(strict=True)
    lexical = repo / relative
    if lexical.is_symlink():
        raise ValueError("private_input_symlink_rejected")
    path = lexical.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("private_input_path_invalid")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
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
        raise ValueError("private_json_object_required")
    return payload, sha256_bytes(raw), path


def validate_frozen_membership(lock: dict, lock_sha: str,
                               pair_filter: dict, pair_filter_sha: str,
                               row: dict, row_ordinal: int,
                               record_index: int) -> dict:
    assignments = lock.get("assignments")
    requested = lock.get("requested_videos")
    effective = lock.get("effective_videos")
    if (lock.get("status") != "frozen_before_claim_generation_and_scoring"
            or lock.get("purpose") != LOCK_PURPOSE
            or lock.get("dataset_split") != "test"
            or type(requested) is not int or type(effective) is not int
            or requested != effective or not isinstance(assignments, list)
            or len(assignments) != effective or effective < 1
            or type(row_ordinal) is not int or not 0 <= row_ordinal < effective
            or type(record_index) is not int or record_index < 0
            or lock.get("selected_pair_filter_sha256") != pair_filter_sha
            or pair_filter.get("schema_version") != 1
            or not isinstance(pair_filter.get("pair_sha256"), list)
            or len(pair_filter["pair_sha256"]) != effective):
        raise ValueError("frozen_lock_pair_filter_contract_invalid")
    # The freezer builds assignments and selected pair hashes from the same
    # ordered `selected` sequence. Preserve that ordinal contract; membership
    # in the set alone would allow swapping two eligible pairs after freezing.
    hashes = pair_filter["pair_sha256"]
    if (not hashes or any(not isinstance(x, str) or len(x) != 64
                          or any(c not in "0123456789abcdef" for c in x) for x in hashes)
            or len(set(hashes)) != len(hashes)):
        raise ValueError("pair_filter_entries_invalid")
    qid, media, pair = (row.get("dataset_question_id"), row.get("video_sha256"),
                        row.get("candidate_pair_sha256"))
    assignment = assignments[row_ordinal]
    if (not isinstance(qid, str) or not qid or not isinstance(media, str)
            or len(media) != 64 or any(c not in "0123456789abcdef" for c in media)
            or pair != hashes[row_ordinal]
            or assignment.get("media_sha256") != media
            or assignment.get("qid_sha256") != sha256_bytes(qid.encode("utf-8"))):
        raise ValueError("packet_row_not_in_frozen_cohort")
    review_index = row.get("review_index")
    if type(review_index) is not int or review_index != record_index:
        raise ValueError("packet_record_order_binding_invalid")
    return {"row_token_sha256": sha256_bytes((media + qid).encode("utf-8")),
            "source_video_sha256": media,
            "dataset_question_id_sha256": sha256_bytes(qid.encode("utf-8")),
            "candidate_pair_sha256": pair,
            "cohort_lock_sha256": lock_sha,
            "selected_pair_filter_sha256": pair_filter_sha,
            "cohort_size": effective,
            "row_ordinal": row_ordinal}


def validate_packet_payload(payload: dict, record_index: int,
                            expected_split: str = "test") -> dict:
    """Validate a passed/partial packet and return the selected row, even if invalid.

    Returning invalid rows is intentional: after lock binding, the caller records
    a zero-forward failure instead of silently dropping the assigned cohort row.
    """
    records = payload.get("records")
    if (payload.get("status") not in {"passed", "partial", "failed"}
            or payload.get("dataset_split") != expected_split
            or payload.get("model_revision") != MODEL_REVISION
            or payload.get("purpose") != PACKET_PURPOSE
            or payload.get("generation_prompt_version") != PROMPT_VERSION
            or payload.get("claim_review_completed") is not False
            or not isinstance(records, list)
            or type(record_index) is not int or not 0 <= record_index < len(records)):
        raise ValueError("claim_packet_contract_invalid")
    if payload.get("status") in {"passed", "partial"}:
        if any(not isinstance(row, dict) for row in records):
            raise ValueError("claim_packet_records_invalid")
        actual_valid = sum(
            isinstance(row.get("generated_parsed"), dict)
            and row["generated_parsed"].get("valid") is True
            and isinstance(row["generated_parsed"].get("claim"), str)
            and bool(row["generated_parsed"]["claim"].strip())
            for row in records)
        declared_valid = payload.get("fully_structured_claim_records")
        if type(declared_valid) is not int or actual_valid != declared_valid:
            raise ValueError("claim_packet_valid_count_mismatch")
    row = records[record_index]
    if not isinstance(row, dict):
        raise ValueError("claim_packet_selected_row_invalid")
    review_index = row.get("review_index")
    if type(review_index) is not int or review_index != record_index:
        raise ValueError("packet_record_order_binding_invalid")
    if row.get("generation_prompt_version") != PROMPT_VERSION:
        raise ValueError("packet_record_prompt_version_mismatch")
    return row


def validate_packet_filter_binding(payload: dict, pair_filter: dict,
                                   pair_filter_sha: str, row: dict) -> dict:
    """Bind the generator's batch-specific allowlist to its packet and row."""
    values = pair_filter.get("pair_sha256")
    if (pair_filter.get("schema_version") != 1 or not isinstance(values, list)
            or not values or any(not isinstance(x, str) or len(x) != 64
                or any(c not in "0123456789abcdef" for c in x) for x in values)
            or len(set(values)) != len(values)):
        raise ValueError("packet_pair_filter_contract_invalid")
    canonical = hashlib.sha256(json.dumps(sorted(values), ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if (payload.get("candidate_pair_filter_sha256") != pair_filter_sha
            or payload.get("candidate_pair_filter_digest") != canonical
            or payload.get("candidate_pair_filter_entries") != len(values)
            or row.get("candidate_pair_sha256") not in set(values)):
        raise ValueError("packet_pair_filter_binding_invalid")
    return {"packet_pair_filter_sha256": pair_filter_sha,
            "packet_pair_filter_digest": canonical,
            "packet_pair_filter_entries": len(values)}


def compute_interaction(original_base: float, mean_base: float,
                        original_biased: float, mean_biased: float) -> dict:
    scores = (original_base, mean_base, original_biased, mean_biased)
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(float(x)) for x in scores):
        raise ValueError("branch_logprob_must_be_finite")
    baseline_gap = float(original_base) - float(mean_base)
    biased_gap = float(original_biased) - float(mean_biased)
    return {
        "baseline_original_minus_frame_mean_nats_per_token": baseline_gap,
        "biased_original_minus_frame_mean_nats_per_token": biased_gap,
        "bias_by_frame_mean_interaction_nats_per_token": biased_gap - baseline_gap,
    }


def _validate_score(score: dict, *, target_count: int, biased: bool) -> None:
    if (not isinstance(score, dict)
            or score.get("score_semantics") != "mean conditional logprob of frozen target token IDs"
            or score.get("target_token_count") != target_count
            or not isinstance(score.get("target_logprob_mean"), (int, float))
            or isinstance(score.get("target_logprob_mean"), bool)
            or not math.isfinite(float(score["target_logprob_mean"]))):
        raise ValueError("score_receipt_contract_invalid")
    # The scoring helper returns integer keys in-memory; JSON serialization
    # later converts mapping keys to strings in the persisted receipt.
    expected = {layer: int(biased) for layer in LAYERS_1_BASED}
    if score.get("intervention_hook_invocations_by_layer_1_based") != expected:
        raise ValueError("attention_hook_count_invalid")
    if biased and score.get("visual_key_position_count", 0) < 1:
        raise ValueError("biased_visual_key_set_empty")


def _sanitized_score(score: dict) -> dict:
    """Keep scalar score/provenance fields; omit per-token values from receipts."""
    fields = ("target_token_count", "first_query_position", "last_target_query_position",
              "target_logprob_sum", "target_logprob_mean", "score_semantics",
              "visual_key_position_count", "visual_key_positions_sha256",
              "intervention_hook_invocations_by_layer_1_based")
    clean = {}
    for key in fields:
        value = score.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, dict):
            value = {str(k): v for k, v in value.items()
                     if type(v) is int and v >= 0}
        if isinstance(value, (str, int, float, dict)) and not isinstance(value, bool):
            clean[key] = value
    return clean


def group_score_receipts(scores: dict) -> dict:
    """Group completed condition scores explicitly as baseline/biased."""
    grouped = {"baseline": {}, "biased": {}}
    allowed = {"original_video", "per_frame_rgb_mean_video"}
    for key, score in scores.items():
        if (not isinstance(key, tuple) or len(key) != 2
                or key[0] not in grouped or key[1] not in allowed
                or key[1] in grouped[key[0]]):
            raise ValueError("score_branch_grouping_invalid")
        grouped[key[0]][key[1]] = (_sanitized_score(score) if isinstance(score, dict)
                                    else {"returned_value_type": type(score).__name__})
    return grouped


def run(*, lock_rel: Path, pair_filter_rel: Path, packet_rel: Path,
        packet_pair_filter_rel: Path,
        record_index: int, row_ordinal: int, output_arg: str,
        user_authorized_local_only: bool) -> int:
    repo = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    attempted = completed = 0
    stage = "private_input_validation"
    output = None
    bound = {}
    source_paths = [Path(__file__).resolve(),
        repo / "scripts" / "run_saved_claim_frame_mean_control.py",
        repo / "scripts" / "run_saved_claim_frame_permutation_attention_smoke.py",
        repo / "scripts" / "qta_multiquery_video_key_intervention.py",
        repo / "scripts" / "claim_mask_video_decoder.py",
        repo / "scripts" / "generate_nextgqa_claim_review_pilot.py",
        repo / "scripts" / "run_saved_claim_video_content_dependence.py"]
    source_hashes = {}
    prompt_hash = None
    scores = {}
    try:
        if not user_authorized_local_only:
            raise ValueError("local_only_authorization_flag_required")
        output = validate_private_output(output_arg, repo)
        if output.parent != repo / PRIVATE or output.suffix != ".json":
            raise ValueError("output_must_be_private_json_child")
        source_hashes = {str(path.relative_to(repo)): sha256_file(path)
                         for path in source_paths}
        lock, lock_sha, _lock_path = read_private_json(repo, lock_rel)
        pair_filter, pair_sha, _pair_path = read_private_json(repo, pair_filter_rel)
        packet_pair_filter, packet_pair_sha, _packet_filter_path = read_private_json(
            repo, packet_pair_filter_rel)
        packet_payload, packet_sha, _packet_path = read_private_json(repo, packet_rel)
        row = validate_packet_payload(packet_payload, record_index)
        packet_filter_binding = validate_packet_filter_binding(
            packet_payload, packet_pair_filter, packet_pair_sha, row)
        bound = validate_frozen_membership(lock, lock_sha, pair_filter,
                                           pair_sha, row, row_ordinal, record_index)
        bound.update(packet_filter_binding)
        parsed = row.get("generated_parsed")
        claim = parsed.get("claim") if isinstance(parsed, dict) else None
        video_path = row.get("local_video_path")
        video_hash = row.get("video_sha256")
        if (not isinstance(video_path, str) or not isinstance(video_hash, str)
                or video_hash != bound["source_video_sha256"]):
            raise ValueError("claim_packet_row_invalid")
        claim_valid = (isinstance(parsed, dict) and parsed.get("valid") is True
                       and isinstance(claim, str) and bool(claim.strip()))
        claim_hash = sha256_bytes(claim.encode("utf-8")) if isinstance(claim, str) else None
        from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_BY_VERSION
        prompt_version, prompt_suffix = resolve_claim_prompt_suffix(row, PROMPT_SUFFIX_BY_VERSION)
        question = row.get("question")
        choices = row.get("choices")
        if (not isinstance(question, str) or not question.strip()
                or not isinstance(choices, list) or len(choices) != len(LABELS)
                or any(not isinstance(choice, str) or not choice.strip() for choice in choices)):
            raise ValueError("claim_prompt_source_invalid")
        prompt_text = (question + "\nOptions:\n"
            + "\n".join(f"{label}. {choice}" for label, choice
                in zip(LABELS, choices, strict=True)) + prompt_suffix)
        prompt_hash = sha256_bytes(prompt_text.encode("utf-8"))
        if not claim_valid:
            raise ValueError("selected_claim_invalid")

        stage = "resource_preflight"
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info()
        host_free = _host_available_bytes()
        if free < MIN_FREE_VRAM or host_free < MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_model_resource_guard_failed")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise ValueError("pinned_local_checkpoint_missing")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        frames, decoded = decode_sampled_video(video_path, num_frames=FRAME_COUNT,
            uniform_tolerance_seconds=Fraction(1, 20))
        verify_sampled_frame_pixels(frames, decoded)
        if decoded["source_sha256"] != video_hash:
            raise ValueError("claim_video_digest_mismatch")
        saved_pts = row.get("sampled_frame_relative_pts_seconds")
        actual_pts = [float(Fraction(*x["relative_presentation_time"]))
                      for x in decoded["sampled_frames"]]
        if (len(frames) != FRAME_COUNT or not isinstance(saved_pts, list)
                or len(saved_pts) != FRAME_COUNT or len(actual_pts) != FRAME_COUNT
                or any(abs(float(a) - b) > 1e-6
                       for a, b in zip(saved_pts, actual_pts, strict=True))):
            raise ValueError("saved_claim_sampling_grid_mismatch")
        fps = float(Fraction(*decoded["effective_sample_fps"]))
        mean_frames, rgb_means = make_frame_mean_control(frames)
        original, original_meta = _prepare_processor_batch(
            processor, process_vision_info, row, frames, fps, prompt_suffix, torch)
        mean_video, mean_meta = _prepare_processor_batch(
            processor, process_vision_info, row, mean_frames, fps, prompt_suffix, torch)
        if original_meta != mean_meta or set(original) != set(mean_video):
            raise ValueError("condition_metadata_or_keys_changed")
        for key in original:
            a, b = original[key], mean_video[key]
            if torch.is_tensor(a) and torch.is_tensor(b):
                if tuple(a.shape) != tuple(b.shape):
                    raise ValueError("condition_tensor_shape_changed")
                if key != "pixel_values_videos" and not torch.equal(a, b):
                    raise ValueError("condition_nonvisual_tensor_changed")
            elif a != b:
                raise ValueError("condition_non_tensor_field_changed")
        pixels = {"original_video": _tensor_sha256(original["pixel_values_videos"], torch),
                  "per_frame_rgb_mean_video": _tensor_sha256(mean_video["pixel_values_videos"], torch)}
        if pixels["original_video"] == pixels["per_frame_rgb_mean_video"]:
            raise ValueError("visual_conditions_not_distinct")
        target_ids = tuple(processor.tokenizer(claim, add_special_tokens=False)["input_ids"])
        if (not target_ids or len(target_ids) > 512
                or any(type(token) is not int or token < 0 for token in target_ids)):
            raise ValueError("claim_retokenization_invalid")
        target_sha = sha256_bytes(json.dumps(target_ids, separators=(",", ":"))
                                  .encode("ascii"))

        stage = "load_pinned_scorer"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("model_processor_video_token_mismatch")
        torch.manual_seed(20261021)
        torch.cuda.manual_seed_all(20261021)
        torch.cuda.reset_peak_memory_stats()
        batches = {"original_video": original, "per_frame_rgb_mean_video": mean_video}
        for biased in (False, True):
            for name, batch in batches.items():
                stage = ("biased_" if biased else "baseline_") + name
                current_free, _ = torch.cuda.mem_get_info()
                if current_free < MIN_FORWARD_FREE_VRAM or _host_available_bytes() < MIN_HOST_AVAILABLE:
                    raise RuntimeError("per_forward_resource_guard_failed")
                attempted += 1
                score = _score_branch(model=model, batch_cpu=batch,
                    target_ids=target_ids, video_token_id=int(processor.video_token_id),
                    biased=biased, torch=torch)
                completed += 1
                scores[("biased" if biased else "baseline", name)] = score
                _validate_score(score, target_count=len(target_ids), biased=biased)
        estimate = compute_interaction(
            scores[("baseline", "original_video")]["target_logprob_mean"],
            scores[("baseline", "per_frame_rgb_mean_video")]["target_logprob_mean"],
            scores[("biased", "original_video")]["target_logprob_mean"],
            scores[("biased", "per_frame_rgb_mean_video")]["target_logprob_mean"])

        payload = {
            "schema_version": 1,
            "status": "completed_one_row_frozen_frame_mean_attention_bias_interaction",
            "started_utc": started_utc,
            "dataset_split": "test",
            "post_result_exploration": True,
            "direct_faithfulness": DIRECT_FAITHFULNESS,
            "faithfulness_result": False,
            "human_review": {"review_actually_performed": False},
            "rights_status": "unverified; authorized local-only use",
            **bound,
            "requested_forwards": 4,
            "attempted_forwards": attempted,
            "completed_forwards": completed,
            "private_input_packet_sha256": packet_sha,
            "private_input_packet_name": packet_rel.name,
            "packet_record_index_zero_based": record_index,
            "packet_status": packet_payload["status"],
            "review_index": row["review_index"],
            "row_pair_binding_rule": "pair_filter.pair_sha256[row_ordinal] equals packet candidate pair; lock assignments and selected pair list share frozen selection order",
            "claim_valid": True,
            "saved_claim_text_sha256": claim_hash,
            "prompt_version": prompt_version,
            "prompt_source_text_sha256": prompt_hash,
            "retokenized_claim_token_ids_sha256": target_sha,
            "target_token_count": len(target_ids),
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": MODEL_REVISION,
            "precision": "4-bit weights; BF16 compute; eager attention",
            "seed": 20261021,
            "intervention": {"layers_1_based": list(LAYERS_1_BASED),
                "logit_bias": LOGIT_BIAS,
                "key_set": "all processor-expanded visual-key positions",
                "query_set": "all re-tokenized claim-token prediction rows"},
            "condition_hashes": {"original_processor_pixels_sha256": pixels["original_video"],
                "frame_mean_processor_pixels_sha256": pixels["per_frame_rgb_mean_video"],
                "per_frame_rgb_means": rgb_means,
                "sampled_frame_pts_seconds": actual_pts,
                "target_token_ids_sha256": target_sha},
            "scores": group_score_receipts(scores),
            "primary_interaction": estimate,
            "hardware": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "free_vram_before_load_gib": free / 1024**3,
                "total_vram_gib": total / 1024**3,
                "host_available_before_load_gib": host_free / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
            "source_sha256": source_hashes,
            "input_sha256": {"cohort_lock": lock_sha,
                "selected_pair_filter": pair_sha,
                "packet_pair_filter": packet_pair_sha,
                "claim_packet": packet_sha},
            "elapsed_seconds": time.monotonic() - started,
            "inferential_status": "post-result protocol replication on one fixed row; descriptive only",
            "limitations": LIMITATIONS,
        }
        _write_receipt_exclusive(output, payload)
        print(json.dumps({"status": payload["status"], "completed_forwards": completed,
            "row_ordinal": row_ordinal,
            "interaction_nats_per_token": estimate["bias_by_frame_mean_interaction_nats_per_token"],
            "output_name": output.name}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema_version": 1,
            "status": "failed_frozen_frame_mean_attention_bias_interaction",
            "stage": stage, "exception_type": type(exc).__name__,
            "failure_code": str(exc)[:180], "started_utc": started_utc,
            "dataset_split": "test", "requested_forwards": 4,
            "attempted_forwards": attempted, "completed_forwards": completed,
            "direct_faithfulness": DIRECT_FAITHFULNESS, "faithfulness_result": False,
            "human_review_performed": False,
            "rights_status": "unverified; authorized local-only use",
            "limitations": LIMITATIONS,
            **bound,
            "private_input_packet_sha256": locals().get("packet_sha"),
            "saved_claim_text_sha256": locals().get("claim_hash"),
            "packet_status": locals().get("packet_payload", {}).get("status"),
            "review_index": locals().get("row", {}).get("review_index"),
            "claim_valid": locals().get("claim_valid", False),
            "prompt_source_text_sha256": prompt_hash,
            "completed_score_branches": group_score_receipts(scores),
            "source_sha256": source_hashes}
        if output is not None:
            try:
                _write_receipt_exclusive(output, failure)
            except FileExistsError:
                pass
        print(json.dumps({"status": failure["status"], "stage": stage,
            "completed_forwards": completed, "failure_code": failure["failure_code"],
            "output_name": output.name if output else None}, sort_keys=True))
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True,
        help="private cohort lock, relative to repository root")
    parser.add_argument("--pair-filter", type=Path, required=True,
        help="private selected-pair filter, relative to repository root")
    parser.add_argument("--packet", type=Path, required=True,
        help="private generated-claim packet, relative to repository root")
    parser.add_argument("--packet-pair-filter", type=Path, required=True,
        help="batch-specific allowlist used to generate this packet")
    parser.add_argument("--record-index", type=int, required=True)
    parser.add_argument("--row-ordinal", type=int, required=True,
        help="row's zero-based assignment ordinal in the frozen lock")
    parser.add_argument("--output", required=True,
        help="new receipt path under data/private_nextgqa")
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args()
    return run(lock_rel=args.lock, pair_filter_rel=args.pair_filter,
        packet_rel=args.packet, packet_pair_filter_rel=args.packet_pair_filter,
        record_index=args.record_index,
        row_ordinal=args.row_ordinal, output_arg=args.output,
        user_authorized_local_only=args.user_authorized_local_only)


if __name__ == "__main__":
    raise SystemExit(main())
