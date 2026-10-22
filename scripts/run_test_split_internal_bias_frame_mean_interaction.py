#!/usr/bin/env python3
"""Run one frozen test-split row for an internal-bias × frame-mean interaction.

This is a local, post-result claim-likelihood sensitivity diagnostic, not a
direct faithfulness score. See the dated experiment protocol.
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
from run_saved_claim_frame_mean_control import make_frame_mean_control, _sha_file
from run_saved_claim_frame_permutation_attention_smoke import (
    FRAME_COUNT, LAYERS_1_BASED, LOGIT_BIAS, MODEL_REVISION,
    MIN_FREE_VRAM, MIN_HOST_AVAILABLE, MIN_FORWARD_FREE_VRAM,
    _host_available_bytes, _prepare_processor_batch, _score_branch,
    _sha256_bytes, _tensor_sha256, _write_receipt_exclusive,
)
from run_saved_claim_video_content_dependence import resolve_claim_prompt_suffix

PRIVATE = Path("data/private_nextgqa")
PACKETS = {
    "smoke": (PRIVATE / "claim_generation_test_frame_mean_smoke_20261021_02.json", 0,
              PRIVATE / "saved_claim_frame_mean_test_smoke_20261021_02_retry01.json"),
    "row0": (PRIVATE / "claim_generation_test_frame_mean_remaining_20261021_02.json", 0,
             PRIVATE / "saved_claim_frame_mean_test_remaining_20261021_02_row0.json"),
    "row1": (PRIVATE / "claim_generation_test_frame_mean_remaining_20261021_02.json", 1,
             PRIVATE / "saved_claim_frame_mean_test_remaining_20261021_02_row1.json"),
}
LOCK_REL = PRIVATE / "saved_claim_frame_mean_test_lock_20261021_02.json"
PAIR_FILTER_REL = PRIVATE / "saved_claim_frame_mean_test_pairs_20261021_02.json"


def _read_private_json(path: Path) -> tuple[dict, str]:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("private_receipt_permissions_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def interaction(original_base: float, mean_base: float,
                original_bias: float, mean_bias: float) -> dict:
    values = (original_base, mean_base, original_bias, mean_bias)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in values):
        raise ValueError("branch_logprob_must_be_finite")
    baseline = float(original_base) - float(mean_base)
    biased = float(original_bias) - float(mean_bias)
    return {"baseline_original_minus_frame_mean_nats_per_token": baseline,
            "biased_original_minus_frame_mean_nats_per_token": biased,
            "bias_by_frame_mean_interaction_nats_per_token": biased - baseline}


def verify_locked_assignment(lock: dict, pair_filter_sha: str, global_index: int,
                             row: dict, expected_filter_sha: str) -> None:
    assignments = lock.get("assignments")
    qid = row.get("dataset_question_id")
    video_hash = row.get("video_sha256")
    if (lock.get("status") != "frozen_before_claim_generation_and_scoring"
            or lock.get("dataset_split") != "test"
            or lock.get("requested_videos") != 3
            or lock.get("effective_videos") != 3
            or lock.get("selected_pair_filter_sha256") != expected_filter_sha
            or pair_filter_sha != expected_filter_sha
            or not isinstance(assignments, list) or len(assignments) != 3
            or global_index not in (0, 1, 2)
            or not isinstance(qid, str) or not isinstance(video_hash, str)):
        raise ValueError("frozen_test_lock_contract_invalid")
    assignment = assignments[global_index]
    if (assignment.get("media_sha256") != video_hash
            or assignment.get("qid_sha256") != hashlib.sha256(qid.encode()).hexdigest()):
        raise ValueError("packet_row_not_in_frozen_lock_assignment")


def run(row: str, output_arg: str) -> int:
    repo = Path(__file__).resolve().parents[1]
    packet_rel, record_index, baseline_rel = PACKETS[row]
    output = validate_private_output(output_arg, repo)
    if output.parent != repo / PRIVATE or output.suffix != ".json":
        raise ValueError("private_output_must_be_json_child_of_private_data_root")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    attempted = completed = 0
    stage = "baseline_receipt_join"
    try:
        baseline, baseline_sha = _read_private_json(repo / baseline_rel)
        lock, lock_sha = _read_private_json(repo / LOCK_REL)
        pair_filter_payload, pair_filter_sha = _read_private_json(repo / PAIR_FILTER_REL)
        from run_saved_claim_frame_permutation_attention_smoke import _read_packet
        record, packet_sha = _read_packet(repo, packet_rel, record_index, expected_split="test")
        global_index = {"smoke": 0, "row0": 1, "row1": 2}[row]
        verify_locked_assignment(lock, pair_filter_sha, global_index, record,
                                 "51658b1e778e91966d0b893c5ffe80727c67ff000e82e7807bc7e531300b7c5b")
        claim = record.get("generated_parsed", {}).get("claim")
        video_path = record.get("local_video_path")
        video_hash = record.get("video_sha256")
        row_token = _sha256_bytes((video_hash + str(record.get("dataset_question_id", ""))).encode())
        if (baseline.get("status") != "completed_one_row_posthoc_frame_mean_sensitivity"
                or baseline.get("dataset_split") != "test"
                or baseline.get("completed_forwards") != 2
                or baseline.get("direct_faithfulness") is not False
                or baseline.get("human_review", {}).get("review_actually_performed") is not False
                or baseline.get("row_token_sha256") != row_token
                or baseline.get("private_input_packet_sha256") != packet_sha
                or baseline.get("private_input_packet_name") != packet_rel.name
                or baseline.get("packet_record_index_zero_based") != record_index
                or baseline.get("source_video_sha256") != video_hash
                or baseline.get("saved_claim_text_sha256") != _sha256_bytes(claim.encode())):
            raise ValueError("baseline_receipt_packet_row_join_invalid")

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
        from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_BY_VERSION

        stage = "reconstruct_matched_inputs"
        prompt_version, prompt_suffix = resolve_claim_prompt_suffix(record, PROMPT_SUFFIX_BY_VERSION)
        if prompt_version != "v2_short":
            raise ValueError("locked_test_prompt_version_mismatch")
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        frames, decoded = decode_sampled_video(video_path, num_frames=FRAME_COUNT,
            uniform_tolerance_seconds=Fraction(1, 20))
        verify_sampled_frame_pixels(frames, decoded)
        if decoded["source_sha256"] != video_hash:
            raise ValueError("claim_video_digest_mismatch")
        saved_pts = record.get("sampled_frame_relative_pts_seconds")
        pts = [float(Fraction(*item["relative_presentation_time"])) for item in decoded["sampled_frames"]]
        if (len(frames) != FRAME_COUNT or not isinstance(saved_pts, list) or len(saved_pts) != FRAME_COUNT
                or len(pts) != FRAME_COUNT or any(abs(float(a)-b) > 1e-6 for a, b in zip(saved_pts, pts, strict=True))):
            raise ValueError("saved_claim_sampling_grid_mismatch")
        fps = float(Fraction(*decoded["effective_sample_fps"]))
        means, rgb_means = make_frame_mean_control(frames)
        original, original_meta = _prepare_processor_batch(processor, process_vision_info, record, frames, fps, prompt_suffix, torch)
        control, mean_meta = _prepare_processor_batch(processor, process_vision_info, record, means, fps, prompt_suffix, torch)
        if original_meta != mean_meta or set(original) != set(control):
            raise ValueError("condition_metadata_or_keys_changed")
        for key in original:
            a, b = original[key], control[key]
            if torch.is_tensor(a) and torch.is_tensor(b):
                if tuple(a.shape) != tuple(b.shape):
                    raise ValueError("condition_tensor_shape_changed")
                if key != "pixel_values_videos" and not torch.equal(a, b):
                    raise ValueError("condition_nonvisual_tensor_changed")
            elif a != b:
                raise ValueError("condition_non_tensor_field_changed")
        original_pixels = _tensor_sha256(original["pixel_values_videos"], torch)
        mean_pixels = _tensor_sha256(control["pixel_values_videos"], torch)
        baseline_scores = baseline.get("branch_target_scores", {})
        base_original = baseline_scores.get("original_video", {})
        base_mean = baseline_scores.get("per_frame_rgb_mean_video", {})
        if (original_pixels != baseline.get("condition", {}).get("original_processor_pixels_sha256")
                or mean_pixels != baseline.get("condition", {}).get("frame_mean_processor_pixels_sha256")
                or base_original.get("visual_key_positions_sha256") != base_mean.get("visual_key_positions_sha256")
                or base_original.get("target_token_count") != baseline.get("target_token_count")):
            raise ValueError("baseline_processor_condition_binding_mismatch")
        target_ids = tuple(processor.tokenizer(claim, add_special_tokens=False)["input_ids"])
        target_hash = _sha256_bytes(json.dumps(target_ids, separators=(",", ":")).encode("ascii"))
        if (target_hash != baseline.get("retokenized_claim_ids_sha256")
                or len(target_ids) != baseline.get("target_token_count")):
            raise ValueError("baseline_target_token_binding_mismatch")

        stage = "load_pinned_scorer"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("model_processor_video_token_mismatch")
        torch.manual_seed(20261021)
        torch.cuda.manual_seed_all(20261021)
        torch.cuda.reset_peak_memory_stats()
        biased = {}
        for name, batch in (("original_video", original), ("per_frame_rgb_mean_video", control)):
            stage = "biased_" + name
            free_now, _ = torch.cuda.mem_get_info()
            if free_now < MIN_FORWARD_FREE_VRAM or _host_available_bytes() < MIN_HOST_AVAILABLE:
                raise RuntimeError("per_forward_resource_guard_failed")
            attempted += 1
            biased[name] = _score_branch(model=model, batch_cpu=batch, target_ids=target_ids,
                video_token_id=int(processor.video_token_id), biased=True, torch=torch)
            completed += 1
        result = interaction(base_original["target_logprob_mean"], base_mean["target_logprob_mean"],
            biased["original_video"]["target_logprob_mean"], biased["per_frame_rgb_mean_video"]["target_logprob_mean"])
        source_files = [Path(__file__).resolve(), Path(__file__).resolve().parent / "run_saved_claim_frame_mean_control.py",
            Path(__file__).resolve().parent / "run_saved_claim_frame_permutation_attention_smoke.py",
            Path(__file__).resolve().parent / "qta_multiquery_video_key_intervention.py",
            Path(__file__).resolve().parent / "claim_mask_video_decoder.py",
            Path(__file__).resolve().parent / "generate_nextgqa_claim_review_pilot.py"]
        payload = {"schema_version": 1, "status": "completed_one_row_test_split_internal_bias_frame_mean_interaction",
            "dataset_split": "test", "post_result_exploration": True, "direct_faithfulness": False,
            "human_review": {"review_actually_performed": False, "collection_waived_by_user_for_this_exploratory_run": True},
            "rights_status": "unverified; local-only per user authorization", "requested_records": 1, "effective_records": 1,
            "requested_forwards": 2, "attempted_forwards": attempted, "completed_forwards": completed,
            "row_token_sha256": row_token, "source_video_sha256": video_hash, "private_input_packet_sha256": packet_sha,
            "private_input_packet_name": packet_rel.name, "packet_record_index_zero_based": record_index,
            "baseline_receipt_name": baseline_rel.name, "baseline_receipt_sha256": baseline_sha,
            "cohort_lock_name": LOCK_REL.name, "cohort_lock_sha256": lock_sha,
            "selected_pair_filter_name": PAIR_FILTER_REL.name,
            "selected_pair_filter_sha256": pair_filter_sha,
            "saved_claim_text_sha256": _sha256_bytes(claim.encode()), "retokenized_claim_ids_sha256": target_hash,
            "target_token_count": len(target_ids), "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": MODEL_REVISION,
            "precision": "4-bit weights; BF16 compute; eager attention", "seed": 20261021,
            "prompt_version": prompt_version, "intervention": {"layers_1_based": list(LAYERS_1_BASED),
                "logit_bias": LOGIT_BIAS, "keys": "all processor-expanded visual keys",
                "queries": "all re-tokenized claim-token prediction rows"},
            "condition_hashes": {"original_processor_pixels_sha256": original_pixels,
                "frame_mean_processor_pixels_sha256": mean_pixels, "per_frame_rgb_means": rgb_means,
                "target_token_ids_sha256": target_hash}, "baseline_scores": {
                "original_video_mean": base_original["target_logprob_mean"],
                "frame_mean_video_mean": base_mean["target_logprob_mean"]},
            "biased_scores": biased, "primary_interaction": result,
            "hardware": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "free_vram_before_load_gib": free / 1024**3, "total_vram_gib": total / 1024**3,
                "host_available_before_load_gib": host_free / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
            "source_sha256": {str(p.relative_to(repo)): _sha_file(p) for p in source_files},
            "elapsed_seconds": time.monotonic() - started,
            "inferential_status": "post-result exploratory fixed test-split n=3; finite-cohort descriptions only",
            "limitations": ["per-frame RGB-mean control alters spatial structure and is out of distribution",
                "global attention bias is not claim-specific or human-grounded", "same-model generated claims and re-tokenized token IDs",
                "test exposure history is incomplete; no independence or population claim", "no human review was performed",
                "this interaction does not establish claim truth or improved explanation faithfulness",
                "rights unverified; protected inputs and detailed outputs remain local"]}
        _write_receipt_exclusive(output, payload)
        print(json.dumps({"status": payload["status"], "completed_forwards": completed,
            "interaction_nats_per_token": result["bias_by_frame_mean_interaction_nats_per_token"],
            "output_name": output.name}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema_version": 1, "status": "failed_test_split_internal_bias_frame_mean_interaction",
            "stage": stage, "exception_type": type(exc).__name__, "failure_code": str(exc)[:180],
            "started_utc": started_utc, "dataset_split": "test", "requested_forwards": 2,
            "attempted_forwards": attempted, "completed_forwards": completed, "direct_faithfulness": False,
            "human_review_performed": False, "rights_status": "unverified; local-only",
            "row_token_sha256": locals().get("row_token"), "source_video_sha256": locals().get("video_hash"),
            "private_input_packet_sha256": locals().get("packet_sha")}
        try:
            _write_receipt_exclusive(output, failure)
        except FileExistsError:
            pass
        print(json.dumps({"status": failure["status"], "stage": stage, "completed_forwards": completed,
            "failure_code": failure["failure_code"], "output_name": output.name}, sort_keys=True))
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", choices=tuple(PACKETS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args()
    return run(args.row, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
