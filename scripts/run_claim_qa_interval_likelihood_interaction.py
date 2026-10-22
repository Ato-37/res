#!/usr/bin/env python3
"""Post-result local diagnostic: claim likelihood × QA interval mask × visual bias."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time

from private_output_paths import validate_private_output
from run_saved_claim_frame_permutation_attention_smoke import (
    FRAME_COUNT, LAYERS_1_BASED, LOGIT_BIAS, MIN_FREE_VRAM,
    MIN_HOST_AVAILABLE, MODEL_REVISION, PACKET_REL, REMAINING_PACKET_REL,
    _host_available_bytes, _prepare_processor_batch, _read_packet, _score_branch,
    _sha256_bytes, _sha256_file, _tensor_sha256,
)
from presentation_time_utils import strict_evidence_indices_from_times
from necessity_choice_margin_utils import apply_mask_fill, nearest_control_indices


def index_map(packet_rel: Path, record_index: int) -> int:
    if type(record_index) is not int or record_index < 0:
        raise ValueError("record_index_invalid")
    if packet_rel == PACKET_REL and record_index == 0:
        return 0
    if packet_rel == REMAINING_PACKET_REL and record_index in (0, 1):
        return record_index + 1
    raise ValueError("record_not_in_frozen_three_row_cohort")


def difference_in_differences(control_base, evidence_base, control_bias, evidence_bias):
    values = (control_base, evidence_base, control_bias, evidence_bias)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in values):
        raise ValueError("nonfinite_condition_score")
    base_sensitivity = float(control_base) - float(evidence_base)
    biased_sensitivity = float(control_bias) - float(evidence_bias)
    return {"baseline_control_minus_qa_interval_nats_per_token": base_sensitivity,
            "biased_control_minus_qa_interval_nats_per_token": biased_sensitivity,
            "bias_by_qa_interval_interaction_nats_per_token": biased_sensitivity - base_sensitivity}


def _score_one(model, batch, target_ids, video_token_id, biased, torch):
    import gc
    result = _score_branch(model=model, batch_cpu=batch, target_ids=target_ids,
                          video_token_id=video_token_id, biased=biased, torch=torch)
    gc.collect()
    return result


def run(output_arg: str, packet_rel: Path, record_index: int) -> int:
    repo = Path(__file__).resolve().parents[1]
    output = validate_private_output(output_arg, repo)
    if output.parent != repo / "data" / "private_nextgqa" or output.suffix != ".json":
        raise ValueError("private_output_must_be_json_child_of_private_data_root")
    started = time.monotonic(); started_utc = datetime.now(timezone.utc).isoformat()
    attempted = completed = 0; stage = "private_packet_validation"
    try:
        global_index = index_map(packet_rel, record_index)
        record, packet_hash = _read_packet(repo, packet_rel, record_index)
        claim = record.get("generated_parsed", {}).get("claim")
        video_path, video_hash = record.get("local_video_path"), record.get("video_sha256")
        spans, review = record.get("gold_interval_seconds"), record.get("human_review")
        if (not isinstance(claim, str) or not claim.strip() or not isinstance(video_path, str)
                or not isinstance(video_hash, str) or len(video_hash) != 64
                or not isinstance(spans, list) or not spans or not isinstance(review, dict)
                or review.get("claim_visually_supported_by_reported_frames") is not None):
            raise ValueError("frozen_claim_row_or_qa_interval_invalid")
        row_token = _sha256_bytes((video_hash + str(record.get("dataset_question_id", ""))).encode())
        stage = "cuda_resource_preflight"
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info(); host_free = _host_available_bytes()
        if free < MIN_FREE_VRAM or host_free < MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_model_resource_guard_failed")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise ValueError("pinned_local_checkpoint_missing")
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
        from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_BY_VERSION
        from run_saved_claim_video_content_dependence import resolve_claim_prompt_suffix
        prompt_version, prompt_suffix = resolve_claim_prompt_suffix(record, PROMPT_SUFFIX_BY_VERSION)

        stage = "decode_and_bind_video"
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        frames, decoded = decode_sampled_video(video_path, num_frames=FRAME_COUNT,
            uniform_tolerance_seconds=Fraction(1, 20))
        verify_sampled_frame_pixels(frames, decoded)
        if decoded["source_sha256"] != video_hash:
            raise ValueError("claim_video_digest_mismatch")
        saved_pts = record.get("sampled_frame_relative_pts_seconds")
        pts = [Fraction(*item["relative_presentation_time"]) for item in decoded["sampled_frames"]]
        if (len(frames) != FRAME_COUNT or not isinstance(saved_pts, list)
                or len(saved_pts) != FRAME_COUNT
                or any(abs(float(saved) - float(actual)) > 1e-6
                       for saved, actual in zip(saved_pts, pts, strict=True))):
            raise ValueError("saved_claim_sampling_grid_mismatch")
        indices = list(range(FRAME_COUNT))
        evidence = strict_evidence_indices_from_times(indices, pts, spans)
        control = nearest_control_indices(indices, evidence)
        if (not evidence or not control or len(control) != len(evidence)
                or set(evidence) & set(control)):
            raise ValueError("qa_interval_equal_count_control_unavailable")
        fps = float(Fraction(*decoded["effective_sample_fps"]))
        evidence_frames = [apply_mask_fill(frame, "black") if i in set(evidence) else frame.copy()
                           for i, frame in enumerate(frames)]
        control_frames = [apply_mask_fill(frame, "black") if i in set(control) else frame.copy()
                          for i, frame in enumerate(frames)]
        stage = "prepare_matched_batches"
        evidence_batch, evidence_meta = _prepare_processor_batch(
            processor, process_vision_info, record, evidence_frames, fps, prompt_suffix, torch)
        control_batch, control_meta = _prepare_processor_batch(
            processor, process_vision_info, record, control_frames, fps, prompt_suffix, torch)
        if evidence_meta != control_meta or set(evidence_batch) != set(control_batch):
            raise ValueError("masked_condition_metadata_changed")
        for key in evidence_batch:
            left, right = evidence_batch[key], control_batch[key]
            if torch.is_tensor(left) and torch.is_tensor(right):
                if tuple(left.shape) != tuple(right.shape):
                    raise ValueError("masked_condition_tensor_shape_changed")
                if key != "pixel_values_videos" and not torch.equal(left, right):
                    raise ValueError("masked_condition_nonvisual_tensor_changed")
            elif left != right:
                raise ValueError("masked_condition_non_tensor_field_changed")
        evidence_pixels = _tensor_sha256(evidence_batch["pixel_values_videos"], torch)
        control_pixels = _tensor_sha256(control_batch["pixel_values_videos"], torch)
        if evidence_pixels == control_pixels:
            raise ValueError("masked_condition_pixels_not_distinct")
        target_ids = tuple(processor.tokenizer(claim, add_special_tokens=False)["input_ids"])
        if not target_ids or len(target_ids) > 512:
            raise ValueError("saved_claim_retokenization_invalid")
        target_hash = _sha256_bytes(json.dumps(target_ids, separators=(",", ":")).encode("ascii"))

        stage = "load_pinned_scorer"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("model_processor_video_token_mismatch")
        torch.manual_seed(20261021); torch.cuda.manual_seed_all(20261021); torch.cuda.reset_peak_memory_stats()
        scores = {}
        for condition, batch, biased in (("qa_interval_baseline", evidence_batch, False),
                ("control_baseline", control_batch, False),
                ("qa_interval_biased", evidence_batch, True),
                ("control_biased", control_batch, True)):
            stage = "score_" + condition
            free_now, _ = torch.cuda.mem_get_info()
            if free_now < 2 * 1024**3 or _host_available_bytes() < MIN_HOST_AVAILABLE:
                raise RuntimeError("per_forward_resource_guard_failed")
            attempted += 1
            scores[condition] = _score_one(model, batch, target_ids,
                int(processor.video_token_id), biased, torch)
            completed += 1
        result = difference_in_differences(
            scores["control_baseline"]["target_logprob_mean"],
            scores["qa_interval_baseline"]["target_logprob_mean"],
            scores["control_biased"]["target_logprob_mean"],
            scores["qa_interval_biased"]["target_logprob_mean"])
        source_files = [Path(__file__).resolve(), Path(__file__).resolve().parent / "run_saved_claim_frame_permutation_attention_smoke.py",
            Path(__file__).resolve().parent / "qta_multiquery_video_key_intervention.py",
            Path(__file__).resolve().parent / "claim_mask_video_decoder.py",
            Path(__file__).resolve().parent / "necessity_choice_margin_utils.py",
            Path(__file__).resolve().parent / "presentation_time_utils.py",
            Path(__file__).resolve().parent / "generate_nextgqa_claim_review_pilot.py"]
        payload = {"schema_version": 1, "status": "completed_one_row_exploratory_qa_interval_claim_likelihood_interaction",
            "started_utc": started_utc, "post_result_exploration": True,
            "purpose": "re-tokenized saved-claim likelihood response to QA interval versus equal-count temporal control masks, with/without fixed visual-key bias",
            "direct_faithfulness": False, "human_review": {"review_actually_performed": False,
                "collection_waived_by_user_for_this_exploratory_run": True},
            "rights_status": "unverified; local-only per user authorization",
            "requested_records": 1, "effective_records": 1, "requested_forwards": 4,
            "attempted_forwards": attempted, "completed_forwards": completed,
            "frozen_global_row_index": global_index, "row_token_sha256": row_token,
            "source_video_sha256": video_hash, "private_input_packet_sha256": packet_hash,
            "private_input_packet_name": packet_rel.name,
            "generation_prompt_version": prompt_version,
            "prompt_suffix_sha256": _sha256_bytes(prompt_suffix.encode("utf-8")),
            "saved_claim_text_sha256": _sha256_bytes(claim.encode("utf-8")),
            "reencoded_claim_token_ids_sha256": target_hash, "target_token_count": len(target_ids),
            "qa_interval_frame_count": len(evidence), "control_frame_count": len(control),
            "qa_interval_frame_indices_sha256": _sha256_bytes(json.dumps(evidence,separators=(",", ":")).encode("ascii")),
            "control_frame_indices_sha256": _sha256_bytes(json.dumps(control,separators=(",", ":")).encode("ascii")),
            "attention_intervention": {"layers_1_based": list(LAYERS_1_BASED), "logit_bias": LOGIT_BIAS,
                "key_set": "all processor-expanded visual-token positions; includes masked-frame tokens",
                "query_set": "all re-tokenized claim-token prediction rows"},
            "condition_scores": scores, "primary_descriptive_interaction": result,
            "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": MODEL_REVISION,
            "precision": "4-bit weights; BF16 compute; eager attention", "seed": 20261021,
            "processor_pixels_sha256": {"qa_interval_mask": evidence_pixels, "control_mask": control_pixels},
            "hardware": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "free_vram_before_load_gib": free/1024**3, "total_vram_gib": total/1024**3,
                "host_available_before_load_gib": host_free/1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3},
            "source_sha256": {str(path.relative_to(repo)): _sha256_file(path) for path in source_files},
            "elapsed_seconds": time.monotonic()-started,
            "inferential_status": "post-result fixed n=3 cohort; finite descriptive values only; no CI or test",
            "limitations": ["QA timing annotations are not claim-specific evidence", "same selected validation cohort and post-result endpoint",
                "same-model generated claims; target IDs are re-tokenized", "black fill is out-of-distribution",
                "global bias affects all visual tokens including masked-frame tokens", "not direct faithfulness",
                "human review not performed", "rights unverified; private local-only"]}
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
        fd = os.open(output, os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
        with os.fdopen(fd,"wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({"status": payload["status"], "completed_forwards": completed,
            "interaction_nats_per_token": result["bias_by_qa_interval_interaction_nats_per_token"],
            "output_name": output.name}, sort_keys=True)); return 0
    except Exception as exc:
        failure = {"schema_version": 1, "status": "failed_exploratory_qa_interval_claim_likelihood_interaction",
            "stage": stage, "exception_type": type(exc).__name__, "failure_code": str(exc)[:160],
            "started_utc": started_utc, "requested_records": 1, "effective_records": 0,
            "requested_forwards": 4, "attempted_forwards": attempted, "completed_forwards": completed,
            "direct_faithfulness": False, "human_review_performed": False,
            "rights_status": "unverified; local-only", "row_token_sha256": locals().get("row_token"),
            "source_video_sha256": locals().get("video_hash"),
            "private_input_packet_sha256": locals().get("packet_hash")}
        try:
            raw = (json.dumps(failure,sort_keys=True,separators=(",", ":"))+"\n").encode()
            fd = os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
            with os.fdopen(fd,"wb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        except FileExistsError:
            pass
        print(json.dumps({"status":failure["status"],"stage":stage,"completed_forwards":completed,
            "failure_code":failure["failure_code"],"output_name":output.name},sort_keys=True)); return 2


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",required=True)
    parser.add_argument("--packet",choices=("smoke","remaining"),default="smoke")
    parser.add_argument("--record-index",type=int,default=0)
    parser.add_argument("--user-authorized-local-only",action="store_true",required=True)
    args=parser.parse_args()
    return run(args.output,PACKET_REL if args.packet=="smoke" else REMAINING_PACKET_REL,args.record_index)


if __name__ == "__main__":
    raise SystemExit(main())
