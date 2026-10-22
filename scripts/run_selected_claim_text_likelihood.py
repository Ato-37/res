#!/usr/bin/env python3
"""Score saved claim text under full, claim-window-mask, and matched-control videos.

This is an exploratory conditional-likelihood diagnostic. The claim text is not
human verified, and its likelihood does not establish truth or faithfulness.
All receipts are private and local.
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
import time

from private_output_paths import validate_private_output
from run_machine_localized_claim_mask_pilot import (
    FRAME_COUNT as SELECTOR_FRAME_COUNT, MODEL_REVISION, MIN_FREE_VRAM, MIN_HOST_AVAILABLE,
    TOLERANCE, _exact_selected_window_interval, _host_available_bytes,
    _read_selected_window_records, _require_model_load_resources,
)
from run_saved_claim_frame_permutation_attention_smoke import (
    _build_prompt, _score_branch, _sha256_file, validate_video_metadata,
)


CONDITIONS = ("full", "claim_window_black", "temporal_control_black")
CLAIM_FRAME_COUNT = 16


def claim_window_contrasts(scores: dict[str, float]) -> dict[str, float]:
    """Return nat/token differences with positive meaning greater claim support."""
    if set(scores) != set(CONDITIONS):
        raise ValueError("claim_score_conditions_invalid")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in scores.values()):
        raise ValueError("claim_score_nonfinite")
    return {
        "control_minus_selected_mask_nats_per_token":
            float(scores["temporal_control_black"] - scores["claim_window_black"]),
        "full_minus_selected_mask_nats_per_token":
            float(scores["full"] - scores["claim_window_black"]),
    }


def validate_saved_generation_pts(saved_pts, actual_pts) -> bool:
    """Fail closed on malformed/nonfinite saved PTS before numeric comparison."""
    if (not isinstance(saved_pts, list) or len(saved_pts) != CLAIM_FRAME_COUNT
            or not isinstance(actual_pts, (list, tuple))
            or len(actual_pts) != CLAIM_FRAME_COUNT):
        return False
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in saved_pts):
        return False
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in actual_pts):
        return False
    return all(abs(float(saved) - float(actual)) <= 1e-6
               for saved, actual in zip(saved_pts, actual_pts, strict=True))


def _batch_for_video(processor, process_vision_info, record, frames, sample_fps,
                     prompt_suffix, torch):
    rendered, messages = _build_prompt(processor, record, frames,
                                       sample_fps, prompt_suffix)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or not isinstance(videos, list) or len(videos) != 1:
        raise ValueError("claim_score_video_preprocess_invalid")
    pixels, metadata = videos[0]
    if int(pixels.shape[0]) != CLAIM_FRAME_COUNT:
        raise ValueError("claim_score_frame_count_invalid")
    validate_video_metadata(metadata, sample_fps=sample_fps, frame_count=CLAIM_FRAME_COUNT)
    batch = processor(text=[rendered], images=images, videos=[pixels],
        video_metadata=[metadata], do_sample_frames=False, cap_pixels_per_frame=True,
        padding=True, return_tensors="pt")
    return batch, metadata


def _build_sampled_conditions(frames, sampled_indices, presentation_times, interval):
    """Mask the selector's frozen time interval on the claim-generation 16-frame grid."""
    from necessity_choice_margin_utils import apply_mask_fill
    from presentation_time_utils import assign_evidence_control_frames_from_times
    if (len(frames) != CLAIM_FRAME_COUNT
            or any(getattr(frame, "mode", None) != "RGB" for frame in frames)):
        raise ValueError("claim_score_generation_frame_contract_invalid")
    assignment = assign_evidence_control_frames_from_times(
        len(presentation_times), CLAIM_FRAME_COUNT, presentation_times, interval)
    if assignment.get("status") != "ok" or assignment["sampled_indices"] != list(sampled_indices):
        raise ValueError("claim_score_generation_grid_assignment_invalid")
    sampled = assignment["sampled_indices"]
    ranks = {frame_index: rank for rank, frame_index in enumerate(sampled)}
    evidence_ranks = sorted(ranks[index] for index in assignment["evidence_indices"])
    control_ranks = sorted(ranks[index] for index in assignment["control_indices"])
    if (not evidence_ranks or len(evidence_ranks) != len(control_ranks)
            or set(evidence_ranks) & set(control_ranks)):
        raise ValueError("claim_score_generation_grid_control_invalid")
    conditions = {}
    for name, masked_ranks in (("full", set()),
            ("claim_window_black", set(evidence_ranks)),
            ("temporal_control_black", set(control_ranks))):
        conditions[name] = [apply_mask_fill(frame, "black") if rank in masked_ranks
                            else frame.copy() for rank, frame in enumerate(frames)]
    def image_hash(frame):
        return hashlib.sha256(frame.tobytes()).hexdigest()
    metadata = {
        "sampled_indices": list(sampled),
        "evidence_indices": list(assignment["evidence_indices"]),
        "control_indices": list(assignment["control_indices"]),
        "evidence_ranks": evidence_ranks,
        "control_ranks": control_ranks,
        "input_frame_sha256": [image_hash(frame) for frame in frames],
        "condition_frame_sha256": {name: [image_hash(frame) for frame in group]
                                   for name, group in conditions.items()},
    }
    return conditions, metadata


def _decode_generation_grid(record, decode_sampled_video, verify_sampled_frame_pixels):
    """Validate the selected 32-frame interval, then map it onto the original 16-frame prompt grid."""
    selector_frames, selector_decode = decode_sampled_video(record["local_video_path"],
        num_frames=SELECTOR_FRAME_COUNT, uniform_tolerance_seconds=TOLERANCE,
        include_timing_map=True)
    verify_sampled_frame_pixels(selector_frames, selector_decode)
    interval = [_exact_selected_window_interval(record["selected_window_receipt"], selector_decode)]
    if selector_decode["source_sha256"] != record["video_sha256"]:
        raise ValueError("claim_score_selector_video_digest_mismatch")

    frames, decoded = decode_sampled_video(record["local_video_path"],
        num_frames=CLAIM_FRAME_COUNT, uniform_tolerance_seconds=TOLERANCE,
        include_timing_map=True)
    verify_sampled_frame_pixels(frames, decoded)
    if decoded["source_sha256"] != record["video_sha256"]:
        raise ValueError("claim_score_generation_video_digest_mismatch")
    saved_pts = record.get("sampled_frame_relative_pts_seconds")
    actual_pts = [float(Fraction(*item["relative_presentation_time"]))
                  for item in decoded["sampled_frames"]]
    if not validate_saved_generation_pts(saved_pts, actual_pts):
        raise ValueError("claim_score_generation_pts_mismatch")
    timing = decoded["full_timing_map"]
    full_times = [Fraction(int(value) - int(timing["pts"][0]))
                  * Fraction(*timing["time_base"]) for value in timing["pts"]]
    conditions, meta = _build_sampled_conditions(frames,
        [item["decoded_frame_index"] for item in decoded["sampled_frames"]],
        full_times, interval)
    if len(meta["evidence_indices"]) != len(meta["control_indices"]):
        raise ValueError("claim_score_generation_grid_count_mismatch")
    signature = {"source_video_sha256": decoded["source_sha256"],
        "selector_pts_sha256": selector_decode["decoded_pts_sequence_sha256"],
        "generation_pts_sha256": decoded["decoded_pts_sequence_sha256"],
        "input_frame_sha256": meta["input_frame_sha256"],
        "condition_frame_sha256": meta["condition_frame_sha256"],
        "evidence_indices": meta["evidence_indices"],
        "control_indices": meta["control_indices"]}
    return frames, decoded, conditions, meta, signature


def run(args) -> int:
    repo = Path(__file__).resolve().parents[1]
    output = validate_private_output(args.output, repo)
    if output.parent != repo / "data" / "private_nextgqa" or output.suffix != ".json":
        raise ValueError("private_output_must_be_json_child_of_private_data_root")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    attempted = completed = 0
    stage = "private_input_validation"
    input_hashes = {}
    try:
        records, input_hashes = _read_selected_window_records(
            repo, args.selection_receipt, args.claim_packet,
            args.cohort_lock, args.cohort_pairs)
        if args.smoke_first_record:
            records = records[:1]

        # Decode and authenticate every selected media/PTS/mask assignment on
        # CPU before loading the large scorer. The scoring loop repeats these
        # checks and requires an exact deterministic signature match.
        from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
        preflight = []
        for position, record in enumerate(records, start=1):
            _frames, _decoded, _conditions, meta, signature = _decode_generation_grid(
                record, decode_sampled_video, verify_sampled_frame_pixels)
            preflight.append({"record_position": position, **signature,
                "selected_frame_count": len(meta["evidence_indices"]),
                "control_frame_count": len(meta["control_indices"])})

        stage = "cuda_resource_preflight"
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info()
        host_free = _host_available_bytes()
        _require_model_load_resources(free, host_free)
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                      / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise ValueError("pinned_local_checkpoint_missing")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  Qwen2_5_VLForConditionalGeneration)
        from qwen_vl_utils import process_vision_info
        from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_BY_VERSION

        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("processor_model_video_token_mismatch")
        torch.manual_seed(20261021)
        torch.cuda.manual_seed_all(20261021)
        torch.cuda.reset_peak_memory_stats()

        rows = []
        for position, record in enumerate(records, start=1):
            stage = f"decode_and_bind_video_{position}"
            frames, decoded, condition_frames, meta, signature = _decode_generation_grid(
                record, decode_sampled_video, verify_sampled_frame_pixels)
            expected_preflight = preflight[position - 1]
            if any(signature[key] != expected_preflight[key] for key in signature):
                raise ValueError("claim_score_cpu_preflight_repeat_mismatch")

            prompt_version = record.get("generation_prompt_version", "v1")
            prompt_suffix = PROMPT_SUFFIX_BY_VERSION.get(prompt_version)
            if not isinstance(prompt_suffix, str):
                raise ValueError("claim_score_prompt_version_invalid")
            fps = float(Fraction(*decoded["effective_sample_fps"]))
            prepared = {}
            metadata = None
            for condition, name in zip(CONDITIONS,
                    ("full", "claim_window_black", "temporal_control_black"), strict=True):
                batch, current_meta = _batch_for_video(processor, process_vision_info,
                    record, condition_frames[name], fps, prompt_suffix, torch)
                metadata = current_meta if metadata is None else metadata
                if current_meta != metadata:
                    raise ValueError("claim_score_video_metadata_changed")
                prepared[condition] = batch
            reference = prepared["full"]
            for condition in CONDITIONS[1:]:
                candidate = prepared[condition]
                if set(reference) != set(candidate):
                    raise ValueError("claim_score_batch_keys_changed")
                for key in reference:
                    left, right = reference[key], candidate[key]
                    if torch.is_tensor(left) and torch.is_tensor(right):
                        if tuple(left.shape) != tuple(right.shape):
                            raise ValueError("claim_score_batch_shape_changed")
                        if key != "pixel_values_videos" and not torch.equal(left, right):
                            raise ValueError("claim_score_nonvisual_input_changed")
                    elif left != right:
                        raise ValueError("claim_score_nonvisual_field_changed")

            claim = record["generated_parsed"]["claim"]
            target_ids = tuple(processor.tokenizer(claim, add_special_tokens=False)["input_ids"])
            if not target_ids or len(target_ids) > 512 or any(type(x) is not int or x < 0 for x in target_ids):
                raise ValueError("claim_score_retokenization_invalid")
            scores = {}
            for condition in CONDITIONS:
                stage = f"score_{condition}_{position}"
                attempted += 1
                scored = _score_branch(model=model, batch_cpu=prepared[condition],
                    target_ids=target_ids, video_token_id=int(processor.video_token_id),
                    biased=False, torch=torch)
                scores[condition] = float(scored["target_logprob_mean"])
                completed += 1
            rows.append({
                "record_position": position,
                "row_token_sha256": hashlib.sha256((record["video_sha256"]
                    + str(record["dataset_question_id"])).encode()).hexdigest(),
                "claim_sha256": hashlib.sha256(claim.encode("utf-8")).hexdigest(),
                "claim_token_count": len(target_ids),
                "claim_token_ids_sha256": hashlib.sha256(json.dumps(
                    target_ids, separators=(",", ":")).encode("ascii")).hexdigest(),
                "condition_mean_token_logprobs_nats": scores,
                **claim_window_contrasts(scores),
                "selected_frame_count": len(meta["evidence_indices"]),
                "control_frame_count": len(meta["control_indices"]),
                "claim_generation_frames": CLAIM_FRAME_COUNT,
                "selected_condition_pixels_sha256": meta["condition_frame_sha256"]["claim_window_black"],
                "control_condition_pixels_sha256": meta["condition_frame_sha256"]["temporal_control_black"],
            })

        complete = len(rows) == (1 if args.smoke_first_record else 3) and completed == len(rows) * 3
        full_run = complete and not args.smoke_first_record
        result = {
            "schema_version": 1,
            "status": "completed_exploratory_claim_text_likelihood" if full_run else "partial_claim_text_smoke",
            "started_utc": started_utc,
            "purpose": "conditional likelihood sensitivity of unreviewed saved claim text to claim-selected versus matched temporal-control frames",
            "direct_faithfulness": False,
            "human_review": {"review_actually_performed": False,
                             "collection_waived_by_user_for_this_exploratory_run": True},
            "rights_status": "unverified; local-only per user authorization",
            "dataset_split": "test",
            "requested_records": 3,
            "effective_records": len(rows) if full_run else 0,
            "requested_forwards": 9 if not args.smoke_first_record else 3,
            "attempted_forwards": attempted,
            "completed_forwards": completed,
            "failure_code": None,
            "selection_receipt_name": Path(args.selection_receipt).name,
            "selection_receipt_sha256": input_hashes[args.selection_receipt],
            "claim_packet_names": [Path(value).name for value in args.claim_packet],
            "cohort_lock_name": Path(args.cohort_lock).name,
            "cohort_pairs_name": Path(args.cohort_pairs).name,
            "inputs_sha256": input_hashes,
            "cpu_preflight_rows": len(preflight),
            # Persist the exact source/PTS/mask signatures that were compared
            # again during scoring; a count alone cannot support artifact-only
            # provenance review after the process exits.
            "cpu_preflight_signatures": preflight,
            "records": rows,
            "primary_metric_name": "control_minus_selected_mask_claim_token_mean_logprob_nats_per_token",
            "primary_metric_mean_nats_per_token": (
                sum(row["control_minus_selected_mask_nats_per_token"] for row in rows) / 3
                if full_run else None),
            "inferential_status": "fixed n=3 development cohort; descriptive only; no population inference",
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": MODEL_REVISION,
            "precision": "4-bit weights; BF16 compute; eager attention",
            "seed": 20261021,
            "hardware": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "free_vram_before_load_gib": free / 1024**3,
                "total_vram_gib": total / 1024**3,
                "host_available_before_load_gib": host_free / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
            "source_sha256": {str(path.relative_to(repo)): _sha256_file(path) for path in (
                Path(__file__).resolve(), repo / "scripts" / "run_machine_localized_claim_mask_pilot.py",
                repo / "scripts" / "run_saved_claim_frame_permutation_attention_smoke.py",
                repo / "scripts" / "claim_mask_video_decoder.py",
                repo / "scripts" / "nextgqa_claim_mask_conditions.py",
                repo / "scripts" / "generate_nextgqa_claim_review_pilot.py",
                repo / "scripts" / "qta_multiquery_video_key_intervention.py",
                repo / "scripts" / "run_nextgqa_gold_interval_margin_pilot.py",
                repo / "scripts" / "presentation_time_utils.py",
                repo / "scripts" / "necessity_choice_margin_utils.py",
                repo / "scripts" / "private_output_paths.py")},
            "elapsed_seconds": time.monotonic() - started,
            "limitations": [
                "saved claims are model-generated and not human checked; likelihood is not truth or semantic support",
                "claim windows were selected from claim likelihood by another Qwen model; selection and scorer share model family",
                "claim text was re-tokenized because original generation token IDs were not retained",
                "black masking can create out-of-distribution video inputs",
                "cohort is sequential exploratory development data on the test split, not independent confirmation",
                "rights remain unverified and artifacts stay local"],
        }
        raw = (json.dumps(result, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n").encode()
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({"status": result["status"], "completed_forwards": completed,
            "effective_records": result["effective_records"],
            "primary_metric_mean_nats_per_token": result["primary_metric_mean_nats_per_token"],
            "elapsed_seconds": round(result["elapsed_seconds"], 1),
            "output_name": output.name}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema_version": 1, "status": "failed_exploratory_claim_text_likelihood",
            "started_utc": started_utc, "stage": stage,
            "exception_type": type(exc).__name__, "failure_code": "claim_text_run_failed",
            "requested_forwards": 3 if args.smoke_first_record else 9,
            "attempted_forwards": attempted,
            "completed_forwards": completed, "inputs_sha256": input_hashes,
            "direct_faithfulness": False, "human_review_performed": False,
            "rights_status": "unverified; local-only"}
        try:
            raw = (json.dumps(failure, sort_keys=True, separators=(",", ":")) + "\n").encode()
            fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        except FileExistsError:
            pass
        print(json.dumps({"status": failure["status"], "stage": stage,
            "exception_type": failure["exception_type"],
            "completed_forwards": completed}, sort_keys=True))
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-receipt", required=True)
    parser.add_argument("--claim-packet", action="append", required=True)
    parser.add_argument("--cohort-lock", required=True)
    parser.add_argument("--cohort-pairs", required=True)
    parser.add_argument("--smoke-first-record", action="store_true")
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        parser.error("claim_text_run_preflight_failed:" + type(exc).__name__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
