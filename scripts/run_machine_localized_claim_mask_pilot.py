#!/usr/bin/env python3
"""Exploratory local GPU pilots on a fixed three-item generated-claim cohort.

The model-localized mode asks Qwen to localize each frozen claim. The separate
benchmark-gold mode uses the dataset's QA intervals. Both measure a fixed QA
gold-vs-best-foil margin under full, interval-black, and matched
temporal-control-black inputs. Neither authenticates human review or proves
claim truth. Treat the results as exploratory diagnostics, not confirmatory
human-validated faithfulness.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import time

from private_output_paths import validate_private_output

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
INDEPENDENT_LOCALIZER_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
LOCALIZER_INSTRUCTION = (
    "Locate the shortest continuous visible-video interval that best supports "
    "the candidate description. Use seconds relative to the first displayed "
    "frame and stay within the stated video duration. Output exactly one JSON "
    "object with the string keys `start_seconds` and `end_seconds`, each mapped "
    "to a number. Do not include "
    "prose, markdown, or analysis. Do not use outside knowledge."
)
FRAME_COUNT = 32
LABELS = "ABCDE"
MIN_FREE_VRAM = 12 * 1024**3
MIN_HOST_AVAILABLE = 8 * 1024**3
TOLERANCE = Fraction(1, 20)
ACTIVE_STAGE = "startup"
COMPLETED_LOCALIZATIONS = 0
ACTIVE_INTERVAL_SOURCE = "unknown"
CURRENT_INPUT_HASHES: dict[str, str] = {}
CURRENT_LOCALIZER_PROMPT_HASHES: list[str] = []
INPUTS = (
    "data/private_nextgqa/claim_review_val_exploratory_gpu_smoke_local_20261019_01.json",
    "data/private_nextgqa/claim_review_val_exploratory_gpu_remaining_local_20261019_01.json",
)


def parse_localized_interval(text: str, duration_seconds: float) -> tuple[float, float]:
    """Parse exactly one JSON interval and reject ambiguous/out-of-range spans."""
    if not isinstance(text, str) or not math.isfinite(float(duration_seconds)) or duration_seconds <= 0:
        raise ValueError("localization_input_invalid")
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.I | re.S)
    if fenced:
        candidate = fenced.group(1).strip()
    def reject_duplicate_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("localization_duplicate_json_key")
            value[key] = item
        return value

    try:
        value = json.loads(candidate, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError):
        raise ValueError("localization_json_invalid") from None
    if not isinstance(value, dict) or set(value) != {"start_seconds", "end_seconds"}:
        raise ValueError("localization_schema_invalid")
    start, end = value["start_seconds"], value["end_seconds"]
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(float(start)) or not math.isfinite(float(end))
            or start < 0 or end <= start or end > duration_seconds):
        raise ValueError("localization_bounds_invalid")
    return float(start), float(end)


def parse_single_embedded_interval(text: str, duration_seconds: float) -> tuple[float, float]:
    """Accept one strict interval object surrounded by model boilerplate.

    The embedded object itself still must have the exact schema and bounds;
    ambiguous multiple-object responses fail closed.
    """
    if not isinstance(text, str):
        raise ValueError("localization_json_invalid")
    objects = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    if len(objects) != 1:
        raise ValueError("localization_single_json_object_required")
    object_match = re.search(r"\{[^{}]*\}", text, flags=re.S)
    surrounding_text = text[:object_match.start()] + text[object_match.end():]
    if re.search(r"\b(?:cannot|can't|unable|uncertain|unsure|impossible|no interval)\b",
                 surrounding_text, flags=re.I):
        raise ValueError("localization_response_disclaims_interval")
    candidate = objects[0]
    try:
        return parse_localized_interval(candidate, duration_seconds)
    except ValueError as original_error:
        # Some instruct checkpoints emit a Python-style dict (single quotes or
        # a trailing comma). Parse only literal syntax, require unique exact
        # keys, then re-serialize through the strict JSON validator.
        keys = ("start_seconds", "end_seconds")
        if any((count := len(re.findall(rf"['\"]{key}['\"]\s*:", candidate))) > 1
               for key in keys):
            raise original_error
        try:
            value = ast.literal_eval(candidate)
            if not isinstance(value, dict) or set(value) != set(keys):
                raise ValueError("localization_schema_invalid")
            strict_json = json.dumps(value, allow_nan=False)
        except (ValueError, SyntaxError, TypeError, OverflowError):
            # Some decoders emit JSON-like unquoted keys. Accept only the exact
            # two-key numeric mapping and no remaining syntax after extraction.
            pair_pattern = re.compile(
                r"(?P<quote>['\"]?)(?P<key>start_seconds|end_seconds)(?P=quote)"
                r"\s*:\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
                r"(?:[eE][+-]?\d+)?)")
            pairs = list(pair_pattern.finditer(candidate))
            if (len(pairs) != 2 or {match.group("key") for match in pairs} != set(keys)):
                raise original_error
            remainder = candidate
            for match in reversed(pairs):
                remainder = remainder[:match.start()] + remainder[match.end():]
            if remainder.strip("{}, \t\r\n"):
                raise original_error
            value = {match.group("key"): float(match.group("value")) for match in pairs}
            strict_json = json.dumps(value, allow_nan=False)
        return parse_localized_interval(strict_json, duration_seconds)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_records(repo: Path) -> tuple[list[dict], dict[str, str]]:
    merged: dict[str, dict] = {}
    input_hashes = {}
    for relative in INPUTS:
        path = repo / relative
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError("claim_input_permissions_invalid")
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
        if (payload.get("status") != "passed" or payload.get("dataset_split") != "val"
                or payload.get("model_revision") != MODEL_REVISION
                or payload.get("claim_review_completed") is not False
                or not isinstance(payload.get("records"), list)):
            raise ValueError("claim_input_contract_invalid")
        input_hashes[relative] = hashlib.sha256(raw).hexdigest()
        for record in payload["records"]:
            pair_hash = record.get("video_sha256")
            claim = record.get("generated_parsed", {}).get("claim")
            if (not isinstance(pair_hash, str) or len(pair_hash) != 64
                    or not isinstance(claim, str) or not claim.strip()
                    or record.get("human_review", {}).get("claim_visually_supported_by_reported_frames") is not None
                    or not isinstance(record.get("local_video_path"), str)):
                raise ValueError("claim_input_record_invalid")
            if pair_hash in merged:
                previous = merged[pair_hash]
                exact_fields = ("video_sha256", "question", "choices", "gold_answer", "generated_raw_text")
                if any(previous.get(key) != record.get(key) for key in exact_fields):
                    raise ValueError("duplicate_claim_pair_payload_mismatch")
            else:
                merged[pair_hash] = record
    records = list(merged.values())
    if len(records) != 3 or len({x.get("video_sha256") for x in records}) != 3:
        raise ValueError("frozen_claim_cohort_not_three_unique_videos")
    return records, input_hashes


def _read_selected_window_records(repo: Path, selection_relative: str):
    """Read the exact test claims/windows bound by a completed local selector receipt."""
    selection_path=(repo/selection_relative).resolve(strict=True)
    private_root=(repo/"data/private_nextgqa").resolve(strict=True)
    if not selection_path.is_relative_to(private_root):
        raise ValueError("selection_receipt_path_invalid")
    selection=json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("status")!="completed_window_likelihood_selection"
            or selection.get("selected_cohort_complete") is not True
            or selection.get("completed_windows")!=12 or len(selection.get("records",[]))!=3):
        raise ValueError("selection_receipt_incomplete")
    packets=("data/private_nextgqa/claim_generation_window_test_smoke_20261021_03.json",
             "data/private_nextgqa/claim_generation_window_test_remaining_20261021_03.json")
    merged={}; hashes={selection_relative:hashlib.sha256(selection_path.read_bytes()).hexdigest()}
    packet_hashes={}
    for relative in packets:
        path=repo/relative; raw=path.read_bytes(); payload=json.loads(raw.decode("utf-8"))
        hashes[relative]=hashlib.sha256(raw).hexdigest()
        packet_hashes[Path(relative).name]=hashes[relative]
        for record in payload.get("records",[]):
            if record.get("video_sha256") in merged:
                raise ValueError("selected_window_duplicate_packet_media")
            merged[record["video_sha256"]]=record
    records=[]; seen=set()
    for selected in selection["records"]:
        media=selected.get("source_video_sha256"); record=merged.get(media)
        if (record is None or media in seen
                or selected.get("dataset_split")!="test"
                or selected.get("packet_sha256")!=packet_hashes.get(selected.get("packet_name"))
                or selected.get("pair_sha256")!=record.get("candidate_pair_sha256")
                or selected.get("saved_claim_sha256")!=hashlib.sha256(
                    record.get("generated_parsed",{}).get("claim","").encode()).hexdigest()
                or selected.get("row_token_sha256")!=hashlib.sha256(
                    (media+str(record.get("dataset_question_id"))).encode()).hexdigest()
                or len(selected.get("selected_sampled_frame_indices",[]))!=8
                or selected.get("window_count")!=4 or selected.get("frames_per_window")!=8):
            raise ValueError("selection_receipt_packet_binding_invalid")
        record=dict(record);record["selected_window_receipt"]=selected
        records.append(record);seen.add(media)
    if len(records)!=3 or len(seen)!=3:
        raise ValueError("selected_window_record_count_invalid")
    return records,hashes


def _host_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("host_memory_probe_unavailable")


def _exact_selected_window_interval(selected: dict, decoded: dict):
    """Recover the selected chunk's exact endpoint PTS after binding the 32-frame decode."""
    chosen=selected.get("selected_sampled_frame_indices")
    displayed=selected.get("sampled_pts_seconds")
    sampled=decoded.get("sampled_frames")
    if (not isinstance(chosen,list) or len(chosen)!=8
            or chosen!=list(range(chosen[0],chosen[0]+8))
            or chosen[0]<0 or chosen[-1]>=FRAME_COUNT
            or not isinstance(displayed,list) or len(displayed)!=FRAME_COUNT
            or decoded.get("decoded_pts_sequence_sha256")!=selected.get("decoded_pts_sha256")
            or len(sampled or [])!=FRAME_COUNT):
        raise ValueError("selected_window_pts_binding_invalid")
    exact=[Fraction(*item["relative_presentation_time"]) for item in sampled]
    if any(abs(float(exact[i])-float(displayed[i]))>1e-7 for i in range(FRAME_COUNT)):
        raise ValueError("selected_window_pts_value_mismatch")
    return exact[chosen[0]],exact[chosen[-1]]


def _require_model_load_resources(free_vram_bytes: int, host_available_bytes: int,
                                 failure_code: str = "pre_model_resource_guard_failed") -> None:
    if (free_vram_bytes < MIN_FREE_VRAM or
            host_available_bytes < MIN_HOST_AVAILABLE):
        raise ValueError(failure_code)


def _endpoint_record_view(records: list[dict], interval_source: str) -> tuple[list[dict], str, str]:
    """Rename internal scorer fields so receipts name the actual interval source."""
    if interval_source == "independent_3b_localized":
        old, new = "claim_evidence_black", "candidate_interval_3b_black"
        old_delta, new_delta = "d_control_minus_claim_mask_nats", "d_control_minus_3b_candidate_interval_nats"
        metric = "D3_control_minus_3b_candidate_interval_nats"
    elif interval_source == "benchmark_gold":
        old, new = "claim_evidence_black", "benchmark_qa_interval_black"
        old_delta, new_delta = "d_control_minus_claim_mask_nats", "d_control_minus_benchmark_qa_interval_nats"
        metric = "D3_control_minus_benchmark_qa_interval_nats"
    elif interval_source == "independent_3b_likelihood_window":
        old, new = "claim_evidence_black", "3b_likelihood_selected_window_black"
        old_delta, new_delta = "d_control_minus_claim_mask_nats", "d_control_minus_3b_likelihood_window_nats"
        metric = "D3_control_minus_3b_likelihood_window_nats"
    else:
        old, new = "claim_evidence_black", "same_model_candidate_interval_black"
        old_delta, new_delta = "d_control_minus_claim_mask_nats", "d_control_minus_same_model_interval_nats"
        metric = "D3_control_minus_same_model_interval_nats"
    renamed = []
    for record in records:
        row = dict(record)
        for field in ("choice_log_probabilities_nats", "gold_vs_best_foil_margins_nats",
                      "best_foil_index_by_condition"):
            if field in row:
                row[field] = dict(row[field])
                if old in row[field]:
                    row[field][new] = row[field].pop(old)
        if old_delta in row:
            row[new_delta] = row.pop(old_delta)
        renamed.append(row)
    return renamed, new, metric


def _write_failure(output_arg: str, exc: Exception) -> None:
    repo = Path(__file__).resolve().parents[1]
    try:
        output = validate_private_output(output_arg, repo)
        known_errors = {
            "localization_json_invalid", "localization_schema_invalid",
            "localization_bounds_invalid", "localization_input_invalid",
            "localization_single_json_object_required",
            "localization_response_disclaims_interval", "localization_literal_invalid",
            "pre_model_resource_guard_failed", "pre_scorer_model_resource_guard_failed",
            "independent_localization_pretransfer_guard_failed",
            "independent_localization_forward_guard_failed",
        }
        detail = str(exc) if type(exc) is ValueError and str(exc) in known_errors else None
        purpose = {
            "model_localized": "exploratory_same_model_localized_claim_span_mask_dependence",
            "independent_3b_localized": "exploratory_cross_scale_localized_claim_span_mask_dependence",
            "benchmark_gold": "exploratory_benchmark_interval_answer_margin_dependence_on_claim_cohort",
        }.get(ACTIVE_INTERVAL_SOURCE, "exploratory_claim_mask_failure")
        localizer_path = (Path.home() /
            ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct" /
            "snapshots" / INDEPENDENT_LOCALIZER_REVISION)
        failed_metric = {
            "model_localized": "D3_control_minus_same_model_interval_nats",
            "independent_3b_localized": "D3_control_minus_3b_candidate_interval_nats",
            "benchmark_gold": "D3_control_minus_benchmark_qa_interval_nats",
        }.get(ACTIVE_INTERVAL_SOURCE, "D3_not_defined")
        payload = {"schema_version": 1, "purpose": purpose,
            "interval_source": ACTIVE_INTERVAL_SOURCE,
            "status": "failed_incomplete_no_effect_estimate",
            "failure_code": detail or ("run_failed_" + type(exc).__name__),
            "failure_stage": ACTIVE_STAGE,
            "completed_localizer_generations": COMPLETED_LOCALIZATIONS,
            "inputs_sha256": CURRENT_INPUT_HASHES,
            "localizer_instruction_sha256": hashlib.sha256(
                LOCALIZER_INSTRUCTION.encode("utf-8")).hexdigest(),
            "localizer_prompt_sha256": CURRENT_LOCALIZER_PROMPT_HASHES,
            "entrypoint_sha256": _sha256(Path(__file__).resolve()),
            "localizer_model_revision": (INDEPENDENT_LOCALIZER_REVISION
                if ACTIVE_INTERVAL_SOURCE == "independent_3b_localized" else
                MODEL_REVISION if ACTIVE_INTERVAL_SOURCE == "model_localized" else None),
            "scorer_model_revision": MODEL_REVISION,
            "localizer_config_sha256": (_sha256(localizer_path / "config.json")
                if ACTIVE_INTERVAL_SOURCE == "independent_3b_localized" and
                   (localizer_path / "config.json").is_file() else None),
            "localizer_processor_config_sha256": (_sha256(
                localizer_path / "preprocessor_config.json")
                if ACTIVE_INTERVAL_SOURCE == "independent_3b_localized" and
                   (localizer_path / "preprocessor_config.json").is_file() else None),
            "requested_records": 3, "effective_records": 0,
            "completed_forwards": 0, "primary_metric_name": failed_metric,
            failed_metric: None,
            "human_review": {"collection_waived_by_user_for_this_run": True,
                "review_actually_performed": False},
            "rights_status": "unverified; local-only per user authorization"}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    except Exception:
        pass


def run(output_arg: str, interval_source: str, selection_receipt: str | None = None,
        smoke_first_record: bool = False) -> int:
    global ACTIVE_STAGE, COMPLETED_LOCALIZATIONS, ACTIVE_INTERVAL_SOURCE, CURRENT_INPUT_HASHES
    global CURRENT_LOCALIZER_PROMPT_HASHES
    ACTIVE_STAGE = "private_input_validation"
    COMPLETED_LOCALIZATIONS = 0
    ACTIVE_INTERVAL_SOURCE = interval_source
    CURRENT_INPUT_HASHES = {}
    CURRENT_LOCALIZER_PROMPT_HASHES = []
    repo = Path(__file__).resolve().parents[1]
    output = validate_private_output(output_arg, repo)
    started = time.monotonic()
    if interval_source=="independent_3b_likelihood_window":
        if not selection_receipt:raise ValueError("selection_receipt_required")
        records,input_hashes=_read_selected_window_records(repo,selection_receipt)
    else:
        records, input_hashes = _read_records(repo)
    if smoke_first_record:
        if interval_source!="independent_3b_likelihood_window":
            raise ValueError("single_record_smoke_only_supported_for_selected_window")
        records=records[:1]
    CURRENT_INPUT_HASHES = input_hashes
    ACTIVE_STAGE = "cuda_and_resource_preflight"
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
    independent_localizer_path = (Path.home() /
        ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct" /
        "snapshots" / INDEPENDENT_LOCALIZER_REVISION)
    if interval_source == "independent_3b_localized" and not independent_localizer_path.is_dir():
        raise ValueError("pinned_independent_localizer_checkpoint_missing")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
    from presentation_time_utils import assign_evidence_control_frames_from_times
    from nextgqa_claim_mask_conditions import build_claim_mask_conditions
    from nextgqa_atomic_batch_freeze import FrozenBatch, _snapshot_tree, batch_sha256
    from nextgqa_claim_mask_scoring import score_preflighted_triplets
    from nextgqa_model_call_boundary import ModelCallBoundary
    from qta_choice_token_binding import resolve_choice_continuation_ids
    from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video

    independent_intervals = []
    independent_prompt_hashes = []
    independent_localizer_peak = 0
    if interval_source == "independent_3b_localized":
        ACTIVE_STAGE = "load_independent_3b_localizer"
        localizer_free, _ = torch.cuda.mem_get_info()
        localizer_host_free = _host_available_bytes()
        _require_model_load_resources(localizer_free, localizer_host_free)
        localizer_processor = AutoProcessor.from_pretrained(
            str(independent_localizer_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        localizer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(independent_localizer_path), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager",
            device_map="cuda", quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(localizer.config.video_token_id) != int(localizer_processor.video_token_id):
            raise ValueError("localizer_processor_video_token_mismatch")
        torch.manual_seed(20261021)
        torch.cuda.manual_seed_all(20261021)
        for position, record in enumerate(records, start=1):
            ACTIVE_STAGE = f"independent_localizer_decode_{position}"
            frames, decoded = decode_sampled_video(record["local_video_path"],
                num_frames=FRAME_COUNT, uniform_tolerance_seconds=TOLERANCE,
                include_timing_map=True)
            verify_sampled_frame_pixels(frames, decoded)
            if decoded["source_sha256"] != record["video_sha256"]:
                raise ValueError("frozen_claim_video_digest_mismatch")
            times_meta = decoded["full_timing_map"]
            pts, tb = times_meta["pts"], Fraction(*times_meta["time_base"])
            full_times = [Fraction(int(value) - int(pts[0])) * tb for value in pts]
            duration = float(full_times[-1])
            fps = Fraction(*decoded["effective_sample_fps"])
            claim = record["generated_parsed"]["claim"]
            ACTIVE_STAGE = f"independent_3b_localize_{position}"
            locator_text = (f"{LOCALIZER_INSTRUCTION} Video duration: {duration:.3f} seconds.\n"
                            f"Candidate description: {claim}")
            locator_messages = [{"role": "user", "content": [
                {"type": "video", "video": frames, "sample_fps": float(fps)},
                {"type": "text", "text": locator_text},
            ]}]
            locator_prompt = localizer_processor.apply_chat_template(
                locator_messages, tokenize=False, add_generation_prompt=True)
            independent_prompt_hashes.append(hashlib.sha256(
                locator_prompt.encode("utf-8")).hexdigest())
            CURRENT_LOCALIZER_PROMPT_HASHES = list(independent_prompt_hashes)
            images, videos = process_vision_info(locator_messages,
                return_video_metadata=True)
            if images is not None or not isinstance(videos, list) or len(videos) != 1:
                raise ValueError("independent_localization_video_preprocess_invalid")
            pixel_values, metadata = videos[0]
            locator_batch = localizer_processor(text=[locator_prompt], images=images,
                videos=[pixel_values], video_metadata=[metadata], do_sample_frames=False,
                cap_pixels_per_frame=True, padding=True, return_tensors="pt")
            if (_host_available_bytes() < MIN_HOST_AVAILABLE or
                    torch.cuda.mem_get_info()[0] < 2 * 1024**3):
                raise ValueError("independent_localization_pretransfer_guard_failed")
            locator_batch = locator_batch.to("cuda")
            if (_host_available_bytes() < MIN_HOST_AVAILABLE or
                    torch.cuda.mem_get_info()[0] < 2 * 1024**3):
                raise ValueError("independent_localization_forward_guard_failed")
            with torch.inference_mode():
                generated = localizer.generate(**locator_batch, do_sample=False,
                    use_cache=True, max_new_tokens=128)
            independent_localizer_peak = max(independent_localizer_peak,
                torch.cuda.max_memory_allocated())
            COMPLETED_LOCALIZATIONS += 1
            ACTIVE_STAGE = f"validate_independent_interval_{position}"
            raw = localizer_processor.tokenizer.decode(
                generated[0, locator_batch["input_ids"].shape[1]:],
                skip_special_tokens=True)
            start_seconds, end_seconds = parse_single_embedded_interval(raw, duration)
            independent_intervals.append((start_seconds, end_seconds))
            del frames, decoded, locator_messages, locator_batch, generated
            del pixel_values, metadata, videos, images
            gc.collect(); torch.cuda.empty_cache()
        ACTIVE_STAGE = "unload_independent_3b_localizer"
        del localizer, localizer_processor
        gc.collect(); torch.cuda.empty_cache()
        ACTIVE_STAGE = "pre_scorer_model_resource_guard"
        scorer_free, scorer_total = torch.cuda.mem_get_info()
        scorer_host_free = _host_available_bytes()
        _require_model_load_resources(scorer_free, scorer_host_free,
            "pre_scorer_model_resource_guard_failed")
    else:
        scorer_free, scorer_total, scorer_host_free = free, total, host_free

    ACTIVE_STAGE = "load_pinned_model_and_processor"
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
    prepared_rows, token_ids, interval_rows = [], None, []
    for position, record in enumerate(records, start=1):
        ACTIVE_STAGE = f"decode_video_{position}"
        frames, decoded = decode_sampled_video(record["local_video_path"],
            num_frames=FRAME_COUNT, uniform_tolerance_seconds=TOLERANCE,
            include_timing_map=True)
        verify_sampled_frame_pixels(frames, decoded)
        if decoded["source_sha256"] != record["video_sha256"]:
            raise ValueError("frozen_claim_video_digest_mismatch")
        times_meta = decoded["full_timing_map"]
        pts, tb = times_meta["pts"], Fraction(*times_meta["time_base"])
        full_times = [Fraction(int(value) - int(pts[0])) * tb for value in pts]
        duration = float(full_times[-1])
        fps = Fraction(*decoded["effective_sample_fps"])
        if interval_source == "model_localized":
            claim = record["generated_parsed"]["claim"]
            ACTIVE_STAGE = f"localize_claim_{position}"
            locator_text = (
                "Locate the shortest continuous visible-video interval that best supports "
                f"the following candidate description. The video begins at 0 s and its last "
                f"displayed frame is at {duration:.3f} s; use only that range, relative to "
                "its first displayed frame. Return only JSON with "
                'numeric keys "start_seconds" and "end_seconds". Do not use outside knowledge.\n'
                f"Candidate description: {claim}"
            )
            locator_messages = [{"role": "user", "content": [
                {"type": "video", "video": frames, "sample_fps": float(fps)},
                {"type": "text", "text": locator_text},
            ]}]
            locator_prompt = processor.apply_chat_template(
                locator_messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(locator_messages, return_video_metadata=True)
            if images is not None or not isinstance(videos, list) or len(videos) != 1:
                raise ValueError("localization_video_preprocess_invalid")
            pixel_values, metadata = videos[0]
            locator_batch = processor(text=[locator_prompt], images=images,
                videos=[pixel_values], video_metadata=[metadata], do_sample_frames=False,
                cap_pixels_per_frame=True, padding=True, return_tensors="pt")
            if (_host_available_bytes() < MIN_HOST_AVAILABLE or
                    torch.cuda.mem_get_info()[0] < 2 * 1024**3):
                raise ValueError("localization_pretransfer_resource_guard_failed")
            locator_batch = locator_batch.to("cuda")
            if _host_available_bytes() < MIN_HOST_AVAILABLE or torch.cuda.mem_get_info()[0] < 2 * 1024**3:
                raise ValueError("localization_forward_resource_guard_failed")
            with torch.inference_mode():
                generated = model.generate(**locator_batch, do_sample=False,
                    use_cache=True, max_new_tokens=64)
            COMPLETED_LOCALIZATIONS += 1
            ACTIVE_STAGE = f"validate_interval_{position}"
            raw = processor.tokenizer.decode(
                generated[0, locator_batch["input_ids"].shape[1]:], skip_special_tokens=True)
            start_seconds, end_seconds = parse_localized_interval(raw, duration)
            interval = [(Fraction(str(start_seconds)), Fraction(str(end_seconds)))]
        elif interval_source == "independent_3b_localized":
            ACTIVE_STAGE = f"validate_independent_interval_{position}"
            start_seconds, end_seconds = independent_intervals[position - 1]
            interval = [(Fraction(str(start_seconds)), Fraction(str(end_seconds)))]
        elif interval_source == "independent_3b_likelihood_window":
            ACTIVE_STAGE = f"validate_selected_window_{position}"
            selected=record["selected_window_receipt"]
            interval=[_exact_selected_window_interval(selected,decoded)]
            start_seconds=float(interval[0][0]);end_seconds=float(interval[0][1])
        elif interval_source == "benchmark_gold":
            ACTIVE_STAGE = f"validate_benchmark_interval_{position}"
            interval = [(Fraction(str(span[0])), Fraction(str(span[1])))
                        for span in record["gold_interval_seconds"]]
            if not interval:
                raise ValueError("benchmark_interval_missing")
            start_seconds = end_seconds = None
        else:
            raise ValueError("interval_source_invalid")
        assignment = assign_evidence_control_frames_from_times(
            len(full_times), FRAME_COUNT, full_times, interval)
        if assignment.get("status") != "ok":
            raise ValueError("localized_interval_has_no_matched_control")
        controls = assignment["control_indices"]
        if not assignment["evidence_indices"] or len(controls) != len(assignment["evidence_indices"]):
            raise ValueError("localized_interval_control_count_invalid")
        qa = {"question": record["question"], "choices": record["choices"],
              "gold": record["choices"].index(record["gold_answer"])}
        condition_payload = build_claim_mask_conditions(
            frames=frames, sampled_indices=[x["decoded_frame_index"] for x in decoded["sampled_frames"]],
            presentation_times=full_times, intervals=interval, include_metadata=True)
        condition_frames = condition_payload["conditions"]
        if interval_source=="independent_3b_likelihood_window":
            expected={decoded["sampled_frames"][i]["decoded_frame_index"] for i in
                      record["selected_window_receipt"]["selected_sampled_frame_indices"]}
            actual=set(condition_payload["metadata"]["evidence_indices"])
            if actual!=expected or len(condition_payload["metadata"]["control_indices"])!=8:
                raise ValueError("selected_window_frame_binding_mismatch")
        row_batches = {}
        rendered_prompt = None
        for condition in ("full", "claim_evidence_black", "temporal_control_black"):
            text, messages = build_qa_prompt(processor, qa, condition_frames[condition],
                                             (fps.numerator, fps.denominator))
            rendered_prompt = rendered_prompt or text
            images, videos = process_vision_info(messages, return_video_metadata=True)
            if images is not None or len(videos) != 1:
                raise ValueError("scoring_video_preprocess_invalid")
            pixels, qwen_metadata = videos[0]
            batch = process_qa_video(processor, text, images, pixels, qwen_metadata)
            row_batches[condition] = FrozenBatch(_snapshot_tree(batch))
        current_token_ids, _prefix = resolve_choice_continuation_ids(
            processor.tokenizer, rendered_prompt)
        if len(current_token_ids) != 5 or len(set(current_token_ids)) != 5:
            raise ValueError("answer_token_ids_invalid")
        if token_ids is None:
            token_ids = current_token_ids
        elif token_ids != current_token_ids:
            raise ValueError("answer_token_binding_differs_across_rows")
        prepared_rows.append({"gold_index": qa["gold"], "batches": row_batches})
        interval_rows.append({"record_position": position,
            "interval_source": interval_source,
            "localizer_prompt_sha256": (independent_prompt_hashes[position - 1]
                if interval_source == "independent_3b_localized" else None),
            "start_seconds": start_seconds, "end_seconds": end_seconds,
            "source_interval_count": len(interval),
            "selected_frame_count": len(assignment["evidence_indices"]),
            "control_frame_count": len(controls),
            "localized_frame_indices": assignment["evidence_indices"],
            "control_frame_indices": controls,
            "interval_assignment_status": condition_payload["metadata"]["status"]})
        if interval_source == "model_localized":
            del locator_batch, generated, pixel_values, metadata
        gc.collect(); torch.cuda.empty_cache()
    ACTIVE_STAGE = "score_nine_frozen_batches"
    row_ids = [f"row_{index}" for index in range(1, len(prepared_rows)+1)]
    expected_hashes = {row_id: {condition: batch_sha256(row["batches"][condition])
        for condition in ("full", "claim_evidence_black", "temporal_control_black")}
        for row_id, row in zip(row_ids, prepared_rows, strict=True)}
    minimum_pretransfer_free = [free]
    minimum_forward_free = [free]
    def check_pretransfer_memory(_row_position, _condition_position, _condition):
        available, _total_now = torch.cuda.mem_get_info()
        minimum_pretransfer_free[0] = min(minimum_pretransfer_free[0], available)
        if available < 2 * 1024**3 or _host_available_bytes() < MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_transfer_resource_guard_failed")
    def check_forward_memory():
        available, _total_now = torch.cuda.mem_get_info()
        minimum_forward_free[0] = min(minimum_forward_free[0], available)
        if available < 2 * 1024**3 or _host_available_bytes() < MIN_HOST_AVAILABLE:
            raise RuntimeError("per_forward_resource_guard_failed")
    call_boundary = ModelCallBoundary(on_root_entry=check_forward_memory)
    score = score_preflighted_triplets(model=model, triplets=prepared_rows,
        token_ids=token_ids, record_ids=row_ids, expected_batch_hashes=expected_hashes,
        torch_module=torch, before_forward=check_pretransfer_memory,
        model_call=call_boundary,requested_records=len(prepared_rows))
    output_records, interval_condition_label, metric_name = _endpoint_record_view(
        score.get("records", []) if score.get("status") == "completed" else [],
        interval_source)
    primary_metric_value = score.get("D3_control_minus_claim_mask_nats")
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    source_files = (
        Path(__file__),
        repo / "scripts/claim_mask_video_decoder.py",
        repo / "scripts/nextgqa_claim_mask_conditions.py",
        repo / "scripts/nextgqa_claim_mask_scoring.py",
        repo / "scripts/nextgqa_condition_batch_builder.py",
        repo / "scripts/nextgqa_atomic_batch_freeze.py",
        repo / "scripts/presentation_time_utils.py",
        repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py",
        repo / "scripts/nextgqa_model_call_boundary.py",
        repo / "scripts/run_claim_likelihood_window_selection.py",
        repo / "scripts/claim_likelihood_window_ranking.py",
    )
    payload = {
        "schema_version": 1,
        "purpose": ("exploratory_same_model_localized_claim_span_mask_dependence"
                    if interval_source == "model_localized"
                    else "exploratory_cross_scale_localized_claim_span_mask_dependence"
                    if interval_source == "independent_3b_localized"
                    else "exploratory_benchmark_interval_answer_margin_dependence_on_claim_cohort"
                    if interval_source=="benchmark_gold" else
                    "exploratory_3b_likelihood_selected_window_answer_margin_dependence"),
        "status": ("partial_three_condition_smoke" if smoke_first_record and score.get("status")=="completed" else score.get("status")),
        "scope": "fixed_three_item_development_cohort; descriptive exploratory only",
        "human_review": {"collection_waived_by_user_for_this_run": True,
                         "review_actually_performed": False,
                         "claim_truth_and_localization_independently_verified": False},
        "requested_records": 3,
        "effective_records": (3 if score.get("status") == "completed" and not smoke_first_record else 0),
        "smoke_only": smoke_first_record,
        "requested_forwards": score.get("requested_forwards"), "completed_forwards": score.get("completed_forwards"),
        "attempted_forwards": score.get("attempted_forwards"),
        "completed_triplets": score.get("completed_triplets"),
        "failure_code": score.get("failure_code"),
        "partial_record_diagnostics_discarded": (
            len(score.get("records", [])) if score.get("status") != "completed" else 0),
        "root_call_entries": call_boundary.root_call_entries,
        "verified_root_calls": call_boundary.verified_root_calls,
        "completed_guarded_calls": call_boundary.completed_calls,
        "interval_source": interval_source,
        "interval_condition_label": interval_condition_label,
        "localizer_generation_calls": COMPLETED_LOCALIZATIONS,
        "records": output_records,
        "intervention_interval_summary": interval_rows,
        "primary_metric_name": metric_name,
        metric_name: primary_metric_value,
        "interpretation": ("Positive means the same-model localized interval mask reduced the fixed QA margin more than equal-count temporal control masking; it does not validate claim truth or independent evidence localization."
            if interval_source == "model_localized" else
            "Positive means the 3B-localized candidate interval mask reduced the fixed 7B QA margin more than equal-count temporal control masking; this cross-scale measure does not validate claim truth, independent evidence localization, or human faithfulness."
            if interval_source == "independent_3b_localized" else
            "Positive means masking a 3B claim-likelihood-selected eight-frame chunk reduced the 7B QA margin more than an equal-count temporal control; four-way adaptive selection and shared Qwen-family priors mean this does not validate claim truth, evidence, or human faithfulness."
            if interval_source == "independent_3b_likelihood_window" else
            "Positive means the benchmark QA gold interval mask reduced the fixed QA margin more than equal-count temporal control masking; it does not measure whether the generated explanation is supported by that interval."),
        "inputs_sha256": input_hashes,
        "source_sha256": {path.name: _sha256(path) for path in source_files},
        "localizer_instruction_sha256": hashlib.sha256(
            LOCALIZER_INSTRUCTION.encode("utf-8")).hexdigest(),
        "processor_config_sha256": _sha256(model_path / "preprocessor_config.json"),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "localizer_config_sha256": (_sha256(independent_localizer_path / "config.json")
            if interval_source == "independent_3b_localized" else None),
        "localizer_processor_config_sha256": (_sha256(
            independent_localizer_path / "preprocessor_config.json")
            if interval_source == "independent_3b_localized" else None),
        "localizer_tokenizer_config_sha256": (_sha256(
            independent_localizer_path / "tokenizer_config.json")
            if interval_source == "independent_3b_localized" else None),
        "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": MODEL_REVISION,
        "localizer_model": ("Qwen/Qwen2.5-VL-3B-Instruct"
            if interval_source == "independent_3b_localized" else
            "Qwen/Qwen2.5-VL-7B-Instruct" if interval_source == "model_localized" else None),
        "localizer_model_revision": (INDEPENDENT_LOCALIZER_REVISION
            if interval_source == "independent_3b_localized" else
            MODEL_REVISION if interval_source == "model_localized" else None),
        "quantization": "4-bit", "compute_dtype": "bfloat16", "attention": "eager",
        "seed": 20261021, "frames_per_video": FRAME_COUNT,
        "intervention": ("black-fill sampled frames in the same-Qwen localized claim interval; equal-count nearest temporal-rank control"
            if interval_source == "model_localized" else
            "black-fill sampled frames in a Qwen2.5-VL-3B localized candidate-claim interval; score with Qwen2.5-VL-7B; equal-count nearest temporal-rank control"
            if interval_source == "independent_3b_localized" else
            "black-fill the 8 sampled frames in a 3B saved-claim-likelihood-selected chunk; score with Qwen2.5-VL-7B; equal-count nearest temporal-rank control"
            if interval_source == "independent_3b_likelihood_window" else
            "black-fill sampled frames in benchmark QA gold interval(s); equal-count nearest temporal-rank control"),
        "hardware": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "free_vram_before_load_gib": free / 1024**3,
            "total_vram_gib": total / 1024**3,
            "host_available_before_load_gib": host_free / 1024**3,
            "free_vram_before_scorer_load_gib": scorer_free / 1024**3,
            "host_available_before_scorer_load_gib": scorer_host_free / 1024**3,
            "independent_localizer_peak_allocated_gib": independent_localizer_peak / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
        "minimum_free_vram_at_scoring_forward_entry_gib": minimum_forward_free[0] / 1024**3,
        "minimum_free_vram_before_scoring_batch_transfer_gib": minimum_pretransfer_free[0] / 1024**3,
        "versions": {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")},
        "elapsed_seconds": elapsed,
        "rights_status": "unverified; local-only per user authorization",
        "user_local_only_authorized": True,
        "inferential_status": "no population inference; n=3 fixed development cohort",
        "limitations": (["candidate claims and QA scores use 7B; a different 3B checkpoint from the same Qwen family localizes intervals; shared-family bias remains",
            "machine-localized interval is not human-validated claim evidence"]
            if interval_source == "independent_3b_localized" else
            ["four-way likelihood selection induces winner's-curse bias",
             "3B and 7B are from the same Qwen family; selected window and claim truth are unreviewed"]
            if interval_source == "independent_3b_likelihood_window" else
            ["same model generates claims, localizes their evidence, and is scored (circularity)"]
            if interval_source == "model_localized" else
            ["benchmark QA intervals are not verified support intervals for the generated claims"])
            + ["no human validation of claim or interval; user waived collection, not evidence truth",
            "black-mask distribution shift; equal frame count does not equal semantics",
            "cohort was reused from claim-review development and is not fresh/confirmatory"],
    }
    out_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(out_bytes + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    del model; gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"status": payload["status"], "requested_records": 3,
        "effective_records": payload["effective_records"],
        "completed_forwards": payload["completed_forwards"],
        "failure_code": payload["failure_code"],
        "primary_metric_name": metric_name,
        metric_name: primary_metric_value,
        "elapsed_seconds": round(elapsed, 1), "output_name": output.name}, sort_keys=True))
    return 0 if score.get("status") == "completed" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new local private output under data/private_nextgqa/")
    parser.add_argument("--interval-source", choices=("model_localized", "independent_3b_localized", "independent_3b_likelihood_window", "benchmark_gold"), required=True)
    parser.add_argument("--selection-receipt", help="completed private 3B selected-window receipt")
    parser.add_argument("--smoke-first-record", action="store_true", help="score one selected-window triplet for path validation only")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args()
    if not args.user_authorized_local_only:
        parser.error("explicit local-only authorization flag required")
    try:
        return run(args.output, args.interval_source,args.selection_receipt,args.smoke_first_record)
    except Exception as exc:
        _write_failure(args.output, exc)
        print(json.dumps({"status": "failed_incomplete_no_effect_estimate",
                          "failure_code": "run_failed_" + type(exc).__name__,
                          "failure_stage": ACTIVE_STAGE,
                          "completed_localizer_generations": COMPLETED_LOCALIZATIONS}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
