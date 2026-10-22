#!/usr/bin/env python3
"""Local exploratory NOAA QA-choice distribution comparison under QTA/sham.

Uses only local samples.json question/choice inputs and video frames. Never
opens gold.json, rights/audit metadata, or human-review files. The result is a
restricted-choice behavior diagnostic, not accuracy, grounding, or faithfulness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import time

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
FRAME_COUNT = 8
SEED = 20261022
QTA_LAYERS = tuple(range(5, 17))
LOGIT_BIAS = math.log(2.0)
SHAM_SEED = "noaa-equal-budget-disjoint-hash-sham-v1-20261022"
MIN_FREE_VRAM = 12 * 1024**3
MIN_HOST_AVAILABLE = 8 * 1024**3
GROUPS = (
    ("private/samples.json", ""),
    ("dive04/private/samples.json", "dive04/"),
    ("dive09/private/samples.json", "dive09/"),
)
SOURCE_PATHS = (
    "scripts/run_noaa_qta_distribution_pilot.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/qta_attention_forward_capture.py",
    "scripts/qta_attention_capture_contract.py",
    "scripts/qta_attention_capture_sidecars.py",
    "scripts/qwen_question_temporal_attention_selector.py",
    "scripts/qta_multiquery_video_key_intervention.py",
    "scripts/qta_choice_token_binding.py",
    "scripts/qta_processor_choice_suffix.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(repo: Path) -> dict[str, str]:
    return {name: sha256_file(repo / name) for name in SOURCE_PATHS}


def choose_equal_budget_sham(*, temporal_group_count: int, qta_groups: tuple[int, ...],
                             opaque_row_token: str) -> tuple[int, ...]:
    """Hash-rank equal-size subsets disjoint from QTA; no evidence overlap claim."""
    if (type(temporal_group_count) is not int or temporal_group_count < 2
            or not qta_groups or tuple(sorted(set(qta_groups))) != qta_groups
            or any(type(x) is not int or x < 0 or x >= temporal_group_count
                   for x in qta_groups)
            or len(qta_groups) >= temporal_group_count
            or not isinstance(opaque_row_token, str) or not opaque_row_token):
        raise ValueError("sham_inputs_invalid")
    size = len(qta_groups)
    qta = set(qta_groups)
    candidates = [candidate for candidate in itertools.combinations(
        range(temporal_group_count), size) if not qta.intersection(candidate)]
    if not candidates:
        raise ValueError("disjoint_equal_budget_sham_unavailable")
    prefix = f"{SHAM_SEED}\0{opaque_row_token}\0"
    return min(candidates, key=lambda candidate: (
        hashlib.sha256((prefix + ",".join(map(str, candidate))).encode()).digest(),
        candidate))


def temporal_token_groups(input_ids, video_token_id: int, grid,
                          spatial_merge_size: int) -> list[tuple[int, ...]]:
    """Map contiguous processor video tokens into equal temporal groups."""
    if (not isinstance(grid, (list, tuple)) or len(grid) != 3
            or any(type(x) is not int or x < 1 for x in grid)
            or type(spatial_merge_size) is not int or spatial_merge_size < 1):
        raise ValueError("video_grid_invalid")
    temporal, height, width = grid
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("video_grid_merge_invalid")
    positions = [i for i, token in enumerate(input_ids) if token == video_token_id]
    tokens_per_group = (height // spatial_merge_size) * (width // spatial_merge_size)
    if (len(positions) != temporal * tokens_per_group or not positions
            or positions != list(range(positions[0], positions[0] + len(positions)))):
        raise ValueError("video_token_group_geometry_invalid")
    return [tuple(positions[i * tokens_per_group:(i + 1) * tokens_per_group])
            for i in range(temporal)]


def restricted_choice_probabilities(logprobs: list[float]) -> list[float]:
    """Normalize five option-token log-probabilities within the candidate set."""
    if (not isinstance(logprobs, list) or len(logprobs) != 5
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(float(x)) for x in logprobs)):
        raise ValueError("choice_logprobs_invalid")
    peak = max(map(float, logprobs))
    weights = [math.exp(float(x) - peak) for x in logprobs]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("choice_probabilities_invalid")
    return [value / total for value in weights]


def total_variation(left: list[float], right: list[float]) -> float:
    if (not isinstance(left, list) or not isinstance(right, list)
            or len(left) != 5 or len(right) != 5
            or any(not math.isfinite(float(x)) for x in left + right)):
        raise ValueError("choice_distribution_invalid")
    return 0.5 * sum(abs(float(a) - float(b))
                     for a, b in zip(left, right, strict=True))


def load_samples(repo: Path) -> tuple[list[dict], dict[str, str]]:
    data_root = (repo / "datasets/noaa_qa_pilot_20261011").resolve(strict=True)
    records, manifest_hashes, seen_rows = [], {}, set()
    for relative, _prefix in GROUPS:
        manifest_path = data_root / relative
        raw = manifest_path.read_bytes()
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError("sample_manifest_invalid")
        manifest_hashes[relative] = sha256_bytes(raw)
        base = repo.resolve()
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("video"), str)
                    or not isinstance(row.get("video_id"), str)
                    or not isinstance(row.get("qid"), str)
                    or not isinstance(row.get("question"), str)
                    or not row["question"].strip()
                    or not isinstance(row.get("choices"), list)
                    or len(row["choices"]) != 5
                    or any(not isinstance(choice, str) or not choice.strip()
                           for choice in row["choices"])):
                raise ValueError("sample_record_contract_invalid")
            row_key = (row["video_id"], row["qid"])
            if row_key in seen_rows:
                raise ValueError("duplicate_sample_row")
            seen_rows.add(row_key)
            relative_video = Path(row["video"])
            if relative_video.is_absolute() or ".." in relative_video.parts:
                raise ValueError("sample_video_path_invalid")
            candidate_video = base / relative_video
            cursor = base
            for part in relative_video.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise ValueError("sample_video_path_invalid")
            video = candidate_video.resolve(strict=True)
            if not video.is_relative_to(data_root):
                raise ValueError("sample_video_path_invalid")
            digest = sha256_file(video)
            if digest != row.get("video_sha256"):
                raise ValueError("sample_video_hash_mismatch")
            records.append({
                "row_token": sha256_bytes((digest + "\0" + row["qid"]).encode()),
                "qid_sha256": sha256_bytes(row["qid"].encode()),
                "video_id_sha256": sha256_bytes(row["video_id"].encode()),
                "video_sha256": digest,
                "video_path": video,
                "question": row["question"],
                "choices": list(row["choices"]),
            })
    if (len(records) != 6 or len({r["video_sha256"] for r in records}) != 3
            or len({r["video_id_sha256"] for r in records}) != 3):
        raise ValueError("fixed_noaa_six_row_three_video_cohort_required")
    return records, manifest_hashes


def _tensor_digest(batch, torch) -> str:
    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        digest.update(key.encode() + b"\0")
        if torch.is_tensor(value):
            cpu = value.detach().cpu().contiguous()
            digest.update(str(cpu.dtype).encode() + b"\0")
            digest.update(json.dumps(list(cpu.shape)).encode() + b"\0")
            digest.update(cpu.numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _score(logits, answer_ids, torch) -> list[float]:
    vector = torch.log_softmax(logits[0, -1].float(), dim=-1)
    result = vector[torch.tensor(answer_ids, device=vector.device)].detach().cpu().tolist()
    if len(result) != 5 or any(not math.isfinite(float(x)) for x in result):
        raise ValueError("choice_scores_nonfinite")
    return [float(x) for x in result]


def _intervention_forward(model, batch, *, key_positions, seq_len, answer_ids,
                          torch, counters, protocol_name: str) -> tuple[list[float], dict]:
    from qta_multiquery_video_key_intervention import (
        install_multiquery_video_key_intervention_hooks, remove_hooks)
    if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
        raise RuntimeError("per_forward_free_vram_guard_failed")
    handles = install_multiquery_video_key_intervention_hooks(
        model=model, layer_indices=tuple(i - 1 for i in QTA_LAYERS),
        biased_positions=key_positions, ablated_positions=(),
        query_positions=(seq_len - 1,), logit_bias=LOGIT_BIAS,
        torch_module=torch)
    calls = {i: 0 for i in QTA_LAYERS}
    observers = []
    try:
        layers = model.model.language_model.layers
        for layer_i in QTA_LAYERS:
            module = layers[layer_i - 1].self_attn
            def observe(_module, args, kwargs, *, layer=layer_i):
                mask = kwargs.get("attention_mask")
                if not torch.is_tensor(mask) or mask.ndim != 4:
                    raise ValueError("attention_mask_observer_invalid")
                expected = float(torch.tensor(LOGIT_BIAS, dtype=mask.dtype).item())
                query = mask[0, 0, seq_len - 1]
                if any(not math.isclose(float(query[p].item()), expected,
                                        rel_tol=0, abs_tol=1e-6)
                       for p in key_positions):
                    raise ValueError("intervention_bias_observer_mismatch")
                calls[layer] += 1
                return args, kwargs
            observers.append(module.register_forward_pre_hook(observe, with_kwargs=True))
        counters["attempted"] += 1
        with torch.inference_mode():
            output = model(**batch, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        counters["completed"] += 1
        if calls != {i: 1 for i in QTA_LAYERS}:
            raise ValueError("intervention_layer_call_count_mismatch")
        scores = _score(output.logits, answer_ids, torch)
        del output
        return scores, {"hook_calls_by_layer": {str(k): v for k, v in calls.items()},
                        "biased_key_count": len(key_positions),
                        "effective_bias_bf16": float(torch.tensor(LOGIT_BIAS,
                            dtype=torch.bfloat16).item()), "arm": protocol_name}
    finally:
        for handle in reversed(observers):
            handle.remove()
        remove_hooks(handles)


def _prepare_record(record, processor, process_vision_info, torch, decoder,
                    decoded_cache: dict):
    from qwen_question_temporal_attention_selector import map_question_token_positions
    from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video
    from qta_choice_token_binding import resolve_choice_continuation_ids
    if record["video_sha256"] in decoded_cache:
        frames, decoded = decoded_cache[record["video_sha256"]]
    else:
        frames, decoded = decoder.decode_sampled_video(
            record["video_path"], num_frames=FRAME_COUNT,
            uniform_tolerance_seconds=Fraction(1, 20))
        decoder.verify_sampled_frame_pixels(frames, decoded)
        decoded_cache[record["video_sha256"]] = (frames, decoded)
    if decoded["source_sha256"] != record["video_sha256"] or len(frames) != FRAME_COUNT:
        raise ValueError("decoded_source_binding_mismatch")
    fps = decoded["effective_sample_fps"]
    rendered, messages = build_qa_prompt(processor,
        {"question": record["question"], "choices": record["choices"]}, frames, fps)
    images, videos = process_vision_info(messages, return_video_metadata=True)
    if images is not None or len(videos) != 1:
        raise ValueError("video_only_processor_contract_failed")
    pixels, metadata = videos[0]
    if (metadata.get("frames_indices") != list(range(FRAME_COUNT))
            or not math.isclose(float(metadata.get("fps", 0)),
                                fps[0] / fps[1], rel_tol=0, abs_tol=1e-7)):
        raise ValueError("video_sample_fps_contract_failed")
    batch = process_qa_video(processor, rendered, images, pixels, metadata)
    grid = [int(x) for x in batch["video_grid_thw"][0].tolist()]
    temporal_patch = int(processor.video_processor.temporal_patch_size)
    if grid[0] * temporal_patch != FRAME_COUNT:
        raise ValueError("temporal_grid_frame_count_mismatch")
    qpos = map_question_token_positions(tokenizer=processor.tokenizer,
        rendered_text=rendered, question_text=record["question"],
        processed_input_ids=batch["input_ids"],
        video_token_id=int(processor.video_token_id))
    answer_ids, _ = resolve_choice_continuation_ids(processor.tokenizer, rendered)
    if len(answer_ids) != 5 or len(set(answer_ids)) != 5:
        raise ValueError("choice_token_binding_invalid")
    from qta_processor_choice_suffix import verify_processor_choice_suffix
    def process_text(text):
        images_for_text, videos_for_text = process_vision_info([{
            "role": "user", "content": [
                {"type": "video", "video": frames,
                 "sample_fps": fps[0] / fps[1]},
                {"type": "text", "text": text},
            ]}], return_video_metadata=True)
        p, m = videos_for_text[0]
        return processor(text=[text], images=images_for_text, videos=[p],
            video_metadata=[m], do_sample_frames=False, cap_pixels_per_frame=True,
            padding=True, return_tensors="pt")["input_ids"][0].tolist()
    suffix_binding = verify_processor_choice_suffix(tokenizer=processor.tokenizer,
        rendered_prompt=rendered, process_text=process_text)
    if suffix_binding["answer_token_ids"] != answer_ids:
        raise ValueError("choice_suffix_binding_mismatch")
    suffix_binding_sha256 = sha256_bytes(json.dumps(suffix_binding,
        sort_keys=True, separators=(",", ":")).encode())
    positions = temporal_token_groups(batch["input_ids"][0].tolist(),
        int(processor.video_token_id), grid,
        int(processor.image_processor.merge_size))
    expected_group_tokens = ((grid[1] // int(processor.image_processor.merge_size))
                             * (grid[2] // int(processor.image_processor.merge_size)))
    if any(len(group) != expected_group_tokens for group in positions):
        raise ValueError("temporal_token_group_size_mismatch")
    prompt_sha = sha256_bytes(rendered.encode("utf-8"))
    return {"frames": frames, "decoded": decoded, "fps": fps,
            "batch": batch, "grid": grid, "groups": positions,
            "qpos": qpos, "answer_ids": answer_ids,
            "input_sha256": _tensor_digest(batch, torch),
            "prompt_sha256": prompt_sha,
            "choice_suffix_binding_sha256": suffix_binding_sha256}


def _run_row(record, prepared, model, torch, counters):
    from qta_attention_forward_capture import capture_qta_attention_forward_once, aggregate_captured_qta_rows
    from qwen_question_temporal_attention_selector import select_top_temporal_groups
    batch = prepared["batch"].to("cuda")
    if _tensor_digest(batch, torch) != prepared["input_sha256"]:
        raise ValueError("frozen_processor_batch_changed")
    seq_len = int(batch["input_ids"].shape[1])
    counters["attempted"] += 1
    def baseline_forward():
        with torch.inference_mode():
            result = model(**batch, use_cache=False, logits_to_keep=1,
                           output_attentions=True)
            counters["completed"] += 1
            return result
    output, attention, capture = capture_qta_attention_forward_once(
        model=model, forward=baseline_forward, question_positions=prepared["qpos"],
        input_length=seq_len, torch_module=torch)
    baseline_scores = _score(output.logits, prepared["answer_ids"], torch)
    group_scores = aggregate_captured_qta_rows(rows=attention,
        group_positions=prepared["groups"], torch_module=torch)
    top_k = max(1, math.ceil(len(prepared["groups"]) * 0.10))
    qta_groups = tuple(sorted(select_top_temporal_groups(group_scores, top_k=top_k)))
    row_token = record["row_token"]
    sham_groups = choose_equal_budget_sham(
        temporal_group_count=len(prepared["groups"]), qta_groups=qta_groups,
        opaque_row_token=row_token)
    qta_keys = tuple(sorted(p for group in qta_groups for p in prepared["groups"][group]))
    sham_keys = tuple(sorted(p for group in sham_groups for p in prepared["groups"][group]))
    del output, attention
    torch.cuda.empty_cache()
    qta_scores, qta_audit = _intervention_forward(model, batch,
        key_positions=qta_keys, seq_len=seq_len, answer_ids=prepared["answer_ids"],
        torch=torch, counters=counters, protocol_name="qta")
    sham_scores, sham_audit = _intervention_forward(model, batch,
        key_positions=sham_keys, seq_len=seq_len, answer_ids=prepared["answer_ids"],
        torch=torch, counters=counters, protocol_name="equal_budget_hash_sham")
    scores = {"baseline": baseline_scores, "qta": qta_scores, "sham": sham_scores}
    distributions = {key: restricted_choice_probabilities(value)
                     for key, value in scores.items()}
    predictions = {key: max(range(5), key=lambda i: (value[i], -i))
                   for key, value in distributions.items()}
    return {
        "row_token_sha256": row_token,
        "video_id_sha256": record["video_id_sha256"],
        "qid_sha256": record["qid_sha256"],
        "source_video_sha256": record["video_sha256"],
        "prompt_sha256": prepared["prompt_sha256"],
        "choice_suffix_binding_sha256": prepared["choice_suffix_binding_sha256"],
        "processor_batch_sha256": prepared["input_sha256"],
        "frame_count": len(prepared["frames"]),
        "sampled_pts_seconds": [float(Fraction(*r["relative_presentation_time"]))
                                 for r in prepared["decoded"]["sampled_frames"]],
        "grid_thw": prepared["grid"],
        "qta_group_indices": list(qta_groups),
        "equal_budget_hash_sham_group_indices": list(sham_groups),
        "baseline_attention_capture": capture,
        "intervention_audits": {"qta": qta_audit, "sham": sham_audit},
        "restricted_choice_probabilities": distributions,
        "predicted_choice_index": predictions,
        "tvd_qta_vs_baseline": total_variation(distributions["qta"], distributions["baseline"]),
        "tvd_sham_vs_baseline": total_variation(distributions["sham"], distributions["baseline"]),
        "tvd_qta_vs_sham": total_variation(distributions["qta"], distributions["sham"]),
        "qta_prediction_flip_from_baseline": predictions["qta"] != predictions["baseline"],
        "sham_prediction_flip_from_baseline": predictions["sham"] != predictions["baseline"],
    }


def _aggregate(records: list[dict]) -> dict:
    if not records:
        raise ValueError("records_required")
    by_video = {}
    for row in records:
        by_video.setdefault(row["source_video_sha256"], []).append(row)
    if len(by_video) != 3:
        raise ValueError("three_video_cluster_contract_failed")
    metrics = ("tvd_qta_vs_baseline", "tvd_sham_vs_baseline", "tvd_qta_vs_sham")
    def mean(vals): return sum(vals) / len(vals) if vals else None
    video_records = []
    for digest, rows in sorted(by_video.items()):
        video_records.append({"source_video_sha256": digest, "question_rows": len(rows),
            **{metric: mean([r[metric] for r in rows]) for metric in metrics},
            "qta_flips_from_baseline": sum(r["qta_prediction_flip_from_baseline"] for r in rows),
            "sham_flips_from_baseline": sum(r["sham_prediction_flip_from_baseline"] for r in rows)})
    return {
        "requested_effective_rows": [6, len(records)],
        "unique_video_clusters": len(by_video),
        "completed_forwards": len(records) * 3,
        "question_level_mean_tvd": {metric: mean([r[metric] for r in records]) for metric in metrics},
        "video_cluster_mean_tvd": {metric: mean([r[metric] for r in video_records]) for metric in metrics},
        "qta_prediction_flips_from_baseline": sum(r["qta_prediction_flip_from_baseline"] for r in records),
        "sham_prediction_flips_from_baseline": sum(r["sham_prediction_flip_from_baseline"] for r in records),
        "video_cluster_summaries": video_records,
        "inference": "descriptive exploratory n=3 source-video clusters; no CI/test/population inference",
    }


def _protocol_sha256() -> str:
    value = {"model_revision": MODEL_REVISION, "frame_count": FRAME_COUNT,
        "seed": SEED, "layers_1_based": list(QTA_LAYERS),
        "logit_bias": LOGIT_BIAS, "qta_selector": "highest ceil(0.10*T) groups",
        "sham": "same-sized disjoint subset; SHA-256 ranked candidate combinations",
        "scored_targets": "five fixed option-label continuation token IDs",
        "analysis": "within-five-option softmax and total variation distance"}
    return sha256_bytes(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode())


def _write_exclusive(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        return sha256_bytes(raw)
    except Exception:
        try: os.close(fd)
        except OSError: pass
        raise


def run(args) -> int:
    repo = Path(__file__).resolve().parents[1]
    data_root = repo / "datasets/noaa_qa_pilot_20261011"
    output = Path(args.output)
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("output_path_invalid")
    output = (repo / output).resolve()
    if not output.is_relative_to((data_root / "private").resolve()):
        raise ValueError("output_must_remain_in_noaa_private_root")
    smoke = None
    if args.mode == "full":
        if not args.smoke_receipt:
            raise ValueError("full_run_requires_smoke_receipt")
        smoke_path = (repo / args.smoke_receipt).resolve(strict=True)
        if not smoke_path.is_relative_to((data_root / "private").resolve()):
            raise ValueError("smoke_receipt_path_invalid")
        smoke = json.loads(smoke_path.read_text())
        if (smoke.get("status") != "completed_smoke"
                or smoke.get("requested_rows") != 1
                or smoke.get("effective_rows") != 1
                or smoke.get("completed_forwards") != 3):
            raise ValueError("smoke_receipt_not_passed")
    started = datetime.now(timezone.utc).isoformat()
    monotonic_started = time.monotonic()
    stage = "load_private_inference_samples"
    attempted = completed = 0
    output_payload = None
    try:
        records, manifest_hashes = load_samples(repo)
        sample_records = records[:1] if args.mode == "smoke" else records
        if args.mode == "full" and (
                smoke.get("sample_manifest_sha256") != manifest_hashes
                or smoke.get("protocol_sha256") != _protocol_sha256()
                or smoke.get("model_revision") != MODEL_REVISION
                or smoke.get("frame_count_per_row") != FRAME_COUNT
                or smoke.get("source_sha256") != source_hashes(repo)
                or not isinstance(smoke.get("rows"), list)
                or smoke["rows"][0].get("row_token_sha256") != sample_records[0]["row_token"]):
            raise ValueError("smoke_receipt_protocol_or_input_mismatch")
        import torch
        import psutil
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info()
        host_available = int(psutil.virtual_memory().available)
        if free < MIN_FREE_VRAM or host_available < MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_model_resource_guard_failed")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
            / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise RuntimeError("pinned_checkpoint_missing")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        import claim_mask_video_decoder as decoder
        stage = "cpu_preflight_all_selected_rows"
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        decoded_cache = {}
        prepared = [_prepare_record(row, processor, process_vision_info, torch,
            decoder, decoded_cache) for row in sample_records]
        if len(prepared) != len(sample_records):
            raise ValueError("prepared_row_count_mismatch")
        stage = "load_pinned_qwen"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("model_processor_video_token_mismatch")
        random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        counters = {"attempted": 0, "completed": 0}
        stage = "baseline_qta_sham_forwards"
        scored = []
        for record, inputs in zip(sample_records, prepared, strict=True):
            scored.append(_run_row(record, inputs, model, torch, counters))
            del inputs["batch"]
            torch.cuda.empty_cache()
        attempted, completed = counters["attempted"], counters["completed"]
        expected_rows = len(sample_records)
        if attempted != expected_rows * 3 or completed != expected_rows * 3:
            raise RuntimeError("incomplete_forward_counts")
        versions = {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av", "Pillow")}
        payload = {
            "schema_version": 1,
            "purpose": "exploratory_noaa_internal_qta_answer_distribution_sensitivity",
            "status": "completed_smoke" if args.mode == "smoke" else "completed_exploratory",
            "dataset": "local NOAA EX2503 pilot; no gold/audit/rights/human-review files opened",
            "run_mode": args.mode,
            "started_utc": started,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=repo, text=True).strip(),
            "source_worktree_dirty": True,
            "source_sha256": source_hashes(repo),
            "sample_manifest_sha256": manifest_hashes,
            "protocol_sha256": _protocol_sha256(),
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": MODEL_REVISION,
            "model_config_sha256": sha256_file(model_path / "config.json"),
            "seed": SEED,
            "frame_count_per_row": FRAME_COUNT,
            "precision": "4-bit weights; BF16 compute; eager attention",
            "intervention": {"selection": "untreated question attention; highest ceil(0.10*T) temporal groups",
                "layers_1_based": list(QTA_LAYERS), "logit_bias_requested": LOGIT_BIAS,
                "sham": "same number of temporal groups, hash-selected from disjoint groups; not evidence-overlap-matched"},
            "requested_rows": 1 if args.mode == "smoke" else 6,
            "effective_rows": len(scored),
            "attempted_forwards": attempted,
            "completed_forwards": completed,
            "rows": scored,
            "summary": None if args.mode == "smoke" else _aggregate(scored),
            "human_review": {"review_performed": False,
                "user_authorized_local_exploratory_inference": True,
                "annotation_truth_or_evidence_support_claimed": False},
            "rights_status": "unverified; local-only under user authorization",
            "gold_used": False,
            "audio_used": False,
            "direct_faithfulness": False,
            "limits": ["restricted A-E next-token distribution only; not the full answer distribution",
                "6 questions are clustered in only 3 source videos; descriptive only",
                "QTA selects question-salient visual keys, not validated explanation evidence",
                "sham matches key budget and is disjoint, but is not evidence-overlap or position matched",
                "does not measure QA accuracy, explanation truth, grounding, or direct faithfulness",
                "human review was not performed; source rights were not independently verified"],
            "gpu": {"name": torch.cuda.get_device_name(0), "cuda_build": torch.version.cuda,
                "free_vram_before_load_bytes": int(free), "total_vram_bytes": int(total),
                "host_available_before_load_bytes": host_available,
                "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated())},
            "runtime_versions": versions,
            "elapsed_seconds": round(time.monotonic() - monotonic_started, 3),
        }
        del model
        output_sha = _write_exclusive(output, payload)
        print(json.dumps({"status": payload["status"], "rows": len(scored),
            "completed_forwards": completed, "output_sha256": output_sha,
            "output_relative_path": output.relative_to(repo).as_posix(),
            "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema_version": 1, "status": "failed_incomplete_no_effect_estimate",
            "purpose": "exploratory_noaa_internal_qta_answer_distribution_sensitivity",
            "run_mode": args.mode, "failure_stage": stage,
            "failure_type": type(exc).__name__, "requested_rows": 1 if args.mode == "smoke" else 6,
            "attempted_forwards": attempted, "completed_forwards": completed,
            "human_review_performed": False, "direct_faithfulness": False,
            "created_utc": datetime.now(timezone.utc).isoformat()}
        if not output.exists():
            _write_exclusive(output, failure)
        print(json.dumps({"status": "failed", "stage": stage,
                          "failure_type": type(exc).__name__}), file=sys.stderr)
        return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", required=True,
        help="unique output path under datasets/noaa_qa_pilot_20261011/private/")
    parser.add_argument("--smoke-receipt")
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "failure_type": type(exc).__name__}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
