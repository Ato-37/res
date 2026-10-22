#!/usr/bin/env python3
"""Exploratory real-video internal key-ablation test for frozen predictions.

Uses a frozen cohort's machine-selected 8-frame claim window and a predeclared
nearest group-level control. A fresh full-video forward freezes the predicted
choice/foil and question-attention QTA keys; four internal ablation forwards
then estimate baseline and fixed-ln(2)-bias selective dependence. Inputs
default to cohort 12 for reproducibility and can be supplied for another
already-frozen cohort. This is local-only and is not human-validated
faithfulness evidence.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from private_output_paths import validate_private_output
from run_machine_localized_claim_mask_pilot import (
    FRAME_COUNT, MODEL_REVISION, _host_available_bytes,
    _read_selected_window_records, _sha256,
)
from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video
from qta_choice_token_binding import resolve_choice_continuation_ids
from qwen_question_temporal_attention_selector import (
    map_question_token_positions, select_top_temporal_groups,
)
from qta_attention_forward_capture import (
    aggregate_captured_qta_rows, capture_qta_attention_forward_once,
)
from qta_multiquery_video_key_intervention import (
    install_multiquery_video_key_intervention_hooks, remove_hooks,
)
from nextgqa_atomic_batch_freeze import FrozenBatch, _snapshot_tree, batch_sha256
from qwen_frozen_frame_window_groups import map_frozen_frame_windows_to_qwen_groups

LAYERS = tuple(range(5, 17))
LOGIT_BIAS = math.log(2.0)
MIN_FREE_VRAM = 12 * 1024**3
MIN_HOST_AVAILABLE = 8 * 1024**3
SELECTION = "data/private_nextgqa/claim_likelihood_selection_remaining_20261021_12.json"
LOCK = "data/private_nextgqa/claim_likelihood_window_cohort_lock_20261021_12.json"
PAIRS = "data/private_nextgqa/claim_likelihood_window_cohort_pairs_20261021_12.json"
PACKETS = ["data/private_nextgqa/claim_likelihood_claims_remaining_20261021_12.json"]
PRIOR_QA = "data/private_nextgqa/claim_likelihood_mask_margin_full_20261021_12_lockbound.json"


def _canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _read_owner_json(repo: Path, relative: str) -> tuple[dict, str]:
    path = repo / relative
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
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
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("private_json_object_required")
    return value, hashlib.sha256(raw).hexdigest()


def select_group_control(evidence_groups, temporal_group_count: int) -> tuple[int, ...]:
    """Nearest disjoint contiguous equal-size control; ties go to earlier groups."""
    evidence = tuple(evidence_groups)
    if (not evidence or any(type(i) is not int for i in evidence)
            or tuple(sorted(set(evidence))) != evidence
            or any(i < 0 or i >= temporal_group_count for i in evidence)):
        raise ValueError("evidence_groups_invalid")
    if evidence != tuple(range(evidence[0], evidence[-1] + 1)):
        raise ValueError("evidence_groups_not_contiguous")
    k = len(evidence)
    candidates = [tuple(range(start, start + k))
                  for start in range(temporal_group_count - k + 1)
                  if not set(range(start, start + k)).intersection(evidence)]
    if not candidates:
        raise ValueError("equal_group_control_unavailable")
    center = (evidence[0] + evidence[-1]) / 2
    return min(candidates, key=lambda c: (abs((c[0] + c[-1]) / 2 - center), c[0]))


def _top_and_foil(values):
    if (not isinstance(values, (list, tuple)) or len(values) != 5
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(float(x)) for x in values)):
        raise ValueError("choice_logprobs_invalid")
    order = sorted(range(5), key=lambda i: (-float(values[i]), i))
    return order[0], order[1]


def _mask_representable_bias(torch_module, dtype, value: float) -> float:
    """Return the intervention value after the additive mask's dtype cast."""
    if not math.isfinite(float(value)):
        raise ValueError("attention_bias_nonfinite")
    return float(torch_module.tensor(float(value), dtype=dtype).item())


def compute_delta_f(branches: dict, predicted: int, foil: int) -> dict:
    required = {"baseline_evidence_ablation", "baseline_control_ablation",
                "qta_evidence_ablation", "qta_control_ablation"}
    if set(branches) != required or predicted == foil:
        raise ValueError("branch_score_set_invalid")
    for values in branches.values():
        if (not isinstance(values, (list, tuple)) or len(values) != 5
                or any(not math.isfinite(float(x)) for x in values)):
            raise ValueError("branch_choice_vector_invalid")
    margins = {name: float(v[predicted]) - float(v[foil])
               for name, v in branches.items()}
    base = margins["baseline_control_ablation"] - margins["baseline_evidence_ablation"]
    qta = margins["qta_control_ablation"] - margins["qta_evidence_ablation"]
    return {"f_baseline_nats": base, "f_qta_nats": qta,
            "delta_f_qta_minus_baseline_nats": qta - base,
            "fixed_prediction_index": predicted, "fixed_foil_index": foil,
            "margins_nats": margins}


def _private_input_path(repo: Path, value: str) -> str:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("private_input_path_invalid")
    try:
        resolved = (repo / relative).resolve(strict=True)
        private = (repo / "data/private_nextgqa").resolve(strict=True)
        resolved.relative_to(private)
        return resolved.relative_to(repo.resolve()).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError("private_input_path_invalid") from exc


def _resolve_inputs(repo: Path, selection: str, lock: str, pairs: str,
                    packets: list[str], prior_qa: str) -> dict:
    if not packets:
        raise ValueError("claim_packet_required")
    return {"selection": _private_input_path(repo, selection),
        "lock": _private_input_path(repo, lock),
        "pairs": _private_input_path(repo, pairs),
        "packets": [_private_input_path(repo, item) for item in packets],
        "prior_qa": _private_input_path(repo, prior_qa)}


def _prepare_rows(repo: Path, processor, torch, process_vision_info,
                  selection: str = SELECTION, lock: str = LOCK,
                  pairs: str = PAIRS, packets: list[str] | None = None):
    records, input_hashes = _read_selected_window_records(
        repo, selection, PACKETS if packets is None else packets, lock, pairs)
    if len(records) != 3:
        raise ValueError("frozen_cohort_record_count_invalid")
    dataset = repo / "NExT-GQA/datasets/nextgqa"
    video_root = Path.home() / "nextgqa_sample/NExTVideo"
    with (dataset / "test.csv").open(newline="", encoding="utf-8-sig") as stream:
        dataset_rows = list(csv.DictReader(stream))
    video_map = json.loads((dataset / "map_vid_vidorID.json").read_text(encoding="utf-8"))
    from run_nextgqa_gold_interval_margin_pilot import candidate_pair_sha256
    from run_nextgqa_real_video_gpu_smoke import video_key
    from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
    prepared = []
    for record in records:
        relative_video = "/".join(Path(record["local_video_path"]).parts[-2:])
        matching_video_ids = [str(vid) for vid, rel in video_map.items()
                              if video_key(rel) == video_key(relative_video)
                              and candidate_pair_sha256(str(vid),
                                  str(record["dataset_question_id"]))
                                  == record["candidate_pair_sha256"]]
        if len(matching_video_ids) != 1:
            raise ValueError("dataset_question_join_invalid")
        matches = [candidate for candidate in dataset_rows
                   if str(candidate["video_id"]) == matching_video_ids[0]
                   and str(candidate["qid"]) == str(record["dataset_question_id"])]
        if len(matches) != 1:
            raise ValueError("dataset_question_join_invalid")
        row = matches[0]
        media = Path(record["local_video_path"])
        if media.is_symlink() or _sha256(media) != record["video_sha256"]:
            raise ValueError("source_video_digest_mismatch")
        frames, decoded = decode_sampled_video(media, num_frames=FRAME_COUNT,
            uniform_tolerance_seconds=__import__("fractions").Fraction(1, 20),
            include_timing_map=True)
        verify_sampled_frame_pixels(frames, decoded)
        selected = record["selected_window_receipt"]
        sampled = decoded["sampled_frames"]
        pts = [float(__import__("fractions").Fraction(*x["relative_presentation_time"]))
               for x in sampled]
        saved_pts = selected.get("sampled_pts_seconds")
        if (decoded["source_sha256"] != record["video_sha256"]
                or selected.get("decoded_pts_sha256") != decoded.get("decoded_pts_sequence_sha256")
                or not isinstance(saved_pts, list) or len(saved_pts) != FRAME_COUNT
                or any(abs(float(a) - b) > 1e-6 for a, b in zip(saved_pts, pts, strict=True))):
            raise ValueError("selected_window_pts_mismatch")
        chosen = selected.get("selected_sampled_frame_indices")
        if (not isinstance(chosen, list) or len(chosen) != 8
                or chosen != list(range(chosen[0], chosen[0] + 8))):
            raise ValueError("selected_window_frame_indices_invalid")
        messages_item = {"question": str(row["question"]).strip(),
                         "choices": [str(row[f"a{i}"]).strip() for i in range(5)]}
        fps = decoded["effective_sample_fps"]
        rendered, messages = build_qa_prompt(processor, messages_item, frames, fps)
        images, videos = process_vision_info(messages, return_video_metadata=True)
        if images is not None or not isinstance(videos, list) or len(videos) != 1:
            raise ValueError("video_processor_input_invalid")
        pixels, metadata = videos[0]
        if (metadata.get("frames_indices") != list(range(FRAME_COUNT))
                or not math.isclose(float(metadata.get("fps", 0)),
                    fps[0] / fps[1], rel_tol=0, abs_tol=1e-7)):
            raise ValueError("processor_frame_index_or_fps_mismatch")
        batch = process_qa_video(processor, rendered, images, pixels, metadata)
        tp = int(processor.video_processor.temporal_patch_size)
        grid = [int(x) for x in batch["video_grid_thw"][0].tolist()]
        seconds_per_grid = batch.get("second_per_grid_ts")
        if (seconds_per_grid is None or len(seconds_per_grid) != 1
                or grid[0] * tp != FRAME_COUNT
                or not math.isclose(float(seconds_per_grid[0]), tp / (fps[0] / fps[1]),
                    rel_tol=5e-8, abs_tol=1e-6)):
            raise ValueError("processor_temporal_timestamp_grid_mismatch")
        evidence_window = (chosen[0], chosen[-1] + 1)
        evidence_group_start = evidence_window[0] // tp
        evidence_group_stop = evidence_window[1] // tp
        control_groups = select_group_control(
            tuple(range(evidence_group_start, evidence_group_stop)), grid[0])
        control_window = (control_groups[0] * tp, (control_groups[-1] + 1) * tp)
        mapped = map_frozen_frame_windows_to_qwen_groups(
            input_ids=batch["input_ids"][0].tolist(),
            video_token_id=int(processor.video_token_id), video_grid_thw=grid,
            spatial_merge_size=int(processor.image_processor.merge_size),
            temporal_patch_size=tp, sampled_pts_seconds=pts,
            evidence_frame_window=evidence_window, control_frame_window=control_window)
        question = str(row["question"]).strip()
        qpos = map_question_token_positions(tokenizer=processor.tokenizer,
            rendered_text=rendered, question_text=question,
            processed_input_ids=batch["input_ids"],
            video_token_id=int(processor.video_token_id))
        answer_ids, _prefix_ids = resolve_choice_continuation_ids(
            processor.tokenizer, rendered)
        # Verify the multimodal processor appends the same single answer token.
        from qta_processor_choice_suffix import verify_processor_choice_suffix
        def process_text(text):
            imgs, vids = process_vision_info([{"role": "user", "content": [
                {"type": "video", "video": frames,
                 "sample_fps": fps[0] / fps[1]}, {"type": "text", "text": text}]}],
                return_video_metadata=True)
            p, m = vids[0]
            return processor(text=[text], images=imgs, videos=[p], video_metadata=[m],
                do_sample_frames=False, cap_pixels_per_frame=True, padding=True,
                return_tensors="pt")["input_ids"][0].tolist()
        choice_binding = verify_processor_choice_suffix(tokenizer=processor.tokenizer,
            rendered_prompt=rendered, process_text=process_text)
        if choice_binding["answer_token_ids"] != answer_ids:
            raise ValueError("choice_id_binding_mismatch")
        batch = FrozenBatch(_snapshot_tree(batch))
        typed_batch_hash = batch_sha256(batch)
        binding = {"row_token_sha256": hashlib.sha256(
                (record["video_sha256"] + str(record["dataset_question_id"])).encode()).hexdigest(),
            "source_video_sha256": record["video_sha256"],
            "saved_claim_sha256": hashlib.sha256(
                record["generated_parsed"]["claim"].encode()).hexdigest()}
        prepared.append({"record": record, "binding": binding, "row": row,
            "batch": batch, "batch_sha256": typed_batch_hash,
            "rendered": rendered, "question": question, "qpos": qpos,
            "answer_ids": answer_ids, "mapping": mapped, "grid": grid,
            "pts": pts, "selected": selected, "choice_binding": choice_binding,
            "processor_metadata": {"frames_indices": metadata.get("frames_indices"),
                "fps": float(metadata["fps"]),
                "second_per_grid_ts": float(seconds_per_grid[0]),
                "temporal_patch_size": tp, "video_grid_thw": grid}})
    return prepared, input_hashes


def _answer_logprobs(logits, answer_ids, torch):
    vector = torch.log_softmax(logits[0, -1].float(), dim=-1)
    values = vector[torch.tensor(answer_ids, device=vector.device)].detach().cpu().tolist()
    if len(values) != 5 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("answer_logprobs_invalid")
    return [float(x) for x in values]


def _run_one(prepared, model, torch, counters, prior_full_scores):
    free_before_transfer, _ = torch.cuda.mem_get_info()
    host_before_transfer = _host_available_bytes()
    if free_before_transfer < 2 * 1024**3 or host_before_transfer < 3 * 1024**3:
        raise RuntimeError("pre_transfer_resource_guard_failed")
    batch = prepared["batch"].to("cuda")
    sequence_len = int(batch["input_ids"].shape[1])
    groups = [tuple(row) for row in prepared["mapping"]["temporal_groups"]]
    visual_groups = groups
    # Full-video baseline simultaneously freezes p/q and QTA attention keys.
    free, _ = torch.cuda.mem_get_info()
    if free < 2 * 1024**3 or _host_available_bytes() < 3 * 1024**3:
        raise RuntimeError("per_forward_resource_guard_failed")
    if batch_sha256(prepared["batch"]) != prepared["batch_sha256"] or \
            batch_sha256(batch) != prepared["batch_sha256"]:
        raise ValueError("frozen_batch_changed_before_baseline_forward")
    counters["attempted"] += 1
    def full_forward():
        result = model(**batch, use_cache=False, logits_to_keep=1)
        counters["completed"] += 1
        return result
    output, rows, capture = capture_qta_attention_forward_once(
        model=model,
        forward=full_forward,
        question_positions=prepared["qpos"], input_length=sequence_len,
        torch_module=torch)
    full_scores = _answer_logprobs(output.logits, prepared["answer_ids"], torch)
    max_prior_deviation = max(abs(a - float(b)) for a, b in
        zip(full_scores, prior_full_scores, strict=True))
    if max_prior_deviation > 1e-4:
        raise ValueError("reconstructed_full_prediction_differs_from_prior_receipt")
    attention_by_layer = rows
    group_attention = aggregate_captured_qta_rows(
        rows=attention_by_layer, group_positions=visual_groups,
        torch_module=torch)
    top_groups = tuple(sorted(select_top_temporal_groups(
        group_attention, top_k=max(1, math.ceil(len(groups) * 0.10)))))
    qta_positions = tuple(sorted(p for i in top_groups for p in groups[i]))
    predicted, foil = _top_and_foil(full_scores)
    del output, rows, attention_by_layer
    torch.cuda.empty_cache()
    ev = tuple(prepared["mapping"]["evidence_token_positions"])
    ctl = tuple(prepared["mapping"]["control_token_positions"])
    branches = {}
    call_details = {"baseline_full": capture}
    for name, bias, ablation in (
        ("baseline_evidence_ablation", (), ev),
        ("baseline_control_ablation", (), ctl),
        ("qta_evidence_ablation", qta_positions, ev),
        ("qta_control_ablation", qta_positions, ctl)):
        free, _ = torch.cuda.mem_get_info()
        if free < 2 * 1024**3 or _host_available_bytes() < 3 * 1024**3:
            raise RuntimeError("per_forward_resource_guard_failed")
        if batch_sha256(prepared["batch"]) != prepared["batch_sha256"] or \
                batch_sha256(batch) != prepared["batch_sha256"]:
            raise ValueError("frozen_batch_changed_before_ablation_forward")
        handles = install_multiquery_video_key_intervention_hooks(
            model=model, layer_indices=tuple(i - 1 for i in LAYERS),
            biased_positions=bias, ablated_positions=ablation,
            query_positions=(sequence_len - 1,), logit_bias=LOGIT_BIAS,
            torch_module=torch)
        observed = {i: 0 for i in LAYERS}
        observer_handles = []
        try:
            layers = model.model.language_model.layers
            for layer_i in LAYERS:
                module = layers[layer_i - 1].self_attn
                def observe(_module, args, kwargs, *, layer=layer_i):
                    mask = kwargs.get("attention_mask")
                    if not torch.is_tensor(mask) or mask.ndim != 4:
                        raise ValueError("intervention_observer_mask_invalid")
                    rows_i = torch.tensor([sequence_len - 1], device=mask.device)
                    expected = mask[0, 0, rows_i[0]]
                    if any(float(expected[p].item()) != float("-inf") for p in ablation):
                        raise ValueError("intervention_ablation_hook_value_mismatch")
                    representable_bias = _mask_representable_bias(
                        torch, mask.dtype, LOGIT_BIAS)
                    for p in bias:
                        if p not in ablation and not math.isclose(float(expected[p].item()),
                                representable_bias, rel_tol=0, abs_tol=1e-6):
                            raise ValueError("intervention_bias_hook_value_mismatch")
                    observed[layer] += 1
                    return args, kwargs
                observer_handles.append(module.register_forward_pre_hook(observe, with_kwargs=True))
            counters["attempted"] += 1
            with torch.inference_mode():
                out = model(**batch, use_cache=False, logits_to_keep=1)
            counters["completed"] += 1
            branches[name] = _answer_logprobs(out.logits, prepared["answer_ids"], torch)
            if observed != {i: 1 for i in LAYERS}:
                raise ValueError("intervention_hook_layer_count_mismatch")
            call_details[name] = {"hook_calls_by_layer": {str(k): v for k,v in observed.items()},
                "bias_key_count": len(bias), "ablation_key_count": len(ablation),
                "query_position": sequence_len - 1}
            del out
        finally:
            for h in reversed(observer_handles):
                h.remove()
            remove_hooks(handles)
        torch.cuda.empty_cache()
    branches["full_video_baseline"] = full_scores
    contrast_branches = {key: branches[key] for key in (
        "baseline_evidence_ablation", "baseline_control_ablation",
        "qta_evidence_ablation", "qta_control_ablation")}
    result = {"row_token_sha256": prepared["binding"]["row_token_sha256"],
        "source_video_sha256": prepared["binding"]["source_video_sha256"],
        "saved_claim_sha256": prepared["binding"]["saved_claim_sha256"],
        "input_batch_sha256": prepared["batch_sha256"],
        "prompt_render_sha256": hashlib.sha256(prepared["rendered"].encode()).hexdigest(),
        "choice_token_binding_sha256": _canonical_hash(prepared["choice_binding"]),
        "selected_window_frame_indices": prepared["mapping"]["evidence_frame_window"],
        "control_window_frame_indices": prepared["mapping"]["control_frame_window"],
        "selected_window_endpoint_pts_seconds": prepared["mapping"]["evidence_pts_seconds"],
        "control_window_endpoint_pts_seconds": prepared["mapping"]["control_pts_seconds"],
        "processor_temporal_metadata": prepared["processor_metadata"],
        "evidence_group_indices": prepared["mapping"]["evidence_group_indices"],
        "control_group_indices": prepared["mapping"]["control_group_indices"],
        "question_attention_group_indices": list(top_groups),
        "question_attention_group_scores": group_attention,
        "evidence_key_count": len(ev), "control_key_count": len(ctl),
        "qta_key_count": len(qta_positions), "full_choice_logprobs_nats": full_scores,
        "branch_choice_logprobs_nats": branches,
        "primary": compute_delta_f(contrast_branches, predicted, foil),
        "forward_count": 5, "calls": call_details,
        "direct_faithfulness": False, "claim_truth_reviewed": False}
    del batch
    gc.collect(); torch.cuda.empty_cache()
    return result


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--selection-receipt", default=SELECTION)
    parser.add_argument("--cohort-lock", default=LOCK)
    parser.add_argument("--cohort-pairs", default=PAIRS)
    parser.add_argument("--claim-packet", action="append")
    parser.add_argument("--prior-qa-receipt", default=PRIOR_QA)
    parser.add_argument("--smoke-first-record", action="store_true",
        help="score one row (5 forwards) for path validation only; no cohort estimate")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    repo = Path(__file__).resolve().parents[1]
    output = validate_private_output(args.output_name, repo)
    private = repo / "data/private_nextgqa"
    if output.parent != private or output.suffix != ".json":
        raise ValueError("output_path_must_be_private_json")
    started = time.monotonic(); started_utc = datetime.now(timezone.utc).isoformat()
    stage = "preflight"
    attempted = completed = 0
    counters = {"attempted": 0, "completed": 0}
    try:
        inputs = _resolve_inputs(repo, args.selection_receipt, args.cohort_lock,
            args.cohort_pairs, PACKETS if args.claim_packet is None else args.claim_packet,
            args.prior_qa_receipt)
        selection_path, lock_path, pairs_path = inputs["selection"], inputs["lock"], inputs["pairs"]
        packet_paths, prior_path = inputs["packets"], inputs["prior_qa"]
        prior, prior_hash = _read_owner_json(repo, prior_path)
        if (prior.get("status") != "completed" or prior.get("effective_records") != 3
                or prior.get("completed_forwards") != 9 or len(prior.get("records", [])) != 3):
            raise ValueError("prior_full_video_receipt_incomplete")
        for position, row in enumerate(prior["records"], start=1):
            if not isinstance(row, dict):
                raise ValueError("prior_qa_row_or_full_vector_invalid")
            values = row.get("choice_log_probabilities_nats", {}).get("full")
            if (row.get("record_position") != position
                    or not isinstance(values, list) or len(values) != 5
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(float(value)) for value in values)):
                raise ValueError("prior_qa_row_or_full_vector_invalid")
        if prior.get("inputs_sha256") != {
                selection_path: _read_owner_json(repo, selection_path)[1],
                lock_path: _read_owner_json(repo, lock_path)[1],
                pairs_path: _read_owner_json(repo, pairs_path)[1],
                packet_paths[0]: _read_owner_json(repo, packet_paths[0])[1]}:
            raise ValueError("prior_qa_inputs_do_not_match_frozen_cohort")
        prior_source_mismatches = []
        for name, expected_hash in prior.get("source_sha256", {}).items():
            candidates = (repo / name, repo / "scripts" / name)
            source = next((candidate for candidate in candidates if candidate.is_file()), None)
            if source is None or _sha256(source) != expected_hash:
                prior_source_mismatches.append(name)
        if prior_source_mismatches:
            raise ValueError("prior_qa_source_hash_mismatch")
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda_bf16_unavailable")
        free, total = torch.cuda.mem_get_info(); host_free = _host_available_bytes()
        if free < MIN_FREE_VRAM or host_free < MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_model_resource_guard_failed")
        model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
            / "snapshots" / MODEL_REVISION)
        if not model_path.is_dir():
            raise ValueError("pinned_model_checkpoint_missing")
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        stage = "cpu_processor_preflight_all_rows"
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        prepared, input_hashes = _prepare_rows(repo, processor, torch, process_vision_info,
            selection_path, lock_path, pairs_path, packet_paths)
        if len(prepared) != 3:
            raise ValueError("cpu_preflight_not_all_rows")
        source_files = [Path(__file__).resolve(), repo / "scripts/run_machine_localized_claim_mask_pilot.py",
            repo / "scripts/claim_mask_video_decoder.py",
            repo / "scripts/run_nextgqa_real_video_gpu_smoke.py",
            repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py",
            repo / "scripts/qta_multiquery_video_key_intervention.py",
            repo / "scripts/qwen_frozen_frame_window_groups.py",
            repo / "scripts/qwen_question_temporal_attention_selector.py",
            repo / "scripts/qta_attention_forward_capture.py",
            repo / "scripts/qta_choice_token_binding.py",
            repo / "scripts/qta_processor_choice_suffix.py",
            repo / "scripts/nextgqa_atomic_batch_freeze.py"]
        source_hashes = {str(p.relative_to(repo)): _sha256(p) for p in source_files}
        input_hashes.update({
            "NExT-GQA/datasets/nextgqa/test.csv": _sha256(repo / "NExT-GQA/datasets/nextgqa/test.csv"),
            "NExT-GQA/datasets/nextgqa/map_vid_vidorID.json": _sha256(repo / "NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"),
        })
        stage = "load_pinned_model"
        free, total = torch.cuda.mem_get_info(); host_free = _host_available_bytes()
        if free < MIN_FREE_VRAM or host_free < MIN_HOST_AVAILABLE:
            raise RuntimeError("pre_model_resource_guard_failed")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if int(model.config.video_token_id) != int(processor.video_token_id):
            raise ValueError("model_processor_video_token_mismatch")
        torch.manual_seed(20261021); torch.cuda.manual_seed_all(20261021)
        torch.cuda.reset_peak_memory_stats()
        records = []
        score_rows = prepared[:1] if args.smoke_first_record else prepared
        cohort_lock, cohort_lock_hash = _read_owner_json(repo, lock_path)
        stage = "score_frozen_cohort"
        for row in score_rows:
            row_start = dict(counters)
            prior_row = prior["records"][len(records)]
            if prior_row.get("record_position") != len(records) + 1:
                raise ValueError("prior_qa_record_order_invalid")
            prior_full = prior_row.get("choice_log_probabilities_nats", {}).get("full")
            result = _run_one(row, model, torch, counters, prior_full)
            attempted, completed = counters["attempted"], counters["completed"]
            result["forward_count"] = {
                "attempted": counters["attempted"] - row_start["attempted"],
                "completed": counters["completed"] - row_start["completed"]}
            result["prior_full_vector_max_abs_deviation_nats"] = max(
                abs(float(a) - float(b)) for a, b in zip(result["full_choice_logprobs_nats"], prior_full, strict=True))
            records.append(result)
        expected_rows = 1 if args.smoke_first_record else 3
        expected_forwards = 5 * expected_rows
        if (len(records) != expected_rows or attempted != expected_forwards
                or completed != expected_forwards):
            raise RuntimeError("incomplete_fixed_cohort")
        del model; gc.collect(); torch.cuda.empty_cache()
        values = [row["primary"]["delta_f_qta_minus_baseline_nats"] for row in records]
        payload = {"schema_version": 1,
            "status": ("partial_internal_key_ablation_smoke" if args.smoke_first_record
                       else "completed_exploratory_internal_key_ablation"),
            "purpose": "frozen_full_video_prediction_selective_dependence_under_global_visual_key_bias",
            "cohort": Path(lock_path).stem,
            "cohort_lock_sha256": cohort_lock_hash,
            "cohort_purpose": cohort_lock.get("purpose"),
            "cohort_reuse": "post-result exploratory test-split follow-up; known-local-exposure-filtered, not confirmatory",
            "human_review": {"review_collection_waived_by_user": True,
                "review_actually_performed": False, "claim_truth_reviewed": False},
            "rights_status": "unverified; local-only per user instruction",
            "requested_records": expected_rows, "effective_records": len(records),
            "requested_forwards": expected_forwards, "attempted_forwards": attempted,
            "completed_forwards": completed,
            "input_sha256": {**input_hashes, prior_path: prior_hash},
            "prior_qa_source_sha256": prior.get("source_sha256"),
            "records": records,
            "finite_cohort_summary": (None if args.smoke_first_record else {"n": len(values), "mean_delta_f_nats": sum(values)/len(values),
                "median_delta_f_nats": sorted(values)[1], "min_delta_f_nats": min(values),
                "max_delta_f_nats": max(values),
                "positive_rows": sum(x > 0 for x in values),
                "interpretation": "finite-cohort exploratory estimate only; no CI, test, or population inference"}),
            "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": MODEL_REVISION,
            "precision": "4-bit weights; BF16 compute; eager attention", "seed": 20261021,
            "layers_1_based": list(LAYERS), "visual_key_bias": LOGIT_BIAS,
            "condition_interpretation": "same full-video pixels in every branch; only internal visual key attention mask differs",
            "prior_qa_receipt_sha256": prior_hash,
            "source_sha256": source_hashes,
            "hardware": {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "free_vram_before_load_gib": free / 1024**3,
                "total_vram_gib": total / 1024**3,
                "host_available_before_load_gib": host_free / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3},
            "elapsed_seconds": time.monotonic() - started,
            "direct_faithfulness": False,
            "limitations": ["machine-selected window is unreviewed and is only an exploratory anchor",
                "global visual-key bias is not claim-specific",
                "test cohort was selected after earlier related outcomes and is exploratory",
                "fixed prediction comes from this baseline full-video forward",
                "no inference to a population or thesis-level faithfulness conclusion",
                "human review was waived as a run prerequisite but not performed",
                "rights remain unverified; protected inputs and scores stay local"]}
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({"status": payload["status"], "records": len(records),
            "forwards": completed, "mean_delta_f_nats": None if args.smoke_first_record else sum(values)/len(values),
            "output_name": output.name}, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {"schema_version": 1, "status": "failed_exploratory_internal_key_ablation",
            "stage": stage, "failure_type": type(exc).__name__,
            "failure_code": str(exc).splitlines()[0][:160],
            "started_utc": started_utc,
            "requested_records": 1 if args.smoke_first_record else 3,
            "requested_forwards": 5 if args.smoke_first_record else 15,
            "attempted_forwards": counters["attempted"],
            "completed_forwards": counters["completed"], "direct_faithfulness": False,
            "human_review_performed": False,
            "completed_rows": locals().get("records", []),
            "effective_records": len(locals().get("records", [])),
            "rights_status": "unverified; local-only",
            "source_sha256": {str(Path(__file__).relative_to(repo)): _sha256(Path(__file__))}}
        try:
            raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"))+"\n").encode()
            fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "wb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        except FileExistsError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(run())
