#!/usr/bin/env python3
"""Pathway attention-knockout replication of Map the Flow (Kim et al., ICLR 2026) on Qwen2.5-VL-7B and TVBench (the base paper's main benchmark; 8 frames).

The cohort is built from the four TVBench tasks whose videos are distributed with the dataset (Action Sequence, Scene Transition, Moving Direction, Object Count; Action Antonym needs NTU RGB+D from its own site and is not included);
one question per video, hash-ranked, at most --per-task videos per task. Questions have 2-4 candidates: labels A.. are used and the readout is restricted to them.
Original description follows (NExT-GQA text of the first version of this runner).

For each video whose unmodified top answer is correct (as in the base paper), block one edge class (cross-frame, video->text,
video->last, text->last) inside a window of k decoder layers centred on layer l, and record the relative change of the
probability of the originally predicted answer token at the last position (fp32 readout). Receipts are local-only (mode 0600),
hash-identified; each video is appended to a partial JSONL so a crash loses at most one video (--resume continues).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_nextgqa_annotation_key_bias_fp32_readout  # noqa: F401  (patches lm_head to an fp32 readout on import)
import run_nextgqa_annotation_key_bias as base
import run_nextgqa_gold_interval_margin_pilot as cohort
import qta_edge_knockout as ko

CENTERS_DEFAULT = (2, 5, 8, 11, 14, 17, 20, 23, 26)
WIDTH_DEFAULT = 9
SEED = 20261023
RUNNER_PATH = Path(__file__).resolve()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--tvbench-root", required=True)
    p.add_argument("--per-task", type=int, default=100)
    p.add_argument("--dataset-split", default="tvbench")
    p.add_argument("--n", type=int, required=True, help="requested videos in total (<= 4 x per-task)")
    p.add_argument("--frames", type=int, choices=(8, 16, 32), default=8)
    p.add_argument("--output", required=True)
    p.add_argument("--centers", default=",".join(map(str, CENTERS_DEFAULT)))
    p.add_argument("--width", type=int, default=WIDTH_DEFAULT)
    p.add_argument("--edges", default=",".join(ko.EDGES))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--user-authorized-local-only", action="store_true")
    return p.parse_args(argv)



TASKS = ("action_sequence", "scene_transition", "moving_direction", "object_count")


def build_qa_prompt_k(processor, item, frames, sample_fps):
    """Same prompt as the NExT-GQA runners but with 2-4 options (labels A..)."""
    k = len(item["choices"])
    options = "\n".join(f"{label}. {choice}" for label, choice in zip(cohort.LABELS[:k], item["choices"], strict=True))
    prompt_text = (f"{item['question']}\nOptions:\n{options}\n"
                   "Choose the best answer. Respond with one letter only.\nAnswer:")
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames, "sample_fps": sample_fps[0] / sample_fps[1]},
        {"type": "text", "text": prompt_text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), messages


def build_tvbench_cohort(tv_root: Path, n: int, per_task: int, frame_count: int):
    import hashlib
    from fractions import Fraction
    rows = []
    for task in TASKS:
        data = json.loads((tv_root / "json" / f"{task}.json").read_text(encoding="utf-8"))
        seen = set()
        ranked = sorted(range(len(data)), key=lambda i: hashlib.sha256(f"tvb-v1|{task}|{data[i]['video']}|{i}".encode()).hexdigest())
        for i in ranked:
            q = data[i]
            path = tv_root / "video" / task / q["video"]
            if q["video"] in seen or not path.is_file() or q["answer"] not in q["candidates"]:
                continue
            seen.add(q["video"])
            rows.append((task, i, q, path))
    selected, counts = [], {t: {"accepted": 0, "decode_rejected": 0} for t in TASKS}
    for task in TASKS:
        for t, i, q, path in rows:
            if t != task or counts[task]["accepted"] >= per_task or len(selected) >= n:
                continue
            try:
                frames, decode = cohort.decode_sampled_video(path, num_frames=frame_count, uniform_tolerance_seconds=cohort.UNIFORM_TOLERANCE)
                cohort.verify_sampled_frame_pixels(frames, decode)
            except Exception:
                counts[task]["decode_rejected"] += 1
                continue
            pts = [Fraction(*r["relative_presentation_time"]) for r in decode["sampled_frames"]]
            item = {"qid": f"{task}:{i}", "question": q["question"], "choices": list(q["candidates"]), "gold": q["candidates"].index(q["answer"]),
                    "frames": cohort.LazyFrames(path, frame_count, decode), "decode": decode, "sampled_times": pts,
                    "evidence_positions": [], "control_positions": [], "task": task}
            selected.append(item)
            counts[task]["accepted"] += 1
    if len(selected) != n:
        raise ValueError(f"requested_{n}_but_cohort_has_{len(selected)}")
    fingerprint = hashlib.sha256(json.dumps([[x["task"], x["qid"], x["decode"]["source_sha256"]] for x in selected]).encode()).hexdigest()
    return selected, fingerprint, counts


def relative_change_percent(p_base: float, p_knockout: float) -> float:
    if not p_base > 0:
        raise ValueError("baseline_probability_nonpositive")
    return (p_knockout - p_base) / p_base * 100.0


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.user_authorized_local_only:
        raise SystemExit("explicit_user_authorized_local_only_required")
    centers = tuple(int(c) for c in args.centers.split(","))
    edges = tuple(args.edges.split(","))
    if any(e not in ko.EDGES for e in edges):
        raise SystemExit("edge_invalid")
    repo = RUNNER_PATH.parents[1]
    output = base.private_output_path(args.output, repo)
    partial = Path(str(output) + ".partial.jsonl")
    if output.exists() or (partial.exists() and not args.resume):
        raise SystemExit("output_or_partial_exists")
    source = base.source_snapshot(repo)
    model_path, model_config_sha256 = base._load_checkpoint_config(repo)
    exclusion_sha256 = None
    tv_root = Path(args.tvbench_root).resolve(strict=True)
    selected, cohort_hash, cohort_counts = build_tvbench_cohort(tv_root, args.n, args.per_task, args.frames)
    cohort.build_qa_prompt = build_qa_prompt_k

    import torch
    from transformers import AutoProcessor
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
                                              min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    label_ids = []
    for label in cohort.LABELS:
        ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"choice_label_tokenization_invalid:{label}")
        label_ids.append(ids[0])
    if len(set(label_ids)) != 5:
        raise ValueError("choice_labels_not_distinct_tokens")
    patch = int(processor.video_processor.temporal_patch_size)

    free, _ = torch.cuda.mem_get_info()
    if free < base.MIN_FREE_VRAM or cohort.host_available_bytes() < base.MIN_HOST_AVAILABLE:
        raise RuntimeError("pre_model_resource_guard_failed")
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map="cuda", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if int(model.config.video_token_id) != int(processor.video_token_id):
        raise ValueError("processor_checkpoint_video_token_mismatch")
    layer_count = len(model.model.language_model.layers)
    label_tensor = torch.tensor(label_ids, device="cuda")

    def forward(device_batch):
        if torch.cuda.mem_get_info()[0] < 2 * base.MIN_FORWARD_VRAM:
            torch.cuda.empty_cache()  # only when VRAM is tight; an unconditional empty_cache per forward forces re-allocation and a sync
        if torch.cuda.mem_get_info()[0] < base.MIN_FORWARD_VRAM or cohort.host_available_bytes() < base.MIN_FORWARD_HOST:
            raise RuntimeError("per_forward_resource_guard_failed")
        with torch.inference_mode():
            out = model(**device_batch, use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].float()
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("nonfinite_answer_logits")
        logp = torch.log_softmax(logits, dim=-1)
        choice_logp = logp[label_tensor].detach().cpu().tolist()
        del out, logits, logp
        return choice_logp

    done_media = set()
    records = []
    if partial.exists():
        for line in partial.read_text().splitlines():
            rec = json.loads(line)
            records.append(rec)
            done_media.add(rec["media_sha256"])
    fd = os.open(partial, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    partial_file = os.fdopen(fd, "a")
    row_indices = range(args.n) if args.mode == "full" else range(min(3, args.n))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = time.monotonic()
    forwards = 0
    try:
        for index in row_indices:
            sample = selected[index]
            media_hash = sample["decode"]["source_sha256"]
            if media_hash in done_media:
                continue
            prepared = cohort.prepare_conditions([sample], processor, torch, args.frames, retain_batches=True)[0]
            batch = prepared["batches"]["full"]
            device_batch = batch.to("cuda")
            input_ids = batch["input_ids"][0].tolist()
            seq_len = len(input_ids)
            geometry = batch["video_grid_thw"][0].tolist()
            groups = base.video_group_key_positions(
                input_ids=input_ids, video_token_id=int(processor.video_token_id), video_grid_thw=geometry,
                spatial_merge_size=int(processor.video_processor.merge_size), temporal_patch_size=patch)
            gold = int(sample["gold"])
            base_logp = forward(device_batch)
            forwards += 1
            k_opts = len(sample["choices"])
            pred = max(range(k_opts), key=lambda j: (base_logp[j], -j))
            correct = pred == gold
            rec = {"task": sample["task"], "n_options": k_opts, "media_sha256": media_hash, "qid_sha256": base.hashlib.sha256(str(sample["qid"]).encode()).hexdigest(),
                   "gold": gold, "predicted": pred, "baseline_correct": correct,
                   "baseline_choice_logprobs": base_logp, "video_grid_thw": geometry, "sequence_length": seq_len,
                   "knockouts": []}
            if correct:
                for edge in edges:
                    blocks = ko.edge_blocks(edge, video_groups=groups, seq_len=seq_len)
                    for center in centers:
                        layers = ko.window_layers(center, args.width, layer_count)
                        handles, state = ko.install_knockout_hooks(
                            model=model, layers_1based=layers, blocks=blocks, torch_module=torch)
                        try:
                            ko_logp = forward(device_batch)
                        finally:
                            ko.remove_hooks(handles)
                        forwards += 1
                        if state["calls"] != {int(i): 1 for i in layers}:
                            raise ValueError("knockout_layer_calls_mismatch")
                        pk = float(torch.tensor(ko_logp[pred]).exp())
                        pb = float(torch.tensor(base_logp[pred]).exp())
                        rec["knockouts"].append({
                            "edge": edge, "center": center, "layers": [layers[0], layers[-1]],
                            "cells": state["expected_cells"], "choice_logprobs": ko_logp,
                            "choice_prob_pred": pk, "base_choice_prob_pred": pb,
                            "relative_change_percent": relative_change_percent(pb, pk),
                            "predicted_after": max(range(k_opts), key=lambda j: (ko_logp[j], -j))})
            partial_file.write(json.dumps(rec) + "\n")
            partial_file.flush()
            os.fsync(partial_file.fileno())
            records.append(rec)
            del prepared, batch, device_batch
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        partial_file.close()
    complete = len(records) >= (args.n if args.mode == "full" else min(3, args.n))
    payload = {
        "schema_version": 1, "purpose": "pathway_attention_knockout_replication_of_map_the_flow_tvbench",
        "status": "completed" if complete else "incomplete", "mode": args.mode, "dataset_split": args.dataset_split,
        "requested_videos": args.n, "attempted_records": len(records),
        "baseline_correct": sum(r["baseline_correct"] for r in records),
        "centers": list(centers), "window_width": args.width, "edges": list(edges), "frames": args.frames,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": base.MODEL_REVISION,
        "model_config_sha256": model_config_sha256, "precision": "4-bit weights, BF16 compute, fp32 lm_head readout",
        "readout": "fp32_lm_head_matmul_of_bf16_final_hidden_state", "seed": SEED, "layer_count": layer_count,
        "cohort_sha256": cohort_hash, "cohort_counts": cohort_counts, "tvbench_tasks": list(TASKS), "per_task": args.per_task, "exclusion_manifest_sha256": exclusion_sha256, "source_snapshot": source,
        "runner_sha256": base.sha256_file(RUNNER_PATH),
        "knockout_module_sha256": base.sha256_file(RUNNER_PATH.parent / "qta_edge_knockout.py"),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0)},
        "elapsed_seconds": time.monotonic() - started, "forwards": forwards,
        "rights_status": "unverified; local-only operation", "human_review_performed": False,
        "interpretation_limit": "attention-pathway dependence of the predicted answer on TVBench (4 of the 5 paper tasks); not explanation faithfulness",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    base.write_private_json_exclusive(output, payload)
    print(json.dumps({"status": payload["status"], "records": len(records), "baseline_correct": payload["baseline_correct"],
                      "forwards": forwards, "output": output.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
