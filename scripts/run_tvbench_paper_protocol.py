#!/usr/bin/env python3
"""PAPER-PROTOCOL analyses of Map the Flow (Kim et al., ICLR 2026) on Qwen2.5-VL-7B and TVBench (plan: experiments/2026-10-29_map_the_flow_protocol_plan.md).

Same cohort as run_tvbench_pathway.py, but with the paper's protocol read from the official repository: prompt template ending in "Best option:(" (the scored token is the option letter), token groups (question = "Question ... Options ... (X) ..." span, non-option question, true option, false options, last token),
and the targets cross-frame, vql-to-ql (video->question, video->last, question->last; last->last is not implemented because the knockout audit forbids the self cell), question-and-options-to-last, vq-to-true-opt,
plus the accuracy drop of cross-frame blocking in the first and the second half of the layers (Table 2 / Table B), a window-size ablation (k=1,5,13; Fig AB), the layer-wise true/false option probabilities (Fig 9) and the same knockouts on baseline-wrong videos (failure cases, Sec 4).

Previous description (run_tvbench_pathway.py) follows.

Pathway attention-knockout replication of Map the Flow (Kim et al., ICLR 2026) on Qwen2.5-VL-7B and TVBench (the base paper's main benchmark; 8 frames).

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
    p.add_argument("--attn", choices=("eager", "sdpa"), default="sdpa")
    p.add_argument("--sample-mode", choices=("correct", "wrong", "all"), default="all")
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


PAPER_SYSTEM = ("Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, "
                "and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.")


def build_qa_prompt_paper(processor, item, frames, sample_fps):
    """The paper's multiple-choice prompt (official repo, conv_mode eval_mvbench) in Qwen chat form; the answer letter is scored after 'Best option:('."""
    options = "\n".join(f"({label}) {choice}" for label, choice in zip(cohort.LABELS[:len(item["choices"])], item["choices"], strict=True))
    user_text = f"Question: {item['question']}\nOptions:\n{options}\nOnly give the best option."
    messages = [{"role": "system", "content": PAPER_SYSTEM},
                {"role": "user", "content": [{"type": "video", "video": frames, "sample_fps": sample_fps[0] / sample_fps[1]},
                                              {"type": "text", "text": user_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Best option:("
    return text, messages


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
    cohort.build_qa_prompt = build_qa_prompt_paper

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
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
        device_map="cuda", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if int(model.config.video_token_id) != int(processor.video_token_id):
        raise ValueError("processor_checkpoint_video_token_mismatch")
    layer_count = len(model.model.language_model.layers)
    label_tensor = torch.tensor(label_ids, device="cuda")

    def forward(device_batch, want_hidden=False):
        if torch.cuda.mem_get_info()[0] < 2 * base.MIN_FORWARD_VRAM:
            torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < base.MIN_FORWARD_VRAM or cohort.host_available_bytes() < base.MIN_FORWARD_HOST:
            raise RuntimeError("per_forward_resource_guard_failed")
        with torch.inference_mode():
            out = model(**device_batch, use_cache=False, logits_to_keep=1, output_hidden_states=want_hidden)
            logits = out.logits[0, -1].float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite_answer_logits")
            logp = torch.log_softmax(logits, dim=-1)
            choice_logp = logp[label_tensor].detach().cpu().tolist()
            lens = None
            if want_hidden:
                hs = out.hidden_states  # embeddings + one entry per layer; the last entry is already after the final norm
                norm = model.model.language_model.norm
                lens = []
                for li in range(1, len(hs)):
                    h = hs[li][0, -1]
                    h = h if li == len(hs) - 1 else norm(h)
                    lg = model.lm_head(h[None, None, :]).float()[0, 0]
                    lens.append(torch.log_softmax(lg, dim=-1)[label_tensor].detach().cpu().tolist())
        del out, logits, logp
        return (choice_logp, lens) if want_hidden else choice_logp

    tok = processor.tokenizer

    def spans_for(text, input_ids, n_vid_tokens, parts):
        """Token index lists (in the expanded sequence) for character spans of `parts` inside the templated prompt text."""
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        vid_pos = [i for i, t in enumerate(ids) if t == int(processor.video_token_id)]
        if len(vid_pos) != 1:
            raise ValueError("video_placeholder_count_invalid")
        p0 = vid_pos[0]
        expanded = ids[:p0] + [ids[p0]] * n_vid_tokens + ids[p0 + 1:]
        if expanded != list(input_ids):
            raise ValueError("prompt_tokenization_mismatch")
        shift = n_vid_tokens - 1
        out = {}
        for name, (a_, b_) in parts.items():
            idx = [i + (shift if i > p0 else 0) for i, (s0, e0) in enumerate(offs) if e0 > a_ and s0 < b_ and e0 > s0]
            if not idx:
                raise ValueError("span_empty:" + name)
            out[name] = tuple(idx)
        return out

    def run_arm(device_batch, blocks, layers, seq_len):
        handles, state = ko.install_knockout_hooks(model=model, layers_1based=layers, blocks=blocks, torch_module=torch, )
        try:
            lp = forward(device_batch)
        finally:
            ko.remove_hooks(handles)
        if state["calls"] != {int(i): 1 for i in layers}:
            raise ValueError("knockout_layer_calls_mismatch")
        return lp, state["expected_cells"]

    done_media = set()
    records = []
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                records.append(rec)
                done_media.add(rec["media_sha256"])
    partial_file = os.fdopen(os.open(partial, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a")
    row_indices = range(args.n) if args.mode == "full" else range(min(3, args.n))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started, forwards = time.monotonic(), 0
    half = layer_count // 2
    try:
        for index in row_indices:
            sample = selected[index]
            media_hash = sample["decode"]["source_sha256"]
            if media_hash in done_media:
                continue
            prepared = cohort.prepare_conditions([sample], processor, torch, args.frames, retain_batches=True, only_full=True)[0]
            batch = prepared["batches"]["full"]
            device_batch = batch.to("cuda")
            input_ids = batch["input_ids"][0].tolist()
            seq_len = len(input_ids)
            geometry = batch["video_grid_thw"][0].tolist()
            groups = base.video_group_key_positions(
                input_ids=input_ids, video_token_id=int(processor.video_token_id), video_grid_thw=geometry,
                spatial_merge_size=int(processor.video_processor.merge_size), temporal_patch_size=patch)
            vision = tuple(sorted(p for g in groups for p in g))
            n_vid = sum(1 for t in input_ids if t == int(processor.video_token_id))
            k_opts = len(sample["choices"])
            gold = int(sample["gold"])
            text, _ = prepared["prompt_text_messages"] if "prompt_text_messages" in prepared else build_qa_prompt_paper(processor, sample, None, sample["decode"]["effective_sample_fps"])
            q_head = f"Question: {sample['question']}"
            opt_lines = [f"({cohort.LABELS[i]}) {c}" for i, c in enumerate(sample["choices"])]
            q_all_text = q_head + "\nOptions:\n" + "\n".join(opt_lines)
            q0 = text.find(q_all_text)
            if q0 < 0:
                raise ValueError("question_text_not_found")
            parts = {"question": (q0, q0 + len(q_all_text)), "nonopt": (q0, q0 + len(q_head))}
            cur = q0 + len(q_all_text) - sum(len(x) for x in opt_lines) - (len(opt_lines) - 1)
            for i, line in enumerate(opt_lines):
                st_ = text.find(line, q0)
                parts[f"opt{i}"] = (st_, st_ + len(line))
            sp = spans_for(text, input_ids, n_vid, parts)
            last = seq_len - 1
            question, nonopt = sp["question"], sp["nonopt"]
            base_logp, lens = forward(device_batch, want_hidden=True)
            forwards += 1
            pred = max(range(k_opts), key=lambda j: (base_logp[j], -j))
            correct = pred == gold
            rec = {"task": sample["task"], "n_options": k_opts, "media_sha256": media_hash, "gold": gold, "predicted": pred, "baseline_correct": correct,
                   "baseline_choice_logprobs": base_logp, "video_grid_thw": geometry, "sequence_length": seq_len,
                   "span_sizes": {k: len(v) for k, v in sp.items()},
                   "lens_choice_logprobs_per_layer": [[round(x, 4) for x in row[:k_opts]] for row in lens], "arms": []}
            do_arms = (args.sample_mode == "all") or (args.sample_mode == "correct" and correct) or (args.sample_mode == "wrong" and not correct)
            if do_arms:
                true_opt = sp[f"opt{gold}"]
                false_opts = tuple(sorted(p for i in range(k_opts) if i != gold for p in sp[f"opt{i}"]))
                targets = [("cross-frame", "No cross-frame interactions", ko.edge_blocks("cross_frame", video_groups=groups, seq_len=seq_len)),
                           ("vql-to-ql", "Video -/-> Question", [(question, vision)]),
                           ("vql-to-ql", "Video -/-> Last", [((last,), vision)]),
                           ("vql-to-ql", "Question -/-> Last", [((last,), question)]),
                           ("question-and-options-to-last", "Non-option question -/-> Last", [((last,), nonopt)]),
                           ("question-and-options-to-last", "True option -/-> Last", [((last,), true_opt)]),
                           ("question-and-options-to-last", "False option -/-> Last", [((last,), false_opts)]),
                           ("vq-to-true-opt", "Video -/-> Non-option question", [(nonopt, vision)]),
                           ("vq-to-true-opt", "Non-option question -/-> True option", [(true_opt, nonopt)]),
                           ("vq-to-true-opt", "Video -/-> True option", [(true_opt, vision)])]
                if not correct:  # failure cases (paper Sec 4): cross-frame, video->question, question->last only
                    targets = [targets[0], targets[1], targets[3]]
                pb = float(torch.tensor(base_logp[pred]).exp())
                def record(target, desc, layers, lp, cells, kind="sweep"):
                    pk = float(torch.tensor(lp[pred]).exp())
                    rec["arms"].append({"kind": kind, "target": target, "desc": desc, "layers": [layers[0], layers[-1]], "n_layers": len(layers), "cells": cells,
                                        "prob_pred": pk, "relative_change_percent": relative_change_percent(pb, pk),
                                        "predicted_after": max(range(k_opts), key=lambda j: (lp[j], -j))})
                for target, desc, blocks in targets:
                    for center in centers:
                        layers = ko.window_layers(center, args.width, layer_count)
                        lp, cells = run_arm(device_batch, blocks, layers, seq_len)
                        forwards += 1
                        record(target, desc, layers, lp, cells)
                cf = targets[0][2]
                for desc, layers in (() if not correct else (("first half of layers", tuple(range(1, half + 1))), ("second half of layers", tuple(range(half + 1, layer_count + 1))))):
                    lp, cells = run_arm(device_batch, cf, layers, seq_len)
                    forwards += 1
                    record("cross-frame", desc, layers, lp, cells, kind="half")
                for width in ((1, 5, 13) if correct else ()):
                    for center in (5, 14, 23):
                        layers = ko.window_layers(center, width, layer_count)
                        lp, cells = run_arm(device_batch, cf, layers, seq_len)
                        forwards += 1
                        record("cross-frame", f"window k={width}", layers, lp, cells, kind="window")
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
        "schema_version": 1, "purpose": "map_the_flow_paper_protocol_analyses_tvbench_qwen",
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
