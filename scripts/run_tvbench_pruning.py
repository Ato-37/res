#!/usr/bin/env python3
"""Effective-pathway pruning (Map the Flow Table 3 / E) on Qwen2.5-VL-7B and TVBench (plan: experiments/2026-10-29_map_the_flow_pruning_plan.md).

Read from the official repository (analysis/effective_pathway_analysis.py, not executed): everything is blocked EXCEPT the effective pathway, i.e. cross-frame attention outside layers 6-15, video->question outside 6-20,
question->last outside 16-25 (LLaVA-NeXT-7B, 32 layers), video->last in all layers (and, in the paper, video->video after L20, question->question after L25, last->last); the random control blocks the same number of cells at random.
Here the layer ranges are rescaled to the 28 layers of Qwen2.5-VL-7B (round(i*28/32)); video->video, question->question and last->last (self cells, forbidden by the knockout audit) are NOT blocked, so the suppressed share is lower than the paper's 58%.
The random control draws, per layer, the same number of causal off-diagonal cells as the pathway arm blocks in that layer-average, uniformly from the region after the first video token (as the official random mode, without the diagonal).
All videos are scored (accuracy over the whole cohort, as in Table 3). Receipts are local-only (mode 0600); each video is appended to a partial JSONL (--resume).
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
import run_tvbench_paper_protocol as pp

SEED = 20261029
RUNNER_PATH = Path(__file__).resolve()
PAPER_LAYERS = 32
# official ranges for LLaVA-NeXT-7B-Video-FT, 0-based inclusive
PAPER_RANGES = {"cross_frame": (5, 14), "video_to_question": (5, 19), "question_to_last": (15, 24)}
N_RANDOM = 3


def scaled_layers(lo: int, hi: int, layer_count: int) -> tuple[int, ...]:
    """1-based layer numbers of the paper range rescaled to layer_count layers."""
    a, b = round(lo * layer_count / PAPER_LAYERS), round(hi * layer_count / PAPER_LAYERS)
    return tuple(range(a + 1, b + 2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--tvbench-root", required=True)
    p.add_argument("--per-task", type=int, default=100)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--frames", type=int, choices=(8, 16, 32), default=8)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--attn", choices=("eager", "sdpa"), default="sdpa")
    p.add_argument("--user-authorized-local-only", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.user_authorized_local_only:
        raise SystemExit("explicit_user_authorized_local_only_required")
    repo = RUNNER_PATH.parents[1]
    output = base.private_output_path(args.output, repo)
    partial = Path(str(output) + ".partial.jsonl")
    if output.exists() or (partial.exists() and not args.resume):
        raise SystemExit("output_or_partial_exists")
    source = base.source_snapshot(repo)
    model_path, model_config_sha256 = base._load_checkpoint_config(repo)
    tv_root = Path(args.tvbench_root).resolve(strict=True)
    selected, cohort_hash, cohort_counts = pp.build_tvbench_cohort(tv_root, args.n, args.per_task, args.frames)
    cohort.build_qa_prompt = pp.build_qa_prompt_paper

    import torch
    from transformers import AutoProcessor
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    label_ids = []
    for label in cohort.LABELS:
        ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"choice_label_tokenization_invalid:{label}")
        label_ids.append(ids[0])
    patch = int(processor.video_processor.temporal_patch_size)
    free, _ = torch.cuda.mem_get_info()
    if free < base.MIN_FREE_VRAM or cohort.host_available_bytes() < base.MIN_HOST_AVAILABLE:
        raise RuntimeError("pre_model_resource_guard_failed")
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
        device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    layer_count = len(model.model.language_model.layers)
    layers_mod = model.model.language_model.layers
    label_tensor = torch.tensor(label_ids, device="cuda")
    ranges = {k: scaled_layers(lo, hi, layer_count) for k, (lo, hi) in PAPER_RANGES.items()}
    all_layers = tuple(range(1, layer_count + 1))
    tok = processor.tokenizer

    def forward(device_batch):
        if torch.cuda.mem_get_info()[0] < 2 * base.MIN_FORWARD_VRAM:
            torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < base.MIN_FORWARD_VRAM or cohort.host_available_bytes() < base.MIN_FORWARD_HOST:
            raise RuntimeError("per_forward_resource_guard_failed")
        with torch.inference_mode():
            out = model(**device_batch, use_cache=False, logits_to_keep=1)
            logits = out.logits[0, -1].float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite_answer_logits")
            lp = torch.log_softmax(logits, dim=-1)[label_tensor].detach().cpu().tolist()
        del out, logits
        return lp

    def question_span(text, input_ids, n_vid, q0, q1):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        vid_pos = [i for i, t in enumerate(ids) if t == int(processor.video_token_id)]
        if len(vid_pos) != 1:
            raise ValueError("video_placeholder_count_invalid")
        p0 = vid_pos[0]
        if ids[:p0] + [ids[p0]] * n_vid + ids[p0 + 1:] != list(input_ids):
            raise ValueError("prompt_tokenization_mismatch")
        shift = n_vid - 1
        idx = [i + (shift if i > p0 else 0) for i, (s0, e0) in enumerate(offs) if e0 > q0 and s0 < q1 and e0 > s0]
        if not idx:
            raise ValueError("span_empty")
        return tuple(idx)

    def pathway_plan(groups, question, vision, last, seq_len):
        """Per layer (1-based): list of (rows, cols) blocks. Everything outside the effective pathway is blocked."""
        cf = ko.edge_blocks("cross_frame", video_groups=groups, seq_len=seq_len)
        plan = {}
        for layer in all_layers:
            blocks = [((last,), vision)]  # video -> last: all layers
            if layer not in ranges["cross_frame"]:
                blocks += cf
            if layer not in ranges["video_to_question"]:
                blocks.append((question, vision))
            if layer not in ranges["question_to_last"]:
                blocks.append(((last,), question))
            plan[layer] = blocks
        return plan

    def run_pathway(device_batch, plan):
        handles, calls = [], {}
        try:
            by_set = {}
            for layer, blocks in plan.items():
                by_set.setdefault(json.dumps([[list(r), list(c)] for r, c in blocks]), (blocks, []))[1].append(layer)
            for blocks, layers in by_set.values():
                hs, st = ko.install_knockout_hooks(model=model, layers_1based=layers, blocks=blocks, torch_module=torch)
                handles += hs
                calls[tuple(layers)] = st
            lp = forward(device_batch)
        finally:
            ko.remove_hooks(handles)
        for layers, st in calls.items():
            if st["calls"] != {int(i): 1 for i in layers}:
                raise ValueError("knockout_layer_calls_mismatch")
        return lp

    def run_random(device_batch, n_cells_per_layer, vision_start, seed):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        handles, hits = [], []

        def make_hook(layer):
            def hook(_module, args_, kwargs):
                mask = kwargs.get("attention_mask")
                if mask is None:
                    from qta_mask_utils import causal_additive_mask
                    mask = causal_additive_mask(kwargs["hidden_states"], torch)
                if not torch.is_tensor(mask) or mask.ndim != 4 or mask.shape[0] != 1:
                    raise ValueError("knockout_attention_mask_invalid")
                s = mask.shape[-1]
                elig = torch.tril(torch.ones(s, s, dtype=torch.bool, device=mask.device), diagonal=-1)
                elig[:vision_start, :] = False
                elig[:, :vision_start] = False
                flat = torch.nonzero(elig.reshape(-1), as_tuple=False).squeeze(1)
                pick = flat[torch.randperm(flat.numel(), generator=gen, device=mask.device)[:n_cells_per_layer]]
                edited = mask.clone()
                low = torch.finfo(edited.dtype).min
                r_, c_ = pick // s, pick % s
                if not bool((mask[0, 0][r_, c_] == 0).all()):
                    raise ValueError("knockout_cells_not_originally_visible")
                edited[0, 0][r_, c_] = low
                hits.append(int(torch.count_nonzero(edited != mask).item()))
                new_kwargs = dict(kwargs)
                new_kwargs["attention_mask"] = edited
                return args_, new_kwargs
            return hook

        try:
            for layer in all_layers:
                handles.append(layers_mod[layer - 1].self_attn.register_forward_pre_hook(make_hook(layer), with_kwargs=True))
            lp = forward(device_batch)
        finally:
            ko.remove_hooks(handles)
        if len(hits) != layer_count or any(h != n_cells_per_layer for h in hits):
            raise ValueError("random_block_count_mismatch")
        return lp

    done_media, records = set(), []
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                records.append(rec)
                done_media.add(rec["media_sha256"])
    partial_file = os.fdopen(os.open(partial, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a")
    row_indices = range(args.n) if args.mode == "full" else range(min(3, args.n))
    torch.manual_seed(SEED)
    started, forwards = time.monotonic(), 0
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
            n_vid = len(vision)
            k_opts = len(sample["choices"])
            gold = int(sample["gold"])
            text, _ = pp.build_qa_prompt_paper(processor, sample, None, sample["decode"]["effective_sample_fps"])
            opt_lines = [f"({cohort.LABELS[i]}) {c}" for i, c in enumerate(sample["choices"])]
            q_all = f"Question: {sample['question']}\nOptions:\n" + "\n".join(opt_lines)
            q0 = text.find(q_all)
            if q0 < 0:
                raise ValueError("question_text_not_found")
            question = question_span(text, input_ids, n_vid, q0, q0 + len(q_all))
            last = seq_len - 1
            base_lp = forward(device_batch)
            forwards += 1
            plan = pathway_plan(groups, question, vision, last, seq_len)
            cells_per_layer = [sum(len(r) * len(c) for r, c in plan[layer]) for layer in all_layers]
            total_cells = sum(cells_per_layer)
            baseline_cells = layer_count * (seq_len * (seq_len + 1) // 2)
            fraction = total_cells / baseline_cells
            path_lp = run_pathway(device_batch, plan)
            forwards += 1
            n_per_layer = round(total_cells / layer_count)
            rand_lps = []
            for r in range(N_RANDOM):
                rand_lps.append(run_random(device_batch, n_per_layer, vision[0], SEED + 1000 * index + r))
                forwards += 1

            def pick(lp):
                return max(range(k_opts), key=lambda j: (lp[j], -j))
            rec = {"task": sample["task"], "n_options": k_opts, "media_sha256": media_hash, "gold": gold, "sequence_length": seq_len,
                   "baseline_choice_logprobs": base_lp[:k_opts], "pathway_choice_logprobs": path_lp[:k_opts],
                   "random_choice_logprobs": [lp[:k_opts] for lp in rand_lps],
                   "predicted": {"baseline": pick(base_lp), "pathway": pick(path_lp), "random": [pick(lp) for lp in rand_lps]},
                   "blocked_cells": total_cells, "baseline_cells": baseline_cells, "blocked_fraction": fraction, "random_cells_per_layer": n_per_layer}
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
        "schema_version": 1, "purpose": "map_the_flow_effective_pathway_pruning_tvbench_qwen", "status": "completed" if complete else "incomplete",
        "mode": args.mode, "requested_videos": args.n, "attempted_records": len(records), "frames": args.frames,
        "paper_ranges_0based_32_layers": PAPER_RANGES, "scaled_layers_1based": {k: [v[0], v[-1]] for k, v in ranges.items()}, "n_random": N_RANDOM,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": base.MODEL_REVISION, "model_config_sha256": model_config_sha256,
        "precision": "4-bit weights, BF16 compute, fp32 lm_head readout", "seed": SEED, "layer_count": layer_count,
        "cohort_sha256": cohort_hash, "cohort_counts": cohort_counts, "per_task": args.per_task, "source_snapshot": source,
        "runner_sha256": base.sha256_file(RUNNER_PATH), "knockout_module_sha256": base.sha256_file(RUNNER_PATH.parent / "qta_edge_knockout.py"),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
        "elapsed_seconds": time.monotonic() - started, "forwards": forwards,
        "rights_status": "unverified; local-only operation", "human_review_performed": False,
        "interpretation_limit": "accuracy under attention pruning on TVBench (4 of 5 paper tasks); not explanation faithfulness",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    base.write_private_json_exclusive(output, payload)
    print(json.dumps({"status": payload["status"], "records": len(records), "forwards": forwards, "output": output.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
