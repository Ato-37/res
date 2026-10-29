#!/usr/bin/env python3
"""Logit lens on video tokens (Map the Flow Figs 4-5 / Table D) on Qwen2.5-VL-7B and the TVBench Action Sequence task (plan: experiments/2026-10-29_map_the_flow_logit_lens_plan.md).

Official protocol (analysis/logit_lens_analysis.py, read not executed): project the residual stream of every video token at every layer through the unembedding, keep the top-1 token, count how often it is one of the task's spatial
(object nouns) or temporal (action verbs) keywords (first sub-word of each word), per layer. Here the residual of layer l (1..28) gets the final norm and the fp32 lm_head; layer 28 is already normalized.
Extension (prespecified): the same count with cross-frame attention blocked in ALL layers, to test whether temporal-keyword emergence depends on cross-frame interaction.
Only counts are stored (no decoded text, no frames). Receipts are local-only (mode 0600), partial JSONL per video (--resume).
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

import run_nextgqa_annotation_key_bias_fp32_readout  # noqa: F401
import run_nextgqa_annotation_key_bias as base
import run_nextgqa_gold_interval_margin_pilot as cohort
import qta_edge_knockout as ko
import run_tvbench_paper_protocol as pp

SEED = 20261029
RUNNER_PATH = Path(__file__).resolve()
SPATIAL = ['bag', 'bed', 'blanket', 'book', 'box', 'cabinet', 'camera', 'clothes', 'cup', 'glass', 'bottle', 'dish', 'door', 'floor', 'food',
           'laptop', 'paper', 'person', 'phone', 'sandwich', 'table']
TEMPORAL = ['eat', 'close', 'do', 'down', 'drink', 'hold', 'on', 'open', 'put', 'sit', 'throw', 'tidy', 'take', 'up']
CHUNK = 256


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--tvbench-root", required=True)
    p.add_argument("--n", type=int, default=100)
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
    selected, cohort_hash, cohort_counts = pp.build_tvbench_cohort(tv_root, args.n, args.n, args.frames)  # per_task = n -> action_sequence only
    if {s["task"] for s in selected} != {"action_sequence"}:
        raise ValueError("cohort_not_action_sequence_only")
    cohort.build_qa_prompt = pp.build_qa_prompt_paper

    import torch
    from transformers import AutoProcessor
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    tok = processor.tokenizer
    patch = int(processor.video_processor.temporal_patch_size)
    free, _ = torch.cuda.mem_get_info()
    if free < base.MIN_FREE_VRAM or cohort.host_available_bytes() < base.MIN_HOST_AVAILABLE:
        raise RuntimeError("pre_model_resource_guard_failed")
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
        device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    layer_count = len(model.model.language_model.layers)
    norm = model.model.language_model.norm
    vocab_n = model.lm_head.out_features
    words = [("spatial", w) for w in SPATIAL] + [("temporal", w) for w in TEMPORAL]
    word_of_token = torch.full((vocab_n,), -1, dtype=torch.long, device="cuda")
    for wi, (_, w) in enumerate(words):
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            ids = tok(variant, add_special_tokens=False).input_ids
            if ids:
                word_of_token[ids[0]] = wi  # first sub-word of the word (official: first tokenized keyword)
    label_ids = [tok(l, add_special_tokens=False).input_ids[0] for l in cohort.LABELS]
    label_tensor = torch.tensor(label_ids, device="cuda")

    def lens_counts(device_batch, vision_idx):
        """Returns per-layer correct-free counts [layer][word] of top-1 keyword hits over the video tokens, and the letter log-probs."""
        if torch.cuda.mem_get_info()[0] < base.MIN_FORWARD_VRAM or cohort.host_available_bytes() < base.MIN_FORWARD_HOST:
            raise RuntimeError("per_forward_resource_guard_failed")
        vid = torch.tensor(vision_idx, device="cuda")
        with torch.inference_mode():
            out = model(**device_batch, use_cache=False, logits_to_keep=1, output_hidden_states=True)
            lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)[label_tensor].cpu().tolist()
            hs = out.hidden_states
            counts = []
            for li in range(1, len(hs)):
                h = hs[li][0, vid]
                h = h if li == len(hs) - 1 else norm(h)
                hit = torch.zeros(len(words), dtype=torch.long, device="cuda")
                for s in range(0, h.shape[0], CHUNK):
                    top = model.lm_head(h[s:s + CHUNK][None]).float()[0].argmax(-1)
                    w = word_of_token[top]
                    w = w[w >= 0]
                    if w.numel():
                        hit += torch.bincount(w, minlength=len(words))
                counts.append(hit.cpu().tolist())
        del out, hs
        return counts, lp

    done, records = set(), []
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                records.append(rec)
                done.add(rec["media_sha256"])
    pf = os.fdopen(os.open(partial, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a")
    rows = range(args.n) if args.mode == "full" else range(min(3, args.n))
    started, forwards = time.monotonic(), 0
    try:
        for index in rows:
            sample = selected[index]
            mh = sample["decode"]["source_sha256"]
            if mh in done:
                continue
            prepared = cohort.prepare_conditions([sample], processor, torch, args.frames, retain_batches=True, only_full=True)[0]
            batch = prepared["batches"]["full"]
            device_batch = batch.to("cuda")
            input_ids = batch["input_ids"][0].tolist()
            seq_len = len(input_ids)
            geometry = batch["video_grid_thw"][0].tolist()
            groups = base.video_group_key_positions(input_ids=input_ids, video_token_id=int(processor.video_token_id), video_grid_thw=geometry,
                                                    spatial_merge_size=int(processor.video_processor.merge_size), temporal_patch_size=patch)
            vision = tuple(sorted(p for g in groups for p in g))
            gold, k = int(sample["gold"]), len(sample["choices"])
            base_counts, base_lp = lens_counts(device_batch, vision)
            forwards += 1
            blocks = ko.edge_blocks("cross_frame", video_groups=groups, seq_len=seq_len)
            handles, st = ko.install_knockout_hooks(model=model, layers_1based=tuple(range(1, layer_count + 1)), blocks=blocks, torch_module=torch)
            try:
                cf_counts, cf_lp = lens_counts(device_batch, vision)
            finally:
                ko.remove_hooks(handles)
            forwards += 1
            if st["calls"] != {i: 1 for i in range(1, layer_count + 1)}:
                raise ValueError("knockout_layer_calls_mismatch")
            rec = {"task": sample["task"], "media_sha256": mh, "gold": gold, "n_options": k, "n_video_tokens": len(vision),
                   "baseline_predicted": max(range(k), key=lambda j: (base_lp[j], -j)), "crossframe_blocked_predicted": max(range(k), key=lambda j: (cf_lp[j], -j)),
                   "counts_baseline": base_counts, "counts_crossframe_blocked": cf_counts}
            pf.write(json.dumps(rec) + "\n")
            pf.flush()
            os.fsync(pf.fileno())
            records.append(rec)
            del prepared, batch, device_batch
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        pf.close()
    complete = len(records) >= (args.n if args.mode == "full" else min(3, args.n))
    payload = {"schema_version": 1, "purpose": "map_the_flow_logit_lens_video_tokens_tvbench_action_sequence_qwen", "status": "completed" if complete else "incomplete",
               "mode": args.mode, "requested_videos": args.n, "attempted_records": len(records), "frames": args.frames,
               "words": [{"kind": kd, "word": w} for kd, w in words], "layer_count": layer_count, "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": base.MODEL_REVISION,
               "model_config_sha256": model_config_sha256, "precision": "4-bit weights, BF16 compute, fp32 lm_head readout", "seed": SEED, "cohort_sha256": cohort_hash,
               "cohort_counts": cohort_counts, "source_snapshot": source, "runner_sha256": base.sha256_file(RUNNER_PATH),
               "runtime": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
               "elapsed_seconds": time.monotonic() - started, "forwards": forwards, "rights_status": "unverified; local-only operation", "human_review_performed": False,
               "interpretation_limit": "keyword counts of top-1 logit-lens tokens of video tokens; descriptive of representations, not explanation faithfulness",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    base.write_private_json_exclusive(output, payload)
    print(json.dumps({"status": payload["status"], "records": len(records), "forwards": forwards, "output": output.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
