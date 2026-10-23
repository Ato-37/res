#!/usr/bin/env python3
"""CSL: causal self-localization on Qwen2.5-VL-7B / NExT-GQA (plan: experiments/2026-10-24_csl_plan.md; companion of SGC).

Per video: unmodified forward; then 16 leave-one-group-out video->text knockouts (layers 6-20, text rows, keys of ONE temporal group
set to -inf; WP1 pathway) -> causal score c_g = self-margin drop (label-free) - margin without g (larger = answer depends more on g). Then keep-mode arms
(additive bias -ln256 on the keys of all NON-kept groups, all post-video rows, all 28 layers): oracle, random (disjoint hash sham), csl (top-|A|
by c_g), csl4 (top-4), random4. No annotation is used for scoring.
Local-only receipts (mode 0600); per-video appends to a partial JSONL (--resume).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_nextgqa_annotation_key_bias_fp32_readout  # noqa: F401  (fp32 lm_head readout patch)
import run_nextgqa_annotation_key_bias as base
import run_nextgqa_gold_interval_margin_pilot as cohort
import qta_attention_scores as sc
from qta_multiquery_video_key_intervention import install_multiquery_video_key_intervention_hooks, remove_hooks

SEED = 20261023
RUNNER_PATH = Path(__file__).resolve()
KNOCK_LAYERS_0B = tuple(range(5, 20))  # layers 6-20, 1-based (WP0/WP1 video->text band)
BIAS = -math.log(256.0)
FIXED_K = 4


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--dataset-split", choices=("val", "test"), required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--frames", type=int, choices=(16, 32), default=32)
    p.add_argument("--exclude-media-sha256", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--user-authorized-local-only", action="store_true")
    return p.parse_args(argv)


def self_margin(lp, pred):
    """Label-free margin: log-prob of the baseline-predicted option minus the best other option."""
    return lp[pred] - max(v for j, v in enumerate(lp) if j != pred)


def complement_keys(groups, kept):
    kept = set(kept)
    return tuple(sorted(p for i, g in enumerate(groups) if i not in kept for p in g))


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.user_authorized_local_only:
        raise SystemExit("explicit_user_authorized_local_only_required")
    repo = RUNNER_PATH.parents[1]
    private_root = (repo / "data/private_nextgqa").resolve(strict=True)
    output = base.private_output_path(args.output, repo)
    partial = Path(str(output) + ".partial.jsonl")
    if output.exists() or (partial.exists() and not args.resume):
        raise SystemExit("output_or_partial_exists")
    source = base.source_snapshot(repo)
    video_root = Path(args.video_root).resolve(strict=True)
    model_path, model_config_sha256 = base._load_checkpoint_config(repo)
    exclusion_payload, exclusion_sha256 = base.read_private_json(Path(args.exclude_media_sha256), private_root)
    exclusion_hashes = cohort.validate_private_media_exclusion(exclusion_payload)
    selected, counts, meta, cohort_hash = base._freeze_exact(
        repo, video_root=video_root, n=args.n, frame_count=args.frames, exclusion=exclusion_hashes,
        lock_assignments=None, dataset_split=args.dataset_split)

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
    layers = model.model.language_model.layers
    layer_count = len(layers)
    label_tensor = torch.tensor(label_ids, device="cuda")

    def forward(device_batch):
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < base.MIN_FORWARD_VRAM or cohort.host_available_bytes() < base.MIN_FORWARD_HOST:
            raise RuntimeError("per_forward_resource_guard_failed")
        with torch.inference_mode():
            out = model(**device_batch, use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].float()
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("nonfinite_answer_logits")
        values = torch.log_softmax(logits, dim=-1)[label_tensor].detach().cpu().tolist()
        del out, logits
        return values

    def knockout_forward(device_batch, text_rows, group_keys):
        handles = install_multiquery_video_key_intervention_hooks(
            model=model, layer_indices=KNOCK_LAYERS_0B, biased_positions=(), ablated_positions=tuple(group_keys),
            query_positions=text_rows, logit_bias=BIAS, torch_module=torch)
        try:
            return forward(device_batch)
        finally:
            remove_hooks(handles)

    def keep_forward(device_batch, query_rows, groups, kept):
        biased = complement_keys(groups, kept)
        if not biased or any(p >= query_rows[0] for p in biased):
            raise ValueError("keep_arm_keys_invalid")
        handles = install_multiquery_video_key_intervention_hooks(
            model=model, layer_indices=tuple(range(layer_count)), biased_positions=biased, ablated_positions=(),
            query_positions=query_rows, logit_bias=BIAS, torch_module=torch)
        try:
            return forward(device_batch), len(biased)
        finally:
            remove_hooks(handles)

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
    torch.cuda.manual_seed_all(SEED)
    started, forwards = time.monotonic(), 0
    try:
        for index in row_indices:
            sample = selected[index]
            media_hash = sample["decode"]["source_sha256"]
            if media_hash in done_media:
                continue
            qid_hash = base.hashlib.sha256(str(sample["qid"]).encode()).hexdigest()
            prepared = cohort.prepare_conditions([sample], processor, torch, args.frames, retain_batches=True)[0]
            batch = prepared["batches"]["full"]
            device_batch = batch.to("cuda")
            input_ids = batch["input_ids"][0].tolist()
            seq_len = len(input_ids)
            geometry = batch["video_grid_thw"][0].tolist()
            groups = base.video_group_key_positions(
                input_ids=input_ids, video_token_id=int(processor.video_token_id), video_grid_thw=geometry,
                spatial_merge_size=int(processor.video_processor.merge_size), temporal_patch_size=patch)
            rec = {"media_sha256": media_hash, "qid_sha256": qid_hash, "video_grid_thw": geometry, "sequence_length": seq_len}
            try:
                ann = base.map_annotation_frames_to_groups(
                    evidence_frame_indices=sample["evidence_positions"], frame_count=args.frames,
                    temporal_patch_size=patch, temporal_group_count=len(groups))
                rand_k = base.choose_disjoint_hash_sham(
                    temporal_group_count=len(groups), annotation_groups=ann, opaque_row_token=base.row_token(media_hash, qid_hash))
            except ValueError as exc:
                rec.update({"excluded": True, "exclusion_reason": str(exc)[:80]})
                partial_file.write(json.dumps(rec) + "\n"); partial_file.flush(); os.fsync(partial_file.fileno())
                records.append(rec)
                del prepared, batch, device_batch
                gc.collect(); torch.cuda.empty_cache()
                continue
            last_video = max(p for g in groups for p in g)
            text_rows = tuple(range(last_video + 1, seq_len - 1))
            query_rows = tuple(range(last_video + 1, seq_len))
            gold = int(sample["gold"])
            base_logp = forward(device_batch)
            forwards += 1
            foil = base.baseline_best_foil(base_logp, gold)
            pred = max(range(5), key=lambda j: (base_logp[j], -j))

            def margin(lp):
                return base.fixed_gold_foil_margin(lp, gold_index=gold, baseline_foil_index=foil)

            ko_lps = []
            for g in groups:
                ko_lps.append(knockout_forward(device_batch, text_rows, g))
                forwards += 1
            scores = [self_margin(base_logp, pred) - self_margin(lp, pred) for lp in ko_lps]
            sg = sc.top_groups(scores, len(ann))
            sg4 = sc.top_groups(scores, FIXED_K)
            r4 = sc.hash_random_groups(FIXED_K, len(groups), base.row_token(media_hash, qid_hash))
            arms = {"oracle": ann, "random": rand_k, "csl": sg, "csl4": sg4, "random4": r4}
            rec.update({"excluded": False, "gold": gold, "predicted": pred, "baseline_correct": pred == gold,
                        "baseline_best_foil": foil, "baseline_margin": margin(base_logp), "baseline_choice_logprobs": base_logp,
                        "temporal_group_count": len(groups), "annotation_group_indices": list(ann),
                        "group_causal_scores": scores, "arms": {}})
            for name, kept in arms.items():
                lp, nkeys = keep_forward(device_batch, query_rows, groups, kept)
                forwards += 1
                rec["arms"][name] = {"kept_groups": list(kept), "edited_keys": nkeys, "margin": margin(lp), "choice_logprobs": lp,
                                      "predicted_after": max(range(5), key=lambda j: (lp[j], -j))}
            rec["localization"] = {"csl_vs_annotation": sc.overlap_with_chance(sg, ann, len(groups)),
                                   "top1_in_annotation": sc.top_groups(scores, 1)[0] in set(ann)}
            partial_file.write(json.dumps(rec) + "\n"); partial_file.flush(); os.fsync(partial_file.fileno())
            records.append(rec)
            del prepared, batch, device_batch
            gc.collect(); torch.cuda.empty_cache()
    finally:
        partial_file.close()
    complete = len(records) >= (args.n if args.mode == "full" else min(3, args.n))
    payload = {
        "schema_version": 1, "purpose": "causal_self_localization_csl", "status": "completed" if complete else "incomplete",
        "mode": args.mode, "dataset_split": args.dataset_split, "requested_videos": args.n, "recorded_videos": len(records),
        "excluded_videos": sum(bool(r.get("excluded")) for r in records), "frames": args.frames,
        "knockout_layers_1_based": [KNOCK_LAYERS_0B[0] + 1, KNOCK_LAYERS_0B[-1] + 1], "keep_bias_nats": BIAS, "fixed_k": FIXED_K,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": base.MODEL_REVISION, "model_config_sha256": model_config_sha256,
        "precision": "4-bit weights, BF16 compute, fp32 lm_head readout", "seed": SEED, "cohort_sha256": cohort_hash,
        "exclusion_manifest_sha256": exclusion_sha256, "source_snapshot": source, "runner_sha256": base.sha256_file(RUNNER_PATH),
        "module_sha256": base.sha256_file(RUNNER_PATH.parent / "qta_attention_scores.py"),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
        "elapsed_seconds": time.monotonic() - started, "forwards": forwards, "rights_status": "unverified; local-only operation",
        "human_review_performed": False, "interpretation_limit": "answer-margin effect of attention-level cropping; not explanation faithfulness",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    base.write_private_json_exclusive(output, payload)
    print(json.dumps({"status": payload["status"], "records": len(records), "excluded": payload["excluded_videos"], "forwards": forwards, "output": output.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
