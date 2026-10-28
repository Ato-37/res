#!/usr/bin/env python3
"""WP1: evidence-specific pathway knockout on Qwen2.5-VL-7B / NExT-GQA (plan: experiments/2026-10-23_evidence_pathway_knockout_plan.md).

Per video: unmodified forward; then knockouts of the video->text and video->last edges restricted to the keys of the annotated evidence
groups or of an equal-count disjoint hash sham, in a middle layer band (6-20) and in all layers, plus a whole-video video->text
positive control. Receipts are local-only (mode 0600) and hash-identified; each video is appended to a partial JSONL (--resume).
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

import run_nextgqa_annotation_key_bias_fp32_readout  # noqa: F401  (fp32 lm_head readout patch)
import run_nextgqa_annotation_key_bias as base
import run_nextgqa_gold_interval_margin_pilot as cohort
import qta_edge_knockout as ko
import qta_evidence_edge_knockout as ev

SEED = 20261023
RUNNER_PATH = Path(__file__).resolve()
(_BA, _BB), _NL = base.layer_band_from_env()
BANDS = {"band": tuple(range(_BA, _BB + 1)), "all": tuple(range(1, _NL + 1))}


def arms_for(groups_count: int, annotation_groups, sham_groups):
    """(name, edge, key_groups or None for whole video, band)."""
    arms = []
    for band in BANDS:
        for edge, short in (("video_to_text", "v2t"), ("video_to_last", "v2l")):
            arms.append((f"{short}_evidence_{band}", edge, tuple(annotation_groups), band))
            arms.append((f"{short}_sham_{band}", edge, tuple(sham_groups), band))
    for band in BANDS:
        arms.append((f"v2t_allvideo_{band}", "video_to_text", None, band))
    return arms


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--dataset-split", choices=("val", "test"), default="test")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--frames", type=int, choices=(16, 32), default=32)
    p.add_argument("--exclude-media-sha256", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--user-authorized-local-only", action="store_true")
    return p.parse_args(argv)


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
        values = logp[label_tensor].detach().cpu().tolist()
        del out, logits, logp
        return values

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
                ann_groups = base.map_annotation_frames_to_groups(
                    evidence_frame_indices=sample["evidence_positions"], frame_count=args.frames,
                    temporal_patch_size=patch, temporal_group_count=len(groups))
                sham_groups = base.choose_disjoint_hash_sham(
                    temporal_group_count=len(groups), annotation_groups=ann_groups,
                    opaque_row_token=base.row_token(media_hash, qid_hash))
            except ValueError as exc:
                rec.update({"excluded": True, "exclusion_reason": str(exc)[:80], "knockouts": []})
                partial_file.write(json.dumps(rec) + "\n"); partial_file.flush(); os.fsync(partial_file.fileno())
                records.append(rec)
                del prepared, batch, device_batch
                gc.collect(); torch.cuda.empty_cache()
                continue
            gold = int(sample["gold"])
            base_logp = forward(device_batch)
            forwards += 1
            pred = max(range(5), key=lambda j: (base_logp[j], -j))
            foil = base.baseline_best_foil(base_logp, gold)
            rec.update({"excluded": False, "gold": gold, "predicted": pred, "baseline_correct": pred == gold,
                        "baseline_best_foil": foil, "baseline_choice_logprobs": base_logp,
                        "baseline_margin": base.fixed_gold_foil_margin(base_logp, gold_index=gold, baseline_foil_index=foil),
                        "annotation_group_indices": list(ann_groups), "sham_group_indices": list(sham_groups),
                        "temporal_group_count": len(groups), "knockouts": []})
            for name, edge, key_groups, band in arms_for(len(groups), ann_groups, sham_groups):
                blocks = (ev.all_video_blocks(edge, video_groups=groups, seq_len=seq_len) if key_groups is None
                          else ev.evidence_blocks(edge, video_groups=groups, seq_len=seq_len, key_groups=key_groups))
                layers = BANDS[band]
                handles, state = ko.install_knockout_hooks(model=model, layers_1based=layers, blocks=blocks, torch_module=torch)
                try:
                    ko_logp = forward(device_batch)
                finally:
                    ko.remove_hooks(handles)
                forwards += 1
                if state["calls"] != {int(i): 1 for i in layers}:
                    raise ValueError("knockout_layer_calls_mismatch")
                pb = float(torch.tensor(base_logp[pred]).exp())
                pk = float(torch.tensor(ko_logp[pred]).exp())
                rec["knockouts"].append({
                    "arm": name, "edge": edge, "band": band, "cells": state["expected_cells"], "choice_logprobs": ko_logp,
                    "margin": base.fixed_gold_foil_margin(ko_logp, gold_index=gold, baseline_foil_index=foil),
                    "relative_change_percent": (pk - pb) / pb * 100.0, "predicted_after": max(range(5), key=lambda j: (ko_logp[j], -j))})
            partial_file.write(json.dumps(rec) + "\n"); partial_file.flush(); os.fsync(partial_file.fileno())
            records.append(rec)
            del prepared, batch, device_batch
            gc.collect(); torch.cuda.empty_cache()
    finally:
        partial_file.close()
    complete = len(records) >= (args.n if args.mode == "full" else min(3, args.n))
    payload = {
        "schema_version": 1, "purpose": "evidence_specific_pathway_knockout_wp1", "status": "completed" if complete else "incomplete",
        "mode": args.mode, "dataset_split": args.dataset_split, "requested_videos": args.n, "recorded_videos": len(records),
        "excluded_videos": sum(bool(r.get("excluded")) for r in records),
        "baseline_correct": sum(bool(r.get("baseline_correct")) for r in records), "bands": {k: [v[0], v[-1]] for k, v in BANDS.items()},
        "frames": args.frames, "model": base.active_model()[0], "model_revision": base.active_model()[1],
        "model_config_sha256": model_config_sha256, "precision": "4-bit weights, BF16 compute, fp32 lm_head readout", "seed": SEED,
        "cohort_sha256": cohort_hash, "exclusion_manifest_sha256": exclusion_sha256, "source_snapshot": source,
        "runner_sha256": base.sha256_file(RUNNER_PATH),
        "module_sha256": {n: base.sha256_file(RUNNER_PATH.parent / n) for n in ("qta_edge_knockout.py", "qta_evidence_edge_knockout.py")},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
        "elapsed_seconds": time.monotonic() - started, "forwards": forwards, "rights_status": "unverified; local-only operation",
        "human_review_performed": False, "interpretation_limit": "pathway-level dependence on the annotated interval; not explanation faithfulness",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    base.write_private_json_exclusive(output, payload)
    print(json.dumps({"status": payload["status"], "records": len(records), "excluded": payload["excluded_videos"],
                      "baseline_correct": payload["baseline_correct"], "forwards": forwards, "output": output.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
