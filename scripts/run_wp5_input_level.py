#!/usr/bin/env python3
"""WP5: input-level necessity/sufficiency of the models own cited interval (plan: experiments/2026-10-28_wp5_wp6_input_level_and_stability_plan.md).

No generation: the cited groups, the equal-count sham and the annotation groups are read from the WP2 receipt (--citations). Arms replace frames of the INPUT video (nearest outside frame,
qta_frame_ops.py): replace cited / sham (necessity), keep only cited / sham / annotation (sufficiency), replace annotation (reference), all frames static (positive control).

Original WP2 description follows.

Per video: (1) the model is asked for its answer AND the time span (seconds) that supports it; the span is parsed and mapped to temporal groups
(its own citation, no annotation); (2) standard forced-choice baseline; (3) video->text knockouts (layers 6-20) of the cited groups, an equal-count
disjoint hash sham, the annotation groups (reference) and the whole video (positive control). Label-free endpoint: self-margin of the baseline-predicted option.
Receipts are local-only (mode 0600) and hash-identified; each video is appended to a partial JSONL (--resume).
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
import qta_cited_evidence as ce
import qta_frame_ops as fo

SEED = 20261023
RUNNER_PATH = Path(__file__).resolve()
(_BA, _BB), _NL = base.layer_band_from_env()
BANDS = {"band": tuple(range(_BA, _BB + 1))}


def self_margin(lp, pred):
    """Label-free margin: log-prob of the baseline-predicted option minus the best other option."""
    return lp[pred] - max(v for j, v in enumerate(lp) if j != pred)


GEN_PROMPT = ("{question}\nOptions:\n{options}\n"
              "Choose the best answer and give the time span of the video, in seconds, that supports it. "
              "Respond exactly in the format: Answer: <letter>. Time: <start>-<end>")


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
    p.add_argument("--attn", choices=("eager", "sdpa"), default="sdpa", help="the mask hooks build an explicit causal mask when sdpa passes None; sdpa is about 2.5x faster than eager, bf16 numerics differ slightly (baseline and arms always share the setting within a run)")
    p.add_argument("--citations", required=True, help="WP2 receipt (json or partial jsonl) with the citation per media hash")
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
    label_tensor = torch.tensor(label_ids, device="cuda")
    cohort.LAZY_FRAMES = True
    selected, counts, meta, cohort_hash = base._freeze_exact(
        repo, video_root=video_root, n=args.n, frame_count=args.frames, exclusion=exclusion_hashes,
        lock_assignments=None, dataset_split=args.dataset_split)

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

    from qwen_vl_utils import process_vision_info  # noqa: F401

    cit_path = Path(args.citations)
    cit_records = (json.loads(cit_path.read_text())["records"] if cit_path.suffix == ".json"
                   else [json.loads(l) for l in cit_path.read_text().splitlines() if l.strip()])
    cit_by_media = {r["media_sha256"]: r for r in cit_records}
    cit_sha256 = base.sha256_file(cit_path)

    def forward_frames(item, frames):
        item2 = dict(item)
        item2["frames"] = frames
        prepared = cohort.prepare_conditions([item2], processor, torch, args.frames, retain_batches=True, only_full=True)[0]
        batch = prepared["batches"]["full"]
        device_batch = batch.to("cuda")
        values = forward(device_batch)
        del prepared, batch, device_batch
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
            rec = {"media_sha256": media_hash, "qid_sha256": qid_hash}
            def exclude(reason):
                rec.update({"excluded": True, "exclusion_reason": reason, "arms": []})
                partial_file.write(json.dumps(rec) + "\n"); partial_file.flush(); os.fsync(partial_file.fileno())
                records.append(rec)
            src = cit_by_media.get(media_hash)
            if src is None or src.get("excluded"):
                exclude("no_usable_wp2_citation")
                continue
            cited, sham, ann = tuple(src["cited_group_indices"]), tuple(src["sham_group_indices"]), tuple(src["annotation_group_indices"])
            frames = [f.copy() for f in sample["frames"]]
            gold = int(sample["gold"])
            base_logp = forward_frames(sample, frames)
            forwards += 1
            pred = max(range(5), key=lambda j: (base_logp[j], -j))
            foil = base.baseline_best_foil(base_logp, gold)
            rec.update({"excluded": False, "gold": gold, "predicted": pred, "baseline_correct": pred == gold,
                        "baseline_choice_logprobs": base_logp, "baseline_self_margin": self_margin(base_logp, pred),
                        "baseline_gold_margin": base.fixed_gold_foil_margin(base_logp, gold_index=gold, baseline_foil_index=foil),
                        "wp2_baseline_max_abs_logprob_diff": max(abs(a - b) for a, b in zip(base_logp, src["baseline_choice_logprobs"])),
                        "cited_group_indices": list(cited), "sham_group_indices": list(sham), "annotation_group_indices": list(ann),
                        "cited_annotation_overlap": src.get("cited_annotation_overlap"), "arms": []})
            arm_frames = [
                ("replace_cited", lambda: fo.replace_groups(frames, cited, patch)),
                ("replace_sham", lambda: fo.replace_groups(frames, sham, patch)),
                ("keep_cited", lambda: fo.keep_only_groups(frames, cited, patch)),
                ("keep_sham", lambda: fo.keep_only_groups(frames, sham, patch)),
                ("replace_annotation", lambda: fo.replace_groups(frames, ann, patch)),
                ("keep_annotation", lambda: fo.keep_only_groups(frames, ann, patch)),
                ("allstatic", lambda: fo.all_static(frames)),
            ]
            for name, make in arm_frames:
                lp = forward_frames(sample, make())
                forwards += 1
                pb = float(torch.tensor(base_logp[pred]).exp())
                pk = float(torch.tensor(lp[pred]).exp())
                rec["arms"].append({"arm": name, "choice_logprobs": lp, "self_margin": self_margin(lp, pred),
                                    "gold_margin": base.fixed_gold_foil_margin(lp, gold_index=gold, baseline_foil_index=foil),
                                    "relative_change_percent": (pk - pb) / pb * 100.0, "predicted_after": max(range(5), key=lambda j: (lp[j], -j))})
            partial_file.write(json.dumps(rec) + "\n"); partial_file.flush(); os.fsync(partial_file.fileno())
            records.append(rec)
            del frames
            gc.collect(); torch.cuda.empty_cache()
    finally:
        partial_file.close()
    complete = len(records) >= (args.n if args.mode == "full" else min(3, args.n))
    payload = {
        "schema_version": 1, "purpose": "wp5_input_level_cited_interval", "status": "completed" if complete else "incomplete",
        "mode": args.mode, "dataset_split": args.dataset_split, "requested_videos": args.n, "recorded_videos": len(records),
        "excluded_videos": sum(bool(r.get("excluded")) for r in records),
        "baseline_correct": sum(bool(r.get("baseline_correct")) for r in records), "citations_receipt_sha256": cit_sha256,
        "frames": args.frames, "model": base.active_model()[0], "model_revision": base.active_model()[1],
        "model_config_sha256": model_config_sha256, "precision": "4-bit weights, BF16 compute, fp32 lm_head readout", "attn_implementation": args.attn, "seed": SEED,
        "cohort_sha256": cohort_hash, "exclusion_manifest_sha256": exclusion_sha256, "source_snapshot": source,
        "runner_sha256": base.sha256_file(RUNNER_PATH),
        "module_sha256": {n: base.sha256_file(RUNNER_PATH.parent / n) for n in ("qta_frame_ops.py", "qta_cited_evidence.py")},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
        "elapsed_seconds": time.monotonic() - started, "forwards": forwards, "rights_status": "unverified; local-only operation",
        "human_review_performed": False, "interpretation_limit": "input-level (frame replacement) dependence of the answer on the models own cited interval; not a general faithfulness result",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    base.write_private_json_exclusive(output, payload)
    print(json.dumps({"status": payload["status"], "records": len(records), "excluded": payload["excluded_videos"],
                      "baseline_correct": payload["baseline_correct"], "forwards": forwards, "output": output.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
