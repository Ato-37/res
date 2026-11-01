#!/usr/bin/env python3
"""WP1-TVB: evidence-interval pathway knockout on Qwen2.5-VL-7B / TVBench Action Sequence (plan: experiments/2026-10-31_tvbench_evidence_knockout_plan.md).

Same design as WP1 (run_evidence_pathway_knockout.py; margin = gold minus frozen baseline foil; video->text and video->last edges restricted to the
keys of the annotated-interval temporal groups vs an equal-count disjoint hash sham, band layers 6-20 and all layers, plus a whole-video video->text
positive control), but the annotation is TVBench Action Sequence's own start/end window and the video is the FULL video (not cropped to the window).
Two options per question (A/B); the readout is restricted to the option labels and the foil is the other option.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, os, platform, sys, time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nextgqa_annotation_key_bias_fp32_readout  # noqa: F401
import run_nextgqa_annotation_key_bias as base
import run_nextgqa_gold_interval_margin_pilot as cohort
import qta_edge_knockout as ko
import qta_evidence_edge_knockout as ev
import run_tvbench_pathway as tvp
from run_evidence_pathway_knockout import arms_for, BANDS, SEED

RUNNER_PATH = Path(__file__).resolve()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--tvbench-root", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--skip", type=int, default=0, help="skip the first N selected videos (disjoint second batch)")
    p.add_argument("--frames", type=int, choices=(16, 32), default=32)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--attn", choices=("eager", "sdpa"), default="sdpa")
    p.add_argument("--user-authorized-local-only", action="store_true")
    return p.parse_args(argv)


def build_cohort(tv_root: Path, n: int, frames: int, skip: int = 0):
    """skip>0 drops the first `skip` selected videos (a disjoint second batch; never pooled with the first batch)."""
    data = json.loads((tv_root / "json" / "action_sequence.json").read_text(encoding="utf-8"))
    ranked = sorted(range(len(data)), key=lambda i: hashlib.sha256(f"tvb-ev-v1|{data[i]['video']}|{i}".encode()).hexdigest())
    selected, seen, rejected = [], set(), 0
    for i in ranked:
        q = data[i]
        path = tv_root / "video" / "action_sequence" / q["video"]
        if q["video"] in seen or not path.is_file() or q["answer"] not in q["candidates"] or len(q["candidates"]) != 2:
            continue
        seen.add(q["video"])
        try:
            fr, decode = cohort.decode_sampled_video(path, num_frames=frames, uniform_tolerance_seconds=cohort.UNIFORM_TOLERANCE)
            cohort.verify_sampled_frame_pixels(fr, decode)
        except Exception:
            rejected += 1
            continue
        pts = [Fraction(*r["relative_presentation_time"]) for r in decode["sampled_frames"]]
        s, e = float(q["start"]), float(q["end"])
        ev_pos = [j for j, t in enumerate(pts) if s <= float(t) <= e]
        selected.append({"qid": f"action_sequence:{i}", "question": q["question"], "choices": list(q["candidates"]),
                         "gold": q["candidates"].index(q["answer"]), "frames": cohort.LazyFrames(path, frames, decode), "decode": decode,
                         "sampled_times": pts, "evidence_positions": ev_pos, "control_positions": [], "task": "action_sequence"})
        if len(selected) >= n + skip:
            break
    selected = selected[skip:]
    if len(selected) != n:
        raise ValueError(f"requested_{n}_but_cohort_has_{len(selected)}")
    fp = hashlib.sha256(json.dumps([[x["qid"], x["decode"]["source_sha256"]] for x in selected]).encode()).hexdigest()
    return selected, fp, {"decode_rejected": rejected}


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
    cohort.LAZY_FRAMES = True
    selected, cohort_hash, counts = build_cohort(Path(args.tvbench_root).resolve(strict=True), args.n, args.frames, args.skip)
    cohort.build_qa_prompt = tvp.build_qa_prompt_k

    import torch
    from transformers import AutoProcessor
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=args.attn, device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    label_tensor = torch.tensor(label_ids, device="cuda")

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
        values = torch.log_softmax(logits, dim=-1)[label_tensor].detach().cpu().tolist()
        del out, logits
        return values

    done, records = set(), []
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                r = json.loads(line); records.append(r); done.add(r["media_sha256"])
    pf = os.fdopen(os.open(partial, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a")
    rows = range(args.n) if args.mode == "full" else range(min(3, args.n))
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    t0, forwards = time.monotonic(), 0
    try:
        for index in rows:
            s = selected[index]; mh = s["decode"]["source_sha256"]
            if mh in done:
                continue
            qh = hashlib.sha256(str(s["qid"]).encode()).hexdigest()
            prepared = cohort.prepare_conditions([s], processor, torch, args.frames, retain_batches=True, only_full=True)[0]
            batch = prepared["batches"]["full"]; dev = batch.to("cuda")
            ids = batch["input_ids"][0].tolist(); seq_len = len(ids); geometry = batch["video_grid_thw"][0].tolist()
            groups = base.video_group_key_positions(input_ids=ids, video_token_id=int(processor.video_token_id), video_grid_thw=geometry,
                                                    spatial_merge_size=int(processor.video_processor.merge_size), temporal_patch_size=patch)
            rec = {"media_sha256": mh, "qid_sha256": qh, "video_grid_thw": geometry, "sequence_length": seq_len,
                   "interval_seconds": [float(s["decode"].get("duration_seconds", 0) or 0), 0.0]}
            try:
                ann = base.map_annotation_frames_to_groups(evidence_frame_indices=s["evidence_positions"], frame_count=args.frames,
                                                           temporal_patch_size=patch, temporal_group_count=len(groups))
                sham = base.choose_disjoint_hash_sham(temporal_group_count=len(groups), annotation_groups=ann,
                                                      opaque_row_token=base.row_token(mh, qh), protocol_seed="tvbench-annotation-key-sham-v1")
            except ValueError as exc:
                rec.update({"excluded": True, "exclusion_reason": str(exc)[:80], "knockouts": []})
                pf.write(json.dumps(rec) + "\n"); pf.flush(); os.fsync(pf.fileno()); records.append(rec)
                del prepared, batch, dev; gc.collect(); torch.cuda.empty_cache(); continue
            gold = int(s["gold"]); foil = 1 - gold
            blp = forward(dev); forwards += 1
            pred = max(range(2), key=lambda j: (blp[j], -j))
            rec.update({"excluded": False, "gold": gold, "predicted": pred, "baseline_correct": pred == gold, "baseline_best_foil": foil,
                        "baseline_choice_logprobs": blp, "baseline_margin": float(blp[gold]) - float(blp[foil]),
                        "annotation_group_indices": list(ann), "sham_group_indices": list(sham), "temporal_group_count": len(groups), "knockouts": []})
            for name, edge, kg, band in arms_for(len(groups), ann, sham):
                blocks = (ev.all_video_blocks(edge, video_groups=groups, seq_len=seq_len) if kg is None
                          else ev.evidence_blocks(edge, video_groups=groups, seq_len=seq_len, key_groups=kg))
                layers = BANDS[band]
                handles, state = ko.install_knockout_hooks(model=model, layers_1based=layers, blocks=blocks, torch_module=torch)
                try:
                    klp = forward(dev)
                finally:
                    ko.remove_hooks(handles)
                forwards += 1
                if state["calls"] != {int(i): 1 for i in layers}:
                    raise ValueError("knockout_layer_calls_mismatch")
                pb = float(torch.tensor(blp[pred]).exp()); pk = float(torch.tensor(klp[pred]).exp())
                rec["knockouts"].append({"arm": name, "edge": edge, "band": band, "cells": state["expected_cells"], "choice_logprobs": klp,
                                         "margin": float(klp[gold]) - float(klp[foil]), "relative_change_percent": (pk - pb) / pb * 100.0,
                                         "predicted_after": max(range(2), key=lambda j: (klp[j], -j))})
            pf.write(json.dumps(rec) + "\n"); pf.flush(); os.fsync(pf.fileno()); records.append(rec)
            del prepared, batch, dev; gc.collect(); torch.cuda.empty_cache()
    finally:
        pf.close()
    complete = len(records) >= (args.n if args.mode == "full" else min(3, args.n))
    payload = {"schema_version": 1, "purpose": "evidence_interval_pathway_knockout_tvbench_action_sequence", "status": "completed" if complete else "incomplete",
               "mode": args.mode, "requested_videos": args.n, "recorded_videos": len(records), "excluded_videos": sum(bool(r.get("excluded")) for r in records),
               "baseline_correct": sum(bool(r.get("baseline_correct")) for r in records), "bands": {k: [v[0], v[-1]] for k, v in BANDS.items()},
               "frames": args.frames, "model": base.active_model()[0], "model_revision": base.active_model()[1], "model_config_sha256": model_config_sha256,
               "attn_implementation": args.attn, "seed": SEED, "cohort_sha256": cohort_hash, "cohort_counts": counts, "source_snapshot": source,
               "runner_sha256": base.sha256_file(RUNNER_PATH), "runtime": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
               "elapsed_seconds": time.monotonic() - t0, "forwards": forwards, "rights_status": "unverified; local-only operation", "human_review_performed": False,
               "interpretation_limit": "pathway-level dependence on the TVBench Action Sequence annotated window; not explanation faithfulness",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    base.write_private_json_exclusive(output, payload)
    print(json.dumps({"status": payload["status"], "records": len(records), "excluded": payload["excluded_videos"], "baseline_correct": payload["baseline_correct"],
                      "forwards": forwards, "output": output.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
