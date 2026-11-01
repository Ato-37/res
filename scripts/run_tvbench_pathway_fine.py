#!/usr/bin/env python3
"""FINE: finer internal-pathway analysis on Qwen2.5-VL-7B / TVBench (plan: experiments/2026-10-31_logit_lens_v2_and_fine_pathway_plan.md).

Per video: baseline, then (a) single-layer knockouts for every layer 1..L of the cross-frame edge and of the video->text edge (all video keys),
(b) single-temporal-group video->text knockouts (key group g, layers 6-20) for every group g (temporal profile; tests a position effect of the earliest groups).
Aggregated by scripts/analyze_fine_pathway.py. Local-only receipts.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, os, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nextgqa_annotation_key_bias_fp32_readout  # noqa: F401
import run_nextgqa_annotation_key_bias as base
import run_nextgqa_gold_interval_margin_pilot as cohort
import qta_edge_knockout as ko
import qta_evidence_edge_knockout as ev
import run_tvbench_pathway as tvp
from run_evidence_pathway_knockout import BANDS, SEED
from run_tvbench_window_expansion import build_cohort

RUNNER_PATH = Path(__file__).resolve()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--tvbench-root", required=True)
    p.add_argument("--task", choices=("action_sequence", "action_localization"), default="action_sequence")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--skip", type=int, default=0, help="skip the first N selected videos (disjoint second batch)")
    p.add_argument("--frames", type=int, choices=(16, 32), default=32)
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
    cohort.LAZY_FRAMES = True
    selected, cohort_hash, counts = build_cohort(Path(args.tvbench_root).resolve(strict=True), args.n, args.frames, args.skip, args.task)
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

    def forward(device_batch, k=2):
        if torch.cuda.mem_get_info()[0] < 2 * base.MIN_FORWARD_VRAM:
            torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < base.MIN_FORWARD_VRAM or cohort.host_available_bytes() < base.MIN_FORWARD_HOST:
            raise RuntimeError("per_forward_resource_guard_failed")
        with torch.inference_mode():
            out = model(**device_batch, use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].float()
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("nonfinite_answer_logits")
        values = torch.log_softmax(logits, dim=-1)[label_tensor[:k]].detach().cpu().tolist()
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
    n_layers = BANDS["all"][-1]
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
            rec = {"media_sha256": mh, "qid_sha256": qh, "video_grid_thw": geometry, "sequence_length": seq_len}
            try:
                ann = base.map_annotation_frames_to_groups(evidence_frame_indices=s["evidence_positions"], frame_count=args.frames,
                                                           temporal_patch_size=patch, temporal_group_count=len(groups))
            except ValueError as exc:
                rec.update({"excluded": True, "exclusion_reason": str(exc)[:80], "knockouts": []})
                pf.write(json.dumps(rec) + "\n"); pf.flush(); os.fsync(pf.fileno()); records.append(rec)
                del prepared, batch, dev; gc.collect(); torch.cuda.empty_cache(); continue
            G = len(groups); gold = int(s["gold"]); kopt = len(s["choices"])
            blp = forward(dev, kopt); forwards += 1
            pred = max(range(kopt), key=lambda j: (blp[j], -j))
            foil = max((j for j in range(kopt) if j != gold), key=lambda j: (blp[j], -j))
            rec.update({"excluded": False, "gold": gold, "predicted": pred, "baseline_correct": pred == gold, "baseline_best_foil": foil,
                        "baseline_choice_logprobs": blp, "baseline_margin": float(blp[gold]) - float(blp[foil]),
                        "annotation_group_indices": list(ann), "temporal_group_count": G, "knockouts": []})
            vid_all = tuple(range(G))
            arms = []
            for l in range(1, n_layers + 1):
                arms.append((f"cf_L{l}", "cross_frame", (l,), None))
                arms.append((f"v2t_L{l}", "video_to_text", (l,), vid_all))
            for g in range(G):
                arms.append((f"v2t_band_g{g}", "video_to_text", BANDS["band"], (g,)))
            for name, edge, layers, kg in arms:
                blocks = (ko.edge_blocks("cross_frame", video_groups=groups, seq_len=seq_len) if edge == "cross_frame"
                          else ev.evidence_blocks(edge, video_groups=groups, seq_len=seq_len, key_groups=kg))
                handles, state = ko.install_knockout_hooks(model=model, layers_1based=layers, blocks=blocks, torch_module=torch)
                try:
                    klp = forward(dev, kopt)
                finally:
                    ko.remove_hooks(handles)
                forwards += 1
                if state["calls"] != {int(i): 1 for i in layers}:
                    raise ValueError("knockout_layer_calls_mismatch")
                pb = float(torch.tensor(blp[pred]).exp()); pk = float(torch.tensor(klp[pred]).exp())
                rec["knockouts"].append({"arm": name, "edge": edge, "layers": list(layers), "key_groups": list(kg) if kg else None,
                                         "overlap": len(set(kg or ()) & set(ann)), "cells": state["expected_cells"], "choice_logprobs": klp,
                                         "margin": float(klp[gold]) - float(klp[foil]), "relative_change_percent": (pk - pb) / pb * 100.0})
            pf.write(json.dumps(rec) + "\n"); pf.flush(); os.fsync(pf.fileno()); records.append(rec)
            del prepared, batch, dev; gc.collect(); torch.cuda.empty_cache()
    finally:
        pf.close()
    complete = len(records) >= (args.n if args.mode == "full" else min(3, args.n))
    payload = {"schema_version": 1, "purpose": f"fine_pathway_tvbench_{args.task}", "status": "completed" if complete else "incomplete",
               "mode": args.mode, "requested_videos": args.n, "recorded_videos": len(records), "excluded_videos": sum(bool(r.get("excluded")) for r in records),
               "baseline_correct": sum(bool(r.get("baseline_correct")) for r in records), "arms": "single-layer cross_frame and video_to_text sweeps (layers 1..L) and single temporal-group video_to_text knockouts in the band layers", "frames": args.frames, "model": base.active_model()[0], "model_revision": base.active_model()[1],
               "model_config_sha256": model_config_sha256, "attn_implementation": args.attn, "seed": SEED, "cohort_sha256": cohort_hash,
               "cohort_counts": counts, "source_snapshot": source, "runner_sha256": base.sha256_file(RUNNER_PATH),
               "runtime": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
               "elapsed_seconds": time.monotonic() - t0, "forwards": forwards, "rights_status": "unverified; local-only operation", "human_review_performed": False,
               "interpretation_limit": "dependence on the annotated window vs amount blocked; not explanation faithfulness",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    base.write_private_json_exclusive(output, payload)
    print(json.dumps({"status": payload["status"], "records": len(records), "excluded": payload["excluded_videos"],
                      "baseline_correct": payload["baseline_correct"], "forwards": forwards, "output": output.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
