#!/usr/bin/env python3
"""Time the Qwen2.5-VL forward used by the runners (one video): baseline forwards and a hooked keep-crop forward, per attention implementation.

Usage: [PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True] bench_forward.py --attn eager|sdpa [--n-forwards 4]. Prints per-forward seconds and peak VRAM. Diagnostic only; no receipt is written.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nextgqa_annotation_key_bias_fp32_readout  # noqa: F401
import run_nextgqa_annotation_key_bias as base
import run_nextgqa_gold_interval_margin_pilot as cohort
import qta_attention_scores as sc
from qta_multiquery_video_key_intervention import install_multiquery_video_key_intervention_hooks, remove_hooks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attn", choices=("eager", "sdpa"), default="eager")
    ap.add_argument("--n-forwards", type=int, default=4)
    ap.add_argument("--dump", type=Path, default=None, help="save baseline and hooked logits (torch.save) for cross-implementation comparison")
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    D = repo / "data/private_nextgqa"
    root = (D / "annotation_key_bias_val_dev_candidates_20261023_01/NExTVideo").resolve()
    payload, _ = base.read_private_json(D / "known_media_sha_union_val_sgc_shamscreen_20261025_83.json", D.resolve())
    excl = cohort.validate_private_media_exclusion(payload)
    cohort.LAZY_FRAMES = True
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    os.environ["HF_HUB_OFFLINE"] = "1"
    model_path, _ = base._load_checkpoint_config(repo)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    t = time.monotonic()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=a.attn, device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    print(f"model load {time.monotonic() - t:.1f}s attn={a.attn} alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)
    sel, *_ = base._freeze_exact(repo, video_root=root, n=1, frame_count=32, exclusion=excl, lock_assignments=None, dataset_split="val")
    prepared = cohort.prepare_conditions(sel, processor, torch, 32, retain_batches=True, only_full=True)[0]
    batch = prepared["batches"]["full"].to("cuda")
    ids = batch["input_ids"][0].tolist()
    groups = base.video_group_key_positions(input_ids=ids, video_token_id=int(processor.video_token_id), video_grid_thw=batch["video_grid_thw"][0].tolist(),
                                            spatial_merge_size=int(processor.video_processor.merge_size), temporal_patch_size=int(processor.video_processor.temporal_patch_size))
    seq = len(ids)
    last_video = max(p for g in groups for p in g)
    rows = tuple(range(last_video + 1, seq))
    print(f"sequence length {seq}", flush=True)

    def fwd():
        torch.cuda.synchronize(); t0 = time.monotonic()
        with torch.inference_mode():
            out = model(**batch, use_cache=False, logits_to_keep=1)
        v = out.logits[0, -1].float().cpu()
        torch.cuda.synchronize()
        return time.monotonic() - t0, v

    torch.cuda.reset_peak_memory_stats()
    ref = None
    for i in range(a.n_forwards):
        dt, v = fwd()
        ref = v if ref is None else ref
        print(f"baseline forward {i}: {dt:.2f}s  peak alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB", flush=True)
    if os.environ.get("BENCH_DEBUG_MASK"):
        lay = model.model.language_model.layers[0].self_attn
        def dbg(_m, args, kwargs):
            m = kwargs.get("attention_mask")
            print("DEBUG mask:", None if m is None else (type(m).__name__, tuple(m.shape), m.dtype), "| kwargs keys:", sorted(kwargs.keys()), flush=True)
            if m is not None:
                print("DEBUG mask min/max/zeros:", float(m.min()), float(m.max()), int((m == 0).sum()), "attn impl:", getattr(model.config, "_attn_implementation", None), flush=True)
        h0 = lay.register_forward_pre_hook(dbg, with_kwargs=True)
        fwd(); h0.remove()
    dumped = {"baseline": ref}
    kept = sc.hash_random_groups(4, len(groups), "bench")
    biased = tuple(sorted(p for i, g in enumerate(groups) if i not in set(kept) for p in g))
    try:
        for i in range(2):
            handles = install_multiquery_video_key_intervention_hooks(model=model, layer_indices=tuple(range(len(model.model.language_model.layers))), biased_positions=biased,
                                                                      ablated_positions=(), query_positions=rows, logit_bias=-math.log(256.0), torch_module=torch)
            try:
                dt, v2 = fwd()
            finally:
                remove_hooks(handles)
            dumped["hooked"] = v2
            print(f"hooked crop forward {i}: {dt:.2f}s (max |logit diff vs baseline| {float((v2 - ref).abs().max()):.2f})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"hooked crop forward failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    if a.dump:
        torch.save(dumped, a.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
