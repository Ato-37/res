#!/usr/bin/env python3
"""Logit lens v2 (plan: experiments/2026-10-31_logit_lens_v2_and_fine_pathway_plan.md): raw lens vs a LINEAR (least-squares) lens, with less floor-limited metrics.

Why: the raw lens (final norm + lm_head on intermediate residuals) showed no mid-layer keyword emergence on Qwen2.5-VL (top-1 keyword rates ~0 until layer 23).
That cannot distinguish "no temporal concepts" from "the probe is bad for intermediate layers". v2 (a) fits, per layer, a ridge-regression translator from the
layer-l normalised video-token state to the final-layer normalised state on TRAIN videos (disjoint from the TEST videos), (b) reports probe fidelity KL(final || lens_l),
(c) reports keyword probability MASS (soft, not top-1) and top-1 rate for both lenses, (d) reports the answer margin (gold-foil option log-prob) read from the last token at every layer.
Test videos = the first 100 of the tvb-v1 Action Sequence ranking (same as the raw-lens run); train videos = ranks 101..(100+--train). Only aggregates are stored.
"""
from __future__ import annotations

import argparse, gc, json, os, platform, sys, time
import torch.nn.functional as F
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nextgqa_annotation_key_bias_fp32_readout  # noqa: F401
import run_nextgqa_annotation_key_bias as base
import run_nextgqa_gold_interval_margin_pilot as cohort
import qta_edge_knockout as ko
import run_tvbench_paper_protocol as pp
from run_tvbench_logit_lens import SPATIAL, TEMPORAL
CHUNK = 64

SEED = 20261031
RUNNER_PATH = Path(__file__).resolve()
RIDGE = 1e-2  # relative to mean diagonal of X^T X


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--tvbench-root", required=True)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--train", type=int, default=40)
    p.add_argument("--frames", type=int, choices=(8, 16, 32), default=8)
    p.add_argument("--output", required=True)
    p.add_argument("--attn", choices=("eager", "sdpa"), default="sdpa")
    p.add_argument("--user-authorized-local-only", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.user_authorized_local_only:
        raise SystemExit("explicit_user_authorized_local_only_required")
    repo = RUNNER_PATH.parents[1]
    output = base.private_output_path(args.output, repo)
    if output.exists():
        raise SystemExit("output_exists")
    source = base.source_snapshot(repo)
    model_path, model_config_sha256 = base._load_checkpoint_config(repo)
    n_test, n_train = (3, 3) if args.mode == "smoke" else (args.n_test, args.train)
    selected, cohort_hash, cohort_counts = pp.build_tvbench_cohort(Path(args.tvbench_root).resolve(strict=True), args.n_test + args.train, args.n_test + args.train, args.frames)
    if {s["task"] for s in selected} != {"action_sequence"}:
        raise ValueError("cohort_not_action_sequence_only")
    test, train = selected[:n_test], selected[args.n_test:args.n_test + n_train]
    cohort.build_qa_prompt = pp.build_qa_prompt_paper

    import torch
    torch.set_grad_enabled(False)  # fix: norm.weight requires grad, so norm(hs) built an autograd graph that XtX[...] += (...).cpu() kept alive (+0.6 GB of CUDA memory per train video -> OOM)
    from transformers import AutoProcessor
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    tok = processor.tokenizer
    patch = int(processor.video_processor.temporal_patch_size)
    if torch.cuda.mem_get_info()[0] < base.MIN_FREE_VRAM or cohort.host_available_bytes() < base.MIN_HOST_AVAILABLE:
        raise RuntimeError("pre_model_resource_guard_failed")
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
        device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    L = len(model.model.language_model.layers); norm = model.model.language_model.norm; V = model.lm_head.out_features
    sets = {"spatial": SPATIAL, "temporal": TEMPORAL}
    kw_mask = {}; word_of_token = torch.full((V,), -1, dtype=torch.long, device="cuda"); words = [(k, w) for k, ws in sets.items() for w in ws]
    for kind in sets: kw_mask[kind] = torch.zeros(V, dtype=torch.bool, device="cuda")
    for wi, (kind, w) in enumerate(words):
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            ids = tok(variant, add_special_tokens=False).input_ids
            if ids:
                word_of_token[ids[0]] = wi; kw_mask[kind][ids[0]] = True
    label_tensor = torch.tensor([tok(l, add_special_tokens=False).input_ids[0] for l in cohort.LABELS], device="cuda")
    D = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size

    def prep(sample):
        prepared = cohort.prepare_conditions([sample], processor, torch, args.frames, retain_batches=True, only_full=True)[0]
        batch = prepared["batches"]["full"]; dev = batch.to("cuda")
        ids = batch["input_ids"][0].tolist(); seq_len = len(ids)
        groups = base.video_group_key_positions(input_ids=ids, video_token_id=int(processor.video_token_id), video_grid_thw=batch["video_grid_thw"][0].tolist(),
                                                spatial_merge_size=int(processor.video_processor.merge_size), temporal_patch_size=patch)
        return prepared, batch, dev, groups, seq_len, torch.tensor(sorted(p for g in groups for p in g), device="cuda")

    def states(dev):
        with torch.inference_mode():
            out = model(**dev, use_cache=False, logits_to_keep=1, output_hidden_states=True)
        return out

    # ---- phase 1: fit per-layer ridge translators (normalised layer-l state -> final normalised state) on TRAIN videos
    XtX = [torch.zeros(D + 1, D + 1, dtype=torch.float32) for _ in range(L - 1)]; XtY = [torch.zeros(D + 1, D, dtype=torch.float32) for _ in range(L - 1)]
    ntok = 0; t0 = time.monotonic(); forwards = 0
    for s in train:
        prepared, batch, dev, groups, seq_len, vid = prep(s)
        out = states(dev); forwards += 1; hs = out.hidden_states
        Y = hs[L][0, vid].float()
        for li in range(1, L):
            X = norm(hs[li][0, vid]).float(); X1 = torch.cat([X, torch.ones(X.shape[0], 1, device="cuda")], 1)
            XtX[li - 1] += (X1.T @ X1).cpu(); XtY[li - 1] += (X1.T @ Y).cpu()
        ntok += int(vid.numel()); del out, hs, prepared, batch, dev, X, X1, Y; gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        print(json.dumps({"train_video": forwards, "cuda_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2), "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
                          "host_available_gb": round(cohort.host_available_bytes() / 2**30, 1)}), flush=True)
    W = []
    for li in range(L - 1):
        A = XtX[li].cuda(); lam = RIDGE * float(A.diagonal().mean()); A += lam * torch.eye(D + 1, device="cuda")
        W.append(torch.linalg.solve(A, XtY[li].cuda()).half().cpu()); del A
    del XtX, XtY; torch.cuda.empty_cache()
    Wg = [w.cuda() for w in W]  # fp16 translators, 27 x ~25 MB

    def lens_logits(h, li, tuned):
        """h: [n, D] raw residual of layer li (1-based). Returns fp32 logits [n, V]."""
        # bf16 lm_head matmul with fp32 output: avoids the 2.2 GB fp32 copy of the unembedding that the fp32-readout wrapper makes on every call (caused a CUDA OOM)
        if li == L:
            return F.linear(h, model.lm_head.weight).float()
        x = norm(h).float()
        if tuned:
            x = torch.cat([x, torch.ones(x.shape[0], 1, device="cuda")], 1) @ Wg[li - 1].float()
        return F.linear(x.to(h.dtype), model.lm_head.weight).float()

    def measure(dev, vid, gold, k):
        out = states(dev); hs = out.hidden_states
        final_lp_last = torch.log_softmax(out.logits[0, -1].float(), -1)[label_tensor]
        rec = {"layers": []}
        for li in range(1, L + 1):
            row = {}
            for tuned in ((False, True) if li < L else (False,)):
                name = "tuned" if tuned else "raw"
                h = hs[li][0, vid]
                top1 = torch.zeros(len(words), dtype=torch.long, device="cuda"); mass = {kk: 0.0 for kk in sets}; kl = 0.0; top5 = {kk: 0 for kk in sets}
                for s0 in range(0, h.shape[0], CHUNK):
                    lg = lens_logits(h[s0:s0 + CHUNK], li, tuned)
                    lp = torch.log_softmax(lg, -1); p = lp.exp()
                    fin = torch.log_softmax(lens_logits(hs[L][0, vid][s0:s0 + CHUNK], L, False), -1)
                    kl += float((fin.exp() * (fin - lp)).sum())
                    w = word_of_token[lg.argmax(-1)]; w = w[w >= 0]
                    if w.numel(): top1 += torch.bincount(w, minlength=len(words))
                    t5 = lg.topk(5, -1).indices
                    for kk in sets:
                        mass[kk] += float(p[:, kw_mask[kk]].sum()); top5[kk] += int(kw_mask[kk][t5].any(-1).sum())
                lt = lens_logits(hs[li][0, -1:], li, tuned)
                lpl = torch.log_softmax(lt, -1)[0][label_tensor].cpu().tolist()
                row[name] = {"top1_counts": top1.cpu().tolist(), "mass_sum": mass, "top5_hits": top5, "kl_sum": kl,
                             "answer_margin": lpl[gold] - lpl[1 - gold] if k == 2 else None}
            rec["layers"].append(row)
        rec["final_answer_margin"] = float(final_lp_last[gold] - final_lp_last[1 - gold]) if k == 2 else None
        del out, hs
        return rec

    records = []
    for s in test:
        prepared, batch, dev, groups, seq_len, vid = prep(s)
        gold, k = int(s["gold"]), len(s["choices"]); rec = {"media_sha256": s["decode"]["source_sha256"], "gold": gold, "n_video_tokens": int(vid.numel())}
        rec["baseline"] = measure(dev, vid, gold, k); forwards += 1
        blocks = ko.edge_blocks("cross_frame", video_groups=groups, seq_len=seq_len)
        handles, st = ko.install_knockout_hooks(model=model, layers_1based=tuple(range(1, L + 1)), blocks=blocks, torch_module=torch)
        try:
            rec["crossframe_blocked"] = measure(dev, vid, gold, k); forwards += 1
        finally:
            ko.remove_hooks(handles)
        records.append(rec); del prepared, batch, dev; gc.collect(); torch.cuda.empty_cache()
    payload = {"schema_version": 1, "purpose": "logit_lens_v2_linear_lens_tvbench_action_sequence", "mode": args.mode, "n_test": len(test), "n_train": len(train),
               "train_video_tokens": ntok, "ridge_relative": RIDGE, "words": [{"kind": kd, "word": w} for kd, w in words], "layer_count": L, "frames": args.frames,
               "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": base.MODEL_REVISION, "model_config_sha256": model_config_sha256,
               "precision": "4-bit weights, BF16 compute, fp32 lens readout; fp16 translators", "seed": SEED, "cohort_sha256": cohort_hash, "source_snapshot": source,
               "runner_sha256": base.sha256_file(RUNNER_PATH), "runtime": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
               "elapsed_seconds": time.monotonic() - t0, "forwards": forwards, "rights_status": "unverified; local-only operation", "human_review_performed": False,
               "interpretation_limit": "probe of representations; train/test videos disjoint; not explanation faithfulness",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    base.write_private_json_exclusive(output, payload)
    print(json.dumps({"status": "completed", "test": len(test), "train": len(train), "train_tokens": ntok, "forwards": forwards, "output": output.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
