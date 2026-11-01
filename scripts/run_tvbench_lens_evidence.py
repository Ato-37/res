#!/usr/bin/env python3
"""WP4-TVB: Logit Lens (linear translator) on evidence vs non-evidence video tokens (plan: experiments/2026-10-31_tvbench_lens_evidence_plan.md).
Per test video: one forward with hidden states; for each layer the lens probability mass of the gold-only / foil-only option words is averaged per temporal group.
Only aggregates are stored (local-only receipt)."""
from __future__ import annotations

import argparse, gc, hashlib, json, os, platform, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nextgqa_annotation_key_bias_fp32_readout  # noqa: F401
import run_nextgqa_annotation_key_bias as base
import run_nextgqa_gold_interval_margin_pilot as cohort
import run_tvbench_pathway as tvp
from run_tvbench_window_expansion import build_cohort

SEED = 20261031
RUNNER_PATH = Path(__file__).resolve()
RIDGE = 1e-2
CHUNK = 256
STOP = set("the a an to of and then after before on in at with from his her their its up down off out over into onto".split()) - {"up", "down", "on", "off", "out", "over"}


def words_of(text):
    return [w for w in re.findall(r"[A-Za-z]+", text.lower()) if len(w) >= 3 and w not in {"the", "and", "then", "after", "before", "with", "from", "his", "her", "their", "its", "into", "onto"}]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--tvbench-root", required=True)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--n-train", type=int, default=40)
    p.add_argument("--frames", type=int, choices=(16, 32), default=32)
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
    n_test, n_train = (3, 3) if args.mode == "smoke" else (args.n_test, args.n_train)
    cohort.LAZY_FRAMES = True
    tv = Path(args.tvbench_root).resolve(strict=True)
    test, test_hash, test_counts = build_cohort(tv, n_test, args.frames, 0, "action_sequence")
    train, train_hash, _ = build_cohort(tv, n_train, args.frames, 200, "action_sequence")
    cohort.build_qa_prompt = tvp.build_qa_prompt_k

    import torch
    import torch.nn.functional as F
    torch.set_grad_enabled(False)
    from transformers import AutoProcessor
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    tok = processor.tokenizer
    patch = int(processor.video_processor.temporal_patch_size)
    if torch.cuda.mem_get_info()[0] < base.MIN_FREE_VRAM or cohort.host_available_bytes() < base.MIN_HOST_AVAILABLE:
        raise RuntimeError("pre_model_resource_guard_failed")
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=args.attn, device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    L = len(model.model.language_model.layers); norm = model.model.language_model.norm; V = model.lm_head.out_features
    D = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    label_tensor = torch.tensor([tok(l, add_special_tokens=False).input_ids[0] for l in cohort.LABELS], device="cuda")
    W_OUT = model.lm_head.weight

    def ids_of(words):
        s = set()
        for w in words:
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                i = tok(v, add_special_tokens=False).input_ids
                if i:
                    s.add(i[0])
        return s

    def prep(sample):
        prepared = cohort.prepare_conditions([sample], processor, torch, args.frames, retain_batches=True, only_full=True)[0]
        batch = prepared["batches"]["full"]; dev = batch.to("cuda")
        ids = batch["input_ids"][0].tolist()
        groups = base.video_group_key_positions(input_ids=ids, video_token_id=int(processor.video_token_id), video_grid_thw=batch["video_grid_thw"][0].tolist(),
                                                spatial_merge_size=int(processor.video_processor.merge_size), temporal_patch_size=patch)
        return prepared, batch, dev, groups

    def states(dev):
        return model(**dev, use_cache=False, logits_to_keep=1, output_hidden_states=True)

    t0 = time.monotonic(); forwards = 0
    # ---- phase 1: ridge translators on the TRAIN videos
    XtX = [torch.zeros(D + 1, D + 1) for _ in range(L - 1)]; XtY = [torch.zeros(D + 1, D) for _ in range(L - 1)]; ntok = 0
    for s in train:
        prepared, batch, dev, groups = prep(s)
        out = states(dev); forwards += 1; hs = out.hidden_states
        vid = torch.tensor(sorted(p for g in groups for p in g), device="cuda")
        Y = hs[L][0, vid].float()
        for li in range(1, L):
            X = norm(hs[li][0, vid]).float(); X1 = torch.cat([X, torch.ones(X.shape[0], 1, device="cuda")], 1)
            XtX[li - 1] += (X1.T @ X1).cpu(); XtY[li - 1] += (X1.T @ Y).cpu()
        ntok += int(vid.numel()); del out, hs, prepared, batch, dev, X, X1, Y; gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        print(json.dumps({"train_video": forwards, "cuda_gb": round(torch.cuda.memory_allocated() / 2**30, 2)}), flush=True)
    Wg = []
    for li in range(L - 1):
        A = XtX[li].cuda(); lam = RIDGE * float(A.diagonal().mean()); A += lam * torch.eye(D + 1, device="cuda")
        Wg.append(torch.linalg.solve(A, XtY[li].cuda()).half()); del A
    del XtX, XtY; torch.cuda.empty_cache()

    def lens_logits(h, li):
        if li == L:
            return F.linear(h, W_OUT).float()
        x = norm(h).float()
        x = torch.cat([x, torch.ones(x.shape[0], 1, device="cuda")], 1) @ Wg[li - 1].float()
        return F.linear(x.to(h.dtype), W_OUT).float()

    records = []
    for s in test:
        mh = s["decode"]["source_sha256"]; rec = {"media_sha256": mh}
        prepared, batch, dev, groups = prep(s)
        G = len(groups)
        try:
            ann = base.map_annotation_frames_to_groups(evidence_frame_indices=s["evidence_positions"], frame_count=args.frames, temporal_patch_size=patch, temporal_group_count=G)
        except ValueError as exc:
            rec.update({"excluded": True, "exclusion_reason": str(exc)[:80]}); records.append(rec); del prepared, batch, dev; gc.collect(); continue
        gold = int(s["gold"]); foil = 1 - gold
        wg, wf = set(words_of(s["choices"][gold])), set(words_of(s["choices"][foil]))
        gids, fids = ids_of(wg - wf), ids_of(wf - wg)
        gids -= fids; fids -= (gids & fids)
        if not gids or not fids:
            rec.update({"excluded": True, "exclusion_reason": "empty_word_set"}); records.append(rec); del prepared, batch, dev; gc.collect(); continue
        gt = torch.tensor(sorted(gids), device="cuda"); ft = torch.tensor(sorted(fids), device="cuda")
        out = states(dev); forwards += 1; hs = out.hidden_states
        fl = torch.log_softmax(out.logits[0, -1].float(), -1)[label_tensor[:2]]
        pred = int(fl.argmax())
        vid_pos = [p for g in groups for p in g]; vid = torch.tensor(vid_pos, device="cuda")
        grp = torch.tensor([gi for gi, g in enumerate(groups) for _ in g], device="cuda")
        cnt = torch.bincount(grp, minlength=G).float()
        dG = [[0.0] * G for _ in range(L)]; dF = [[0.0] * G for _ in range(L)]; last_margin = []
        for li in range(1, L + 1):
            sg = torch.zeros(G, device="cuda"); sf = torch.zeros(G, device="cuda")
            h_all = hs[li][0, vid]
            for c0 in range(0, h_all.shape[0], CHUNK):
                lg = lens_logits(h_all[c0:c0 + CHUNK], li)
                lse = torch.logsumexp(lg, -1)
                lG = torch.logsumexp(lg[:, gt], -1) - lse; lF = torch.logsumexp(lg[:, ft], -1) - lse
                gi = grp[c0:c0 + CHUNK]
                sg.index_add_(0, gi, lG); sf.index_add_(0, gi, lF)
            dG[li - 1] = (sg / cnt).cpu().tolist(); dF[li - 1] = (sf / cnt).cpu().tolist()
            lt = torch.log_softmax(lens_logits(hs[li][0, -1:], li)[0], -1)[label_tensor[:2]].cpu().tolist()
            last_margin.append(lt[gold] - lt[foil])
        rec.update({"excluded": False, "gold": gold, "predicted": pred, "baseline_correct": pred == gold, "annotation_group_indices": list(ann), "temporal_group_count": G,
                    "group_tokens": cnt.cpu().tolist(), "log_mass_gold": dG, "log_mass_foil": dF, "last_token_margin": last_margin,
                    "n_gold_ids": len(gids), "n_foil_ids": len(fids)})
        records.append(rec); del out, hs, prepared, batch, dev; gc.collect(); torch.cuda.empty_cache()
    payload = {"schema_version": 1, "purpose": "logit_lens_evidence_tokens_tvbench_action_sequence", "mode": args.mode, "n_test": len(test), "n_train": len(train), "train_video_tokens": ntok,
               "frames": args.frames, "layer_count": L, "ridge_relative": RIDGE, "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": base.MODEL_REVISION,
               "model_config_sha256": model_config_sha256, "precision": "4-bit weights, BF16 compute, bf16 lens readout", "seed": SEED, "test_cohort_sha256": test_hash, "train_cohort_sha256": train_hash,
               "source_snapshot": source, "runner_sha256": base.sha256_file(RUNNER_PATH), "runtime": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
               "elapsed_seconds": time.monotonic() - t0, "forwards": forwards, "rights_status": "unverified; local-only operation", "human_review_performed": False,
               "interpretation_limit": "representation-level reading of evidence vs other video tokens; not causal, not explanation faithfulness",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    base.write_private_json_exclusive(output, payload)
    print(json.dumps({"status": "completed", "test": len(test), "analysed": sum(1 for r in records if not r.get("excluded")), "train_tokens": ntok, "forwards": forwards, "output": output.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
