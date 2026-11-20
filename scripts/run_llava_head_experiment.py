#!/usr/bin/env python3
"""LLaVA-NeXT-Video head-level 'Video -/-> Question' knockout (plan: experiments/2026-11-03_llava_head_plan.md). Phases: sweep -> scan -> group.
Run with ~/external/mtf-venv/bin/python scripts/run_llava_head_experiment.py --mode smoke|full --out-prefix <new path prefix> --user-authorized-local-only
Per-head blocking: additive mask [1,H,n,n] = causal everywhere + (-inf) on (question query x video key) for the chosen heads, installed on layer.self_attn.forward with the official wrapper pattern.
Endpoint: relative change (%) of the gold-option-letter probability. Only derived numbers stored."""
import argparse, functools, hashlib, json, os, platform, random, sys, time
from datetime import datetime, timezone
from pathlib import Path

OFFICIAL = Path.home() / "external" / "map-the-flow-fast"
MODEL = Path.home() / "external" / "models" / "LLaVA-NeXT-7B-Video-FT"
G_SIZE, N_LAYERS_SEL = 14, 3
SWEEP_LAYERS0 = list(range(5, 20))  # 0-based = layers 6-20 (1-based), the paper's video->question range; smoke sweeps the same layers


def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=("smoke", "full"), required=True); p.add_argument("--out-prefix", required=True); p.add_argument("--sweep-json", default=None, help="skip phase 1 and reuse the saved sweep receipt (candidate layers and correct selection videos)"); p.add_argument("--user-authorized-local-only", action="store_true")
    a = p.parse_args()
    if not a.user_authorized_local_only: raise SystemExit("explicit_user_authorized_local_only_required")
    if a.sweep_json: a.sweep_json = str(Path(a.sweep_json).resolve())  # resolve before chdir
    prefix = Path(a.out_prefix).resolve(); outs = {k: Path(f"{prefix}_{k}.json") for k in ("sweep", "scan", "group")}
    if any(o.exists() for k, o in outs.items() if not (a.sweep_json and k == "sweep")): raise SystemExit("output_exists")
    sys.path.insert(0, str(OFFICIAL)); os.chdir(OFFICIAL)
    import torch, numpy as np
    torch.set_grad_enabled(False)
    from tasks.eval.eval_utils import conv_templates
    from tasks.eval.model_utils import load_model_and_dataset
    from analysis.causal_intervention_tools import precompute_attention_masks, predict_from_input, decode_tokens, find_token_range
    pooling = (8, 12, 12); nvis = pooling[0] * pooling[1] * pooling[2]
    model, processor, dataset = load_model_and_dataset(rank=0, world_size=1, pretrained_model_name_or_path=str(MODEL), num_frames=8, use_lora=False, weight_dir=str(MODEL), dataset_name="tvbench",
                                                       lora_alpha=0, lora_target_modules=["q_proj", "v_proj"], pooling_shape=pooling, force_eager=True)
    vocab = processor.tokenizer.get_vocab(); img_idx = model.config.image_token_index; target = model.language_model.model
    L = len(target.layers); H = model.config.text_config.num_attention_heads
    smoke = a.mode == "smoke"; t0 = time.monotonic(); forwards = [0]

    def ranked(task, two_cands):
        ents = [(i, e) for i, e in enumerate(dataset.data_list) if e["task_type"] == task]
        r = sorted(ents, key=lambda ie: hashlib.sha256(f"llava-head-v1|{ie[1]['data']['video']}|{ie[0]}".encode()).hexdigest()); seen, out = set(), []
        for i, e in r:
            d = e["data"]
            if d["video"] in seen or (two_cands and len(d["candidates"]) != 2) or not Path(e["prefix"], d["video"]).is_file(): continue
            seen.add(d["video"]); out.append(i)
        return out
    as_ranked = ranked("Action Sequence", True)
    n_sel, n_astest, n_other = (3, 2, 1) if smoke else (30, 80, 50)
    sel_ids = as_ranked[:n_sel]; astest_ids = as_ranked[n_sel:n_sel + n_astest]
    others = {t: ranked(t, False)[:n_other] for t in ("Moving Direction", "Scene Transition", "Object Count")}

    def prepare(idx):
        ex = dataset[idx]
        if ex["video_pils"] is None: return None
        conv = conv_templates["eval_mvbench"].copy(); conv.user_query(ex["question"], None, "\nOnly give the best option.", is_mm=True); conv.assistant_response("Best option:(")
        inputs = processor(text=conv.get_prompt(), images=ex["video_pils"], return_tensors="pt").to(model.device); inputs["media_type"] = "video"
        ids = inputs["input_ids"][0]; start = int(torch.where(ids == img_idx)[0].item()); n = inputs["input_ids"].shape[1] + nvis - 1
        qr = find_token_range(processor.tokenizer, ids, ex["question"]); assert qr[0] > -1
        qrange = [x + (nvis - 1) for x in range(qr[0], qr[1])]; vision = list(range(start, start + nvis))
        gt = ex["answer"][1]; gt_t = vocab[gt]
        pred_t, p_pred, probs = [x[0] for x in predict_from_input(model, inputs)]; forwards[0] += 1
        correct = decode_tokens(processor.tokenizer, [pred_t.cpu().item()])[0].lower() == gt.lower()
        causal = precompute_attention_masks(n, 1, [], [], model.dtype, model.device)  # [1,1,n,n] pure causal additive
        blocked = precompute_attention_masks(n, 1, [qrange], [vision], model.dtype, model.device)  # causal + question<-video blocked
        return {"idx": idx, "inputs": inputs, "n": n, "gt_t": gt_t, "base_gt": float(probs[gt_t]), "correct": bool(correct), "causal": causal, "blocked": blocked}

    def run_block(v, layer_heads):
        """layer_heads: {layer0: None (all heads) | list of head ids}. Returns relative change (%) of the gold-letter probability. Per-layer mask buffers are reused (v['bufs']) to avoid a 100 MB allocation per forward."""
        saved, restore = {}, []
        try:
            for l, heads in layer_heads.items():
                if heads is None: mask = v["blocked"]
                else:
                    b = v.setdefault("bufs", {}).get(l)
                    if b is None: b = v["bufs"][l] = v["causal"].expand(1, H, v["n"], v["n"]).clone()
                    b[:, heads] = v["blocked"]; restore.append((b, heads)); mask = b
                f = target.layers[l].self_attn.forward; saved[l] = f

                def wrap(fn, m):
                    @functools.wraps(fn)
                    def w(*args, **kwargs):
                        if kwargs["position_ids"].size(-1) == kwargs["attention_mask"].size(-1): kwargs["attention_mask"] = m
                        return fn(*args, **kwargs)
                    return w
                target.layers[l].self_attn.forward = wrap(f, mask)
            out = model(**v["inputs"])
        finally:
            for l, f in saved.items(): target.layers[l].self_attn.forward = f
            for b, heads in restore: b[:, heads] = v["causal"]
        forwards[0] += 1
        prob = float(torch.softmax(out.logits[0, -1, :], dim=0)[v["gt_t"]]); del out
        return (prob - v["base_gt"]) * 100.0 / v["base_gt"]

    def save(path, payload):
        payload.update({"model": "LLaVA-NeXT-7B-Video-FT", "precision": "bf16", "layers": L, "heads": H, "mode": a.mode, "runtime": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
                        "elapsed_seconds": time.monotonic() - t0, "forwards": forwards[0], "rights_status": "unverified; local-only operation", "human_review_performed": False,
                        "interpretation_limit": "dependence of the prediction on video->question attention in selected heads, not explanation faithfulness", "created_at_utc": datetime.now(timezone.utc).isoformat()})
        with open(path, "x") as f: json.dump(payload, f)
        os.chmod(path, 0o600)

    # ---- phase 1 sweep (or reuse of a saved sweep) + phase 2 scan on the selection videos
    sweep_recs, sel_vids = [], []
    if a.sweep_json:
        sw = json.load(open(a.sweep_json)); cand = [c - 1 for c in sw["candidate_layers_1based"]]
        for r in sw["records"]:
            if not r["baseline_correct"]: continue
            v = prepare(r["idx"])
            if v is None or not v["correct"]: print(json.dumps({"reuse_skip": r["idx"]}), flush=True); continue
            sel_vids.append((v, {"idx": r["idx"]})); print(json.dumps({"reuse": len(sel_vids), "elapsed_s": round(time.monotonic() - t0)}), flush=True)
    else:
        for i in sel_ids:
            v = prepare(i)
            if v is None: continue
            rec = {"idx": i, "baseline_correct": v["correct"]}
            if v["correct"]:
                rec["layer_all_heads"] = [run_block(v, {l: None}) for l in SWEEP_LAYERS0]; sel_vids.append((v, rec))
            sweep_recs.append(rec); print(json.dumps({"sweep": len(sweep_recs), "correct": v["correct"], "elapsed_s": round(time.monotonic() - t0)}), flush=True)
            if not v["correct"]: del v
        layer_mean = np.mean([r["layer_all_heads"] for _, r in sel_vids], axis=0)
        cand = sorted(SWEEP_LAYERS0[int(x)] for x in np.argsort(layer_mean)[:N_LAYERS_SEL])  # most negative mean change among the swept layers
        save(outs["sweep"], {"purpose": "llava_head_phase1_layer_sweep", "records": sweep_recs, "swept_layers_1based": [l + 1 for l in SWEEP_LAYERS0], "layer_mean_pct": [float(x) for x in layer_mean], "candidate_layers_1based": [c + 1 for c in cand], "n_selection_requested": len(sel_ids)})
    scan_recs = []
    for v, rec in sel_vids:
        rec["head_scan"] = {str(l + 1): [run_block(v, {l: [h]}) for h in range(H)] for l in cand}
        scan_recs.append({"idx": rec["idx"], "head_scan": rec["head_scan"]}); v.pop("bufs", None); torch.cuda.empty_cache(); print(json.dumps({"scan": len(scan_recs), "elapsed_s": round(time.monotonic() - t0)}), flush=True)
    cells = [(l + 1, h) for l in cand for h in range(H)]
    mean_cell = {c: float(np.mean([r["head_scan"][str(c[0])][c[1]] for r in scan_recs])) for c in cells}
    G = sorted(sorted(cells, key=lambda c: mean_cell[c])[:G_SIZE]); others_cells = [c for c in cells if c not in G]
    rands = [sorted(random.Random(s).sample(others_cells, G_SIZE)) for s in range(3)]
    save(outs["scan"], {"purpose": "llava_head_phase2_head_scan", "records": scan_recs, "candidate_layers_1based": [c + 1 for c in cand], "cell_mean_pct": {f"{c[0]}:{c[1]}": mean_cell[c] for c in cells},
                        "G": G, "RAND": rands, "rule": "14 heads with the most negative mean change over the 3 candidate layers x 32 heads; RAND = 3 random 14-head sets from the remaining heads (seeds 0-2)"})
    for v, _ in sel_vids: del v
    torch.cuda.empty_cache()
    # ---- phase 3 group test on disjoint videos
    def arm_heads(cellset):
        d = {}
        for l1, h in cellset: d.setdefault(l1 - 1, []).append(h)
        return d
    arms = {"G": arm_heads(G), **{f"RAND{k}": arm_heads(r) for k, r in enumerate(rands)}, "ALL": {l: None for l in cand}}
    group_recs = []
    cohorts = [("Action Sequence", astest_ids)] + [(t, ids) for t, ids in others.items()]
    for task, ids in cohorts:
        for i in ids:
            v = prepare(i)
            if v is None: continue
            rec = {"task": task, "idx": i, "baseline_correct": v["correct"]}
            if v["correct"]:
                for name, lh in arms.items(): rec[name] = run_block(v, lh)
            group_recs.append(rec); v.pop("bufs", None); del v; torch.cuda.empty_cache()
        print(json.dumps({"group_task": task, "records": len(group_recs), "elapsed_s": round(time.monotonic() - t0)}), flush=True)
    save(outs["group"], {"purpose": "llava_head_phase3_group_test", "records": group_recs, "G": G, "RAND": rands, "candidate_layers_1based": [c + 1 for c in cand], "arms": list(arms)})
    print(json.dumps({"status": "completed", "forwards": forwards[0], "elapsed_s": round(time.monotonic() - t0)}))


if __name__ == "__main__":
    main()
