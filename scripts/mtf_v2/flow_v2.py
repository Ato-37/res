"""Window sweep v2 (official LLaVA-NeXT-7B-Video-FT + official code in ~/external/map-the-flow-fast).
Differences from the official information_flow_analysis.py (target cross-frame only):
 - all questions (no baseline-correct filter); records per-option-letter log-probs before/after -> margin (gold - best foil, nat)
 - videos chosen from a frozen split file (split/pilot size), not random sampling
 - optional null: block the SAME NUMBER of random (query,key) cells inside the vision x vision causal region, same layers
 - several window sizes k in one process (model loaded once); layers swept 0..31 (centre window, clipped at edges)
Output: one JSON-lines file per run (new path only)."""
import sys, os, json, argparse, time, hashlib, random
sys.path.insert(0, os.path.expanduser(os.environ.get("MTF_CODE_DIR", "~/external/map-the-flow-fast")))  # official byminji/map-the-flow checkout (commit 566f3ed + DATASET_PATH edit)
import torch
from collections import defaultdict
from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_model_and_dataset
from analysis.causal_intervention_tools import (precompute_attention_masks, precompute_random_block_attention_masks,
    _set_block_attn_hooks, _remove_wrapper, find_inter_frame_block_ranges)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--split", required=True); ap.add_argument("--part", choices=["dev", "test"], default="dev")
ap.add_argument("--tasks", default="Action Antonym,Action Sequence,Object Count,Scene Transition,Moving Direction")
ap.add_argument("--n_per_task", type=int, default=5); ap.add_argument("--ks", default="1,5,9,13"); ap.add_argument("--null", action="store_true")
ap.add_argument("--model_path", default=os.path.expanduser(os.environ.get("MTF_MODEL_PATH", "~/external/models/LLaVA-NeXT-7B-Video-FT"))); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--fp32_readout", action="store_true"); ap.add_argument("--skip_n", type=int, default=0); ap.add_argument("--done_from", default=None)
a = ap.parse_args()
if os.path.exists(a.out): raise SystemExit(f"exists {a.out}")
split = json.load(open(a.split)); ks = [int(x) for x in a.ks.split(",")]; pooling = (8, 12, 12)
model, processor, dataset = load_model_and_dataset(rank=0, world_size=1, pretrained_model_name_or_path=a.model_path, num_frames=8, use_lora=False,
    weight_dir=a.model_path, dataset_name="tvbench", lora_alpha=0, lora_target_modules=["q_proj", "v_proj"], pooling_shape=pooling, force_eager=True)
if a.fp32_readout:
    import torch.nn.functional as F
    _h = None
    for _n, _m in model.named_modules():
        if _n.endswith("lm_head"): _h = _m; break
    assert _h is not None and _h.bias is None and _h.weight.dtype == torch.bfloat16
    _h.forward = lambda hidden, _w=_h.weight: F.linear(hidden.float(), _w.float())
num_layers = model.config.text_config.num_hidden_layers; image_token_index = model.config.image_token_index; mask_heads = 1
vocab = processor.tokenizer.get_vocab(); conv_mode = "eval_mvbench"
idx_by = defaultdict(list)
for i, e in enumerate(dataset.data_list): idx_by[e["task_type"]].append(i)
meta = {"script": "flow_v2.py", "split": a.split, "split_sha256": hashlib.sha256(open(a.split, "rb").read()).hexdigest(), "part": a.part, "ks": ks, "null": a.null,
        "n_per_task": a.n_per_task, "skip_n": a.skip_n, "done_from": a.done_from, "seed": a.seed, "host": __import__("socket").gethostname(), "transformers": __import__("transformers").__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "dtype": str(model.dtype), "attn": "eager", "readout": "fp32_lm_head" if a.fp32_readout else "bf16", "mask_num_heads": mask_heads,
        "note": "mask_num_heads=1 broadcast (same as official fast copy); layers 0-based"}
fo = open(a.out, "w"); fo.write(json.dumps({"meta": meta}) + "\n"); fo.flush()

@torch.no_grad()
def full_probs(inputs, attn_mask=None, layerlist=None):
    if attn_mask is None: return torch.softmax(model(**inputs).logits[0, -1].float(), 0)
    hooks = _set_block_attn_hooks(model, attn_mask, layerlist, "llava")
    try: out = model(**inputs)
    finally: _remove_wrapper(model, hooks, "llava")
    return torch.softmax(out.logits[0, -1].float(), 0)

def margin(p, letters, gt):
    lp = torch.log(p[[vocab[l] for l in letters]].clamp_min(1e-30)); g = letters.index(gt)
    foil = torch.cat([lp[:g], lp[g + 1:]]).max()
    return (lp[g] - foil).item(), lp.tolist()

done_vids = set()
if a.done_from:
    # a video counts as done only if every requested k has all num_layers x conds rows (partial trailing writes are not done)
    _c = defaultdict(lambda: defaultdict(set))
    for _f in a.done_from.split(","):  # comma-separated list of earlier output files
        for _l in open(_f):
            _r = json.loads(_l)
            if "meta" in _r: continue
            _c[(_r["task"], _r["video"])][_r["k"]].add((_r["layer"], _r["cond"]))
    _need = num_layers * (2 if a.null else 1)
    done_vids = {key for key, v in _c.items() if all(len(v.get(k, ())) >= _need for k in ks)}
t0 = time.time(); nvid = 0
for task in a.tasks.split(","):
    want = split["tasks"][task][a.part][a.skip_n:a.skip_n + a.n_per_task]; seen = set()
    for vid in want:
        if (task, vid) in done_vids: continue
        qi = [i for i in idx_by[task] if dataset.data_list[i]["data"]["video"] == vid]
        if not qi: continue
        ex = dataset[qi[0]]  # one question per video (video = unit)
        letters = list("ABCD"[:1 + len(ex["false_options"])]); gt = ex["answer"][1]
        conv = conv_templates[conv_mode].copy(); conv.user_query(ex["question"], None, "\nOnly give the best option.", is_mm=True); conv.assistant_response("Best option:(")
        prompt = conv.get_prompt(); inputs = processor(text=prompt, images=ex["video_pils"], return_tensors="pt").to(model.device); inputs["media_type"] = "video"
        ids = inputs["input_ids"][0]; ph = torch.where(ids == image_token_index)[0].item(); nvis = 8 * 144
        vision_range = [x + ph for x in range(nvis)]; ntoks = ids.shape[0] + nvis - 1
        qlists, klists = find_inter_frame_block_ranges(vision_range, num_frames=8, num_vis_one_frame=144, vis_start_id=ph)
        am = precompute_attention_masks(ntoks, mask_heads, qlists, klists, model.dtype, model.device)  # (query_ranges, key_ranges) as in the official call
        base = torch.tril(torch.ones((ntoks, ntoks), device=model.device)); n_block = int(((am[0, 0] < -1e4).float() * base).sum().item())
        vr = torch.tensor(vision_range, device=model.device); vv = torch.tril(torch.ones(nvis, nvis, device=model.device)).nonzero() + ph
        pb = full_probs(inputs); mb, lpb = margin(pb, letters, gt); pred = letters[int(torch.tensor(lpb).argmax())]
        rec_base = {"task": task, "video": vid, "letters": letters, "gt": gt, "pred": pred, "base_margin": mb, "base_logp": lpb, "n_block_cells": n_block}
        for k in ks:
            for layer in range(num_layers):
                ll = list(range(max(0, layer - k // 2), min(num_layers, layer - (-k // 2))))
                for cond in (["real", "null"] if a.null else ["real"]):
                    if cond == "real": m = am
                    else:
                        torch.manual_seed(a.seed * 1000003 + layer * 31 + k)
                        m = precompute_random_block_attention_masks(ntoks, mask_heads, vv, n_block, model.dtype, model.device)
                    p = full_probs(inputs, m, ll); mn, lpn = margin(p, letters, gt)
                    fo.write(json.dumps({**rec_base, "k": k, "layer": layer, "cond": cond, "new_margin": mn, "new_logp": lpn, "d_margin": mn - mb}) + "\n")
        fo.flush(); nvid += 1; print(f"[{time.time()-t0:.0f}s] {task} {vid} done ({nvid})", flush=True)
fo.close(); print("done", time.time() - t0)
