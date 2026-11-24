import sys, os, json, argparse, time, hashlib, random
sys.path.insert(0, os.path.expanduser(os.environ.get("MTF_CODE_DIR", "~/external/map-the-flow-fast")))  # official byminji/map-the-flow checkout (commit 566f3ed + DATASET_PATH edit)
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from head_mask import sample_subset, layer_mask, restore_layer_mask, install
from collections import defaultdict
from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_model_and_dataset
from analysis.causal_intervention_tools import (precompute_attention_masks, precompute_random_block_attention_masks,
    _set_block_attn_hooks, _remove_wrapper, find_inter_frame_block_ranges)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--split", required=True); ap.add_argument("--part", choices=["dev", "test"], default="dev")
ap.add_argument("--tasks", default="Action Antonym,Action Sequence,Object Count,Scene Transition,Moving Direction")
ap.add_argument("--n_per_task", type=int, default=5); ap.add_argument("--layers", default="6-15", help="0-based inclusive range; exploratory default until criteria are frozen"); ap.add_argument("--p", type=float, default=0.3); ap.add_argument("--n_subsets", type=int, default=20); ap.add_argument("--complement", action="store_true", help="also block the complement of every sampled subset (additivity test: y(S)+y(S^c) vs y(all))")
ap.add_argument("--model_path", default=os.path.expanduser(os.environ.get("MTF_MODEL_PATH", "~/external/models/LLaVA-NeXT-7B-Video-FT"))); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--fp32_readout", action="store_true"); ap.add_argument("--skip_n", type=int, default=0); ap.add_argument("--done_from", default=None)
a = ap.parse_args()
if os.path.exists(a.out): raise SystemExit(f"exists {a.out}")
split = json.load(open(a.split)); pooling = (8, 12, 12)
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
lo, hi = [int(x) for x in a.layers.split("-")]; layers = list(range(lo, hi + 1)); H = model.config.text_config.num_attention_heads
meta = {"script": "flow_heads_v2.py", "split": a.split, "split_sha256": hashlib.sha256(open(a.split, "rb").read()).hexdigest(), "part": a.part, "layers": layers, "p": a.p, "n_subsets": a.n_subsets, "complement": a.complement,
        "n_per_task": a.n_per_task, "skip_n": a.skip_n, "seed": a.seed, "H": H, "host": __import__("socket").gethostname(), "transformers": __import__("transformers").__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "readout": "fp32_lm_head" if a.fp32_readout else "bf16", "note": "per-head [1,H,n,n] masks; subset seed = seed*1000003 + video_index*131 + s; layers 0-based"}
fo = open(a.out, "w"); fo.write(json.dumps({"meta": meta}) + "\n"); fo.flush()
target = model.language_model.model

@torch.no_grad()
def probs_with(inputs, masks=None):
    undo = install(target.layers, masks) if masks else (lambda: None)
    try: out = model(**inputs)
    finally: undo()
    return torch.softmax(out.logits[0, -1].float(), 0)

def margin(p, letters, gt):
    lp = torch.log(p[[vocab[l] for l in letters]].clamp_min(1e-30)); g = letters.index(gt)
    foil = torch.cat([lp[:g], lp[g + 1:]]).max()
    return (lp[g] - foil).item(), lp.tolist()

t0 = time.time(); nvid = 0; vcount = 0
for task in a.tasks.split(","):
    for vid in split["tasks"][task][a.part][a.skip_n:a.skip_n + a.n_per_task]:
        vcount += 1
        qi = [i for i in idx_by[task] if dataset.data_list[i]["data"]["video"] == vid]
        if not qi: continue
        ex = dataset[qi[0]]
        letters = list("ABCD"[:1 + len(ex["false_options"])]); gt = ex["answer"][1]
        conv = conv_templates[conv_mode].copy(); conv.user_query(ex["question"], None, "\nOnly give the best option.", is_mm=True); conv.assistant_response("Best option:(")
        inputs = processor(text=conv.get_prompt(), images=ex["video_pils"], return_tensors="pt").to(model.device); inputs["media_type"] = "video"
        ids = inputs["input_ids"][0]; ph = torch.where(ids == image_token_index)[0].item(); nvis = 8 * 144
        vision_range = [x + ph for x in range(nvis)]; ntoks = ids.shape[0] + nvis - 1
        qlists, klists = find_inter_frame_block_ranges(vision_range, num_frames=8, num_vis_one_frame=144, vis_start_id=ph)
        block = precompute_attention_masks(ntoks, 1, qlists, klists, model.dtype, model.device)
        causal = precompute_attention_masks(ntoks, 1, [], [], model.dtype, model.device)
        mb, lpb = margin(probs_with(inputs), letters, gt); pred = letters[int(torch.tensor(lpb).argmax())]
        rec_base = {"task": task, "video": vid, "letters": letters, "gt": gt, "pred": pred, "base_margin": mb, "base_logp": lpb}
        bufs = {l: causal.expand(1, H, ntoks, ntoks).clone() for l in layers}
        def run(sub):
            for l, hs in sub.items(): layer_mask(causal, block, hs, H, bufs[l])
            try: p = probs_with(inputs, {l: bufs[l] for l in sub})
            finally:
                for l, hs in sub.items(): restore_layer_mask(bufs[l], causal, hs)
            return margin(p, letters, gt)
        allsub = {l: list(range(H)) for l in layers}
        jobs = [("all", allsub)]
        for s in range(a.n_subsets):
            sub = sample_subset(layers, H, a.p, a.seed * 1000003 + vcount * 131 + s); jobs.append((f"s{s}", sub))
            if a.complement: jobs.append((f"c{s}", {l: [h for h in range(H) if h not in set(sub.get(l, []))] for l in layers}))
        for name, sub in jobs:
            mn, lpn = run(sub)
            fo.write(json.dumps({**rec_base, "cond": name, "subset": {str(l): hs for l, hs in sub.items()}, "n_heads": sum(len(h) for h in sub.values()), "new_margin": mn, "new_logp": lpn, "d_margin": mn - mb}) + "\n")
        del bufs; torch.cuda.empty_cache(); fo.flush(); nvid += 1; print(f"[{time.time()-t0:.0f}s] {task} {vid} done ({nvid})", flush=True)
fo.close(); print("done", time.time() - t0)
