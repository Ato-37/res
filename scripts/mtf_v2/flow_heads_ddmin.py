import sys, os, json, argparse, time, hashlib, random
sys.path.insert(0, os.path.expanduser(os.environ.get("MTF_CODE_DIR", "~/external/map-the-flow-fast")))  # official byminji/map-the-flow checkout (commit 566f3ed + DATASET_PATH edit)
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddmin_heads import ddmin
from head_mask import sample_subset, layer_mask, restore_layer_mask, install
from collections import defaultdict
from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_model_and_dataset
from analysis.causal_intervention_tools import (precompute_attention_masks, precompute_random_block_attention_masks,
    _set_block_attn_hooks, _remove_wrapper, find_inter_frame_block_ranges)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--split", required=True); ap.add_argument("--part", choices=["dev", "test"], default="dev")
ap.add_argument("--tasks", default="Action Antonym,Action Sequence,Object Count,Scene Transition,Moving Direction")
ap.add_argument("--n_per_task", type=int, default=5); ap.add_argument("--layers", default="6-15", help="0-based inclusive range; exploratory default until criteria are frozen"); ap.add_argument("--p", type=float, default=0.3); ap.add_argument("--n_subsets", type=int, default=20); ap.add_argument("--max_depth", type=int, default=5); ap.add_argument("--expand_ratio", type=float, default=1.0); ap.add_argument("--max_nodes", type=int, default=80); ap.add_argument("--frac", type=float, default=0.8); ap.add_argument("--max_calls", type=int, default=150)
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
meta = {"script": "flow_heads_ddmin.py", "split": a.split, "split_sha256": hashlib.sha256(open(a.split, "rb").read()).hexdigest(), "part": a.part, "layers": layers, "p": a.p, "n_subsets": a.n_subsets, "max_depth": a.max_depth, "expand_ratio": a.expand_ratio, "max_nodes": a.max_nodes,
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


t0 = time.time(); cohort = []
for task in a.tasks.split(","):
    for vid in split["tasks"][task][a.part][a.skip_n:a.skip_n + a.n_per_task]:
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
        mb, lpb = margin(probs_with(inputs), letters, gt)
        cohort.append({"task": task, "video": vid, "letters": letters, "gt": gt, "inputs": inputs, "block": block, "causal": causal, "ntoks": ntoks, "mb": mb})
        print(f"[{time.time()-t0:.0f}s] cached {vid} ({len(cohort)})", flush=True)

def eval_node(elems):
    sub = {}
    for l, h in elems: sub.setdefault(l, []).append(h)
    out = []
    for c in cohort:
        bufs = {l: c["causal"].expand(1, H, c["ntoks"], c["ntoks"]).clone() for l in sub}
        for l, hs in sub.items(): layer_mask(c["causal"], c["block"], hs, H, bufs[l])
        p = probs_with(c["inputs"], bufs); mn, _ = margin(p, c["letters"], c["gt"]); out.append(mn - c["mb"]); del bufs
    torch.cuda.empty_cache(); return out

rng = random.Random(a.seed); root = [(l, h) for l in layers for h in range(H)]; rng.shuffle(root)
ncall = [0]
def effect(elems):
    v = eval_node(elems); ncall[0] += 1; loss = -sum(v) / len(v)
    fo.write(json.dumps({"call": ncall[0], "size": len(elems), "loss": loss, "per_video": v, "videos": [c["video"] for c in cohort], "elems": elems}) + "\n"); fo.flush()
    print(f"[{time.time()-t0:.0f}s] call {ncall[0]} size {len(elems)} loss {loss:+.3f}", flush=True)
    return loss
res, full, calls = ddmin(root, effect, frac=a.frac, max_calls=a.max_calls)
fo.write(json.dumps({"result": {"minimal": res, "size": None if res is None else len(res), "full_loss": full, "frac": a.frac, "calls": calls, "complete": calls < a.max_calls}}) + "\n"); fo.close(); print("done", time.time() - t0)
