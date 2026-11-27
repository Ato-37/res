"""Confirmation of the top-10 heads on other videos (restore them jointly vs 20 random sets of 10).
Per-pathway head leave-one-out on the paper's layer ranges (0-based, official effective_pathway_analysis.py):
v2q = layers 5..19 (question rows x video keys), q2l = 15..24 (last row x question keys), ff = 5..14 (cross-frame vision x vision).
mode gate: block the path in all heads of its range; also the same block with the official 1-head broadcast mask (equality check).
mode loo: block all heads of the range, restore exactly one (layer, head) to the plain causal mask; every head once, nothing is discarded.
Record: d = margin(intervention) - margin(no block)."""
import sys, os, json, argparse, time, hashlib
sys.path.insert(0, os.path.expanduser(os.environ.get("MTF_CODE_DIR", "~/external/map-the-flow-fast")))
import torch
from collections import defaultdict
from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_model_and_dataset
from analysis.causal_intervention_tools import precompute_attention_masks, find_token_range, find_inter_frame_block_ranges
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from head_mask import install

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--split", required=True); ap.add_argument("--part", choices=["dev"], default="dev")
ap.add_argument("--tasks", default="Action Antonym,Scene Transition"); ap.add_argument("--n_per_task", type=int, default=10); ap.add_argument("--skip_n", type=int, default=0)
ap.add_argument("--mode", default="confirm"); ap.add_argument("--select", required=True); ap.add_argument("--n_random", type=int, default=20); ap.add_argument("--rand_seed", type=int, default=777); ap.add_argument("--pairs", default="", help="path:task items separated by ;"); ap.add_argument("--done_from", default=""); ap.add_argument("--max_heads", type=int, default=0, help="smoke only")
ap.add_argument("--model_path", default=os.path.expanduser(os.environ.get("MTF_MODEL_PATH", "~/external/models/LLaVA-NeXT-7B-Video-FT"))); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--fp32_readout", action="store_true")
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
RANGES = {"v2q": range(5, 20), "q2l": range(15, 25), "ff": range(5, 15)}
H = model.config.text_config.num_attention_heads; target = model.language_model.model
meta = {"script": "flow_loo_confirm.py", "mode": a.mode, "split": a.split, "split_sha256": hashlib.sha256(open(a.split, "rb").read()).hexdigest(), "ranges": {k: [v.start, v.stop - 1] for k, v in RANGES.items()}, "tasks": a.tasks, "n_per_task": a.n_per_task, "skip_n": a.skip_n, "pairs": a.pairs, "done_from": a.done_from,
        "host": __import__("socket").gethostname(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "readout": "fp32_lm_head" if a.fp32_readout else "bf16", "H": H}
done = set()
for f in [x for x in a.done_from.split(",") if x]:
    for line in open(f):
        try: r = json.loads(line)
        except ValueError: continue
        if "path" in r: done.add((r["path"], r["task"], r["video"], r["layer"], r["head"]))
fo = open(a.out, "w"); fo.write(json.dumps({"meta": meta}) + "\n"); fo.flush()

@torch.no_grad()
def margin_with(c, masks):
    torch.cuda.empty_cache()  # keep the allocator under 16 GB (WSL spills to host memory otherwise)
    undo = install(target.layers, masks) if masks else (lambda: None)
    try: out = model(**c["inputs"], use_cache=False)  # no KV cache: avoids spilling past 16 GB
    finally: undo()
    lp = torch.log_softmax(out.logits[0, -1].float(), 0)[[vocab[l] for l in c["letters"]]]
    g = c["letters"].index(c["gt"])
    return (lp[g] - torch.cat([lp[:g], lp[g + 1:]]).max()).item()

def prepare(task, vid):
    qi = [i for i in idx_by[task] if dataset.data_list[i]["data"]["video"] == vid]
    if not qi: return None
    ex = dataset[qi[0]]
    letters = list("ABCD"[:1 + len(ex["false_options"])]); gt = ex["answer"][1]
    conv = conv_templates[conv_mode].copy(); conv.user_query(ex["question"], None, "\nOnly give the best option.", is_mm=True); conv.assistant_response("Best option:(")
    inputs = processor(text=conv.get_prompt(), images=ex["video_pils"], return_tensors="pt").to(model.device); inputs["media_type"] = "video"
    ids = inputs["input_ids"][0]; ph = torch.where(ids == image_token_index)[0].item(); nvis = 8 * 144
    vision = list(range(ph, ph + nvis)); ntoks = ids.shape[0] + nvis - 1; last = ntoks - 1
    s, e = find_token_range(processor.tokenizer, ids, ex["question"]); question = [x + nvis - 1 for x in range(s, e)]
    assert question[0] >= ph + nvis and question[-1] < last
    q_i, k_i = find_inter_frame_block_ranges(vision, num_frames=8, num_vis_one_frame=144, vis_start_id=ph)
    c = {"inputs": inputs, "letters": letters, "gt": gt, "block": {
        "v2q": precompute_attention_masks(ntoks, 1, [question], [vision], model.dtype, model.device),
        "q2l": precompute_attention_masks(ntoks, 1, [[last]], [question], model.dtype, model.device),
        "ff": precompute_attention_masks(ntoks, 1, q_i, k_i, model.dtype, model.device)},
        "causal": precompute_attention_masks(ntoks, 1, [], [], model.dtype, model.device)}
    c["mb"] = margin_with(c, None)
    return c

import functools, random
sel = json.load(open(a.select))

@torch.no_grad()
def margin_restore(c, path, heads):
    """block the path in every head of its range except `heads`; the 32-head mask of a layer is built inside its forward and freed right after."""
    blk = c["block"][path]; n = blk.shape[-1]; by = defaultdict(list)
    for l, h in heads: by[l].append(h)
    saved = {}
    def wrap(fn, l):
        @functools.wraps(fn)
        def w(*args, **kw):
            if kw["position_ids"].size(-1) == kw["attention_mask"].size(-1):
                if l in by:
                    m = blk.expand(1, H, n, n).clone(); m[:, by[l]] = c["causal"][:, 0:1]
                else: m = blk
                kw["attention_mask"] = m
            return fn(*args, **kw)
        return w
    for l in RANGES[path]:
        saved[l] = target.layers[l].self_attn.forward; target.layers[l].self_attn.forward = wrap(saved[l], l)
    torch.cuda.empty_cache()
    try: out = model(**c["inputs"], use_cache=False)
    finally:
        for l, f in saved.items(): target.layers[l].self_attn.forward = f
    lp = torch.log_softmax(out.logits[0, -1].float(), 0)[[vocab[x] for x in c["letters"]]]; g = c["letters"].index(c["gt"])
    return (lp[g] - torch.cat([lp[:g], lp[g + 1:]]).max()).item()

t0 = time.time()
for key, info in sel.items():
    path, task = key.split(":"); top = [tuple(h) for h in info["top"]]
    universe = [(l, h) for l in RANGES[path] for h in range(H)]; rr = random.Random(a.rand_seed)
    conds = [("full", []), ("top", top)] + [(f"rand{i}", rr.sample(universe, len(top))) for i in range(a.n_random)]
    fo.write(json.dumps({"pair": key, "direction": info["direction"], "conds": dict(conds)}) + "\n")
    nv = 0
    for vid in split["tasks"][task][a.part][a.skip_n:a.skip_n + a.n_per_task]:
        c = prepare(task, vid)
        if c is None: continue
        rec = {"path": path, "task": task, "video": vid, "base_margin": c["mb"], "d": {n: margin_restore(c, path, hs) - c["mb"] for n, hs in conds}}
        if a.max_heads: rec["check_single"] = [[l, h, margin_restore(c, path, [(l, h)]) - c["mb"]] for l, h in universe[:a.max_heads]]
        fo.write(json.dumps(rec) + "\n"); fo.flush(); nv += 1; print(f"[{time.time()-t0:.0f}s] {key} {vid} done ({nv}) full {rec['d']['full']:+.3f} top {rec['d']['top']:+.3f}", flush=True)
fo.close(); print("done", time.time() - t0)
