"""Path patching: single-head attention knockout with and without freezing the detour routes (see pathpatch.py).
sender / live receiver positions: v2q = question / last; q2l = last / none; ff = vision / question + last.
d_free = margin(plain single-head knockout) - margin(no block); d_frozen = the same with detours frozen. Row layer=-1 is the whole range."""
import sys, os, json, argparse, time, hashlib
sys.path.insert(0, os.path.expanduser(os.environ.get("MTF_CODE_DIR", "~/external/map-the-flow-fast")))
import torch
from collections import defaultdict
from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_model_and_dataset
from analysis.causal_intervention_tools import precompute_attention_masks, find_token_range, find_inter_frame_block_ranges
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from head_mask import install
from pathpatch import record, freeze

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--split", required=True); ap.add_argument("--part", choices=["dev"], default="dev")
ap.add_argument("--tasks", default="Action Antonym,Scene Transition"); ap.add_argument("--n_per_task", type=int, default=10); ap.add_argument("--skip_n", type=int, default=0)
ap.add_argument("--check", action="store_true", help="identity checks on the real model"); ap.add_argument("--pairs", default="", help="path:task items separated by ;"); ap.add_argument("--done_from", default=""); ap.add_argument("--max_heads", type=int, default=0, help="smoke only")
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
assert model.config.text_config.pretraining_tp == 1
meta = {"script": "flow_pathpatch.py", "split_sha256": hashlib.sha256(open(a.split, "rb").read()).hexdigest(), "ranges": {k: [v.start, v.stop - 1] for k, v in RANGES.items()}, "n_per_task": a.n_per_task, "skip_n": a.skip_n, "pairs": a.pairs, "max_heads": a.max_heads,
        "host": __import__("socket").gethostname(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "readout": "fp32_lm_head" if a.fp32_readout else "bf16", "H": H, "attn": type(target.layers[0].self_attn).__name__}
fo = open(a.out, "w"); fo.write(json.dumps({"meta": meta}) + "\n"); fo.flush()

@torch.no_grad()
def margin_with(c, masks=None, patch=None):
    torch.cuda.empty_cache()
    u1 = install(target.layers, masks) if masks else (lambda: None)
    u2 = freeze(target.layers, c["cache"], patch[0], patch[1], patch[2], H) if patch else (lambda: None)
    try: out = model(**c["inputs"], use_cache=False)
    finally: u2(); u1()
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
    c["pos"] = {"v2q": (question, [last]), "q2l": ([last], []), "ff": (vision, question + [last])}
    c["cache"], undo = record(target.layers)
    try: c["mb"] = margin_with(c)
    finally: undo()
    return c

pairs = [tuple(x.split(":")) for x in a.pairs.split(";") if x] or [(p, t) for p in RANGES for t in a.tasks.split(",")]
t0 = time.time()
for path, task in pairs:
    for vid in split["tasks"][task][a.part][a.skip_n:a.skip_n + a.n_per_task]:
        R = RANGES[path]
        todo = [(l, h) for l in R for h in range(H)]
        if a.max_heads: todo = todo[:a.max_heads]
        todo = [x for x in todo if (path, task, vid, *x) not in done]; whole = (path, task, vid, -1, -1) not in done
        if not todo and not whole: continue
        c = prepare(task, vid)
        if c is None: continue
        blk = c["block"][path]; n = blk.shape[-1]; spos, rpos = c["pos"][path]
        one = c["causal"].expand(1, H, n, n).clone()
        base = {"path": path, "task": task, "video": vid, "base_margin": c["mb"]}
        if a.check:
            l = R.start
            print("CHECK no-block+freeze - base", margin_with(c, {l: one}, ({l: [0]}, spos, rpos)) - c["mb"],
                  "| no hooks, causal mask - base", margin_with(c, {x: one for x in R}) - c["mb"], flush=True)
            full = {x: blk for x in R}; send = {x: range(H) for x in R}
            print("CHECK full block", margin_with(c, full) - c["mb"], "| frozen, no receivers", margin_with(c, full, (send, spos, [])) - c["mb"], "| frozen, everything live", margin_with(c, full, (send, list(range(n)), list(range(n)))) - c["mb"], flush=True)
        full = {x: blk for x in R}
        if whole: fo.write(json.dumps({**base, "layer": -1, "head": -1, "d_free": margin_with(c, full) - c["mb"], "d_frozen": margin_with(c, full, ({x: range(H) for x in R}, spos, rpos)) - c["mb"]}) + "\n")
        t1 = time.time()
        for l, h in todo:
            one[:, h] = blk[:, 0]
            free = margin_with(c, {l: one}) - c["mb"]
            froz = margin_with(c, {l: one}, ({l: [h]}, spos, rpos)) - c["mb"]
            one[:, h] = c["causal"][:, 0]
            fo.write(json.dumps({**base, "layer": l, "head": h, "d_free": free, "d_frozen": froz}) + "\n"); fo.flush()
        del one; torch.cuda.empty_cache()
        print(f"[{time.time()-t0:.0f}s] {path} {task} {vid} done, {(time.time()-t1)/max(1,2*len(todo)):.2f} s/forward", flush=True)
fo.close(); print("done", time.time() - t0)
