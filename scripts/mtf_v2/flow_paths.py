"""Per-pathway knockout for video->question (v2q) and question->last (q2l), token ranges as in the official effective_pathway_analysis.py.
v2q: question rows cannot attend to the 1152 video keys. q2l: the last row cannot attend to the question keys.
q2l_null: the last row blocks the same number of random other earlier keys (not question, not itself). v2q has no matched null (too few non-video keys).
Layer sweep: window of k layers centred on each layer; endpoint d_margin (gold - best foil, nat), fp32 readout."""
import sys, os, json, argparse, time, hashlib
sys.path.insert(0, os.path.expanduser(os.environ.get("MTF_CODE_DIR", "~/external/map-the-flow-fast")))
import torch
from collections import defaultdict
from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_model_and_dataset
from analysis.causal_intervention_tools import precompute_attention_masks, find_token_range, _set_block_attn_hooks, _remove_wrapper

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--split", required=True); ap.add_argument("--part", choices=["dev"], default="dev")
ap.add_argument("--tasks", default="Action Antonym,Scene Transition"); ap.add_argument("--n_per_task", type=int, default=10); ap.add_argument("--skip_n", type=int, default=0)
ap.add_argument("--k", type=int, default=7); ap.add_argument("--layer_lo", type=int, default=0); ap.add_argument("--layer_hi", type=int, default=31)
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
meta = {"script": "flow_paths.py", "split": a.split, "split_sha256": hashlib.sha256(open(a.split, "rb").read()).hexdigest(), "part": a.part, "k": a.k, "layer_lo": a.layer_lo, "layer_hi": a.layer_hi, "tasks": a.tasks, "n_per_task": a.n_per_task, "skip_n": a.skip_n, "seed": a.seed,
        "host": __import__("socket").gethostname(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "readout": "fp32_lm_head" if a.fp32_readout else "bf16", "mask_num_heads": 1, "note": "layers 0-based"}
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

t0 = time.time(); nvid = 0
for task in a.tasks.split(","):
    for vid in split["tasks"][task][a.part][a.skip_n:a.skip_n + a.n_per_task]:
        qi = [i for i in idx_by[task] if dataset.data_list[i]["data"]["video"] == vid]
        if not qi: continue
        ex = dataset[qi[0]]
        letters = list("ABCD"[:1 + len(ex["false_options"])]); gt = ex["answer"][1]
        conv = conv_templates[conv_mode].copy(); conv.user_query(ex["question"], None, "\nOnly give the best option.", is_mm=True); conv.assistant_response("Best option:(")
        inputs = processor(text=conv.get_prompt(), images=ex["video_pils"], return_tensors="pt").to(model.device); inputs["media_type"] = "video"
        ids = inputs["input_ids"][0]; ph = torch.where(ids == image_token_index)[0].item(); nvis = 8 * 144
        vision = list(range(ph, ph + nvis)); ntoks = ids.shape[0] + nvis - 1; last = ntoks - 1
        s, e = find_token_range(processor.tokenizer, ids, ex["question"]); question = [x + nvis - 1 for x in range(s, e)]
        assert question[0] >= ph + nvis and question[-1] < last, (ph, question[0], question[-1], last)
        qtext = processor.tokenizer.decode(ids[s:e])
        m_v2q = precompute_attention_masks(ntoks, 1, [question], [vision], model.dtype, model.device)
        m_q2l = precompute_attention_masks(ntoks, 1, [[last]], [question], model.dtype, model.device)
        pool = [x for x in range(last) if x < question[0] or x > question[-1]]
        g = torch.Generator(); g.manual_seed(a.seed * 1000003 + nvid)
        null_keys = [pool[i] for i in torch.randperm(len(pool), generator=g)[:len(question)].tolist()]
        m_null = precompute_attention_masks(ntoks, 1, [[last]], [null_keys], model.dtype, model.device)
        pb = full_probs(inputs); mb, lpb = margin(pb, letters, gt); pred = letters[int(torch.tensor(lpb).argmax())]
        rec = {"task": task, "video": vid, "gt": gt, "pred": pred, "base_margin": mb, "n_question_tokens": len(question)}
        if nvid == 0: print("question span decodes to:", repr(qtext[:80]), "...", repr(qtext[-60:]), flush=True)
        for layer in range(a.layer_lo, a.layer_hi + 1):
            ll = list(range(max(0, layer - a.k // 2), min(num_layers, layer - (-a.k // 2))))
            for path, m in (("v2q", m_v2q), ("q2l", m_q2l), ("q2l_null", m_null)):
                mn, _ = margin(full_probs(inputs, m, ll), letters, gt)
                fo.write(json.dumps({**rec, "layer": layer, "path": path, "d_margin": mn - mb}) + "\n")
        fo.flush(); nvid += 1; print(f"[{time.time()-t0:.0f}s] {task} {vid} done ({nvid})", flush=True)
fo.close(); print("done", time.time() - t0)
