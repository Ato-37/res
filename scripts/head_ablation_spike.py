"""Head-ablation (diagnosis) spike: for one or more video+question+known-answer
samples, zero out one attention head's contribution at a time (via a
forward_pre_hook on that layer's o_proj) and see how much the log-probability
of the correct answer's first token drops. A bigger drop means that head
mattered more for producing the correct answer.

Two modes:
  --mode layer  ablates an entire layer's attention output (all heads at
                once) for every layer, to cheaply find promising layers.
  --mode head   ablates each head individually within the chosen layer(s)
                (--layers), to localize which head(s) matter.

Single sample:
    python scripts/head_ablation_spike.py --video clip.mp4 \
        --question "What did the baby do ...?" --answer "lay on floor" \
        --mode layer

Multiple samples (loads the model once, reuses it for all samples), file is a
JSON list of {"video": ..., "question": ..., "answer": ...} objects:
    python scripts/head_ablation_spike.py --samples-file samples.json --mode layer
    python scripts/head_ablation_spike.py --samples-file samples.json --mode head --layers 0,23,27
"""

import argparse
import json
import os
import shutil
import tempfile

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Observed to be 3584/28 (7B) and 2048/16 (3B) for Qwen2.5-VL checkpoints so
# far -- i.e. head_dim=128 in both. Re-check if a different model is used.
HEAD_DIM = 128


from closed_research_branch import reject_inference

def extract_frames(video_path, num_frames, max_dim):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Could not read frame count from {video_path}")

    indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    tmp_dir = tempfile.mkdtemp(prefix="head_ablation_frames_")
    frame_paths = []

    for pos, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail((max_dim, max_dim))
        path = f"{tmp_dir}/frame_{pos:03d}.jpg"
        img.save(path)
        frame_paths.append(path)

    cap.release()
    return frame_paths, tmp_dir, total, indices


def logprobs_and_entropy(model, inputs):
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    probs = logprobs.exp()
    entropy = -(probs * logprobs).sum().item()
    return logprobs, entropy


def compute_metrics(model, inputs, token_ids):
    """One forward pass -> ([logprob per token_id], entropy).

    entropy is over the full next-token distribution: if ablation raises the
    correct token's logprob mainly by flattening the whole distribution, a
    "control" token matched to the correct token's baseline probability
    should rise by a similar amount. If the correct token rises much more
    than the matched control, that is evidence of a specific/selective
    effect rather than generic flattening.
    """
    logprobs, entropy = logprobs_and_entropy(model, inputs)
    return [logprobs[t].item() for t in token_ids], entropy


def build_valid_word_mask(tokenizer, vocab_size):
    """True for token ids that decode to a clean alphabetic word (>=2 letters,
    no replacement-character byte-fallback junk). Used so the control token is
    a real word, not an incomplete UTF-8 byte fragment."""
    valid = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(vocab_size):
        s = tokenizer.decode([i]).strip()
        if s and "�" not in s and s.isalpha() and len(s) >= 2:
            valid[i] = True
    return valid


def pick_control_token(base_logprobs, exclude_ids, target_logprob, valid_mask):
    """Real-word token whose baseline logprob is closest to target_logprob, excluding exclude_ids."""
    diffs = (base_logprobs - target_logprob).abs().clone()
    diffs[~valid_mask] = float("inf")
    for tid in exclude_ids:
        diffs[tid] = float("inf")
    return int(diffs.argmin().item())


def zero_head_hook(head_idx, head_dim):
    def hook(module, args):
        x = args[0].clone()
        x[..., head_idx * head_dim:(head_idx + 1) * head_dim] = 0
        return (x,) + args[1:]
    return hook


def zero_layer_hook():
    def hook(module, args):
        x = torch.zeros_like(args[0])
        return (x,) + args[1:]
    return hook


def run_sample(model, processor, device, layers, valid_mask, video, question, answer, num_frames, frame_size, mode, target_layers):
    """Returns (baseline_logprob, {layer_idx: drop} or {(layer_idx, head_idx): drop})."""
    frame_paths, tmp_dir, total, indices = extract_frames(video, num_frames, frame_size)
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frame_paths},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
        ).to(device)

        answer_token_id = processor.tokenizer(answer, add_special_tokens=False).input_ids[0]

        base_logprobs, base_entropy = logprobs_and_entropy(model, inputs)
        wrong_token_id = int(base_logprobs.argmax().item())
        base_correct = base_logprobs[answer_token_id].item()
        control_token_id = pick_control_token(base_logprobs, [answer_token_id, wrong_token_id], base_correct, valid_mask)
        token_ids = [answer_token_id, wrong_token_id, control_token_id]
        base_correct, base_wrong, base_control = [base_logprobs[t].item() for t in token_ids]

        print(f"  video={video} total_frames={total} sampled={indices} "
              f"total_input_tokens={inputs['input_ids'].shape[1]} "
              f"answer_token={processor.tokenizer.decode([answer_token_id])!r} "
              f"model_top_token={processor.tokenizer.decode([wrong_token_id])!r} "
              f"control_token={processor.tokenizer.decode([control_token_id])!r} "
              f"baseline: correct_logprob={base_correct:.4f} wrong_logprob={base_wrong:.4f} "
              f"control_logprob={base_control:.4f} entropy={base_entropy:.4f}")

        def deltas(c, w, ctl, e):
            return {
                "correct_drop": base_correct - c,
                "wrong_drop": base_wrong - w,
                "control_drop": base_control - ctl,
                "entropy_delta": e - base_entropy,
            }

        results = {}
        if mode == "layer":
            for i, layer in enumerate(layers):
                handle = layer.self_attn.o_proj.register_forward_pre_hook(zero_layer_hook())
                (c, w, ctl), e = compute_metrics(model, inputs, token_ids)
                handle.remove()
                results[i] = deltas(c, w, ctl, e)
        else:
            for li in target_layers:
                layer = layers[li]
                num_heads = layer.self_attn.o_proj.in_features // HEAD_DIM
                for h in range(num_heads):
                    handle = layer.self_attn.o_proj.register_forward_pre_hook(zero_head_hook(h, HEAD_DIM))
                    (c, w, ctl), e = compute_metrics(model, inputs, token_ids)
                    handle.remove()
                    results[(li, h)] = deltas(c, w, ctl, e)
        return (base_correct, base_wrong, base_control), results
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Single-sample mode: path to one video")
    parser.add_argument("--question", help="Single-sample mode: question text")
    parser.add_argument("--answer", help="Single-sample mode: ground-truth answer text")
    parser.add_argument("--samples-file", help="Multi-sample mode: JSON list of {video, question, answer}")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--mode", choices=["layer", "head"], default="layer")
    parser.add_argument("--layers", default="0", help="Comma-separated layer indices to sweep heads in (--mode head only)")
    args = parser.parse_args()
    reject_inference(parser, 'legacy_head_ablation',
                     'retired head/layer search is closed')

    if args.samples_file:
        samples = json.load(open(args.samples_file))
    else:
        assert args.video and args.question and args.answer, "Provide --samples-file, or all of --video/--question/--answer"
        samples = [{"video": args.video, "question": args.question, "answer": args.answer}]
    for s in samples:
        s["video"] = os.path.expanduser(s["video"])

    target_layers = [int(x) for x in args.layers.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, quant={args.quant}, mode={args.mode}, n_samples={len(samples)}")

    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif args.quant == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
        quantization_config=quant_config, device_map=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    layers = model.model.language_model.layers

    vocab_size = model.get_output_embeddings().weight.shape[0]
    print(f"building valid-word mask over vocab_size={vocab_size} (one-time cost)...")
    valid_mask = build_valid_word_mask(processor.tokenizer, vocab_size)
    print(f"valid_word_tokens={int(valid_mask.sum())}")

    all_results = []
    for i, s in enumerate(samples):
        print(f"\n=== sample {i}: {s['question']!r} -> {s['answer']!r} ===")
        _, results = run_sample(
            model, processor, device, layers, valid_mask, s["video"], s["question"], s["answer"],
            args.num_frames, args.frame_size, args.mode, target_layers,
        )
        def fmt(idx, m):
            return (f"idx={idx} correct_drop={m['correct_drop']:+.4f} "
                    f"wrong_drop={m['wrong_drop']:+.4f} control_drop={m['control_drop']:+.4f} "
                    f"entropy_delta={m['entropy_delta']:+.4f}")

        ranked = sorted(results.items(), key=lambda t: -t[1]["correct_drop"])
        print("  top (helps correct answer):")
        for idx, m in ranked[:5]:
            print(f"    {fmt(idx, m)}")
        print("  bottom (shortcut candidate: ablation helps the correct answer):")
        for idx, m in ranked[-5:]:
            print(f"    {fmt(idx, m)}")
        all_results.append(results)

    if len(samples) > 1:
        keys = all_results[0].keys()
        metric_names = ["correct_drop", "wrong_drop", "control_drop", "entropy_delta"]
        avg = {k: {mn: sum(r[k][mn] for r in all_results) / len(all_results) for mn in metric_names} for k in keys}

        def fmt_avg(idx, m):
            return (f"idx={idx} avg_correct_drop={m['correct_drop']:+.4f} "
                    f"avg_wrong_drop={m['wrong_drop']:+.4f} avg_control_drop={m['control_drop']:+.4f} "
                    f"avg_entropy_delta={m['entropy_delta']:+.4f}")

        ranked_avg = sorted(avg.items(), key=lambda t: -t[1]["correct_drop"])
        print(f"\n=== average over {len(samples)} samples ===")
        print("  top:")
        for idx, m in ranked_avg[:10]:
            print(f"    {fmt_avg(idx, m)}")
        print("  bottom (shortcut candidates):")
        for idx, m in ranked_avg[-10:]:
            print(f"    {fmt_avg(idx, m)}")


if __name__ == "__main__":
    main()
