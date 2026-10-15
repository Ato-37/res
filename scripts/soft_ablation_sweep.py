"""Measure continuous attention-head attenuation (soft ablation).

Unlike hard ablation, this script scales one head's input to ``o_proj`` by
an arbitrary ``alpha`` and records a dose-response curve.  The main outcome
is the change in the correct answer token's log-probability relative to the
unmodified input.  A matched control token, entropy, and the model's top
token are recorded to help distinguish selective effects from generic
distribution collapse.

The default setup compares the previously screened candidate layer=6 with
layer=12 as a pre-specified control, but ``--layers`` can select any layers.
This is an evaluation/diagnosis script; it does not update model weights.

Usage:
    python scripts/soft_ablation_sweep.py \
        --samples-file data/nextgqa_samples_big_heldout.json \
        --n-samples 2 --layers 6,12 \
        --scales 1,0.75,0.5,0.25,0.1,0.05,0.01,0 \
    --metric sequence_logprob --num-controls 5 \
        --output results/soft_ablation_smoke.json
"""

import argparse
import json
import math
import os
import platform
import shutil
import tempfile

import cv2
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
import transformers


HEAD_DIM = 128


from closed_research_branch import reject_inference

def extract_frames(video_path, num_frames, max_dim):
    # Some legacy videos (notably VP6F) report a frame count that is slightly
    # larger than the number of frames OpenCV can actually decode. Count the
    # decodable frames first, then sample from that observed count instead of
    # trusting CAP_PROP_FRAME_COUNT for the final frame.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    decoded_total = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded_total += 1
    finally:
        cap.release()
    if decoded_total <= 0:
        raise RuntimeError(f"Could not decode any frames from {video_path}")

    indices = [
        round(i * (decoded_total - 1) / max(num_frames - 1, 1))
        for i in range(num_frames)
    ]
    tmp_dir = tempfile.mkdtemp(prefix="soft_ablation_frames_")
    frame_paths = []
    target_positions = set(indices)
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen video {video_path}")
        # Decode sequentially instead of seeking to each frame. This is
        # reliable for legacy codecs and uses the observed decodable length.
        for pos in range(decoded_total):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not sequentially read frame {pos} from {video_path}")
            if pos not in target_positions:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image.thumbnail((max_dim, max_dim))
            path = os.path.join(tmp_dir, f"frame_{len(frame_paths):03d}.jpg")
            image.save(path)
            frame_paths.append(path)
            if len(frame_paths) == len(indices):
                break
        if len(frame_paths) != len(indices):
            raise RuntimeError(
                f"Decoded {len(frame_paths)} of {len(indices)} requested frames from {video_path}"
            )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        cap.release()
    return frame_paths, tmp_dir


def attenuation_hook(head_idx, scale):
    def hook(module, args):
        x = args[0].clone()
        start = head_idx * HEAD_DIM
        x[..., start:start + HEAD_DIM] *= scale
        return (x,) + args[1:]
    return hook


def append_answer_tokens(inputs, answer_ids):
    """Return prompt inputs with the answer tokens appended for teacher forcing."""
    augmented = dict(inputs)
    answer = torch.tensor(answer_ids, dtype=inputs["input_ids"].dtype,
                          device=inputs["input_ids"].device).unsqueeze(0)
    augmented["input_ids"] = torch.cat([inputs["input_ids"], answer], dim=1)
    if "attention_mask" in inputs:
        extra_mask = torch.ones_like(answer, dtype=inputs["attention_mask"].dtype)
        augmented["attention_mask"] = torch.cat([inputs["attention_mask"], extra_mask], dim=1)
    # Recent Transformers versions pass Qwen2.5-VL's multimodal token-type
    # sequence to get_rope_index. Answer tokens are ordinary text tokens, so
    # append type 0; leaving this tensor at the prompt length causes a mask
    # shape mismatch inside get_rope_index.
    for key in ("mm_token_type_ids", "input_token_type_ids", "token_type_ids"):
        if key in inputs:
            extra_types = torch.zeros_like(answer, dtype=inputs[key].dtype)
            augmented[key] = torch.cat([inputs[key], extra_types], dim=1)
    return augmented


def sequence_logprob_from_output(output, sequence_len, answer_ids):
    answer_logprobs = sequence_answer_logprobs(output, sequence_len, answer_ids)
    return sum(
        answer_logprobs[pos, token_id].item()
        for pos, token_id in enumerate(answer_ids)
    )


def sequence_answer_logprobs(output, sequence_len, answer_ids):
    start = sequence_len - len(answer_ids) - 1
    answer_logits = output.logits[0, start:start + len(answer_ids), :].float()
    if answer_logits.shape[0] != len(answer_ids):
        raise RuntimeError(
            f"Expected {len(answer_ids)} answer logits, got {answer_logits.shape[0]}"
        )
    return torch.log_softmax(answer_logits, dim=-1)


def forward_metrics(model, prompt_inputs, answer_ids, top_id, control_ids, metric):
    inputs = prompt_inputs
    if metric == "sequence_logprob":
        inputs = append_answer_tokens(prompt_inputs, answer_ids)
    with torch.no_grad():
        output = model(**inputs)
    if metric == "sequence_logprob":
        prompt_len = prompt_inputs["input_ids"].shape[1]
        first_logits = output.logits[0, prompt_len - 1, :].float()
        answer_logprobs = sequence_answer_logprobs(
            output, inputs["input_ids"].shape[1], answer_ids
        )
        correct_logprob = sum(
            answer_logprobs[pos, token_id].item()
            for pos, token_id in enumerate(answer_ids)
        )
    else:
        first_logits = output.logits[0, -1, :].float()
        correct_logprob = torch.log_softmax(first_logits, dim=-1)[answer_ids[0]].item()
    first_logprobs = torch.log_softmax(first_logits, dim=-1)
    probs = first_logprobs.exp()
    entropy = -(probs * first_logprobs).sum().item()
    if metric == "sequence_logprob":
        # Each answer position has its own matched controls, selected under
        # the same correct-answer prefix. This avoids changing the prefix and
        # making the suffix likelihood incomparable to the correct sequence.
        control_logprobs = [
            sum(answer_logprobs[pos, control_ids[pos][rank]].item()
                for pos in range(len(answer_ids)))
            for rank in range(len(control_ids[0]))
        ]
    else:
        control_logprobs = [first_logprobs[token_id].item() for token_id in control_ids]
    return {
        "correct_logprob": correct_logprob,
        "first_token_correct_logprob": first_logprobs[answer_ids[0]].item(),
        "top_logprob": first_logprobs[top_id].item(),
        "control_logprobs": control_logprobs,
        "control_logprob": sum(control_logprobs) / len(control_logprobs),
        "entropy": entropy,
        "top_token_id": int(first_logits.argmax().item()),
    }


def valid_word_mask(tokenizer, vocab_size):
    valid = torch.zeros(vocab_size, dtype=torch.bool)
    for token_id in range(vocab_size):
        text = tokenizer.decode([token_id]).strip()
        if text and "�" not in text and text.isalpha() and len(text) >= 2:
            valid[token_id] = True
    return valid


def choose_controls(logprobs, answer_id, top_id, valid_mask, num_controls):
    distances = (logprobs - logprobs[answer_id]).abs().clone()
    distances[~valid_mask] = float("inf")
    distances[answer_id] = float("inf")
    distances[top_id] = float("inf")
    return [int(token_id) for token_id in torch.argsort(distances)[:num_controls].tolist()]


def choose_position_controls(answer_logprobs, answer_ids, valid_mask, num_controls):
    controls = []
    for position, answer_id in enumerate(answer_ids):
        logprobs = answer_logprobs[position]
        top_id = int(logprobs.argmax().item())
        controls.append(choose_controls(
            logprobs, answer_id, top_id, valid_mask, num_controls
        ))
    return controls


def parse_scales(value):
    scales = [float(x) for x in value.split(",")]
    if not scales or any(not math.isfinite(x) or x < 0 for x in scales):
        raise argparse.ArgumentTypeError("scales must be finite non-negative numbers")
    if any(x > 1 for x in scales):
        raise argparse.ArgumentTypeError("scales above 1 are not attenuation; use a separate amplification run")
    return scales


def sample_id(sample, index):
    return str(sample.get("qid", sample.get("video_id", index)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--layers", default="6,12", help="Comma-separated layer indices")
    parser.add_argument("--head", type=int, default=16)
    parser.add_argument("--scales", type=parse_scales,
                        default=parse_scales("1,0.75,0.5,0.25,0.1,0.05,0.01,0"))
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--metric", choices=["first_token", "sequence_logprob"],
                        default="first_token",
                        help="Evaluate the first answer token or the full answer sequence")
    parser.add_argument("--num-controls", type=int, default=5,
                        help="Number of matched first-token control tokens")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--seed", type=int, default=0,
                        help="Torch seed recorded for reproducibility (forward pass is eval-mode)")
    args = parser.parse_args()
    reject_inference(parser, 'legacy_soft_ablation',
                     'retired head/layer/alpha search is closed')

    if args.num_controls < 1:
        raise ValueError("--num-controls must be at least 1")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    layers_to_run = [int(x) for x in args.layers.split(",")]
    samples = json.load(open(args.samples_file))
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    for sample in samples:
        sample["video"] = os.path.expanduser(sample["video"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, quant={args.quant}, layers={layers_to_run}, head={args.head}")
    print(f"scales={','.join(str(x) for x in args.scales)}")

    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif args.quant == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        quantization_config=quant_config,
        device_map=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels
    )
    layers = model.model.language_model.layers
    for layer_idx in layers_to_run:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"layer {layer_idx} is outside 0..{len(layers) - 1}")
    num_heads = layers[layers_to_run[0]].self_attn.o_proj.in_features // HEAD_DIM
    if args.head < 0 or args.head >= num_heads:
        raise ValueError(f"head {args.head} is outside 0..{num_heads - 1}")

    vocab_size = model.get_output_embeddings().weight.shape[0]
    print(f"building valid-word mask over vocab_size={vocab_size}...")
    word_mask = valid_word_mask(processor.tokenizer, vocab_size)

    prepared = []
    failures = []
    for index, sample in enumerate(samples):
        sid = sample_id(sample, index)
        try:
            frame_paths, tmp_dir = extract_frames(sample["video"], args.num_frames, args.frame_size)
            messages = [{"role": "user", "content": [
                {"type": "video", "video": frame_paths},
                {"type": "text", "text": sample["question"]},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(device)
            answer_ids = processor.tokenizer(
                sample["answer"], add_special_tokens=False
            ).input_ids
            if not answer_ids:
                raise ValueError(f"Answer has no tokens for sample {sid}")
            prepared.append({
                "id": sid,
                "inputs": inputs,
                "answer_ids": answer_ids,
                "frame_paths": frame_paths,
                "tmp_dir": tmp_dir,
            })
        except Exception as exc:
            failures.append({"id": sid, "video": sample.get("video"), "error": repr(exc)})
            print(f"WARNING: skipping sample {sid}: {exc}")

    if not prepared:
        raise RuntimeError("No samples could be prepared")

    records = []
    for item in prepared:
        baseline_logits = None
        with torch.no_grad():
            baseline_output = model(**item["inputs"])
        baseline_logits = baseline_output.logits[0, -1, :].float()
        baseline_logprobs = torch.log_softmax(baseline_logits, dim=-1)
        answer_ids = item["answer_ids"]
        answer_id = answer_ids[0]
        top_id = int(baseline_logits.argmax().item())
        if args.metric == "sequence_logprob":
            baseline_sequence_inputs = append_answer_tokens(item["inputs"], answer_ids)
            with torch.no_grad():
                baseline_sequence_output = model(**baseline_sequence_inputs)
            baseline_answer_logprobs = sequence_answer_logprobs(
                baseline_sequence_output,
                baseline_sequence_inputs["input_ids"].shape[1],
                answer_ids,
            )
            control_ids = choose_position_controls(
                baseline_answer_logprobs, answer_ids, word_mask, args.num_controls
            )
        else:
            control_ids = choose_controls(
                baseline_logprobs, answer_id, top_id, word_mask, args.num_controls
            )
        baseline = forward_metrics(
            model, item["inputs"], answer_ids, top_id, control_ids, args.metric
        )
        print(f"sample={item['id']} baseline metric={args.metric} "
              f"correct={baseline['correct_logprob']:.4f} "
              f"first_token={baseline['first_token_correct_logprob']:.4f} "
              f"top={baseline['top_logprob']:.4f} "
              f"control_mean={baseline['control_logprob']:.4f}")

        for layer_idx in layers_to_run:
            layer = layers[layer_idx]
            for scale in args.scales:
                handle = None
                if scale != 1.0:
                    handle = layer.self_attn.o_proj.register_forward_pre_hook(
                        attenuation_hook(args.head, scale)
                    )
                try:
                    metrics = forward_metrics(
                        model, item["inputs"], answer_ids, top_id, control_ids, args.metric
                    )
                finally:
                    if handle is not None:
                        handle.remove()
                record = {
                    "sample_id": item["id"],
                    "layer": layer_idx,
                    "head": args.head,
                    "scale": scale,
                    "metric": args.metric,
                    "num_controls": args.num_controls,
                    "correct_delta": metrics["correct_logprob"] - baseline["correct_logprob"],
                    "first_token_correct_delta": (
                        metrics["first_token_correct_logprob"]
                        - baseline["first_token_correct_logprob"]
                    ),
                    "top_delta": metrics["top_logprob"] - baseline["top_logprob"],
                    "control_delta": metrics["control_logprob"] - baseline["control_logprob"],
                    "control_deltas": [
                        current - initial
                        for current, initial in zip(
                            metrics["control_logprobs"], baseline["control_logprobs"]
                        )
                    ],
                    "entropy_delta": metrics["entropy"] - baseline["entropy"],
                    "baseline": baseline,
                    "metrics": metrics,
                }
                records.append(record)

    for item in prepared:
        shutil.rmtree(item["tmp_dir"], ignore_errors=True)

    summaries = []
    for layer_idx in layers_to_run:
        for scale in args.scales:
            subset = [r for r in records if r["layer"] == layer_idx and r["scale"] == scale]
            values = [r["correct_delta"] for r in subset]
            summaries.append({
                "layer": layer_idx,
                "head": args.head,
                "scale": scale,
                "n": len(values),
                "mean_correct_delta": sum(values) / len(values),
                "sd_correct_delta": (
                    sum((x - sum(values) / len(values)) ** 2 for x in values) / (len(values) - 1)
                ) ** 0.5 if len(values) > 1 else None,
                "mean_control_delta": sum(r["control_delta"] for r in subset) / len(subset),
                "mean_selective_delta": sum(
                    r["correct_delta"] - r["control_delta"] for r in subset
                ) / len(subset),
                "mean_entropy_delta": sum(r["entropy_delta"] for r in subset) / len(subset),
            })
    print(f"\n=== mean {args.metric} delta relative to alpha=1 ===")
    for summary in summaries:
        print(f"layer={summary['layer']:>2} scale={summary['scale']:>5}: "
              f"correct={summary['mean_correct_delta']:+.4f} n={summary['n']} "
              f"control={summary['mean_control_delta']:+.4f} "
              f"entropy={summary['mean_entropy_delta']:+.4f}")

    result = {
        "script": "soft_ablation_sweep.py",
        "samples_file": args.samples_file,
        "model": args.model,
        "quant": args.quant,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "layers": layers_to_run,
        "head": args.head,
        "scales": args.scales,
        "metric": args.metric,
        "num_controls": args.num_controls,
        "seed": args.seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "opencv_version": cv2.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_frames": args.num_frames,
        "n_requested": len(samples),
        "n_effective": len(prepared),
        "failures": failures,
        "summaries": summaries,
        "records": records,
    }
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as file:
            json.dump(result, file, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
