"""First activation-steering (correction) pilot. Instead of zeroing the
candidate grounding head (layer=6, head=16 by default -- the one head that
replicated across two independent sample sets in head_grounding_scan.py), we
AMPLIFY its contribution by a scale factor and see whether:

  1. the correct answer's token log-probability goes up (mirrors the
     ablation experiment's correct_drop metric, but testing the opposite
     direction: does boosting a validated head help rather than hurt?)
  2. the model's overall (all-head-averaged) attention-based grounding hit
     rate improves -- does amplifying one head shift the WHOLE model's
     effective behavior toward the right video segment, not just that
     head's own narrow view of it?

This is a minimal causal test: if scaling this head's output shifts model
behavior in the predicted direction (and does so more as the scale
increases), that is real evidence of a causal, not just correlational, role.
A steering vector framed inside CRA-GQA's front-door/back-door formulation is
a later refinement -- this spike checks the simplest possible version first.

Usage:
    python scripts/activation_steering_spike.py --samples-file data/nextgqa_samples_big_heldout.json
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

HEAD_DIM = 128


def extract_frames(video_path, num_frames, max_dim):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        raise RuntimeError(f"Could not read frame count from {video_path}")
    indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    tmp_dir = tempfile.mkdtemp(prefix="steering_frames_")
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
    timestamps = [idx / fps for idx in indices]
    return frame_paths, tmp_dir, timestamps


def find_video_token_id(model, processor):
    tid = getattr(model.config, "video_token_id", None)
    if tid is not None:
        return tid
    for name in ["<|video_pad|>", "<|video_placeholder|>"]:
        tid = processor.tokenizer.convert_tokens_to_ids(name)
        if tid is not None and tid != processor.tokenizer.unk_token_id:
            return tid
    raise RuntimeError("Could not find a video token id")


def video_span(input_ids, video_token_id):
    positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise RuntimeError("No video tokens found in input_ids")
    return int(positions[0]), int(positions[-1]) + 1


def overlaps(a_start, a_end, intervals):
    return any(a_start < e and s < a_end for s, e in intervals)


def scale_head_hook(head_idx, head_dim, scale):
    def hook(module, args):
        x = args[0].clone()
        x[..., head_idx * head_dim:(head_idx + 1) * head_dim] *= scale
        return (x,) + args[1:]
    return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--head", type=int, default=16)
    parser.add_argument("--scales", default="1,2,4,8")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    args = parser.parse_args()
    scales = [float(x) for x in args.scales.split(",")]

    samples = json.load(open(args.samples_file))
    for s in samples:
        s["video"] = os.path.expanduser(s["video"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, quant={args.quant}, target=layer{args.layer}/head{args.head}, scales={scales}")

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
    video_token_id = find_video_token_id(model, processor)
    layer = model.model.language_model.layers[args.layer]

    results = {scale: {"correct_logprob_sum": 0.0, "hits": 0} for scale in scales}
    n_run = 0

    for i, s in enumerate(samples):
        print(f"sample {i}: {s['question'][:60]!r}...")
        try:
            frame_paths, tmp_dir, timestamps = extract_frames(s["video"], args.num_frames, args.frame_size)
        except Exception as e:
            print(f"  WARNING: could not read video ({e}), skipping")
            continue
        try:
            messages = [{"role": "user", "content": [
                {"type": "video", "video": frame_paths},
                {"type": "text", "text": s["question"]},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
            ).to(device)

            answer_token_id = processor.tokenizer(s["answer"], add_special_tokens=False).input_ids[0]
            v_start, v_end = video_span(inputs["input_ids"], video_token_id)
            span_len = v_end - v_start
            grid_thw = inputs.get("video_grid_thw")
            t = int(grid_thw[0, 0]) if grid_thw is not None else args.num_frames // 2
            if span_len % t != 0:
                print(f"  WARNING: span_len={span_len} not divisible by t={t}, skipping")
                continue
            chunk_size = span_len // t
            frames_per_chunk = max(len(timestamps) // t, 1)

            for scale in scales:
                handle = None
                if scale != 1:
                    handle = layer.self_attn.o_proj.register_forward_pre_hook(
                        scale_head_hook(args.head, HEAD_DIM, scale))
                with torch.no_grad():
                    out = model(**inputs, output_attentions=True)
                if handle is not None:
                    handle.remove()

                logits = out.logits[0, -1, :].float()
                logprob = torch.log_softmax(logits, dim=-1)[answer_token_id].item()
                results[scale]["correct_logprob_sum"] += logprob

                last_pos_attn = torch.stack([la[0, :, -1, v_start:v_end] for la in out.attentions])
                per_token = last_pos_attn.float().mean(dim=(0, 1))
                per_chunk = per_token.view(t, chunk_size).sum(dim=1)
                peak_chunk = int(per_chunk.argmax().item())
                lo = timestamps[peak_chunk * frames_per_chunk]
                hi = timestamps[min((peak_chunk + 1) * frames_per_chunk, len(timestamps)) - 1]
                if overlaps(lo, hi, s["grounding"]):
                    results[scale]["hits"] += 1

            n_run += 1
        except Exception as e:
            print(f"  WARNING: error processing sample ({e}), skipping")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n=== ran {n_run} samples, steering layer={args.layer} head={args.head} ===")
    print(f"{'scale':>6} {'avg_correct_logprob':>20} {'grounding_hit_rate':>19}")
    for scale in scales:
        r = results[scale]
        avg_lp = r["correct_logprob_sum"] / n_run
        hr = r["hits"] / n_run
        print(f"{scale:>6} {avg_lp:>20.4f} {hr:>18.2%}")


if __name__ == "__main__":
    main()
