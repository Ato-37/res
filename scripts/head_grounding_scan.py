"""Scan every individual (layer, head) pair for temporal grounding behavior,
instead of averaging over all heads or a whole layer (which the earlier
grounding_check.py showed is dominated by attention-sink heads that always
stare at the start of the video regardless of content).

This reuses the SAME output_attentions forward pass per sample -- extracting
per-head profiles is just slicing a tensor we already have, no extra model
calls -- so scanning all ~28x28 heads costs nothing beyond the 16 forward
passes grounding_check.py already does.

For each head we track two things across all samples:
  - hit_rate: how often that head's peak attention chunk overlaps the
    ground-truth grounding interval.
  - sink_fraction: how often that head's peak chunk is the SAME chunk
    (usually chunk 0), regardless of the sample/question. A head that always
    points at chunk 0 can still show a nonzero hit_rate purely by coincidence
    (some ground-truth intervals start near t=0) -- high sink_fraction is a
    warning that a high hit_rate is not real localization.

Usage:
    python scripts/head_grounding_scan.py --samples-file data/nextgqa_samples.json
"""

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def extract_frames(video_path, num_frames, max_dim):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        raise RuntimeError(f"Could not read frame count from {video_path}")

    indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    tmp_dir = tempfile.mkdtemp(prefix="head_grounding_frames_")
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
    raise RuntimeError("Could not find a video token id; check model.config or tokenizer special tokens")


def video_span(input_ids, video_token_id):
    positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise RuntimeError("No video tokens found in input_ids")
    return int(positions[0]), int(positions[-1]) + 1


def overlaps(a_start, a_end, intervals):
    return any(a_start < e and s < a_end for s, e in intervals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--heads", default=None,
                         help="Comma-separated layer:head pairs to validate specifically, e.g. '0:0,4:12,8:24,1:7' "
                              "(for testing candidates found on a different sample set, to check they're not just noise)")
    args = parser.parse_args()
    head_filter = None
    if args.heads:
        head_filter = [tuple(int(x) for x in pair.split(":")) for pair in args.heads.split(",")]

    samples = json.load(open(args.samples_file))
    for s in samples:
        s["video"] = os.path.expanduser(s["video"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, quant={args.quant}, n_samples={len(samples)}")

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

    n_layers = len(model.model.language_model.layers)
    num_heads = None  # discovered from the first sample's attention shape
    hits = None
    peak_counts = None
    n_run = 0
    random_baseline_total = 0.0

    for i, s in enumerate(samples):
        print(f"sample {i}: {s['question'][:60]!r}...")
        try:
            frame_paths, tmp_dir, timestamps = extract_frames(s["video"], args.num_frames, args.frame_size)
        except Exception as e:
            print(f"  WARNING: could not read video {s['video']!r} ({e}), skipping sample")
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

            v_start, v_end = video_span(inputs["input_ids"], video_token_id)
            span_len = v_end - v_start
            grid_thw = inputs.get("video_grid_thw")
            t = int(grid_thw[0, 0]) if grid_thw is not None else args.num_frames // 2
            if span_len % t != 0:
                print(f"  WARNING: span_len={span_len} not divisible by t={t}, skipping sample")
                continue
            chunk_size = span_len // t
            frames_per_chunk = max(len(timestamps) // t, 1)

            # chance baseline: if a chunk were picked uniformly at random,
            # what fraction of the time would it overlap ground truth?
            n_overlapping_chunks = sum(
                1 for c in range(t)
                if overlaps(timestamps[c * frames_per_chunk],
                            timestamps[min((c + 1) * frames_per_chunk, len(timestamps)) - 1],
                            s["grounding"])
            )
            random_baseline_total += n_overlapping_chunks / t

            with torch.no_grad():
                out = model(**inputs, output_attentions=True)

            if num_heads is None:
                num_heads = out.attentions[0].shape[1]
                hits = torch.zeros(n_layers, num_heads)
                peak_counts = [[Counter() for _ in range(num_heads)] for _ in range(n_layers)]

            layers_to_scan = range(n_layers) if head_filter is None else sorted({li for li, _ in head_filter})
            for li in layers_to_scan:
                layer_attn = out.attentions[li]
                heads_here = range(num_heads) if head_filter is None else [h for l2, h in head_filter if l2 == li]
                attn_span = layer_attn[0, :, -1, v_start:v_end].float()  # [heads, span_len]
                per_chunk = attn_span.view(num_heads, t, chunk_size).sum(dim=2)  # [heads, t]
                peak = per_chunk.argmax(dim=1)  # [heads]
                for h in heads_here:
                    pc = int(peak[h].item())
                    peak_counts[li][h][pc] += 1
                    lo = timestamps[pc * frames_per_chunk]
                    hi = timestamps[min((pc + 1) * frames_per_chunk, len(timestamps)) - 1]
                    if overlaps(lo, hi, s["grounding"]):
                        hits[li, h] += 1
            n_run += 1
        except Exception as e:
            print(f"  WARNING: error processing sample ({e}), skipping")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    hit_rate = hits / n_run
    random_baseline = random_baseline_total / n_run
    scan_pairs = head_filter if head_filter else [(li, h) for li in range(n_layers) for h in range(num_heads)]
    rows = []
    for li, h in scan_pairs:
        counts = peak_counts[li][h]
        sink_fraction = max(counts.values()) / n_run
        n_distinct_chunks = len(counts)
        rows.append((li, h, hit_rate[li, h].item(), sink_fraction, n_distinct_chunks))

    print(f"\n=== ran {n_run} samples, scanned {len(scan_pairs)} head(s) ===")
    print(f"random-chunk baseline (chance level for THIS sample set) = {random_baseline:.2%}\n")

    rows.sort(key=lambda r: (-r[2], r[3]))  # highest hit_rate first, then lowest sink_fraction
    label = "Specified heads" if head_filter else f"Top {args.top_n} heads by hit_rate"
    print(f"{label} (tie-broken by low sink_fraction = not just always picking one chunk):")
    print(f"{'layer':>5} {'head':>4} {'hit_rate':>9} {'sink_fraction':>13} {'distinct_chunks':>16}")
    for li, h, hr, sink, nd in rows[:args.top_n]:
        flag = "  <- likely just a sink head" if sink >= 0.85 else ""
        print(f"{li:>5} {h:>4} {hr:>9.2%} {sink:>13.2%} {nd:>16}{flag}")


if __name__ == "__main__":
    main()
