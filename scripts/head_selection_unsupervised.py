"""Unsupervised head selection for temporal grounding, following
LocalizationHeads (CVPR'25 Highlight)'s ACTUAL method -- not the
ground-truth hit_rate screening `head_grounding_scan.py` used earlier in
this project. LocalizationHeads selects heads purely from attention *shape*,
with no grounding labels touched at selection time:

  Criterion-1 (value-based elbow / chord method): sum each head's attention
    mass toward the video tokens, sort heads descending by this value, and
    find the "elbow" of that curve (the point of max perpendicular distance
    from the line joining the first and last points of the sorted curve).
    Keep only heads at or above the elbow (with a --min-keep floor, since a
    flat curve can give a degenerate elbow near the very start).

  Criterion-2 (spatial entropy in the original paper, lower = better): among
    the Criterion-1 survivors, prefer heads whose attention is most
    *concentrated* (low entropy). This project has no single-image spatial
    axis; the paper's own suggested video extension (see the LocalizationHeads
    summary artifact, "見直すべき点") is to substitute *temporal* entropy: the
    Shannon entropy of each head's attention distribution over the video's
    time chunks (already computed in `head_grounding_scan.py`'s per_chunk
    tensor, just missing an entropy calculation).

NExT-GQA's ground-truth grounding intervals are used ONLY afterward, to
evaluate whether the unsupervised top candidates actually localize
correctly against a separate --eval-samples-file -- never to select them.
This is the key methodological fix versus the earlier ground-truth-hit-rate
based selection, which risked overfitting to the screening set even with a
held-out check (selecting AND evaluating both used labels; here selection
uses none).

Also tracks `sink_fraction` per head (same diagnostic as
`head_grounding_scan.py`): how often a head's peak attention chunk is the
SAME chunk regardless of the sample/question. A head can score well on both
Criterion-1 (high magnitude) and Criterion-2 (low entropy) simply by always
staring at one fixed chunk (e.g. the video's start) -- that is an attention
sink, not real temporal localization, and low entropy alone cannot tell the
two apart. This was added after the 2026-09-30 run showed layer=9 heads
dominating the unsupervised top-20 despite none of them beating the random
grounding baseline (see experiments/2026-09-30_unsupervised_head_selection.md).

Usage:
    python scripts/head_selection_unsupervised.py \
        --samples-file data/nextgqa_samples_big.json \
        --eval-samples-file data/nextgqa_samples_big_heldout.json
"""

import argparse
import json
import math
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
    tmp_dir = tempfile.mkdtemp(prefix="head_selection_frames_")
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


def elbow_index(values_desc, min_keep):
    """Chord method: find the index of max perpendicular distance from the
    line joining the first and last points of the sorted-descending curve.
    Returns how many heads to keep (at least min_keep)."""
    n = len(values_desc)
    x1, y1 = 0, values_desc[0]
    x2, y2 = n - 1, values_desc[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    best_i, best_dist = 0, -1.0
    for i, y0 in enumerate(values_desc):
        dist = abs(dy * i - dx * y0 + x2 * y1 - y2 * x1) / norm if norm > 0 else 0.0
        if dist > best_dist:
            best_dist, best_i = dist, i
    return min(max(best_i + 1, min_keep), n)


def load_model(args, device):
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
    return model, processor


def scan_criteria(samples_path, model, processor, video_token_id, device, args):
    """Unsupervised pass: for every (layer, head), accumulate attention mass
    into the video span (Criterion-1 input) and temporal entropy of the
    per-chunk distribution (Criterion-2 input). No grounding labels used."""
    samples = json.load(open(samples_path))
    for s in samples:
        s["video"] = os.path.expanduser(s["video"])

    n_layers = len(model.model.language_model.layers)
    num_heads = None
    magnitude_sum = None
    entropy_sum = None
    peak_counts = None  # [layer][head] -> Counter of which chunk was the argmax, across samples
    n_run = 0

    for i, s in enumerate(samples):
        print(f"[criteria] sample {i}: {s['question'][:60]!r}...")
        try:
            frame_paths, tmp_dir, _ = extract_frames(s["video"], args.num_frames, args.frame_size)
        except Exception as e:
            print(f"  WARNING: could not read {s['video']!r} ({e}), skipping")
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
                print(f"  WARNING: span_len={span_len} not divisible by t={t}, skipping")
                continue
            chunk_size = span_len // t

            with torch.no_grad():
                out = model(**inputs, output_attentions=True)

            if num_heads is None:
                num_heads = out.attentions[0].shape[1]
                magnitude_sum = torch.zeros(n_layers, num_heads)
                entropy_sum = torch.zeros(n_layers, num_heads)
                peak_counts = [[Counter() for _ in range(num_heads)] for _ in range(n_layers)]

            for li in range(n_layers):
                attn_span = out.attentions[li][0, :, -1, v_start:v_end].float()  # [heads, span_len]
                per_chunk = attn_span.view(num_heads, t, chunk_size).sum(dim=2)  # [heads, t]
                mass = per_chunk.sum(dim=1)  # [heads] -- total attention mass into the video span
                probs = per_chunk / mass.clamp_min(1e-12).unsqueeze(1)
                ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
                magnitude_sum[li] += mass.cpu()  # accumulators are CPU tensors; attn tensors are on `device`
                entropy_sum[li] += ent.cpu()
                peak = per_chunk.argmax(dim=1).cpu()
                for h in range(num_heads):
                    peak_counts[li][h][int(peak[h].item())] += 1
            n_run += 1
        except Exception as e:
            print(f"  WARNING: error processing sample ({e}), skipping")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if n_run == 0:
        raise RuntimeError("No samples could be processed for criteria computation")
    sink_fraction = torch.zeros(n_layers, num_heads)
    for li in range(n_layers):
        for h in range(num_heads):
            sink_fraction[li, h] = max(peak_counts[li][h].values()) / n_run
    return magnitude_sum / n_run, entropy_sum / n_run, sink_fraction, n_run


def evaluate_hit_rate(samples_path, heads, model, processor, video_token_id, device, args):
    """Evaluation-only pass: for a fixed list of (layer, head) pairs, compute
    hit_rate against NExT-GQA ground-truth grounding intervals. Ground truth
    is used here and only here -- never during scan_criteria above.

    Also computes a FIXED-CHUNK baseline (always predict chunk c, for each
    c) alongside the uniform-random baseline. NExT-GQA's grounding intervals
    are not necessarily uniformly distributed over a video's duration (e.g.
    many "what happened after X" questions may skew grounding toward later
    chunks) -- a head that just has some content-independent positional bias
    (attends to "late" positions generally, not the specific answer-relevant
    moment) could beat the uniform-random baseline for the wrong reason. If
    the best fixed-chunk baseline is already close to a candidate head's
    hit_rate, that head's apparent grounding is not distinguishable from
    this positional artifact."""
    samples = json.load(open(samples_path))
    for s in samples:
        s["video"] = os.path.expanduser(s["video"])

    layers_needed = sorted({li for li, _ in heads})
    hits = {pair: 0 for pair in heads}
    n_run = 0
    random_baseline_total = 0.0
    fixed_chunk_hits = Counter()
    fixed_chunk_totals = Counter()

    for i, s in enumerate(samples):
        print(f"[eval] sample {i}: {s['question'][:60]!r}...")
        try:
            frame_paths, tmp_dir, timestamps = extract_frames(s["video"], args.num_frames, args.frame_size)
        except Exception as e:
            print(f"  WARNING: could not read {s['video']!r} ({e}), skipping")
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
                print(f"  WARNING: span_len={span_len} not divisible by t={t}, skipping")
                continue
            chunk_size = span_len // t
            frames_per_chunk = max(len(timestamps) // t, 1)

            n_overlapping_chunks = 0
            for c in range(t):
                lo = timestamps[c * frames_per_chunk]
                hi = timestamps[min((c + 1) * frames_per_chunk, len(timestamps)) - 1]
                hit = overlaps(lo, hi, s["grounding"])
                n_overlapping_chunks += int(hit)
                fixed_chunk_totals[c] += 1
                if hit:
                    fixed_chunk_hits[c] += 1
            random_baseline_total += n_overlapping_chunks / t

            with torch.no_grad():
                out = model(**inputs, output_attentions=True)

            for li in layers_needed:
                attn_span = out.attentions[li][0, :, -1, v_start:v_end].float()
                num_heads = attn_span.shape[0]
                per_chunk = attn_span.view(num_heads, t, chunk_size).sum(dim=2)
                peak = per_chunk.argmax(dim=1)
                for pair in heads:
                    l2, h = pair
                    if l2 != li:
                        continue
                    pc = int(peak[h].item())
                    lo = timestamps[pc * frames_per_chunk]
                    hi = timestamps[min((pc + 1) * frames_per_chunk, len(timestamps)) - 1]
                    if overlaps(lo, hi, s["grounding"]):
                        hits[pair] += 1
            n_run += 1
        except Exception as e:
            print(f"  WARNING: error processing sample ({e}), skipping")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if n_run == 0:
        raise RuntimeError("No samples could be processed for evaluation")
    random_baseline = random_baseline_total / n_run
    fixed_chunk_baselines = {c: fixed_chunk_hits[c] / fixed_chunk_totals[c] for c in fixed_chunk_totals}
    return {pair: hits[pair] / n_run for pair in heads}, random_baseline, fixed_chunk_baselines, n_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True,
                         help="Samples used to compute the unsupervised criteria (no grounding labels touched here)")
    parser.add_argument("--eval-samples-file", default=None,
                         help="Optional separate samples file to evaluate the selected heads' hit_rate "
                              "against ground truth -- evaluation only, never used for selection")
    parser.add_argument("--min-keep", type=int, default=10,
                         help="Floor on how many heads Criterion-1's elbow cutoff must keep")
    parser.add_argument("--max-sink-fraction", type=float, default=0.5,
                         help="Exclude Criterion-1 survivors whose peak attention chunk is the same in at "
                              "least this fraction of samples, before ranking by entropy. LocalizationHeads' "
                              "own paper combines its entropy criterion with an analogous extra filter "
                              "(excluding a degenerate 'bottom-row focus' pattern in images) for the same "
                              "reason: entropy alone cannot tell a content-independent attention sink apart "
                              "from genuine concentration. Set to 1.0 to disable this filter.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, quant={args.quant}, min_keep={args.min_keep}")

    model, processor = load_model(args, device)
    video_token_id = find_video_token_id(model, processor)

    magnitude, entropy, sink_fraction, n_run = scan_criteria(
        args.samples_file, model, processor, video_token_id, device, args)
    n_layers, num_heads = magnitude.shape
    print(f"\n=== criteria computed from {n_run} samples in {args.samples_file} (no ground truth used) ===")

    flat = [(li, h, magnitude[li, h].item()) for li in range(n_layers) for h in range(num_heads)]
    flat.sort(key=lambda r: -r[2])
    values_desc = [v for _, _, v in flat]
    k = elbow_index(values_desc, args.min_keep)
    survivors = flat[:k]
    print(f"Criterion-1 (value-based elbow/chord on attention magnitude): kept {k} of {len(flat)} heads")

    non_sink_survivors = [(li, h, mag) for li, h, mag in survivors
                           if sink_fraction[li, h].item() < args.max_sink_fraction]
    n_excluded = len(survivors) - len(non_sink_survivors)
    print(f"Sink filter (sink_fraction >= {args.max_sink_fraction:.0%}): excluded {n_excluded} of "
          f"{len(survivors)} Criterion-1 survivors before ranking by entropy")

    survivors_ranked = sorted(
        ((li, h, mag, entropy[li, h].item(), sink_fraction[li, h].item()) for li, h, mag in non_sink_survivors),
        key=lambda r: r[3],  # ascending temporal entropy = more concentrated = better
    )

    print(f"\nTop {min(args.top_n, len(survivors_ranked))} unsupervised candidates "
          f"(Criterion-1 survivors, ranked by Criterion-2 temporal entropy ascending):")
    print(f"{'rank':>4} {'layer':>5} {'head':>4} {'attn_mag':>9} {'temporal_entropy':>17} {'sink_fraction':>13}")
    top_candidates = survivors_ranked[:args.top_n]
    for rank, (li, h, mag, ent, sink) in enumerate(top_candidates, 1):
        flags = []
        if (li, h) == (6, 16):
            flags.append("layer6/head16 (previous ground-truth-based candidate)")
        if sink >= 0.85:
            flags.append("likely just a sink head")
        flag = "  <- " + ", ".join(flags) if flags else ""
        print(f"{rank:>4} {li:>5} {h:>4} {mag:>9.4f} {ent:>17.4f} {sink:>13.2%}{flag}")

    n_sink_in_top = sum(1 for *_, sink in top_candidates if sink >= 0.85)
    if n_sink_in_top:
        print(f"\n{n_sink_in_top} of the top {len(top_candidates)} candidates have sink_fraction >= 85% "
              f"(their peak chunk is the same regardless of sample/question -- likely attention sinks, "
              f"not real temporal localization; low entropy alone could not tell them apart from genuine "
              f"concentration).")

    rank_among_survivors = next((i + 1 for i, r in enumerate(survivors_ranked) if (r[0], r[1]) == (6, 16)), None)
    if rank_among_survivors:
        print(f"\nlayer=6/head=16 rank among {len(survivors_ranked)} Criterion-1 survivors "
              f"by temporal entropy: {rank_among_survivors}")
    else:
        li6_mag = magnitude[6, 16].item()
        rank_all = next(i + 1 for i, (li, h, _) in enumerate(flat) if (li, h) == (6, 16))
        print(f"\nlayer=6/head=16 did NOT survive Criterion-1's elbow cutoff "
              f"(rank {rank_all} of {len(flat)} by attention magnitude={li6_mag:.4f}, cutoff was top {k})")

    if args.eval_samples_file:
        eval_heads = [(li, h) for li, h, _, _, _ in top_candidates]
        if (6, 16) not in eval_heads:
            eval_heads.append((6, 16))
        hit_rates, random_baseline, fixed_chunk_baselines, n_eval = evaluate_hit_rate(
            args.eval_samples_file, eval_heads, model, processor, video_token_id, device, args)
        print(f"\n=== evaluation on {args.eval_samples_file} (n={n_eval}, ground truth used HERE ONLY) ===")
        print(f"random-chunk (uniform) baseline = {random_baseline:.2%}")
        best_fixed_c, best_fixed_rate = max(fixed_chunk_baselines.items(), key=lambda kv: kv[1])
        print(f"best FIXED-chunk baseline (always predict chunk {best_fixed_c}, ignoring content) "
              f"= {best_fixed_rate:.2%}")
        print("  " + ", ".join(f"chunk={c}: {r:.2%}" for c, r in sorted(fixed_chunk_baselines.items())))
        print("  (if a head's hit_rate below is close to this best-fixed-chunk number, its apparent "
              "grounding may just be a content-independent positional bias, not real localization)\n")
        print(f"{'layer':>5} {'head':>4} {'hit_rate':>9}")
        for li, h in sorted(eval_heads, key=lambda p: -hit_rates[p]):
            flag = "  <- previous ground-truth-based candidate" if (li, h) == (6, 16) else ""
            hr = hit_rates[(li, h)]
            if hr <= best_fixed_rate + 1e-9:
                flag += "  <- does not beat the best fixed-chunk baseline"
            print(f"{li:>5} {h:>4} {hr:>9.2%}{flag}")


if __name__ == "__main__":
    main()
