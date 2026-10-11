"""Retroactive sanity check: how much of any attention-based grounding
hit_rate result in this project could be explained by a purely
content-independent positional bias in NExT-GQA's grounding intervals,
rather than real model behavior?

Motivation (2026-09-30): `head_selection_unsupervised.py`'s evaluation
showed that on `data/nextgqa_samples_big_heldout.json`, just always
predicting a fixed chunk (chunk=2 of 4) -- ignoring the video and question
entirely -- already hits the ground-truth grounding interval 50.00% of the
time, well above the uniform-random baseline (36.36%) and *above*
layer=6/head=16's own attention-based hit_rate (45.45%) on that same file.
This calls every earlier "head X beats the random baseline" claim in this
project into question, since none of them were checked against this
stronger, position-only baseline. This script recomputes it for any
existing samples file, using ONLY video duration/fps (via OpenCV metadata,
no frame decoding, no model, no GPU) -- fast enough to re-check every
samples file used so far.

Usage:
    python scripts/chunk_baseline_check.py --samples-file data/nextgqa_samples_big.json
    python scripts/chunk_baseline_check.py --samples-file data/nextgqa_samples_big_heldout.json
    python scripts/chunk_baseline_check.py --samples-file data/nextgqa_samples.json
    python scripts/chunk_baseline_check.py --samples-file data/nextgqa_samples_heldout.json

Note: this approximates the number of temporal chunks as num_frames // 2,
matching Qwen2.5-VL's observed 2x temporal patch merge (t = grid_thw[0,0]
was consistently num_frames // 2 in every run so far in this project). If a
particular video's actual t differs from this, this script's chunk
boundaries won't exactly match that sample's real evaluation -- treat this
as an approximate diagnostic, not a byte-for-byte replay of the model runs.
"""

import argparse
import json
import os
from collections import Counter

import cv2


def overlaps(a_start, a_end, intervals):
    return any(a_start < e and s < a_end for s, e in intervals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--num-frames", type=int, default=8,
                         help="Must match the --num-frames used in the actual model run being checked")
    args = parser.parse_args()
    t = max(args.num_frames // 2, 1)

    samples = json.load(open(args.samples_file))
    for s in samples:
        s["video"] = os.path.expanduser(s["video"])

    fixed_chunk_hits = Counter()
    fixed_chunk_totals = Counter()
    random_baseline_total = 0.0
    n_run = 0

    for i, s in enumerate(samples):
        cap = cv2.VideoCapture(s["video"])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        if total_frames <= 0:
            print(f"  WARNING: could not read metadata for {s['video']!r}, skipping")
            continue

        indices = [round(j * (total_frames - 1) / max(args.num_frames - 1, 1)) for j in range(args.num_frames)]
        timestamps = [idx / fps for idx in indices]
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
        n_run += 1

    if n_run == 0:
        raise RuntimeError("No samples could be processed")

    print(f"=== {args.samples_file}: n={n_run}, t={t} chunks (num_frames={args.num_frames}) ===")
    print(f"uniform-random baseline = {random_baseline_total / n_run:.2%}")
    fixed_rates = {c: fixed_chunk_hits[c] / fixed_chunk_totals[c] for c in range(t)}
    best_c, best_rate = max(fixed_rates.items(), key=lambda kv: kv[1])
    print(f"best FIXED-chunk baseline (always predict chunk {best_c}, ignoring content) = {best_rate:.2%}")
    print("  " + ", ".join(f"chunk={c}: {fixed_rates[c]:.2%}" for c in range(t)))
    print("\nCompare any previously reported hit_rate for a head evaluated on this exact samples file "
          "(and this exact --num-frames) against the best-fixed-chunk number above. A hit_rate that "
          "doesn't clear it is not distinguishable from ignoring the video/question entirely.")


if __name__ == "__main__":
    main()
