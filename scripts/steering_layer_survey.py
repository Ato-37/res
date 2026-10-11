"""Is layer=6's head=16 dose-response curve (attenuating helps, amplifying
hurts) actually unusual, or would most layers' head=16 show some directional
effect just from being scaled at all? Fix the head index at 16 and sweep
every layer (0..27), at just 3 scale points (0.25, 1.0, 2.0), on a smaller
sample set, to build an empirical distribution to compare layer=6 against
instead of just one hand-picked control layer.

Also runs a paired significance test (not just point estimates) for a
handful of layers of interest: per-sample prefers_low = lp_low - lp_base is
recorded (not just the cross-sample average), so we can report a mean,
sample std, paired t-test against 0 (df = n-1, exact incomplete-beta based
p-value, no scipy dependency), and a percentile bootstrap 95% CI. This is
needed to say "layer=6's attenuation effect is significant" rather than just
"layer=6 ranked highly in this sample" -- a point-estimate rank alone is not
a statistical claim.

Usage:
    python scripts/steering_layer_survey.py --samples-file data/nextgqa_samples_big_heldout.json --n-samples 25 --stats-layers 6,12,13 --seed 0
"""

import argparse
import json
import math
import os
import random
import shutil
import tempfile

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

HEAD_DIM = 128


def _betacf(a, b, x):
    """Continued-fraction evaluation of the incomplete beta function
    (Numerical Recipes 6.4.6), used below to get an exact t-test p-value
    without requiring scipy."""
    maxit, eps, fpmin = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_ttest_vs_zero(diffs):
    """One-sample (paired) t-test of `diffs` against 0. Returns
    (mean, sample_std, t_stat, df, two_tailed_p)."""
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = var ** 0.5
    se = sd / (n ** 0.5)
    t = mean / se if se > 0 else float("inf")
    df = n - 1
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return mean, sd, t, df, p


def bootstrap_ci(values, n_boot, alpha=0.05):
    """Percentile bootstrap CI for the mean of `values`. Caller must seed
    `random` beforehand for reproducibility."""
    n = len(values)
    means = []
    for _ in range(n_boot):
        resample = [values[random.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return means[lo_idx], means[hi_idx]


def extract_frames(video_path, num_frames, max_dim):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Could not read frame count from {video_path}")
    indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    tmp_dir = tempfile.mkdtemp(prefix="steering_survey_frames_")
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
    return frame_paths, tmp_dir


def scale_head_hook(head_idx, head_dim, scale):
    def hook(module, args):
        x = args[0].clone()
        x[..., head_idx * head_dim:(head_idx + 1) * head_dim] *= scale
        return (x,) + args[1:]
    return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--head", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--stats-layers", default="6,12",
                         help="Comma-separated layer indices to run a paired significance test on "
                              "(per-sample prefers_low, not just the cross-sample average printed in the table)")
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap resamples for the 95%% CI")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the bootstrap (for reproducibility)")
    args = parser.parse_args()
    random.seed(args.seed)
    stats_layers = {int(x) for x in args.stats_layers.split(",")}

    samples = json.load(open(args.samples_file))[:args.n_samples]
    for s in samples:
        s["video"] = os.path.expanduser(s["video"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, quant={args.quant}, head={args.head}, n_samples={len(samples)}, seed={args.seed}")

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
    n_layers = len(layers)

    # pre-build inputs once per sample (reused across all layers/scales)
    prepared = []
    for s in samples:
        try:
            frame_paths, tmp_dir = extract_frames(s["video"], args.num_frames, args.frame_size)
        except Exception as e:
            print(f"  WARNING: could not read {s['video']!r} ({e}), skipping")
            continue
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
        prepared.append((inputs, answer_token_id, frame_paths, tmp_dir))

    def logprobs_at_scale(li, scale):
        """Per-sample log-prob of the correct answer token, same order as
        `prepared`. The cross-sample average of this is what earlier runs
        reported; keeping the full list lets us also run a paired test."""
        layer = layers[li]
        handle = None
        if scale != 1:
            handle = layer.self_attn.o_proj.register_forward_pre_hook(scale_head_hook(args.head, HEAD_DIM, scale))
        values = []
        try:
            for inputs, answer_token_id, _, _ in prepared:
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1, :].float()
                values.append(torch.log_softmax(logits, dim=-1)[answer_token_id].item())
        finally:
            if handle is not None:
                handle.remove()
        return values

    print(f"\n{'layer':>5} {'lp@0.25':>9} {'lp@1.0':>9} {'lp@2.0':>9} {'prefers_low':>12} {'prefers_high':>13}")
    rows = []
    per_sample_cache = {}  # li -> (lp_low_list, lp_base_list), only kept for --stats-layers
    for li in range(n_layers):
        lp_low_list = logprobs_at_scale(li, 0.25)
        lp_base_list = logprobs_at_scale(li, 1.0)
        lp_high_list = logprobs_at_scale(li, 2.0)
        lp_low = sum(lp_low_list) / len(lp_low_list)
        lp_base = sum(lp_base_list) / len(lp_base_list)
        lp_high = sum(lp_high_list) / len(lp_high_list)
        prefers_low = lp_low - lp_base
        prefers_high = lp_high - lp_base
        rows.append((li, lp_low, lp_base, lp_high, prefers_low, prefers_high))
        if li in stats_layers:
            per_sample_cache[li] = (lp_low_list, lp_base_list)
        flag = "  <- candidate" if li == 6 else ("  <- earlier control" if li == 12 else "")
        print(f"{li:>5} {lp_low:>9.3f} {lp_base:>9.3f} {lp_high:>9.3f} {prefers_low:>12.3f} {prefers_high:>13.3f}{flag}")

    for _, _, _, tmp_dir in prepared:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # noqa: kept 4-way unpack to match `prepared` tuples

    rows.sort(key=lambda r: r[4], reverse=True)
    print(f"\nranked by 'prefers_low' (attenuation helps most) -- layer=6's rank:")
    for rank, r in enumerate(rows):
        if r[0] == 6:
            print(f"  layer=6 is rank {rank+1} of {n_layers} by prefers_low (value={r[4]:.3f})")
            break

    n_eff = len(prepared)
    print(f"\n=== paired significance test on prefers_low (per-sample lp@0.25 - lp@1.0), n={n_eff}, "
          f"n_boot={args.n_boot}, seed={args.seed} ===")
    print("NOTE: tests H0: mean(prefers_low) == 0 for each layer in isolation. This is NOT a test of "
          "'layer=6 is different from layer=12' directly (that would need the paired difference-of-differences); "
          "see the two-layer comparison line below for that.")
    for li in sorted(stats_layers):
        if li not in per_sample_cache:
            continue
        lows, bases = per_sample_cache[li]
        diffs = [lo - ba for lo, ba in zip(lows, bases)]
        mean, sd, t, df, p = paired_ttest_vs_zero(diffs)
        ci_lo, ci_hi = bootstrap_ci(diffs, args.n_boot)
        sig = "significant at alpha=0.05" if p < 0.05 else "NOT significant at alpha=0.05"
        print(f"  layer={li}: mean={mean:+.4f} nats, sd={sd:.4f}, t({df})={t:.3f}, p={p:.4f}, "
              f"95% bootstrap CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  -> {sig}")

    stats_layers_present = sorted(li for li in stats_layers if li in per_sample_cache)
    if len(stats_layers_present) >= 2:
        a, b = stats_layers_present[0], stats_layers_present[1]
        lows_a, bases_a = per_sample_cache[a]
        lows_b, bases_b = per_sample_cache[b]
        diffs_a = [lo - ba for lo, ba in zip(lows_a, bases_a)]
        diffs_b = [lo - ba for lo, ba in zip(lows_b, bases_b)]
        dod = [da - db for da, db in zip(diffs_a, diffs_b)]  # difference-of-differences, paired by sample
        mean, sd, t, df, p = paired_ttest_vs_zero(dod)
        ci_lo, ci_hi = bootstrap_ci(dod, args.n_boot)
        sig = "significant at alpha=0.05" if p < 0.05 else "NOT significant at alpha=0.05"
        print(f"\n  paired diff-of-diffs, layer={a} vs layer={b} (H0: their prefers_low effects are equal): "
              f"mean={mean:+.4f}, sd={sd:.4f}, t({df})={t:.3f}, p={p:.4f}, "
              f"95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  -> {sig}")


if __name__ == "__main__":
    main()
