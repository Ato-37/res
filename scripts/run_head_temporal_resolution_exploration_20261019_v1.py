"""Exploratory, label-free head scan at two temporal resolutions.

This deliberately does not reopen the closed head-faithfulness confirmation.
It computes attention-shape summaries only and writes aggregate output locally.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

import head_selection_unsupervised as legacy


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_one(sample, frames, model, processor, token_id, device):
    paths, tmp_dir, _ = legacy.extract_frames(sample["video"], frames, 224)
    try:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": paths},
            {"type": "text", "text": sample["question"]},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = legacy.process_vision_info(messages)
        inputs = processor(text=[prompt], images=images, videos=videos,
                           padding=True, return_tensors="pt").to(device)
        start, end = legacy.video_span(inputs["input_ids"], token_id)
        grid = inputs.get("video_grid_thw")
        t = int(grid[0, 0]) if grid is not None else frames // 2
        span = end - start
        if t < 2 or span % t:
            raise RuntimeError("unexpected_video_token_geometry")
        with torch.inference_mode():
            out = model(**inputs, output_attentions=True)
        masses, entropies, peaks = [], [], []
        for attn in out.attentions:
            # Attention from the final text token to each video token.
            a = attn[0, :, -1, start:end].float()
            chunks = a.reshape(a.shape[0], t, span // t).sum(dim=-1)
            mass = chunks.sum(dim=1)
            probs = chunks / mass.clamp_min(1e-12).unsqueeze(1)
            entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
            masses.append(mass.cpu())
            entropies.append(entropy.cpu())
            peaks.append(chunks.argmax(dim=1).cpu())
        del out, inputs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return torch.stack(masses), torch.stack(entropies), torch.stack(peaks), t
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def summarize(records, n_layers, n_heads):
    # records: [(magnitude, entropy, peak)] where each tensor is [layer, head]
    mass = torch.stack([r[0] for r in records]).mean(0)
    entropy = torch.stack([r[1] for r in records]).mean(0)
    peak_stack = torch.stack([r[2] for r in records])
    sink = torch.zeros(n_layers, n_heads)
    for li in range(n_layers):
        for hi in range(n_heads):
            vals = peak_stack[:, li, hi].tolist()
            counts = {x: vals.count(x) for x in set(vals)}
            sink[li, hi] = max(counts.values()) / len(vals)
    flat = [(li, hi, float(mass[li, hi]), float(entropy[li, hi]), float(sink[li, hi]))
            for li in range(n_layers) for hi in range(n_heads)]
    flat.sort(key=lambda x: (-x[2], x[0], x[1]))
    k = legacy.elbow_index([x[2] for x in flat], min_keep=10)
    survivors = [x for x in flat[:k] if x[4] < 0.5]
    ranked = sorted(survivors, key=lambda x: (x[3], -x[2], x[0], x[1]))
    magnitude_rank = {(x[0], x[1]): i + 1 for i, x in enumerate(flat)}
    entropy_rank = {(x[0], x[1]): i + 1 for i, x in enumerate(ranked)}
    head_metrics = [{"layer": x[0], "head": x[1], "mean_video_mass": x[2],
                     "mean_temporal_entropy_nats": x[3], "sink_fraction": x[4],
                     "magnitude_rank": magnitude_rank[(x[0], x[1])],
                     "survives_elbow_and_sink_filter": (x[0], x[1]) in entropy_rank,
                     "entropy_rank_among_survivors": entropy_rank.get((x[0], x[1]))}
                    for x in flat]
    return {"num_heads_total": len(flat), "elbow_survivors": k,
            "non_sink_survivors": len(survivors),
            "all_head_metrics": head_metrics,
            "top20": [{"layer": x[0], "head": x[1], "mean_video_mass": x[2],
                       "mean_temporal_entropy_nats": x[3], "sink_fraction": x[4]}
                      for x in ranked[:20]],
            "ranked_pairs": [[x[0], x[1]] for x in ranked]}


def spearman(xs, ys):
    common = sorted(set(xs) & set(ys))
    if len(common) < 3:
        return None, len(common)
    rx = {v: i + 1 for i, v in enumerate(sorted(common, key=lambda v: xs[v]))}
    ry = {v: i + 1 for i, v in enumerate(sorted(common, key=lambda v: ys[v]))}
    mx, my = statistics.mean(rx.values()), statistics.mean(ry.values())
    cov = sum((rx[v] - mx) * (ry[v] - my) for v in common)
    denom = math.sqrt(sum((rx[v] - mx) ** 2 for v in common) * sum((ry[v] - my) ** 2 for v in common))
    return (cov / denom if denom else None), len(common)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=1, help="1 for smoke; 0 for all rows")
    p.add_argument("--exploratory", action="store_true", required=True,
                   help="required acknowledgement: output is exploratory, not confirmation")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    p.add_argument("--max-pixels", type=int, default=200704)
    args = p.parse_args()
    src = Path(args.samples_file)
    out = Path(args.output)
    if out.exists():
        p.error("refusing_to_overwrite_output")
    rows = json.loads(src.read_text())
    for row in rows:
        row["video"] = os.path.expanduser(row["video"])
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        p.error("empty_cohort")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = legacy.load_model(args, device)
    token_id = legacy.find_video_token_id(model, processor)
    n_layers = len(model.model.language_model.layers)
    n_heads = model.config.text_config.num_attention_heads
    result = {"schema": "head_temporal_resolution_exploration_v1",
              "research_status": "exploratory_only",
              "claim": "attention-shape temporal-resolution stability only; not faithfulness",
              "data_manifest_sha256": sha256(src), "requested_samples": len(rows),
              "sources": {"runner_sha256": sha256(Path(__file__)),
                          "head_helper_sha256": sha256(Path(legacy.__file__))},
              "command": {"limit": args.limit, "frames": [8, 16],
                          "min_keep": 10, "sink_fraction_exclusion_ge": 0.5,
                          "ranking": "temporal_entropy_ascending_after_attention_mass_elbow_and_sink_filter"},
              "conditions": {}, "device": device,
              "torch": torch.__version__, "python": platform.python_version(),
              "model": args.model, "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
              "quantization": args.quant, "dtype": "bfloat16",
              "max_pixels": args.max_pixels, "failures": 0}
    if torch.cuda.is_available():
        result["hardware"] = {"gpu": torch.cuda.get_device_name(0),
                               "cuda_runtime": torch.version.cuda}
    all_summaries = {}
    for frames in (8, 16):
        recs = []
        errors = 0
        error_types = {}
        started = time.monotonic()
        error_messages = {}
        for sample in rows:
            try:
                m, e, pk, t = scan_one(sample, frames, model, processor, token_id, device)
                recs.append((m, e, pk))
            except Exception as exc:
                errors += 1
                name = type(exc).__name__
                error_types[name] = error_types.get(name, 0) + 1
                # Keep only fixed diagnostic messages; never persist paths or
                # user/data text in the private result summary.
                msg = str(exc).replace(sample.get("video", ""), "[video]")
                msg = msg.replace(sample.get("question", ""), "[question]")[:160]
                error_messages[msg] = error_messages.get(msg, 0) + 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if not recs:
            raise RuntimeError(f"condition_{frames}f_no_successful_samples; sanitized_errors={error_messages}")
        summary = summarize(recs, n_layers, n_heads)
        summary.update({"frames_requested": frames, "temporal_bins_observed": t,
                        "effective_samples": len(recs), "failures": errors,
                        "sanitized_error_types": error_types,
                        "sanitized_error_messages": error_messages,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "max_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
                        if torch.cuda.is_available() else None})
        all_summaries[frames] = summary
        result["conditions"][str(frames)] = {k: v for k, v in summary.items() if k != "ranked_pairs"}
        result["failures"] += errors
    rank8 = {tuple(pair): i + 1 for i, pair in enumerate(all_summaries[8]["ranked_pairs"])}
    rank16 = {tuple(pair): i + 1 for i, pair in enumerate(all_summaries[16]["ranked_pairs"])}
    rho, n_shared = spearman(rank8, rank16)
    top8 = {tuple((x["layer"], x["head"])) for x in result["conditions"]["8"]["top20"]}
    top16 = {tuple((x["layer"], x["head"])) for x in result["conditions"]["16"]["top20"]}
    result["comparison"] = {"rank_spearman_on_shared_ranked_survivors": rho,
                            "shared_survivor_count": n_shared,
                            "top20_overlap_count": len(top8 & top16),
                            "top20_jaccard": len(top8 & top16) / len(top8 | top16) if top8 | top16 else None}
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"output": str(out), "requested": len(rows),
                      "effective_8f": result["conditions"]["8"]["effective_samples"],
                      "effective_16f": result["conditions"]["16"]["effective_samples"],
                      "failures": result["failures"], "comparison": result["comparison"]}))


if __name__ == "__main__":
    main()
