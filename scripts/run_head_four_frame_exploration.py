"""Single-resolution, label-free low-resolution head ranking diagnostic.

Exploratory attention-shape stability only; not a faithfulness experiment.
"""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import head_selection_unsupervised as legacy
from run_head_temporal_resolution_exploration import scan_one, summarize

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-file", required=True)
    p.add_argument("--protocol-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=3, help="3 for smoke; 0 for all rows")
    p.add_argument("--frames", type=int, choices=(4, 6), default=4)
    p.add_argument("--max-pixels", type=int, default=200704)
    p.add_argument("--exploratory", action="store_true", required=True)
    args = p.parse_args()
    manifest, output = Path(args.samples_file), Path(args.output)
    if output.exists():
        p.error("refusing_to_overwrite_output")
    rows = json.loads(manifest.read_text())
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        p.error("empty_cohort")
    for row in rows:
        row["video"] = os.path.expanduser(row["video"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_args = SimpleNamespace(model="Qwen/Qwen2.5-VL-7B-Instruct", quant="4bit",
                                 max_pixels=args.max_pixels)
    model, processor = legacy.load_model(model_args, device)
    token_id = legacy.find_video_token_id(model, processor)
    n_layers = len(model.model.language_model.layers)
    n_heads = model.config.text_config.num_attention_heads
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    records, error_types, error_messages, failed_indices = [], {}, {}, []
    started = time.monotonic()
    observed_bins = set()
    for index, sample in enumerate(rows):
        try:
            mass, entropy, peaks, bins = scan_one(
                sample, args.frames, model, processor, token_id, device)
            if mass.shape != (n_layers, n_heads):
                raise RuntimeError("unexpected_attention_shape")
            records.append((mass, entropy, peaks))
            observed_bins.add(bins)
        except Exception as exc:
            name = type(exc).__name__
            error_types[name] = error_types.get(name, 0) + 1
            msg = str(exc).replace(sample.get("video", ""), "[video]")
            msg = msg.replace(sample.get("question", ""), "[question]")[:160]
            error_messages[msg] = error_messages.get(msg, 0) + 1
            failed_indices.append(index)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if len(observed_bins) != 1 or (observed_bins and next(iter(observed_bins)) < 2):
        raise RuntimeError("inconsistent_or_insufficient_temporal_bins")
    if not records:
        raise RuntimeError("no_successful_samples")
    result = summarize(records, n_layers, n_heads)
    result.update({
        "schema": "head_four_frame_exploration_v1",
        "research_status": "exploratory_only",
        "scope": "attention-shape ranking stability only; not faithfulness",
        "data_manifest_sha256": sha256(manifest),
        "protocol_sha256": sha256(Path(args.protocol_file)),
        "requested_samples": len(rows), "effective_samples": len(records),
        "failures": len(failed_indices), "failed_manifest_indices": failed_indices,
        "sanitized_error_types": error_types,
        "sanitized_error_messages": error_messages,
        "frames_requested": args.frames, "temporal_bins_observed": next(iter(observed_bins)),
        "ranking": "mass elbow min_keep=10; sink fraction <0.5; entropy ascending; top20",
        "runner_sha256": sha256(Path(__file__)),
        "head_helper_sha256": sha256(Path(legacy.__file__)),
        "base_runner_sha256": sha256(Path(__import__("run_head_temporal_resolution_exploration").__file__)),
        "model_revision": MODEL_REVISION, "quantization": "4bit",
        "dtype": "bfloat16", "max_pixels": args.max_pixels,
        "device": device, "python": platform.python_version(),
        "torch": torch.__version__,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "max_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if torch.cuda.is_available() else None,
    })
    if torch.cuda.is_available():
        result["hardware"] = {"gpu": torch.cuda.get_device_name(0),
                              "cuda_runtime": torch.version.cuda}
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"output": str(output), "requested": len(rows),
                      "effective": len(records), "failures": len(failed_indices),
                      "temporal_bins": next(iter(observed_bins)),
                      "elapsed_seconds": result["elapsed_seconds"],
                      "max_allocated_vram_gib": result["max_allocated_vram_gib"]}))


if __name__ == "__main__":
    main()
