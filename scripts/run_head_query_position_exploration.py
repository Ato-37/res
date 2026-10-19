#!/usr/bin/env python3
"""Paired exploratory head attention summaries at two query-position definitions.

Local-only, post-selection attention-shape measurement. It is not a
faithfulness experiment. See the dated frozen protocol before invoking.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from PIL import __version__ as PIL_VERSION
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import __version__ as TRANSFORMERS_VERSION
from tokenizers import __version__ as TOKENIZERS_VERSION
import cv2
from qwen_vl_utils import process_vision_info

import head_selection_unsupervised as head_helper
import run_head_bootstrap_exploration as base


PROTOCOL = Path("experiments/2026-10-19_head_query_position_sensitivity_protocol.md")
MODE_NAMES = ("generation_prefix", "question_body_mean")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_unique_question_span(input_ids, question_token_ids, video_start, video_end):
    """Locate exact question token IDs after the video span; reject ambiguity."""
    ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    needle = question_token_ids.tolist() if hasattr(question_token_ids, "tolist") else list(question_token_ids)
    if not ids or not needle:
        raise ValueError("question_span_empty")
    if (type(video_start) is not int or type(video_end) is not int
            or video_start < 0 or video_end <= video_start or video_end > len(ids)):
        raise ValueError("video_span_invalid")
    hits = [i for i in range(len(ids) - len(needle) + 1)
            if ids[i:i + len(needle)] == needle]
    if len(hits) != 1:
        raise ValueError("question_span_not_unique")
    start = hits[0]
    end = start + len(needle)
    if start < video_end or end > len(ids):
        raise ValueError("question_span_outside_text_after_video")
    return start, end


def temporal_summary(attention, *, video_start, video_end, temporal_bins,
                     query_start, query_end):
    """Reduce one forward to per-head mass, entropy, and one peak bin."""
    if attention.ndim != 4 or attention.shape[0] != 1:
        raise ValueError("attention_shape_invalid")
    if not (0 <= video_start < video_end <= attention.shape[-1]):
        raise ValueError("video_span_invalid")
    if not (0 <= query_start < query_end <= attention.shape[-2]):
        raise ValueError("query_span_invalid")
    span = video_end - video_start
    if span % temporal_bins:
        raise ValueError("temporal_span_not_divisible")
    chunk_size = span // temporal_bins
    values = attention[0, :, query_start:query_end, video_start:video_end].float()
    # [heads, query subtokens, temporal bins, spatial tokens] -> equal-token
    # mean after summing each token's visual mass into temporal bins.
    chunks = values.reshape(values.shape[0], values.shape[1], temporal_bins,
                            chunk_size).sum(-1).mean(1)
    mass = chunks.sum(-1)
    probs = chunks / mass.clamp_min(1e-12).unsqueeze(-1)
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
    return (mass.cpu().numpy(), entropy.cpu().numpy(),
            chunks.argmax(-1).cpu().numpy().astype(np.uint8))


def summarize_all_heads(mass_rows, entropy_rows, peak_rows, *, min_keep=10,
                        max_sink=0.5, top_n=20, temporal_bins=8):
    """Apply the fixed mass-elbow, sink exclusion, entropy ranking rule."""
    mass_rows = np.asarray(mass_rows, dtype=np.float64)
    entropy_rows = np.asarray(entropy_rows, dtype=np.float64)
    peaks = np.asarray(peak_rows)
    if (mass_rows.ndim != 3 or mass_rows.shape[0] < 1
            or entropy_rows.shape != mass_rows.shape or peaks.shape != mass_rows.shape):
        raise ValueError("metric_array_shape_or_range_invalid")
    if (not np.isfinite(mass_rows).all() or not np.isfinite(entropy_rows).all()
            or np.any(mass_rows < -1e-6) or np.any(mass_rows > 1.00001)
            or np.any(entropy_rows < -1e-6)
            or np.any(entropy_rows > np.log(temporal_bins) + 1e-4)
            or np.any(peaks < 0) or np.any(peaks >= temporal_bins)):
        raise ValueError("metric_value_range_invalid")
    mass = mass_rows.mean(axis=0)
    entropy = entropy_rows.mean(axis=0)
    sink = np.zeros(mass.shape, dtype=np.float64)
    for li in range(mass.shape[0]):
        for hi in range(mass.shape[1]):
            counts = np.bincount(peaks[:, li, hi].astype(np.int64), minlength=temporal_bins)
            sink[li, hi] = counts.max() / len(peaks)
    flat = [(li, hi, float(mass[li, hi]), float(entropy[li, hi]),
             float(sink[li, hi]))
            for li in range(mass.shape[0]) for hi in range(mass.shape[1])]
    flat.sort(key=lambda row: (-row[2], row[0], row[1]))
    elbow_n = head_helper.elbow_index([row[2] for row in flat], min_keep=min_keep)
    eligible = [row for row in flat[:elbow_n] if row[4] < max_sink]
    eligible.sort(key=lambda row: (row[3], -row[2], row[0], row[1]))
    return eligible, elbow_n


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        rank = ((pos + 1) + end) / 2.0
        for index in range(pos, end):
            ranks[order[index]] = rank
        pos = end
    return ranks


def spearman_average_ties(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    dx = sum((x - mx) ** 2 for x in rx)
    dy = sum((y - my) ** 2 for y in ry)
    denominator = (dx * dy) ** 0.5
    return numerator / denominator if denominator else None


def compare_rankings(prefix_rows, question_rows, *, temporal_bins=8):
    prefix, elbow_prefix = summarize_all_heads(*prefix_rows, temporal_bins=temporal_bins)
    question, elbow_question = summarize_all_heads(*question_rows, temporal_bins=temporal_bins)
    prefix_map = {(x[0], x[1]): x for x in prefix}
    question_map = {(x[0], x[1]): x for x in question}
    shared = sorted(set(prefix_map) & set(question_map))
    top_prefix = prefix[:20]
    top_question = question[:20]
    top_a = {(x[0], x[1]) for x in top_prefix}
    top_b = {(x[0], x[1]) for x in top_question}
    union = top_a | top_b
    rho = spearman_average_ties(
        [prefix_map[k][3] for k in shared],
        [question_map[k][3] for k in shared])
    return {
        "generation_prefix": {
            "elbow_survivors": elbow_prefix,
            "non_sink_survivors": len(prefix),
            "top20": [{"layer": x[0], "head": x[1], "mean_video_mass": x[2],
                       "mean_temporal_entropy_nats": x[3], "sink_fraction": x[4]}
                      for x in top_prefix],
        },
        "question_body_mean": {
            "elbow_survivors": elbow_question,
            "non_sink_survivors": len(question),
            "top20": [{"layer": x[0], "head": x[1], "mean_video_mass": x[2],
                       "mean_temporal_entropy_nats": x[3], "sink_fraction": x[4]}
                      for x in top_question],
        },
        "comparison": {
            "top20_intersection": len(top_a & top_b),
            "top20_union": len(union),
            "top20_jaccard": len(top_a & top_b) / len(union) if union else None,
            "shared_non_sink_survivors": len(shared),
            "spearman_average_ties_on_shared_survivor_entropy": rho,
        },
    }


def scan_one(row, frames, model, processor, token_id):
    paths, tmp_dir = base.extract_frames(row["video"], frames)
    try:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": paths},
            {"type": "text", "text": row["question"]},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos,
                           padding=True, return_tensors="pt").to("cuda")
        video_start, video_end = head_helper.video_span(inputs["input_ids"], token_id)
        question_ids = processor.tokenizer.encode(row["question"],
                                                  add_special_tokens=False)
        question_start, question_end = find_unique_question_span(
            inputs["input_ids"][0], question_ids, video_start, video_end)
        grid = inputs.get("video_grid_thw")
        temporal_bins = int(grid[0, 0]) if grid is not None else frames // 2
        if temporal_bins != frames // 2:
            raise ValueError("unexpected_temporal_geometry")
        with torch.inference_mode():
            out = model(**inputs, output_attentions=True)
        prefix_metrics, question_metrics = [], []
        for attention in out.attentions:
            prefix_metrics.append(temporal_summary(
                attention, video_start=video_start, video_end=video_end,
                temporal_bins=temporal_bins, query_start=attention.shape[-2] - 1,
                query_end=attention.shape[-2]))
            question_metrics.append(temporal_summary(
                attention, video_start=video_start, video_end=video_end,
                temporal_bins=temporal_bins, query_start=question_start,
                query_end=question_end))
        del out, inputs
        torch.cuda.synchronize()
        def pack(metrics):
            return (np.stack([x[0] for x in metrics]),
                    np.stack([x[1] for x in metrics]),
                    np.stack([x[2] for x in metrics]))
        return pack(prefix_metrics), pack(question_metrics)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def processor_video_token_id(processor):
    token = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    if token is None or token == processor.tokenizer.unk_token_id:
        raise ValueError("video_token_id_unavailable")
    return token


def preflight_one(row, frames, processor, token_id):
    """CPU-only decode/processor/span validation before loading model weights."""
    paths, tmp_dir = base.extract_frames(row["video"], frames)
    try:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": paths},
            {"type": "text", "text": row["question"]},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[prompt], images=images, videos=videos,
                           padding=True, return_tensors="pt")
        video_start, video_end = head_helper.video_span(inputs["input_ids"], token_id)
        question_ids = processor.tokenizer.encode(row["question"],
                                                  add_special_tokens=False)
        question_start, question_end = find_unique_question_span(
            inputs["input_ids"][0], question_ids, video_start, video_end)
        grid = inputs.get("video_grid_thw")
        temporal_bins = int(grid[0, 0]) if grid is not None else frames // 2
        if (temporal_bins != frames // 2 or
                (video_end - video_start) % temporal_bins != 0 or
                question_start >= question_end):
            raise ValueError("unexpected_temporal_geometry")
        return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def preflight_rows(rows, frames, processor, token_id):
    eligible, failed, reasons = [], [], Counter()
    for index, row in enumerate(rows):
        try:
            preflight_one(row, frames, processor, token_id)
            eligible.append(index)
        except RuntimeError as exc:
            if str(exc) not in {"unreadable_video_metadata", "unreadable_sampled_frame"}:
                raise RuntimeError("cpu_preflight_failed") from None
            failed.append(index)
            reasons["frame_decode"] += 1
    if len(eligible) + len(failed) != len(rows):
        raise RuntimeError("cpu_preflight_denominator_mismatch")
    return eligible, failed, dict(reasons)


def validate_paired_metrics(prefix_rows, question_rows, *, temporal_bins=8):
    if len(prefix_rows) != 3 or len(question_rows) != 3:
        raise ValueError("metric_tuple_invalid")
    if any(a.shape != b.shape for a, b in zip(prefix_rows, question_rows)):
        raise ValueError("paired_metric_shape_mismatch")
    if prefix_rows[0].shape != prefix_rows[1].shape or prefix_rows[0].shape != prefix_rows[2].shape:
        raise ValueError("generation_prefix_metric_shape_mismatch")
    if question_rows[0].shape != question_rows[1].shape or question_rows[0].shape != question_rows[2].shape:
        raise ValueError("question_metric_shape_mismatch")
    for name, triplet in (("generation_prefix", prefix_rows),
                          ("question_body_mean", question_rows)):
        mass, entropy, peaks = triplet
        if (mass.ndim != 3 or mass.shape[0] < 1 or not np.isfinite(mass).all()
                or not np.isfinite(entropy).all()):
            raise ValueError(f"{name}_metric_nonfinite_or_shape_invalid")
        if (np.any(mass < -1e-6) or np.any(mass > 1.00001)
                or np.any(entropy < -1e-6)
                or np.any(entropy > np.log(temporal_bins) + 1e-4)
                or np.any(peaks < 0) or np.any(peaks >= temporal_bins)):
            raise ValueError(f"{name}_metric_value_out_of_range")


def distribution_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed-as-distribution"


def _write_exclusive(path: Path, raw: bytes):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-file", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--protocol-file", type=Path, default=PROTOCOL,
                        help="protocol whose content hash is recorded in the run summary")
    parser.add_argument("--limit", type=int, default=1,
                        help="1 for smoke; 0 for all manifest rows")
    parser.add_argument("--frames", type=int, choices=(8, 16), default=16)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--exploratory", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.summary_output.exists() or args.metrics_output.exists():
        parser.error("refusing_to_overwrite_output")
    if not args.protocol_file.is_file():
        parser.error("protocol_file_not_found")
    if args.summary_output.parent.resolve() != args.metrics_output.parent.resolve():
        parser.error("summary_and_metrics_must_share_private_directory")
    rows = json.loads(args.samples_file.read_text())
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        parser.error("empty_cohort")
    for row in rows:
        row["video"] = os.path.expanduser(row["video"])
    media_digests, group_ids, media_commitment = base.media_inventory(rows)

    quant = BitsAndBytesConfig(load_in_4bit=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(
        model_name, revision=base.MODEL_REVISION, local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    token_id = processor_video_token_id(processor)
    preflight_indices, preflight_failed, preflight_reasons = preflight_rows(
        rows, args.frames, processor, token_id)
    if not preflight_indices:
        raise RuntimeError("no_cpu_preflight_eligible_samples")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, revision=base.MODEL_REVISION, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
        quantization_config=quant, device_map="cuda")
    model.eval()
    n_layers = len(model.model.language_model.layers)
    n_heads = model.config.text_config.num_attention_heads
    torch.cuda.reset_peak_memory_stats()
    values = {name: {metric: [] for metric in ("mass", "entropy", "peak")}
              for name in MODE_NAMES}
    sample_indices, successes = [], []
    errors = Counter(preflight_reasons)
    started = time.monotonic()
    for index in preflight_indices:
        prefix, question = scan_one(rows[index], args.frames, model, processor, token_id)
        for metrics in (prefix, question):
            if metrics[0].shape != (n_layers, n_heads):
                raise ValueError("unexpected_attention_shape")
        for name, metrics in zip(MODE_NAMES, (prefix, question)):
            for metric, value in zip(("mass", "entropy", "peak"), metrics):
                values[name][metric].append(value)
        sample_indices.append(index)
        successes.append(int(group_ids[index]))
    if sample_indices != preflight_indices:
        raise RuntimeError("gpu_successes_differ_from_cpu_preflight")
    if not sample_indices:
        raise RuntimeError("no_successful_samples")
    if [sha256(Path(row["video"])) for row in rows] != media_digests:
        raise RuntimeError("media_changed_during_scan")
    packed = {}
    for name in MODE_NAMES:
        packed[name] = (
            np.stack(values[name]["mass"]).astype(np.float32),
            np.stack(values[name]["entropy"]).astype(np.float32),
            np.stack(values[name]["peak"]).astype(np.uint8),
        )
    temporal_bins = args.frames // 2
    validate_paired_metrics(packed[MODE_NAMES[0]], packed[MODE_NAMES[1]],
                            temporal_bins=temporal_bins)
    comparison = compare_rankings(packed[MODE_NAMES[0]], packed[MODE_NAMES[1]],
                                  temporal_bins=temporal_bins)
    torch.cuda.synchronize()
    args.summary_output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.summary_output.parent.stat().st_mode & 0o777 != 0o700:
        raise RuntimeError("private_output_directory_must_be_0700")
    with args.metrics_output.open("xb") as stream:
        os.chmod(args.metrics_output, 0o600)
        np.savez_compressed(
            stream, sample_index=np.asarray(sample_indices, dtype=np.int32),
            media_group_id=np.asarray(successes, dtype=np.int32),
            failed_sample_index=np.asarray(preflight_failed, dtype=np.int32),
            **{f"{name}_{metric}": array
               for name in MODE_NAMES
               for metric, array in zip(("mass", "entropy", "peak"), packed[name])})
        stream.flush()
        os.fsync(stream.fileno())
    summary = {
        "schema": "head_query_position_sensitivity_v1",
        "research_status": "exploratory_post_selection_only",
        "claim": "attention-shape query-position sensitivity only; not faithfulness",
        "protocol_file": args.protocol_file.as_posix(),
        "protocol_sha256": sha256(args.protocol_file),
        "data_manifest_sha256": sha256(args.samples_file),
        "source_manifest_rows": len(json.loads(args.samples_file.read_text())),
        "requested_samples": len(rows), "sample_limit": args.limit,
        "cohort_definition": "first limit rows of manifest in order; limit=0 means all rows",
        "effective_samples": len(sample_indices), "failed_samples": dict(errors),
        "failed_sample_count": len(preflight_failed),
        "sample_index_sha256": hashlib.sha256(
            np.asarray(sample_indices, dtype=np.int32).tobytes()).hexdigest(),
        "failed_sample_index_sha256": hashlib.sha256(
            np.asarray(preflight_failed, dtype=np.int32).tobytes()).hexdigest(),
        "unique_media_count": len(set(media_digests)),
        "effective_unique_media_count": len(set(successes)),
        "ordered_media_content_commitment_sha256": media_commitment,
        "resampling_unit": "paired, same forward per media; no inferential resampling",
        "metrics_artifact": args.metrics_output.name,
        "metrics_sha256": sha256(args.metrics_output),
        "runner_sha256": sha256(Path(__file__)),
        "helper_sha256": sha256(Path(head_helper.__file__)),
        "base_runner_sha256": sha256(Path(base.__file__)),
        "model": model_name, "model_revision": base.MODEL_REVISION,
        "model_loaded_revision": base.MODEL_REVISION,
        "model_resolution": "explicit_commit_revision_local_files_only",
        "quantization": "4bit", "dtype": "bfloat16", "attention": "eager",
        "frames": args.frames, "temporal_bins": args.frames // 2,
        "max_pixels": args.max_pixels, "query_summary_methods": {
            "generation_prefix": "final input attention query at input_ids position -1",
            "question_body_mean": "equal mean over exact question subtoken query positions; no special tokens",
        },
        "ranking": {"min_keep": 10, "sink_excluded_at_or_above": 0.5,
                    "top_n": 20, "tie_break": "entropy,mass_desc,layer,head"},
        "comparison": comparison,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "max_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__, "python": platform.python_version(),
        "transformers": TRANSFORMERS_VERSION, "tokenizers": TOKENIZERS_VERSION,
        "opencv": cv2.__version__, "pillow": PIL_VERSION,
        "numpy": np.__version__, "qwen_vl_utils": distribution_version("qwen-vl-utils"),
    }
    _write_exclusive(args.summary_output,
                     (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"requested": len(rows), "effective": len(sample_indices),
                      "failures": dict(errors),
                      "elapsed_seconds": summary["elapsed_seconds"],
                      "peak_vram_gib": summary["max_allocated_vram_gib"],
                      "summary": str(args.summary_output),
                      "metrics": str(args.metrics_output)}, sort_keys=True))


if __name__ == "__main__":
    main()
