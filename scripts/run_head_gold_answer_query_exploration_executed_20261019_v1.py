#!/usr/bin/env python3
"""Post-selection, local-only head query-position exploration.

See experiments/2026-10-19_head_gold_answer_query_protocol.md. Gold answers
only define appended answer-token query positions; they are not outcomes or
head-selection labels. This is not a faithfulness test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import head_selection_unsupervised as head_helper
import run_head_bootstrap_exploration as base
import run_head_query_position_exploration as paired

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = base.MODEL_REVISION
NAMES = ("generation_prefix", "question_body_mean", "gold_answer_mean")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan_one(row, frames, model, processor, token_id):
    paths, temp_dir = base.extract_frames(row["video"], frames)
    try:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": paths},
            {"type": "text", "text": row["question"]}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[prompt], images=images, videos=videos,
                           padding=True, return_tensors="pt").to("cuda")
        video_start, video_end = head_helper.video_span(inputs["input_ids"], token_id)
        question_ids = processor.tokenizer.encode(row["question"], add_special_tokens=False)
        q_start, q_end = paired.find_unique_question_span(
            inputs["input_ids"][0], question_ids, video_start, video_end)
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("gold_answer_missing_or_empty")
        answer_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
        if not answer_ids:
            raise ValueError("gold_answer_tokenization_empty")
        prefix_len = int(inputs["input_ids"].shape[1])
        # The prompt ends at the assistant generation prefix. Appending exactly
        # tokenized answer text makes these positions causal assistant queries.
        answer_tensor = torch.tensor([answer_ids], device=inputs["input_ids"].device,
                                     dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], answer_tensor), dim=1)
        if "attention_mask" in inputs:
            ones = torch.ones_like(answer_tensor, dtype=inputs["attention_mask"].dtype)
            inputs["attention_mask"] = torch.cat((inputs["attention_mask"], ones), dim=1)
        if "mm_token_type_ids" in inputs:
            # Qwen2.5-VL uses 0 for ordinary text, 1 for images, and 2 for
            # video. The appended assistant answer is plain text.
            text_types = torch.zeros_like(answer_tensor,
                                          dtype=inputs["mm_token_type_ids"].dtype)
            inputs["mm_token_type_ids"] = torch.cat(
                (inputs["mm_token_type_ids"], text_types), dim=1)
        answer_start, answer_end = prefix_len, prefix_len + len(answer_ids)
        if answer_start < q_end or answer_end != int(inputs["input_ids"].shape[1]):
            raise ValueError("gold_answer_span_invalid")
        temporal_bins = int(inputs["video_grid_thw"][0, 0])
        if temporal_bins != frames // 2:
            raise ValueError("unexpected_temporal_geometry")
        with torch.inference_mode():
            output = model(**inputs, output_attentions=True)
        packed = []
        for attention in output.attentions:
            spans = ((prefix_len - 1, prefix_len),
                     (q_start, q_end), (answer_start, answer_end))
            per_mode = [paired.temporal_summary(
                attention, video_start=video_start, video_end=video_end,
                temporal_bins=temporal_bins, query_start=start, query_end=end)
                for start, end in spans]
            packed.append(per_mode)
        result = []
        for mode in range(3):
            result.append(tuple(np.stack([layer[mode][metric]
                                         for layer in packed])
                                for metric in range(3)))
        del output, inputs
        torch.cuda.synchronize()
        return tuple(tuple(x.astype(np.uint8) if i == 2 else x.astype(np.float32)
                           for i, x in enumerate(metrics)) for metrics in result)
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-file", required=True, type=Path)
    p.add_argument("--summary-output", required=True, type=Path)
    p.add_argument("--metrics-output", required=True, type=Path)
    p.add_argument("--protocol-file", required=True, type=Path)
    p.add_argument("--limit", type=int, default=1, help="1 for smoke; 0 for all rows")
    p.add_argument("--exploratory", action="store_true", required=True)
    a = p.parse_args(argv)
    if a.summary_output.exists() or a.metrics_output.exists():
        p.error("refusing_to_overwrite_output")
    if a.summary_output.parent.resolve() != a.metrics_output.parent.resolve():
        p.error("outputs_must_share_private_directory")
    rows = json.loads(a.samples_file.read_text())
    source_rows = len(rows)
    if a.limit:
        rows = rows[:a.limit]
    for row in rows:
        row["video"] = os.path.expanduser(row["video"])
    if not rows:
        p.error("empty_cohort")
    media_digests, group_ids, commitment = base.media_inventory(rows)
    quant = BitsAndBytesConfig(load_in_4bit=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=200704)
    token_id = paired.processor_video_token_id(processor)
    eligible, failed, reasons = paired.preflight_rows(rows, 8, processor, token_id)
    if not eligible:
        raise RuntimeError("no_cpu_preflight_eligible_samples")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
        quantization_config=quant, device_map="cuda")
    model.eval()
    n_layers = len(model.model.language_model.layers)
    n_heads = model.config.text_config.num_attention_heads
    torch.cuda.reset_peak_memory_stats()
    all_values = {name: [[], [], []] for name in NAMES}
    successes, indices = [], []
    started = time.monotonic()
    for index in eligible:
        triplets = scan_one(rows[index], 8, model, processor, token_id)
        if any(x[0].shape != (n_layers, n_heads) for x in triplets):
            raise RuntimeError("unexpected_attention_shape")
        for name, metrics in zip(NAMES, triplets):
            for bucket, value in zip(all_values[name], metrics):
                bucket.append(value)
        indices.append(index)
        successes.append(int(group_ids[index]))
    if indices != eligible:
        raise RuntimeError("gpu_successes_differ_from_cpu_preflight")
    if [sha256(Path(row["video"])) for row in rows] != media_digests:
        raise RuntimeError("media_changed_during_scan")
    arrays = {name: tuple(np.stack(bucket) for bucket in values)
              for name, values in all_values.items()}
    for triplet in arrays.values():
        paired.validate_paired_metrics(triplet, triplet, temporal_bins=4)
    summaries = {}
    for name in ("question_body_mean", "gold_answer_mean"):
        comparison = paired.compare_rankings(
            arrays["generation_prefix"], arrays[name], temporal_bins=4)
        comparison[name] = comparison.pop("question_body_mean")
        comparison["comparison_condition"] = name
        summaries[name] = comparison
    a.summary_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if a.summary_output.parent.stat().st_mode & 0o777 != 0o700:
        raise RuntimeError("private_output_directory_must_be_0700")
    with a.metrics_output.open("xb") as stream:
        os.chmod(a.metrics_output, 0o600)
        np.savez_compressed(stream, sample_index=np.asarray(indices, dtype=np.int32),
                            media_group_id=np.asarray(successes, dtype=np.int32),
                            failed_sample_index=np.asarray(failed, dtype=np.int32),
                            **{f"{name}_{metric}": value
                               for name, triple in arrays.items()
                               for metric, value in zip(("mass", "entropy", "peak"), triple)})
        stream.flush()
        os.fsync(stream.fileno())
    summary = {
        "schema": "head_gold_answer_query_exploration_v1",
        "research_status": "exploratory_post_selection_only",
        "claim": "gold-answer-conditioned attention-shape sensitivity only; not faithfulness",
        "gold_answer_use": "query-position definition only; not a selection outcome",
        "protocol_sha256": sha256(a.protocol_file),
        "data_manifest_sha256": sha256(a.samples_file), "source_manifest_rows": source_rows,
        "requested_samples": len(rows), "effective_samples": len(indices),
        "failed_sample_count": len(failed), "failed_samples": reasons,
        "sample_index_sha256": hashlib.sha256(np.asarray(indices, dtype=np.int32).tobytes()).hexdigest(),
        "failed_sample_index_sha256": hashlib.sha256(np.asarray(failed, dtype=np.int32).tobytes()).hexdigest(),
        "ordered_media_content_commitment_sha256": commitment,
        "effective_unique_media_count": len(set(successes)),
        "metrics_artifact": a.metrics_output.name, "metrics_sha256": sha256(a.metrics_output),
        "runner_sha256": sha256(Path(__file__)), "helper_sha256": sha256(Path(head_helper.__file__)),
        "paired_runner_sha256": sha256(Path(paired.__file__)),
        "base_runner_sha256": sha256(Path(base.__file__)),
        "model": MODEL, "model_revision": REVISION, "model_resolution": "local_files_only_pinned_commit",
        "quantization": "4bit", "dtype": "bfloat16", "attention": "eager",
        "frames": 8, "temporal_bins": 4, "max_pixels": 200704,
        "ranking": {"min_keep": 10, "sink_excluded_at_or_above": 0.5,
                    "top_n": 20, "entropy_ascending": True},
        "comparisons": summaries,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "max_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
    }
    with a.summary_output.open("xb") as stream:
        os.chmod(a.summary_output, 0o600)
        stream.write((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"requested": len(rows), "effective": len(indices),
                      "failures": reasons, "elapsed_seconds": summary["elapsed_seconds"],
                      "peak_vram_gib": summary["max_allocated_vram_gib"],
                      "output": str(a.summary_output), "metrics": str(a.metrics_output)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
