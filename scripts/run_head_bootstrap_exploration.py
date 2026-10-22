"""Collect anonymous per-video head attention statistics for bootstrap stability.

Exploratory only: no gold labels, generated explanations, or faithfulness scores.
The model revision and processor are pinned to a local cached commit.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import head_selection_unsupervised as head_helper

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def media_inventory(rows):
    digests = []
    groups = {}
    group_ids = []
    for index, row in enumerate(rows):
        path = row["video"]
        if not os.path.isfile(path):
            raise RuntimeError("manifest_media_missing")
        digest = sha256(path)
        digests.append(digest)
        if digest not in groups:
            groups[digest] = len(groups)
        group_ids.append(groups[digest])
    commitment = hashlib.sha256("\n".join(
        f"{i}:{digest}" for i, digest in enumerate(digests)).encode()).hexdigest()
    return digests, np.asarray(group_ids, dtype=np.int32), commitment


def load_model(model_name, max_pixels):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, revision=MODEL_REVISION, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
        quantization_config=quant, device_map="cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_name, revision=MODEL_REVISION, local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=max_pixels)
    return model, processor


def extract_frames(video_path, num_frames, max_dim=224):
    tmp_dir = tempfile.mkdtemp(prefix="head_bootstrap_frames_")
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total <= 0:
            raise RuntimeError("unreadable_video_metadata")
        indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
        paths = []
        for pos, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("unreadable_sampled_frame")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image.thumbnail((max_dim, max_dim))
            path = os.path.join(tmp_dir, f"frame_{pos:03d}.jpg")
            image.save(path)
            paths.append(path)
        return paths, tmp_dir
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        cap.release()


def scan_one(sample, frames, model, processor, token_id):
    paths, tmp_dir = extract_frames(sample["video"], frames)
    try:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": paths},
            {"type": "text", "text": sample["question"]}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos,
                           padding=True, return_tensors="pt").to("cuda")
        start, end = head_helper.video_span(inputs["input_ids"], token_id)
        grid = inputs.get("video_grid_thw")
        temporal_bins = int(grid[0, 0]) if grid is not None else frames // 2
        expected_bins = frames // 2
        if temporal_bins != expected_bins or (end - start) % temporal_bins:
            raise RuntimeError("unexpected_temporal_geometry")
        with torch.inference_mode():
            out = model(**inputs, output_attentions=True)
        masses, entropies, peaks = [], [], []
        span = end - start
        for attention in out.attentions:
            values = attention[0, :, -1, start:end].float()
            chunks = values.reshape(values.shape[0], temporal_bins, span // temporal_bins).sum(-1)
            mass = chunks.sum(-1)
            probs = chunks / mass.clamp_min(1e-12).unsqueeze(-1)
            entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
            masses.append(mass.cpu().numpy())
            entropies.append(entropy.cpu().numpy())
            peaks.append(chunks.argmax(-1).cpu().numpy())
        del out, inputs
        torch.cuda.synchronize()
        return (np.stack(masses), np.stack(entropies), np.stack(peaks).astype(np.uint8))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def summarize(mass_rows, entropy_rows, peak_rows, top_n=20, min_keep=10, max_sink=0.5):
    mass = mass_rows.mean(axis=0)
    entropy = entropy_rows.mean(axis=0)
    # [samples, layers, heads] -> per-head dominant temporal bin frequency.
    sink = np.zeros(mass.shape, dtype=np.float64)
    for li in range(mass.shape[0]):
        for hi in range(mass.shape[1]):
            counts = np.bincount(peak_rows[:, li, hi].astype(int))
            sink[li, hi] = counts.max() / len(peak_rows)
    flat = [(li, hi, float(mass[li, hi]), float(entropy[li, hi]), float(sink[li, hi]))
            for li in range(mass.shape[0]) for hi in range(mass.shape[1])]
    flat.sort(key=lambda x: (-x[2], x[0], x[1]))
    k = head_helper.elbow_index([x[2] for x in flat], min_keep=min_keep)
    eligible = [x for x in flat[:k] if x[4] < max_sink]
    eligible.sort(key=lambda x: (x[3], -x[2], x[0], x[1]))
    return eligible[:top_n], k, len(eligible)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-file", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--metrics-output", required=True)
    p.add_argument("--limit", type=int, default=1, help="1 for smoke; 0 for all manifest rows")
    p.add_argument("--frames", type=int, choices=(8, 16), default=16)
    p.add_argument("--max-pixels", type=int, default=200704)
    p.add_argument("--exploratory", action="store_true", required=True)
    args = p.parse_args()
    manifest, summary_path, metrics_path = map(Path, (args.samples_file, args.summary_output, args.metrics_output))
    if summary_path.exists() or metrics_path.exists():
        p.error("refusing_to_overwrite_output")
    if summary_path.parent.resolve() != metrics_path.parent.resolve():
        p.error("summary_and_metrics_must_share_private_directory")
    rows = json.loads(manifest.read_text())
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        p.error("empty_cohort")
    for row in rows:
        row["video"] = os.path.expanduser(row["video"])
    media_digests, all_group_ids, media_commitment = media_inventory(rows)

    model, processor = load_model("Qwen/Qwen2.5-VL-7B-Instruct", args.max_pixels)
    token_id = head_helper.find_video_token_id(model, processor)
    n_layers = len(model.model.language_model.layers)
    n_heads = model.config.text_config.num_attention_heads
    torch.cuda.reset_peak_memory_stats()
    masses, entropies, peaks, sample_indices, success_group_ids, failed_indices = [], [], [], [], [], []
    errors = Counter()
    started = time.monotonic()
    for index, row in enumerate(rows):
        try:
            mass, entropy, peak = scan_one(row, args.frames, model, processor, token_id)
            if mass.shape != (n_layers, n_heads):
                raise RuntimeError("unexpected_attention_shape")
            masses.append(mass)
            entropies.append(entropy)
            peaks.append(peak)
            sample_indices.append(index)
            success_group_ids.append(all_group_ids[index])
        except Exception as exc:
            errors[type(exc).__name__] += 1
            failed_indices.append(index)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not masses:
        raise RuntimeError("no_successful_samples")
    # Verify that no input media bytes changed while the model was running.
    if [sha256(row["video"]) for row in rows] != media_digests:
        raise RuntimeError("media_changed_during_scan")
    mass_rows = np.stack(masses).astype(np.float32)
    entropy_rows = np.stack(entropies).astype(np.float32)
    peak_rows = np.stack(peaks).astype(np.uint8)
    candidates, elbow_n, eligible_n = summarize(mass_rows, entropy_rows, peak_rows)
    torch.cuda.synchronize()

    summary_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if summary_path.parent.stat().st_mode & 0o777 != 0o700:
        raise RuntimeError("private_output_directory_must_be_0700")
    mfd = os.open(metrics_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(mfd, "wb") as f:
        np.savez_compressed(f, sample_index=np.asarray(sample_indices, dtype=np.int32),
                            media_group_id=np.asarray(success_group_ids, dtype=np.int32),
                            failed_sample_index=np.asarray(failed_indices, dtype=np.int32),
                            attention_mass=mass_rows, temporal_entropy=entropy_rows,
                            peak_bin=peak_rows)
    summary = {"schema": "head_bootstrap_exploration_v1", "research_status": "exploratory_only",
               "claim": "per-video stability of attention-shape head selection only; not faithfulness",
               "data_manifest_sha256": sha256(manifest), "source_manifest_rows": len(json.loads(manifest.read_text())),
               "sample_limit": args.limit, "requested_samples": len(rows),
               "cohort_definition": "first limit rows of manifest in order; limit=0 means all rows",
               "effective_samples": len(sample_indices), "failures": dict(errors),
               "failed_sample_count": len(failed_indices),
               "sample_index_sha256": hashlib.sha256(np.asarray(sample_indices, dtype=np.int32).tobytes()).hexdigest(),
               "failed_sample_index_sha256": hashlib.sha256(np.asarray(failed_indices, dtype=np.int32).tobytes()).hexdigest(),
               "unique_media_count": len(set(media_digests)),
               "effective_unique_media_count": len(set(int(x) for x in success_group_ids)),
               "ordered_media_content_commitment_sha256": media_commitment,
               "resampling_unit": "exact-media-sha256 cluster; unique media are sampled with replacement",
               "metrics_artifact": metrics_path.name, "metrics_sha256": sha256(metrics_path),
               "runner_sha256": sha256(Path(__file__)),
               "helper_sha256": sha256(Path(head_helper.__file__)),
               "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": MODEL_REVISION,
               "model_loaded_revision": MODEL_REVISION,
               "model_resolution": "explicit_commit_revision_local_files_only",
               "quantization": "4bit", "dtype": "bfloat16", "attention": "eager",
               "frames": args.frames, "temporal_bins": args.frames // 2,
               "max_pixels": args.max_pixels, "elbow_min_keep": 10,
               "sink_fraction_excluded_at_or_above": 0.5, "eligible_heads": eligible_n,
               "elbow_survivors": elbow_n, "elapsed_seconds": round(time.monotonic() - started, 3),
               "max_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
               "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
               "torch": torch.__version__, "python": platform.python_version(),
               "candidate_top20": [{"layer": x[0], "head": x[1], "mean_video_mass": x[2],
                                    "mean_temporal_entropy_nats": x[3], "sink_fraction": x[4]}
                                   for x in candidates]}
    sfd = os.open(summary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(sfd, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"requested": len(rows), "effective": len(sample_indices),
                      "failure_types": dict(errors), "elapsed_seconds": summary["elapsed_seconds"],
                      "peak_vram_gib": summary["max_allocated_vram_gib"],
                      "summary": str(summary_path), "metrics": str(metrics_path)}))


if __name__ == "__main__":
    main()
